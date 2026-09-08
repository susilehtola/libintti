// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Geometric gradient of the RI (density-fitting) Coulomb energy -- the force
// that matches the RI-J Fock build (ri.hpp), so an RI-SCF differentiates the
// surface it actually optimises on. With the fit coefficients gamma = M^{-1} d,
// d_P = sum_mn (mn|P) D_mn, M_PQ = (P|Q), the RI-J energy is
//   E_J = 1/2 d^T M^{-1} d = 1/2 sum_P d_P gamma_P,
// and (the metric inverse being stationary) its gradient is
//   dE_J/dx = sum_P gamma_P dd_P/dx  -  1/2 sum_PQ gamma_P gamma_Q dM_PQ/dx.
// Both terms need only derivatives of the 2-/3-centre Coulomb integrals, which
// are ghost-augmented quartets (ncenter.hpp), so the erigrad centre-shift
// (2 alpha [.+1] - m [.-1]) drives them; the zero-exponent ghost never moves.
// Matrix-level: density in, per-shell forces out; no quartet is exposed.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "blas.hpp"    // detail::gemm (row-major GEMM)
#include "erigrad.hpp" // detail::comp_index, detail::eri_block4
#include "fock.hpp"
#include "gto.hpp"
#include "jk.hpp"      // JKRequest, JKDerivResult
#include "ncenter.hpp" // detail::ghost_pair pattern (ghost shell)
#include "ri.hpp"      // detail::syevd
#include "tgrid.hpp"

namespace intti {

/// Per-shell forces of the RI energy: forb[shell] over the orbital basis,
/// faux[shell] over the auxiliary basis. The caller maps both to atoms.
template <class Real> struct RIGrad {
  std::vector<std::array<Real, 3>> forb, faux;
};

namespace detail {

/// Device digestion shared by the RI gradient terms: a list of (quartet, pos)
/// jobs whose promoted/demoted blocks have been batched, contracted with a
/// per-job coefficient pattern and accumulated into a force array with atomics.
/// This is the quartet_pos_grad body on device -- same MD centre shift
/// 2 alpha [pos+1] - m_e [pos-1], same ghost-shell quartets.
template <class Real> struct RIGradJobs {
  std::vector<ShellPair<Real>> plist;
  std::vector<std::pair<int, int>> quartets;
  std::vector<int> plusE, minusE; ///< batch entry per job (-1 if absent)
  std::vector<int> jl0, jl1, jl2, jl3;   ///< the four base momenta per job
  std::vector<int> jpos, jtgt;           ///< differentiated slot, force target
  std::vector<Real> jalpha;              ///< exponent of the differentiated shell
  std::vector<int> jo0, jo1, jo2;        ///< AO offsets used by the coefficient
  int njob{0};

  /// Add one (quartet, pos) job. `sh` are the four shells, `tgt` the index into
  /// the force array the derivative accumulates to.
  void add(const PrimitiveShell<Real> sh[4], int pos, int tgt, int o0, int o1, int o2) {
    auto mk = [&](int dl) {
      PrimitiveShell<Real> s[4] = {sh[0], sh[1], sh[2], sh[3]};
      s[pos].l += dl;
      const int ib = static_cast<int>(plist.size());
      plist.push_back(make_pair(s[0], s[1]));
      const int ik = static_cast<int>(plist.size());
      plist.push_back(make_pair(s[2], s[3]));
      const int e = static_cast<int>(quartets.size());
      quartets.push_back({ib, ik});
      return e;
    };
    plusE.push_back(mk(1));
    minusE.push_back(sh[pos].l >= 1 ? mk(-1) : -1);
    jl0.push_back(sh[0].l); jl1.push_back(sh[1].l);
    jl2.push_back(sh[2].l); jl3.push_back(sh[3].l);
    jpos.push_back(pos);
    jtgt.push_back(tgt);
    jalpha.push_back(sh[pos].alpha);
    jo0.push_back(o0); jo1.push_back(o1); jo2.push_back(o2);
    ++njob;
  }
};

/// Run the batched jobs and accumulate the forces. The coefficient pattern is
/// selected by Mode (RI-J uses 0/1, RI-K uses 2/3):
///   0: D_mn gamma_P                     (RI-J three-centre, Term A)
///   1: -1/2 gamma_P gamma_Q             (RI-J two-centre,  Term B)
///   2: c3[(mu,nu,P)]                    (RI-K three-centre, Term 1)
///   3: c2[(T,U)]                        (RI-K two-centre,  Term 2)
/// Forces are indexed shell-major over the concatenated [orbital shells,
/// auxiliary shells] list. Unused coefficient views may be empty.
template <class Real, int Mode>
void ri_grad_digest(RIGradJobs<Real> &jobs, const TGrid<Real> &grid,
                    Kokkos::View<const Real *> Dd, int nao,
                    Kokkos::View<const Real *> gam,
                    Kokkos::View<const Real *> c3v, Kokkos::View<const Real *> c2v,
                    int naux,
                    Kokkos::View<Real *[3], Kokkos::LayoutLeft> force) {
  if (jobs.njob == 0) return;
  auto tab = make_pair_table(jobs.plist);
  auto batch = make_batch(tab, jobs.quartets);
  QuartetWorkspace<Real> ws;
  Kokkos::View<Real *> out("intti::rig::out", batch.nout_total);
  eri_quartets(tab, batch, grid, out, ws);
  auto pE = to_device(jobs.plusE, "rig::pE"), mE = to_device(jobs.minusE, "rig::mE");
  auto l0 = to_device(jobs.jl0, "rig::l0"), l1 = to_device(jobs.jl1, "rig::l1");
  auto l2 = to_device(jobs.jl2, "rig::l2"), l3 = to_device(jobs.jl3, "rig::l3");
  auto posv = to_device(jobs.jpos, "rig::pos"), tgtv = to_device(jobs.jtgt, "rig::tgt");
  auto alv = to_device(jobs.jalpha, "rig::al");
  auto o0v = to_device(jobs.jo0, "rig::o0"), o1v = to_device(jobs.jo1, "rig::o1");
  auto o2v = to_device(jobs.jo2, "rig::o2");
  auto offv = batch.out_offset;
  const int njob = jobs.njob;
  Kokkos::parallel_for(
      "intti::rig::digest", Kokkos::RangePolicy<>(0, njob), KOKKOS_LAMBDA(int j) {
        const int L[4] = {l0(j), l1(j), l2(j), l3(j)};
        const int nc[4] = {ncart(L[0]), ncart(L[1]), ncart(L[2]), ncart(L[3])};
        const int p = posv(j), lp = L[p], tgt = tgtv(j);
        const Real ap = alv(j);
        const std::int64_t pb = offv(pE(j));
        const bool hasm = mE(j) >= 0;
        const std::int64_t mb = hasm ? offv(mE(j)) : 0;
        int npl[4], nmi[4];
        for (int i = 0; i < 4; ++i) { npl[i] = nc[i]; nmi[i] = nc[i]; }
        npl[p] = ncart(lp + 1);
        if (lp >= 1) nmi[p] = ncart(lp - 1);
        auto idx = [](const int n[4], int a, int b, int c, int d) {
          return ((static_cast<std::size_t>(a) * n[1] + b) * n[2] + c) * n[3] + d;
        };
        const int oa = o0v(j), ob = o1v(j), oc = o2v(j);
        int k[4];
        for (k[0] = 0; k[0] < nc[0]; ++k[0])
          for (k[1] = 0; k[1] < nc[1]; ++k[1])
            for (k[2] = 0; k[2] < nc[2]; ++k[2])
              for (k[3] = 0; k[3] < nc[3]; ++k[3]) {
                Real cf;
                if constexpr (Mode == 0)
                  cf = Dd(static_cast<std::size_t>(oa + k[0]) * nao + ob + k[1]) * gam(oc + k[2]);
                else if constexpr (Mode == 1)
                  cf = Real(-0.5) * gam(oa + k[0]) * gam(oc + k[2]);
                else if constexpr (Mode == 2)
                  cf = c3v(((static_cast<std::size_t>(oa + k[0]) * nao + ob + k[1]) * naux) +
                           oc + k[2]);
                else
                  cf = c2v(static_cast<std::size_t>(oa + k[0]) * naux + oc + k[2]);
                if (cf == Real(0)) continue;
                int b3[3];
                cart_comp(lp, k[p], b3[0], b3[1], b3[2]);
                for (int e = 0; e < 3; ++e) {
                  int m3[3] = {b3[0], b3[1], b3[2]};
                  m3[e] += 1;
                  const int ip = comp_index(lp + 1, m3[0], m3[1]);
                  int ii[4] = {k[0], k[1], k[2], k[3]};
                  ii[p] = ip;
                  Real term = 2 * ap * out(pb + idx(npl, ii[0], ii[1], ii[2], ii[3]));
                  if (b3[e] >= 1 && hasm) {
                    int mm[3] = {b3[0], b3[1], b3[2]};
                    mm[e] -= 1;
                    const int im = comp_index(lp - 1, mm[0], mm[1]);
                    int jj[4] = {k[0], k[1], k[2], k[3]};
                    jj[p] = im;
                    term -= Real(b3[e]) * out(mb + idx(nmi, jj[0], jj[1], jj[2], jj[3]));
                  }
                  Kokkos::atomic_add(&force(tgt, e), cf * term);
                }
              }
      });
}

/// Contribution of differentiating shell position `pos` (0..3) of the quartet
/// (s0 s1 | s2 s3) to a force vector out[3], contracted with a coefficient
/// coeff(k0,k1,k2,k3) over the four Cartesian component indices. Uses the MD
/// centre-shift d/dR = 2 alpha [.+1] - m [.-1] (erigrad). A zero-exponent ghost
/// must never be passed as `pos` (its derivative is zero and is simply skipped
/// by the caller).
template <class Real, class CoeffFn>
void quartet_pos_grad(const PrimitiveShell<Real> &s0, const PrimitiveShell<Real> &s1,
                      const PrimitiveShell<Real> &s2, const PrimitiveShell<Real> &s3,
                      int pos, const TGrid<Real> &grid, CoeffFn coeff, Real out[3]) {
  const PrimitiveShell<Real> sh[4] = {s0, s1, s2, s3};
  const int L[4] = {s0.l, s1.l, s2.l, s3.l};
  const int nc[4] = {ncart(L[0]), ncart(L[1]), ncart(L[2]), ncart(L[3])};
  const int lp = L[pos];
  const Real ap = sh[pos].alpha;
  auto promote = [&](int dl) {
    PrimitiveShell<Real> s[4] = {s0, s1, s2, s3};
    s[pos].l += dl;
    return detail::eri_block4(s[0], s[1], s[2], s[3], grid);
  };
  auto plus = promote(1);
  std::vector<Real> minus;
  if (lp >= 1) minus = promote(-1);
  int npl[4], nmi[4];
  for (int i = 0; i < 4; ++i) {
    npl[i] = nc[i];
    nmi[i] = nc[i];
  }
  npl[pos] = ncart(lp + 1);
  if (lp >= 1) nmi[pos] = ncart(lp - 1);
  auto idx = [](const int n[4], int a, int b, int c, int d) {
    return ((static_cast<std::size_t>(a) * n[1] + b) * n[2] + c) * n[3] + d;
  };
  int k[4];
  for (k[0] = 0; k[0] < nc[0]; ++k[0])
    for (k[1] = 0; k[1] < nc[1]; ++k[1])
      for (k[2] = 0; k[2] < nc[2]; ++k[2])
        for (k[3] = 0; k[3] < nc[3]; ++k[3]) {
          const Real cf = coeff(k[0], k[1], k[2], k[3]);
          if (cf == Real(0)) continue;
          int b3[3];
          cart_comp(lp, k[pos], b3[0], b3[1], b3[2]);
          for (int e = 0; e < 3; ++e) {
            int m3[3] = {b3[0], b3[1], b3[2]};
            m3[e] += 1;
            const int ip = detail::comp_index(lp + 1, m3[0], m3[1]);
            int ii[4] = {k[0], k[1], k[2], k[3]};
            ii[pos] = ip;
            Real term = 2 * ap * plus[idx(npl, ii[0], ii[1], ii[2], ii[3])];
            if (b3[e] >= 1) {
              int mm[3] = {b3[0], b3[1], b3[2]};
              mm[e] -= 1;
              const int im = detail::comp_index(lp - 1, mm[0], mm[1]);
              int jj[4] = {k[0], k[1], k[2], k[3]};
              jj[pos] = im;
              term -= Real(b3[e]) * minus[idx(nmi, jj[0], jj[1], jj[2], jj[3])];
            }
            out[e] += cf * term;
          }
        }
}

/// Zero-exponent unit-s ghost at a shell's centre (the RI ket partner).
template <class Real>
PrimitiveShell<Real> ghost_shell(const PrimitiveShell<Real> &s) {
  return {Real(0), {s.center[0], s.center[1], s.center[2]}, 0};
}

/// Full (uncontracted) derivative block d/dR_{pos,dir} of the quartet
/// (s0 s1 | s2 s3): out has the base component shape, out[idx] = the MD
/// centre-shift 2 alpha [.+1_dir] - m [.-1_dir] of position `pos`.
template <class Real>
std::vector<Real> quartet_pos_deriv_block(const PrimitiveShell<Real> &s0,
                                          const PrimitiveShell<Real> &s1,
                                          const PrimitiveShell<Real> &s2,
                                          const PrimitiveShell<Real> &s3, int pos, int dir,
                                          const TGrid<Real> &grid) {
  const PrimitiveShell<Real> sh[4] = {s0, s1, s2, s3};
  const int L[4] = {s0.l, s1.l, s2.l, s3.l};
  const int nc[4] = {ncart(L[0]), ncart(L[1]), ncart(L[2]), ncart(L[3])};
  const int lp = L[pos];
  const Real ap = sh[pos].alpha;
  auto promote = [&](int dl) {
    PrimitiveShell<Real> s[4] = {s0, s1, s2, s3};
    s[pos].l += dl;
    return detail::eri_block4(s[0], s[1], s[2], s[3], grid);
  };
  auto plus = promote(1);
  std::vector<Real> minus;
  if (lp >= 1) minus = promote(-1);
  int npl[4], nmi[4];
  for (int i = 0; i < 4; ++i) {
    npl[i] = nc[i];
    nmi[i] = nc[i];
  }
  npl[pos] = ncart(lp + 1);
  if (lp >= 1) nmi[pos] = ncart(lp - 1);
  auto idx = [](const int n[4], int a, int b, int c, int d) {
    return ((static_cast<std::size_t>(a) * n[1] + b) * n[2] + c) * n[3] + d;
  };
  std::vector<Real> out(static_cast<std::size_t>(nc[0]) * nc[1] * nc[2] * nc[3], Real(0));
  int k[4];
  for (k[0] = 0; k[0] < nc[0]; ++k[0])
    for (k[1] = 0; k[1] < nc[1]; ++k[1])
      for (k[2] = 0; k[2] < nc[2]; ++k[2])
        for (k[3] = 0; k[3] < nc[3]; ++k[3]) {
          int b3[3];
          cart_comp(lp, k[pos], b3[0], b3[1], b3[2]);
          int m3[3] = {b3[0], b3[1], b3[2]};
          m3[dir] += 1;
          const int ip = detail::comp_index(lp + 1, m3[0], m3[1]);
          int ii[4] = {k[0], k[1], k[2], k[3]};
          ii[pos] = ip;
          Real term = 2 * ap * plus[idx(npl, ii[0], ii[1], ii[2], ii[3])];
          if (b3[dir] >= 1) {
            int mm[3] = {b3[0], b3[1], b3[2]};
            mm[dir] -= 1;
            const int im = detail::comp_index(lp - 1, mm[0], mm[1]);
            int jj[4] = {k[0], k[1], k[2], k[3]};
            jj[pos] = im;
            term -= Real(b3[dir]) * minus[idx(nmi, jj[0], jj[1], jj[2], jj[3])];
          }
          out[idx(nc, k[0], k[1], k[2], k[3])] = term;
        }
  return out;
}

/// Device jobs for the RI Hessians: one job per ghost-shell quartet, carrying
/// the batch entry for each per-position l-offset pattern the second-derivative
/// digestion asks for. Same construction as the 2e Hessian (erihess): patterns
/// are single +-2 at an allowed position (the p == q terms) and +-1 at two
/// allowed positions (p != q), looked up in the kernel by a 625-entry key table.
/// `positions` lists the non-ghost slots (the ghost's derivative is zero).
template <class Real> struct RIHessJobs {
  std::vector<std::array<int, 4>> pats; ///< the offset patterns, in a fixed order
  std::vector<int> idxkey;              ///< key(o) -> pattern index, or -1
  std::vector<ShellPair<Real>> plist;
  std::vector<std::pair<int, int>> quartets;
  std::vector<int> ent;                  ///< npat entries per job (-1 if absent)
  std::vector<int> jl0, jl1, jl2, jl3;   ///< base momenta
  std::vector<Real> ja0, ja1, ja2, ja3;  ///< exponents
  std::vector<int> jo0, jo1, jo2;        ///< AO offsets for the coefficient
  std::vector<int> jt0, jt1, jt2;        ///< Hessian centre index per position
  int npat{0}, njob{0};

  static int key(const std::array<int, 4> &o) {
    return (((o[0] + 2) * 5 + (o[1] + 2)) * 5 + (o[2] + 2)) * 5 + (o[3] + 2);
  }
  /// Build the pattern list for the given non-ghost positions.
  void build_patterns(const std::vector<int> &positions) {
    pats.clear();
    pats.push_back({0, 0, 0, 0});
    for (int p : positions) {
      std::array<int, 4> a{}, b{};
      a[p] = 2;
      b[p] = -2;
      pats.push_back(a);
      pats.push_back(b);
    }
    const int sg[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    for (std::size_t i = 0; i < positions.size(); ++i)
      for (std::size_t j = i + 1; j < positions.size(); ++j)
        for (int s = 0; s < 4; ++s) {
          std::array<int, 4> o{};
          o[positions[i]] = sg[s][0];
          o[positions[j]] = sg[s][1];
          pats.push_back(o);
        }
    npat = static_cast<int>(pats.size());
    idxkey.assign(625, -1);
    for (int i = 0; i < npat; ++i) idxkey[key(pats[i])] = i;
  }
  /// Add one quartet job; `tg` are the Hessian centre indices per position.
  void add(const PrimitiveShell<Real> sh[4], const int tg[3], int o0, int o1, int o2) {
    const int L[4] = {sh[0].l, sh[1].l, sh[2].l, sh[3].l};
    for (int i = 0; i < npat; ++i) {
      const auto &o = pats[i];
      bool ok = true;
      for (int t = 0; t < 4; ++t)
        if (L[t] + o[t] < 0) ok = false;
      if (!ok) { ent.push_back(-1); continue; }
      PrimitiveShell<Real> s[4] = {sh[0], sh[1], sh[2], sh[3]};
      for (int t = 0; t < 4; ++t) s[t].l += o[t];
      const int ib = static_cast<int>(plist.size());
      plist.push_back(make_pair(s[0], s[1]));
      const int ik = static_cast<int>(plist.size());
      plist.push_back(make_pair(s[2], s[3]));
      ent.push_back(static_cast<int>(quartets.size()));
      quartets.push_back({ib, ik});
    }
    jl0.push_back(L[0]); jl1.push_back(L[1]); jl2.push_back(L[2]); jl3.push_back(L[3]);
    ja0.push_back(sh[0].alpha); ja1.push_back(sh[1].alpha);
    ja2.push_back(sh[2].alpha); ja3.push_back(sh[3].alpha);
    jo0.push_back(o0); jo1.push_back(o1); jo2.push_back(o2);
    jt0.push_back(tg[0]); jt1.push_back(tg[1]); jt2.push_back(tg[2]);
    ++njob;
  }
};

/// Run the RI Hessian jobs: the quartet_pos_hess body on device, over the
/// allowed (p,q) position pairs, accumulating cf * d2 into the (3 ncen)^2
/// Hessian with atomics. Mode selects the coefficient exactly as the gradient
/// digest does (0: D_mn gamma_P, 1: -1/2 gamma_P gamma_Q, 2: c3, 3: c2).
template <class Real, int Mode>
void ri_hess_digest(RIHessJobs<Real> &jobs, const std::vector<int> &positions,
                    const TGrid<Real> &grid, Kokkos::View<const Real *> Dd, int nao,
                    Kokkos::View<const Real *> gam, Kokkos::View<const Real *> c3v,
                    Kokkos::View<const Real *> c2v, int naux, int dim,
                    Kokkos::View<Real *> H) {
  if (jobs.njob == 0) return;
  auto tab = make_pair_table(jobs.plist);
  auto batch = make_batch(tab, jobs.quartets);
  QuartetWorkspace<Real> ws;
  Kokkos::View<Real *> out("intti::rih::out", batch.nout_total);
  eri_quartets(tab, batch, grid, out, ws);
  auto entv = to_device(jobs.ent, "rih::ent");
  auto idxv = to_device(jobs.idxkey, "rih::idx");
  auto posv = to_device(positions, "rih::pos");
  auto l0 = to_device(jobs.jl0, "rih::l0"), l1 = to_device(jobs.jl1, "rih::l1");
  auto l2 = to_device(jobs.jl2, "rih::l2"), l3 = to_device(jobs.jl3, "rih::l3");
  auto a0 = to_device(jobs.ja0, "rih::a0"), a1 = to_device(jobs.ja1, "rih::a1");
  auto a2 = to_device(jobs.ja2, "rih::a2"), a3 = to_device(jobs.ja3, "rih::a3");
  auto o0v = to_device(jobs.jo0, "rih::o0"), o1v = to_device(jobs.jo1, "rih::o1");
  auto o2v = to_device(jobs.jo2, "rih::o2");
  auto t0 = to_device(jobs.jt0, "rih::t0"), t1 = to_device(jobs.jt1, "rih::t1");
  auto t2 = to_device(jobs.jt2, "rih::t2");
  auto offv = batch.out_offset;
  const int npat = jobs.npat, njob = jobs.njob;
  const int npos = static_cast<int>(positions.size());
  Kokkos::parallel_for(
      "intti::rih::digest", Kokkos::RangePolicy<>(0, njob), KOKKOS_LAMBDA(int j) {
        const int L[4] = {l0(j), l1(j), l2(j), l3(j)};
        const Real al[4] = {a0(j), a1(j), a2(j), a3(j)};
        const int nc[4] = {ncart(L[0]), ncart(L[1]), ncart(L[2]), ncart(L[3])};
        const int tg[3] = {t0(j), t1(j), t2(j)};
        const int oa = o0v(j), ob = o1v(j), oc = o2v(j);
        auto rawval = [&](const int o[4], const int m[4][3]) -> Real {
          const int kk = (((o[0] + 2) * 5 + (o[1] + 2)) * 5 + (o[2] + 2)) * 5 + (o[3] + 2);
          const int pi2 = idxv(kk);
          if (pi2 < 0) return Real(0);
          const int e2 = entv(j * npat + pi2);
          if (e2 < 0) return Real(0);
          int nn[4], id[4];
          for (int t = 0; t < 4; ++t) {
            const int lt = L[t] + o[t];
            if (m[t][0] < 0 || m[t][1] < 0 || m[t][2] < 0 ||
                m[t][0] + m[t][1] + m[t][2] != lt)
              return Real(0);
            nn[t] = ncart(lt);
            id[t] = comp_index(lt, m[t][0], m[t][1]);
          }
          return out(offv(e2) +
                     ((((static_cast<std::size_t>(id[0]) * nn[1] + id[1]) * nn[2] + id[2]) *
                       nn[3]) + id[3]));
        };
        int bm[4][3];
        int k[4];
        for (k[0] = 0; k[0] < nc[0]; ++k[0])
          for (k[1] = 0; k[1] < nc[1]; ++k[1])
            for (k[2] = 0; k[2] < nc[2]; ++k[2])
              for (k[3] = 0; k[3] < nc[3]; ++k[3]) {
                Real cf;
                if constexpr (Mode == 0)
                  cf = Dd(static_cast<std::size_t>(oa + k[0]) * nao + ob + k[1]) * gam(oc + k[2]);
                else if constexpr (Mode == 1)
                  cf = Real(-0.5) * gam(oa + k[0]) * gam(oc + k[2]);
                else if constexpr (Mode == 2)
                  cf = c3v(((static_cast<std::size_t>(oa + k[0]) * nao + ob + k[1]) * naux) +
                           oc + k[2]);
                else
                  cf = c2v(static_cast<std::size_t>(oa + k[0]) * naux + oc + k[2]);
                if (cf == Real(0)) continue;
                for (int t = 0; t < 4; ++t) cart_comp(L[t], k[t], bm[t][0], bm[t][1], bm[t][2]);
                auto mset = [&](int p, const int mp[3], int q, const int mq[3], int mm[4][3]) {
                  for (int r = 0; r < 4; ++r)
                    for (int t = 0; t < 3; ++t) mm[r][t] = bm[r][t];
                  for (int t = 0; t < 3; ++t) mm[p][t] = mp[t];
                  if (q != p)
                    for (int t = 0; t < 3; ++t) mm[q][t] = mq[t];
                };
                for (int pi = 0; pi < npos; ++pi)
                  for (int qi = 0; qi < npos; ++qi) {
                    const int p = posv(pi), q = posv(qi);
                    const Real ap = al[p], aq = al[q];
                    for (int e = 0; e < 3; ++e)
                      for (int f = 0; f < 3; ++f) {
                        Real d2 = 0;
                        int mm[4][3];
                        if (p == q) {
                          const int *mp = bm[p];
                          const int de = (e == f) ? 1 : 0;
                          int T[4][3];
                          for (int t = 0; t < 3; ++t)
                            T[0][t] = T[1][t] = T[2][t] = T[3][t] = mp[t];
                          T[0][e] += 1; T[0][f] += 1;
                          T[1][f] += 1; T[1][e] -= 1;
                          T[2][e] += 1; T[2][f] -= 1;
                          T[3][e] -= 1; T[3][f] -= 1;
                          int o2a[4] = {0, 0, 0, 0}, o0a[4] = {0, 0, 0, 0}, om2[4] = {0, 0, 0, 0};
                          o2a[p] = 2;
                          om2[p] = -2;
                          mset(p, T[0], q, T[0], mm);
                          d2 += 2 * ap * (2 * ap * rawval(o2a, mm));
                          mset(p, T[1], q, T[1], mm);
                          d2 += 2 * ap * (-(Real(mp[e]) + de) * rawval(o0a, mm));
                          mset(p, T[2], q, T[2], mm);
                          d2 += -Real(mp[f]) * (2 * ap * rawval(o0a, mm));
                          mset(p, T[3], q, T[3], mm);
                          d2 += -Real(mp[f]) * (-(Real(mp[e]) - de) * rawval(om2, mm));
                        } else {
                          const int *mp = bm[p], *mq = bm[q];
                          int Pe1[3], Pe0[3], Qf1[3], Qf0[3];
                          for (int t = 0; t < 3; ++t) {
                            Pe1[t] = mp[t]; Pe0[t] = mp[t];
                            Qf1[t] = mq[t]; Qf0[t] = mq[t];
                          }
                          Pe1[e] += 1; Pe0[e] -= 1;
                          Qf1[f] += 1; Qf0[f] -= 1;
                          int oo[4];
                          auto seto = [&](int dp, int dq) {
                            for (int t = 0; t < 4; ++t) oo[t] = 0;
                            oo[p] += dp;
                            oo[q] += dq;
                          };
                          mset(p, Pe1, q, Qf1, mm); seto(1, 1);
                          d2 += 4 * ap * aq * rawval(oo, mm);
                          mset(p, Pe1, q, Qf0, mm); seto(1, -1);
                          d2 += -2 * ap * Real(mq[f]) * rawval(oo, mm);
                          mset(p, Pe0, q, Qf1, mm); seto(-1, 1);
                          d2 += -Real(mp[e]) * 2 * aq * rawval(oo, mm);
                          mset(p, Pe0, q, Qf0, mm); seto(-1, -1);
                          d2 += Real(mp[e]) * Real(mq[f]) * rawval(oo, mm);
                        }
                        if (d2 != Real(0))
                          Kokkos::atomic_add(
                              &H((3 * tg[pi] + e) * static_cast<std::size_t>(dim) + 3 * tg[qi] + f),
                              cf * d2);
                      }
                  }
              }
      });
}

/// Second geometric derivative d^2/dR_{p,e} dR_{q,f} of the quartet
/// (s0 s1 | s2 s3), contracted with coeff(k0,k1,k2,k3) over the components,
/// returned as out[e][f]. The MD centre-shift applied twice (l+/-2 for p==q,
/// l+/-1 x l+/-1 for p!=q); p, q must be non-ghost positions.
template <class Real, class CoeffFn>
void quartet_pos_hess(const PrimitiveShell<Real> &s0, const PrimitiveShell<Real> &s1,
                      const PrimitiveShell<Real> &s2, const PrimitiveShell<Real> &s3,
                      int p, int q, const TGrid<Real> &grid, CoeffFn coeff,
                      Real out[3][3]) {
  const PrimitiveShell<Real> sh[4] = {s0, s1, s2, s3};
  const int L[4] = {s0.l, s1.l, s2.l, s3.l};
  const int nc[4] = {ncart(L[0]), ncart(L[1]), ncart(L[2]), ncart(L[3])};
  const Real ap = sh[p].alpha, aq = sh[q].alpha;
  std::map<std::array<int, 4>, std::vector<Real>> cache;
  auto block = [&](const std::array<int, 4> &o) -> const std::vector<Real> * {
    for (int i = 0; i < 4; ++i)
      if (L[i] + o[i] < 0) return nullptr;
    auto it = cache.find(o);
    if (it == cache.end()) {
      PrimitiveShell<Real> s[4] = {s0, s1, s2, s3};
      for (int i = 0; i < 4; ++i) s[i].l += o[i];
      it = cache.emplace(o, detail::eri_block4(s[0], s[1], s[2], s[3], grid)).first;
    }
    return &it->second;
  };
  auto rawval = [&](const std::array<int, 4> &o, const int m[4][3]) -> Real {
    const auto *blk = block(o);
    if (!blk) return Real(0);
    int nn[4], id[4];
    for (int i = 0; i < 4; ++i) {
      const int li = L[i] + o[i];
      if (m[i][0] < 0 || m[i][1] < 0 || m[i][2] < 0 || m[i][0] + m[i][1] + m[i][2] != li)
        return Real(0);
      nn[i] = ncart(li);
      id[i] = detail::comp_index(li, m[i][0], m[i][1]);
    }
    return (*blk)[((static_cast<std::size_t>(id[0]) * nn[1] + id[1]) * nn[2] + id[2]) * nn[3] +
                  id[3]];
  };
  for (int e = 0; e < 3; ++e)
    for (int f = 0; f < 3; ++f) out[e][f] = Real(0);
  int bm[4][3];
  int k[4];
  for (k[0] = 0; k[0] < nc[0]; ++k[0])
    for (k[1] = 0; k[1] < nc[1]; ++k[1])
      for (k[2] = 0; k[2] < nc[2]; ++k[2])
        for (k[3] = 0; k[3] < nc[3]; ++k[3]) {
          const Real cf = coeff(k[0], k[1], k[2], k[3]);
          if (cf == Real(0)) continue;
          for (int i = 0; i < 4; ++i) cart_comp(L[i], k[i], bm[i][0], bm[i][1], bm[i][2]);
          auto set = [&](int pp, const int mp[3], int qq, const int mq[3], int mm[4][3]) {
            for (int r = 0; r < 4; ++r)
              for (int t = 0; t < 3; ++t) mm[r][t] = bm[r][t];
            for (int t = 0; t < 3; ++t) mm[pp][t] = mp[t];
            if (qq != pp)
              for (int t = 0; t < 3; ++t) mm[qq][t] = mq[t];
          };
          for (int e = 0; e < 3; ++e)
            for (int f = 0; f < 3; ++f) {
              Real d2 = 0;
              int mm[4][3];
              if (p == q) {
                const int *mp = bm[p];
                const int de = (e == f) ? 1 : 0;
                int T[4][3];
                for (int t = 0; t < 3; ++t)
                  T[0][t] = T[1][t] = T[2][t] = T[3][t] = mp[t];
                T[0][e] += 1; T[0][f] += 1;
                T[1][f] += 1; T[1][e] -= 1;
                T[2][e] += 1; T[2][f] -= 1;
                T[3][e] -= 1; T[3][f] -= 1;
                const std::array<int, 4> o2 = {p == 0 ? 2 : 0, p == 1 ? 2 : 0, p == 2 ? 2 : 0,
                                               p == 3 ? 2 : 0};
                const std::array<int, 4> o0 = {0, 0, 0, 0};
                const std::array<int, 4> om2 = {p == 0 ? -2 : 0, p == 1 ? -2 : 0,
                                                p == 2 ? -2 : 0, p == 3 ? -2 : 0};
                set(p, T[0], q, T[0], mm);
                d2 += 2 * ap * (2 * ap * rawval(o2, mm));
                set(p, T[1], q, T[1], mm);
                d2 += 2 * ap * (-(Real(mp[e]) + de) * rawval(o0, mm));
                set(p, T[2], q, T[2], mm);
                d2 += -Real(mp[f]) * (2 * ap * rawval(o0, mm));
                set(p, T[3], q, T[3], mm);
                d2 += -Real(mp[f]) * (-(Real(mp[e]) - de) * rawval(om2, mm));
              } else {
                const int *mp = bm[p], *mq = bm[q];
                int Pe1[3], Pe0[3], Qf1[3], Qf0[3];
                for (int t = 0; t < 3; ++t) {
                  Pe1[t] = mp[t]; Pe0[t] = mp[t];
                  Qf1[t] = mq[t]; Qf0[t] = mq[t];
                }
                Pe1[e] += 1; Pe0[e] -= 1;
                Qf1[f] += 1; Qf0[f] -= 1;
                auto oo = [&](int dp, int dq) {
                  std::array<int, 4> o = {0, 0, 0, 0};
                  o[p] += dp; o[q] += dq;
                  return o;
                };
                set(p, Pe1, q, Qf1, mm);
                d2 += 4 * ap * aq * rawval(oo(1, 1), mm);
                set(p, Pe1, q, Qf0, mm);
                d2 += -2 * ap * Real(mq[f]) * rawval(oo(1, -1), mm);
                set(p, Pe0, q, Qf1, mm);
                d2 += -Real(mp[e]) * 2 * aq * rawval(oo(-1, 1), mm);
                set(p, Pe0, q, Qf0, mm);
                d2 += Real(mp[e]) * Real(mq[f]) * rawval(oo(-1, -1), mm);
              }
              out[e][f] += cf * d2;
            }
        }
}

} // namespace detail

/// Geometric gradient of the RI Coulomb energy E_J = 1/2 sum_mn D_mn J_mn
/// (J from ri_jk) w.r.t. the orbital- and auxiliary-shell centres. D is the
/// nao x nao symmetric AO density (row-major). tau_lin drops metric
/// eigenvalues below tau_lin*max, matching ri_fit.
template <class Real>
RIGrad<Real> ri_j_gradient(const ShellBasis<Real> &orb, const ShellBasis<Real> &aux,
                           const Real *D, const TGrid<Real> &grid,
                           Real tau_lin = Real(1e-10), int aux_tile_shells = 0) {
  const int nao = orb.nao, naux = aux.nao;
  auto M = coulomb_2c(aux, grid);             // naux x naux
  // The three-centre tensor is consumed exactly once here, as d = T^T D, so it
  // is taken one auxiliary tile at a time and never materialised: nao^2 x naux
  // is 32 GB at nao = 1000, naux = 4000.
  const int nsa_t = static_cast<int>(aux.shells.size());
  if (aux_tile_shells < 1) aux_tile_shells = nsa_t;
  std::vector<Real> d(naux, Real(0));
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  for (int A0 = 0; A0 < nsa_t; A0 += aux_tile_shells) {
    const int A1 = std::min(A0 + aux_tile_shells, nsa_t);
    const int p0 = aux.ao_off[A0], blk = aux.ao_off[A1] - p0;
    const auto Tblk = coulomb_3c_auxblock(orb, aux, grid, A0, A1);
    detail::gemm('T', 'N', blk, 1, static_cast<int>(N), Real(1), Tblk.data(), blk, D, 1,
                 Real(0), d.data() + p0, 1);
  }
  // gamma = M^{-1} d via the eigendecomposition (pseudo-inverse with cutoff)
  std::vector<Real> V = M, eval(naux);
  detail::syevd(naux, V.data(), eval.data()); // V: eigenvectors (columns, col-major)
  Real emax = 0;
  for (Real e : eval) emax = std::max(emax, e);
  std::vector<Real> gamma(naux, Real(0));
  for (int kk = 0; kk < naux; ++kk) {
    if (eval[kk] <= tau_lin * emax) continue;
    Real vd = 0;
    for (int P = 0; P < naux; ++P) vd += V[kk * naux + P] * d[P];
    const Real s = vd / eval[kk];
    for (int P = 0; P < naux; ++P) gamma[P] += s * V[kk * naux + P];
  }

  RIGrad<Real> g;
  g.forb.assign(orb.shells.size(), {Real(0), Real(0), Real(0)});
  g.faux.assign(aux.shells.size(), {Real(0), Real(0), Real(0)});
  auto Dm = [&](int i, int j) { return D[static_cast<std::size_t>(i) * nao + j]; };

  // Term A: sum_P gamma_P sum_mn D_mn d(mn|P)/dx, quartet (m n | P ghost)
  const int nso = static_cast<int>(orb.shells.size());
  const int nsa = static_cast<int>(aux.shells.size());

  if constexpr (kokkos_scalar_v<Real>) {
    // Device path: the fit (gemm/syevd) stays on the host -- it is dense linear
    // algebra, not integrals -- while both derivative terms are batched into one
    // eri_quartets call each and digested on device. Forces are accumulated into
    // a single [orbital shells, auxiliary shells] array and split on the way out.
    detail::RIGradJobs<Real> jobA, jobB;
    for (int m = 0; m < nso; ++m)
      for (int n = 0; n < nso; ++n)
        for (int a = 0; a < nsa; ++a) {
          const PrimitiveShell<Real> sh[4] = {orb.shells[m], orb.shells[n], aux.shells[a],
                                              detail::ghost_shell(aux.shells[a])};
          for (int pos = 0; pos < 3; ++pos) {
            const int tgt = (pos == 0) ? m : (pos == 1) ? n : nso + a;
            jobA.add(sh, pos, tgt, orb.ao_off[m], orb.ao_off[n], aux.ao_off[a]);
          }
        }
    for (int a = 0; a < nsa; ++a)
      for (int b = 0; b < nsa; ++b) {
        const PrimitiveShell<Real> sh[4] = {aux.shells[a], detail::ghost_shell(aux.shells[a]),
                                            aux.shells[b], detail::ghost_shell(aux.shells[b])};
        for (int pos : {0, 2}) {
          const int tgt = (pos == 0) ? nso + a : nso + b;
          jobB.add(sh, pos, tgt, aux.ao_off[a], 0, aux.ao_off[b]);
        }
      }
    auto Dd = detail::to_device(D, static_cast<std::size_t>(nao) * nao, "rig::D");
    auto gam = detail::to_device(gamma, "rig::gamma");
    Kokkos::View<const Real *> none;
    Kokkos::View<Real *[3], Kokkos::LayoutLeft> force("rig::force", nso + nsa);
    detail::ri_grad_digest<Real, 0>(jobA, grid, Dd, nao, gam, none, none, naux, force);
    detail::ri_grad_digest<Real, 1>(jobB, grid, Dd, nao, gam, none, none, naux, force);
    auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, force);
    for (int s = 0; s < nso; ++s)
      for (int e = 0; e < 3; ++e) g.forb[s][e] = hf(s, e);
    for (int s = 0; s < nsa; ++s)
      for (int e = 0; e < 3; ++e) g.faux[s][e] = hf(nso + s, e);
    return g;
  }

  for (int m = 0; m < nso; ++m)
    for (int n = 0; n < nso; ++n)
      for (int a = 0; a < nsa; ++a) {
        const auto &sm = orb.shells[m], &sn = orb.shells[n], &sP = aux.shells[a];
        const auto gh = detail::ghost_shell(sP);
        const int om = orb.ao_off[m], on = orb.ao_off[n], oP = aux.ao_off[a];
        const int nn = ncart(sn.l), nPc = ncart(sP.l);
        auto coeff = [&](int km, int kn, int kP, int) {
          return Dm(om + km, on + kn) * gamma[oP + kP];
        };
        (void)nn;
        (void)nPc;
        const int posshell[3] = {m, n, a};
        for (int pos = 0; pos < 3; ++pos) {
          Real out[3] = {0, 0, 0};
          detail::quartet_pos_grad(sm, sn, sP, gh, pos, grid, coeff, out);
          auto &F = (pos == 2) ? g.faux[a] : g.forb[posshell[pos]];
          for (int e = 0; e < 3; ++e) F[e] += out[e];
        }
      }

  // Term B: -1/2 sum_PQ gamma_P gamma_Q dM_PQ/dx, quartet (P ghost | Q ghost)
  for (int a = 0; a < nsa; ++a)
    for (int b = 0; b < nsa; ++b) {
      const auto &sP = aux.shells[a], &sQ = aux.shells[b];
      const auto ghP = detail::ghost_shell(sP), ghQ = detail::ghost_shell(sQ);
      const int oP = aux.ao_off[a], oQ = aux.ao_off[b];
      auto coeff = [&](int kP, int, int kQ, int) {
        return Real(-0.5) * gamma[oP + kP] * gamma[oQ + kQ];
      };
      for (int pos : {0, 2}) {
        Real out[3] = {0, 0, 0};
        detail::quartet_pos_grad(sP, ghP, sQ, ghQ, pos, grid, coeff, out);
        auto &F = (pos == 0) ? g.faux[a] : g.faux[b];
        for (int e = 0; e < 3; ++e) F[e] += out[e];
      }
    }
  return g;
}

/// Geometric Hessian of the RI Coulomb energy E_J, as a (3 ncen) x (3 ncen)
/// matrix with ncen = (#orbital shells) + (#auxiliary shells), ordered orbital
/// shells first then auxiliary shells (Cartesian-minor).
///
/// E_J = 1/2 d^T M^{-1} d with gamma = M^{-1} d, so differentiating twice gives
///   d^2E_J/dxdy = gamma^T d_xy - 1/2 gamma^T M_xy gamma + r_x^T M^{-1} r_y,
///   r_x = d_x - M_x gamma  ( = M gamma_x ).
/// This is just the exact second derivative of that expression -- nothing
/// exotic, and the standard density-fitting gradient/Hessian algebra. Its
/// structural content is the VARIATIONAL / 2n+1 rule: because gamma is the
/// stationary point of the fitting functional, the GRADIENT needs no gamma_x at
/// all, and the HESSIAN needs only gamma_x (through r_x), never gamma_xy.
/// (Earlier revisions of this comment called it "the envelope form"; that is
/// loose -- the envelope theorem names the first-order statement, and the
/// second-order one is the 2n+1 rule, which in fact would allow up to the THIRD
/// derivative from gamma_x alone.) The caller maps shells to atoms and sums.
template <class Real>
std::vector<Real> ri_j_hessian(const ShellBasis<Real> &orb, const ShellBasis<Real> &aux,
                               const Real *D, const TGrid<Real> &grid,
                               Real tau_lin = Real(1e-10), int aux_tile_shells = 0) {
  const int nao = orb.nao, naux = aux.nao;
  const int nso = static_cast<int>(orb.shells.size());
  const int nsa = static_cast<int>(aux.shells.size());
  const int ncen = nso + nsa, dim = 3 * ncen;
  auto M = coulomb_2c(aux, grid);
  // one auxiliary tile at a time; the tensor is used once, as d = T^T D
  const int nsa_t = static_cast<int>(aux.shells.size());
  if (aux_tile_shells < 1) aux_tile_shells = nsa_t;
  auto Dm = [&](int i, int j) { return D[static_cast<std::size_t>(i) * nao + j]; };
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  // d_P and M^{-1}: d_P = sum_mn (mn|P) D_mn = (T^T D)[P]
  std::vector<Real> d(naux, Real(0));
  for (int A0 = 0; A0 < nsa_t; A0 += aux_tile_shells) {
    const int A1 = std::min(A0 + aux_tile_shells, nsa_t);
    const int p0 = aux.ao_off[A0], blk_ = aux.ao_off[A1] - p0;
    const auto Tblk = coulomb_3c_auxblock(orb, aux, grid, A0, A1);
    detail::gemm('T', 'N', blk_, 1, static_cast<int>(N), Real(1), Tblk.data(), blk_, D, 1,
                 Real(0), d.data() + p0, 1);
  }
  std::vector<Real> Vv = M, eval(naux);
  detail::syevd(naux, Vv.data(), eval.data());
  Real emax = 0;
  for (Real e : eval) emax = std::max(emax, e);
  std::vector<Real> Minv(static_cast<std::size_t>(naux) * naux, Real(0)), gamma(naux, Real(0));
  for (int kk = 0; kk < naux; ++kk) {
    if (eval[kk] <= tau_lin * emax) continue;
    const Real inv = Real(1) / eval[kk];
    Real vd = 0;
    for (int P = 0; P < naux; ++P) vd += Vv[kk * naux + P] * d[P];
    for (int P = 0; P < naux; ++P) {
      gamma[P] += inv * vd * Vv[kk * naux + P];
      for (int Q = 0; Q < naux; ++Q)
        Minv[P * naux + Q] += inv * Vv[kk * naux + P] * Vv[kk * naux + Q];
    }
  }
  auto cshell = [&](bool isaux, int s) { return isaux ? nso + s : s; };

  // first-derivative residual tensor r[x][P] = d_x[P] - (M_x gamma)[P]
  std::vector<Real> r(static_cast<std::size_t>(dim) * naux, Real(0));
  auto radd = [&](int cs, int dir, int P, Real v) {
    r[(static_cast<std::size_t>(3 * cs + dir)) * naux + P] += v;
  };
  const auto ghost = [&](const PrimitiveShell<Real> &s) { return detail::ghost_shell(s); };
  // d_x: 3-centre (m n | a ghost)
  for (int m = 0; m < nso; ++m)
    for (int n = 0; n < nso; ++n)
      for (int a = 0; a < nsa; ++a) {
        const auto &sm = orb.shells[m], &sn = orb.shells[n], &sP = aux.shells[a];
        const auto gh = ghost(sP);
        const int om = orb.ao_off[m], on = orb.ao_off[n], oP = aux.ao_off[a];
        const int nm = ncart(sm.l), nn = ncart(sn.l), nP = ncart(sP.l);
        const int cs[3] = {cshell(false, m), cshell(false, n), cshell(true, a)};
        for (int pos = 0; pos < 3; ++pos)
          for (int dir = 0; dir < 3; ++dir) {
            auto blk = detail::quartet_pos_deriv_block(sm, sn, sP, gh, pos, dir, grid);
            for (int km = 0; km < nm; ++km)
              for (int kn = 0; kn < nn; ++kn) {
                const Real dmn = Dm(om + km, on + kn);
                if (dmn == Real(0)) continue;
                for (int kP = 0; kP < nP; ++kP)
                  radd(cs[pos], dir, oP + kP,
                       dmn * blk[((static_cast<std::size_t>(km) * nn + kn) * nP + kP)]);
              }
          }
      }
  // -(M_x gamma): 2-centre (a ghost | b ghost), free aux index = a
  for (int a = 0; a < nsa; ++a)
    for (int b = 0; b < nsa; ++b) {
      const auto &sA = aux.shells[a], &sB = aux.shells[b];
      const auto ghA = ghost(sA), ghB = ghost(sB);
      const int oA = aux.ao_off[a], oB = aux.ao_off[b];
      const int nA = ncart(sA.l), nB = ncart(sB.l);
      for (int pos : {0, 2})
        for (int dir = 0; dir < 3; ++dir) {
          auto blk = detail::quartet_pos_deriv_block(sA, ghA, sB, ghB, pos, dir, grid);
          const int cs = (pos == 0) ? cshell(true, a) : cshell(true, b);
          for (int ka = 0; ka < nA; ++ka) {
            Real acc = 0;
            for (int kb = 0; kb < nB; ++kb)
              acc += blk[static_cast<std::size_t>(ka) * nB + kb] * gamma[oB + kb];
            radd(cs, dir, oA + ka, -acc);
          }
        }
    }
  // s[x] = M^{-1} r[x]: s[x][P] = sum_Q r[x][Q] Minv[P][Q] = (r Minv^T)[x][P]
  std::vector<Real> s(static_cast<std::size_t>(dim) * naux, Real(0));
  detail::gemm('N', 'T', dim, naux, naux, Real(1), r.data(), naux, Minv.data(), naux,
               Real(0), s.data(), naux);

  std::vector<Real> H(static_cast<std::size_t>(dim) * dim, Real(0));
  auto Hadd = [&](int x, int y, Real v) { H[static_cast<std::size_t>(x) * dim + y] += v; };
  // response term H[x][y] = sum_P r[x][P] s[y][P] = (r s^T)[x][y]
  detail::gemm('N', 'T', dim, dim, naux, Real(1), r.data(), naux, s.data(), naux, Real(1),
               H.data(), dim);
  if constexpr (kokkos_scalar_v<Real>) {
    // Device direct terms. The response term r^T M^{-1} r above is dense linear
    // algebra over first derivatives and stays on the host; H already holds it,
    // so the device contribution is ADDED, not assigned.
    detail::RIHessJobs<Real> jA, jB;
    const std::vector<int> posA{0, 1, 2}, posB{0, 2};
    jA.build_patterns(posA);
    for (int m = 0; m < nso; ++m)
      for (int n = 0; n < nso; ++n)
        for (int a = 0; a < nsa; ++a) {
          const PrimitiveShell<Real> sh[4] = {orb.shells[m], orb.shells[n], aux.shells[a],
                                              ghost(aux.shells[a])};
          const int tg[3] = {cshell(false, m), cshell(false, n), cshell(true, a)};
          jA.add(sh, tg, orb.ao_off[m], orb.ao_off[n], aux.ao_off[a]);
        }
    jB.build_patterns(posB);
    for (int a = 0; a < nsa; ++a)
      for (int b = 0; b < nsa; ++b) {
        const PrimitiveShell<Real> sh[4] = {aux.shells[a], ghost(aux.shells[a]),
                                            aux.shells[b], ghost(aux.shells[b])};
        const int tg[3] = {cshell(true, a), cshell(true, b), 0};
        jB.add(sh, tg, aux.ao_off[a], 0, aux.ao_off[b]);
      }
    auto Dd = detail::to_device(D, static_cast<std::size_t>(nao) * nao, "rih::D");
    auto gam = detail::to_device(gamma, "rih::gamma");
    Kokkos::View<const Real *> none;
    Kokkos::View<Real *> Hd("rih::H", static_cast<std::size_t>(dim) * dim);
    detail::ri_hess_digest<Real, 0>(jA, posA, grid, Dd, nao, gam, none, none, naux, dim, Hd);
    detail::ri_hess_digest<Real, 1>(jB, posB, grid, Dd, nao, gam, none, none, naux, dim, Hd);
    auto hh = detail::to_host(Hd);
    for (std::size_t i = 0; i < H.size(); ++i) H[i] += hh[i];
    return H;
  }

  // direct term 1: gamma^T d_xy, 3-centre with coeff D_mn gamma_P
  for (int m = 0; m < nso; ++m)
    for (int n = 0; n < nso; ++n)
      for (int a = 0; a < nsa; ++a) {
        const auto &sm = orb.shells[m], &sn = orb.shells[n], &sP = aux.shells[a];
        const auto gh = ghost(sP);
        const int om = orb.ao_off[m], on = orb.ao_off[n], oP = aux.ao_off[a];
        auto coeff = [&](int km, int kn, int kP, int) {
          return Dm(om + km, on + kn) * gamma[oP + kP];
        };
        const int cs[3] = {cshell(false, m), cshell(false, n), cshell(true, a)};
        for (int p = 0; p < 3; ++p)
          for (int q = 0; q < 3; ++q) {
            Real o[3][3];
            detail::quartet_pos_hess(sm, sn, sP, gh, p, q, grid, coeff, o);
            for (int e = 0; e < 3; ++e)
              for (int f = 0; f < 3; ++f) Hadd(3 * cs[p] + e, 3 * cs[q] + f, o[e][f]);
          }
      }
  // direct term 2: -1/2 gamma^T M_xy gamma, 2-centre with coeff -1/2 gamma gamma
  for (int a = 0; a < nsa; ++a)
    for (int b = 0; b < nsa; ++b) {
      const auto &sA = aux.shells[a], &sB = aux.shells[b];
      const auto ghA = ghost(sA), ghB = ghost(sB);
      const int oA = aux.ao_off[a], oB = aux.ao_off[b];
      auto coeff = [&](int ka, int, int kb, int) {
        return Real(-0.5) * gamma[oA + ka] * gamma[oB + kb];
      };
      const int cs[2] = {cshell(true, a), cshell(true, b)};
      const int poss[2] = {0, 2};
      for (int pi = 0; pi < 2; ++pi)
        for (int qi = 0; qi < 2; ++qi) {
          Real o[3][3];
          detail::quartet_pos_hess(sA, ghA, sB, ghB, poss[pi], poss[qi], grid, coeff, o);
          for (int e = 0; e < 3; ++e)
            for (int f = 0; f < 3; ++f) Hadd(3 * cs[pi] + e, 3 * cs[qi] + f, o[e][f]);
        }
    }
  return H;
}

/// Geometric gradient of the RI exchange energy E_K = -1/4 sum_mn D_mn K_mn
/// (K from ri_jk) w.r.t. the orbital- and auxiliary-shell centres. With the
/// density-transformed 3-index H_{sn}^Q = sum_l D_sl (ln|Q), its fit
/// G_{ab}^R = sum_Q M^{-1}_RQ H_{ab}^Q, and E_K = -1/4 sum_Q H_{ns}^Q G_{sn}^Q,
///   dE_K/dx = -1/2 sum_{lnR} [sum_s D_ls G_{ns}^R] d(ln|R)/dx
///           + 1/4  sum_{TU}   [sum_sn G_{ns}^T G_{sn}^U] d(T|U)/dx.
template <class Real>
RIGrad<Real> ri_k_gradient(const ShellBasis<Real> &orb, const ShellBasis<Real> &aux,
                           const Real *D, const TGrid<Real> &grid,
                           Real tau_lin = Real(1e-10)) {
  const int nao = orb.nao, naux = aux.nao;
  auto M = coulomb_2c(aux, grid);
  auto T = coulomb_3c(orb, aux, grid); // T[(mu*nao+la)*naux+P] = (mu la|P)
  auto Dm = [&](int i, int j) { return D[static_cast<std::size_t>(i) * nao + j]; };
  const std::size_t no2 = static_cast<std::size_t>(nao) * nao;

  // M^{-1} (pseudo-inverse with cutoff)
  std::vector<Real> V = M, eval(naux);
  detail::syevd(naux, V.data(), eval.data());
  Real emax = 0;
  for (Real e : eval) emax = std::max(emax, e);
  std::vector<Real> Minv(static_cast<std::size_t>(naux) * naux, Real(0));
  for (int k = 0; k < naux; ++k) {
    if (eval[k] <= tau_lin * emax) continue;
    const Real s = Real(1) / eval[k];
    for (int R = 0; R < naux; ++R)
      for (int Q = 0; Q < naux; ++Q) Minv[R * naux + Q] += s * V[k * naux + R] * V[k * naux + Q];
  }

  // H[(s*nao+n)*naux+Q] = sum_l D_sl (l n|Q): H[s][(n,Q)] = sum_l D[s][l] T[l][(n,Q)]
  // = (D * T_reshaped)[s][(n,Q)] with T viewed as nao x (nao*naux).
  std::vector<Real> H(no2 * naux, Real(0));
  detail::gemm('N', 'N', nao, nao * naux, nao, Real(1), D, nao, T.data(), nao * naux, Real(0),
               H.data(), nao * naux);
  // G[(a*nao+b)*naux+R] = sum_Q M^{-1}_RQ H[ab][Q] = (H Minv^T)[ab][R]
  std::vector<Real> G(no2 * naux, Real(0));
  detail::gemm('N', 'T', static_cast<int>(no2), naux, naux, Real(1), H.data(), naux, Minv.data(),
               naux, Real(0), G.data(), naux);
  // c3[(l*nao+n)*naux+R] = -1/2 sum_s D_ls G[(n*nao+s)*naux+R]. Batched over n:
  // for each n, c3[l][n][R] = -1/2 (D * G[n])[l][R], G[n] the (sig,R) slice.
  std::vector<Real> c3(no2 * naux, Real(0));
  for (int n = 0; n < nao; ++n)
    detail::gemm('N', 'N', nao, naux, nao, Real(-0.5), D, nao,
                 G.data() + static_cast<std::size_t>(n) * nao * naux, naux, Real(0),
                 c3.data() + static_cast<std::size_t>(n) * naux, nao * naux);
  // c2[T*naux+U] = 1/4 sum_{sn} G[(n*nao+s)*naux+T] G[(s*nao+n)*naux+U]. With the
  // (n,s)->(s,n) inner transpose as a permuted copy Gp, c2 = 1/4 G^T Gp.
  std::vector<Real> Gp(no2 * naux, Real(0));
  for (int n = 0; n < nao; ++n)
    for (int sig = 0; sig < nao; ++sig)
      for (int R = 0; R < naux; ++R)
        Gp[(static_cast<std::size_t>(n) * nao + sig) * naux + R] =
            G[(static_cast<std::size_t>(sig) * nao + n) * naux + R];
  std::vector<Real> c2(static_cast<std::size_t>(naux) * naux, Real(0));
  detail::gemm('T', 'N', naux, naux, static_cast<int>(no2), Real(0.25), G.data(), naux, Gp.data(),
               naux, Real(0), c2.data(), naux);

  RIGrad<Real> g;
  g.forb.assign(orb.shells.size(), {Real(0), Real(0), Real(0)});
  g.faux.assign(aux.shells.size(), {Real(0), Real(0), Real(0)});
  const int nso = static_cast<int>(orb.shells.size());
  const int nsa = static_cast<int>(aux.shells.size());

  if constexpr (kokkos_scalar_v<Real>) {
    // same batched ghost-quartet digestion as the RI-J gradient, with the RI-K
    // coefficient tensors c3 (three-centre) and c2 (two-centre).
    detail::RIGradJobs<Real> jobA, jobB;
    for (int lsh = 0; lsh < nso; ++lsh)
      for (int nsh = 0; nsh < nso; ++nsh)
        for (int a = 0; a < nsa; ++a) {
          const PrimitiveShell<Real> sh[4] = {orb.shells[lsh], orb.shells[nsh], aux.shells[a],
                                              detail::ghost_shell(aux.shells[a])};
          for (int pos = 0; pos < 3; ++pos) {
            const int tgt = (pos == 0) ? lsh : (pos == 1) ? nsh : nso + a;
            jobA.add(sh, pos, tgt, orb.ao_off[lsh], orb.ao_off[nsh], aux.ao_off[a]);
          }
        }
    for (int a = 0; a < nsa; ++a)
      for (int b = 0; b < nsa; ++b) {
        const PrimitiveShell<Real> sh[4] = {aux.shells[a], detail::ghost_shell(aux.shells[a]),
                                            aux.shells[b], detail::ghost_shell(aux.shells[b])};
        for (int pos : {0, 2}) {
          const int tgt = (pos == 0) ? nso + a : nso + b;
          jobB.add(sh, pos, tgt, aux.ao_off[a], 0, aux.ao_off[b]);
        }
      }
    auto c3d = detail::to_device(c3, "rik::c3");
    auto c2d = detail::to_device(c2, "rik::c2");
    Kokkos::View<const Real *> none;
    Kokkos::View<Real *[3], Kokkos::LayoutLeft> force("rik::force", nso + nsa);
    detail::ri_grad_digest<Real, 2>(jobA, grid, none, nao, none, c3d, c2d, naux, force);
    detail::ri_grad_digest<Real, 3>(jobB, grid, none, nao, none, c3d, c2d, naux, force);
    auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, force);
    for (int s = 0; s < nso; ++s)
      for (int e = 0; e < 3; ++e) g.forb[s][e] = hf(s, e);
    for (int s = 0; s < nsa; ++s)
      for (int e = 0; e < 3; ++e) g.faux[s][e] = hf(nso + s, e);
    return g;
  }

  // Term 1: 3-centre d(l n|R) contracted with c3, quartet (l n | R ghost)
  for (int lsh = 0; lsh < nso; ++lsh)
    for (int nsh = 0; nsh < nso; ++nsh)
      for (int a = 0; a < nsa; ++a) {
        const auto &sl = orb.shells[lsh], &sn = orb.shells[nsh], &sR = aux.shells[a];
        const auto gh = detail::ghost_shell(sR);
        const int ol = orb.ao_off[lsh], on = orb.ao_off[nsh], oR = aux.ao_off[a];
        auto coeff = [&](int kl, int kn, int kR, int) {
          return c3[((static_cast<std::size_t>(ol + kl) * nao + on + kn) * naux) + oR + kR];
        };
        const int posshell[3] = {lsh, nsh, a};
        for (int pos = 0; pos < 3; ++pos) {
          Real out[3] = {0, 0, 0};
          detail::quartet_pos_grad(sl, sn, sR, gh, pos, grid, coeff, out);
          auto &F = (pos == 2) ? g.faux[a] : g.forb[posshell[pos]];
          for (int e = 0; e < 3; ++e) F[e] += out[e];
        }
      }
  // Term 2: 2-centre d(T|U) contracted with c2, quartet (T ghost | U ghost)
  for (int a = 0; a < nsa; ++a)
    for (int b = 0; b < nsa; ++b) {
      const auto &sT = aux.shells[a], &sU = aux.shells[b];
      const auto ghT = detail::ghost_shell(sT), ghU = detail::ghost_shell(sU);
      const int oT = aux.ao_off[a], oU = aux.ao_off[b];
      auto coeff = [&](int kT, int, int kU, int) {
        return c2[(static_cast<std::size_t>(oT + kT) * naux) + oU + kU];
      };
      for (int pos : {0, 2}) {
        Real out[3] = {0, 0, 0};
        detail::quartet_pos_grad(sT, ghT, sU, ghU, pos, grid, coeff, out);
        auto &F = (pos == 0) ? g.faux[a] : g.faux[b];
        for (int e = 0; e < 3; ++e) F[e] += out[e];
      }
    }
  return g;
}

/// Geometric Hessian of the RI exchange energy E_K, as a (3 ncen) x (3 ncen)
/// matrix (orbital shells then auxiliary shells). Envelope form: direct terms
/// (coeff3, coeff2 of ri_k_gradient contracted with the integral Hessians) plus
/// the response -1/2 sum_{ab} R_x[a,b]^T M^{-1} R_y[a,b],
///   R_x[a,b,Q] = dH[a,b,Q]/dx - sum_R dM_QR/dx G[a,b,R],
/// H[a,b,Q] = sum_l D_al (l b|Q), G = M^{-1} H.
template <class Real>
std::vector<Real> ri_k_hessian(const ShellBasis<Real> &orb, const ShellBasis<Real> &aux,
                               const Real *D, const TGrid<Real> &grid,
                               Real tau_lin = Real(1e-10)) {
  const int nao = orb.nao, naux = aux.nao;
  const int nso = static_cast<int>(orb.shells.size());
  const int nsa = static_cast<int>(aux.shells.size());
  const int ncen = nso + nsa, dim = 3 * ncen;
  auto M = coulomb_2c(aux, grid);
  auto T = coulomb_3c(orb, aux, grid); // (mu la|P)
  auto Dm = [&](int i, int j) { return D[static_cast<std::size_t>(i) * nao + j]; };
  const std::size_t no2 = static_cast<std::size_t>(nao) * nao;
  // M^{-1}
  std::vector<Real> Vv = M, eval(naux);
  detail::syevd(naux, Vv.data(), eval.data());
  Real emax = 0;
  for (Real e : eval) emax = std::max(emax, e);
  std::vector<Real> Minv(static_cast<std::size_t>(naux) * naux, Real(0));
  for (int k = 0; k < naux; ++k) {
    if (eval[k] <= tau_lin * emax) continue;
    const Real s = Real(1) / eval[k];
    for (int P = 0; P < naux; ++P)
      for (int Q = 0; Q < naux; ++Q) Minv[P * naux + Q] += s * Vv[k * naux + P] * Vv[k * naux + Q];
  }
  // H[(a*nao+b)*naux+Q] = sum_l D_al (l b|Q) = (D * T_reshaped); G = M^{-1} H = H Minv^T
  std::vector<Real> Hm(no2 * naux, Real(0)), G(no2 * naux, Real(0));
  detail::gemm('N', 'N', nao, nao * naux, nao, Real(1), D, nao, T.data(), nao * naux, Real(0),
               Hm.data(), nao * naux);
  detail::gemm('N', 'T', static_cast<int>(no2), naux, naux, Real(1), Hm.data(), naux, Minv.data(),
               naux, Real(0), G.data(), naux);
  // coeff3[(l*nao+n)*naux+R] = -1/2 sum_s D_ls G[(n*nao+s)*naux+R], batched over n
  std::vector<Real> coeff3(no2 * naux, Real(0));
  for (int n = 0; n < nao; ++n)
    detail::gemm('N', 'N', nao, naux, nao, Real(-0.5), D, nao,
                 G.data() + static_cast<std::size_t>(n) * nao * naux, naux, Real(0),
                 coeff3.data() + static_cast<std::size_t>(n) * naux, nao * naux);
  // coeff2[T*naux+U] = 1/4 sum_{sn} G[(n*nao+s)*T] G[(s*nao+n)*U] = 1/4 G^T Gp,
  // Gp the (n,s)->(s,n) inner-transposed copy of G.
  std::vector<Real> Gp(no2 * naux, Real(0));
  for (int n = 0; n < nao; ++n)
    for (int sig = 0; sig < nao; ++sig)
      for (int R = 0; R < naux; ++R)
        Gp[(static_cast<std::size_t>(n) * nao + sig) * naux + R] =
            G[(static_cast<std::size_t>(sig) * nao + n) * naux + R];
  std::vector<Real> coeff2(static_cast<std::size_t>(naux) * naux, Real(0));
  detail::gemm('T', 'N', naux, naux, static_cast<int>(no2), Real(0.25), G.data(), naux, Gp.data(),
               naux, Real(0), coeff2.data(), naux);

  auto cshell = [&](bool isaux, int sidx) { return isaux ? nso + sidx : sidx; };
  const auto ghost = [&](const PrimitiveShell<Real> &sx) { return detail::ghost_shell(sx); };
  std::vector<Real> Hess(static_cast<std::size_t>(dim) * dim, Real(0));
  auto Hadd = [&](int x, int y, Real v) { Hess[static_cast<std::size_t>(x) * dim + y] += v; };

  if constexpr (kokkos_scalar_v<Real>) {
    // device direct terms (the response term below stays on the host)
    detail::RIHessJobs<Real> jA, jB;
    const std::vector<int> posA{0, 1, 2}, posB{0, 2};
    jA.build_patterns(posA);
    for (int l = 0; l < nso; ++l)
      for (int n = 0; n < nso; ++n)
        for (int a = 0; a < nsa; ++a) {
          const PrimitiveShell<Real> sh[4] = {orb.shells[l], orb.shells[n], aux.shells[a],
                                              ghost(aux.shells[a])};
          const int tg[3] = {cshell(false, l), cshell(false, n), cshell(true, a)};
          jA.add(sh, tg, orb.ao_off[l], orb.ao_off[n], aux.ao_off[a]);
        }
    jB.build_patterns(posB);
    for (int a = 0; a < nsa; ++a)
      for (int b = 0; b < nsa; ++b) {
        const PrimitiveShell<Real> sh[4] = {aux.shells[a], ghost(aux.shells[a]),
                                            aux.shells[b], ghost(aux.shells[b])};
        const int tg[3] = {cshell(true, a), cshell(true, b), 0};
        jB.add(sh, tg, aux.ao_off[a], 0, aux.ao_off[b]);
      }
    auto c3d = detail::to_device(coeff3, "rikh::c3");
    auto c2d = detail::to_device(coeff2, "rikh::c2");
    Kokkos::View<const Real *> none;
    Kokkos::View<Real *> Hd("rikh::H", static_cast<std::size_t>(dim) * dim);
    detail::ri_hess_digest<Real, 2>(jA, posA, grid, none, nao, none, c3d, c2d, naux, dim, Hd);
    detail::ri_hess_digest<Real, 3>(jB, posB, grid, none, nao, none, c3d, c2d, naux, dim, Hd);
    auto hh = detail::to_host(Hd);
    for (std::size_t i = 0; i < Hess.size(); ++i) Hess[i] += hh[i];
  } else {
  // direct terms
  for (int l = 0; l < nso; ++l)
    for (int n = 0; n < nso; ++n)
      for (int a = 0; a < nsa; ++a) {
        const auto &sl = orb.shells[l], &sn = orb.shells[n], &sR = aux.shells[a];
        const auto gh = ghost(sR);
        const int ol = orb.ao_off[l], on = orb.ao_off[n], oR = aux.ao_off[a];
        auto coeff = [&](int kl, int kn, int kR, int) {
          return coeff3[((static_cast<std::size_t>(ol + kl) * nao + on + kn) * naux) + oR + kR];
        };
        const int cs[3] = {cshell(false, l), cshell(false, n), cshell(true, a)};
        for (int p = 0; p < 3; ++p)
          for (int q = 0; q < 3; ++q) {
            Real o[3][3];
            detail::quartet_pos_hess(sl, sn, sR, gh, p, q, grid, coeff, o);
            for (int e = 0; e < 3; ++e)
              for (int f = 0; f < 3; ++f) Hadd(3 * cs[p] + e, 3 * cs[q] + f, o[e][f]);
          }
      }
  for (int a = 0; a < nsa; ++a)
    for (int b = 0; b < nsa; ++b) {
      const auto &sT = aux.shells[a], &sU = aux.shells[b];
      const auto ghT = ghost(sT), ghU = ghost(sU);
      const int oT = aux.ao_off[a], oU = aux.ao_off[b];
      auto coeff = [&](int kT, int, int kU, int) {
        return coeff2[(static_cast<std::size_t>(oT + kT) * naux) + oU + kU];
      };
      const int cs[2] = {cshell(true, a), cshell(true, b)};
      const int poss[2] = {0, 2};
      for (int pi = 0; pi < 2; ++pi)
        for (int qi = 0; qi < 2; ++qi) {
          Real o[3][3];
          detail::quartet_pos_hess(sT, ghT, sU, ghU, poss[pi], poss[qi], grid, coeff, o);
          for (int e = 0; e < 3; ++e)
            for (int f = 0; f < 3; ++f) Hadd(3 * cs[pi] + e, 3 * cs[qi] + f, o[e][f]);
        }
    }
  }

  // response: R_x[(a*nao+b)*naux+Q] = dH[a,b,Q]/dx - sum_R dM_QR/dx G[a,b,R]
  std::vector<Real> R(static_cast<std::size_t>(dim) * no2 * naux, Real(0));
  auto Ridx = [&](int x, int a, int b, int Q) {
    return (static_cast<std::size_t>(x) * no2 + (static_cast<std::size_t>(a) * nao + b)) * naux + Q;
  };
  // part 1: dH/dx = sum_l D_al d(l b|Q)/dx, 3-centre (l n | c ghost)
  for (int l = 0; l < nso; ++l)
    for (int n = 0; n < nso; ++n)
      for (int c = 0; c < nsa; ++c) {
        const auto &sl = orb.shells[l], &sn = orb.shells[n], &sc = aux.shells[c];
        const auto gh = ghost(sc);
        const int ol = orb.ao_off[l], on = orb.ao_off[n], oc = aux.ao_off[c];
        const int nl = ncart(sl.l), nn = ncart(sn.l), nP = ncart(sc.l);
        const int cs[3] = {cshell(false, l), cshell(false, n), cshell(true, c)};
        for (int pos = 0; pos < 3; ++pos)
          for (int dir = 0; dir < 3; ++dir) {
            auto blk = detail::quartet_pos_deriv_block(sl, sn, sc, gh, pos, dir, grid);
            const int xi = 3 * cs[pos] + dir;
            for (int kl = 0; kl < nl; ++kl)
              for (int kn = 0; kn < nn; ++kn)
                for (int kQ = 0; kQ < nP; ++kQ) {
                  const Real dv = blk[((static_cast<std::size_t>(kl) * nn + kn) * nP + kQ)];
                  if (dv == Real(0)) continue;
                  const int lam = ol + kl, bb = on + kn, QQ = oc + kQ;
                  for (int aa = 0; aa < nao; ++aa) {
                    const Real dal = Dm(aa, lam);
                    if (dal != Real(0)) R[Ridx(xi, aa, bb, QQ)] += dal * dv;
                  }
                }
          }
      }
  // part 2: - sum_R dM_QR/dx G[a,b,R], 2-centre (c ghost | e ghost), Q from c
  for (int c = 0; c < nsa; ++c)
    for (int ee = 0; ee < nsa; ++ee) {
      const auto &sC = aux.shells[c], &sE = aux.shells[ee];
      const auto ghC = ghost(sC), ghE = ghost(sE);
      const int oC = aux.ao_off[c], oE = aux.ao_off[ee];
      const int nC = ncart(sC.l), nE = ncart(sE.l);
      for (int pos : {0, 2})
        for (int dir = 0; dir < 3; ++dir) {
          auto blk = detail::quartet_pos_deriv_block(sC, ghC, sE, ghE, pos, dir, grid);
          const int cs = (pos == 0) ? cshell(true, c) : cshell(true, ee);
          const int xi = 3 * cs + dir;
          for (int kQ = 0; kQ < nC; ++kQ)
            for (int kR = 0; kR < nE; ++kR) {
              const Real dv = blk[static_cast<std::size_t>(kQ) * nE + kR];
              if (dv == Real(0)) continue;
              const int QQ = oC + kQ, RR = oE + kR;
              for (int aa = 0; aa < nao; ++aa)
                for (int bb = 0; bb < nao; ++bb)
                  R[Ridx(xi, aa, bb, QQ)] -=
                      dv * G[(static_cast<std::size_t>(aa) * nao + bb) * naux + RR];
            }
        }
    }
  // S = M^{-1} R (over Q): S[(x,ab)][Q] = sum_Rr R[(x,ab)][Rr] Minv[Q][Rr]
  // = (R Minv^T) with (x,ab) flattened into one dim*no2 row index.
  std::vector<Real> S(static_cast<std::size_t>(dim) * no2 * naux, Real(0));
  detail::gemm('N', 'T', static_cast<int>(static_cast<std::size_t>(dim) * no2), naux, naux,
               Real(1), R.data(), naux, Minv.data(), naux, Real(0), S.data(), naux);
  // response Hess[x][y] += -1/2 sum_{a,b,Q} R_x[b,a,Q] S_y[a,b,Q]. L_GG couples
  // (a,b) with (b,a): form the (a,b)->(b,a) inner-permuted copy Rp of R, then the
  // sum over the flattened (a,b,Q) index is the GEMM Hess += -1/2 Rp S^T.
  std::vector<Real> Rp(static_cast<std::size_t>(dim) * no2 * naux, Real(0));
  for (int x = 0; x < dim; ++x)
    for (int a = 0; a < nao; ++a)
      for (int b = 0; b < nao; ++b)
        std::copy(R.data() + Ridx(x, b, a, 0), R.data() + Ridx(x, b, a, 0) + naux,
                  Rp.data() + Ridx(x, a, b, 0));
  const int Kab = static_cast<int>(no2 * naux);
  detail::gemm('N', 'T', dim, dim, Kab, Real(-0.5), Rp.data(), Kab, S.data(), Kab, Real(1),
               Hess.data(), dim);
  return Hess;
}

/// Derivative RI Coulomb MATRICES: for each shell centre (orbital shells first,
/// then auxiliary, as in RIGrad) and Cartesian direction,
///   dJ_mn/dx = sum_P d(mn|P)/dx gamma_P + sum_P (mn|P) dgamma_P/dx.
///
/// The second term is what makes this more than a re-scatter of ri_j_gradient.
/// The GRADIENT never needs gamma_x: gamma is the stationary point of the
/// fitting functional, so by the 2n+1 rule the first-order energy is complete
/// without it. The matrix is not a stationary quantity, so gamma_x is required,
/// and it comes from differentiating M gamma = d:
///   M gamma_x = d_x - M_x gamma,
/// one metric solve per perturbation reusing the eigendecomposition already
/// formed for gamma. d_x and the first term are two contractions of the SAME
/// derivative three-centre pass -- over mn with the density, and over P with
/// gamma -- so it is accumulated once and consumed twice.
///
/// The auxiliary-shell perturbations come from translational invariance of the
/// three-centre integral, d/dP = -(d/dm + d/dn); likewise d/dQ = -d/dP for the
/// two-centre metric derivative. Those are exact and save evaluating the shifted
/// auxiliary quartets.
template <class Real>
JKDerivResult<Real> ri_j_deriv_build(const ShellBasis<Real> &orb, const ShellBasis<Real> &aux,
                                     const std::vector<JKRequest<Real>> &reqs,
                                     const TGrid<Real> &grid,
                                     Real tau_lin = Real(1e-10),
                                     int aux_tile_shells = 0) {
  const int nao = orb.nao, naux = aux.nao;
  const int nso = static_cast<int>(orb.shells.size());
  const int nsa = static_cast<int>(aux.shells.size());
  const int ncen = nso + nsa, npert = 3 * ncen;
  const int nreq = static_cast<int>(reqs.size());
  const std::size_t N = static_cast<std::size_t>(nao) * nao;

  if (aux_tile_shells < 1) aux_tile_shells = nsa;
  auto M = coulomb_2c(aux, grid);
  // The three-centre tensor is NEVER materialised: it is nao^2 x naux, the same
  // shape the reverted exchange-derivative intermediate was rejected for. It is
  // consumed in two auxiliary-tiled passes instead (the ri_j_tiled idiom), and
  // the second pass is hoisted so the tile loop is OUTSIDE the perturbation
  // loop -- tiling naively inside it would recompute the whole tensor
  // nreq x npert times.
  // metric pseudo-inverse, shared by gamma and every gamma_x
  std::vector<Real> V = M, eval(naux);
  detail::syevd(naux, V.data(), eval.data());
  Real emax = 0;
  for (Real e : eval) emax = std::max(emax, e);
  auto solve = [&](const std::vector<Real> &rhs) {
    std::vector<Real> x(naux, Real(0));
    for (int k = 0; k < naux; ++k) {
      if (eval[k] <= tau_lin * emax) continue;
      Real vd = 0;
      for (int P = 0; P < naux; ++P) vd += V[k * naux + P] * rhs[P];
      const Real sc = vd / eval[k];
      for (int P = 0; P < naux; ++P) x[P] += sc * V[k * naux + P];
    }
    return x;
  };
  // pass 1: d_r,P = sum_mn (mn|P) D_r,mn, one auxiliary tile at a time
  std::vector<std::vector<Real>> gamma(nreq);
  {
    std::vector<std::vector<Real>> d(nreq, std::vector<Real>(naux, Real(0)));
    for (int A0 = 0; A0 < nsa; A0 += aux_tile_shells) {
      const int A1 = std::min(A0 + aux_tile_shells, nsa);
      const int p0 = aux.ao_off[A0], blk = aux.ao_off[A1] - p0;
      const auto Tblk = coulomb_3c_auxblock(orb, aux, grid, A0, A1);
      for (int r = 0; r < nreq; ++r)
        detail::gemm('T', 'N', blk, 1, static_cast<int>(N), Real(1), Tblk.data(), blk,
                     reqs[r].D, 1, Real(0), d[r].data() + p0, 1);
    }
    for (int r = 0; r < nreq; ++r) gamma[r] = solve(d[r]);
  }

  // ---- derivative three-centre pass -------------------------------------
  // accumulates, per perturbation x and request r:
  //   A[r][x] (nao x nao) = sum_P d(mn|P)/dx gamma_P
  //   dx[r][x] (naux)     = sum_mn d(mn|P)/dx D_mn
  std::vector<Real> A(static_cast<std::size_t>(nreq) * npert * N, Real(0));
  std::vector<Real> dxv(static_cast<std::size_t>(nreq) * npert * naux, Real(0));
  {
    std::vector<ShellPair<Real>> plist;
    std::vector<int> pid(static_cast<std::size_t>(nso) * 3 * nso * 3, -1);
    auto orb_pair = [&](int si, int di, int sj, int dj) {
      if (orb.shells[si].l + di < 0 || orb.shells[sj].l + dj < 0) return -1;
      const std::size_t key =
          ((static_cast<std::size_t>(si) * 3 + (di + 1)) * nso + sj) * 3 + (dj + 1);
      if (pid[key] < 0) {
        PrimitiveShell<Real> a = orb.shells[si], b = orb.shells[sj];
        a.l += di;
        b.l += dj;
        plist.push_back(make_pair(a, b));
        pid[key] = static_cast<int>(plist.size()) - 1;
      }
      return pid[key];
    };
    for (int m = 0; m < nso; ++m)
      for (int n = 0; n < nso; ++n) {
        orb_pair(m, 0, n, 0);
        orb_pair(m, 1, n, 0);
        orb_pair(m, -1, n, 0);
        orb_pair(m, 0, n, 1);
        orb_pair(m, 0, n, -1);
      }
    const int nbra = static_cast<int>(plist.size());
    for (int a = 0; a < nsa; ++a) plist.push_back(detail::ghost_pair(aux.shells[a]));
    std::vector<std::pair<int, int>> quartets;
    std::vector<int> jm, jn, jaux, emp, emm, enp, enm;
    for (int m = 0; m < nso; ++m)
      for (int n = 0; n < nso; ++n)
        for (int a = 0; a < nsa; ++a) {
          auto emit = [&](int bra) {
            if (bra < 0) return -1;
            const int e = static_cast<int>(quartets.size());
            quartets.push_back({bra, nbra + a});
            return e;
          };
          jm.push_back(m); jn.push_back(n); jaux.push_back(a);
          emp.push_back(emit(orb_pair(m, 1, n, 0)));
          emm.push_back(emit(orb_pair(m, -1, n, 0)));
          enp.push_back(emit(orb_pair(m, 0, n, 1)));
          enm.push_back(emit(orb_pair(m, 0, n, -1)));
        }
    const int njob = static_cast<int>(jm.size());
    auto tab = make_pair_table(plist);
    auto batch = make_batch(tab, quartets);
    QuartetWorkspace<Real> ws;
    Kokkos::View<Real *> out("intti::rijd::out", batch.nout_total);
    eri_quartets(tab, batch, grid, out, ws);
    // host-side flattened inputs
    std::vector<int> hlm(nso), hln(nso), hom(nso);
    std::vector<Real> ham(nso);
    for (int i = 0; i < nso; ++i) {
      hlm[i] = orb.shells[i].l;
      hom[i] = orb.ao_off[i];
      ham[i] = orb.shells[i].alpha;
    }
    std::vector<int> hla(nsa), hoa(nsa);
    for (int i = 0; i < nsa; ++i) {
      hla[i] = aux.shells[i].l;
      hoa[i] = aux.ao_off[i];
    }
    std::vector<Real> hg(static_cast<std::size_t>(nreq) * naux), hD(nreq * N);
    for (int r = 0; r < nreq; ++r) {
      for (int P = 0; P < naux; ++P) hg[r * naux + P] = gamma[r][P];
      for (std::size_t i = 0; i < N; ++i) hD[r * N + i] = reqs[r].D[i];
    }
    auto dlm = detail::to_device(hlm, "rijd::lm"), dom = detail::to_device(hom, "rijd::om");
    auto dam = detail::to_device(ham, "rijd::am");
    auto dla = detail::to_device(hla, "rijd::la"), doa = detail::to_device(hoa, "rijd::oa");
    auto djm = detail::to_device(jm, "rijd::jm"), djn = detail::to_device(jn, "rijd::jn");
    auto dja = detail::to_device(jaux, "rijd::ja");
    auto dmp = detail::to_device(emp, "rijd::mp"), dmm = detail::to_device(emm, "rijd::mm");
    auto dnp = detail::to_device(enp, "rijd::np"), dnm = detail::to_device(enm, "rijd::nm");
    auto dg = detail::to_device(hg, "rijd::g"), dD = detail::to_device(hD, "rijd::D");
    auto offv = batch.out_offset;
    Kokkos::View<Real *> Ad("rijd::A", A.size()), Dxd("rijd::dx", dxv.size());
    const std::size_t nn2 = N, nax = naux;
    Kokkos::parallel_for(
        "intti::rijd::digest", Kokkos::RangePolicy<>(0, njob), KOKKOS_LAMBDA(int j) {
          const int m = djm(j), n = djn(j), a = dja(j);
          const int lm = dlm(m), ln = dlm(n), lp = dla(a);
          const int nm = ncart(lm), nn = ncart(ln), nP = ncart(lp);
          const int om = dom(m), on = dom(n), oP = doa(a);
          const int ent[2][2] = {{dmp(j), dmm(j)}, {dnp(j), dnm(j)}};
          for (int km = 0; km < nm; ++km) {
            int m3[3];
            cart_comp(lm, km, m3[0], m3[1], m3[2]);
            for (int kn = 0; kn < nn; ++kn) {
              int n3[3];
              cart_comp(ln, kn, n3[0], n3[1], n3[2]);
              const std::size_t imn = static_cast<std::size_t>(om + km) * nao + on + kn;
              for (int kp = 0; kp < nP; ++kp) {
                const int Pg = oP + kp;
                for (int e = 0; e < 3; ++e) {
                  Real dv[3];
                  for (int slot = 0; slot < 2; ++slot) {
                    int sg[2], ci[2];
                    Real co[2];
                    const int nt = slot == 0
                                       ? detail::md_grad_terms(lm, m3, dam(m), e, sg, ci, co)
                                       : detail::md_grad_terms(ln, n3, dam(n), e, sg, ci, co);
                    Real v = 0;
                    for (int t = 0; t < nt; ++t) {
                      const int ee = ent[slot][sg[t]];
                      if (ee < 0) continue;
                      const int sh = (sg[t] == 0) ? 1 : -1;
                      const int mm = (slot == 0) ? ncart(lm + sh) : nm;
                      const int mn2 = (slot == 1) ? ncart(ln + sh) : nn;
                      const int ia = (slot == 0) ? ci[t] : km;
                      const int ib = (slot == 1) ? ci[t] : kn;
                      (void)mm;
                      const std::size_t idx =
                          ((static_cast<std::size_t>(ia) * mn2 + ib) * nP + kp);
                      v += co[t] * out(offv(ee) + idx);
                    }
                    dv[slot] = -v; // md_grad_terms is d/dx; the centre derivative is -it
                  }
                  dv[2] = -(dv[0] + dv[1]); // d/dP by translational invariance
                  const int tgt[3] = {m, n, nso + a};
                  for (int slot = 0; slot < 3; ++slot) {
                    const std::size_t px = static_cast<std::size_t>(3 * tgt[slot] + e);
                    for (int r = 0; r < nreq; ++r) {
                      Kokkos::atomic_add(&Ad((static_cast<std::size_t>(r) * (3 * (nso + nsa)) +
                                              px) * nn2 + imn),
                                         dv[slot] * dg(r * nax + Pg));
                      Kokkos::atomic_add(&Dxd((static_cast<std::size_t>(r) * (3 * (nso + nsa)) +
                                               px) * nax + Pg),
                                         dv[slot] * dD(r * nn2 + imn));
                    }
                  }
                }
              }
            }
          }
        });
    Kokkos::fence();
    A = detail::to_host(Ad);
    dxv = detail::to_host(Dxd);
  }

  // ---- derivative two-centre pass: (M_x gamma)_P ------------------------
  std::vector<Real> Mxg(static_cast<std::size_t>(nreq) * npert * naux, Real(0));
  {
    std::vector<ShellPair<Real>> plist;
    std::vector<int> pid(static_cast<std::size_t>(nsa) * 3, -1);
    auto aux_pair = [&](int si, int di) {
      if (aux.shells[si].l + di < 0) return -1;
      const std::size_t key = static_cast<std::size_t>(si) * 3 + (di + 1);
      if (pid[key] < 0) {
        PrimitiveShell<Real> a = aux.shells[si];
        a.l += di;
        plist.push_back(detail::ghost_pair(a));
        pid[key] = static_cast<int>(plist.size()) - 1;
      }
      return pid[key];
    };
    for (int a = 0; a < nsa; ++a) {
      aux_pair(a, 0);
      aux_pair(a, 1);
      aux_pair(a, -1);
    }
    std::vector<std::pair<int, int>> quartets;
    std::vector<int> jp, jq, epp, epm;
    for (int P = 0; P < nsa; ++P)
      for (int Qs = 0; Qs < nsa; ++Qs) {
        auto emit = [&](int bra) {
          if (bra < 0) return -1;
          const int e = static_cast<int>(quartets.size());
          quartets.push_back({bra, aux_pair(Qs, 0)});
          return e;
        };
        jp.push_back(P); jq.push_back(Qs);
        epp.push_back(emit(aux_pair(P, 1)));
        epm.push_back(emit(aux_pair(P, -1)));
      }
    const int njob = static_cast<int>(jp.size());
    auto tab = make_pair_table(plist);
    auto batch = make_batch(tab, quartets);
    QuartetWorkspace<Real> ws;
    Kokkos::View<Real *> out("intti::rijd2::out", batch.nout_total);
    eri_quartets(tab, batch, grid, out, ws);
    std::vector<int> hla(nsa), hoa(nsa);
    std::vector<Real> haa(nsa);
    for (int i = 0; i < nsa; ++i) {
      hla[i] = aux.shells[i].l;
      hoa[i] = aux.ao_off[i];
      haa[i] = aux.shells[i].alpha;
    }
    std::vector<Real> hg(static_cast<std::size_t>(nreq) * naux);
    for (int r = 0; r < nreq; ++r)
      for (int P = 0; P < naux; ++P) hg[r * naux + P] = gamma[r][P];
    auto dla = detail::to_device(hla, "rijd2::la"), doa = detail::to_device(hoa, "rijd2::oa");
    auto daa = detail::to_device(haa, "rijd2::aa");
    auto djp = detail::to_device(jp, "rijd2::p"), djq = detail::to_device(jq, "rijd2::q");
    auto dpp = detail::to_device(epp, "rijd2::pp"), dpm = detail::to_device(epm, "rijd2::pm");
    auto dg = detail::to_device(hg, "rijd2::g");
    auto offv = batch.out_offset;
    Kokkos::View<Real *> Mg("rijd2::Mg", Mxg.size());
    const std::size_t nax = naux;
    const int nso_ = nso, ncen_ = ncen;
    Kokkos::parallel_for(
        "intti::rijd2::digest", Kokkos::RangePolicy<>(0, njob), KOKKOS_LAMBDA(int j) {
          const int P = djp(j), Qs = djq(j);
          const int lP = dla(P), lQ = dla(Qs);
          const int nP = ncart(lP), nQ = ncart(lQ);
          const int oP = doa(P), oQ = doa(Qs);
          const int ent[2] = {dpp(j), dpm(j)};
          for (int kp = 0; kp < nP; ++kp) {
            int p3[3];
            cart_comp(lP, kp, p3[0], p3[1], p3[2]);
            for (int kq = 0; kq < nQ; ++kq)
              for (int e = 0; e < 3; ++e) {
                int sg[2], ci[2];
                Real co[2];
                const int nt = detail::md_grad_terms(lP, p3, daa(P), e, sg, ci, co);
                Real v = 0;
                for (int t = 0; t < nt; ++t) {
                  const int ee = ent[sg[t]];
                  if (ee < 0) continue;
                  const int sh = (sg[t] == 0) ? 1 : -1;
                  const std::size_t idx = static_cast<std::size_t>(ci[t]) * nQ + kq;
                  (void)sh;
                  v += co[t] * out(offv(ee) + idx);
                }
                const Real dP = -v;              // centre derivative
                const Real dQ = -dP;             // only two centres: d/dQ = -d/dP
                const int tgt[2] = {nso_ + P, nso_ + Qs};
                const Real dvv[2] = {dP, dQ};
                for (int slot = 0; slot < 2; ++slot) {
                  const std::size_t px = static_cast<std::size_t>(3 * tgt[slot] + e);
                  for (int r = 0; r < nreq; ++r)
                    Kokkos::atomic_add(
                        &Mg((static_cast<std::size_t>(r) * (3 * ncen_) + px) * nax + oP + kp),
                        dvv[slot] * dg(r * nax + oQ + kq));
                }
              }
          }
        });
    Kokkos::fence();
    Mxg = detail::to_host(Mg);
  }

  // ---- assemble: dJ^x = A^x + T gamma_x, gamma_x = M^+ (d_x - M_x gamma) --
  JKDerivResult<Real> res;
  res.nshell = ncen;
  res.nao = nao;
  res.J.resize(nreq);
  res.K.resize(nreq);
  // gamma_x for every (request, perturbation) first: each is only naux long, so
  // all of them together are nreq x npert x naux -- negligible beside the output
  std::vector<Real> gx(static_cast<std::size_t>(nreq) * npert * naux, Real(0));
  for (int r = 0; r < nreq; ++r) {
    res.J[r].assign(static_cast<std::size_t>(npert) * N, Real(0));
    for (int x = 0; x < npert; ++x) {
      std::vector<Real> rhs(naux);
      const std::size_t b = (static_cast<std::size_t>(r) * npert + x) * naux;
      for (int P = 0; P < naux; ++P) rhs[P] = dxv[b + P] - Mxg[b + P];
      const auto g = solve(rhs);
      for (int P = 0; P < naux; ++P) gx[b + P] = g[P];
    }
  }
  // pass 2: J^x += T gamma_x, tile loop outermost so the three-centre tensor is
  // built once per tile for ALL requests and perturbations
  for (int A0 = 0; A0 < nsa; A0 += aux_tile_shells) {
    const int A1 = std::min(A0 + aux_tile_shells, nsa);
    const int p0 = aux.ao_off[A0], blk = aux.ao_off[A1] - p0;
    const auto Tblk = coulomb_3c_auxblock(orb, aux, grid, A0, A1);
    for (int r = 0; r < nreq; ++r)
      for (int x = 0; x < npert; ++x)
        detail::gemm('N', 'N', static_cast<int>(N), 1, blk, Real(1), Tblk.data(), blk,
                     gx.data() + (static_cast<std::size_t>(r) * npert + x) * naux + p0, 1,
                     Real(1), res.J[r].data() + static_cast<std::size_t>(x) * N, 1);
  }
  // add the perturbation-independent term
  for (int r = 0; r < nreq; ++r)
    for (int x = 0; x < npert; ++x) {
      const std::size_t o = static_cast<std::size_t>(x) * N;
      const std::size_t ao = (static_cast<std::size_t>(r) * npert + x) * N;
      for (std::size_t i = 0; i < N; ++i) res.J[r][o + i] += A[ao + i];
    }
  return res;
}

// RI EXCHANGE derivative matrices are deliberately ABSENT.
//
// A dense-density formulation was implemented and removed: it is not tractable
// and cannot be made so by blocking. With
//   G^P_{ms} = sum_l (ml|P) D_ls,  M Ghat = G,  K_mn = sum_{Q,s} Ghat^Q_{ms} (ns|Q),
// the response M dGhat/dx = dG/dx - (dM/dx) Ghat has a naux x nao x nao
// intermediate PER PERTURBATION -- 32 GB at nao = 1000, naux = 4000, and 256 GB
// at twice that. Blocking the perturbation loop does not help, because the
// object is already too large at a block size of one.
//
// The tractable form needs two changes together, and neither alone suffices:
//
//  1. Keep the density FACTORISED, D = C_L C_R^T. Then
//       G^P_{ms} = sum_k X^P_{mk} C_R[s,k],  X^P_{mk} = sum_l (ml|P) C_L[l,k],
//     and since M^{-1} touches only P, Ghat inherits the factorisation: the
//     nao x nao slab never has to exist, only naux x nao x nvec. This is the
//     same reason ri_k_occ2 is orbital-driven -- a perturbed density is a
//     product of two orbital sets, and keeping it that way is what makes the
//     response affordable.
//
//  2. Move M^{-1} onto the perturbation-INDEPENDENT side, so dGhat/dx is never
//     formed at all:
//       sum_{Q,s} dGhat^Q_{ms}/dx (ns|Q)
//         = sum_{P,s} [dG/dx - (dM/dx) Ghat]^P_{ms} Chat^P_{ns},
//       Chat^P_{ns} = sum_Q M^{-1}_{PQ} (ns|Q)  -- the fitted three-index,
//     computed once and reused by every perturbation. The remaining sum over P
//     is then a plain contraction, so it BLOCKS over the auxiliary index, and
//     the only per-perturbation storage left is the output matrix itself.
//
// With both, per-perturbation memory is nao^2 (the requested output) and the
// auxiliary block size becomes a cache/BLAS choice rather than a memory limit.
// Note the Coulomb builder above does not have this problem: its response is a
// length-naux vector, so it is affordable as written.

} // namespace intti
