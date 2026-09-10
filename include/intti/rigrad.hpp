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
                    Kokkos::View<Real *[3], Kokkos::LayoutLeft> force,
                    Kokkos::View<const Real *> clv = {},
                    Kokkos::View<const Real *> zv = {}, int nvec = 0) {
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
                else if constexpr (Mode == 4) {
                  // c3^R_ln = sum_jk C_L[l,j] C_L[n,k] Z^R_jk -- evaluated here
                  // rather than read from a dense nao^2 x naux array, which is
                  // the whole point of the factorised form
                  const int li = oa + k[0], ni = ob + k[1], R = oc + k[2];
                  Real acc = 0;
                  for (int jj = 0; jj < nvec; ++jj) {
                    const Real cl = clv(static_cast<std::size_t>(li) * nvec + jj);
                    if (cl == Real(0)) continue;
                    Real inner = 0;
                    for (int kk = 0; kk < nvec; ++kk)
                      inner += clv(static_cast<std::size_t>(ni) * nvec + kk) *
                               zv((static_cast<std::size_t>(R) * nvec + jj) * nvec + kk);
                    acc += cl * inner;
                  }
                  cf = acc;
                } else
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

/// All THREE Cartesian directions of the position-`pos` derivative at once.
///
/// The two promoted/demoted integral blocks depend only on `pos`; `dir` enters
/// nowhere but the component bookkeeping of the assembly below. Calling the
/// single-direction form for dir = 0, 1, 2 therefore evaluated the same pair of
/// eri_block4 three times over -- and those evaluations, not the digest, are
/// what these kernels spend their time on (ri_k_deriv_kernel measured 68 s
/// against ri_j_deriv_kernel's 1 s for the same number of derivative blocks,
/// the difference being that the latter goes through the batched engine).
template <class Real>
std::array<std::vector<Real>, 3>
quartet_pos_deriv_blocks3(const PrimitiveShell<Real> &s0, const PrimitiveShell<Real> &s1,
                          const PrimitiveShell<Real> &s2, const PrimitiveShell<Real> &s3,
                          int pos, const TGrid<Real> &grid) {
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
  const auto plus = promote(1);
  std::vector<Real> minus;
  if (lp >= 1) minus = promote(-1);
  int npl[4], nmi[4];
  for (int i = 0; i < 4; ++i) npl[i] = nmi[i] = nc[i];
  npl[pos] = ncart(lp + 1);
  if (lp >= 1) nmi[pos] = ncart(lp - 1);
  auto idx = [](const int n[4], int a, int b, int c, int d) {
    return ((static_cast<std::size_t>(a) * n[1] + b) * n[2] + c) * n[3] + d;
  };
  const std::size_t sz = static_cast<std::size_t>(nc[0]) * nc[1] * nc[2] * nc[3];
  std::array<std::vector<Real>, 3> out;
  for (int d = 0; d < 3; ++d) out[d].assign(sz, Real(0));
  int k[4];
  for (k[0] = 0; k[0] < nc[0]; ++k[0])
    for (k[1] = 0; k[1] < nc[1]; ++k[1])
      for (k[2] = 0; k[2] < nc[2]; ++k[2])
        for (k[3] = 0; k[3] < nc[3]; ++k[3]) {
          int b3[3];
          cart_comp(lp, k[pos], b3[0], b3[1], b3[2]);
          const std::size_t o = idx(nc, k[0], k[1], k[2], k[3]);
          for (int dir = 0; dir < 3; ++dir) {
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
            out[dir][o] = term;
          }
        }
  return out;
}

/// Full (uncontracted) derivative block d/dR_{pos,dir} of the quartet
/// (s0 s1 | s2 s3): out has the base component shape, out[idx] = the MD
/// centre-shift 2 alpha [.+1_dir] - m [.-1_dir] of position `pos`. Prefer
/// quartet_pos_deriv_blocks3 when all three directions are wanted, which is the
/// usual case.
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

/// The three positions of a (bra1 bra2 | aux ghost) derivative, with the AUX
/// position obtained from translational invariance rather than evaluated.
///
/// The ghost has zero exponent, so it is a constant and carries no centre
/// dependence: d/dbra1 + d/dbra2 + d/daux = 0 exactly. That makes the third
/// position free, cutting the eri_block4 evaluations -- which are what these
/// kernels actually spend their time on -- by a third. ri_j_deriv_kernel has
/// always done this; the exchange and Hessian passes were evaluating all three.
///
/// blk[pos][dir], each in the base component shape of (s0 s1 | s2 ghost).
template <class Real>
std::array<std::array<std::vector<Real>, 3>, 3>
ghost_quartet_deriv(const PrimitiveShell<Real> &s0, const PrimitiveShell<Real> &s1,
                    const PrimitiveShell<Real> &s2, const PrimitiveShell<Real> &gh,
                    const TGrid<Real> &grid) {
  std::array<std::array<std::vector<Real>, 3>, 3> blk;
  blk[0] = quartet_pos_deriv_blocks3(s0, s1, s2, gh, 0, grid);
  blk[1] = quartet_pos_deriv_blocks3(s0, s1, s2, gh, 1, grid);
  for (int dir = 0; dir < 3; ++dir) {
    blk[2][dir].assign(blk[0][dir].size(), Real(0));
    for (std::size_t i = 0; i < blk[0][dir].size(); ++i)
      blk[2][dir][i] = -(blk[0][dir][i] + blk[1][dir][i]);
  }
  return blk;
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
                    Kokkos::View<Real *> H, Kokkos::View<const Real *> clv = {},
                    Kokkos::View<const Real *> zv = {}, int nvec = 0) {
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
                else if constexpr (Mode == 4) {
                  // c3^R_ln = sum_jk C_L[l,j] C_L[n,k] Z^R_jk -- evaluated here
                  // rather than read from a dense nao^2 x naux array, which is
                  // the whole point of the factorised form
                  const int li = oa + k[0], ni = ob + k[1], R = oc + k[2];
                  Real acc = 0;
                  for (int jj = 0; jj < nvec; ++jj) {
                    const Real cl = clv(static_cast<std::size_t>(li) * nvec + jj);
                    if (cl == Real(0)) continue;
                    Real inner = 0;
                    for (int kk = 0; kk < nvec; ++kk)
                      inner += clv(static_cast<std::size_t>(ni) * nvec + kk) *
                               zv((static_cast<std::size_t>(R) * nvec + jj) * nvec + kk);
                    acc += cl * inner;
                  }
                  cf = acc;
                } else
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

namespace detail {

/// ri_j_hessian's derivative kernel, over PRIMITIVE shells, with the fit
/// supplied already solved.
///
/// Everything the kernel contracts against enters LINEARLY in the orbital and
/// auxiliary indices, so a contracted basis needs no second kernel: since
///     sum_IJP D_IJ (IJ|P) g_P = sum_ijp (C^T D C)_ij (Ca^T g)_p (ij|p),
/// the fit is solved in the CONTRACTED auxiliary space and the density and
/// gamma are pushed down to the primitive space, after which this kernel runs
/// unchanged. Each primitive triple's derivative is still evaluated exactly
/// once -- the contraction is folded into the coefficients, not paid for in
/// integrals -- so this is the shared-intermediate form, not a
/// decontract/recontract.
///
/// `Ca` (nauxc x naux, row-major) lifts a primitive auxiliary index back to the
/// contracted one, where M^{-1} lives; empty means the two spaces coincide.
/// `parent` folds shell centres onto perturbation groups; `gam` is the
/// pushed-down gamma, in the PRIMITIVE auxiliary space.
template <class Real>
std::vector<Real> ri_j_hessian_kernel(const ShellBasis<Real> &orb,
                                      const ShellBasis<Real> &aux, const Real *D,
                                      const std::vector<Real> &Ca, int nauxc,
                                      const std::vector<Real> &Minv,
                                      const std::vector<Real> &gamma,
                                      const std::vector<int> &parent, int ngrp,
                                      const TGrid<Real> &grid) {
  const int nao = orb.nao, naux = aux.nao;
  const int nso = static_cast<int>(orb.shells.size());
  const int nsa = static_cast<int>(aux.shells.size());
  const int dim = 3 * ngrp;
  auto Dm = [&](int i, int j) { return D[static_cast<std::size_t>(i) * nao + j]; };
  auto cshell = [&](bool isaux, int s) { return parent[isaux ? nso + s : s]; };

  // first-derivative residual tensor r[x][P] = d_x[P] - (M_x gamma)[P]
  std::vector<Real> r(static_cast<std::size_t>(dim) * naux, Real(0));
  auto radd = [&](int cs, int dir, int P, Real v) {
    r[(static_cast<std::size_t>(3 * cs + dir)) * naux + P] += v;
  };
  const auto ghost = [&](const PrimitiveShell<Real> &s) { return detail::ghost_shell(s); };
  // d_x: 3-centre (m n | a ghost), through the BATCHED engine -- the same
  // construction ri_j_deriv_kernel and ri_k_deriv_kernel use. Only the two bra
  // positions are emitted; the auxiliary one follows from translational
  // invariance, the ghost being exponent-free.
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
        orb_pair(m, 1, n, 0);
        orb_pair(m, -1, n, 0);
        orb_pair(m, 0, n, 1);
        orb_pair(m, 0, n, -1);
      }
    const int nbra = static_cast<int>(plist.size());
    for (int a = 0; a < nsa; ++a) plist.push_back(detail::ghost_pair(aux.shells[a]));
    std::vector<std::pair<int, int>> quartets;
    std::vector<int> jm, jn, ja, emp, emm, enp, enm;
    for (int m = 0; m < nso; ++m)
      for (int n = 0; n < nso; ++n)
        for (int a = 0; a < nsa; ++a) {
          auto emit = [&](int bra) {
            if (bra < 0) return -1;
            const int e = static_cast<int>(quartets.size());
            quartets.push_back({bra, nbra + a});
            return e;
          };
          jm.push_back(m); jn.push_back(n); ja.push_back(a);
          emp.push_back(emit(orb_pair(m, 1, n, 0)));
          emm.push_back(emit(orb_pair(m, -1, n, 0)));
          enp.push_back(emit(orb_pair(m, 0, n, 1)));
          enm.push_back(emit(orb_pair(m, 0, n, -1)));
        }
    auto tab = make_pair_table(plist);
    auto batch = make_batch(tab, quartets);
    QuartetWorkspace<Real> ws;
    Kokkos::View<Real *> qout("intti::rijh::out", batch.nout_total);
    eri_quartets(tab, batch, grid, qout, ws);
    // digest ON DEVICE: the quartet buffer is the largest object, so it stays
    // where it was produced. Parallel over jobs; r is dim x naux, small enough
    // that atomics into it are cheap and no per-thread storage is needed.
    std::vector<int> hoL(nso), hoO(nso), haL(nsa), haO(nsa);
    std::vector<Real> hoA(nso);
    for (int i2 = 0; i2 < nso; ++i2) {
      hoL[i2] = orb.shells[i2].l;
      hoO[i2] = orb.ao_off[i2];
      hoA[i2] = orb.shells[i2].alpha;
    }
    for (int i2 = 0; i2 < nsa; ++i2) {
      haL[i2] = aux.shells[i2].l;
      haO[i2] = aux.ao_off[i2];
    }
    auto doL = detail::to_device(hoL, "rijh::oL"), doO = detail::to_device(hoO, "rijh::oO");
    auto doA = detail::to_device(hoA, "rijh::oA");
    auto daL = detail::to_device(haL, "rijh::aL"), daO = detail::to_device(haO, "rijh::aO");
    auto djm = detail::to_device(jm, "rijh::jm"), djn = detail::to_device(jn, "rijh::jn");
    auto dja = detail::to_device(ja, "rijh::ja");
    auto demp = detail::to_device(emp, "rijh::emp"), demm = detail::to_device(emm, "rijh::emm");
    auto denp = detail::to_device(enp, "rijh::enp"), denm = detail::to_device(enm, "rijh::enm");
    auto dpar = detail::to_device(parent, "rijh::parent");
    auto dD = detail::to_device(D, static_cast<std::size_t>(nao) * nao, "rijh::D");
    auto offv = batch.out_offset;
    Kokkos::View<Real *> rd("rijh::r", static_cast<std::size_t>(dim) * naux);
    const int njob = static_cast<int>(jm.size()), nsoK = nso, naoK = nao, nauxK = naux;
    Kokkos::parallel_for(
        "intti::rijh::rpass", Kokkos::RangePolicy<>(0, njob), KOKKOS_LAMBDA(int j) {
          const int m = djm(j), n = djn(j), a = dja(j);
          const int lm = doL(m), ln = doL(n), lp = daL(a);
          const int nm = ncart(lm), nn = ncart(ln), nP = ncart(lp);
          const int om = doO(m), on = doO(n), oP = daO(a);
          const int cs[3] = {dpar(m), dpar(n), dpar(nsoK + a)};
          const int ent[2][2] = {{demp(j), demm(j)}, {denp(j), denm(j)}};
          for (int km = 0; km < nm; ++km) {
            int m3[3];
            cart_comp(lm, km, m3[0], m3[1], m3[2]);
            for (int kn = 0; kn < nn; ++kn) {
              const Real dmn = dD(static_cast<std::size_t>(om + km) * naoK + on + kn);
              if (dmn == Real(0)) continue;
              int n3[3];
              cart_comp(ln, kn, n3[0], n3[1], n3[2]);
              for (int kP = 0; kP < nP; ++kP)
                for (int dir = 0; dir < 3; ++dir) {
                  Real dv[3];
                  for (int slot = 0; slot < 2; ++slot) {
                    int sg[2], ci[2];
                    Real co[2];
                    const int nt =
                        slot == 0
                            ? detail::md_grad_terms(lm, m3, doA(m), dir, sg, ci, co)
                            : detail::md_grad_terms(ln, n3, doA(n), dir, sg, ci, co);
                    Real v = 0;
                    for (int t = 0; t < nt; ++t) {
                      const int ee = ent[slot][sg[t]];
                      if (ee < 0) continue;
                      const int sh = (sg[t] == 0) ? 1 : -1;
                      const int mn2 = (slot == 1) ? ncart(ln + sh) : nn;
                      const int ia = (slot == 0) ? ci[t] : km;
                      const int ib = (slot == 1) ? ci[t] : kn;
                      v += co[t] * qout(offv(ee) +
                                        (static_cast<std::size_t>(ia) * mn2 + ib) * nP + kP);
                    }
                    dv[slot] = -v;
                  }
                  dv[2] = -(dv[0] + dv[1]); // ghost is exponent-free
                  for (int pos = 0; pos < 3; ++pos)
                    if (dv[pos] != Real(0))
                      Kokkos::atomic_add(
                          &rd(static_cast<std::size_t>(3 * cs[pos] + dir) * nauxK + oP + kP),
                          dmn * dv[pos]);
                }
            }
          }
        });
    Kokkos::fence();
    const auto hr = detail::to_host(rd);
    for (std::size_t i2 = 0; i2 < r.size(); ++i2) r[i2] += hr[i2];
  }
  // -(M_x gamma): 2-centre (a ghost | b ghost), free aux index = a
  for (int a = 0; a < nsa; ++a)
    for (int b = 0; b < nsa; ++b) {
      const auto &sA = aux.shells[a], &sB = aux.shells[b];
      const auto ghA = ghost(sA), ghB = ghost(sB);
      const int oA = aux.ao_off[a], oB = aux.ao_off[b];
      const int nA = ncart(sA.l), nB = ncart(sB.l);
      for (int pos : {0, 2}) {
        auto blk3 = detail::quartet_pos_deriv_blocks3(sA, ghA, sB, ghB, pos, grid);
        for (int dir = 0; dir < 3; ++dir) {
          const auto &blk = blk3[dir];
          const int cs = (pos == 0) ? cshell(true, a) : cshell(true, b);
          for (int ka = 0; ka < nA; ++ka) {
            Real acc = 0;
            for (int kb = 0; kb < nB; ++kb)
              acc += blk[static_cast<std::size_t>(ka) * nB + kb] * gamma[oB + kb];
            radd(cs, dir, oA + ka, -acc);
          }
        }
      }
    }
  // r was accumulated at PRIMITIVE auxiliary indices; M^{-1} lives in the
  // contracted space, so lift it there first: r_c = r Ca^T. Ca empty means the
  // two spaces coincide and the lift is the identity.
  std::vector<Real> rc;
  if (Ca.empty()) {
    rc = r;
  } else {
    rc.assign(static_cast<std::size_t>(dim) * nauxc, Real(0));
    detail::gemm('N', 'T', dim, nauxc, naux, Real(1), r.data(), naux, Ca.data(), naux,
                 Real(0), rc.data(), nauxc);
  }
  // s[x] = M^{-1} r_c[x]: s[x][P] = sum_Q r_c[x][Q] Minv[P][Q] = (r_c Minv^T)[x][P]
  std::vector<Real> s(static_cast<std::size_t>(dim) * nauxc, Real(0));
  detail::gemm('N', 'T', dim, nauxc, nauxc, Real(1), rc.data(), nauxc, Minv.data(), nauxc,
               Real(0), s.data(), nauxc);

  std::vector<Real> H(static_cast<std::size_t>(dim) * dim, Real(0));
  auto Hadd = [&](int x, int y, Real v) { H[static_cast<std::size_t>(x) * dim + y] += v; };
  // response term H[x][y] = sum_P r_c[x][P] s[y][P] = (r_c s^T)[x][y]
  detail::gemm('N', 'T', dim, dim, nauxc, Real(1), rc.data(), nauxc, s.data(), nauxc,
               Real(1), H.data(), dim);
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

/// Solve the RI fit: M^{-1} (with a relative eigenvalue cutoff) and
/// gamma = M^{-1} d, d_P = sum_mn (mn|P) D_mn. Templated on the basis types so
/// the contracted builders serve it unchanged.
/// d_P = sum_mn (mn|P) D_mn. Overloaded rather than templated so the primitive
/// path keeps its AUXILIARY TILING -- the three-centre tensor is used once here,
/// so materialising all of it would be a needless nao^2 x naux. The contracted
/// path has no tiled three-centre builder yet and takes the whole tensor.
template <class Real>
void ri_fit_rhs(const ShellBasis<Real> &orb, const ShellBasis<Real> &aux, const Real *D,
                const TGrid<Real> &grid, int aux_tile_shells, std::vector<Real> &d) {
  const int naux = aux.nao, nsa = static_cast<int>(aux.shells.size());
  const std::size_t N = static_cast<std::size_t>(orb.nao) * orb.nao;
  if (aux_tile_shells < 1) aux_tile_shells = nsa;
  d.assign(naux, Real(0));
  for (int A0 = 0; A0 < nsa; A0 += aux_tile_shells) {
    const int A1 = std::min(A0 + aux_tile_shells, nsa);
    const int p0 = aux.ao_off[A0], blk = aux.ao_off[A1] - p0;
    const auto Tblk = coulomb_3c_auxblock(orb, aux, grid, A0, A1);
    gemm('T', 'N', blk, 1, static_cast<int>(N), Real(1), Tblk.data(), blk, D, 1, Real(0),
         d.data() + p0, 1);
  }
}

template <class Real>
void ri_fit_rhs(const ContractedBasis<Real> &orb, const ContractedBasis<Real> &aux,
                const Real *D, const TGrid<Real> &grid, int aux_tile_shells,
                std::vector<Real> &d) {
  const int naux = aux.nao, nsa = static_cast<int>(aux.shells.size());
  const std::size_t N = static_cast<std::size_t>(orb.nao) * orb.nao;
  if (aux_tile_shells < 1) aux_tile_shells = nsa;
  d.assign(naux, Real(0));
  for (int A0 = 0; A0 < nsa; A0 += aux_tile_shells) {
    const int A1 = std::min(A0 + aux_tile_shells, nsa);
    const int p0 = aux.ao_off[A0], blk = aux.ao_off[A1] - p0;
    const auto Tblk = coulomb_3c_auxblock(orb, aux, grid, A0, A1);
    gemm('T', 'N', blk, 1, static_cast<int>(N), Real(1), Tblk.data(), blk, D, 1, Real(0),
         d.data() + p0, 1);
  }
}

/// M^{-1} alone (no right-hand side), with the same relative eigenvalue cutoff.
template <class Real, class AuxBasis>
std::vector<Real> ri_metric_inverse(const AuxBasis &aux, const TGrid<Real> &grid,
                                    Real tau_lin) {
  const int naux = aux.nao;
  auto V = coulomb_2c(aux, grid);
  std::vector<Real> eval(naux);
  syevd(naux, V.data(), eval.data());
  Real emax = 0;
  for (Real e : eval) emax = std::max(emax, e);
  std::vector<Real> Minv(static_cast<std::size_t>(naux) * naux, Real(0));
  for (int k = 0; k < naux; ++k) {
    if (eval[k] <= tau_lin * emax) continue;
    const Real sc = Real(1) / eval[k];
    for (int R = 0; R < naux; ++R)
      for (int Q = 0; Q < naux; ++Q)
        Minv[R * naux + Q] += sc * V[k * naux + R] * V[k * naux + Q];
  }
  return Minv;
}

template <class Real, class OrbBasis, class AuxBasis>
void ri_solve_fit(const OrbBasis &orb, const AuxBasis &aux, const Real *D,
                  const TGrid<Real> &grid, Real tau_lin, int aux_tile_shells,
                  std::vector<Real> &Minv, std::vector<Real> &gamma) {
  const int naux = aux.nao;
  auto M = coulomb_2c(aux, grid);
  std::vector<Real> d;
  ri_fit_rhs(orb, aux, D, grid, aux_tile_shells, d);
  std::vector<Real> Vv = M, eval(naux);
  detail::syevd(naux, Vv.data(), eval.data());
  Real emax = 0;
  for (Real e : eval) emax = std::max(emax, e);
  Minv.assign(static_cast<std::size_t>(naux) * naux, Real(0));
  gamma.assign(naux, Real(0));
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
}

} // namespace detail

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
  std::vector<Real> Minv, gamma;
  detail::ri_solve_fit(orb, aux, D, grid, tau_lin, aux_tile_shells, Minv, gamma);
  const int ncen =
      static_cast<int>(orb.shells.size()) + static_cast<int>(aux.shells.size());
  std::vector<int> parent(ncen);
  for (int i = 0; i < ncen; ++i) parent[i] = i;
  return detail::ri_j_hessian_kernel(orb, aux, D, {}, aux.nao, Minv, gamma, parent, ncen,
                                     grid);
}

/// Same, over generally-contracted orbital and auxiliary bases. The fit is
/// solved in the CONTRACTED auxiliary space -- which is the physically right
/// space, since splitting an auxiliary contraction would enlarge the fitting
/// span -- and the density and gamma are then pushed down to the primitive
/// space the derivative kernel runs in. Result is indexed by CONTRACTED shell
/// centre, orbital shells then auxiliary.
template <class Real>
std::vector<Real> ri_j_hessian(const ContractedBasis<Real> &orb,
                               const ContractedBasis<Real> &aux, const Real *D,
                               const TGrid<Real> &grid, Real tau_lin = Real(1e-10),
                               int aux_tile_shells = 0) {
  std::vector<Real> Minv, gamma;
  detail::ri_solve_fit(orb, aux, D, grid, tau_lin, aux_tile_shells, Minv, gamma);
  ShellBasis<Real> po, pa;
  const auto fo = detail::expand_contracted(orb, po);
  const auto fa = detail::expand_contracted(aux, pa);
  const auto Co = detail::fanout_matrix(fo, po); // nao_c x nao_p
  const auto Ca = detail::fanout_matrix(fa, pa); // naux_c x naux_p
  // D_p = C^T D C and gamma_p = Ca^T gamma
  const int naoc = orb.nao, naop = po.nao, nauxc = aux.nao, nauxp = pa.nao;
  std::vector<Real> tmp(static_cast<std::size_t>(naoc) * naop, Real(0));
  detail::gemm('N', 'N', naoc, naop, naoc, Real(1), D, naoc, Co.data(), naop, Real(0),
               tmp.data(), naop);
  std::vector<Real> Dp(static_cast<std::size_t>(naop) * naop, Real(0));
  detail::gemm('T', 'N', naop, naop, naoc, Real(1), Co.data(), naop, tmp.data(), naop,
               Real(0), Dp.data(), naop);
  std::vector<Real> gp(nauxp, Real(0));
  for (int P = 0; P < nauxc; ++P)
    for (int p = 0; p < nauxp; ++p)
      gp[p] += Ca[static_cast<std::size_t>(P) * nauxp + p] * gamma[P];
  // shell centres: orbital primitives then auxiliary primitives, each folded
  // onto its parent CONTRACTED shell
  const int ncen = static_cast<int>(po.shells.size()) + static_cast<int>(pa.shells.size());
  const int ngrp = fo.nsh + fa.nsh;
  std::vector<int> parent(ncen);
  for (int i = 0; i < static_cast<int>(po.shells.size()); ++i) parent[i] = fo.parent[i];
  for (int i = 0; i < static_cast<int>(pa.shells.size()); ++i)
    parent[static_cast<int>(po.shells.size()) + i] = fo.nsh + fa.parent[i];
  return detail::ri_j_hessian_kernel(po, pa, Dp.data(), Ca, nauxc, Minv, gp, parent, ngrp,
                                     grid);
}



namespace detail {
/// Expand an optional shell-centre -> perturbation-group map. Empty means one
/// group per centre (the identity), which is the shell-resolved behaviour.
/// Folding shell perturbations onto ATOMS is exact: moving an atom moves every
/// shell centred on it, so d/dR_atom is the sum of the shell derivatives, and
/// the sum is linear so it may be taken at any stage. Taking it at the
/// accumulation keeps the npert-sized objects at atom resolution -- the
/// difference between 43 GB and 2.4 GB at nao = 1000 with ~1800 shells.
inline std::vector<int> perturbation_groups(const std::vector<int> &group, int ncen,
                                            int &ngrp) {
  std::vector<int> g(group);
  if (g.empty()) {
    g.resize(ncen);
    for (int i = 0; i < ncen; ++i) g[i] = i;
  }
  ngrp = 0;
  for (int v : g) ngrp = std::max(ngrp, v + 1);
  return g;
}
} // namespace detail

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
namespace detail {

/// ri_j_deriv_build's kernel, over PRIMITIVE shells. Unlike the Hessians, the
/// OUTPUT here is AO-indexed per perturbation, so the contraction cannot be
/// folded entirely into the coefficients: the derivative three-centre pass has
/// to fan its result out onto contracted AOs in the DIGEST (detail::ShellFanout,
/// as the direct two-electron derivatives do), because lifting a
/// per-perturbation nao^2 matrix afterwards would materialise the
/// primitive-sized output -- the thing the atom fold exists to avoid.
///
/// Everything else follows the Coulomb Hessian: D and gamma arrive pushed DOWN
/// to the primitive space, the metric stays CONTRACTED, and d_x and M_x gamma
/// are accumulated at primitive auxiliary indices and lifted with Ca before the
/// solve. `fo` null and `Ca` empty is the primitive case, where every one of
/// those is the identity.
template <class Real, class Tile>
JKDerivResult<Real> ri_j_deriv_kernel(
    const ShellBasis<Real> &orb, const ShellBasis<Real> &aux,
    const std::vector<JKRequest<Real>> &reqs, const TGrid<Real> &grid, int aux_tile_shells,
    const std::vector<int> &group, const ShellFanout<Real> *fo, const std::vector<Real> &Ca,
    int nauxc, const std::vector<Real> &Minv, const std::vector<std::vector<Real>> &gamma_c,
    int nsa_out, Tile &&tile) {
  const int nao = orb.nao, naux = aux.nao;
  const int nao_out = fo ? fo->nao : nao;
  const std::size_t Nout = static_cast<std::size_t>(nao_out) * nao_out;
  ShellFanout<Real> idf;
  if (!fo) idf = identity_fanout(orb);
  const ShellFanout<Real> &fan = fo ? *fo : idf;
  const int nso = static_cast<int>(orb.shells.size());
  const int nsa = static_cast<int>(aux.shells.size());
  const int ncen = nso + nsa;
  int ngrp = 0;
  const auto grp = detail::perturbation_groups(group, ncen, ngrp);
  const int npert = 3 * ngrp;
  const int nreq = static_cast<int>(reqs.size());
  const std::size_t N = static_cast<std::size_t>(nao) * nao;

  if (aux_tile_shells < 1) aux_tile_shells = nsa;
  // The three-centre tensor is NEVER materialised in the derivative passes: it
  // is nao^2 x naux, the same shape the reverted exchange-derivative
  // intermediate was rejected for. It is consumed in an auxiliary-tiled pass
  // instead (the ri_j_tiled idiom), hoisted so the tile loop is OUTSIDE the
  // perturbation loop -- tiling naively inside it would recompute the whole
  // tensor nreq x npert times.
  //
  // The fit is solved in the CONTRACTED auxiliary space and handed in; the
  // solve here is just the application of M^{-1}.
  auto solve = [&](const std::vector<Real> &rhs_c) {
    std::vector<Real> x(nauxc, Real(0));
    for (int P = 0; P < nauxc; ++P) {
      Real acc = 0;
      for (int Q = 0; Q < nauxc; ++Q) acc += Minv[static_cast<std::size_t>(P) * nauxc + Q] * rhs_c[Q];
      x[P] = acc;
    }
    return x;
  };
  // lift a per-(request, perturbation) primitive-auxiliary array to contracted
  auto lift_aux = [&](const std::vector<Real> &Ap) {
    if (Ca.empty()) return Ap;
    std::vector<Real> out(static_cast<std::size_t>(nreq) * npert * nauxc, Real(0));
    detail::gemm('N', 'T', nreq * npert, nauxc, naux, Real(1), Ap.data(), naux, Ca.data(),
                 naux, Real(0), out.data(), nauxc);
    return out;
  };
  // gamma pushed DOWN to primitive auxiliary indices, which is where the
  // derivative blocks meet it
  std::vector<std::vector<Real>> gamma(nreq, std::vector<Real>(naux, Real(0)));
  for (int r = 0; r < nreq; ++r) {
    if (Ca.empty()) {
      gamma[r] = gamma_c[r];
    } else {
      for (int P = 0; P < nauxc; ++P)
        for (int p = 0; p < naux; ++p)
          gamma[r][p] += Ca[static_cast<std::size_t>(P) * naux + p] * gamma_c[r][P];
    }
  }

  // ---- derivative three-centre pass -------------------------------------
  // accumulates, per perturbation x and request r:
  //   A[r][x] (nao x nao) = sum_P d(mn|P)/dx gamma_P
  //   dx[r][x] (naux)     = sum_mn d(mn|P)/dx D_mn
  std::vector<Real> A(static_cast<std::size_t>(nreq) * npert * Nout, Real(0));
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
    auto dgrp = detail::to_device(grp, "rijd::grp");
    auto fnc = detail::to_device(fan.nctr, "rijd::nctr");
    auto fco = detail::to_device(fan.coff, "rijd::coff");
    auto fbs = detail::to_device(fan.base, "rijd::base");
    auto fw = detail::to_device(fan.w, "rijd::w");
    const std::size_t npert_ = static_cast<std::size_t>(npert);
    const std::size_t nn2 = N, nax = naux, nout2 = Nout;
    const int naoo = nao_out;
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
                    // fold onto the perturbation group (atoms, typically)
                    const std::size_t px = static_cast<std::size_t>(3 * dgrp(tgt[slot]) + e);
                    for (int r = 0; r < nreq; ++r) {
                      // A is an AO-indexed OUTPUT, so the contraction fans out
                      // here rather than folding into a coefficient. All trip
                      // counts are 1 for a primitive basis.
                      const Real gv = dv[slot] * dg(r * nax + Pg);
                      const std::size_t abase =
                          (static_cast<std::size_t>(r) * npert_ + px) * nout2;
                      for (int cM = 0; cM < fnc(m); ++cM) {
                        const Real wm = fw(fco(m) + cM) * gv;
                        const int I = fbs(m) + cM * nm + km;
                        for (int cN = 0; cN < fnc(n); ++cN) {
                          const Real w = wm * fw(fco(n) + cN);
                          const int Jn = fbs(n) + cN * nn + kn;
                          Kokkos::atomic_add(
                              &Ad(abase + static_cast<std::size_t>(I) * naoo + Jn), w);
                        }
                      }
                      // d_x keeps its primitive auxiliary index; it is lifted
                      // once, after the pass.
                      Kokkos::atomic_add(
                          &Dxd((static_cast<std::size_t>(r) * npert_ + px) * nax + Pg),
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
    const int nso_ = nso;
    auto dgrp = detail::to_device(grp, "rijd::grp2");
    const std::size_t npert_ = static_cast<std::size_t>(npert);
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
                  const std::size_t px = static_cast<std::size_t>(3 * dgrp(tgt[slot]) + e);
                  for (int r = 0; r < nreq; ++r)
                    Kokkos::atomic_add(
                        &Mg((static_cast<std::size_t>(r) * npert_ + px) * nax + oP + kp),
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
  res.nshell = ngrp;
  res.nao = nao_out;
  res.J.resize(nreq);
  res.K.resize(nreq);
  // d_x and M_x gamma were accumulated at PRIMITIVE auxiliary indices; the
  // metric lives in the contracted space, so lift both before the solve.
  const auto dxc = lift_aux(dxv), Mxc = lift_aux(Mxg);
  // gamma_x for every (request, perturbation): each is only nauxc long, so all
  // of them together are negligible beside the output
  std::vector<Real> gx(static_cast<std::size_t>(nreq) * npert * nauxc, Real(0));
  for (int r = 0; r < nreq; ++r) {
    res.J[r].assign(static_cast<std::size_t>(npert) * Nout, Real(0));
    for (int x = 0; x < npert; ++x) {
      std::vector<Real> rhs(nauxc);
      const std::size_t b = (static_cast<std::size_t>(r) * npert + x) * nauxc;
      for (int P = 0; P < nauxc; ++P) rhs[P] = dxc[b + P] - Mxc[b + P];
      const auto g = solve(rhs);
      for (int P = 0; P < nauxc; ++P) gx[b + P] = g[P];
    }
  }
  // pass 2: J^x += T gamma_x, with T in the OUTPUT (contracted) basis on both
  // orbital indices and the contracted auxiliary index. When a contracted
  // tensor is supplied it is used whole; otherwise the tile loop is outermost
  // so the tensor is built once per tile for ALL requests and perturbations.
  // `tile(A0, A1)` yields the three-centre block for auxiliary shells [A0, A1)
  // in the OUTPUT basis, so this loop is the same whether the caller is
  // primitive or contracted -- and neither ever holds nao^2 x naux.
  for (int A0 = 0; A0 < nsa_out; A0 += aux_tile_shells) {
    const int A1 = std::min(A0 + aux_tile_shells, nsa_out);
    int p0 = 0, blk = 0;
    const auto Tblk = tile(A0, A1, p0, blk);
    for (int r = 0; r < nreq; ++r)
      for (int x = 0; x < npert; ++x)
        detail::gemm('N', 'N', static_cast<int>(Nout), 1, blk, Real(1), Tblk.data(), blk,
                     gx.data() + (static_cast<std::size_t>(r) * npert + x) * nauxc + p0, 1,
                     Real(1), res.J[r].data() + static_cast<std::size_t>(x) * Nout, 1);
  }
  // add the perturbation-independent term
  for (int r = 0; r < nreq; ++r)
    for (int x = 0; x < npert; ++x) {
      const std::size_t o = static_cast<std::size_t>(x) * Nout;
      const std::size_t ao = (static_cast<std::size_t>(r) * npert + x) * Nout;
      for (std::size_t i = 0; i < Nout; ++i) res.J[r][o + i] += A[ao + i];
    }
  return res;
}

} // namespace detail

/// Derivative RI Coulomb matrices over primitive shells.
template <class Real>
JKDerivResult<Real> ri_j_deriv_build(const ShellBasis<Real> &orb, const ShellBasis<Real> &aux,
                                     const std::vector<JKRequest<Real>> &reqs,
                                     const TGrid<Real> &grid,
                                     Real tau_lin = Real(1e-10),
                                     int aux_tile_shells = 0,
                                     const std::vector<int> &group = {}) {
  const int nreq = static_cast<int>(reqs.size());
  const auto Minv = detail::ri_metric_inverse(aux, grid, tau_lin);
  std::vector<std::vector<Real>> gam(nreq);
  for (int r = 0; r < nreq; ++r) {
    std::vector<Real> d;
    detail::ri_fit_rhs(orb, aux, reqs[r].D, grid, aux_tile_shells, d);
    gam[r].assign(aux.nao, Real(0));
    detail::gemm('N', 'N', aux.nao, 1, aux.nao, Real(1), Minv.data(), aux.nao, d.data(), 1,
                 Real(0), gam[r].data(), 1);
  }
  const std::vector<Real> no_lift; // primitive: the two auxiliary spaces coincide
  const int nsa = static_cast<int>(aux.shells.size());
  auto tile = [&](int A0, int A1, int &p0, int &blk) {
    p0 = aux.ao_off[A0];
    blk = aux.ao_off[A1] - p0;
    return coulomb_3c_auxblock(orb, aux, grid, A0, A1);
  };
  // explicit cast: nullptr alone gives the compiler nothing to deduce Real from
  return detail::ri_j_deriv_kernel(
      orb, aux, reqs, grid, aux_tile_shells, group,
      static_cast<const detail::ShellFanout<Real> *>(nullptr), no_lift, aux.nao, Minv, gam,
      nsa, tile);
}

/// Same, over generally-contracted bases. The densities and the result are in
/// CONTRACTED AOs; the fit is solved in the contracted auxiliary space.
template <class Real>
JKDerivResult<Real> ri_j_deriv_build(const ContractedBasis<Real> &orb,
                                     const ContractedBasis<Real> &aux,
                                     const std::vector<JKRequest<Real>> &reqs,
                                     const TGrid<Real> &grid,
                                     Real tau_lin = Real(1e-10),
                                     int aux_tile_shells = 0,
                                     const std::vector<int> &group = {}) {
  const int nreq = static_cast<int>(reqs.size());
  const int naoc = orb.nao, nauxc = aux.nao;
  const auto Minv = detail::ri_metric_inverse(aux, grid, tau_lin);
  std::vector<std::vector<Real>> gam(nreq);
  for (int r = 0; r < nreq; ++r) {
    std::vector<Real> d;
    detail::ri_fit_rhs(orb, aux, reqs[r].D, grid, aux_tile_shells, d);
    gam[r].assign(nauxc, Real(0));
    detail::gemm('N', 'N', nauxc, 1, nauxc, Real(1), Minv.data(), nauxc, d.data(), 1,
                 Real(0), gam[r].data(), 1);
  }
  ShellBasis<Real> po, pa;
  const auto fo = detail::expand_contracted(orb, po);
  const auto fa = detail::expand_contracted(aux, pa);
  const auto Co = detail::fanout_matrix(fo, po);
  const auto Ca = detail::fanout_matrix(fa, pa);
  // densities push DOWN to the primitive space for the derivative pass
  const int naop = po.nao;
  std::vector<std::vector<Real>> Dp(nreq);
  std::vector<JKRequest<Real>> preq(reqs);
  for (int r = 0; r < nreq; ++r) {
    std::vector<Real> tmp(static_cast<std::size_t>(naoc) * naop, Real(0));
    detail::gemm('N', 'N', naoc, naop, naoc, Real(1), reqs[r].D, naoc, Co.data(), naop,
                 Real(0), tmp.data(), naop);
    Dp[r].assign(static_cast<std::size_t>(naop) * naop, Real(0));
    detail::gemm('T', 'N', naop, naop, naoc, Real(1), Co.data(), naop, tmp.data(), naop,
                 Real(0), Dp[r].data(), naop);
    preq[r].D = Dp[r].data();
  }
  const int ncen = static_cast<int>(po.shells.size()) + static_cast<int>(pa.shells.size());
  std::vector<int> parent(ncen);
  for (int i = 0; i < static_cast<int>(po.shells.size()); ++i) parent[i] = fo.parent[i];
  for (int i = 0; i < static_cast<int>(pa.shells.size()); ++i)
    parent[static_cast<int>(po.shells.size()) + i] = fo.nsh + fa.parent[i];
  // the caller's group map is over CONTRACTED centres; compose it with parent
  std::vector<int> grp2(ncen);
  for (int i = 0; i < ncen; ++i) grp2[i] = group.empty() ? parent[i] : group[parent[i]];
  // tiles in the CONTRACTED basis on every index -- which is the output basis
  const int nsa_c = static_cast<int>(aux.shells.size());
  auto tile = [&](int A0, int A1, int &p0, int &blk) {
    p0 = aux.ao_off[A0];
    blk = aux.ao_off[A1] - p0;
    return coulomb_3c_auxblock(orb, aux, grid, A0, A1);
  };
  return detail::ri_j_deriv_kernel(po, pa, preq, grid, aux_tile_shells, grp2, &fo, Ca,
                                   nauxc, Minv, gam, nsa_c, tile);
}

// The RI exchange derivative matrices, absent for a while, are reinstated below
// as ri_k_deriv_occ. The dense-density version was reverted because its response
// object was naux x nao^2 PER PERTURBATION -- 32 GB for a single perturbation at
// nao = 1000, naux = 4000, so no blocking could rescue it. Factorising the
// density and moving M^{-1} onto the perturbation-independent side fixes it;
// see ri_k_deriv_occ.

/// RI exchange gradient for a FACTORISED density, D = C_L C_R^T. Same quantity
/// as ri_k_gradient, computed without ever forming an nao^2 x naux object.
///
/// ri_k_gradient builds four of them -- the three-centre tensor T, the
/// density-transformed H, its fit G, and the coefficients c3 -- which is ~128 GB
/// at nao = 1000, naux = 4000. Factorising collapses all but one:
///
///   Y^Q_nj  = sum_l C_R[l,j] (ln|Q)                  naux x nao x nvec
///   Yhat    = M^{-1} Y            (M^{-1} touches only the auxiliary index)
///   H^Q_sn  = sum_j C_L[s,j] Y^Q_nj,  G^R_sn = sum_j C_L[s,j] Yhat^R_nj
///   c3^R_ln = sum_ij C_L[l,i] C_L[n,j] Z^R_ij,  Z^R_ij = -1/2 sum_s C_R[s,i] Yhat^R_sj
///   c2_TU   = 1/4 sum_ij W^T_ij W^U_ji,          W^T_ij =      sum_s C_L[s,i] Yhat^T_sj
///
/// so H, G and c3 never exist: Z and W are naux x nvec x nvec, c2 is naux x naux,
/// and the digest evaluates c3 from its factors per quartet component (Mode 4)
/// instead of reading a dense array. The three-centre tensor itself is consumed
/// one auxiliary tile at a time.
///
/// Only Yhat remains at naux x nao x nvec -- the same irreducible object as
/// ri_k_occ_tiled. It is NOT blocked over nvec here: unlike the exchange build,
/// where K is a plain sum over the vector index so blocks accumulate, Z and c2
/// need all (i,j) pairs, so nvec blocking would require every pair of blocks.
/// That is a further step, not a correctness issue.
template <class Real>
RIGrad<Real> ri_k_gradient_occ(const ShellBasis<Real> &orb, const ShellBasis<Real> &aux,
                               const Real *CL, const Real *CR, int nvec,
                               const TGrid<Real> &grid, Real tau_lin = Real(1e-10),
                               int aux_tile_shells = 0) {
  const int nao = orb.nao, naux = aux.nao;
  const int nso = static_cast<int>(orb.shells.size());
  const int nsa = static_cast<int>(aux.shells.size());
  if (aux_tile_shells < 1) aux_tile_shells = nsa;
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  const std::size_t slab = static_cast<std::size_t>(nao) * nvec;
  const std::size_t vv = static_cast<std::size_t>(nvec) * nvec;

  auto M = coulomb_2c(aux, grid);
  std::vector<Real> V = M, eval(naux);
  detail::syevd(naux, V.data(), eval.data());
  Real emax = 0;
  for (Real e : eval) emax = std::max(emax, e);
  std::vector<Real> Minv(static_cast<std::size_t>(naux) * naux, Real(0));
  for (int k = 0; k < naux; ++k) {
    if (eval[k] <= tau_lin * emax) continue;
    const Real sc = Real(1) / eval[k];
    for (int R = 0; R < naux; ++R)
      for (int Q = 0; Q < naux; ++Q)
        Minv[R * naux + Q] += sc * V[k * naux + R] * V[k * naux + Q];
  }
  // Y^Q_nj = sum_l C_R[l,j] (l n|Q), one auxiliary tile at a time
  std::vector<Real> Y(static_cast<std::size_t>(naux) * slab, Real(0));
  {
    std::vector<Real> Tq(N);
    for (int A0 = 0; A0 < nsa; A0 += aux_tile_shells) {
      const int A1 = std::min(A0 + aux_tile_shells, nsa);
      const int p0 = aux.ao_off[A0], blk = aux.ao_off[A1] - p0;
      const auto Tblk = coulomb_3c_auxblock(orb, aux, grid, A0, A1);
      for (int Q = 0; Q < blk; ++Q) {
        for (std::size_t mn = 0; mn < N; ++mn) Tq[mn] = Tblk[mn * blk + Q];
        // Y_Q = T_Q^T C_R  (n,j) = sum_l T_Q[l][n] C_R[l][j]
        detail::gemm('T', 'N', nao, nvec, nao, Real(1), Tq.data(), nao, CR, nvec, Real(0),
                     Y.data() + static_cast<std::size_t>(p0 + Q) * slab, nvec);
      }
    }
  }
  // Yhat = M^{-1} Y over the auxiliary index: one GEMM
  std::vector<Real> Yh(Y.size(), Real(0));
  detail::gemm('N', 'N', naux, static_cast<int>(slab), naux, Real(1), Minv.data(), naux,
               Y.data(), static_cast<int>(slab), Real(0), Yh.data(),
               static_cast<int>(slab));
  // Z^R = -1/2 C_R^T Yhat^R ; W^T = C_L^T Yhat^T   (both naux x nvec x nvec)
  std::vector<Real> Z(static_cast<std::size_t>(naux) * vv, Real(0));
  std::vector<Real> W(static_cast<std::size_t>(naux) * vv, Real(0));
  for (int R = 0; R < naux; ++R) {
    detail::gemm('T', 'N', nvec, nvec, nao, Real(-0.5), CR, nvec,
                 Yh.data() + static_cast<std::size_t>(R) * slab, nvec, Real(0),
                 Z.data() + static_cast<std::size_t>(R) * vv, nvec);
    detail::gemm('T', 'N', nvec, nvec, nao, Real(1), CL, nvec,
                 Yh.data() + static_cast<std::size_t>(R) * slab, nvec, Real(0),
                 W.data() + static_cast<std::size_t>(R) * vv, nvec);
  }
  // c2_TU = 1/4 sum_ij W^T_ij W^U_ji: transpose the inner block of one factor
  std::vector<Real> Wp(W.size(), Real(0));
  for (int T = 0; T < naux; ++T)
    for (int i2 = 0; i2 < nvec; ++i2)
      for (int j2 = 0; j2 < nvec; ++j2)
        Wp[static_cast<std::size_t>(T) * vv + i2 * nvec + j2] =
            W[static_cast<std::size_t>(T) * vv + j2 * nvec + i2];
  std::vector<Real> c2(static_cast<std::size_t>(naux) * naux, Real(0));
  detail::gemm('N', 'T', naux, naux, static_cast<int>(vv), Real(0.25), W.data(),
               static_cast<int>(vv), Wp.data(), static_cast<int>(vv), Real(0), c2.data(),
               naux);

  RIGrad<Real> g;
  g.forb.assign(orb.shells.size(), {Real(0), Real(0), Real(0)});
  g.faux.assign(aux.shells.size(), {Real(0), Real(0), Real(0)});
  if constexpr (kokkos_scalar_v<Real>) {
    detail::RIGradJobs<Real> jobA, jobB;
    for (int lsh = 0; lsh < nso; ++lsh)
      for (int nsh = 0; nsh < nso; ++nsh)
        for (int a = 0; a < nsa; ++a) {
          const PrimitiveShell<Real> sh[4] = {orb.shells[lsh], orb.shells[nsh],
                                              aux.shells[a],
                                              detail::ghost_shell(aux.shells[a])};
          for (int pos = 0; pos < 3; ++pos) {
            const int tgt = (pos == 0) ? lsh : (pos == 1) ? nsh : nso + a;
            jobA.add(sh, pos, tgt, orb.ao_off[lsh], orb.ao_off[nsh], aux.ao_off[a]);
          }
        }
    for (int P = 0; P < nsa; ++P)
      for (int Qs = 0; Qs < nsa; ++Qs) {
        const PrimitiveShell<Real> sh[4] = {aux.shells[P], detail::ghost_shell(aux.shells[P]),
                                            aux.shells[Qs],
                                            detail::ghost_shell(aux.shells[Qs])};
        for (int pos : {0, 2}) {
          const int tgt = nso + ((pos == 0) ? P : Qs);
          jobB.add(sh, pos, tgt, aux.ao_off[P], 0, aux.ao_off[Qs]);
        }
      }
    const int ncen = nso + nsa;
    Kokkos::View<Real *[3], Kokkos::LayoutLeft> force("intti::rikg::f", ncen);
    auto clv = detail::to_device(CL, static_cast<std::size_t>(nao) * nvec, "rikg::cl");
    auto zv = detail::to_device(Z, "rikg::z");
    auto c2d = detail::to_device(c2, "rikg::c2");
    Kokkos::View<const Real *> none;
    detail::ri_grad_digest<Real, 4>(jobA, grid, none, nao, none, none, none, naux, force,
                                    clv, zv, nvec);
    detail::ri_grad_digest<Real, 3>(jobB, grid, none, nao, none, none, c2d, naux, force);
    auto fh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, force);
    for (int i = 0; i < nso; ++i)
      for (int e = 0; e < 3; ++e) g.forb[i][e] = fh(i, e);
    for (int i = 0; i < nsa; ++i)
      for (int e = 0; e < 3; ++e) g.faux[i][e] = fh(nso + i, e);
  }
  return g;
}

/// RI exchange HESSIAN for a factorised density, D = C_L C_R^T. Same quantity as
/// ri_k_hessian, without any nao^2 x naux -- let alone dim x nao^2 x naux --
/// storage.
///
/// ri_k_hessian holds R, S and Rp, each dim x nao^2 x naux. At nao = 1000,
/// naux = 4000 and ~520 shells that is 150 TB against a 20 MB output. TWO
/// changes are needed and neither suffices alone:
///
///  1. FACTORISE. The response inherits the density's factorisation on its first
///     orbital index, R_x[a,b,Q] = sum_i C_L[a,i] r_x^Q[b,i], and the final
///     contraction then collapses BOTH orbital indices onto vector indices:
///       Hess[x][y] = -1/2 sum_{Q,R,i,j} P_x^Q[i,j] M^{-1}_QR P_y^R[j,i],
///       P_x^Q[i,j] = sum_a r_x^Q[a,i] C_L[a,j].
///     nao^2 -> nvec^2. Note the second factor is P with its vector indices
///     TRANSPOSED, so only one object is needed, not two.
///
///  2. BLOCK over the VECTOR index. M^{-1} couples only the auxiliary index, so
///     i is free; the sum over i then accumulates over blocks. Blocking over
///     PERTURBATIONS instead would also bound the memory but needs O((dim/b)^2)
///     passes, trading the memory problem for a worse time one -- the same trap
///     that had to be avoided in ri_j_deriv_build.
///
/// P is never built for all i at once and r is never built at all (it would be
/// dim x naux x nao x nvec, 4.8 TB). Working set is
/// 2 x dim x naux x vec_block x nvec: ~10 GB at vec_block = 1, ~50 GB at 5.
// CONTRACTION, for the exchange derivative kernels below (ri_k_gradient_occ,
// ri_k_hessian_occ, ri_k_deriv_occ). None of them takes a contracted basis yet;
// ri_j_hessian does, and the same recipe carries over. Written down here so the
// work starts from the design rather than from a re-derivation.
//
// The argument is the one that made the Coulomb case free: every three-centre
// integral enters LINEARLY in each of its own indices, so the contraction can be
// folded into whatever it is contracted against instead of into the integrals.
// Concretely, with C the orbital and Ca the auxiliary fan-out matrices
// (contracted x primitive, detail::fanout_matrix):
//
//   * ORBITAL-side factors push DOWN once, outside everything:
//         CLp = C^T CL,  CRp = C^T CR      (nao_primitive x nvec)
//     The exchange case is where this differs from Coulomb: the density arrives
//     factorised, so it is the FACTORS that transform, not D by congruence.
//
//   * The metric M and hence M^{-1} stay in the CONTRACTED auxiliary space --
//     the physically right space, since splitting an auxiliary contraction
//     enlarges the fitting span and changes the answer.
//
//   * The mixed tensor (m n | P), primitive orbital and contracted auxiliary,
//     is what Y and X are built from. There is no mixed builder, but none is
//     needed: wrap the primitive orbital basis as a trivially-contracted
//     ContractedBasis (one primitive per shell, coefficient
//     1/cart_norm_pyscf(l, alpha) so the effective coefficient is 1) and call
//     the contracted three-centre builder. tests/test_contracted.cpp already
//     uses exactly this wrapper for its reference.
//
//   * Anything meeting a DERIVATIVE block at a primitive auxiliary index must be
//     lifted there: Yhat_p[p] = sum_P Ca[P][p] Yhat[P]. That object is
//     naux_primitive x nao_primitive x nvec -- a constant factor above the
//     contracted one, and exactly the size an uncontracted run of the same
//     primitive basis already pays, which is the honest baseline.
//
//   * Anything accumulated AT a primitive auxiliary index and then meeting
//     M^{-1} is lifted the other way, r_c = r Ca^T, as ri_j_hessian_kernel does.
//
//   * Shell-indexed outputs (the Hessians) fold through a parent map;
//     AO-indexed outputs (ri_k_deriv_occ's matrices) need the digest fan-out of
//     detail::ShellFanout instead, since lifting a per-perturbation nao^2 matrix
//     afterwards would materialise the primitive-sized output.
//
// The one piece with no counterpart in the Coulomb case is c2/W/Z, which are
// nvec x nvec per auxiliary function: their auxiliary index is contracted
// throughout and only meets M^{-1}, so they need no lift at all.

namespace detail {

/// ri_k_hessian_occ's derivative kernel, over PRIMITIVE shells, with the fit and
/// the Y intermediate supplied. See the CONTRACTION note above: orbital-side
/// factors (CL, CR) arrive already pushed DOWN to the primitive space, the
/// metric inverse stays in the CONTRACTED auxiliary space, and anything that
/// meets a derivative block at a primitive auxiliary index is LIFTED there with
/// Ca (nauxc x naux, row-major; empty when the two spaces coincide).
///
/// Y is nauxc x nao_primitive x nvec: its auxiliary index is contracted, because
/// it only ever meets M^{-1}; its orbital index is primitive, because it is
/// contracted away against CL/CR, which are primitive.
template <class Real>
std::vector<Real> ri_k_hessian_kernel(const ShellBasis<Real> &orb,
                                      const ShellBasis<Real> &aux, const Real *CL,
                                      const Real *CR, int nvec, const std::vector<Real> &Y,
                                      const std::vector<Real> &Minv,
                                      const std::vector<Real> &Ca, int nauxc,
                                      const std::vector<int> &parent, int ngrp,
                                      const TGrid<Real> &grid, int vec_block) {
  const int nao = orb.nao, naux = aux.nao;
  const int nso = static_cast<int>(orb.shells.size());
  const int nsa = static_cast<int>(aux.shells.size());
  const int dim = 3 * ngrp;
  if (vec_block < 1) vec_block = nvec;
  const std::size_t slab = static_cast<std::size_t>(nao) * nvec;
  const std::size_t vv = static_cast<std::size_t>(nvec) * nvec;
  auto cshell = [&](bool isaux, int i) { return parent[isaux ? nso + i : i]; };
  const auto ghost = [&](const PrimitiveShell<Real> &sx) { return detail::ghost_shell(sx); };
  // Lift a contracted-auxiliary-indexed array (nauxc x w) to primitive auxiliary
  // indices (naux x w) -- the form every object takes when it meets a derivative
  // block. Identity when the two spaces coincide.
  auto lift = [&](const std::vector<Real> &A, std::size_t w) {
    if (Ca.empty()) return A;
    std::vector<Real> out(static_cast<std::size_t>(naux) * w, Real(0));
    detail::gemm('T', 'N', naux, static_cast<int>(w), nauxc, Real(1), Ca.data(), naux,
                 A.data(), static_cast<int>(w), Real(0), out.data(), static_cast<int>(w));
    return out;
  };

  std::vector<Real> Yh(Y.size(), Real(0));
  detail::gemm('N', 'N', nauxc, static_cast<int>(slab), nauxc, Real(1), Minv.data(), nauxc,
               Y.data(), static_cast<int>(slab), Real(0), Yh.data(), static_cast<int>(slab));
  std::vector<Real> Z(static_cast<std::size_t>(nauxc) * vv, Real(0));
  std::vector<Real> W(static_cast<std::size_t>(nauxc) * vv, Real(0));
  std::vector<Real> Vm(static_cast<std::size_t>(nauxc) * vv, Real(0)); // V^R[i,j]
  for (int R = 0; R < nauxc; ++R) {
    detail::gemm('T', 'N', nvec, nvec, nao, Real(-0.5), CR, nvec,
                 Yh.data() + static_cast<std::size_t>(R) * slab, nvec, Real(0),
                 Z.data() + static_cast<std::size_t>(R) * vv, nvec);
    detail::gemm('T', 'N', nvec, nvec, nao, Real(1), CL, nvec,
                 Yh.data() + static_cast<std::size_t>(R) * slab, nvec, Real(0),
                 W.data() + static_cast<std::size_t>(R) * vv, nvec);
    // V^R[i,j] = sum_a Yhat^R[a,i] C_L[a,j]
    detail::gemm('T', 'N', nvec, nvec, nao, Real(1),
                 Yh.data() + static_cast<std::size_t>(R) * slab, nvec, CL, nvec, Real(0),
                 Vm.data() + static_cast<std::size_t>(R) * vv, nvec);
  }
  std::vector<Real> Wp(W.size(), Real(0));
  for (int T = 0; T < nauxc; ++T)
    for (int i2 = 0; i2 < nvec; ++i2)
      for (int j2 = 0; j2 < nvec; ++j2)
        Wp[static_cast<std::size_t>(T) * vv + i2 * nvec + j2] =
            W[static_cast<std::size_t>(T) * vv + j2 * nvec + i2];
  std::vector<Real> c2(static_cast<std::size_t>(nauxc) * nauxc, Real(0));
  detail::gemm('N', 'T', nauxc, nauxc, static_cast<int>(vv), Real(0.25), W.data(),
               static_cast<int>(vv), Wp.data(), static_cast<int>(vv), Real(0), c2.data(),
               nauxc);
  // Objects that meet a derivative block live at PRIMITIVE auxiliary indices.
  // c2 meets TWO of them (the two-centre second derivative), so it is lifted on
  // both sides; Z and Vm meet one each.
  const std::vector<Real> Zp = lift(Z, vv);
  const std::vector<Real> Vmp = lift(Vm, vv);
  std::vector<Real> c2p;
  if (Ca.empty()) {
    c2p = c2;
  } else {
    const auto half = lift(c2, static_cast<std::size_t>(nauxc)); // naux x nauxc
    c2p.assign(static_cast<std::size_t>(naux) * naux, Real(0));
    detail::gemm('N', 'N', naux, naux, nauxc, Real(1), half.data(), nauxc, Ca.data(), naux,
                 Real(0), c2p.data(), naux);
  }

  std::vector<Real> Hess(static_cast<std::size_t>(dim) * dim, Real(0));
  // ---- skeleton terms, factorised c3 (Mode 4) and c2 (Mode 3) -------------
  if constexpr (kokkos_scalar_v<Real>) {
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
    auto clv = detail::to_device(CL, static_cast<std::size_t>(nao) * nvec, "rikh::cl");
    auto zv = detail::to_device(Zp, "rikh::z");
    auto c2d = detail::to_device(c2p, "rikh::c2");
    Kokkos::View<const Real *> none;
    Kokkos::View<Real *> Hd("rikh::H", static_cast<std::size_t>(dim) * dim);
    detail::ri_hess_digest<Real, 4>(jA, posA, grid, none, nao, none, none, none, naux, dim,
                                    Hd, clv, zv, nvec);
    detail::ri_hess_digest<Real, 3>(jB, posB, grid, none, nao, none, none, c2d, naux, dim,
                                    Hd);
    auto hh = detail::to_host(Hd);
    for (std::size_t i = 0; i < Hess.size(); ++i) Hess[i] += hh[i];
  }

  // ---- response term, blocked over the vector index -----------------------
  // The derivative quartets do NOT depend on the vector block, so they are
  // built and evaluated ONCE, above the loop. Leaving them inside meant a
  // caller who blocked the vector index -- the whole point of that blocking,
  // which exists to bound memory -- paid for the integrals again per block.
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
    for (int l = 0; l < nso; ++l)
      for (int n = 0; n < nso; ++n) {
        orb_pair(l, 1, n, 0);
        orb_pair(l, -1, n, 0);
        orb_pair(l, 0, n, 1);
        orb_pair(l, 0, n, -1);
      }
    const int nbra = static_cast<int>(plist.size());
    for (int c = 0; c < nsa; ++c) plist.push_back(detail::ghost_pair(aux.shells[c]));
    std::vector<std::pair<int, int>> quartets;
    std::vector<int> jl, jn2, jc, elp, elm, enp2, enm2;
    for (int l = 0; l < nso; ++l)
      for (int n = 0; n < nso; ++n)
        for (int c = 0; c < nsa; ++c) {
          auto emit = [&](int bra) {
            if (bra < 0) return -1;
            const int e = static_cast<int>(quartets.size());
            quartets.push_back({bra, nbra + c});
            return e;
          };
          jl.push_back(l); jn2.push_back(n); jc.push_back(c);
          elp.push_back(emit(orb_pair(l, 1, n, 0)));
          elm.push_back(emit(orb_pair(l, -1, n, 0)));
          enp2.push_back(emit(orb_pair(l, 0, n, 1)));
          enm2.push_back(emit(orb_pair(l, 0, n, -1)));
        }
    auto tab = make_pair_table(plist);
    auto batch = make_batch(tab, quartets);
    QuartetWorkspace<Real> ws;
    Kokkos::View<Real *> qout("intti::rikh::out", batch.nout_total);
    eri_quartets(tab, batch, grid, qout, ws);
  std::vector<int> hoL(nso), hoO(nso), haL(nsa), haO(nsa);
  std::vector<Real> hoA(nso);
  for (int i2 = 0; i2 < nso; ++i2) {
    hoL[i2] = orb.shells[i2].l;
    hoO[i2] = orb.ao_off[i2];
    hoA[i2] = orb.shells[i2].alpha;
  }
  for (int i2 = 0; i2 < nsa; ++i2) {
    haL[i2] = aux.shells[i2].l;
    haO[i2] = aux.ao_off[i2];
  }
  auto doL = detail::to_device(hoL, "rikh::oL"), doO = detail::to_device(hoO, "rikh::oO");
  auto doA = detail::to_device(hoA, "rikh::oA");
  auto daL = detail::to_device(haL, "rikh::aL"), daO = detail::to_device(haO, "rikh::aO");
  auto djl = detail::to_device(jl, "rikh::jl"), djn2 = detail::to_device(jn2, "rikh::jn");
  auto djc = detail::to_device(jc, "rikh::jc");
  auto delp = detail::to_device(elp, "rikh::elp"), delm = detail::to_device(elm, "rikh::elm");
  auto denp2 = detail::to_device(enp2, "rikh::enp"), denm2 = detail::to_device(enm2, "rikh::enm");
  auto dpar = detail::to_device(parent, "rikh::parent");
  auto dCL = detail::to_device(CL, static_cast<std::size_t>(orb.nao) * nvec, "rikh::cl");
  auto dCR = detail::to_device(CR, static_cast<std::size_t>(orb.nao) * nvec, "rikh::cr");
  auto offv = batch.out_offset;

  for (int i0 = 0; i0 < nvec; i0 += vec_block) {
    const int i1 = std::min(i0 + vec_block, nvec);
    const int ib = i1 - i0;
    // Pa[x][Q][ii][j] : first vector index in the block
    // Pb[y][R][ii][j] = P[y][R][j][i0+ii] : SECOND vector index in the block,
    // stored with the block index outermost so the metric solve is one GEMM
    const std::size_t pstride = static_cast<std::size_t>(naux) * ib * nvec;
    std::vector<Real> Pa(static_cast<std::size_t>(dim) * pstride, Real(0));
    std::vector<Real> Pb(static_cast<std::size_t>(dim) * pstride, Real(0));
    auto Pidx = [&](int x, int Q, int ii, int j) {
      return static_cast<std::size_t>(x) * pstride +
             (static_cast<std::size_t>(Q) * ib + ii) * nvec + j;
    };
    // three-centre: P_x^Q[i,j] += C_R[l,i] C_L[a,j] d(l a|Q)/dx, digested ON
    // DEVICE from the quartet buffer built above -- the buffer is the largest
    // object here and does not travel. Pa/Pb are dim x naux x ib x nvec, far
    // smaller, and come back for the host metric solve below.
    {
      Kokkos::View<Real *> Pad("rikh::Pa", static_cast<std::size_t>(dim) * pstride);
      Kokkos::View<Real *> Pbd("rikh::Pb", static_cast<std::size_t>(dim) * pstride);
      const int njob = static_cast<int>(jl.size()), nsoK = nso;
      const int nvecK = nvec, ibK = ib, i0K = i0, nauxK = naux;
      const std::size_t pstr = pstride;
      Kokkos::parallel_for(
          "intti::rikh::pxpass", Kokkos::RangePolicy<>(0, njob), KOKKOS_LAMBDA(int j) {
            const int l = djl(j), n = djn2(j), c = djc(j);
            const int ll = doL(l), ln = doL(n), lp = daL(c);
            const int nl = ncart(ll), nn = ncart(ln), nP = ncart(lp);
            const int ol = doO(l), on = doO(n), oc = daO(c);
            const int cs[3] = {dpar(l), dpar(n), dpar(nsoK + c)};
            const int ent[2][2] = {{delp(j), delm(j)}, {denp2(j), denm2(j)}};
            for (int kl = 0; kl < nl; ++kl) {
              int l3[3];
              cart_comp(ll, kl, l3[0], l3[1], l3[2]);
              for (int kn = 0; kn < nn; ++kn) {
                int n3[3];
                cart_comp(ln, kn, n3[0], n3[1], n3[2]);
                for (int kQ = 0; kQ < nP; ++kQ)
                  for (int dir = 0; dir < 3; ++dir) {
                    Real dvv[3];
                    for (int slot = 0; slot < 2; ++slot) {
                      int sg[2], ci[2];
                      Real co[2];
                      const int nt =
                          slot == 0
                              ? detail::md_grad_terms(ll, l3, doA(l), dir, sg, ci, co)
                              : detail::md_grad_terms(ln, n3, doA(n), dir, sg, ci, co);
                      Real v = 0;
                      for (int t = 0; t < nt; ++t) {
                        const int ee = ent[slot][sg[t]];
                        if (ee < 0) continue;
                        const int sh = (sg[t] == 0) ? 1 : -1;
                        const int mn2 = (slot == 1) ? ncart(ln + sh) : nn;
                        const int ia = (slot == 0) ? ci[t] : kl;
                        const int ib2 = (slot == 1) ? ci[t] : kn;
                        v += co[t] *
                             qout(offv(ee) +
                                  (static_cast<std::size_t>(ia) * mn2 + ib2) * nP + kQ);
                      }
                      dvv[slot] = -v;
                    }
                    dvv[2] = -(dvv[0] + dvv[1]); // ghost is exponent-free
                    for (int pos = 0; pos < 3; ++pos) {
                      const Real dv = dvv[pos];
                      if (dv == Real(0)) continue;
                      const int xi = 3 * cs[pos] + dir;
                      const int lam = ol + kl, aa = on + kn, QQ = oc + kQ;
                      for (int ii = 0; ii < ibK; ++ii) {
                        const Real cri =
                            dCR(static_cast<std::size_t>(lam) * nvecK + i0K + ii);
                        const Real clj =
                            dCL(static_cast<std::size_t>(aa) * nvecK + i0K + ii);
                        for (int jj = 0; jj < nvecK; ++jj) {
                          const std::size_t o =
                              static_cast<std::size_t>(xi) * pstr +
                              (static_cast<std::size_t>(QQ) * ibK + ii) * nvecK + jj;
                          Kokkos::atomic_add(
                              &Pad(o),
                              dv * cri * dCL(static_cast<std::size_t>(aa) * nvecK + jj));
                          Kokkos::atomic_add(
                              &Pbd(o),
                              dv * clj * dCR(static_cast<std::size_t>(lam) * nvecK + jj));
                        }
                      }
                    }
                  }
              }
            }
            (void)nauxK;
          });
      Kokkos::fence();
      const auto hPa = detail::to_host(Pad), hPb = detail::to_host(Pbd);
      for (std::size_t z = 0; z < Pa.size(); ++z) {
        Pa[z] += hPa[z];
        Pb[z] += hPb[z];
      }
    }
    // two-centre: P_x^Q[i,j] -= dM_QR/dx V^R[i,j]
    for (int c = 0; c < nsa; ++c)
      for (int ee = 0; ee < nsa; ++ee) {
        const auto &sC = aux.shells[c], &sE = aux.shells[ee];
        const auto ghC = ghost(sC), ghE = ghost(sE);
        const int oC = aux.ao_off[c], oE = aux.ao_off[ee];
        const int nC = ncart(sC.l), nE = ncart(sE.l);
        for (int pos : {0, 2}) {
          auto blk3 = detail::quartet_pos_deriv_blocks3(sC, ghC, sE, ghE, pos, grid);
          for (int dir = 0; dir < 3; ++dir) {
            const auto &blk = blk3[dir];
            const int cs = (pos == 0) ? cshell(true, c) : cshell(true, ee);
            const int xi = 3 * cs + dir;
            for (int kQ = 0; kQ < nC; ++kQ)
              for (int kR = 0; kR < nE; ++kR) {
                const Real dv = blk[static_cast<std::size_t>(kQ) * nE + kR];
                if (dv == Real(0)) continue;
                const int QQ = oC + kQ, RR = oE + kR;
                for (int ii = 0; ii < ib; ++ii)
                  for (int j = 0; j < nvec; ++j) {
                    Pa[Pidx(xi, QQ, ii, j)] -=
                        dv * Vmp[static_cast<std::size_t>(RR) * vv + (i0 + ii) * nvec + j];
                    Pb[Pidx(xi, QQ, ii, j)] -=
                        dv * Vmp[static_cast<std::size_t>(RR) * vv + j * nvec + (i0 + ii)];
                  }
              }
          }
        }
      }
    // Pa and Pb were accumulated at PRIMITIVE auxiliary indices, since that is
    // what a derivative block carries; M^{-1} lives in the contracted space, so
    // lift both first: P_c = Ca P_p. (This is the counterpart of r -> r Ca^T in
    // the Coulomb kernel -- there the free index is the row, here the column.)
    const int inner = static_cast<int>(static_cast<std::size_t>(ib) * nvec);
    const std::size_t cstride = static_cast<std::size_t>(nauxc) * ib * nvec;
    std::vector<Real> Pac, Pbc;
    if (Ca.empty()) {
      Pac = std::move(Pa);
      Pbc = std::move(Pb);
    } else {
      Pac.assign(static_cast<std::size_t>(dim) * cstride, Real(0));
      Pbc.assign(static_cast<std::size_t>(dim) * cstride, Real(0));
      for (int x = 0; x < dim; ++x) {
        detail::gemm('N', 'N', nauxc, inner, naux, Real(1), Ca.data(), naux,
                     Pa.data() + static_cast<std::size_t>(x) * pstride, inner, Real(0),
                     Pac.data() + static_cast<std::size_t>(x) * cstride, inner);
        detail::gemm('N', 'N', nauxc, inner, naux, Real(1), Ca.data(), naux,
                     Pb.data() + static_cast<std::size_t>(x) * pstride, inner, Real(0),
                     Pbc.data() + static_cast<std::size_t>(x) * cstride, inner);
      }
    }
    // Hess[x][y] += -1/2 sum_{Q,R,i,j} Pa_x[Q,i,j] Minv[Q,R] Pb_y[R,i,j]
    std::vector<Real> MPb(Pbc.size(), Real(0));
    for (int y = 0; y < dim; ++y)
      detail::gemm('N', 'N', nauxc, inner, nauxc, Real(1), Minv.data(), nauxc,
                   Pbc.data() + static_cast<std::size_t>(y) * cstride, inner, Real(0),
                   MPb.data() + static_cast<std::size_t>(y) * cstride, inner);
    detail::gemm('N', 'T', dim, dim, static_cast<int>(cstride), Real(-0.5), Pac.data(),
                 static_cast<int>(cstride), MPb.data(), static_cast<int>(cstride), Real(1),
                 Hess.data(), dim);
  }
  return Hess;
}

/// Y^Q_{n,i} = sum_s (n s|Q) C_R[s,i], auxiliary index CONTRACTED, orbital index
/// primitive. Overloaded on the basis types so the primitive path keeps its
/// auxiliary tiling; the contracted path uses the mixed (primitive orbital |
/// contracted auxiliary) tensor, obtained by wrapping the primitive orbital
/// basis as trivially-contracted shells.
template <class Real>
std::vector<Real> ri_k_build_Y(const ShellBasis<Real> &orb, const ShellBasis<Real> &aux,
                               const Real *CR, int nvec, const TGrid<Real> &grid,
                               int aux_tile_shells) {
  const int nao = orb.nao, naux = aux.nao;
  const int nsa = static_cast<int>(aux.shells.size());
  if (aux_tile_shells < 1) aux_tile_shells = nsa;
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  const std::size_t slab = static_cast<std::size_t>(nao) * nvec;
  std::vector<Real> Y(static_cast<std::size_t>(naux) * slab, Real(0)), Tq(N);
  for (int A0 = 0; A0 < nsa; A0 += aux_tile_shells) {
    const int A1 = std::min(A0 + aux_tile_shells, nsa);
    const int p0 = aux.ao_off[A0], blk = aux.ao_off[A1] - p0;
    const auto Tblk = coulomb_3c_auxblock(orb, aux, grid, A0, A1);
    for (int Q = 0; Q < blk; ++Q) {
      for (std::size_t mn = 0; mn < N; ++mn) Tq[mn] = Tblk[mn * blk + Q];
      gemm('T', 'N', nao, nvec, nao, Real(1), Tq.data(), nao, CR, nvec, Real(0),
           Y.data() + static_cast<std::size_t>(p0 + Q) * slab, nvec);
    }
  }
  return Y;
}

template <class Real>
std::vector<Real> ri_k_build_Y(const ContractedBasis<Real> &orb,
                               const ContractedBasis<Real> &aux, const Real *CR, int nvec,
                               const TGrid<Real> &grid, int aux_tile_shells) {
  const int nao = orb.nao, naux = aux.nao;
  const int nsa = static_cast<int>(aux.shells.size());
  if (aux_tile_shells < 1) aux_tile_shells = nsa;
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  const std::size_t slab = static_cast<std::size_t>(nao) * nvec;
  std::vector<Real> Y(static_cast<std::size_t>(naux) * slab, Real(0)), Tq(N);
  for (int A0 = 0; A0 < nsa; A0 += aux_tile_shells) {
    const int A1 = std::min(A0 + aux_tile_shells, nsa);
    const int p0 = aux.ao_off[A0], blk = aux.ao_off[A1] - p0;
    const auto Tblk = coulomb_3c_auxblock(orb, aux, grid, A0, A1);
    for (int Q = 0; Q < blk; ++Q) {
      for (std::size_t mn = 0; mn < N; ++mn) Tq[mn] = Tblk[mn * blk + Q];
      gemm('T', 'N', nao, nvec, nao, Real(1), Tq.data(), nao, CR, nvec, Real(0),
           Y.data() + static_cast<std::size_t>(p0 + Q) * slab, nvec);
    }
  }
  return Y;
}

} // namespace detail

template <class Real>
std::vector<Real> ri_k_hessian_occ(const ShellBasis<Real> &orb, const ShellBasis<Real> &aux,
                                   const Real *CL, const Real *CR, int nvec,
                                   const TGrid<Real> &grid, Real tau_lin = Real(1e-10),
                                   int aux_tile_shells = 0, int vec_block = 0) {
  const auto Minv = detail::ri_metric_inverse(aux, grid, tau_lin);
  const auto Y = detail::ri_k_build_Y(orb, aux, CR, nvec, grid, aux_tile_shells);
  const int ncen =
      static_cast<int>(orb.shells.size()) + static_cast<int>(aux.shells.size());
  std::vector<int> parent(ncen);
  for (int i = 0; i < ncen; ++i) parent[i] = i;
  return detail::ri_k_hessian_kernel(orb, aux, CL, CR, nvec, Y, Minv, {}, aux.nao, parent,
                                     ncen, grid, vec_block);
}

/// Same, over generally-contracted orbital and auxiliary bases. CL and CR are
/// nao_contracted x nvec; the result is indexed by CONTRACTED shell centre.
template <class Real>
std::vector<Real> ri_k_hessian_occ(const ContractedBasis<Real> &orb,
                                   const ContractedBasis<Real> &aux, const Real *CL,
                                   const Real *CR, int nvec, const TGrid<Real> &grid,
                                   Real tau_lin = Real(1e-10), int aux_tile_shells = 0,
                                   int vec_block = 0) {
  const auto Minv = detail::ri_metric_inverse(aux, grid, tau_lin);
  ShellBasis<Real> po, pa;
  const auto fo = detail::expand_contracted(orb, po);
  const auto fa = detail::expand_contracted(aux, pa);
  const auto Co = detail::fanout_matrix(fo, po); // nao_c x nao_p
  const auto Ca = detail::fanout_matrix(fa, pa); // naux_c x naux_p
  // orbital-side factors push DOWN once: CLp = C^T CL. The density arrives
  // factorised, so it is the FACTORS that transform, not D by congruence.
  const int naoc = orb.nao, naop = po.nao;
  auto pushdown = [&](const Real *X) {
    std::vector<Real> Xp(static_cast<std::size_t>(naop) * nvec, Real(0));
    detail::gemm('T', 'N', naop, nvec, naoc, Real(1), Co.data(), naop, X, nvec, Real(0),
                 Xp.data(), nvec);
    return Xp;
  };
  const auto CLp = pushdown(CL), CRp = pushdown(CR);
  // Y wants a primitive orbital index and a contracted auxiliary one: wrap the
  // primitive orbitals as trivially-contracted shells and use the contracted
  // three-centre builder.
  std::vector<ContractedShell<Real>> tw;
  tw.reserve(po.shells.size());
  for (const auto &sh : po.shells) {
    ContractedShell<Real> t;
    t.l = sh.l;
    for (int k = 0; k < 3; ++k) t.center[k] = sh.center[k];
    t.alpha = {sh.alpha};
    t.coeff = {Real(1) / cart_norm_pyscf(sh.l, sh.alpha)};
    tw.push_back(std::move(t));
  }
  const auto pow_ = make_contracted_basis<Real>(std::move(tw));
  const auto Y = detail::ri_k_build_Y(pow_, aux, CRp.data(), nvec, grid, 0);
  const int ncen = static_cast<int>(po.shells.size()) + static_cast<int>(pa.shells.size());
  const int ngrp = fo.nsh + fa.nsh;
  std::vector<int> parent(ncen);
  for (int i = 0; i < static_cast<int>(po.shells.size()); ++i) parent[i] = fo.parent[i];
  for (int i = 0; i < static_cast<int>(pa.shells.size()); ++i)
    parent[static_cast<int>(po.shells.size()) + i] = fo.nsh + fa.parent[i];
  return detail::ri_k_hessian_kernel(po, pa, CLp.data(), CRp.data(), nvec, Y, Minv, Ca,
                                     aux.nao, parent, ngrp, grid, vec_block);
}

/// Derivative RI EXCHANGE matrices for a factorised density, D = C_L C_R^T.
/// Reinstates what was reverted for being untractable, in a form that is not.
///
/// The dense version's response object Ghat_x was naux x nao^2 PER PERTURBATION,
/// 32 GB for one perturbation, so blocking could not help. Two changes fix it,
/// the same pair as for the exchange Hessian:
///
///  1. FACTORISE. With X^P_mi = sum_l (ml|P) C_L[l,i], Y^Q_ni = sum_s (ns|Q)
///     C_R[s,i] and Xhat = M^{-1}X, Yhat = M^{-1}Y, we have
///     K_mn = sum_{Q,i} Xhat^Q_mi Y^Q_ni: the nao x nao slab per auxiliary
///     function becomes nao x nvec.
///  2. Move M^{-1} onto the perturbation-INDEPENDENT side, so the response is
///     never formed. Differentiating M Xhat = X and folding M^{-1} onto Y:
///       dK_mn/dx = sum_{P,i}   dX^P_mi/dx  Yhat^P_ni
///                - sum_{P,R,i} dM_PR/dx Xhat^R_mi Yhat^P_ni
///                + sum_{Q,i}   Xhat^Q_mi  dY^Q_ni/dx
///     Every term contracts a derivative integral directly against something
///     precomputed; nothing of size naux x nao x nvec exists per perturbation.
///
/// Terms 1 and 3 use the SAME derivative three-centre integrals in two index
/// roles, so one pass serves both. The persistent working set is Xhat and Yhat
/// (2 x naux x nao x nvec, built once); the total is dominated by the requested
/// OUTPUT, npert x nao^2.
namespace detail {

/// ri_k_deriv_occ's kernel, over PRIMITIVE shells. Like the Coulomb derivative
/// matrices the output is AO-indexed per perturbation, so the orbital index that
/// comes from a derivative block is fanned out onto contracted AOs here.
///
/// Xh and Yh arrive at PRIMITIVE auxiliary indices (they meet derivative blocks
/// on that index, in all three terms) but CONTRACTED orbital indices (that index
/// is an output). CL and CR arrive pushed down to the primitive orbital space.
/// Term 2 therefore needs no fan-out at all: both of its orbital indices come
/// from Xh and Yh and are already contracted.
template <class Real>
JKDerivResult<Real> ri_k_deriv_kernel(const ShellBasis<Real> &orb,
                                      const ShellBasis<Real> &aux, const Real *CL,
                                      const Real *CR, int nvec,
                                      const std::vector<Real> &Xh,
                                      const std::vector<Real> &Yh,
                                      const ShellFanout<Real> *fo,
                                      const std::vector<int> &grp, int ngrp,
                                      const TGrid<Real> &grid, int aux_tile_shells) {
  const int naux = aux.nao;
  const int nso = static_cast<int>(orb.shells.size());
  const int nsa = static_cast<int>(aux.shells.size());
  const int npert = 3 * ngrp;
  ShellFanout<Real> idf;
  if (!fo) idf = identity_fanout(orb);
  const ShellFanout<Real> &fan = fo ? *fo : idf;
  const int nao_out = fan.nao;
  const std::size_t N = static_cast<std::size_t>(nao_out) * nao_out;
  const std::size_t slab = static_cast<std::size_t>(nao_out) * nvec;
  auto cshell = [&](bool isaux, int i) { return isaux ? nso + i : i; };
  const auto ghost = [&](const PrimitiveShell<Real> &sx) { return detail::ghost_shell(sx); };

  JKDerivResult<Real> res;
  res.nshell = ngrp;
  res.nao = nao_out;
  res.J.resize(1);
  res.K.resize(1);
  res.K[0].assign(static_cast<std::size_t>(npert) * N, Real(0));
  auto Kadd = [&](int x, int m, int n, Real v) {
    res.K[0][static_cast<std::size_t>(x) * N + static_cast<std::size_t>(m) * nao_out + n] +=
        v;
  };
  // ---- three-centre derivative pass, BATCHED -----------------------------
  // The quartets go through eri_quartets rather than one eri_block4 call per
  // shell triple, which is what ri_j_deriv_kernel has always done and is the
  // reason it measured 1 s where this kernel measured 68 s for the same number
  // of blocks. Pair intermediates are shared across the batch and the
  // evaluation is parallel; the digest below stays on the host, because its
  // inner work is a length-nao_out gemv and axpy rather than a scalar scatter.
  //
  // Only the two BRA positions are emitted -- the auxiliary derivative comes
  // from translational invariance, the ghost being exponent-free.
  if (aux_tile_shells < 1) aux_tile_shells = nsa;
  for (int C0 = 0; C0 < nsa; C0 += aux_tile_shells) {
  const int C1 = std::min(C0 + aux_tile_shells, nsa);
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
  for (int l = 0; l < nso; ++l)
    for (int n = 0; n < nso; ++n) {
      orb_pair(l, 1, n, 0);
      orb_pair(l, -1, n, 0);
      orb_pair(l, 0, n, 1);
      orb_pair(l, 0, n, -1);
    }
  const int nbra = static_cast<int>(plist.size());
  for (int c = C0; c < C1; ++c) plist.push_back(detail::ghost_pair(aux.shells[c]));
  std::vector<std::pair<int, int>> quartets;
  std::vector<int> jl, jn, jc, elp, elm, enp, enm;
  for (int l = 0; l < nso; ++l)
    for (int n = 0; n < nso; ++n)
      for (int c = C0; c < C1; ++c) {
        auto emit = [&](int bra) {
          if (bra < 0) return -1;
          const int e = static_cast<int>(quartets.size());
          quartets.push_back({bra, nbra + (c - C0)});
          return e;
        };
        jl.push_back(l); jn.push_back(n); jc.push_back(c);
        elp.push_back(emit(orb_pair(l, 1, n, 0)));
        elm.push_back(emit(orb_pair(l, -1, n, 0)));
        enp.push_back(emit(orb_pair(l, 0, n, 1)));
        enm.push_back(emit(orb_pair(l, 0, n, -1)));
      }
  const int njob = static_cast<int>(jl.size());
  auto tab = make_pair_table(plist);
  auto batch = make_batch(tab, quartets);
  QuartetWorkspace<Real> ws;
  Kokkos::View<Real *> qout("intti::rikd::out", batch.nout_total);
  eri_quartets(tab, batch, grid, qout, ws);

  // ---- digest, ON DEVICE -------------------------------------------------
  // The quartet buffer is the largest object in this kernel, so it must not
  // travel: the digest goes to it, not the other way round. Pulling it back to
  // host was a regression -- invisible on an OpenMP backend, where the copy is
  // nearly free, and the dominant cost on a real GPU.
  //
  // The parallel axis is (job, OUTPUT index n2), not the job alone, and that is
  // what makes it scratch-free. y1 and y3 contract over the vector index and
  // their only free index is the output one, so for a fixed n2 each is a SCALAR
  // rather than a length-nao_out vector -- computed once per (job, kn, kQ) and
  // reused across dir, kl, pos and the contraction fan-out. That is the same
  // reuse hoisting bought on the host, without any per-thread storage.
  auto dCL = detail::to_device(CL, static_cast<std::size_t>(orb.nao) * nvec, "rikd::cl");
  auto dCR = detail::to_device(CR, static_cast<std::size_t>(orb.nao) * nvec, "rikd::cr");
  auto dXh = detail::to_device(Xh, "rikd::xh"), dYh = detail::to_device(Yh, "rikd::yh");
  auto djl = detail::to_device(jl, "rikd::jl"), djn = detail::to_device(jn, "rikd::jn");
  auto djc = detail::to_device(jc, "rikd::jc");
  auto delp = detail::to_device(elp, "rikd::elp"), delm = detail::to_device(elm, "rikd::elm");
  auto denp = detail::to_device(enp, "rikd::enp"), denm = detail::to_device(enm, "rikd::enm");
  auto dgrp = detail::to_device(grp, "rikd::grp");
  auto dfnc = detail::to_device(fan.nctr, "rikd::nctr");
  auto dfco = detail::to_device(fan.coff, "rikd::coff");
  auto dfbs = detail::to_device(fan.base, "rikd::base");
  auto dfw = detail::to_device(fan.w, "rikd::w");
  std::vector<int> hoL(nso), hoO(nso), haL(nsa), haO(nsa);
  std::vector<Real> hoA(nso);
  for (int i2 = 0; i2 < nso; ++i2) {
    hoL[i2] = orb.shells[i2].l;
    hoO[i2] = orb.ao_off[i2];
    hoA[i2] = orb.shells[i2].alpha;
  }
  for (int i2 = 0; i2 < nsa; ++i2) {
    haL[i2] = aux.shells[i2].l;
    haO[i2] = aux.ao_off[i2];
  }
  auto doL = detail::to_device(hoL, "rikd::oL"), doO = detail::to_device(hoO, "rikd::oO");
  auto doA = detail::to_device(hoA, "rikd::oA");
  auto daL = detail::to_device(haL, "rikd::aL"), daO = detail::to_device(haO, "rikd::aO");
  auto offv = batch.out_offset;
  Kokkos::View<Real *> Kd("rikd::K", static_cast<std::size_t>(npert) * N);
  const int nsoK = nso, nvecK = nvec, naoK = nao_out;
  const std::size_t slabK = slab, NK = N;
  Kokkos::parallel_for(
      "intti::rikd::digest",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {njob, nao_out}),
      KOKKOS_LAMBDA(int j, int n2) {
        const int l = djl(j), n = djn(j), c = djc(j);
        const int ll = doL(l), ln = doL(n), lp = daL(c);
        const int nl = ncart(ll), nn = ncart(ln), nP = ncart(lp);
        const int on = doO(n), oc = daO(c);
        const int cs[3] = {dgrp(l), dgrp(n), dgrp(nsoK + c)};
        const int ent[2][2] = {{delp(j), delm(j)}, {denp(j), denm(j)}};
        for (int kn = 0; kn < nn; ++kn) {
          int n3[3];
          cart_comp(ln, kn, n3[0], n3[1], n3[2]);
          for (int kQ = 0; kQ < nP; ++kQ) {
            const int B = on + kn, P = oc + kQ;
            Real y1 = 0, y3 = 0; // scalars: n2 is this thread's own index
            for (int i3 = 0; i3 < nvecK; ++i3) {
              y1 += dCL(static_cast<std::size_t>(B) * nvecK + i3) *
                    dYh(static_cast<std::size_t>(P) * slabK + n2 * nvecK + i3);
              y3 += dXh(static_cast<std::size_t>(P) * slabK + n2 * nvecK + i3) *
                    dCR(static_cast<std::size_t>(B) * nvecK + i3);
            }
            if (y1 == Real(0) && y3 == Real(0)) continue;
            for (int dir = 0; dir < 3; ++dir)
              for (int kl = 0; kl < nl; ++kl) {
                int l3[3];
                cart_comp(ll, kl, l3[0], l3[1], l3[2]);
                Real dv[3];
                for (int slot = 0; slot < 2; ++slot) {
                  int sg[2], ci[2];
                  Real co[2];
                  const int nt = slot == 0
                                     ? detail::md_grad_terms(ll, l3, doA(l), dir, sg, ci, co)
                                     : detail::md_grad_terms(ln, n3, doA(n), dir, sg, ci, co);
                  Real v = 0;
                  for (int t = 0; t < nt; ++t) {
                    const int ee = ent[slot][sg[t]];
                    if (ee < 0) continue;
                    const int sh = (sg[t] == 0) ? 1 : -1;
                    const int mn2 = (slot == 1) ? ncart(ln + sh) : nn;
                    const int ia = (slot == 0) ? ci[t] : kl;
                    const int ib = (slot == 1) ? ci[t] : kn;
                    v += co[t] * qout(offv(ee) +
                                      (static_cast<std::size_t>(ia) * mn2 + ib) * nP + kQ);
                  }
                  dv[slot] = -v; // md_grad_terms is d/dx; the centre derivative is -it
                }
                dv[2] = -(dv[0] + dv[1]); // ghost is exponent-free
                for (int pos = 0; pos < 3; ++pos) {
                  if (dv[pos] == Real(0)) continue;
                  const std::size_t xo = static_cast<std::size_t>(3 * cs[pos] + dir) * NK;
                  // the orbital index out of the derivative block is PRIMITIVE
                  // and fans out onto the contracted AOs it feeds; n2 is an
                  // output and is already contracted.
                  for (int cM = 0; cM < dfnc(l); ++cM) {
                    const Real w = dfw(dfco(l) + cM) * dv[pos];
                    const int I = dfbs(l) + cM * nl + kl;
                    Kokkos::atomic_add(&Kd(xo + static_cast<std::size_t>(I) * naoK + n2),
                                       w * y1);
                    Kokkos::atomic_add(&Kd(xo + static_cast<std::size_t>(n2) * naoK + I),
                                       w * y3);
                  }
                }
              }
          }
        }
      });
  Kokkos::fence();
  {
    const auto hK = detail::to_host(Kd);
    for (std::size_t i2 = 0; i2 < hK.size(); ++i2) res.K[0][i2] += hK[i2];
  }
  }
  // term 2, one perturbation at a time
  {
    std::vector<Real> Amat(static_cast<std::size_t>(naux) * slab);
    std::vector<Real> dM(static_cast<std::size_t>(naux) * naux);
    for (int x = 0; x < npert; ++x) {
      std::fill(dM.begin(), dM.end(), Real(0));
      bool any = false;
      for (int c = 0; c < nsa; ++c)
        for (int ee = 0; ee < nsa; ++ee) {
          const auto &sC = aux.shells[c], &sE = aux.shells[ee];
          const auto ghC = ghost(sC), ghE = ghost(sE);
          const int oC = aux.ao_off[c], oE = aux.ao_off[ee];
          const int nC = ncart(sC.l), nE = ncart(sE.l);
          for (int pos : {0, 2}) {
            const int cs = (pos == 0) ? cshell(true, c) : cshell(true, ee);
            // only one direction survives the filter below, so the
            // single-direction form is the right one here -- computing all
            // three would discard two of them.
            for (int dir = 0; dir < 3; ++dir) {
              // every shell in the group contributes to the group's perturbation
              if (3 * grp[cs] + dir != x) continue;
              const auto blk =
                  detail::quartet_pos_deriv_block(sC, ghC, sE, ghE, pos, dir, grid);
              for (int kQ = 0; kQ < nC; ++kQ)
                for (int kR = 0; kR < nE; ++kR) {
                  const Real dv = blk[static_cast<std::size_t>(kQ) * nE + kR];
                  if (dv == Real(0)) continue;
                  dM[static_cast<std::size_t>(oC + kQ) * naux + oE + kR] += dv;
                  any = true;
                }
            }
          }
        }
      if (!any) continue;
      detail::gemm('N', 'N', naux, static_cast<int>(slab), naux, Real(1), dM.data(), naux,
                   Xh.data(), static_cast<int>(slab), Real(0), Amat.data(),
                   static_cast<int>(slab));
      for (int P = 0; P < naux; ++P)
        // both orbital indices here come from Xh and Yh, which are already in
        // the OUTPUT basis -- so nao_out, not the primitive nao
        detail::gemm('N', 'T', nao_out, nao_out, nvec, Real(-1),
                     Amat.data() + static_cast<std::size_t>(P) * slab, nvec,
                     Yh.data() + static_cast<std::size_t>(P) * slab, nvec, Real(1),
                     res.K[0].data() + static_cast<std::size_t>(x) * N, nao_out);
    }
  }
  return res;
}

} // namespace detail

/// Derivative RI exchange matrices for a factorised density, over primitive
/// shells.
template <class Real>
JKDerivResult<Real> ri_k_deriv_occ(const ShellBasis<Real> &orb, const ShellBasis<Real> &aux,
                                   const Real *CL, const Real *CR, int nvec,
                                   const TGrid<Real> &grid, Real tau_lin = Real(1e-10),
                                   int aux_tile_shells = 0,
                                   const std::vector<int> &group = {}) {
  const auto Minv = detail::ri_metric_inverse(aux, grid, tau_lin);
  const auto X = detail::ri_k_build_Y(orb, aux, CL, nvec, grid, aux_tile_shells);
  const auto Y = detail::ri_k_build_Y(orb, aux, CR, nvec, grid, aux_tile_shells);
  const int naux = aux.nao;
  const int sl = orb.nao * nvec;
  std::vector<Real> Xh(X.size(), Real(0)), Yh(Y.size(), Real(0));
  detail::gemm('N', 'N', naux, sl, naux, Real(1), Minv.data(), naux, X.data(), sl, Real(0),
               Xh.data(), sl);
  detail::gemm('N', 'N', naux, sl, naux, Real(1), Minv.data(), naux, Y.data(), sl, Real(0),
               Yh.data(), sl);
  const int ncen =
      static_cast<int>(orb.shells.size()) + static_cast<int>(aux.shells.size());
  int ngrp = 0;
  const auto grp = detail::perturbation_groups(group, ncen, ngrp);
  return detail::ri_k_deriv_kernel(orb, aux, CL, CR, nvec, Xh, Yh,
                                   static_cast<const detail::ShellFanout<Real> *>(nullptr),
                                   grp, ngrp, grid, aux_tile_shells);
}

/// Same, over generally-contracted bases. CL, CR and the result are in
/// CONTRACTED AOs.
template <class Real>
JKDerivResult<Real> ri_k_deriv_occ(const ContractedBasis<Real> &orb,
                                   const ContractedBasis<Real> &aux, const Real *CL,
                                   const Real *CR, int nvec, const TGrid<Real> &grid,
                                   Real tau_lin = Real(1e-10), int aux_tile_shells = 0,
                                   const std::vector<int> &group = {}) {
  const int naoc = orb.nao, nauxc = aux.nao;
  const auto Minv = detail::ri_metric_inverse(aux, grid, tau_lin);
  // X and Y with CONTRACTED indices throughout: their orbital index is an
  // output, and their auxiliary index meets M^{-1}.
  const auto X = detail::ri_k_build_Y(orb, aux, CL, nvec, grid, aux_tile_shells);
  const auto Y = detail::ri_k_build_Y(orb, aux, CR, nvec, grid, aux_tile_shells);
  const int sl = naoc * nvec;
  std::vector<Real> Xhc(X.size(), Real(0)), Yhc(Y.size(), Real(0));
  detail::gemm('N', 'N', nauxc, sl, nauxc, Real(1), Minv.data(), nauxc, X.data(), sl,
               Real(0), Xhc.data(), sl);
  detail::gemm('N', 'N', nauxc, sl, nauxc, Real(1), Minv.data(), nauxc, Y.data(), sl,
               Real(0), Yhc.data(), sl);

  ShellBasis<Real> po, pa;
  const auto fo = detail::expand_contracted(orb, po);
  const auto fa = detail::expand_contracted(aux, pa);
  const auto Co = detail::fanout_matrix(fo, po);
  const auto Ca = detail::fanout_matrix(fa, pa);
  const int naop = po.nao, nauxp = pa.nao;
  // ...then LIFT the auxiliary index to primitive, which is where every
  // derivative block meets it.
  auto lift = [&](const std::vector<Real> &A) {
    std::vector<Real> out(static_cast<std::size_t>(nauxp) * sl, Real(0));
    detail::gemm('T', 'N', nauxp, sl, nauxc, Real(1), Ca.data(), nauxp, A.data(), sl,
                 Real(0), out.data(), sl);
    return out;
  };
  const auto Xh = lift(Xhc), Yh = lift(Yhc);
  // orbital factors push DOWN
  auto pushdown = [&](const Real *Z) {
    std::vector<Real> Zp(static_cast<std::size_t>(naop) * nvec, Real(0));
    detail::gemm('T', 'N', naop, nvec, naoc, Real(1), Co.data(), naop, Z, nvec, Real(0),
                 Zp.data(), nvec);
    return Zp;
  };
  const auto CLp = pushdown(CL), CRp = pushdown(CR);

  const int ncen = static_cast<int>(po.shells.size()) + static_cast<int>(pa.shells.size());
  std::vector<int> parent(ncen);
  for (int i = 0; i < static_cast<int>(po.shells.size()); ++i) parent[i] = fo.parent[i];
  for (int i = 0; i < static_cast<int>(pa.shells.size()); ++i)
    parent[static_cast<int>(po.shells.size()) + i] = fo.nsh + fa.parent[i];
  std::vector<int> grp(ncen);
  int ngrp = fo.nsh + fa.nsh;
  for (int i = 0; i < ncen; ++i) grp[i] = group.empty() ? parent[i] : group[parent[i]];
  if (!group.empty()) {
    ngrp = 0;
    for (int v : grp) ngrp = std::max(ngrp, v + 1);
  }
  return detail::ri_k_deriv_kernel(po, pa, CLp.data(), CRp.data(), nvec, Xh, Yh, &fo, grp,
                                   ngrp, grid, aux_tile_shells);
}

} // namespace intti
