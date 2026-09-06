// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Two-electron energy gradient -- the black-box forces piece. The MD shift
// relation extends to the quartet: differentiating shell p of (ab|cd) w.r.t.
// its centre raises/lowers that shell's angular momentum
//   nabla_e phi_p[m] = m_e phi_p[m_e-1] - 2 alpha_p phi_p[m_e+1],
// so d/dR_p (ab|cd) = -(nabla_e) is a linear combination of quartets with
// shell p promoted and demoted. Contracted with the closed-shell two-particle
// density it gives dE_2e/dR per shell centre. Matrix-level: density in, forces
// out; individual quartets stay internal.
//
// Permutational symmetry: (ab|cd) is invariant under a<->b, c<->d and
// (ab)<->(cd), an 8-element group. We loop over CANONICAL shell quartets only
// (a>=b, c>=d, pair(a,b)>=pair(c,d)), evaluate the promoted/demoted integral
// blocks once, and replay the *unchanged* per-quartet contraction on every
// distinct orbit member by permuting the block axes. Correctness of the sign
// bookkeeping and centre routing therefore rides entirely on the original body,
// which recomputes everything from each member's own shell data; only the
// (permutation-invariant) integral values are shared. ~8x fewer quartet evals.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "batch.hpp"  // PairTable, make_batch, eri_quartets (device driver)
#include "device.hpp" // detail::to_device / to_host
#include "fock.hpp"
#include "gto.hpp"
#include "quartet.hpp"
#include "tgrid.hpp"

namespace intti {

namespace detail {
/// index in shell l of the Cartesian component (lx,ly,lz) (cart_comp order).
KOKKOS_INLINE_FUNCTION int comp_index(int l, int lx, int ly) {
  return (l - lx) * (l - lx + 1) / 2 + (l - lx - ly);
}

/// (ab|cd) block for four explicit shells, layout ((ka*ncb+kb)*ncc+kc)*ncd+kd.
template <class Real>
std::vector<Real> eri_block4(const PrimitiveShell<Real> &a,
                             const PrimitiveShell<Real> &b,
                             const PrimitiveShell<Real> &c,
                             const PrimitiveShell<Real> &d, const TGrid<Real> &grid) {
  std::vector<Real> blk(static_cast<std::size_t>(ncart(a.l)) * ncart(b.l) *
                        ncart(c.l) * ncart(d.l));
  eri_quartet(make_pair(a, b), make_pair(c, d), grid, blk.data());
  return blk;
}

/// The 8 permutations of the ERI symmetry group, as maps member-slot -> source
/// (canonical) slot for the four positions (a,b,c,d).
inline constexpr int eri_perms[8][4] = {
    {0, 1, 2, 3}, {1, 0, 2, 3}, {0, 1, 3, 2}, {1, 0, 3, 2},
    {2, 3, 0, 1}, {3, 2, 0, 1}, {2, 3, 1, 0}, {3, 2, 1, 0}};

/// Reindex a 4-axis Cartesian block under an ERI symmetry permutation.
/// perm[j] is the canonical (source) slot supplying member slot j; dimC are the
/// canonical per-axis Cartesian dimensions. Returns the block in member layout.
template <class Real>
std::vector<Real> permute_block(const std::vector<Real> &canon, const int dimC[4],
                                const int perm[4]) {
  int inv[4];
  for (int j = 0; j < 4; ++j) inv[perm[j]] = j;
  int dimM[4];
  for (int j = 0; j < 4; ++j) dimM[j] = dimC[perm[j]];
  std::vector<Real> out(static_cast<std::size_t>(dimM[0]) * dimM[1] * dimM[2] * dimM[3]);
  int jm[4];
  for (jm[0] = 0; jm[0] < dimM[0]; ++jm[0])
    for (jm[1] = 0; jm[1] < dimM[1]; ++jm[1])
      for (jm[2] = 0; jm[2] < dimM[2]; ++jm[2])
        for (jm[3] = 0; jm[3] < dimM[3]; ++jm[3]) {
          int ic[4];
          for (int c = 0; c < 4; ++c) ic[c] = jm[inv[c]];
          const std::size_t ci =
              ((static_cast<std::size_t>(ic[0]) * dimC[1] + ic[1]) * dimC[2] + ic[2]) * dimC[3] +
              ic[3];
          const std::size_t mi =
              ((static_cast<std::size_t>(jm[0]) * dimM[1] + jm[1]) * dimM[2] + jm[2]) * dimM[3] +
              jm[3];
          out[mi] = canon[ci];
        }
  return out;
}

/// Schwarz bound sqrt(max|(ab|ab)|) for a shell pair, taken as the max over the
/// base pair and its two single (l+1) promotions so it also bounds the
/// first-derivative integrals of the pair. Cauchy--Schwarz makes the block
/// maximum equal to the diagonal maximum, so max-over-entries is exact.
template <class Real>
Real pair_schwarz_margin(const PrimitiveShell<Real> &a, const PrimitiveShell<Real> &b,
                         const TGrid<Real> &grid) {
  auto q = [&](const PrimitiveShell<Real> &x, const PrimitiveShell<Real> &y) {
    auto blk = eri_block4(x, y, x, y, grid);
    Real m = 0;
    for (Real v : blk) m = std::max(m, std::abs(v));
    return std::sqrt(m);
  };
  PrimitiveShell<Real> ap = a, bp = b;
  ap.l += 1;
  bp.l += 1;
  return std::max({q(a, b), q(ap, b), q(a, bp)});
}

// ---- device (GPU) two-electron gradient -------------------------------------
// Drop the 8-fold symmetry replay: loop ALL ordered quartets (a,b,c,d) once and
// contribute each, which equals the canonical+replay sum. Batch every quartet's
// promoted/demoted blocks through eri_quartets (one device pass) and run the
// gradient contraction on device with atomic accumulation into the per-shell
// forces. Exact (tau = 0) path only; correctness-first, so it materialises the
// full quartet list (fine for the validation sizes; screened/streamed + symmetry
// reinstated is the production follow-up). l <= LMAX-1 (each shell is promoted).
template <class Real>
std::vector<std::array<Real, 3>>
two_electron_gradient_dev(const ShellBasis<Real> &basis, const Real *D,
                          const TGrid<Real> &grid) {
  const int ns = static_cast<int>(basis.shells.size());
  const int nao = basis.nao;
  // per-shell device data
  std::vector<int> hL(ns), hOff(ns);
  std::vector<Real> hAl(ns);
  for (int i = 0; i < ns; ++i) {
    hL[i] = basis.shells[i].l;
    hOff[i] = basis.ao_off[i];
    hAl[i] = basis.shells[i].alpha;
  }
  auto shL = to_device(hL, "intti::g2::L");
  auto shOff = to_device(hOff, "intti::g2::off");
  auto shAl = to_device(hAl, "intti::g2::al");
  auto Dd = to_device(D, static_cast<std::size_t>(nao) * nao, "intti::g2::D");

  // build the pair list, the batch (bra,ket) list, and per-quartet block entries
  std::vector<ShellPair<Real>> plist;
  auto add_pair = [&](int si, int di, int sj, int dj) {
    PrimitiveShell<Real> a = basis.shells[si], b = basis.shells[sj];
    a.l += di;
    b.l += dj;
    plist.push_back(make_pair(a, b));
    return static_cast<int>(plist.size()) - 1;
  };
  std::vector<std::pair<int, int>> quartets;
  std::vector<int> qa, qb, qc, qd; // shells per job (ordered quartet)
  std::vector<int> plusE, minusE;  // 4 per job: batch-entry index, -1 if absent
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b)
      for (int c = 0; c < ns; ++c)
        for (int d = 0; d < ns; ++d) {
          const int L[4] = {hL[a], hL[b], hL[c], hL[d]};
          qa.push_back(a); qb.push_back(b); qc.push_back(c); qd.push_back(d);
          for (int pos = 0; pos < 4; ++pos) {
            // plus: shell at pos promoted
            int bp = (pos == 0) ? add_pair(a, 1, b, 0)
                     : (pos == 1) ? add_pair(a, 0, b, 1)
                                  : add_pair(a, 0, b, 0);
            int kp = (pos == 2) ? add_pair(c, 1, d, 0)
                     : (pos == 3) ? add_pair(c, 0, d, 1)
                                  : add_pair(c, 0, d, 0);
            plusE.push_back(static_cast<int>(quartets.size()));
            quartets.push_back({bp, kp});
            // minus: shell at pos demoted (only if l>=1)
            if (L[pos] >= 1) {
              int bm = (pos == 0) ? add_pair(a, -1, b, 0)
                       : (pos == 1) ? add_pair(a, 0, b, -1)
                                    : add_pair(a, 0, b, 0);
              int km = (pos == 2) ? add_pair(c, -1, d, 0)
                       : (pos == 3) ? add_pair(c, 0, d, -1)
                                    : add_pair(c, 0, d, 0);
              minusE.push_back(static_cast<int>(quartets.size()));
              quartets.push_back({bm, km});
            } else {
              minusE.push_back(-1);
            }
          }
        }
  const int njob = static_cast<int>(qa.size());
  auto tab = make_pair_table(plist);
  auto batch = make_batch(tab, quartets);
  QuartetWorkspace<Real> ws;
  Kokkos::View<Real *> out("intti::g2::out", batch.nout_total);
  eri_quartets(tab, batch, grid, out, ws);

  auto dqa = to_device(qa, "intti::g2::qa"), dqb = to_device(qb, "intti::g2::qb");
  auto dqc = to_device(qc, "intti::g2::qc"), dqd = to_device(qd, "intti::g2::qd");
  auto dplus = to_device(plusE, "intti::g2::plusE");
  auto dminus = to_device(minusE, "intti::g2::minusE");
  auto offv = batch.out_offset;
  Kokkos::View<Real *[3], Kokkos::LayoutLeft> gradd("intti::g2::grad", ns);
  Kokkos::parallel_for(
      "intti::g2::digest", Kokkos::RangePolicy<>(0, njob), KOKKOS_LAMBDA(int j) {
        const int sh[4] = {dqa(j), dqb(j), dqc(j), dqd(j)};
        const int La = shL(sh[0]), Lb = shL(sh[1]), Lc = shL(sh[2]), Ld = shL(sh[3]);
        const int na = ncart(La), nb = ncart(Lb), nc = ncart(Lc), nd = ncart(Ld);
        const int off0 = shOff(sh[0]), off1 = shOff(sh[1]), off2 = shOff(sh[2]), off3 = shOff(sh[3]);
        auto Dm = [&](int i, int k) { return Dd(static_cast<std::size_t>(i) * nao + k); };
        for (int pos = 0; pos < 4; ++pos) {
          const int Lp = shL(sh[pos]);
          const int pe = dplus(j * 4 + pos), me = dminus(j * 4 + pos);
          const int pbase = offv(pe);
          const int mbase = (me >= 0) ? offv(me) : 0;
          // block dims for plus (pos promoted) and minus (pos demoted)
          const int dP[4] = {ncart(pos == 0 ? La + 1 : La), ncart(pos == 1 ? Lb + 1 : Lb),
                             ncart(pos == 2 ? Lc + 1 : Lc), ncart(pos == 3 ? Ld + 1 : Ld)};
          const int dM[4] = {ncart(pos == 0 ? La - 1 : La), ncart(pos == 1 ? Lb - 1 : Lb),
                             ncart(pos == 2 ? Lc - 1 : Lc), ncart(pos == 3 ? Ld - 1 : Ld)};
          const Real ap = shAl(sh[pos]);
          for (int ka = 0; ka < na; ++ka) {
            int a3[3];
            cart_comp(La, ka, a3[0], a3[1], a3[2]);
            for (int kb = 0; kb < nb; ++kb) {
              int b3[3];
              cart_comp(Lb, kb, b3[0], b3[1], b3[2]);
              for (int kc = 0; kc < nc; ++kc) {
                int c3[3];
                cart_comp(Lc, kc, c3[0], c3[1], c3[2]);
                for (int kd = 0; kd < nd; ++kd) {
                  int d3[3];
                  cart_comp(Ld, kd, d3[0], d3[1], d3[2]);
                  const Real coeff =
                      Real(0.5) * Dm(off0 + ka, off1 + kb) * Dm(off2 + kc, off3 + kd) -
                      Real(0.25) * Dm(off0 + ka, off2 + kc) * Dm(off1 + kb, off3 + kd);
                  if (coeff == Real(0)) continue;
                  const int *base3 = pos == 0 ? a3 : pos == 1 ? b3 : pos == 2 ? c3 : d3;
                  const int idxc[4] = {ka, kb, kc, kd};
                  for (int e = 0; e < 3; ++e) {
                    int mm[3] = {base3[0], base3[1], base3[2]};
                    mm[e] = base3[e] + 1;
                    const int ip = comp_index(Lp + 1, mm[0], mm[1]);
                    int ic[4] = {idxc[0], idxc[1], idxc[2], idxc[3]};
                    ic[pos] = ip;
                    const std::size_t pidx =
                        (((static_cast<std::size_t>(ic[0]) * dP[1] + ic[1]) * dP[2] + ic[2]) *
                             dP[3] + ic[3]);
                    Real term = 2 * ap * out(pbase + pidx);
                    if (base3[e] >= 1 && me >= 0) {
                      mm[e] = base3[e] - 1;
                      const int im = comp_index(Lp - 1, mm[0], mm[1]);
                      int icm[4] = {idxc[0], idxc[1], idxc[2], idxc[3]};
                      icm[pos] = im;
                      const std::size_t midx =
                          (((static_cast<std::size_t>(icm[0]) * dM[1] + icm[1]) * dM[2] + icm[2]) *
                               dM[3] + icm[3]);
                      term -= Real(base3[e]) * out(mbase + midx);
                    }
                    Kokkos::atomic_add(&gradd(sh[pos], e), coeff * term);
                  }
                }
              }
            }
          }
        }
      });
  auto gh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gradd);
  std::vector<std::array<Real, 3>> grad(ns);
  for (int i = 0; i < ns; ++i) grad[i] = {gh(i, 0), gh(i, 1), gh(i, 2)};
  return grad;
}
} // namespace detail

/// Gradient of the closed-shell two-electron energy
///   E_2e = 1/2 sum D_mn D_ls (mn|ls) - 1/4 sum D_ml D_ns (mn|ls)
/// with respect to each shell centre. Returns forces[nshell][3] = dE_2e/dR_s.
/// D is nao x nao symmetric (row-major). tau > 0 enables Schwarz + density
/// screening of shell quartets (tau = 0, the default, is exact).
template <class Real>
std::vector<std::array<Real, 3>>
two_electron_gradient(const ShellBasis<Real> &basis, const Real *D,
                      const TGrid<Real> &grid, Real tau = Real(0)) {
  if constexpr (kokkos_scalar_v<Real>) {
    // exact (unscreened) case -> GPU; screened path keeps the host symmetry replay
    if (tau == Real(0)) {
      bool ok = true;
      for (const auto &s : basis.shells)
        if (s.l >= LMAX) ok = false; // each shell is promoted to l+1
      if (ok) return detail::two_electron_gradient_dev(basis, D, grid);
    }
  }
  const int ns = static_cast<int>(basis.shells.size());
  const int nao = basis.nao;
  std::vector<std::array<Real, 3>> grad(ns, {Real(0), Real(0), Real(0)});
  auto Dm = [&](int i, int j) { return D[static_cast<std::size_t>(i) * nao + j]; };

  // per-shell-pair Schwarz bounds and density-block maxima for screening
  std::vector<Real> Q, Dmax;
  if (tau > Real(0)) {
    Q.assign(static_cast<std::size_t>(ns) * ns, Real(0));
    Dmax.assign(static_cast<std::size_t>(ns) * ns, Real(0));
    for (int i = 0; i < ns; ++i)
      for (int j = 0; j < ns; ++j) {
        Q[i * ns + j] = detail::pair_schwarz_margin(basis.shells[i], basis.shells[j], grid);
        Real m = 0;
        for (int ki = 0; ki < ncart(basis.shells[i].l); ++ki)
          for (int kj = 0; kj < ncart(basis.shells[j].l); ++kj)
            m = std::max(m, std::abs(Dm(basis.ao_off[i] + ki, basis.ao_off[j] + kj)));
        Dmax[i * ns + j] = m;
      }
  }

  // contract one (member) shell quartet against the density, given its
  // promoted/demoted blocks in member layout. This is the original, unmodified
  // per-quartet gradient body -- all signs and centre routing live here.
  auto contribute = [&](const int shm[4], const int Lm[4], const Real alm[4],
                        const int offm[4], const std::vector<Real> plus[4],
                        const std::vector<Real> minus[4]) {
    const int na = ncart(Lm[0]), nb = ncart(Lm[1]), nc = ncart(Lm[2]), nd = ncart(Lm[3]);
    auto idx = [](const int n[4], int i0, int i1, int i2, int i3) {
      return ((static_cast<std::size_t>(i0) * n[1] + i1) * n[2] + i2) * n[3] + i3;
    };
    for (int pos = 0; pos < 4; ++pos) {
      const int lp = Lm[pos];
      const int npl[4] = {ncart(pos == 0 ? Lm[0] + 1 : Lm[0]), ncart(pos == 1 ? Lm[1] + 1 : Lm[1]),
                          ncart(pos == 2 ? Lm[2] + 1 : Lm[2]), ncart(pos == 3 ? Lm[3] + 1 : Lm[3])};
      const int nmi[4] = {ncart(pos == 0 ? Lm[0] - 1 : Lm[0]), ncart(pos == 1 ? Lm[1] - 1 : Lm[1]),
                          ncart(pos == 2 ? Lm[2] - 1 : Lm[2]), ncart(pos == 3 ? Lm[3] - 1 : Lm[3])};
      for (int ka = 0; ka < na; ++ka) {
        int a3[3];
        cart_comp(Lm[0], ka, a3[0], a3[1], a3[2]);
        for (int kb = 0; kb < nb; ++kb) {
          int b3[3];
          cart_comp(Lm[1], kb, b3[0], b3[1], b3[2]);
          for (int kc = 0; kc < nc; ++kc) {
            int c3[3];
            cart_comp(Lm[2], kc, c3[0], c3[1], c3[2]);
            for (int kd = 0; kd < nd; ++kd) {
              int d3[3];
              cart_comp(Lm[3], kd, d3[0], d3[1], d3[2]);
              const Real coeff =
                  Real(0.5) * Dm(offm[0] + ka, offm[1] + kb) * Dm(offm[2] + kc, offm[3] + kd) -
                  Real(0.25) * Dm(offm[0] + ka, offm[2] + kc) * Dm(offm[1] + kb, offm[3] + kd);
              if (coeff == Real(0)) continue;
              int m[3];
              const int *base3 = pos == 0 ? a3 : pos == 1 ? b3 : pos == 2 ? c3 : d3;
              for (int e = 0; e < 3; ++e) {
                m[0] = base3[0]; m[1] = base3[1]; m[2] = base3[2];
                m[e] = base3[e] + 1;
                const int ip = detail::comp_index(lp + 1, m[0], m[1]);
                Real term = 2 * alm[pos] *
                            plus[pos][pos == 0 ? idx(npl, ip, kb, kc, kd)
                                      : pos == 1 ? idx(npl, ka, ip, kc, kd)
                                      : pos == 2 ? idx(npl, ka, kb, ip, kd)
                                                 : idx(npl, ka, kb, kc, ip)];
                if (base3[e] >= 1) {
                  m[e] = base3[e] - 1;
                  const int im = detail::comp_index(lp - 1, m[0], m[1]);
                  term -= Real(base3[e]) *
                          minus[pos][pos == 0 ? idx(nmi, im, kb, kc, kd)
                                     : pos == 1 ? idx(nmi, ka, im, kc, kd)
                                     : pos == 2 ? idx(nmi, ka, kb, im, kd)
                                                : idx(nmi, ka, kb, kc, im)];
                }
                grad[shm[pos]][e] += coeff * term;
              }
            }
          }
        }
      }
    }
  };

  // canonical shell quartets: a>=b, c>=d, pair(a,b) >= pair(c,d)
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b <= a; ++b) {
      const int Pab = a * ns + b;
      for (int cc = 0; cc < ns; ++cc)
        for (int dd = 0; dd <= cc; ++dd) {
          if (cc * ns + dd > Pab) continue;
          const int canon[4] = {a, b, cc, dd};
          const int Lc[4] = {basis.shells[a].l, basis.shells[b].l, basis.shells[cc].l,
                             basis.shells[dd].l};

          if (tau > Real(0)) {
            const Real qab = Q[a * ns + b], qcd = Q[cc * ns + dd];
            Real dw = 0; // max |D| across the six shell-pair blocks of the orbit
            const int sh[4] = {a, b, cc, dd};
            for (int i = 0; i < 4; ++i)
              for (int j = 0; j < 4; ++j) dw = std::max(dw, Dmax[sh[i] * ns + sh[j]]);
            const Real almax =
                std::max({basis.shells[a].alpha, basis.shells[b].alpha,
                          basis.shells[cc].alpha, basis.shells[dd].alpha});
            if (2 * almax * qab * qcd * dw * dw < tau) continue;
          }

          // canonical promoted/demoted blocks, one per canonical slot
          std::vector<Real> cplus[4], cminus[4];
          for (int p = 0; p < 4; ++p) {
            auto mk = [&](int slot, int dl) {
              auto s = basis.shells[canon[slot]];
              s.l += dl;
              return s;
            };
            cplus[p] = detail::eri_block4(mk(0, p == 0), mk(1, p == 1), mk(2, p == 2),
                                          mk(3, p == 3), grid);
            if (Lc[p] >= 1)
              cminus[p] = detail::eri_block4(mk(0, p == 0 ? -1 : 0), mk(1, p == 1 ? -1 : 0),
                                             mk(2, p == 2 ? -1 : 0), mk(3, p == 3 ? -1 : 0), grid);
          }
          const int dC[4] = {ncart(Lc[0]), ncart(Lc[1]), ncart(Lc[2]), ncart(Lc[3])};

          // enumerate distinct orbit members and contract each once
          int seen[8][4];
          int nseen = 0;
          for (int g = 0; g < 8; ++g) {
            const int *pm = detail::eri_perms[g];
            const int mem[4] = {canon[pm[0]], canon[pm[1]], canon[pm[2]], canon[pm[3]]};
            bool dup = false;
            for (int t = 0; t < nseen && !dup; ++t)
              dup = seen[t][0] == mem[0] && seen[t][1] == mem[1] && seen[t][2] == mem[2] &&
                    seen[t][3] == mem[3];
            if (dup) continue;
            for (int t = 0; t < 4; ++t) seen[nseen][t] = mem[t];
            ++nseen;

            const int Lm[4] = {Lc[pm[0]], Lc[pm[1]], Lc[pm[2]], Lc[pm[3]]};
            const Real alm[4] = {basis.shells[mem[0]].alpha, basis.shells[mem[1]].alpha,
                                 basis.shells[mem[2]].alpha, basis.shells[mem[3]].alpha};
            const int offm[4] = {basis.ao_off[mem[0]], basis.ao_off[mem[1]],
                                 basis.ao_off[mem[2]], basis.ao_off[mem[3]]};
            std::vector<Real> mplus[4], mminus[4];
            for (int pos = 0; pos < 4; ++pos) {
              const int cp = pm[pos];
              int dPlus[4];
              for (int t = 0; t < 4; ++t) dPlus[t] = dC[t];
              dPlus[cp] = ncart(Lc[cp] + 1);
              mplus[pos] = detail::permute_block(cplus[cp], dPlus, pm);
              if (Lc[cp] >= 1) {
                int dMinus[4];
                for (int t = 0; t < 4; ++t) dMinus[t] = dC[t];
                dMinus[cp] = ncart(Lc[cp] - 1);
                mminus[pos] = detail::permute_block(cminus[cp], dMinus, pm);
              }
            }
            contribute(mem, Lm, alm, offm, mplus, mminus);
          }
        }
    }
  return grad;
}

} // namespace intti
