// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// One-electron spin-orbit spatial integrals (roadmap M17). Per component k:
//
//   W_k,uv = sum_A Z_A  eps_kij  <chi_u| (r-R_A)_i / |r-R_A|^3  d/dr_j |chi_v>,
//
// the field of nucleus A crossed with the momentum. The key identity keeps it on
// the ordinary attraction t-quadrature -- no 1/r^3 kernel is needed:
//
//   (r-R_A)_i / |r-R_A|^3 = d/dR_A,i (1/|r-R_A|),
//
// so W_k = eps_kij sum_A Z_A d/dR_A,i <chi_u| 1/r_A | d_j chi_v>: the nuclear-
// attraction integral between chi_u and the ket derivative d_j chi_v,
// differentiated with respect to the charge position R_A. In the McMurchie-
// Davidson / t-quadrature axis factorisation the attraction axis factor is
// g_d = pref sum_tau E^d_tau B_tau(theta, P_d - R_A,d) with B_n = (d/dX)^n
// exp(-theta X^2) (hermite1d.hpp), so per axis we form three 1D factors from the
// same E and B arrays:
//   plain_d  = pref sum_tau E_tau B_tau                 (1/r_A)
//   field_d  = -pref sum_tau E_tau B_{tau+1}            (d/dR_A,d, since dX=-dR_A)
//   ketder_d = b_d plain_d(l_b-1) - 2 beta plain_d(l_b+1)  (d/dr_d on the ket)
// and assemble W_x = plain_x(field_y ketder_z - field_z ketder_y), cyclically.
//
// Real and antisymmetric (the -i of p and the alpha^2/2 spin-coupling prefactor
// are the caller's convention; cf. PySCF int1e_prinvxp per atom, up to the -i).
// Matrix-level API: whole nao x nao matrices, one per Cartesian component.
//
// The two-electron spin-orbit (spin-same-orbit) builder below uses the same
// idea on the ERI: (r12)_i/r12^3 = -d_{1,i}(1/r12) integrated by parts moves the
// operator onto electron-1's two functions, and the symmetric mu d_i d_j lambda
// term dies against eps_kij, leaving plain Coulomb integrals with mu,lambda
// differentiated -- promoted/demoted quartets, no 1/r12^3 kernel.

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "erigrad.hpp" // detail::eri_block4, detail::comp_index
#include "fock.hpp"
#include "gto.hpp"
#include "hermite1d.hpp"
#include "math.hpp"
#include "nuclear.hpp" // PointCharge
#include "screening.hpp"
#include "tgrid.hpp"

namespace intti {

/// One-electron spin-orbit spatial matrices {W_x, W_y, W_z} (each nao x nao,
/// row-major, unnormalized Cartesian). `charges` carry the operator weight per
/// nucleus (use weight = Z_A). `grid` is the Coulomb t-grid.
template <class Real>
std::array<std::vector<Real>, 3>
spin_orbit_1e(const ShellBasis<Real> &basis,
              const std::vector<PointCharge<Real>> &charges, const TGrid<Real> &grid) {
  const int nao = basis.nao;
  std::array<std::vector<Real>, 3> W;
  for (auto &m : W) m.assign(static_cast<std::size_t>(nao) * nao, Real(0));
  const Real pi = pi_v<Real>();
  const int ns = static_cast<int>(basis.shells.size());
  const int nt = grid.n();
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const int la = sa.l, lb = sb.l, lbe = lb + 1; // ket extended by one (d_j)
      const Real p = sa.alpha + sb.alpha, mu = sa.alpha * sb.alpha / p;
      const int nE = la + lbe + 1; // Hermite length
      const int esz = (la + 1) * (lbe + 1) * nE;
      Real Pd[3];
      std::vector<Real> E(static_cast<std::size_t>(3) * esz);
      for (int d = 0; d < 3; ++d) {
        Pd[d] = (sa.alpha * sa.center[d] + sb.alpha * sb.center[d]) / p;
        const Real ab = sa.center[d] - sb.center[d];
        e_coeffs(la, lbe, p, Pd[d] - sa.center[d], Pd[d] - sb.center[d],
                 exp_(-mu * ab * ab), E.data() + d * esz);
      }
      const int nca = ncart(la), ncb = ncart(lb);
      std::array<std::vector<Real>, 3> acc;
      for (auto &x : acc) x.assign(static_cast<std::size_t>(nca) * ncb, Real(0));
      const int stride = lbe + 1;
      std::vector<Real> B(nE + 1);
      // per-axis 1D factors: plain[d][i*stride+j] (j<=lbe), field[d] (j<=lb)
      std::vector<Real> plain[3], field[3];
      for (int d = 0; d < 3; ++d) {
        plain[d].assign(static_cast<std::size_t>(la + 1) * stride, Real(0));
        field[d].assign(static_cast<std::size_t>(la + 1) * stride, Real(0));
      }
      for (const auto &c : charges)
        for (int it = 0; it < nt; ++it) {
          const Real t = grid.t[it], denom = p + t * t;
          const Real theta = p * t * t / denom, pref = sqrt_(pi / denom);
          const Real wt = grid.w[it] * c.weight;
          for (int d = 0; d < 3; ++d) {
            hermite_b(nE, theta, Pd[d] - c.R[d], B.data()); // B[0..nE]
            const Real *Ed = E.data() + d * esz;
            for (int i = 0; i <= la; ++i)
              for (int j = 0; j <= lbe; ++j) {
                Real sp = 0, sf = 0;
                const Real *e = Ed + (i * (lbe + 1) + j) * nE;
                for (int tau = 0; tau <= i + j; ++tau) {
                  sp += e[tau] * B[tau];
                  sf += e[tau] * B[tau + 1];
                }
                plain[d][i * stride + j] = pref * sp;
                field[d][i * stride + j] = -pref * sf;
              }
          }
          auto P = [&](int d, int i, int j) { return plain[d][i * stride + j]; };
          auto Fld = [&](int d, int i, int j) { return field[d][i * stride + j]; };
          for (int ka = 0; ka < nca; ++ka) {
            int a3[3];
            cart_comp(la, ka, a3[0], a3[1], a3[2]);
            for (int kb = 0; kb < ncb; ++kb) {
              int b3[3];
              cart_comp(lb, kb, b3[0], b3[1], b3[2]);
              Real pl[3], fl[3], kd[3];
              for (int d = 0; d < 3; ++d) {
                pl[d] = P(d, a3[d], b3[d]);
                fl[d] = Fld(d, a3[d], b3[d]);
                Real k = -2 * sb.alpha * P(d, a3[d], b3[d] + 1);
                if (b3[d] >= 1) k += Real(b3[d]) * P(d, a3[d], b3[d] - 1);
                kd[d] = k;
              }
              acc[0][ka * ncb + kb] += wt * pl[0] * (fl[1] * kd[2] - fl[2] * kd[1]);
              acc[1][ka * ncb + kb] += wt * pl[1] * (fl[2] * kd[0] - fl[0] * kd[2]);
              acc[2][ka * ncb + kb] += wt * pl[2] * (fl[0] * kd[1] - fl[1] * kd[0]);
            }
          }
        }
      for (int d = 0; d < 3; ++d)
        for (int ka = 0; ka < nca; ++ka)
          for (int kb = 0; kb < ncb; ++kb)
            W[d][(basis.ao_off[a] + ka) * static_cast<std::size_t>(nao) +
                 basis.ao_off[b] + kb] = acc[d][ka * ncb + kb];
    }
  return W;
}

/// Two-electron spin-orbit "Coulomb-type" matrices {Y_x, Y_y, Y_z} (each nao x
/// nao, row-major, unnormalized Cartesian). The spin-same-orbit two-electron
/// operator (r12 x grad_1)/r12^3 contracted with a density over electron 2:
///
///   Y_k,ul = sum_{n,s} D_ns  (u l | SO_k | n s),
///
/// where mu,lambda are on electron 1 and nu,sigma (contracted with D) on
/// electron 2. The operator reduces to ordinary Coulomb integrals with
/// electron-1's two functions differentiated: integrating
/// (r12)_i/r12^3 = -d_{1,i}(1/r12) by parts in r1, the term with mu d_i d_j
/// lambda is symmetric in (i,j) and dies against eps_kij, leaving
///
///   (u l | SO_k | n s) = eps_kij ( d_i u  d_j l | n s ).
///
/// Each d_i is the MD centre-shift combination (l+1 with -2 alpha, l-1 with the
/// Cartesian power), so this is a linear combination of promoted/demoted plain
/// quartets -- no new kernel. Correctness-first O(N^4) shell loop (like the 1e
/// SO double loop); the fused/screened build is a later optimisation. D is
/// nao x nao row-major; grid the Coulomb t-grid. Antisymmetric: Y_k = -Y_k^T.
namespace detail {

/// Per-quartet two-electron spin-orbit block for four explicit primitive shells:
///   out[(((k*nca+ka)*ncb+kb)*ncc+kc)*ncd+kd] = eps_kij ( d_i sa  d_j sb | sc sd )
/// = (sa sb | SO_k | sc sd), the spin-same-orbit integral with electron-1's pair
/// (sa,sb) differentiated. Built from promoted/demoted plain quartets
/// (eri_block4) via the MD centre-shift combination -- no 1/r12^3 kernel.
template <class Real>
std::vector<Real> spin_orbit_2e_block(const PrimitiveShell<Real> &sa,
                                      const PrimitiveShell<Real> &sb,
                                      const PrimitiveShell<Real> &sc,
                                      const PrimitiveShell<Real> &sd,
                                      const TGrid<Real> &grid) {
  const int la = sa.l, lb = sb.l;
  const int nca = ncart(la), ncb = ncart(lb), ncc = ncart(sc.l), ncd = ncart(sd.l);
  auto shl = [](PrimitiveShell<Real> s, int dl) { s.l += dl; return s; };
  // promoted/demoted electron-1 blocks: blk[a-sign][b-sign], sign 0 = +1 (l+1).
  std::vector<Real> blk[2][2];
  blk[0][0] = eri_block4(shl(sa, 1), shl(sb, 1), sc, sd, grid);
  if (lb >= 1) blk[0][1] = eri_block4(shl(sa, 1), shl(sb, -1), sc, sd, grid);
  if (la >= 1) blk[1][0] = eri_block4(shl(sa, -1), shl(sb, 1), sc, sd, grid);
  if (la >= 1 && lb >= 1) blk[1][1] = eri_block4(shl(sa, -1), shl(sb, -1), sc, sd, grid);
  const int nb_s[2] = {ncart(lb + 1), lb >= 1 ? ncart(lb - 1) : 0};
  // derivative-in-direction terms for one Cartesian component of a shell of
  // momentum l: sgn 0 -> (l+1) block, coeff -2 alpha; sgn 1 -> (l-1), coeff the
  // Cartesian power in that direction. Returns the term count (1 or 2).
  auto terms = [](int l, const int c3[3], Real alpha, int dir, int sgn[2],
                  int ci[2], Real co[2]) -> int {
    int nt = 0;
    int t[3] = {c3[0], c3[1], c3[2]};
    t[dir] += 1;
    sgn[nt] = 0;
    ci[nt] = comp_index(l + 1, t[0], t[1]);
    co[nt] = -2 * alpha;
    ++nt;
    if (c3[dir] >= 1) {
      int u[3] = {c3[0], c3[1], c3[2]};
      u[dir] -= 1;
      sgn[nt] = 1;
      ci[nt] = comp_index(l - 1, u[0], u[1]);
      co[nt] = static_cast<Real>(c3[dir]);
      ++nt;
    }
    return nt;
  };
  std::vector<Real> out(static_cast<std::size_t>(3) * nca * ncb * ncc * ncd, Real(0));
  for (int ka = 0; ka < nca; ++ka) {
    int a3[3];
    cart_comp(la, ka, a3[0], a3[1], a3[2]);
    for (int kb = 0; kb < ncb; ++kb) {
      int b3[3];
      cart_comp(lb, kb, b3[0], b3[1], b3[2]);
      // (d_i sa)(d_j sb | sc sd) for one (kc,kd) ket component
      auto val = [&](int i, int j, int kc, int kd) -> Real {
        int as[2], aci[2], bs[2], bci[2];
        Real aco[2], bco[2];
        const int nat = terms(la, a3, sa.alpha, i, as, aci, aco);
        const int nbt = terms(lb, b3, sb.alpha, j, bs, bci, bco);
        Real v = 0;
        for (int u = 0; u < nat; ++u)
          for (int w = 0; w < nbt; ++w) {
            const std::vector<Real> &B = blk[as[u]][bs[w]];
            const int nbc = nb_s[bs[w]];
            const std::size_t idx =
                (((static_cast<std::size_t>(aci[u]) * nbc + bci[w]) * ncc + kc) * ncd + kd);
            v += aco[u] * bco[w] * B[idx];
          }
        return v;
      };
      for (int kc = 0; kc < ncc; ++kc)
        for (int kd = 0; kd < ncd; ++kd) {
          const std::size_t base =
              ((static_cast<std::size_t>(ka) * ncb + kb) * ncc + kc) * ncd + kd;
          const std::size_t plane = static_cast<std::size_t>(nca) * ncb * ncc * ncd;
          out[0 * plane + base] = val(1, 2, kc, kd) - val(2, 1, kc, kd);
          out[1 * plane + base] = val(2, 0, kc, kd) - val(0, 2, kc, kd);
          out[2 * plane + base] = val(0, 1, kc, kd) - val(1, 0, kc, kd);
        }
    }
  }
  return out;
}

// ---- device (GPU) two-electron spin-orbit -----------------------------------
// Same drop-symmetry batched-quartet-consumer pattern as the 2e gradient: for
// every ordered quartet (mu,lambda|nu,sigma) batch the 4 shifted-bra blocks
// (a+-1, b+-1) x the unshifted ket into one eri_quartets call, then run the SO
// digestion -- eps_kij (d_i mu)(d_j lambda) with the MD centre-shift terms --
// on device, contracting the density and scattering with atomics.

/// d/dir terms for one Cartesian component of a shell of momentum l (device).
/// This is exactly the MD electronic-gradient shift, shared with the geometric
/// derivative builders; see detail::md_grad_terms in erigrad.hpp.
template <class Real>
KOKKOS_INLINE_FUNCTION int so_terms(int l, const int c3[3], Real alpha, int dir,
                                    int sgn[2], int ci[2], Real co[2]) {
  return md_grad_terms(l, c3, alpha, dir, sgn, ci, co);
}

/// Shared batch: the 4 shifted-bra x unshifted-ket quartets per ordered quartet.
template <class Real> struct SO2eBatch {
  Kokkos::View<Real *> out, shAl;
  Kokkos::View<int *> qa, qb, qc, qd, e00, e01, e10, e11, shL, shOff, ksw;
  Kokkos::View<std::int64_t *> boff; ///< 64-bit: see QuartetBatch::out_offset
  int njob{0}, nao{0};
  /// Fraction of the full grid the screened batch actually evaluates (1 when
  /// screening is off). Reported so the saving can be measured rather than
  /// assumed.
  double node_fraction{1.0};
};

/// `tau` (0 = off) screens the ordered quartet loop with a Schwarz bound on the
/// DERIVATIVE-expanded quartets: the digest reads shifted-bra blocks weighted by
/// the MD centre-shift coefficients (-2 alpha on the raised term, the Cartesian
/// power on the lowered one), so the bound carries max(2 alpha, l) per bra shell
/// on top of the usual Q_bra Q_ket. Conservative, hence safe.
template <class Real>
SO2eBatch<Real> build_so2e_batch(const ShellBasis<Real> &basis, const TGrid<Real> &grid,
                                 Real tau = Real(0)) {
  const int ns = static_cast<int>(basis.shells.size());
  SO2eBatch<Real> B;
  B.nao = basis.nao;
  std::vector<int> hL(ns), hOff(ns);
  std::vector<Real> hAl(ns);
  for (int i = 0; i < ns; ++i) {
    hL[i] = basis.shells[i].l;
    hOff[i] = basis.ao_off[i];
    hAl[i] = basis.shells[i].alpha;
  }
  // Every DISTINCT pair the digest can ask for -- the unshifted kets and the
  // four shifted bras -- built once and memoised on (shell, shift) x (shell,
  // shift). Constructing them inside the quartet loop instead would put ~5 ns^4
  // pairs (and their E tables) in the pair table where only ~5 ns^2 are distinct.
  std::vector<ShellPair<Real>> plist;
  std::vector<int> pid(static_cast<std::size_t>(ns) * 3 * ns * 3, -1);
  auto pair_id = [&](int si, int di, int sj, int dj) {
    if (hL[si] + di < 0 || hL[sj] + dj < 0) return -1;
    const std::size_t key =
        ((static_cast<std::size_t>(si) * 3 + (di + 1)) * ns + sj) * 3 + (dj + 1);
    if (pid[key] < 0) {
      PrimitiveShell<Real> a = basis.shells[si], b = basis.shells[sj];
      a.l += di;
      b.l += dj;
      plist.push_back(make_pair(a, b));
      pid[key] = static_cast<int>(plist.size()) - 1;
    }
    return pid[key];
  };
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      pair_id(a, 0, b, 0);
      for (int da = 1; da >= -1; da -= 2)
        for (int db = 1; db >= -1; db -= 2) pair_id(a, da, b, db);
    }
  auto tab = make_pair_table(plist);
  std::vector<Real> Q;
  if (tau > Real(0)) Q = schwarz(tab, plist, grid);
  auto scale = [&](int s) {
    const Real two_al = 2 * hAl[s];
    return two_al > Real(hL[s]) ? two_al : Real(hL[s]);
  };
  // Permutational dedup. The quartets here are ordinary ERIs over a SHIFTED bra
  // pair and an unshifted ket, so the two intra-pair swaps hold: (ab|cd) =
  // (ba|cd) = (ab|dc). The bra-ket swap buys nothing -- a shifted bra can never
  // coincide with an unshifted ket -- so the fold is 4-fold, not 8. Each block is
  // stored once in its canonical orientation (the smaller pair id) and the digest
  // reads it back through the permutation; the entry arrays pack that as
  // 2*index + braswap, with -1 for an absent shift.
  std::vector<std::pair<int, int>> quartets;
  std::unordered_map<long long, int> seen;
  const long long npair_all = static_cast<long long>(plist.size());
  auto canon = [&](int p1, int p2, int &swap) {
    swap = (p2 >= 0 && p2 < p1) ? 1 : 0;
    return swap ? p2 : p1;
  };
  std::vector<int> qa, qb, qc, qd, e00, e01, e10, e11, ksw;
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b)
      for (int c = 0; c < ns; ++c)
        for (int d = 0; d < ns; ++d) {
          const int la = hL[a], lb = hL[b];
          int kswap = 0;
          const int ket = canon(pair_id(c, 0, d, 0), pair_id(d, 0, c, 0), kswap);
          if (tau > Real(0)) {
            Real qbra = 0;
            for (int da = 1; da >= -1; da -= 2)
              for (int db = 1; db >= -1; db -= 2) {
                const int p = pair_id(a, da, b, db);
                if (p >= 0 && Q[p] > qbra) qbra = Q[p];
              }
            if (scale(a) * scale(b) * qbra * Q[ket] < tau) continue;
          }
          qa.push_back(a); qb.push_back(b); qc.push_back(c); qd.push_back(d);
          ksw.push_back(kswap);
          auto emit = [&](int da, int db) {
            int bswap = 0;
            const int bp = canon(pair_id(a, da, b, db), pair_id(b, db, a, da), bswap);
            const long long key = static_cast<long long>(bp) * npair_all + ket;
            auto it = seen.find(key);
            int e;
            if (it == seen.end()) {
              e = static_cast<int>(quartets.size());
              quartets.push_back({bp, ket});
              seen.emplace(key, e);
            } else {
              e = it->second;
            }
            return 2 * e + bswap;
          };
          e00.push_back(emit(1, 1));
          e01.push_back(lb >= 1 ? emit(1, -1) : -1);
          e10.push_back(la >= 1 ? emit(-1, 1) : -1);
          e11.push_back((la >= 1 && lb >= 1) ? emit(-1, -1) : -1);
        }
  auto batch = make_batch(tab, quartets);
  // t-resolved node truncation (screening.hpp). The 2e spin-orbit quartets are
  // ordinary Coulomb ERIs over shifted pairs, so the per-node bound applies
  // unchanged -- what does not carry over is the tolerance: the digest weights
  // each block by the MD centre-shift coefficients, so a quartet truncated at
  // tau reaches W multiplied by up to max(2 alpha, l) per bra shell. That factor
  // is handed to the screener per quartet (the max over the jobs that read it),
  // exactly the factor the Schwarz pass above already uses.
  if (tau > Real(0)) {
    std::vector<Real> hamp(quartets.size(), Real(1));
    for (std::size_t j = 0; j < qa.size(); ++j) {
      const Real f = scale(qa[j]) * scale(qb[j]);
      const int ee[4] = {e00[j], e01[j], e10[j], e11[j]};
      for (int k = 0; k < 4; ++k)
        if (ee[k] >= 0) {
          const int e = ee[k] / 2;
          if (f > hamp[e]) hamp[e] = f;
        }
    }
    auto damp = to_device(hamp, "intti::so2e::amp");
    t_screen_batch(batch, plist, grid, tau, 16, damp);
    auto hkeep = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, batch.keep);
    std::int64_t kept = 0;
    for (int q = 0; q < batch.nq; ++q) kept += hkeep(q);
    B.node_fraction = batch.nq ? static_cast<double>(kept) /
                                     (static_cast<double>(batch.nq) * grid.n())
                               : 1.0;
  }
  QuartetWorkspace<Real> ws;
  B.out = Kokkos::View<Real *>("intti::so2e::out", batch.nout_total);
  eri_quartets(tab, batch, grid, B.out, ws);
  B.boff = batch.out_offset;
  B.shL = to_device(hL, "intti::so2e::L");
  B.shOff = to_device(hOff, "intti::so2e::off");
  B.shAl = to_device(hAl, "intti::so2e::al");
  B.qa = to_device(qa, "intti::so2e::qa");
  B.qb = to_device(qb, "intti::so2e::qb");
  B.qc = to_device(qc, "intti::so2e::qc");
  B.qd = to_device(qd, "intti::so2e::qd");
  B.e00 = to_device(e00, "intti::so2e::e00");
  B.e01 = to_device(e01, "intti::so2e::e01");
  B.e10 = to_device(e10, "intti::so2e::e10");
  B.e11 = to_device(e11, "intti::so2e::e11");
  B.ksw = to_device(ksw, "intti::so2e::ksw");
  B.njob = static_cast<int>(qa.size());
  return B;
}

template <class Real>
std::array<std::vector<Real>, 3>
spin_orbit_2e_coulomb_dev(const ShellBasis<Real> &basis, const Real *D,
                          const TGrid<Real> &grid, Real tau) {
  const int nao = basis.nao;
  auto B = build_so2e_batch(basis, grid, tau);
  auto Dd = to_device(D, static_cast<std::size_t>(nao) * nao, "intti::so2e::D");
  Kokkos::View<Real *> Yd("intti::so2e::Y", static_cast<std::size_t>(3) * nao * nao);
  const std::size_t plane = static_cast<std::size_t>(nao) * nao;
  auto out = B.out, shAl = B.shAl;
  auto shL = B.shL, shOff = B.shOff, boff = B.boff;
  auto qa = B.qa, qb = B.qb, qc = B.qc, qd = B.qd, e00 = B.e00, e01 = B.e01, e10 = B.e10, e11 = B.e11;
  auto ksw = B.ksw;
  Kokkos::parallel_for(
      "intti::so2e::coulomb", Kokkos::RangePolicy<>(0, B.njob), KOKKOS_LAMBDA(int j) {
        const int a = qa(j), b = qb(j), c = qc(j), d = qd(j);
        const int la = shL(a), lb = shL(b), lc = shL(c), ld = shL(d);
        const int nc = ncart(lc), nd = ncart(ld);
        const Real ala = shAl(a), alb = shAl(b);
        const int oa = shOff(a), ob = shOff(b), oc = shOff(c), od = shOff(d);
        // entry arrays pack 2*quartet + braswap; -1 marks an absent shift
        const int ent[2][2] = {{e00(j), e01(j)}, {e10(j), e11(j)}};
        std::int64_t base[2][2];
        int bsw[2][2];
        for (int u = 0; u < 2; ++u)
          for (int w = 0; w < 2; ++w) {
            base[u][w] = ent[u][w] >= 0 ? boff(ent[u][w] >> 1) : 0;
            bsw[u][w] = ent[u][w] >= 0 ? (ent[u][w] & 1) : 0;
          }
        const int kswap = ksw(j);
        for (int ka = 0; ka < ncart(la); ++ka) {
          int a3[3];
          cart_comp(la, ka, a3[0], a3[1], a3[2]);
          for (int kb = 0; kb < ncart(lb); ++kb) {
            int b3[3];
            cart_comp(lb, kb, b3[0], b3[1], b3[2]);
            Real yx = 0, yy = 0, yz = 0;
            auto val = [&](int i, int jj, int kc, int kd) -> Real {
              int as[2], aci[2], bs[2], bci[2];
              Real aco[2], bco[2];
              const int na = so_terms(la, a3, ala, i, as, aci, aco);
              const int nb = so_terms(lb, b3, alb, jj, bs, bci, bco);
              Real v = 0;
              for (int u = 0; u < na; ++u)
                for (int w = 0; w < nb; ++w) {
                  const int lap = as[u] == 0 ? la + 1 : la - 1;
                  const int lbp = bs[w] == 0 ? lb + 1 : lb - 1;
                  const int nac = ncart(lap), nbc = ncart(lbp);
                  // read the canonical block through the stored permutation
                  const std::size_t ibra = bsw[as[u]][bs[w]]
                                               ? static_cast<std::size_t>(bci[w]) * nac + aci[u]
                                               : static_cast<std::size_t>(aci[u]) * nbc + bci[w];
                  const std::size_t iket = kswap ? static_cast<std::size_t>(kd) * nc + kc
                                                 : static_cast<std::size_t>(kc) * nd + kd;
                  const std::size_t idx = ibra * (static_cast<std::size_t>(nc) * nd) + iket;
                  v += aco[u] * bco[w] * out(base[as[u]][bs[w]] + idx);
                }
              return v;
            };
            for (int kc = 0; kc < nc; ++kc)
              for (int kd = 0; kd < nd; ++kd) {
                const Real Dcd = Dd(static_cast<std::size_t>(oc + kc) * nao + od + kd);
                if (Dcd == Real(0)) continue;
                yx += Dcd * (val(1, 2, kc, kd) - val(2, 1, kc, kd));
                yy += Dcd * (val(2, 0, kc, kd) - val(0, 2, kc, kd));
                yz += Dcd * (val(0, 1, kc, kd) - val(1, 0, kc, kd));
              }
            const std::size_t mu = static_cast<std::size_t>(oa + ka) * nao + ob + kb;
            Kokkos::atomic_add(&Yd(0 * plane + mu), yx);
            Kokkos::atomic_add(&Yd(1 * plane + mu), yy);
            Kokkos::atomic_add(&Yd(2 * plane + mu), yz);
          }
        }
      });
  auto flat = to_host(Yd);
  std::array<std::vector<Real>, 3> Y;
  for (int k = 0; k < 3; ++k)
    Y[k].assign(flat.begin() + k * plane, flat.begin() + (k + 1) * plane);
  return Y;
}

template <class Real>
std::array<std::vector<Real>, 3>
spin_orbit_2e_exchange_dev(const ShellBasis<Real> &basis, const Real *D,
                           const TGrid<Real> &grid, Real tau) {
  const int nao = basis.nao;
  auto B = build_so2e_batch(basis, grid, tau);
  auto Dd = to_device(D, static_cast<std::size_t>(nao) * nao, "intti::so2ex::D");
  Kokkos::View<Real *> Ked("intti::so2ex::Ke", static_cast<std::size_t>(3) * nao * nao);
  const std::size_t plane = static_cast<std::size_t>(nao) * nao;
  auto out = B.out, shAl = B.shAl;
  auto shL = B.shL, shOff = B.shOff, boff = B.boff;
  auto qa = B.qa, qb = B.qb, qc = B.qc, qd = B.qd, e00 = B.e00, e01 = B.e01, e10 = B.e10, e11 = B.e11;
  auto ksw = B.ksw;
  Kokkos::parallel_for(
      "intti::so2e::exchange", Kokkos::RangePolicy<>(0, B.njob), KOKKOS_LAMBDA(int j) {
        const int a = qa(j), b = qb(j), c = qc(j), d = qd(j);
        const int la = shL(a), lb = shL(b), lc = shL(c), ld = shL(d);
        const int nc = ncart(lc), nd = ncart(ld);
        const Real ala = shAl(a), alb = shAl(b);
        const int oa = shOff(a), ob = shOff(b), oc = shOff(c), od = shOff(d);
        // entry arrays pack 2*quartet + braswap; -1 marks an absent shift
        const int ent[2][2] = {{e00(j), e01(j)}, {e10(j), e11(j)}};
        std::int64_t base[2][2];
        int bsw[2][2];
        for (int u = 0; u < 2; ++u)
          for (int w = 0; w < 2; ++w) {
            base[u][w] = ent[u][w] >= 0 ? boff(ent[u][w] >> 1) : 0;
            bsw[u][w] = ent[u][w] >= 0 ? (ent[u][w] & 1) : 0;
          }
        const int kswap = ksw(j);
        for (int ka = 0; ka < ncart(la); ++ka) {
          int a3[3];
          cart_comp(la, ka, a3[0], a3[1], a3[2]);
          for (int kb = 0; kb < ncart(lb); ++kb) {
            int b3[3];
            cart_comp(lb, kb, b3[0], b3[1], b3[2]);
            auto val = [&](int i, int jj, int kc, int kd) -> Real {
              int as[2], aci[2], bs[2], bci[2];
              Real aco[2], bco[2];
              const int na = so_terms(la, a3, ala, i, as, aci, aco);
              const int nb = so_terms(lb, b3, alb, jj, bs, bci, bco);
              Real v = 0;
              for (int u = 0; u < na; ++u)
                for (int w = 0; w < nb; ++w) {
                  const int lap = as[u] == 0 ? la + 1 : la - 1;
                  const int lbp = bs[w] == 0 ? lb + 1 : lb - 1;
                  const int nac = ncart(lap), nbc = ncart(lbp);
                  // read the canonical block through the stored permutation
                  const std::size_t ibra = bsw[as[u]][bs[w]]
                                               ? static_cast<std::size_t>(bci[w]) * nac + aci[u]
                                               : static_cast<std::size_t>(aci[u]) * nbc + bci[w];
                  const std::size_t iket = kswap ? static_cast<std::size_t>(kd) * nc + kc
                                                 : static_cast<std::size_t>(kc) * nd + kd;
                  const std::size_t idx = ibra * (static_cast<std::size_t>(nc) * nd) + iket;
                  v += aco[u] * bco[w] * out(base[as[u]][bs[w]] + idx);
                }
              return v;
            };
            for (int kc = 0; kc < nc; ++kc) {
              const Real Dbc = Dd(static_cast<std::size_t>(ob + kb) * nao + oc + kc);
              if (Dbc == Real(0)) continue;
              for (int kd = 0; kd < nd; ++kd) {
                const Real vx = val(1, 2, kc, kd) - val(2, 1, kc, kd);
                const Real vy = val(2, 0, kc, kd) - val(0, 2, kc, kd);
                const Real vz = val(0, 1, kc, kd) - val(1, 0, kc, kd);
                const std::size_t ms = static_cast<std::size_t>(oa + ka) * nao + od + kd;
                Kokkos::atomic_add(&Ked(0 * plane + ms), Dbc * vx);
                Kokkos::atomic_add(&Ked(1 * plane + ms), Dbc * vy);
                Kokkos::atomic_add(&Ked(2 * plane + ms), Dbc * vz);
              }
            }
          }
        }
      });
  auto flat = to_host(Ked);
  std::array<std::vector<Real>, 3> Ke;
  for (int k = 0; k < 3; ++k)
    Ke[k].assign(flat.begin() + k * plane, flat.begin() + (k + 1) * plane);
  return Ke;
}

} // namespace detail

template <class Real>
std::array<std::vector<Real>, 3>
spin_orbit_2e_coulomb(const ShellBasis<Real> &basis, const Real *D,
                      const TGrid<Real> &grid, Real tau = Real(0)) {
  if constexpr (kokkos_scalar_v<Real>) {
    bool ok = true;
    for (const auto &s : basis.shells)
      if (s.l >= LMAX) ok = false; // bra shells are promoted to l+1
    if (ok) return detail::spin_orbit_2e_coulomb_dev(basis, D, grid, tau);
  }
  const int nao = basis.nao;
  std::array<std::vector<Real>, 3> Y;
  for (auto &m : Y) m.assign(static_cast<std::size_t>(nao) * nao, Real(0));
  const int ns = static_cast<int>(basis.shells.size());
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const int nca = ncart(sa.l), ncb = ncart(sb.l);
      for (int c = 0; c < ns; ++c)
        for (int d = 0; d < ns; ++d) {
          const auto &sc = basis.shells[c], &sd = basis.shells[d];
          const int ncc = ncart(sc.l), ncd = ncart(sd.l);
          const auto blk = detail::spin_orbit_2e_block(sa, sb, sc, sd, grid);
          const std::size_t plane = static_cast<std::size_t>(nca) * ncb * ncc * ncd;
          for (int ka = 0; ka < nca; ++ka)
            for (int kb = 0; kb < ncb; ++kb) {
              const std::size_t mu = basis.ao_off[a] + ka, lam = basis.ao_off[b] + kb;
              for (int kc = 0; kc < ncc; ++kc)
                for (int kd = 0; kd < ncd; ++kd) {
                  const Real Dcd = D[static_cast<std::size_t>(basis.ao_off[c] + kc) * nao +
                                     basis.ao_off[d] + kd];
                  if (Dcd == Real(0)) continue;
                  const std::size_t base =
                      ((static_cast<std::size_t>(ka) * ncb + kb) * ncc + kc) * ncd + kd;
                  for (int k = 0; k < 3; ++k)
                    Y[k][mu * nao + lam] += Dcd * blk[k * plane + base];
                }
            }
        }
    }
  return Y;
}

/// Two-electron spin-orbit "exchange-type" matrices {Ke_x, Ke_y, Ke_z} (each
/// nao x nao). Ke_k,us = sum_{l,n} D_ln (u l | SO_k | n s): the same
/// spin-same-orbit integral as spin_orbit_2e_coulomb, but with the density
/// bridging one electron-1 index (lambda) and one electron-2 index (nu), leaving
/// mu (electron 1) and sigma (electron 2) as the matrix indices. Together with
/// the Coulomb-type matrix and the 1e SO matrix this is the building block for a
/// spin-orbit mean-field (SOMF/AMFI) effective one-electron operator; the
/// mean-field linear combination and its coefficients are a downstream modelling
/// choice (Breit-Pauli spin-same/other-orbit) and are not fixed here. Same
/// correctness-first O(N^4) shell loop; D is nao x nao row-major.
template <class Real>
std::array<std::vector<Real>, 3>
spin_orbit_2e_exchange(const ShellBasis<Real> &basis, const Real *D,
                       const TGrid<Real> &grid, Real tau = Real(0)) {
  if constexpr (kokkos_scalar_v<Real>) {
    bool ok = true;
    for (const auto &s : basis.shells)
      if (s.l >= LMAX) ok = false; // bra shells are promoted to l+1
    if (ok) return detail::spin_orbit_2e_exchange_dev(basis, D, grid, tau);
  }
  const int nao = basis.nao;
  std::array<std::vector<Real>, 3> Ke;
  for (auto &m : Ke) m.assign(static_cast<std::size_t>(nao) * nao, Real(0));
  const int ns = static_cast<int>(basis.shells.size());
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const int nca = ncart(sa.l), ncb = ncart(sb.l);
      for (int c = 0; c < ns; ++c)
        for (int d = 0; d < ns; ++d) {
          const auto &sc = basis.shells[c], &sd = basis.shells[d];
          const int ncc = ncart(sc.l), ncd = ncart(sd.l);
          const auto blk = detail::spin_orbit_2e_block(sa, sb, sc, sd, grid);
          const std::size_t plane = static_cast<std::size_t>(nca) * ncb * ncc * ncd;
          for (int ka = 0; ka < nca; ++ka)
            for (int kb = 0; kb < ncb; ++kb)
              for (int kc = 0; kc < ncc; ++kc) {
                const Real Dln = D[static_cast<std::size_t>(basis.ao_off[b] + kb) * nao +
                                   basis.ao_off[c] + kc];
                if (Dln == Real(0)) continue;
                for (int kd = 0; kd < ncd; ++kd) {
                  const std::size_t mu = basis.ao_off[a] + ka, sig = basis.ao_off[d] + kd;
                  const std::size_t base =
                      ((static_cast<std::size_t>(ka) * ncb + kb) * ncc + kc) * ncd + kd;
                  for (int k = 0; k < 3; ++k)
                    Ke[k][mu * nao + sig] += Dln * blk[k * plane + base];
                }
              }
        }
    }
  return Ke;
}

} // namespace intti
