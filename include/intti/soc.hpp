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
#include <vector>

#include "erigrad.hpp" // detail::eri_block4, detail::comp_index
#include "fock.hpp"
#include "gto.hpp"
#include "hermite1d.hpp"
#include "math.hpp"
#include "nuclear.hpp" // PointCharge
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

} // namespace detail

template <class Real>
std::array<std::vector<Real>, 3>
spin_orbit_2e_coulomb(const ShellBasis<Real> &basis, const Real *D,
                      const TGrid<Real> &grid) {
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
                       const TGrid<Real> &grid) {
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
