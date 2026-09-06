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

#include <array>
#include <cstddef>
#include <vector>

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

} // namespace intti
