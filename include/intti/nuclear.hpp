// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Nuclear attraction and Coulomb-potential collocation matrices via the
// t-quadrature. At fixed t the interaction of an AO pair with a point at C is
// the overlap of the pair's Hermite expansion with a Gaussian exp(-t^2(r-C)^2):
//
//   <a|1/|r-C||b> = (2/sqrt(pi)) int_0^inf dt  prod_d G_d(t),
//   G_d(t) = sqrt(pi/(p+t^2)) sum_tau E^d_tau B_tau(theta_t, P_d - C_d),
//   theta_t = p t^2/(p+t^2),
//
// with E the McMurchie-Davidson coefficients (e_coeffs) and B the
// Gaussian-derivative array (hermite_b). This is the critical primitive for
// local hybrids (the local exchange energy density is dominated by these), so
// it carries distance/Schwarz screening. Matrix-level API: whole nao x nao
// matrices, never per-shell blocks.

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "fock.hpp"
#include "gto.hpp"
#include "hermite1d.hpp"
#include "math.hpp"
#include "tgrid.hpp"

namespace intti {

/// A weighted point (a nuclear charge Z at R, or a unit collocation point).
template <class Real> struct PointCharge {
  Real weight; ///< contribution weight (e.g. -Z for nuclear attraction)
  Real R[3];
};

namespace detail {

/// Accumulate sum_c weight_c <a|1/|r-R_c||b> into V (nao x nao, row-major).
/// tau > 0 enables a per-(pair,centre) Schwarz/decay screen.
template <class Real>
void attraction_accumulate(const ShellBasis<Real> &basis,
                           const std::vector<PointCharge<Real>> &centers,
                           const TGrid<Real> &grid, Real tau, Real *V) {
  const int nao = basis.nao;
  const int nt = grid.n();
  const Real pi = pi_v<Real>();
  const int ns = static_cast<int>(basis.shells.size());
  const bool screen = tau > Real(0);
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const int la = sa.l, lb = sb.l, esz = (la + 1) * (lb + 1) * (la + lb + 1);
      const Real p = sa.alpha + sb.alpha;
      const Real mu = sa.alpha * sb.alpha / p;
      Real Pd[3], expmu = 1;
      std::vector<Real> E(static_cast<std::size_t>(3) * esz);
      for (int d = 0; d < 3; ++d) {
        Pd[d] = (sa.alpha * sa.center[d] + sb.alpha * sb.center[d]) / p;
        const Real ab = sa.center[d] - sb.center[d];
        const Real Kd = exp_(-mu * ab * ab);
        expmu *= Kd;
        e_coeffs(la, lb, p, Pd[d] - sa.center[d], Pd[d] - sb.center[d], Kd,
                 E.data() + d * esz);
      }
      const Real pair_bound = (2 * pi / p) * expmu; // (2 pi/p) exp(-mu R_AB^2)
      // per-component accumulation buffer
      const int nca = ncart(la), ncb = ncart(lb);
      std::vector<Real> acc(static_cast<std::size_t>(nca) * ncb, Real(0));
      std::vector<Real> Bx(la + lb + 1), By(la + lb + 1), Bz(la + lb + 1);
      for (const auto &c : centers) {
        if (screen) {
          Real d2 = 0;
          for (int d = 0; d < 3; ++d) d2 += (Pd[d] - c.R[d]) * (Pd[d] - c.R[d]);
          using std::sqrt;
          const Real fb = d2 * p > Real(1) ? Real(0.5) * sqrt(pi / (p * d2)) : Real(1);
          if (std::abs(c.weight) * pair_bound * fb < tau) continue;
        }
        for (int it = 0; it < nt; ++it) {
          const Real t = grid.t[it];
          const Real denom = p + t * t;
          const Real theta = p * t * t / denom;
          const Real pref = sqrt_(pi / denom);
          hermite_b(la + lb, theta, Pd[0] - c.R[0], Bx.data());
          hermite_b(la + lb, theta, Pd[1] - c.R[1], By.data());
          hermite_b(la + lb, theta, Pd[2] - c.R[2], Bz.data());
          const Real wt = grid.w[it] * c.weight;
          for (int ka = 0; ka < nca; ++ka) {
            int a3[3];
            cart_comp(la, ka, a3[0], a3[1], a3[2]);
            for (int kb = 0; kb < ncb; ++kb) {
              int b3[3];
              cart_comp(lb, kb, b3[0], b3[1], b3[2]);
              auto gd = [&](int d, const Real *B) {
                const Real *Ed = E.data() + d * esz +
                                 (a3[d] * (lb + 1) + b3[d]) * (la + lb + 1);
                Real s = 0;
                for (int tau_ = 0; tau_ <= a3[d] + b3[d]; ++tau_)
                  s += Ed[tau_] * B[tau_];
                return pref * s;
              };
              acc[ka * ncb + kb] += wt * gd(0, Bx.data()) * gd(1, By.data()) *
                                    gd(2, Bz.data());
            }
          }
        }
      }
      for (int ka = 0; ka < nca; ++ka)
        for (int kb = 0; kb < ncb; ++kb)
          V[(basis.ao_off[a] + ka) * nao + basis.ao_off[b] + kb] +=
              acc[ka * ncb + kb];
    }
}

} // namespace detail

/// Nuclear attraction matrix V_ab = -sum_C Z_C <a|1/|r-R_C||b> (matches
/// PySCF int1e_nuc). charges carry Z (positive); tau enables screening.
template <class Real>
std::vector<Real> nuclear_matrix(const ShellBasis<Real> &basis,
                                 const std::vector<PointCharge<Real>> &charges,
                                 const TGrid<Real> &grid, Real tau = Real(0)) {
  std::vector<Real> V(static_cast<std::size_t>(basis.nao) * basis.nao, Real(0));
  // charges here already carry weight = -Z (caller sets it); provide a helper
  detail::attraction_accumulate(basis, charges, grid, tau, V.data());
  return V;
}

/// Convenience: build charges with weight = -Z from (Z, R) nuclei.
template <class Real>
std::vector<PointCharge<Real>>
nuclei_as_charges(const std::vector<Real> &Z,
                  const std::vector<std::array<Real, 3>> &R) {
  std::vector<PointCharge<Real>> c(Z.size());
  for (std::size_t i = 0; i < Z.size(); ++i)
    c[i] = {-Z[i], {R[i][0], R[i][1], R[i][2]}};
  return c;
}

/// Coulomb-potential collocation: for each point r_g, the matrix
/// <mu|1/|r-r_g||nu>. This is the primitive local hybrids consume. Returned
/// as one nao x nao matrix per point.
template <class Real>
std::vector<std::vector<Real>>
potential_matrices(const ShellBasis<Real> &basis,
                   const std::vector<std::array<Real, 3>> &points,
                   const TGrid<Real> &grid, Real tau = Real(0)) {
  std::vector<std::vector<Real>> out;
  out.reserve(points.size());
  const std::size_t n2 = static_cast<std::size_t>(basis.nao) * basis.nao;
  for (const auto &pt : points) {
    std::vector<Real> V(n2, Real(0));
    std::vector<PointCharge<Real>> one{{Real(1), {pt[0], pt[1], pt[2]}}};
    detail::attraction_accumulate(basis, one, grid, tau, V.data());
    out.push_back(std::move(V));
  }
  return out;
}

} // namespace intti
