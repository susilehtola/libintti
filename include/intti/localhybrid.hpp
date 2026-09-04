// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Local and locally range-separated hybrids: the exact-exchange energy
// density on a real-space grid -- the library's raison d'etre.
//
// For occupied orbitals psi_i = sum_mu C_{mu i} phi_mu, the conventional HF
// exchange energy density is
//   eps_x(r) = -1/2 sum_ij psi_i(r) psi_j(r) K_ij(r),
//   K_ij(r)  = integral psi_i(r') psi_j(r') / |r - r'| dr'
//            = (C^T V^r C)_ij,   V^r_{mu nu} = <phi_mu|1/|r-r'||phi_nu> (r'),
// where V^r is the Coulomb-potential collocation matrix of nuclear.hpp. Its
// grid integral is the total exact-exchange energy,
//   integral eps_x(r) dr = -1/2 sum_ij (ij|ij) = -1/2 Tr(D K),  D = sum_i C_i C_i^T,
// which is the validation oracle.
//
// Locally range-separated hybrids: pass a range-separated t grid
// (make_tgrid(erf_rs(omega))) so V^r uses erf(omega r)/r instead of 1/r; a
// position-dependent omega(r) is then just a per-point choice of truncation
// of the shared t nodes -- the t-quadrature does this natively.
//
// Matrix/array-level API: orbitals in, energy density (and total) out.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "fock.hpp"
#include "gto.hpp"
#include "math.hpp"
#include "nuclear.hpp"
#include "tgrid.hpp"

namespace intti {

/// AO values phi_mu(r_g) at grid points: row-major [npoint * nao].
template <class Real>
std::vector<Real> ao_values(const ShellBasis<Real> &basis,
                            const std::vector<std::array<Real, 3>> &points) {
  const int nao = basis.nao;
  const int np = static_cast<int>(points.size());
  std::vector<Real> out(static_cast<std::size_t>(np) * nao, Real(0));
  const int ns = static_cast<int>(basis.shells.size());
  for (int g = 0; g < np; ++g) {
    const Real *r = points[g].data();
    for (int s = 0; s < ns; ++s) {
      const auto &sh = basis.shells[s];
      Real d[3], r2 = 0;
      for (int k = 0; k < 3; ++k) {
        d[k] = r[k] - sh.center[k];
        r2 += d[k] * d[k];
      }
      const Real rad = exp_(-sh.alpha * r2);
      for (int c = 0; c < ncart(sh.l); ++c) {
        int l3[3];
        cart_comp(sh.l, c, l3[0], l3[1], l3[2]);
        Real v = rad;
        for (int k = 0; k < 3; ++k)
          for (int e = 0; e < l3[k]; ++e) v *= d[k];
        out[static_cast<std::size_t>(g) * nao + basis.ao_off[s] + c] = v;
      }
    }
  }
  return out;
}

/// Result of a local-exchange evaluation.
template <class Real> struct LocalExchange {
  std::vector<Real> eps; ///< energy density eps_x(r_g) per grid point
  Real energy{0};        ///< sum_g w_g eps_x(r_g) (if weights supplied)
};

/// Exact-exchange energy density on a grid for occupied orbitals C
/// (nao x nocc, row-major C[mu*nocc+i]). Pass the Coulomb t grid for full-
/// range HF exchange, or an erf/erfc grid for (locally) range-separated
/// exchange. If weights is non-empty, energy = sum_g w_g eps_x(r_g).
template <class Real>
LocalExchange<Real> local_exchange(const ShellBasis<Real> &basis, const Real *C,
                                   int nocc,
                                   const std::vector<std::array<Real, 3>> &points,
                                   const std::vector<Real> &weights,
                                   const TGrid<Real> &grid) {
  const int nao = basis.nao;
  const int np = static_cast<int>(points.size());
  auto phi = ao_values(basis, points); // [np*nao]
  LocalExchange<Real> out;
  out.eps.assign(np, Real(0));
  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
  std::vector<Real> V(n2), psi(nocc), u(nao);
  for (int g = 0; g < np; ++g) {
    // potential collocation matrix V^g (unit charge at r_g)
    std::fill(V.begin(), V.end(), Real(0));
    std::vector<PointCharge<Real>> one{{Real(1), {points[g][0], points[g][1], points[g][2]}}};
    detail::attraction_accumulate(basis, one, grid, Real(0), V.data());
    // psi_i(r_g) = sum_mu phi_mu C_mu i
    const Real *phg = &phi[static_cast<std::size_t>(g) * nao];
    for (int i = 0; i < nocc; ++i) {
      Real s = 0;
      for (int mu = 0; mu < nao; ++mu) s += phg[mu] * C[mu * nocc + i];
      psi[i] = s;
    }
    // eps = -1/2 sum_ij psi_i psi_j (C^T V C)_ij collapses (sum over i,j first):
    //     = -1/2 sum_{mu nu} V_{mu nu} u_mu u_nu,  u_mu = sum_i C_{mu i} psi_i.
    // This drops the occupied index -- O(nao^2 nocc)+O(nocc^2 nao) -> O(nao^2).
    for (int mu = 0; mu < nao; ++mu) {
      Real s = 0;
      for (int i = 0; i < nocc; ++i) s += C[mu * nocc + i] * psi[i];
      u[mu] = s;
    }
    Real e = 0;
    for (int mu = 0; mu < nao; ++mu) {
      Real row = 0;
      for (int nu = 0; nu < nao; ++nu) row += V[mu * nao + nu] * u[nu];
      e += u[mu] * row;
    }
    out.eps[g] = Real(-0.5) * e;
  }
  if (!weights.empty()) {
    Real E = 0;
    for (int g = 0; g < np; ++g) E += weights[g] * out.eps[g];
    out.energy = E;
  }
  return out;
}

} // namespace intti
