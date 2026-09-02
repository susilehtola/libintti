// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Three-electron Coulomb integrals by numerical integration -- the same
// t-quadrature that resolves the two-electron 1/r, applied twice.
//
//   G_abcdef = <a(1)b(2)c(3)| r12^{-1} r13^{-1} |d(1)e(2)f(3)>.
//
// Both operators are replaced by their Gaussian integral transform
//   1/r12 = (2/sqrt(pi)) int exp(-t^2 r12^2) dt,
//   1/r13 = (2/sqrt(pi)) int exp(-s^2 r13^2) ds,
// so the integral becomes a TWO-dimensional (t,s) quadrature whose integrand
// factorises over the Cartesian directions and is analytic at each node
// (Mehine, Losilla & Sundholm, Mol. Phys. 111, 2536 (2013) -- the same group's
// generalisation of the two-electron scheme libintti already uses; both the
// scaled/Mobius grid and the delta-function tail correction carry over per
// dimension). Electron 1 (density P = a d) is shared by both operators;
// electron 2 (Q = b e) couples through t, electron 3 (S = c f) through s.
//
// This header provides the s-type (l=0) primitive as the first, analytically
// checkable case (one centre: G_aaaaaa = 4 zeta_a / 3). Higher angular momentum
// follows from the same recursion machinery as the two-electron path.

#include "gto.hpp"
#include "math.hpp"
#include "tgrid.hpp"

namespace intti {

namespace detail {
template <class Real> Real pow34(Real x) { // x^{3/4} = x^{1/2} x^{1/4}
  return sqrt_(x) * sqrt_(sqrt_(x));
}
} // namespace detail

/// Three-electron Coulomb integral G_abcdef over six s-type primitive Gaussians
/// (normalised), via the 2D (t,s) quadrature. `grid` is an ordinary Coulomb
/// t-grid (make_tgrid(coulomb())), used for both auxiliary dimensions.
template <class Real>
Real three_electron_coulomb(const PrimitiveShell<Real> &a, const PrimitiveShell<Real> &b,
                            const PrimitiveShell<Real> &c, const PrimitiveShell<Real> &d,
                            const PrimitiveShell<Real> &e, const PrimitiveShell<Real> &f,
                            const TGrid<Real> &grid) {
  // Gaussian product of an electron's two functions -> density (alpha, R, K)
  auto pairdens = [](const PrimitiveShell<Real> &x, const PrimitiveShell<Real> &y,
                     Real &alpha, Real R[3], Real &K) {
    alpha = x.alpha + y.alpha;
    Real ab2 = 0;
    for (int i = 0; i < 3; ++i) {
      R[i] = (x.alpha * x.center[i] + y.alpha * y.center[i]) / alpha;
      const Real dd = x.center[i] - y.center[i];
      ab2 += dd * dd;
    }
    K = exp_(-x.alpha * y.alpha / alpha * ab2);
  };
  Real aP, aQ, aS, RP[3], RQ[3], RS[3], Kad, Kbe, Kcf;
  pairdens(a, d, aP, RP, Kad); // electron 1
  pairdens(b, e, aQ, RQ, Kbe); // electron 2
  pairdens(c, f, aS, RS, Kcf); // electron 3
  Real RPQ2 = 0, RPS2 = 0;
  for (int i = 0; i < 3; ++i) {
    RPQ2 += (RP[i] - RQ[i]) * (RP[i] - RQ[i]);
    RPS2 += (RP[i] - RS[i]) * (RP[i] - RS[i]);
  }
  const Real pi = pi_v<Real>();
  const Real pi92 = pi * pi * pi * pi * sqrt_(pi); // pi^{9/2}
  const int nt = grid.n();
  Real acc = 0;
  for (int it = 0; it < nt; ++it) {
    const Real t2 = grid.t[it] * grid.t[it];
    const Real LQ = t2 * aQ / (t2 + aQ); // Lambda(t, alpha_Q)
    const Real wt = grid.w[it];
    for (int is = 0; is < nt; ++is) {
      const Real s2 = grid.t[is] * grid.t[is];
      const Real LS = s2 * aS / (s2 + aS);
      const Real denom = (aP + LQ + LS) * (aQ + t2) * (aS + s2);
      const Real M = pi92 * exp_(-LQ * RPQ2 - LS * RPS2) / (denom * sqrt_(denom));
      acc += wt * grid.w[is] * M;
    }
  }
  // s-type normalisation N0(z) = (2 z / pi)^{3/4} per function
  const Real N = detail::pow34(2 * a.alpha / pi) * detail::pow34(2 * b.alpha / pi) *
                 detail::pow34(2 * c.alpha / pi) * detail::pow34(2 * d.alpha / pi) *
                 detail::pow34(2 * e.alpha / pi) * detail::pow34(2 * f.alpha / pi);
  return N * Kad * Kbe * Kcf * acc;
}

} // namespace intti
