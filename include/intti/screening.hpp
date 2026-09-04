// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Distance-including integral screening -- the Ochsenfeld QQR / MBIE upgrade to
// plain Cauchy-Schwarz (Lambrecht & Ochsenfeld, JCP 123, 184101 (2005); Maurer
// et al., JCP 136, 144107 (2012)). Schwarz |(ab|cd)| <= Q_ab Q_cd is separable
// and throws away the bra-ket distance decay. For non-overlapping clouds the
// Coulomb kernel is bounded by its value at closest approach, giving the
// monopole estimate
//   |(ab|cd)| <= M_ab M_cd / (R - ext_ab - ext_cd),   R = |P_ab - P_cd|,
// where M is an upper bound on the pair charge int|rho| and ext its radius. Both
// this and Schwarz are valid upper bounds, so their MIN is the tightest valid
// estimate -- Schwarz-tight for close pairs, ~1/R for separated ones. (For
// SHORT-RANGE operators no such helper is needed: the geminal e^{-theta R^2}
// decay is already distance-including, see tc.hpp / the M-HK note.)

#include <cmath>

#include "gto.hpp"
#include "math.hpp"

namespace intti {
namespace detail {

/// sqrt( int (x-A)^{2l} e^{-p (x-P)^2} dx ), dPA = P - A. Used for the per-axis
/// Cauchy-Schwarz bound on the 1-D absolute charge of a Gaussian pair factor.
template <class Real>
Real gauss_moment_sqrt(Real p, Real dPA, int l) {
  // sum_{j=0}^{l} C(2l,2j) dPA^{2(l-j)} (2j-1)!!/(2p)^j, then * sqrt(pi/p).
  auto binom = [](int n, int k) {
    Real r = 1;
    for (int i = 0; i < k; ++i) r = r * Real(n - i) / Real(i + 1);
    return r;
  };
  const Real pi = pi_v<Real>();
  Real S = 0, twop = 2 * p;
  for (int j = 0; j <= l; ++j) {
    Real df = 1; // (2j-1)!!
    for (int m = 1; m <= 2 * j - 1; m += 2) df *= Real(m);
    Real pw = 1; // dPA^{2(l-j)}
    for (int m = 0; m < 2 * (l - j); ++m) pw *= dPA;
    Real tp = 1; // (2p)^j
    for (int m = 0; m < j; ++m) tp *= twop;
    S += binom(2 * l, 2 * j) * pw * df / tp;
  }
  return sqrt_(S * sqrt_(pi / p));
}

} // namespace detail

/// Upper bound M_ab on the pair charge int|rho_ab| over a shell pair: the max
/// over Cartesian components of prod_d |K_d| g(p,P_d-A_d,a_d) g(p,P_d-B_d,b_d)
/// with g = gauss_moment_sqrt (Cauchy-Schwarz per axis). Real pairs only.
template <class Real>
Real pair_charge_bound(const ShellPair<Real> &sp) {
  const int la = sp.la, lb = sp.lb;
  // per-axis, per-power tables g_A[d][a], g_B[d][b]
  Real gA[3][8], gB[3][8]; // powers up to LMAX
  for (int d = 0; d < 3; ++d) {
    for (int a = 0; a <= la; ++a)
      gA[d][a] = detail::gauss_moment_sqrt(sp.p, Real(sp.P[d] - sp.A[d]), a);
    for (int b = 0; b <= lb; ++b)
      gB[d][b] = detail::gauss_moment_sqrt(sp.p, Real(sp.P[d] - sp.B[d]), b);
  }
  Real Km = 1;
  for (int d = 0; d < 3; ++d) Km *= std::abs(Real(sp.K[d]));
  Real best = 0;
  for (int ka = 0; ka < ncart(la); ++ka) {
    int a3[3];
    cart_comp(la, ka, a3[0], a3[1], a3[2]);
    for (int kb = 0; kb < ncart(lb); ++kb) {
      int b3[3];
      cart_comp(lb, kb, b3[0], b3[1], b3[2]);
      Real m = Km;
      for (int d = 0; d < 3; ++d) m *= gA[d][a3[d]] * gB[d][b3[d]];
      if (m > best) best = m;
    }
  }
  return best;
}

/// Radius beyond which the pair density is below eps: the Gaussian e^{-p r^2}
/// reaches eps at sqrt(ln(1/eps)/p), plus a polynomial margin (la+lb)/sqrt(p).
template <class Real>
Real pair_extent(const ShellPair<Real> &sp, Real eps) {
  using std::log;
  using std::sqrt;
  const Real g = sqrt(log(Real(1) / eps) / sp.p);
  return g + Real(sp.la + sp.lb) / sqrt(sp.p);
}

/// MBIE-1 / QQR distance-including estimate of |(bra|ket)| for the Coulomb ERI:
/// min(Q_bra Q_ket, M_bra M_ket / (R - ext_bra - ext_ket)). Qbra/Qket are the
/// caller's Schwarz factors. Always a valid upper bound; tighter than Schwarz
/// once the clouds separate. eps sets the extent tolerance.
template <class Real>
Real mbie_estimate(const ShellPair<Real> &bra, const ShellPair<Real> &ket,
                   Real Qbra, Real Qket, Real eps = Real(1e-10)) {
  const Real schwarz = Qbra * Qket;
  Real R2 = 0;
  for (int d = 0; d < 3; ++d) {
    const Real dx = Real(bra.P[d]) - Real(ket.P[d]);
    R2 += dx * dx;
  }
  const Real sep =
      sqrt_(R2) - pair_extent(bra, eps) - pair_extent(ket, eps);
  if (sep <= Real(0)) return schwarz; // clouds may overlap -> Schwarz
  const Real mono = pair_charge_bound(bra) * pair_charge_bound(ket) / sep;
  return mono < schwarz ? mono : schwarz;
}

} // namespace intti
