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
#include "hermite1d.hpp"
#include "math.hpp"
#include "batch.hpp"
#include "tgrid.hpp"

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


// ---------------------------------------------------------------------------
// t-RESOLVED screening: truncate the quadrature per quartet, not the quartet.
//
// This one has no counterpart in a Boys-function code, because it needs the
// Coulomb kernel resolved into Gaussians. At node t the kernel is e^{-t^2 r12^2}
// and the two-pair matrix element carries an explicit e^{-theta(t) R^2}, with
//     theta(t) = t^2 p q / (p q + t^2 (p + q)),   R = |P_bra - P_ket|.
// theta is monotone in t and SATURATES at mu = pq/(p+q) rather than diverging,
// so the damping is bounded by e^{-mu R^2} and never removes the small-t nodes
// -- which is correct, because the small-t region is precisely where the 1/R
// tail comes from. Taking theta ~ t^2 there, the contributing window is
// t <~ sqrt(ln(1/eps))/R, and the integral over it reproduces q_ab q_cd / R:
// the monopole estimate is not an approximation bolted on, it is what the
// small-t part of the integrand IS.
//
// So the payoff is NOT rejecting distant quartets -- their interaction is real
// -- but evaluating them on a short PREFIX of the grid. This composes with
// Schwarz rather than replacing it.
//
// The bound. hermite_b gives B_n = (d/dX)^n e^{-theta X^2} = (-sqrt(theta))^n
// H_n(sqrt(theta) X) e^{-theta X^2}. Cramer's inequality,
// |H_n(u)| <= k 2^{n/2} sqrt(n!) e^{u^2/2} with k = 1.086435, absorbs the
// polynomial growth into half the exponent:
//     |B_n(theta, X)| <= k (2 theta)^{n/2} sqrt(n!) e^{-theta X^2 / 2}.
// Multiplying the three directions and the absolute E-sums of the two pairs
// bounds a node's contribution, and the tail is summed from the top of the grid
// down. Loose by design -- it is a screening estimate, and every step is an
// upper bound.

namespace detail {

/// Per-direction max over Cartesian components of sum_t |E_t^{ab}|: the
/// t-independent half of a node's bound.
template <class Real> void pair_e_absmax(const ShellPair<Real> &sp, Real out[3]) {
  const int la = sp.la, lb = sp.lb, nt = la + lb + 1;
  std::vector<Real> E(static_cast<std::size_t>(la + 1) * (lb + 1) * nt);
  for (int d = 0; d < 3; ++d) {
    const Real ab = sp.A[d] - sp.B[d];
    // K is already folded into E_0^{00} by e_coeffs when passed here
    e_coeffs(la, lb, sp.p, Real(sp.P[d] - sp.A[d]), Real(sp.P[d] - sp.B[d]),
             Real(sp.K[d]), E.data());
    (void)ab;
    Real best = 0;
    for (int i = 0; i <= la; ++i)
      for (int j = 0; j <= lb; ++j) {
        Real acc = 0;
        for (int t = 0; t <= i + j; ++t)
          acc += std::abs(E[(static_cast<std::size_t>(i) * (lb + 1) + j) * nt + t]);
        if (acc > best) best = acc;
      }
    out[d] = best;
  }
}

/// k^3 * max_{m <= L} (2 theta)^{m/2} sqrt(m!) -- the Cramer factor, maximised
/// over the Hermite order because (2 theta)^{m/2} sqrt(m!) is not monotone in m
/// when 2 theta < 1.
template <class Real> Real cramer_factor(Real theta, int L) {
  const Real k = Real(1.086435);
  const Real two_theta = 2 * theta;
  Real best = 0, pw = 1, fact = 1;
  for (int m = 0; m <= L; ++m) {
    if (m > 0) {
      pw *= std::sqrt(two_theta);
      fact *= std::sqrt(Real(m));
    }
    const Real v = pw * fact;
    if (v > best) best = v;
  }
  return k * k * k * best;
}

} // namespace detail

/// Number of leading quadrature nodes that must be kept for this quartet: nodes
/// at index >= the return value contribute less than `eps` in total. Returns
/// grid.n() when no truncation is justified.
///
/// theta is monotone in t, so this is a one-sided prefix -- the small-t nodes
/// are never discarded.
namespace detail {

/// t_screen_keep with the per-pair E factors already computed.
template <class Real>
int t_screen_keep_pre(const ShellPair<Real> &bra, const ShellPair<Real> &ket,
                      const Real *Ea, const Real *Eb, const TGrid<Real> &grid, Real eps) {
  const int nt = grid.n();
  Real Epref = 1;
  for (int d = 0; d < 3; ++d) Epref *= Ea[d] * Eb[d];
  if (!(Epref > Real(0))) return 0;
  Real R2 = 0;
  for (int d = 0; d < 3; ++d) {
    const Real dx = Real(bra.P[d]) - Real(ket.P[d]);
    R2 += dx * dx;
  }
  const int L = bra.la + bra.lb + ket.la + ket.lb;
  const Real p = bra.p, q = ket.p;
  const Real pi = pi_v<Real>();
  // walk down from the top of the grid, accumulating the discarded tail
  Real tail = 0;
  for (int i = nt - 1; i >= 0; --i) {
    const Real t = grid.t[i];
    const Real D = p * q + t * t * (p + q);
    const Real theta = t * t * p * q / D;
    const Real b = std::abs(grid.w[i]) * (pi / std::sqrt(D)) * Epref *
                   detail::cramer_factor(theta, L) * std::exp(-theta * R2 / 2);
    if (tail + b > eps) return i + 1; // node i must be kept
    tail += b;
  }
  return 0;
}

} // namespace detail

/// Convenience form that computes the pair factors itself. Prefer
/// t_screen_batch for a whole batch, which hoists them.
template <class Real>
int t_screen_keep(const ShellPair<Real> &bra, const ShellPair<Real> &ket,
                  const TGrid<Real> &grid, Real eps) {
  Real Ea[3], Eb[3];
  detail::pair_e_absmax(bra, Ea);
  detail::pair_e_absmax(ket, Eb);
  return detail::t_screen_keep_pre(bra, ket, Ea, Eb, grid, eps);
}


/// Fill a batch's per-quartet node counts from t_screen_keep, so the engine
/// evaluates each quartet on only the prefix of the grid it needs.
///
/// Screening is OPT-IN: make_batch leaves every quartet at the full grid, and
/// this narrows it. eps is an absolute bound on the discarded contribution per
/// quartet, so it should be set against the tolerance the caller already uses
/// for Schwarz rather than independently.
template <class Real>
void t_screen_batch(QuartetBatch<Real> &batch,
                    const std::vector<ShellPair<Real>> &pair_list,
                    const TGrid<Real> &grid, Real eps) {
  const int nt = grid.n();
  // The E-coefficient factors depend on the PAIR, not the quartet, so they are
  // built once per pair rather than twice per quartet. Without this the
  // screening pass calls e_coeffs O(nq) times and can cost more than it saves.
  const int npair = static_cast<int>(pair_list.size());
  std::vector<Real> Eabs(static_cast<std::size_t>(npair) * 3);
  for (int i = 0; i < npair; ++i) detail::pair_e_absmax(pair_list[i], &Eabs[3 * i]);
  std::vector<int> keep(batch.nq, nt);
  for (int q = 0; q < batch.nq; ++q) {
    const auto [ib, ik] = batch.h_quartets[q];
    keep[q] = detail::t_screen_keep_pre(pair_list[ib], pair_list[ik], &Eabs[3 * ib],
                                        &Eabs[3 * ik], grid, eps);
  }
  auto h = Kokkos::create_mirror_view(batch.keep);
  for (int q = 0; q < batch.nq; ++q) h(q) = keep[q];
  Kokkos::deep_copy(batch.keep, h);
  batch.nt_full = nt;
}

/// Total nodes that would be evaluated, before and after screening -- for
/// callers that want to report or tune the saving.
template <class Real>
std::pair<std::size_t, std::size_t> t_screen_nodes(const QuartetBatch<Real> &batch,
                                                   int nt) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, batch.keep);
  std::size_t kept = 0;
  for (int q = 0; q < batch.nq; ++q) kept += std::min(h(q), nt);
  return {static_cast<std::size_t>(batch.nq) * nt, kept};
}

} // namespace intti
