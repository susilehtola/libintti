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

/// Per-direction, PER HERMITE ORDER max over Cartesian components of |E_n^{ab}|.
///
/// Weighting each order by its own E mass, rather than factoring out a single
/// max_n |B_n| and multiplying by sum_n |E_n|, is what makes the bound usable at
/// high angular momentum. The Cramer factor carries sqrt(n!) -- 21886 at n = 12
/// -- but the E coefficients that actually reach those orders are small, so the
/// factorised form charges the largest |B_n| against the whole E mass. Measured
/// against the ideal truncation, that cost up to 43 of 64 nodes on (ff|ff)
/// quartets, including ones that are entirely negligible.
///
/// out must hold 3 * (la + lb + 1).
template <class Real>
void pair_e_absmax_n(const ShellPair<Real> &sp, Real *out) {
  const int la = sp.la, lb = sp.lb, nt = la + lb + 1;
  std::vector<Real> E(static_cast<std::size_t>(la + 1) * (lb + 1) * nt);
  for (int d = 0; d < 3; ++d) {
    e_coeffs(la, lb, sp.p, Real(sp.P[d] - sp.A[d]), Real(sp.P[d] - sp.B[d]),
             Real(sp.K[d]), E.data());
    for (int n = 0; n < nt; ++n) {
      Real best = 0;
      for (int i = 0; i <= la; ++i)
        for (int j = 0; j <= lb; ++j) {
          if (n > i + j) continue;
          const Real v =
              std::abs(E[(static_cast<std::size_t>(i) * (lb + 1) + j) * nt + n]);
          if (v > best) best = v;
        }
      out[d * nt + n] = best;
    }
  }
}

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

/// Largest total Hermite order the full-Gaussian branch will handle; above it
/// that branch is skipped and the other two stand alone, so exceeding it costs
/// tightness and never correctness.
///
/// 21 covers (hh|hh), i.e. every quartet a real basis set produces -- g appears
/// in quadruple-zeta sets and h in quintuple. LMAX is 8, so a pathological
/// (ll|ll) would reach 33; the fallback handles it. The cost of the choice is
/// stack: 3*21 + 21 doubles of per-thread scratch, against the 5.8 KB the
/// J-build kernel already carries.
inline constexpr int TSCR_NMAX = 21;

/// Per-direction coefficients of the full-Gaussian Hermite bound, as a
/// polynomial in theta.
///
/// This is the other half of the story Cramer's inequality tells. Cramer bounds
/// |H_n(u)| by k 2^{n/2} sqrt(n!) e^{u^2/2}, which is sharp near u ~ sqrt(n) but
/// spends HALF THE GAUSSIAN to do it: |B_n(theta,X)| <= k (2 theta)^{n/2}
/// sqrt(n!) e^{-theta X^2/2} keeps only e^{-theta X^2/2} where the true object
/// decays as e^{-theta X^2}. For a distant pair that missing half IS the
/// estimate -- at theta R^2/2 = 37, measured on a diffuse f-shell quartet at
/// R = 20, it is a factor e^{37} = 1e16, and it was the largest single source of
/// slack left in the bound.
///
/// Bounding |H_n| by its own all-positive coefficients instead keeps
/// e^{-theta X^2} intact at the price of a polynomial in |u|:
///   |H_n(u)| <= G_n(|u|),  G_0 = 1, G_n = 2u G_{n-1} + 2(n-1) G_{n-2},
/// and substituting u = sqrt(theta) a collapses the whole per-direction factor
/// to a polynomial in theta alone,
///   sum_n c_n theta^{n/2} G_n(sqrt(theta) a) = sum_k b_k theta^k,
///   b_k = sum_{n=k}^{min(2k,N)} c_n n!/((n-k)! (2k-n)!) (2a)^{2k-n},
/// which is what this returns. Writing it in theta matters: every term is then
/// theta^k e^{-theta a^2}, whose peak in theta is known exactly, and that is
/// what lets the tail be charged at the right place instead of at saturation.
///
/// c is the convolution of the bra and ket per-order E masses (the coupling
/// runs over B_{tau_bra + tau_ket}), and the recurrences avoid dividing by a,
/// so a = 0 is not special-cased.
template <class Real>
KOKKOS_INLINE_FUNCTION void t_screen_bpoly(const Real *c, int n1, Real a, Real *b) {
  for (int k = 0; k < n1; ++k) b[k] = 0;
  const Real a2 = 2 * a, a4 = a2 * a2;
  for (int n = 0; n < n1; ++n) {
    if (!(c[n] > Real(0))) continue;
    const int mmax = n / 2;
    // T = n!/(m! j!) (2a)^j at m = mmax, j = n - 2 mmax (0 or 1)
    Real T = 1;
    for (int f = mmax + 1; f <= n; ++f) T *= Real(f); // n!/mmax!
    int j = n - 2 * mmax;
    if (j == 1) T *= a2;
    for (int m = mmax; m >= 0; --m) {
      b[n - m] += c[n] * T;
      if (m == 0) break;
      // step to (m-1, j+2): T *= m (2a)^2 / ((j+1)(j+2))
      T *= Real(m) * a4 / (Real(j + 1) * Real(j + 2));
      j += 2;
    }
  }
}

/// The Cramer-derived Hermite factor at a GIVEN theta, without the prefactor:
/// min of the factorised bound (max_m (2 theta)^{m/2} sqrt(m!) against the whole
/// E mass) and the per-order one (each order weighted by its own E mass).
template <class Real>
KOKKOS_INLINE_FUNCTION Real t_screen_phi(Real th, Real Epref, int L, int nbh, int nkh,
                                         const Real *Fb, const Real *Fk) {
  Real bf = 0, fact = 1, pw = 1;
  const Real s2 = sqrt_(2 * th);
  for (int m = 0; m <= L; ++m) {
    if (m > 0) { pw *= s2; fact *= sqrt_(Real(m)); }
    const Real v = pw * fact;
    if (v > bf) bf = v;
  }
  bf *= Epref;
  Real bo = 1;
  const Real s4 = sqrt_(4 * th);
  for (int d = 0; d < 3; ++d) {
    Real gb = 0, pwb = 1, fb = 1;
    for (int n = 0; n < nbh; ++n) {
      if (n > 0) { pwb *= s4; fb *= sqrt_(Real(n)); }
      gb += Fb[d * nbh + n] * fb * pwb;
    }
    Real gk = 0, pwk = 1, fk = 1;
    for (int n = 0; n < nkh; ++n) {
      if (n > 0) { pwk *= s4; fk *= sqrt_(Real(n)); }
      gk += Fk[d * nkh + n] * fk * pwk;
    }
    bo *= gb * gk;
  }
  return bf < bo ? bf : bo;
}

/// Largest the product pref^3(u) * Phi(theta(u)) can be anywhere on the tail
/// u >= ui, where Phi is the Cramer-derived Hermite factor. Both branches are
/// rigorous upper bounds; the smaller wins. Epref (the total E mass) belongs to
/// the factorised branch only -- the per-order branch carries its E mass inside
/// the F arrays.
template <class Real>
KOKKOS_INLINE_FUNCTION Real t_screen_M(Real ui, Real pq, Real ps, Real mu, Real Epref,
                                       int L, int nbh, int nkh, const Real *Fb,
                                       const Real *Fk, Real uord) {
  const Real pi = pi_v<Real>();
  // (a) FACTORISED: max_m (2 theta)^{m/2} sqrt(m!) against the whole E mass.
  //     Each order m is charged at its OWN peak max(ui, m mu/3) -- past that
  //     point the term decays, so charging it at saturation was pure slack.
  Real Mfac = 0, fact = 1;
  for (int m = 0; m <= L; ++m) {
    if (m > 0) fact *= sqrt_(Real(m));
    const Real upk = Real(m) * mu / 3;
    const Real um = ui > upk ? ui : upk;
    const Real D = pq + um * ps;
    const Real pr = pi / sqrt_(D);
    const Real sc = sqrt_(2 * um * pq / D);
    Real v = pr * pr * pr * fact * Epref;
    for (int j = 0; j < m; ++j) v *= sc;
    if (v > Mfac) Mfac = v;
  }
  // (b) PER ORDER: each Hermite order weighted by its own E mass, which is far
  //     tighter at high angular momentum -- (a) charges sqrt(L!), 21886 at
  //     L = 12, against E mass that is not there. The six polynomial factors
  //     multiply out to total degree Nord, so the product may be charged at
  //     theta(ui) only once ui is past the last peak, uord = Nord mu / 3;
  //     below that the prefactor still moves but Phi is held at saturation.
  const Real D = pq + ui * ps;
  const Real pr = pi / sqrt_(D);
  const Real thu = (ui >= uord) ? ui * pq / D : mu;
  Real Mord = pr * pr * pr;
  const Real s4 = sqrt_(4 * thu);
  for (int d = 0; d < 3; ++d) {
    Real gb = 0, pwb = 1, fb = 1;
    for (int n = 0; n < nbh; ++n) {
      if (n > 0) { pwb *= s4; fb *= sqrt_(Real(n)); }
      gb += Fb[d * nbh + n] * fb * pwb;
    }
    Real gk = 0, pwk = 1, fk = 1;
    for (int n = 0; n < nkh; ++n) {
      if (n > 0) { pwk *= s4; fk *= sqrt_(Real(n)); }
      gk += Fk[d * nkh + n] * fk * pwk;
    }
    Mord *= gb * gk;
  }
  return Mfac < Mord ? Mfac : Mord;
}

/// Upper bound on everything node i and beyond can contribute.
///
/// TWO independent factorisations of the same sum, both rigorous, smaller wins:
///
///  (1) WEIGHT x PEAK.  sum_j |w_j| <= W(i) times the largest the integrand
///      pref^3 Phi can be anywhere on the tail. Good when Phi climbs fast --
///      high angular momentum -- because each order is charged at its own peak.
///
///  (2) WEIGHTED PREFACTOR x SATURATION.  Phi is held at its saturating value
///      Phi(mu) and the PREFACTOR is summed node by node, which is where this
///      one wins: (1) charges pref^3 at its largest value against every weight
///      in the tail, and most of that weight sits where pref^3 is orders of
///      magnitude smaller. At l = 0, where Phi == 1 and there is no angular
///      momentum in play at all, that single substitution was the whole
///      remaining slack -- a measured factor of 2e4 to 5e5.
///
///      The node sum is closed-form because pref^3(u) = pi^3 (p+q)^{-3/2}
///      (mu + u)^{-3/2}, and (mu + u)^{-3/2} <= max(mu, u)^{-3/2}, which is
///      mu^{-3/2} below t^2 = mu and t^{-3} above it. So with W and V the
///      suffix sums of |w| and |w|/t^3, and jm the first node past t^2 = mu,
///        sum_{j>=i} |w_j| pref^3(u_j)
///          <= pi^3 (p+q)^{-3/2} [ mu^{-3/2} (W(i) - W(m)) + V(m) ],  m = max(i, jm).
///      The substitution costs at most 2^{3/2} = 2.83 (at t^2 = mu exactly,
///      where max(mu,u) = mu but mu + u = 2mu), uniformly -- a bounded 2.8x for
///      an unbounded 1e5x.
///  (3) FULL GAUSSIAN.  The same weighted prefactor sum as (2), but with the
///      Hermite factor bounded by t_screen_bpoly, which keeps e^{-theta R^2}
///      instead of e^{-theta R^2/2}. Its polynomial is bigger than Cramer's near
///      contact and irrelevant beside a squared exponential once the pairs are
///      apart -- which is the case screening exists for.
///
///      Per direction the factor is sum_k b_k theta^k e^{-theta a^2}, and EACH
///      TERM is charged at its own constrained peak,
///        max_{theta in [theta_i, mu]} theta^k e^{-theta a^2}
///          at theta_k* = clamp(k / a^2, theta_i, mu),
///      which is the last instance of the mistake this bound kept making --
///      charging the polynomial at one end of the grid and the exponential at
///      the other. The three cases (past peak, at peak, short of peak) split the
///      k range into three contiguous pieces, so one pass and two exponentials
///      per direction cover it; only the middle piece needs a power, and it is
///      usually narrow or empty. `bv` null means the caller skipped the branch.
template <class Real>
KOKKOS_INLINE_FUNCTION Real t_screen_tail(int i, Real p, Real qq, Real R2, Real Epref,
                                          int L, int nbh, int nkh, const Real *Fb,
                                          const Real *Fk, const Real *tv,
                                          const Real *wsuf, const Real *vsuf, int jm,
                                          Real phimu, const Real *bv, int n1,
                                          const Real *ax, int nt, Real uord) {
  if (i >= nt) return Real(0);
  const Real kc = Real(1.086435);
  const Real k3 = kc * kc * kc;
  const Real pi = pi_v<Real>();
  const Real pq = p * qq, ps = p + qq, mu = pq / ps;
  const Real ui = tv[i] * tv[i];
  const Real th = ui * pq / (pq + ui * ps);

  const Real b1 = wsuf[i] * t_screen_M(ui, pq, ps, mu, Epref, L, nbh, nkh, Fb, Fk, uord);

  const int m = i > jm ? i : jm;
  const Real rps = Real(1) / sqrt_(ps);
  const Real rmu = Real(1) / sqrt_(mu);
  const Real pre = pi * pi * pi * rps * rps * rps;
  const Real S = pre * (rmu * rmu * rmu * (wsuf[i] - wsuf[m]) + vsuf[m]);
  const Real b2 = S * phimu;

  Real best = k3 * (b1 < b2 ? b1 : b2) * exp_(-th * R2 / 2);
  if (bv != nullptr) {
    const Real ee = exp_(Real(1));
    Real b3 = S;
    for (int d = 0; d < 3; ++d) {
      const Real a2 = ax[d] * ax[d];
      const Real *bd = bv + d * TSCR_NMAX;
      const Real klo = th * a2, khi = mu * a2;
      const Real eth = exp_(-th * a2), emu = exp_(-mu * a2);
      Real acc = 0, thp = 1, mup = 1;
      for (int k = 0; k < n1; ++k) {
        if (k > 0) {
          thp *= th;
          mup *= mu;
        }
        const Real kk = Real(k);
        if (kk <= klo)
          acc += bd[k] * thp * eth;
        else if (kk >= khi)
          acc += bd[k] * mup * emu;
        else {
          const Real r = kk / (a2 * ee);
          Real v = bd[k];
          for (int j = 0; j < k; ++j) v *= r;
          acc += v;
        }
      }
      b3 *= acc;
    }
    if (b3 < best) best = b3;
  }
  return best;
}

/// THE t-screening estimate -- ONE implementation, called from both the host
/// entry (t_screen_keep_pre) and the device kernel (t_screen_batch). They were
/// briefly two, which meant the same quartet could be screened two ways
/// depending on which entry point a caller used, and the sweep tests validated
/// only the host one.
///
/// Returns how many leading t nodes must be evaluated for the discarded
/// remainder to stay under eps.
///
/// WHAT IS BOUNDED. Node j contributes, per Cartesian direction, a factor
/// w_j (pi/sqrt(D_j)) B_n(theta_j, X) with D_j = pq + t_j^2 (p+q) and
/// theta_j = t_j^2 pq / D_j, and Cramer's inequality gives
///   |B_n(theta, X)| <= k (2 theta)^{n/2} sqrt(n!) e^{-theta X^2/2}, k = 1.086435.
/// Collecting the three directions, the tail from node i onward is at most
///   T(i) = k^3 W(i) M(i) e^{-theta_i R^2/2},  W(i) = sum_{j>=i} |w_j|,
/// with M as in t_screen_M. Every factor is nonincreasing in i, so the smallest
/// i with T(i) <= eps is found by bisection: 6 evaluations at nt = 64.
///
/// WHY M IS TAKEN AT THE TRUNCATION POINT, which is the tightening. The earlier
/// form bounded M's two factors at OPPOSITE ends of the grid -- the prefactor at
/// D >= pq (its t = 0 value) and the Hermite factor at theta <= mu (its t = inf
/// value). No node is at both, and that gap was most of the nine orders of
/// unused budget this used to carry. Two facts close it: pref^3 is decreasing in
/// u, so on the tail it is at most its value at u_i; and a term of order N rides
/// theta^{N/2}, whose product with pref^3 is UNIMODAL in u with its peak at
/// u = N mu / 3 (set d/du of (N/2) ln u - (N+3)/2 ln D to zero), so past the
/// peak it may be charged at u_i rather than at saturation.
///
/// A consequence worth naming: T(i) now falls with i even at R = 0, because the
/// prefactor and the weights decay on their own. Coincident-centre quartets --
/// which a distance-only bound can never truncate, and which are exactly the
/// compact high-L case -- are screened like any other.
template <class Real>
KOKKOS_INLINE_FUNCTION int
t_screen_scan(Real p, Real qq, const Real *X, Real Epref, int L, int nbh, int nkh,
              const Real *Fb, const Real *Fk, const Real *tv, const Real *wv,
              const Real *wsuf, const Real *vsuf, int nt, Real eps, int refine) {
  const Real kc = Real(1.086435);
  const Real k3 = kc * kc * kc;
  const Real pi = pi_v<Real>();
  const Real pq = p * qq, ps = p + qq, mu = pq / ps;
  const Real uord = Real(3 * (nbh - 1) + 3 * (nkh - 1)) * mu / 3;
  Real R2 = 0, ax[3];
  for (int d = 0; d < 3; ++d) {
    ax[d] = X[d] < Real(0) ? -X[d] : X[d];
    R2 += X[d] * X[d];
  }
  // quartet invariants, hoisted out of the bisection
  const Real phimu = t_screen_phi(mu, Epref, L, nbh, nkh, Fb, Fk);
  // Per-direction convolution of the bra and ket E masses: the coupling in
  // direction d runs over B_{tau_b + tau_k}, so the coefficient of total order n
  // is sum_{a+b=n} F^bra_a F^ket_b. Built ONCE here rather than inside the
  // bisection, which is what keeps the third branch to O(L) per evaluation.
  const int n1 = nbh + nkh - 1;
  const bool use_g = n1 <= TSCR_NMAX;
  Real bv[3 * TSCR_NMAX];
  if (use_g) {
    Real cv[TSCR_NMAX];
    for (int d = 0; d < 3; ++d) {
      for (int n = 0; n < n1; ++n) cv[n] = 0;
      for (int a = 0; a < nbh; ++a)
        for (int b = 0; b < nkh; ++b) cv[a + b] += Fb[d * nbh + a] * Fk[d * nkh + b];
      t_screen_bpoly(cv, n1, ax[d], &bv[d * TSCR_NMAX]);
    }
  }
  const Real *bvp = use_g ? bv : nullptr;
  int jm = 0, jhi = nt; // first node with t^2 > mu
  while (jm < jhi) {
    const int mid = (jm + jhi) / 2;
    if (tv[mid] * tv[mid] > mu)
      jhi = mid;
    else
      jm = mid + 1;
  }

  int lo = 0, hi = nt;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (t_screen_tail(mid, p, qq, R2, Epref, L, nbh, nkh, Fb, Fk, tv, wsuf, vsuf, jm,
                      phimu, bvp, n1, ax, nt, uord) <= eps)
      hi = mid;
    else
      lo = mid + 1;
  }
  if (lo == 0 || refine <= 0) return lo;

  // REFINE. The closed form still charges the whole tail against one bounding
  // node; walking down from its answer and accumulating the EXACT per-node
  // bound recovers the rest, and stays rigorous because everything at or above
  // `lo` is already covered by T(lo). Capped, so the cost is O(refine) rather
  // than the O(nt) node-by-node scan this replaced.
  const Real Tc = t_screen_tail(lo, p, qq, R2, Epref, L, nbh, nkh, Fb, Fk, tv, wsuf,
                                vsuf, jm, phimu, bvp, n1, ax, nt, uord);
  Real S = 0;
  int i = lo - 1;
  const int stop = lo - refine > 0 ? lo - refine : 0;
  for (; i >= stop; --i) {
    const Real u = tv[i] * tv[i];
    const Real D = pq + u * ps;
    const Real th = u * pq / D;
    const Real pr = pi / sqrt_(D);
    // at this node's exact theta -- a single node needs no peak argument
    const Real wabs = wv[i] < Real(0) ? -wv[i] : wv[i];
    const Real pr3 = pr * pr * pr;
    Real b = wabs * pr3 * k3 * t_screen_phi(th, Epref, L, nbh, nkh, Fb, Fk) *
             exp_(-th * R2 / 2);
    if (use_g) {
      Real g = 1;
      for (int d = 0; d < 3; ++d) {
        const Real *bd = &bv[d * TSCR_NMAX];
        Real acc = 0, thp = 1;
        for (int k = 0; k < n1; ++k) {
          if (k > 0) thp *= th;
          acc += bd[k] * thp;
        }
        g *= acc * exp_(-th * ax[d] * ax[d]);
      }
      const Real bg = wabs * pr3 * g;
      if (bg < b) b = bg;
    }
    if (S + b + Tc > eps) break;
    S += b;
  }
  return i + 1;
}

/// t_screen_keep with the per-pair E factors already computed.
///
/// `wsuf` holds the suffix sums sum_{j>=i} |w_j| of the grid weights -- the
/// caller builds it once rather than paying O(nt) per quartet for what used to
/// be a single total. Using the SUFFIX is itself a small tightening: the old
/// form charged the whole grid's weight against a tail that starts partway up.
template <class Real>
int t_screen_keep_pre(const ShellPair<Real> &bra, const ShellPair<Real> &ket,
                      const Real *Ea, const Real *Eb, const Real *Fa, const Real *Fb,
                      const TGrid<Real> &grid, const Real *wsuf, const Real *vsuf,
                      Real eps, int refine = 16) {
  Real Epref = 1;
  for (int d = 0; d < 3; ++d) Epref *= Ea[d] * Eb[d];
  if (!(Epref > Real(0))) return 0;
  Real X[3];
  for (int d = 0; d < 3; ++d) X[d] = Real(bra.P[d]) - Real(ket.P[d]);
  return t_screen_scan<Real>(bra.p, ket.p, X, Epref, bra.la + bra.lb + ket.la + ket.lb,
                             bra.la + bra.lb + 1, ket.la + ket.lb + 1, Fa, Fb,
                             grid.t.data(), grid.w.data(), wsuf, vsuf, grid.n(), eps,
                             refine);
}

/// The two suffix sums t_screen_scan needs: W(i) = sum_{j>=i} |w_j| and
/// V(i) = sum_{j>=i} |w_j| / t_j^3. Both are built once per grid.
///
/// W keeps an (nt+1)-th entry of zero so the tail bound can difference it
/// without a bounds check. V is only ever read at indices past t^2 = mu > 0,
/// so its small-t entries are never used; they are floored anyway rather than
/// left to overflow on a grid whose first node is near zero.
template <class Real>
std::pair<std::vector<Real>, std::vector<Real>> weight_suffix(const TGrid<Real> &grid) {
  const int nt = grid.n();
  std::vector<Real> w(nt + 1, Real(0)), v(nt + 1, Real(0));
  const Real tiny = Real(1e-8);
  for (int i = nt - 1; i >= 0; --i) {
    const Real wa = grid.w[i] < Real(0) ? -grid.w[i] : grid.w[i];
    const Real t = grid.t[i] > tiny ? grid.t[i] : tiny;
    w[i] = w[i + 1] + wa;
    v[i] = v[i + 1] + wa / (t * t * t);
  }
  return {std::move(w), std::move(v)};
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
  std::vector<Real> Fa(3 * (bra.la + bra.lb + 1)), Fb(3 * (ket.la + ket.lb + 1));
  detail::pair_e_absmax_n(bra, Fa.data());
  detail::pair_e_absmax_n(ket, Fb.data());
  const auto [wsuf, vsuf] = detail::weight_suffix(grid);
  return detail::t_screen_keep_pre(bra, ket, Ea, Eb, Fa.data(), Fb.data(), grid,
                                   wsuf.data(), vsuf.data(), eps);
}


// WHICH CALL SITES MAY USE THIS, and why the rest do not yet.
//
// Wired: jk_build's general path (its tau IS the Schwarz tolerance and bounds
// the same thing), the two RI Hessians (tau_screen, default off), the 2e
// spin-orbit build, the 2-/3-centre n-centre builds (primitive and contracted),
// and the cross-basis J/K. Everything but jk_build defaults to off.
//
// The pattern that made the last four possible is the `amp` argument: the
// screener bounds the QUARTET, but what the caller cares about is the error in
// the matrix or tensor its digest builds, and every digest but a plain ERI one
// multiplies the quartet by something first -- an MD centre-shift coefficient
// (SOC), a contraction coefficient (contracted n-centre), a density element
// (cross-basis). Handing that factor to the screener per quartet keeps eps
// meaning "absolute error in the output".
//
//   * GIAO (giao2e.hpp) -- CANNOT, as written. t_screen_keep reads
//     Real(sp.P[d]) and pair_charge_bound is real-pairs-only, while a London
//     pair carries a complex product centre and prefactor. Note though that
//     hermite_b keeps theta REAL for GIAOs and only X goes complex, so the
//     per-node bound needs |e^{-theta X^2/2}| = e^{-theta Re(X^2)/2} rather
//     than a fresh derivation -- more tractable than the real monopole
//     construction was.
//
//   * Derivative quartets (erigrad, erihess, the remaining rigrad passes) --
//     VALID but needs the check below. Their quartets are promoted/demoted
//     pairs recombined with md_grad_terms coefficients of -2*alpha and l, so a
//     per-quartet bound of eps could in principle emerge multiplied by 2*alpha,
//     which is 1e6 for a tight function. Measured across six decades of
//     exponent it does not (see DerivativeScreeningDoesNotAmplify), but that
//     test could not construct a case where screening materially moves the
//     result, so the path is guarded rather than demonstrated.
//
//   * Cholesky (cholesky.hpp) -- NOT without thought. The pivoted decomposition
//     selects on diagonal magnitudes; perturbing them by eps perturbs the pivot
//     ORDER, which is a discrete choice rather than a bounded error.
//
//   * The Schwarz pass itself (fock.hpp, make_batch(pairs, diag)) -- NEVER.
//     It defines the tolerance the screening is measured against.
//
//   * 2c/3c (ncenter.hpp) and cross-basis (crossbasis.hpp) -- WIRED, opt-in
//     tau_screen on each public entry. Real pairs and the plain Coulomb kernel,
//     so the bound is the unmodified one; only the budget changes. Contracted
//     n-centre divides by the largest effective coefficient (prim_eff_max),
//     cross-basis by max|D|. The cross-basis factor is a single scalar rather
//     than per quartet: the bound is over the whole density matrix, not the
//     block a given quartet reaches. Looser than it needs to be, and cheap
//     enough that tightening it has not been worth doing.
//
//   * SOC (soc.hpp) -- unexamined. Its operator is not the plain Coulomb
//     kernel, so the e^{-theta R^2} factor this rests on needs re-deriving
//     before the bound can be claimed.

/// Fill a batch's per-quartet node counts from t_screen_keep, so the engine
/// evaluates each quartet on only the prefix of the grid it needs.
///
/// Screening is OPT-IN: make_batch leaves every quartet at the full grid, and
/// this narrows it. eps is an absolute bound on the discarded contribution per
/// quartet, so it should be set against the tolerance the caller already uses
/// for Schwarz rather than independently.
///
/// `amp` (optional, one entry per batch quartet) is the factor by which the
/// caller's digest multiplies that quartet before it reaches the output. The
/// per-quartet budget becomes eps/amp(q), so eps keeps meaning "absolute error
/// in the quantity the caller actually builds". Derivative-like digests need
/// this: an MD centre-shift coefficient is -2 alpha, which is 1e6 for a tight
/// function, and a quartet screened at eps would land in the output at 1e6 eps.
/// Leave it empty for a plain ERI digest, where the factor is 1.
template <class Real>
void t_screen_batch(QuartetBatch<Real> &batch,
                    const std::vector<ShellPair<Real>> &pair_list,
                    const TGrid<Real> &grid, Real eps, int refine = 16,
                    const Kokkos::View<Real *> &amp = Kokkos::View<Real *>()) {
  const int nt = grid.n();
  // Two things keep this from costing more than it saves.
  //
  // The E-coefficient factors depend on the PAIR, not the quartet, so they are
  // built once per pair rather than twice per quartet -- otherwise e_coeffs runs
  // O(nq) times.
  //
  // And the per-quartet scan runs ON DEVICE. As a serial host loop it measured
  // 156 ms against 70 ms of (screened) integral evaluation on a 9-atom chain --
  // the screening dominated the work it was removing.
  //
  // Walking the node list term by term made the scan ~450 transcendentals per
  // quartet, and it became the limit: 48-51% of the screened runtime on chains,
  // so a 100x cut in nodes realised only ~6x of wall clock. The closed form
  // below removed it:
  //
  //     scan time, 9-atom chain    224 ms -> 6 ms
  //     scan time, 16-atom chain  1553 ms -> ~100-270 ms
  //     nodes kept, 16-atom chain  1.0%  -> 1.5%
  //
  // -- a little truncation depth traded for essentially the whole scan. (Net
  // speedups on this machine are not quotable: load average 19 on 6 cores while
  // measuring, and identical runs varied 2x. The scan reduction is a 15-30x
  // change and survives that; the net does not.)
  //
  // WHAT THE CLOSED FORM COSTS, measured against the per-node scan it replaced:
  //
  //     sweep truncations   2554/4500 -> 1518/4500
  //     worst loss          6.9e-12   -> 7.4e-20  (of a 1e-11 budget)
  //     pre-filter nout     14.2%     -> 17.4%
  //
  // That nine orders of unused budget is now largely closed -- see
  // t_screen_scan and t_screen_tail for the three bounds that did it. What the
  // tightening bought, on identical systems, is node work:
  //
  //     SOC chain, 5 / 7 / 9 centres    0.651 / 0.524 / 0.450  ->  0.491 / 0.376 / 0.323
  //     ERI chain, 6 / 10 / 16 centres  0.672 / 0.263 / 0.090  ->  0.454 / 0.165 / 0.057
  //
  // and against a brute-force ideal (the smallest truncation that actually
  // stays under eps, found by bisection on a 60-quartet probe spanning l <= 3,
  // alpha over two decades and R from 0 to 20), waste fell from 17.6% of the
  // grid to 7.7%, worst case 11.5x to 9.0x.
  //
  // A NOTE ON WHAT eps BUYS. It is a PER-QUARTET budget -- the same convention
  // as the Schwarz tolerance elsewhere in the library -- so an output element
  // that sums N quartets is guaranteed only N eps. That was always true; it
  // became visible only once the estimate was tight enough to actually spend
  // what it was given, which is why two n-centre tests had to be restated
  // rather than the bound loosened.
  //
  // RECOVERED by the capped refinement walk below: the closed form gives a valid
  // starting index, then the walk accumulates the EXACT per-node bound downward
  // while S + T_closed stays under eps (everything at or above the start is
  // already bounded by the closed form there, so this is rigorous). The cap
  // keeps the worst case O(refine) instead of O(nt).
  //
  // Measured on a 6-centre spd system, 52650 quartets:
  //
  //     refine  nodes   scan     eval
  //        0    5.2%    9.4 ms   2.00 s
  //        8    4.4%   12.8 ms   2.00 s
  //       16    4.2%   18.0 ms   2.12 s
  //       64    4.2%   21.8 ms   1.87 s
  //
  // Node work falls 5.2 -> 4.2% and saturates by 16, for a scan that stays ~1%
  // of the total. Hence the default. (Wall-clock nets came out 3.2-3.7x across
  // all settings, i.e. within this machine's noise -- the node fraction is the
  // measurement that means anything here.)
  const int npair = static_cast<int>(pair_list.size());
  std::vector<Real> hE(static_cast<std::size_t>(npair) * 3);
  std::vector<Real> hp(npair), hP(static_cast<std::size_t>(npair) * 3);
  std::vector<int> hla(npair), hlb(npair);
  for (int i = 0; i < npair; ++i) {
    detail::pair_e_absmax(pair_list[i], &hE[3 * i]);
    hp[i] = pair_list[i].p;
    hla[i] = pair_list[i].la;
    hlb[i] = pair_list[i].lb;
    for (int d = 0; d < 3; ++d) hP[3 * i + d] = Real(pair_list[i].P[d]);
  }
  // per-order E mass, ragged: pair i occupies 3*(la+lb+1) entries at hEoff[i]
  std::vector<int> hEoff(npair);
  int etot = 0;
  for (int i = 0; i < npair; ++i) {
    hEoff[i] = etot;
    etot += 3 * (pair_list[i].la + pair_list[i].lb + 1);
  }
  std::vector<Real> hEn(etot);
  for (int i = 0; i < npair; ++i) detail::pair_e_absmax_n(pair_list[i], &hEn[hEoff[i]]);
  auto dEn = detail::to_device(hEn, "tscr::En");
  auto dEoff = detail::to_device(hEoff, "tscr::Eoff");
  auto dE = detail::to_device(hE, "tscr::E"), dp = detail::to_device(hp, "tscr::p");
  auto dP = detail::to_device(hP, "tscr::P");
  auto dla = detail::to_device(hla, "tscr::la"), dlb = detail::to_device(hlb, "tscr::lb");
  std::vector<int> hqb(batch.nq), hqk(batch.nq);
  for (int q = 0; q < batch.nq; ++q) {
    hqb[q] = batch.h_quartets[q].first;
    hqk[q] = batch.h_quartets[q].second;
  }
  auto dqb = detail::to_device(hqb, "tscr::qb"), dqk = detail::to_device(hqk, "tscr::qk");
  auto tv = grid.t_dev, wv = grid.w_dev;
  const auto wvpair = detail::weight_suffix(grid);
  auto wsv = detail::to_device(wvpair.first, "tscr::wsuf");
  auto vsv = detail::to_device(wvpair.second, "tscr::vsuf");
  auto keep = batch.keep;
  auto ampv = amp;
  const Real eps0 = eps;
  const Real pi = pi_v<Real>();
  const Real kc = Real(1.086435);
  const int refine_n = refine;
  Kokkos::parallel_for(
      "intti::tscreen", Kokkos::RangePolicy<>(0, batch.nq), KOKKOS_LAMBDA(int q) {
        const int ib = dqb(q), ik = dqk(q);
        const Real epsq = ampv.extent(0) ? eps0 / ampv(q) : eps0;
        Real Epref = 1;
        for (int d = 0; d < 3; ++d) Epref *= dE(3 * ib + d) * dE(3 * ik + d);
        if (!(Epref > Real(0))) {
          keep(q) = 0;
          return;
        }
        Real X[3];
        for (int d = 0; d < 3; ++d) X[d] = dP(3 * ib + d) - dP(3 * ik + d);
        const int nbh = dla(ib) + dlb(ib) + 1, nkh = dla(ik) + dlb(ik) + 1;
        keep(q) = detail::t_screen_scan<Real>(
            dp(ib), dp(ik), X, Epref, nbh - 1 + nkh - 1, nbh, nkh,
            &dEn(dEoff(ib)), &dEn(dEoff(ik)), tv.data(), wv.data(), wsv.data(),
            vsv.data(), nt, epsq, refine_n);
      });
  Kokkos::fence();
  batch.nt_full = nt;
}

/// Per-quartet node counts for a quartet LIST, before a batch exists.
///
/// Screening after make_batch leaves fully-screened quartets (keep == 0) in the
/// batch: they evaluate no nodes, but still pay the f-phase E-coefficient
/// assembly and still occupy output slots, so nout_total is unchanged. Running
/// the estimate first lets the caller drop them outright.
template <class Real>
std::vector<int> t_screen_keeps(const std::vector<std::pair<int, int>> &quartets,
                                const std::vector<ShellPair<Real>> &pair_list,
                                const TGrid<Real> &grid, Real eps) {
  const int npair = static_cast<int>(pair_list.size());
  std::vector<Real> Eabs(static_cast<std::size_t>(npair) * 3);
  std::vector<int> Foff(npair);
  int ftot = 0;
  for (int i = 0; i < npair; ++i) {
    Foff[i] = ftot;
    ftot += 3 * (pair_list[i].la + pair_list[i].lb + 1);
  }
  std::vector<Real> Fn(ftot);
  for (int i = 0; i < npair; ++i) {
    detail::pair_e_absmax(pair_list[i], &Eabs[3 * i]);
    detail::pair_e_absmax_n(pair_list[i], &Fn[Foff[i]]);
  }
  const auto [wsuf, vsuf] = detail::weight_suffix(grid);
  std::vector<int> keep(quartets.size());
  for (std::size_t q = 0; q < quartets.size(); ++q) {
    const auto [ib, ik] = quartets[q];
    keep[q] = detail::t_screen_keep_pre(pair_list[ib], pair_list[ik], &Eabs[3 * ib],
                                        &Eabs[3 * ik], &Fn[Foff[ib]], &Fn[Foff[ik]], grid,
                                        wsuf.data(), vsuf.data(), eps);
  }
  return keep;
}

/// Drop the quartets that contribute nothing, returning the survivors and their
/// node counts. Feed the survivors to make_batch and the counts to
/// t_screen_apply.
template <class Real>
std::pair<std::vector<std::pair<int, int>>, std::vector<int>>
t_screen_filter(const std::vector<std::pair<int, int>> &quartets,
                const std::vector<ShellPair<Real>> &pair_list, const TGrid<Real> &grid,
                Real eps) {
  const auto keep = t_screen_keeps(quartets, pair_list, grid, eps);
  std::vector<std::pair<int, int>> live;
  std::vector<int> livekeep;
  live.reserve(quartets.size());
  livekeep.reserve(quartets.size());
  for (std::size_t q = 0; q < quartets.size(); ++q)
    if (keep[q] > 0) {
      live.push_back(quartets[q]);
      livekeep.push_back(keep[q]);
    }
  return {std::move(live), std::move(livekeep)};
}

/// Install already-computed node counts on a batch built from the survivors.
template <class Real>
void t_screen_apply(QuartetBatch<Real> &batch, const std::vector<int> &keep, int nt) {
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
