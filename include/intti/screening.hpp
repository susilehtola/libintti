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

/// t_screen_keep with the per-pair E factors already computed.
///
/// ONE formula, shared with the device kernel in t_screen_batch. They were
/// briefly different -- this one scanning node by node while the device used the
/// collective bound -- which meant the same quartet could be screened two ways
/// depending on which entry point a caller used, and the sweep tests validated
/// only the host one.
template <class Real>
int t_screen_keep_pre(const ShellPair<Real> &bra, const ShellPair<Real> &ket,
                      const Real *Ea, const Real *Eb, const Real *Fa, const Real *Fb,
                      const TGrid<Real> &grid, Real eps) {
  const int nt = grid.n();
  const Real pi = pi_v<Real>();
  const Real kc = Real(1.086435);
  Real Epref = 1;
  for (int d = 0; d < 3; ++d) Epref *= Ea[d] * Eb[d];
  if (!(Epref > Real(0))) return 0;
  Real R2 = 0;
  for (int d = 0; d < 3; ++d) {
    const Real dx = Real(bra.P[d]) - Real(ket.P[d]);
    R2 += dx * dx;
  }
  const int L = bra.la + bra.lb + ket.la + ket.lb;
  const int nbh = bra.la + bra.lb + 1, nkh = ket.la + ket.lb + 1;
  const Real p = bra.p, q = ket.p;
  const Real mu = p * q / (p + q);
  const Real pr = pi / std::sqrt(p * q);
  const Real pref3 = pr * pr * pr;
  Real wsum = 0;
  for (int i = 0; i < nt; ++i) wsum += std::abs(grid.w[i]);

  // (a) factorised, (b) per Hermite order -- see t_screen_batch for the algebra
  Real best = 0, pw = 1, fact = 1;
  const Real smu = std::sqrt(2 * mu);
  for (int m = 0; m <= L; ++m) {
    if (m > 0) {
      pw *= smu;
      fact *= std::sqrt(Real(m));
    }
    best = std::max(best, pw * fact);
  }
  const Real A_fac = pref3 * kc * kc * kc * best * wsum * Epref;
  const Real s4 = std::sqrt(4 * mu);
  Real A_ord = pref3 * kc * kc * kc * wsum;
  for (int d = 0; d < 3; ++d) {
    Real gb = 0, pwb = 1, fb = 1;
    for (int n = 0; n < nbh; ++n) {
      if (n > 0) { pwb *= s4; fb *= std::sqrt(Real(n)); }
      gb += Fa[d * nbh + n] * fb * pwb;
    }
    Real gk = 0, pwk = 1, fk = 1;
    for (int n = 0; n < nkh; ++n) {
      if (n > 0) { pwk *= s4; fk *= std::sqrt(Real(n)); }
      gk += Fb[d * nkh + n] * fk * pwk;
    }
    A_ord *= gb * gk;
  }
  const Real C = std::min(A_fac, A_ord);

  if (!(C > eps)) return 0;
  if (!(R2 > Real(0))) return nt;
  const Real thstar = 2 * std::log(C / eps) / R2;
  if (thstar >= mu) return nt;
  const Real den = p * q - thstar * (p + q);
  if (!(den > Real(0))) return nt;
  const Real t2star = thstar * p * q / den;
  int lo = 0, hi = nt;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (grid.t[mid] * grid.t[mid] > t2star)
      hi = mid;
    else
      lo = mid + 1;
  }
  return lo;
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
  return detail::t_screen_keep_pre(bra, ket, Ea, Eb, Fa.data(), Fb.data(), grid, eps);
}


// WHICH CALL SITES MAY USE THIS, and why the rest do not yet.
//
// Wired: jk_build's general path (its tau IS the Schwarz tolerance and bounds
// the same thing) and the two RI Hessians (new tau_screen, default off).
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
//   * 2c/3c (ncenter.hpp) and cross-basis (crossbasis.hpp) -- should be fine,
//     real pairs and the plain Coulomb kernel, but they take no screening
//     tolerance today and are not on any measured hot path.
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
template <class Real>
void t_screen_batch(QuartetBatch<Real> &batch,
                    const std::vector<ShellPair<Real>> &pair_list,
                    const TGrid<Real> &grid, Real eps) {
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
  // Nine orders of unused budget: the collective bound charges F(mu), the
  // largest theta, against every node including the small-t ones that dominate
  // the sum. Weighting each Hermite order by its own E mass (the A_ord branch)
  // was added to recover high-L tightness and does not close this -- the
  // looseness is the mu substitution, not the L factor.
  //
  // The way back, not taken here: use the closed form to get a valid starting
  // index cheaply, then walk DOWN from it accumulating the exact tail while it
  // stays under eps. Cost is O(keep_closed - keep_exact) rather than O(nt), so
  // it is cheap exactly when the two are close and expensive when the gain is
  // largest -- which needs measuring before it is worth doing.
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
  Real wsum_h = 0;
  for (int i = 0; i < nt; ++i) wsum_h += std::abs(grid.w[i]);
  const Real wsum = wsum_h;
  auto keep = batch.keep;
  const Real pi = pi_v<Real>();
  const Real kc = Real(1.086435);
  Kokkos::parallel_for(
      "intti::tscreen", Kokkos::RangePolicy<>(0, batch.nq), KOKKOS_LAMBDA(int q) {
        const int ib = dqb(q), ik = dqk(q);
        Real Epref = 1;
        for (int d = 0; d < 3; ++d) Epref *= dE(3 * ib + d) * dE(3 * ik + d);
        if (!(Epref > Real(0))) {
          keep(q) = 0;
          return;
        }
        const int nbh = dla(ib) + dlb(ib) + 1, nkh = dla(ik) + dlb(ik) + 1;
        Real R2 = 0;
        for (int d = 0; d < 3; ++d) {
          const Real dx = dP(3 * ib + d) - dP(3 * ik + d);
          R2 += dx * dx;
        }
        const int L = dla(ib) + dlb(ib) + dla(ik) + dlb(ik);
        const Real p = dp(ib), qq = dp(ik);
        // The discarded tail is bounded COLLECTIVELY rather than summed term by
        // term, which turns an O(nt) walk with ~450 transcendentals into O(1):
        //
        //   sum_{j>=i} w_j (pi/sqrt(D_j)) F(theta_j) e^{-theta_j R^2/2}
        //     <= e^{-theta_i R^2/2} * (pi/sqrt(pq)) * F(mu) * sum_j |w_j|
        //
        // using theta_j increasing (so the exponential is largest at j = i),
        // D_j >= pq, theta_j <= mu = pq/(p+q), and F increasing in theta. Every
        // step is an upper bound, so the result is still rigorous -- just looser,
        // which costs some truncation depth and buys the whole scan.
        const Real mu = p * qq / (p + qq);
        // pref^3, one power per Cartesian direction (see the host path)
        const Real pr = pi / sqrt_(p * qq);
        const Real pref3 = pr * pr * pr;

        // (a) FACTORISED: max_n |B_n| charged against the whole E mass.
        Real best = 0, pw = 1, fact = 1;
        const Real smu = sqrt_(2 * mu);
        for (int m = 0; m <= L; ++m) {
          if (m > 0) {
            pw *= smu;
            fact *= sqrt_(Real(m));
          }
          const Real v = pw * fact;
          if (v > best) best = v;
        }
        const Real A_fac = pref3 * kc * kc * kc * best * wsum * Epref;

        // (b) PER ORDER: weight each Hermite order by its own E mass. Using
        //     sqrt((ta+tb)!) <= sqrt(ta!) sqrt(tb!) 2^{(ta+tb)/2} the double sum
        //     factorises into one polynomial per pair per direction,
        //       G(theta) = sum_n F_n sqrt(n!) (4 theta)^{n/2},
        //     which is far tighter at high angular momentum: (a) charges
        //     sqrt(L!) -- 21886 at L = 12 -- against E mass that is not there.
        const Real s4 = sqrt_(4 * mu);
        Real A_ord = pref3 * kc * kc * kc * wsum;
        for (int d = 0; d < 3; ++d) {
          Real gb = 0, pwb = 1, fb = 1;
          for (int n = 0; n < nbh; ++n) {
            if (n > 0) { pwb *= s4; fb *= sqrt_(Real(n)); }
            gb += dEn(dEoff(ib) + d * nbh + n) * fb * pwb;
          }
          Real gk = 0, pwk = 1, fk = 1;
          for (int n = 0; n < nkh; ++n) {
            if (n > 0) { pwk *= s4; fk *= sqrt_(Real(n)); }
            gk += dEn(dEoff(ik) + d * nkh + n) * fk * pwk;
          }
          A_ord *= gb * gk;
        }
        // both are valid upper bounds, so use the smaller
        const Real C = A_fac < A_ord ? A_fac : A_ord;
        if (!(C > eps)) { // even the whole tail is negligible
          keep(q) = 0;
          return;
        }
        if (!(R2 > Real(0))) { // coincident centres: no distance decay to exploit
          keep(q) = nt;
          return;
        }
        // need theta_i > theta* for the tail beyond i to fall under eps
        const Real thstar = 2 * log_(C / eps) / R2;
        if (thstar >= mu) { // theta saturates below the threshold: keep all
          keep(q) = nt;
          return;
        }
        // theta(t) > thstar  <=>  t^2 > thstar pq / (pq - thstar (p+q))
        const Real den = p * qq - thstar * (p + qq);
        if (!(den > Real(0))) {
          keep(q) = nt;
          return;
        }
        const Real t2star = thstar * p * qq / den;
        // keep the nodes with t^2 <= t2star; grid.t is ascending
        int lo = 0, hi = nt; // first index with t^2 > t2star
        while (lo < hi) {
          const int mid = (lo + hi) / 2;
          if (tv(mid) * tv(mid) > t2star)
            hi = mid;
          else
            lo = mid + 1;
        }
        keep(q) = lo;
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
  std::vector<int> keep(quartets.size());
  for (std::size_t q = 0; q < quartets.size(); ++q) {
    const auto [ib, ik] = quartets[q];
    keep[q] = detail::t_screen_keep_pre(pair_list[ib], pair_list[ik], &Eabs[3 * ib],
                                        &Eabs[3 * ik], &Fn[Foff[ib]], &Fn[Foff[ik]], grid,
                                        eps);
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
