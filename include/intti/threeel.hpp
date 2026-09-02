// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Three-electron Coulomb integrals by numerical integration -- the same
// t-quadrature that resolves the two-electron 1/r, applied twice.
//
//   G_abcdef = <a(1)b(2)c(3)| r12^{-1} r13^{-1} |d(1)e(2)f(3)>.
//
// Both operators are replaced by their Gaussian integral transform, so the
// integral becomes a TWO-dimensional (t,s) quadrature (Mehine, Losilla &
// Sundholm, Mol. Phys. 111, 2536 (2013)). At each (t,s) node the six-fold
// integral over the three electron coordinates is a product of Gaussians:
// electron 1 (density P = a d, exponent aP) is shared, electron 2 (Q = b e)
// couples to it through the geminal exponent t^2, electron 3 (S = c f) through
// s^2. The integral factorises over the Cartesian directions; in each direction
// it is a 3-variable Gaussian in (x1, x2, x3) with quadratic form
//
//   A = [[aP+t^2+s^2, -t^2,    -s^2   ],
//        [-t^2,        aQ+t^2,  0      ],
//        [-s^2,        0,       aS+s^2 ]],   det A = (aP+LQ+LS)(aQ+t^2)(aS+s^2),
//
// (LQ = aQ t^2/(aQ+t^2), LS = aS s^2/(aS+s^2)) and a linear source set by the
// centre offsets. The s-value is pi^{9/2}/det(A)^{3/2} exp(-Phi), with the FULL
// three-term exponent Phi = [aP LQ R_PQ^2 + aP LS R_PS^2 + LQ LS R_QS^2]/(aP+
// LQ+LS) -- all three pair distances P-Q, P-S AND Q-S enter. Arbitrary angular
// momentum is a polynomial moment of that Gaussian: the Cartesian GTO product
// is expanded on the pair centre (T coefficients, Gaussian product theorem) and
// the moment E[(x1-RP)^nP (x2-RQ)^nQ (x3-RS)^nS] is evaluated in closed form
// from the mean/covariance of the 3-variable Gaussian (Isserlis recurrence).

#include <algorithm>
#include <cmath>
#include <vector>

#include "math.hpp"
#include "tgrid.hpp"

namespace intti {

/// A single Cartesian primitive Gaussian x^lx y^ly z^lz exp(-alpha (r-C)^2).
template <class Real> struct CartGauss {
  Real alpha;
  Real center[3];
  int l[3];
};

namespace detail {

/// Cartesian product coefficients T_u of (x-A)^la (x-D)^ld = sum_u T_u (x-P)^u,
///   T_u = sum_{k+m=u} C(la,k)(P-A)^{la-k} C(ld,m)(P-D)^{ld-m}.
template <class Real>
void te_prod_coeffs(int la, Real PA, int ld, Real PD, std::vector<Real> &T) {
  auto C = [](int n, int k) {
    double r = 1;
    for (int i = 0; i < k; ++i) r = r * (n - i) / (i + 1);
    return r;
  };
  auto ipow = [](Real x, int n) {
    Real r = 1;
    for (int i = 0; i < n; ++i) r *= x;
    return r;
  };
  T.assign(la + ld + 1, Real(0));
  for (int k = 0; k <= la; ++k)
    for (int m = 0; m <= ld; ++m)
      T[k + m] += Real(C(la, k)) * ipow(PA, la - k) * Real(C(ld, m)) * ipow(PD, ld - m);
}

/// 3x3 matrix and its inverse (with determinant) for the (t,s)-node quadratic
/// form. A is small and SPD, so a closed-form adjugate is both exact and fast.
template <class Real> struct Mat3 {
  Real m[3][3];
};
template <class Real> Mat3<Real> te_inv3(const Mat3<Real> &A, Real &det) {
  const auto &a = A.m;
  const Real c00 = a[1][1] * a[2][2] - a[1][2] * a[2][1];
  const Real c01 = -(a[1][0] * a[2][2] - a[1][2] * a[2][0]);
  const Real c02 = a[1][0] * a[2][1] - a[1][1] * a[2][0];
  det = a[0][0] * c00 + a[0][1] * c01 + a[0][2] * c02;
  Mat3<Real> I;
  I.m[0][0] = c00 / det;
  I.m[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / det;
  I.m[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det;
  I.m[1][0] = c01 / det;
  I.m[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / det;
  I.m[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / det;
  I.m[2][0] = c02 / det;
  I.m[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / det;
  I.m[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det;
  return I;
}

/// Central moments E[e1^i e2^j e3^k] of a zero-mean Gaussian with covariance
/// Sigma, tabulated for i<=Im, j<=Jm, k<=Km via the Isserlis (Stein) recurrence
/// E[e_a prod] = sum_b Sigma_ab E[d_{e_b} prod]; reducing the first positive
/// index references only strictly earlier table entries.
template <class Real>
std::vector<Real> te_central_moments(const Real Sig[3][3], int Im, int Jm, int Km) {
  const int SJ = Jm + 1, SK = Km + 1;
  auto id = [&](int i, int j, int k) { return (i * SJ + j) * SK + k; };
  std::vector<Real> c(static_cast<std::size_t>(Im + 1) * SJ * SK, Real(0));
  auto get = [&](int i, int j, int k) -> Real {
    if (i < 0 || j < 0 || k < 0) return Real(0);
    return c[id(i, j, k)];
  };
  c[id(0, 0, 0)] = 1;
  for (int i = 0; i <= Im; ++i)
    for (int j = 0; j <= Jm; ++j)
      for (int k = 0; k <= Km; ++k) {
        if (i == 0 && j == 0 && k == 0) continue;
        Real v;
        if (i > 0)
          v = Real(i - 1) * Sig[0][0] * get(i - 2, j, k) +
              Real(j) * Sig[0][1] * get(i - 1, j - 1, k) +
              Real(k) * Sig[0][2] * get(i - 1, j, k - 1);
        else if (j > 0)
          v = Real(j - 1) * Sig[1][1] * get(i, j - 2, k) +
              Real(k) * Sig[1][2] * get(i, j - 1, k - 1);
        else
          v = Real(k - 1) * Sig[2][2] * get(i, j, k - 2);
        c[id(i, j, k)] = v;
      }
  return c;
}

/// One quadrature node for an inter-electronic operator: the operator, sampled
/// as weight * exp(-t^2 r^2). Coulomb r^{-1} = (2/sqrt pi) int exp(-t^2 r^2) dt
/// is the whole t-grid (weights carry 2/sqrt pi); a Gaussian geminal
/// sum_k c_k exp(-g_k r^2) is the fixed nodes {(c_k, sqrt g_k)} (no integration).
template <class Real> struct OpNode {
  Real weight, t;
};

/// Coulomb operator r^{-1} as node list (the t-grid).
template <class Real> std::vector<OpNode<Real>> coulomb_nodes(const TGrid<Real> &grid) {
  std::vector<OpNode<Real>> nd(grid.n());
  for (int i = 0; i < grid.n(); ++i) nd[i] = {grid.w[i], grid.t[i]};
  return nd;
}
/// Gaussian geminal sum_k c[k] exp(-g[k] r^2) as node list.
template <class Real>
std::vector<OpNode<Real>> gaussian_nodes(const std::vector<Real> &c,
                                         const std::vector<Real> &g) {
  std::vector<OpNode<Real>> nd(c.size());
  for (std::size_t k = 0; k < c.size(); ++k) nd[k] = {c[k], sqrt_(g[k])};
  return nd;
}
/// Geminal-over-r operator (sum_k c[k] e^{-g[k] r^2}) / r as a node list: each
/// term times 1/r = (2/sqrt pi) int e^{-t^2 r^2} dt is the Coulomb grid with the
/// exponent shifted, t -> sqrt(g[k] + t_i^2). g[k]=0 recovers plain Coulomb.
/// This is the F12 "f/r" operator (f a Gaussian geminal).
template <class Real>
std::vector<OpNode<Real>> geminal_over_r_nodes(const std::vector<Real> &c,
                                               const std::vector<Real> &g,
                                               const TGrid<Real> &grid) {
  std::vector<OpNode<Real>> nd;
  nd.reserve(c.size() * grid.n());
  for (std::size_t k = 0; k < c.size(); ++k)
    for (int i = 0; i < grid.n(); ++i)
      nd.push_back({c[k] * grid.w[i], sqrt_(g[k] + grid.t[i] * grid.t[i])});
  return nd;
}

/// Core three-electron accumulation shared by the plain integral and the moment
/// variants. `moment` selects an inserted scalar (summed over Cartesian axes):
///   0 -> none:        G = <ab c| op12(r12) op13(r13) |d e f>;
///   1 -> r12^2:       <ab c| r12^2 op12 op13 |def>, r12^2 = sum (x1-x2)^2;
///   2 -> r12 . r13:   the vector dot product, sum (x1-x2)(x1-x3).
/// The r12^2 operator is chosen by the caller through the node lists: Coulomb
/// nodes give the linear operator r12 = r12^2 r12^{-1}; the Gaussian nodes
/// {(4 g_k g_l c_k c_l, sqrt(g_k+g_l))} give (grad_1 f12).(grad_1 f12). The
/// r12.r13 moment gives the cross gradient (grad_1 f12).(grad_1 f13) with
/// Gaussian nodes on both slots, an F12 commutator / B-matrix ingredient.
template <class Real>
Real te_core(const CartGauss<Real> &a, const CartGauss<Real> &b, const CartGauss<Real> &c,
             const CartGauss<Real> &d, const CartGauss<Real> &e, const CartGauss<Real> &f,
             const std::vector<OpNode<Real>> &nodes12,
             const std::vector<OpNode<Real>> &nodes13, int moment) {
  auto pairdens = [](const CartGauss<Real> &x, const CartGauss<Real> &y, Real &alpha,
                     Real R[3], Real &K) {
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
  pairdens(a, d, aP, RP, Kad);
  pairdens(b, e, aQ, RQ, Kbe);
  pairdens(c, f, aS, RS, Kcf);
  // per-direction centre offsets P-Q and P-S
  Real DPQ[3], DPS[3];
  for (int i = 0; i < 3; ++i) {
    DPQ[i] = RP[i] - RQ[i];
    DPS[i] = RP[i] - RS[i];
  }
  // per-direction product coefficients and max angular indices
  std::vector<Real> TP[3], TQ[3], TS[3];
  int mP[3], mQ[3], mS[3];
  for (int i = 0; i < 3; ++i) {
    te_prod_coeffs(a.l[i], RP[i] - a.center[i], d.l[i], RP[i] - d.center[i], TP[i]);
    te_prod_coeffs(b.l[i], RQ[i] - b.center[i], e.l[i], RQ[i] - e.center[i], TQ[i]);
    te_prod_coeffs(c.l[i], RS[i] - c.center[i], f.l[i], RS[i] - f.center[i], TS[i]);
    mP[i] = a.l[i] + d.l[i];
    mQ[i] = b.l[i] + e.l[i];
    mS[i] = c.l[i] + f.l[i];
  }
  const int ext = moment ? 2 : 0;
  const int Im = std::max({mP[0], mP[1], mP[2]}) + ext;
  const int Jm = std::max({mQ[0], mQ[1], mQ[2]}) + ext;
  const int Km = std::max({mS[0], mS[1], mS[2]}) + ext; // r12.r13 raises S by 1
  // binomials up to the orders we need
  const int Bn = std::max({Im, Jm, Km}) + 1;
  std::vector<std::vector<Real>> C(Bn, std::vector<Real>(Bn, Real(0)));
  for (int n = 0; n < Bn; ++n) {
    C[n][0] = 1;
    for (int k = 1; k <= n; ++k) C[n][k] = C[n - 1][k - 1] + C[n - 1][k];
  }
  const Real pi = pi_v<Real>();
  const Real pi92 = pi * pi * pi * pi * sqrt_(pi);
  Real acc = 0;
  for (const auto &n12 : nodes12) {
    const Real t2 = n12.t * n12.t;
    for (const auto &n13 : nodes13) {
      const Real s2 = n13.t * n13.t;
      Mat3<Real> A;
      A.m[0][0] = aP + t2 + s2;
      A.m[0][1] = A.m[1][0] = -t2;
      A.m[0][2] = A.m[2][0] = -s2;
      A.m[1][1] = aQ + t2;
      A.m[1][2] = A.m[2][1] = 0;
      A.m[2][2] = aS + s2;
      Real det;
      const Mat3<Real> Ai = te_inv3(A, det);
      Real Sig[3][3];
      for (int p = 0; p < 3; ++p)
        for (int q = 0; q < 3; ++q) Sig[p][q] = Ai.m[p][q] / 2;
      const auto cmom = te_central_moments(Sig, Im, Jm, Km);
      const int SJ = Jm + 1, SK = Km + 1;
      auto cm = [&](int i, int j, int k) { return cmom[(i * SJ + j) * SK + k]; };
      const Real prefac = pi92 / (det * sqrt_(det));
      Real F[3], Fr2[3];
      for (int dir = 0; dir < 3; ++dir) {
        // linear source b and constant c0 of this direction's Gaussian
        const Real dpq = DPQ[dir], dps = DPS[dir];
        const Real b0 = -(t2 * dpq + s2 * dps), b1 = t2 * dpq, b2 = s2 * dps;
        const Real mu0 = Ai.m[0][0] * b0 + Ai.m[0][1] * b1 + Ai.m[0][2] * b2;
        const Real mu1 = Ai.m[1][0] * b0 + Ai.m[1][1] * b1 + Ai.m[1][2] * b2;
        const Real mu2 = Ai.m[2][0] * b0 + Ai.m[2][1] * b1 + Ai.m[2][2] * b2;
        const Real c0 = t2 * dpq * dpq + s2 * dps * dps;
        const Real base = exp_(b0 * mu0 + b1 * mu1 + b2 * mu2 - c0);
        // powers of the mean
        std::vector<Real> P0(Im + 3, Real(1)), P1(Jm + 3, Real(1)), P2(Km + 3, Real(1));
        for (int i = 1; i < (int)P0.size(); ++i) P0[i] = P0[i - 1] * mu0;
        for (int i = 1; i < (int)P1.size(); ++i) P1[i] = P1[i - 1] * mu1;
        for (int i = 1; i < (int)P2.size(); ++i) P2[i] = P2[i - 1] * mu2;
        // raw moment E[(x1-RP)^nP (x2-RQ)^nQ (x3-RS)^nS] = shift by mean + central
        auto mom = [&](int nP, int nQ, int nS) {
          Real s = 0;
          for (int k1 = 0; k1 <= nP; ++k1)
            for (int k2 = 0; k2 <= nQ; ++k2)
              for (int k3 = 0; k3 <= nS; ++k3)
                s += C[nP][k1] * C[nQ][k2] * C[nS][k3] * P0[nP - k1] * P1[nQ - k2] *
                     P2[nS - k3] * cm(k1, k2, k3);
          return s;
        };
        Real Th = 0, Thm = 0;
        for (int nP = 0; nP <= mP[dir]; ++nP)
          for (int nQ = 0; nQ <= mQ[dir]; ++nQ)
            for (int nS = 0; nS <= mS[dir]; ++nS) {
              const Real coef = TP[dir][nP] * TQ[dir][nQ] * TS[dir][nS];
              Th += coef * mom(nP, nQ, nS);
              // shifted coords: (x1-x2) = (x1-RP)-(x2-RQ)+DPQ, likewise (x1-x3).
              if (moment == 1) { // r12^2 = (x1-x2)^2
                const Real m = mom(nP + 2, nQ, nS) + mom(nP, nQ + 2, nS) +
                               dpq * dpq * mom(nP, nQ, nS) - 2 * mom(nP + 1, nQ + 1, nS) +
                               2 * dpq * mom(nP + 1, nQ, nS) - 2 * dpq * mom(nP, nQ + 1, nS);
                Thm += coef * m;
              } else if (moment == 2) { // r12.r13 = (x1-x2)(x1-x3), per direction
                const Real m = mom(nP + 2, nQ, nS) + (dpq + dps) * mom(nP + 1, nQ, nS) -
                               mom(nP + 1, nQ + 1, nS) - mom(nP + 1, nQ, nS + 1) -
                               dps * mom(nP, nQ + 1, nS) - dpq * mom(nP, nQ, nS + 1) +
                               mom(nP, nQ + 1, nS + 1) + dpq * dps * mom(nP, nQ, nS);
                Thm += coef * m;
              }
            }
        F[dir] = base * Th;
        Fr2[dir] = base * Thm;
      }
      Real contr;
      if (!moment)
        contr = F[0] * F[1] * F[2];
      else
        contr = Fr2[0] * F[1] * F[2] + F[0] * Fr2[1] * F[2] + F[0] * F[1] * Fr2[2];
      acc += n12.weight * n13.weight * prefac * contr;
    }
  }
  return Kad * Kbe * Kcf * acc;
}

/// Unnormalised three-electron integral of six Cartesian Gaussians with
/// arbitrary inter-electronic operators on the 1-2 and 1-3 pairs, given as node
/// lists (Coulomb -> coulomb_nodes, Gaussian geminal -> gaussian_nodes).
template <class Real>
Real three_electron_raw_nodes(const CartGauss<Real> &a, const CartGauss<Real> &b,
                              const CartGauss<Real> &c, const CartGauss<Real> &d,
                              const CartGauss<Real> &e, const CartGauss<Real> &f,
                              const std::vector<OpNode<Real>> &nodes12,
                              const std::vector<OpNode<Real>> &nodes13) {
  return te_core(a, b, c, d, e, f, nodes12, nodes13, 0);
}

/// Unnormalised three-electron integral with an r^2 moment on the 1-2 operator:
/// the 1-2 slot carries r12^2 exp(-t^2 r12^2). With Coulomb nodes on 1-2 this is
/// the linear operator r12 = r12^2 r12^{-1}; with a Gaussian-geminal node list
/// {(4 g_k g_l c_k c_l, sqrt(g_k+g_l))} it is (grad_1 f12).(grad_1 f12).
template <class Real>
Real three_electron_raw_moment12(const CartGauss<Real> &a, const CartGauss<Real> &b,
                                 const CartGauss<Real> &c, const CartGauss<Real> &d,
                                 const CartGauss<Real> &e, const CartGauss<Real> &f,
                                 const std::vector<OpNode<Real>> &nodes12,
                                 const std::vector<OpNode<Real>> &nodes13) {
  return te_core(a, b, c, d, e, f, nodes12, nodes13, 1);
}

/// Unnormalised three-electron integral with the cross moment r12 . r13 (vector
/// dot product) inserted. With Gaussian-geminal derivative nodes on both slots
/// it is the F12 cross gradient (grad_1 f12).(grad_1 f13).
template <class Real>
Real three_electron_raw_moment_cross(const CartGauss<Real> &a, const CartGauss<Real> &b,
                                     const CartGauss<Real> &c, const CartGauss<Real> &d,
                                     const CartGauss<Real> &e, const CartGauss<Real> &f,
                                     const std::vector<OpNode<Real>> &nodes12,
                                     const std::vector<OpNode<Real>> &nodes13) {
  return te_core(a, b, c, d, e, f, nodes12, nodes13, 2);
}

/// Unnormalised three-electron Coulomb integral (both operators = r^{-1}).
template <class Real>
Real three_electron_raw(const CartGauss<Real> &a, const CartGauss<Real> &b,
                        const CartGauss<Real> &c, const CartGauss<Real> &d,
                        const CartGauss<Real> &e, const CartGauss<Real> &f,
                        const TGrid<Real> &grid) {
  const auto nd = coulomb_nodes(grid);
  return three_electron_raw_nodes(a, b, c, d, e, f, nd, nd);
}

/// s-type normalisation product N0(z) = (2 z / pi)^{3/4} (l=0) generalised:
/// per function N = N0 * gamma(lx) gamma(ly) gamma(lz), gamma(l)=[(2l-1)!!]^{-1/2},
/// N0 = pi^{-3/4} 2^{L+3/4} z^{L/2+3/4}, L = lx+ly+lz.
template <class Real> Real te_norm(const CartGauss<Real> &g) {
  const Real pi = pi_v<Real>();
  const int L = g.l[0] + g.l[1] + g.l[2];
  Real N0 = std::pow(pi, Real(-0.75)) * std::pow(Real(2), Real(L) + Real(0.75)) *
            std::pow(g.alpha, Real(L) / 2 + Real(0.75));
  Real gam = 1;
  for (int d = 0; d < 3; ++d) {
    Real df = 1; // (2l-1)!!
    for (int k = 2 * g.l[d] - 1; k > 0; k -= 2) df *= k;
    gam /= sqrt_(df);
  }
  return N0 * gam;
}

} // namespace detail

/// Three-electron Coulomb integral G_abcdef over six NORMALISED Cartesian
/// primitive Gaussians, via the 2D (t,s) quadrature. `grid` is an ordinary
/// Coulomb t-grid (make_tgrid(coulomb())), used for both auxiliary dimensions.
template <class Real>
Real three_electron_coulomb(const CartGauss<Real> &a, const CartGauss<Real> &b,
                            const CartGauss<Real> &c, const CartGauss<Real> &d,
                            const CartGauss<Real> &e, const CartGauss<Real> &f,
                            const TGrid<Real> &grid) {
  const Real N = detail::te_norm(a) * detail::te_norm(b) * detail::te_norm(c) *
                 detail::te_norm(d) * detail::te_norm(e) * detail::te_norm(f);
  return N * detail::three_electron_raw(a, b, c, d, e, f, grid);
}

/// Three-electron integral over six NORMALISED Cartesian Gaussians with
/// arbitrary inter-electronic operators on the 1-2 and 1-3 pairs, given as node
/// lists (detail::coulomb_nodes for r^{-1}, detail::gaussian_nodes for a
/// Gaussian-geminal expansion sum_k c_k e^{-g_k r^2}). This covers the
/// explicitly-correlated / transcorrelated three-electron integrals: r12^{-1}
/// r13^{-1}, f12 r13^{-1}, f12 f13, ... where f is a Gaussian geminal (a
/// Gaussian geminal needs no t-integration, so its slot is just its fixed
/// nodes).
template <class Real>
Real three_electron(const CartGauss<Real> &a, const CartGauss<Real> &b,
                    const CartGauss<Real> &c, const CartGauss<Real> &d,
                    const CartGauss<Real> &e, const CartGauss<Real> &f,
                    const std::vector<detail::OpNode<Real>> &op12,
                    const std::vector<detail::OpNode<Real>> &op13) {
  const Real N = detail::te_norm(a) * detail::te_norm(b) * detail::te_norm(c) *
                 detail::te_norm(d) * detail::te_norm(e) * detail::te_norm(f);
  return N * detail::three_electron_raw_nodes(a, b, c, d, e, f, op12, op13);
}

/// Three-electron integral with an r^2 moment on the 1-2 operator (op12), over
/// six NORMALISED Cartesian Gaussians. The linear operator r12 = r12^2 r12^{-1}
/// uses coulomb_nodes for op12; the F12 (grad_1 f12).(grad_1 f12) uses the
/// Gaussian nodes {(4 g_k g_l c_k c_l, sqrt(g_k+g_l))}.
template <class Real>
Real three_electron_moment12(const CartGauss<Real> &a, const CartGauss<Real> &b,
                             const CartGauss<Real> &c, const CartGauss<Real> &d,
                             const CartGauss<Real> &e, const CartGauss<Real> &f,
                             const std::vector<detail::OpNode<Real>> &op12,
                             const std::vector<detail::OpNode<Real>> &op13) {
  const Real N = detail::te_norm(a) * detail::te_norm(b) * detail::te_norm(c) *
                 detail::te_norm(d) * detail::te_norm(e) * detail::te_norm(f);
  return N * detail::three_electron_raw_moment12(a, b, c, d, e, f, op12, op13);
}

/// Three-electron integral with the cross moment r12 . r13 (vector dot product),
/// over six NORMALISED Cartesian Gaussians. With Gaussian derivative nodes on
/// both slots it is the F12 cross gradient (grad_1 f12).(grad_1 f13) -- a
/// commutator / B-matrix ingredient coupling the two inter-electronic distances.
template <class Real>
Real three_electron_moment_cross(const CartGauss<Real> &a, const CartGauss<Real> &b,
                                 const CartGauss<Real> &c, const CartGauss<Real> &d,
                                 const CartGauss<Real> &e, const CartGauss<Real> &f,
                                 const std::vector<detail::OpNode<Real>> &op12,
                                 const std::vector<detail::OpNode<Real>> &op13) {
  const Real N = detail::te_norm(a) * detail::te_norm(b) * detail::te_norm(c) *
                 detail::te_norm(d) * detail::te_norm(e) * detail::te_norm(f);
  return N * detail::three_electron_raw_moment_cross(a, b, c, d, e, f, op12, op13);
}

/// Three-body energy from a set of normalised Cartesian Gaussians and an AO
/// density matrix D (n x n, row-major): the fully-contracted three-electron
/// integral
///   E = sum_{abcdef} G_{abcdef} D_{ad} D_{be} D_{cf},
/// electron 1 = (a,d), 2 = (b,e), 3 = (c,f), with the operators op12 (1-2) and
/// op13 (1-3) given as node lists. Matrix-level: density in, scalar out; the
/// individual sextet stays internal. This is the mean-field 3-body contribution
/// transcorrelated / F12 methods build. O(n^6) reference (the sextet loop);
/// production would fold the density in earlier and screen.
template <class Real>
Real three_electron_energy(const std::vector<CartGauss<Real>> &basis,
                           const std::vector<Real> &D,
                           const std::vector<detail::OpNode<Real>> &op12,
                           const std::vector<detail::OpNode<Real>> &op13) {
  const int n = static_cast<int>(basis.size());
  auto Dm = [&](int i, int j) { return D[static_cast<std::size_t>(i) * n + j]; };
  Real E = 0;
  for (int a = 0; a < n; ++a)
    for (int d = 0; d < n; ++d) {
      const Real Dad = Dm(a, d);
      if (Dad == Real(0)) continue;
      for (int b = 0; b < n; ++b)
        for (int e = 0; e < n; ++e) {
          const Real Dbe = Dm(b, e);
          if (Dbe == Real(0)) continue;
          for (int c = 0; c < n; ++c)
            for (int f = 0; f < n; ++f) {
              const Real Dcf = Dm(c, f);
              if (Dcf == Real(0)) continue;
              E += Dad * Dbe * Dcf *
                   three_electron(basis[a], basis[b], basis[c], basis[d], basis[e], basis[f],
                                  op12, op13);
            }
        }
    }
  return E;
}

/// Effective one-body (Fock) contribution of the three-body term: the
/// functional derivative F_pq = dE/dD_pq of the energy above, an n x n matrix
/// (row-major). Since the density enters E three times (once per electron), each
/// sextet scatters into the three slots its pairs occupy:
///   F_ad += G D_be D_cf,  F_be += G D_ad D_cf,  F_cf += G D_ad D_be.
/// Matrix in, matrix out. The cubic homogeneity of E gives sum_pq F_pq D_pq = 3E.
template <class Real>
std::vector<Real> three_electron_fock(const std::vector<CartGauss<Real>> &basis,
                                      const std::vector<Real> &D,
                                      const std::vector<detail::OpNode<Real>> &op12,
                                      const std::vector<detail::OpNode<Real>> &op13) {
  const int n = static_cast<int>(basis.size());
  auto Dm = [&](int i, int j) { return D[static_cast<std::size_t>(i) * n + j]; };
  std::vector<Real> F(static_cast<std::size_t>(n) * n, Real(0));
  auto Fadd = [&](int i, int j, Real v) { F[static_cast<std::size_t>(i) * n + j] += v; };
  for (int a = 0; a < n; ++a)
    for (int d = 0; d < n; ++d) {
      const Real Dad = Dm(a, d);
      for (int b = 0; b < n; ++b)
        for (int e = 0; e < n; ++e) {
          const Real Dbe = Dm(b, e);
          for (int c = 0; c < n; ++c)
            for (int f = 0; f < n; ++f) {
              const Real Dcf = Dm(c, f);
              if (Dad == Real(0) && Dbe == Real(0) && Dcf == Real(0)) continue;
              const Real G = three_electron(basis[a], basis[b], basis[c], basis[d], basis[e],
                                            basis[f], op12, op13);
              Fadd(a, d, G * Dbe * Dcf);
              Fadd(b, e, G * Dad * Dcf);
              Fadd(c, f, G * Dad * Dbe);
            }
        }
    }
  return F;
}

} // namespace intti
