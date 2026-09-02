// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Three-electron Coulomb integrals by numerical integration -- the same
// t-quadrature that resolves the two-electron 1/r, applied twice.
//
//   G_abcdef = <a(1)b(2)c(3)| r12^{-1} r13^{-1} |d(1)e(2)f(3)>.
//
// Both operators are replaced by their Gaussian integral transform, so the
// integral becomes a TWO-dimensional (t,s) quadrature whose integrand
// factorises over the Cartesian directions and is analytic at each node
// (Mehine, Losilla & Sundholm, Mol. Phys. 111, 2536 (2013) -- the same group's
// generalisation of the two-electron scheme libintti already uses; the
// scaled/Mobius grid and the delta tail carry over per dimension). Electron 1
// (density P = a d) is shared by both operators; electron 2 (Q = b e) couples
// through t, electron 3 (S = c f) through s.
//
// Arbitrary angular momentum: the Cartesian GTO product is expanded on the pair
// centre (T coefficients, Gaussian product theorem) and the angular dependence
// of the (t,s) integrand is the polynomial Phi_{nP nQ nS}(Lambda_Q, Lambda_S,
// Xi_PQ, Xi_PS) built by the recursion (Eqs. 19-20 of the paper), with
// Xi_PQ = R_Q - R_P, Xi_PS = R_S - R_P per direction.

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

/// Bivariate polynomials Phi_{nP nQ nS} in (u = Xi_PQ, v = Xi_PS) at one (t,s)
/// node. Returns a flat table indexed [(nP*(mQ+1)+nQ)*(mS+1)+nS], each entry a
/// (Dg+1)x(Dg+1) coefficient array, Dg = mP+mQ+mS, coeff[iu*(Dg+1)+iv].
template <class Real>
std::vector<std::vector<Real>> te_build_phi(int mP, int mQ, int mS, Real LQ, Real LS,
                                            Real aP, Real aQ, Real aS) {
  const int Dg = mP + mQ + mS, W = Dg + 1;
  const int nphi = (mP + 1) * (mQ + 1) * (mS + 1);
  std::vector<std::vector<Real>> phi(nphi, std::vector<Real>(static_cast<std::size_t>(W) * W,
                                                             Real(0)));
  auto idx = [&](int nP, int nQ, int nS) { return (nP * (mQ + 1) + nQ) * (mS + 1) + nS; };
  auto at = [&](std::vector<Real> &p, int iu, int iv) -> Real & { return p[iu * W + iv]; };
  auto du = [&](const std::vector<Real> &p, std::vector<Real> &r) { // d/du
    std::fill(r.begin(), r.end(), Real(0));
    for (int iu = 1; iu < W; ++iu)
      for (int iv = 0; iv < W; ++iv) r[(iu - 1) * W + iv] += Real(iu) * p[iu * W + iv];
  };
  auto dv = [&](const std::vector<Real> &p, std::vector<Real> &r) {
    std::fill(r.begin(), r.end(), Real(0));
    for (int iu = 0; iu < W; ++iu)
      for (int iv = 1; iv < W; ++iv) r[iu * W + iv - 1] += Real(iv) * p[iu * W + iv];
  };
  // combine helper writing into dst: dst = (1/den)[ terms ]
  phi[idx(0, 0, 0)][0] = 1;
  std::vector<Real> ta(static_cast<std::size_t>(W) * W), tb(ta);
  // increase nP
  for (int i = 0; i < mP; ++i) {
    const auto &cur = phi[idx(i, 0, 0)];
    auto &out = phi[idx(i + 1, 0, 0)];
    du(cur, ta);
    dv(cur, tb);
    for (int iu = 0; iu < W; ++iu)
      for (int iv = 0; iv < W; ++iv) {
        Real val = -(ta[iu * W + iv] + tb[iu * W + iv]);
        // +2(LQ u + LS v) cur
        if (iu >= 1) val += 2 * LQ * cur[(iu - 1) * W + iv];
        if (iv >= 1) val += 2 * LS * cur[iu * W + (iv - 1)];
        out[iu * W + iv] = val / (2 * aP);
      }
    if (i >= 1)
      for (std::size_t k = 0; k < out.size(); ++k)
        out[k] += Real(i) / (2 * aP) * phi[idx(i - 1, 0, 0)][k];
  }
  // increase nQ
  for (int i = 0; i <= mP; ++i)
    for (int j = 0; j < mQ; ++j) {
      const auto &cur = phi[idx(i, j, 0)];
      auto &out = phi[idx(i, j + 1, 0)];
      du(cur, ta);
      for (int iu = 0; iu < W; ++iu)
        for (int iv = 0; iv < W; ++iv) {
          Real val = ta[iu * W + iv];
          if (iu >= 1) val -= 2 * LQ * cur[(iu - 1) * W + iv]; // -2 LQ u cur
          out[iu * W + iv] = val / (2 * aQ);
        }
      if (j >= 1)
        for (std::size_t k = 0; k < out.size(); ++k)
          out[k] += Real(j) / (2 * aQ) * phi[idx(i, j - 1, 0)][k];
    }
  // increase nS
  for (int i = 0; i <= mP; ++i)
    for (int j = 0; j <= mQ; ++j)
      for (int k = 0; k < mS; ++k) {
        const auto &cur = phi[idx(i, j, k)];
        auto &out = phi[idx(i, j, k + 1)];
        dv(cur, tb);
        for (int iu = 0; iu < W; ++iu)
          for (int iv = 0; iv < W; ++iv) {
            Real val = tb[iu * W + iv];
            if (iv >= 1) val -= 2 * LS * cur[iu * W + (iv - 1)]; // -2 LS v cur
            out[iu * W + iv] = val / (2 * aS);
          }
        if (k >= 1)
          for (std::size_t q = 0; q < out.size(); ++q)
            out[q] += Real(k) / (2 * aS) * phi[idx(i, j, k - 1)][q];
      }
  return phi;
}

/// Unnormalised three-electron Coulomb integral of six Cartesian Gaussians.
template <class Real>
Real three_electron_raw(const CartGauss<Real> &a, const CartGauss<Real> &b,
                        const CartGauss<Real> &c, const CartGauss<Real> &d,
                        const CartGauss<Real> &e, const CartGauss<Real> &f,
                        const TGrid<Real> &grid) {
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
  Real RPQ2 = 0, RPS2 = 0, XiPQ[3], XiPS[3];
  for (int i = 0; i < 3; ++i) {
    RPQ2 += (RP[i] - RQ[i]) * (RP[i] - RQ[i]);
    RPS2 += (RP[i] - RS[i]) * (RP[i] - RS[i]);
    XiPQ[i] = RQ[i] - RP[i];
    XiPS[i] = RS[i] - RP[i];
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
  const int MP = std::max({mP[0], mP[1], mP[2]});
  const int MQ = std::max({mQ[0], mQ[1], mQ[2]});
  const int MS = std::max({mS[0], mS[1], mS[2]});
  const int W = MP + MQ + MS + 1;
  const Real pi = pi_v<Real>();
  const Real pi92 = pi * pi * pi * pi * sqrt_(pi);
  const int nt = grid.n();
  Real acc = 0;
  for (int it = 0; it < nt; ++it) {
    const Real t2 = grid.t[it] * grid.t[it];
    const Real LQ = t2 * aQ / (t2 + aQ);
    for (int is = 0; is < nt; ++is) {
      const Real s2 = grid.t[is] * grid.t[is];
      const Real LS = s2 * aS / (s2 + aS);
      const Real denom = (aP + LQ + LS) * (aQ + t2) * (aS + s2);
      const Real M = pi92 * exp_(-LQ * RPQ2 - LS * RPS2) / (denom * sqrt_(denom));
      auto phi = te_build_phi(MP, MQ, MS, LQ, LS, aP, aQ, aS);
      Real theta = 1;
      for (int dir = 0; dir < 3; ++dir) {
        Real Th = 0;
        for (int nP = 0; nP <= mP[dir]; ++nP)
          for (int nQ = 0; nQ <= mQ[dir]; ++nQ)
            for (int nS = 0; nS <= mS[dir]; ++nS) {
              const auto &p = phi[(nP * (MQ + 1) + nQ) * (MS + 1) + nS];
              // evaluate poly at (u=XiPQ, v=XiPS)
              Real val = 0, up = 1;
              for (int iu = 0; iu < W; ++iu) {
                Real vp = 1;
                for (int iv = 0; iv < W; ++iv) {
                  val += p[iu * W + iv] * up * vp;
                  vp *= XiPS[dir];
                }
                up *= XiPQ[dir];
              }
              Th += TP[dir][nP] * TQ[dir][nQ] * TS[dir][nS] * val;
            }
        theta *= Th;
      }
      acc += grid.w[it] * grid.w[is] * M * theta;
    }
  }
  return Kad * Kbe * Kcf * acc;
}

/// s-type normalisation product N0(z) = (2 z / pi)^{3/4} (l=0) generalised:
/// per function N = N0 * gamma(lx) gamma(ly) gamma(lz), gamma(l)=[(2l-1)!!]^{-1/2},
/// N0 = pi^{-3/4} 2^{L+3/4} z^{L/2+3/4}, L = lx+ly+lz.
template <class Real> Real te_norm(const CartGauss<Real> &g) {
  const Real pi = pi_v<Real>();
  const int L = g.l[0] + g.l[1] + g.l[2];
  auto ipow = [](Real x, int n) {
    Real r = 1;
    for (int i = 0; i < n; ++i) r *= x;
    return r;
  };
  Real N0 = std::pow(pi, Real(-0.75)) * std::pow(Real(2), Real(L) + Real(0.75)) *
            std::pow(g.alpha, Real(L) / 2 + Real(0.75));
  Real gam = 1;
  for (int d = 0; d < 3; ++d) {
    Real df = 1; // (2l-1)!!
    for (int k = 2 * g.l[d] - 1; k > 0; k -= 2) df *= k;
    gam /= sqrt_(df);
  }
  (void)ipow;
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

} // namespace intti
