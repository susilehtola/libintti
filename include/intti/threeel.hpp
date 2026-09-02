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

/// Unnormalised three-electron integral of six Cartesian Gaussians with
/// arbitrary inter-electronic operators on the 1-2 and 1-3 pairs, given as node
/// lists (Coulomb -> coulomb_nodes, Gaussian geminal -> gaussian_nodes).
template <class Real>
Real three_electron_raw_nodes(const CartGauss<Real> &a, const CartGauss<Real> &b,
                              const CartGauss<Real> &c, const CartGauss<Real> &d,
                              const CartGauss<Real> &e, const CartGauss<Real> &f,
                              const std::vector<OpNode<Real>> &nodes12,
                              const std::vector<OpNode<Real>> &nodes13) {
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
  const int n12 = static_cast<int>(nodes12.size()), n13 = static_cast<int>(nodes13.size());
  Real acc = 0;
  for (int it = 0; it < n12; ++it) {
    const Real t2 = nodes12[it].t * nodes12[it].t;
    const Real LQ = t2 * aQ / (t2 + aQ);
    for (int is = 0; is < n13; ++is) {
      const Real s2 = nodes13[is].t * nodes13[is].t;
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
      acc += nodes12[it].weight * nodes13[is].weight * M * theta;
    }
  }
  return Kad * Kbe * Kcf * acc;
}

/// Like te_build_phi, but also returns dphi = d Phi / d Lambda_Q (same layout),
/// via the recursion differentiated w.r.t. Lambda_Q (explicit +2u/-2u terms).
template <class Real>
void te_build_phi_dLQ(int mP, int mQ, int mS, Real LQ, Real LS, Real aP, Real aQ, Real aS,
                      std::vector<std::vector<Real>> &phi,
                      std::vector<std::vector<Real>> &dphi) {
  const int Dg = mP + mQ + mS, W = Dg + 1;
  const int nphi = (mP + 1) * (mQ + 1) * (mS + 1);
  const std::size_t sz = static_cast<std::size_t>(W) * W;
  phi.assign(nphi, std::vector<Real>(sz, Real(0)));
  dphi.assign(nphi, std::vector<Real>(sz, Real(0)));
  auto id = [&](int nP, int nQ, int nS) { return (nP * (mQ + 1) + nQ) * (mS + 1) + nS; };
  auto du = [&](const std::vector<Real> &p, std::vector<Real> &r) {
    std::fill(r.begin(), r.end(), Real(0));
    for (int iu = 1; iu < W; ++iu)
      for (int iv = 0; iv < W; ++iv) r[(iu - 1) * W + iv] += Real(iu) * p[iu * W + iv];
  };
  auto dv = [&](const std::vector<Real> &p, std::vector<Real> &r) {
    std::fill(r.begin(), r.end(), Real(0));
    for (int iu = 0; iu < W; ++iu)
      for (int iv = 1; iv < W; ++iv) r[iu * W + iv - 1] += Real(iv) * p[iu * W + iv];
  };
  phi[id(0, 0, 0)][0] = 1;
  std::vector<Real> ta(sz), tb(sz), tc(sz), td(sz);
  for (int i = 0; i < mP; ++i) {
    const auto &c = phi[id(i, 0, 0)];
    const auto &dc = dphi[id(i, 0, 0)];
    auto &o = phi[id(i + 1, 0, 0)];
    auto &dobj = dphi[id(i + 1, 0, 0)];
    du(c, ta); dv(c, tb); du(dc, tc); dv(dc, td);
    for (int iu = 0; iu < W; ++iu)
      for (int iv = 0; iv < W; ++iv) {
        Real v = -(ta[iu * W + iv] + tb[iu * W + iv]);
        Real dvv = -(tc[iu * W + iv] + td[iu * W + iv]);
        if (iu >= 1) { v += 2 * LQ * c[(iu - 1) * W + iv]; dvv += 2 * c[(iu - 1) * W + iv] + 2 * LQ * dc[(iu - 1) * W + iv]; }
        if (iv >= 1) { v += 2 * LS * c[iu * W + iv - 1]; dvv += 2 * LS * dc[iu * W + iv - 1]; }
        o[iu * W + iv] = v / (2 * aP);
        dobj[iu * W + iv] = dvv / (2 * aP);
      }
    if (i >= 1)
      for (std::size_t k = 0; k < sz; ++k) {
        o[k] += Real(i) / (2 * aP) * phi[id(i - 1, 0, 0)][k];
        dobj[k] += Real(i) / (2 * aP) * dphi[id(i - 1, 0, 0)][k];
      }
  }
  for (int i = 0; i <= mP; ++i)
    for (int j = 0; j < mQ; ++j) {
      const auto &c = phi[id(i, j, 0)];
      const auto &dc = dphi[id(i, j, 0)];
      auto &o = phi[id(i, j + 1, 0)];
      auto &dobj = dphi[id(i, j + 1, 0)];
      du(c, ta); du(dc, tc);
      for (int iu = 0; iu < W; ++iu)
        for (int iv = 0; iv < W; ++iv) {
          Real v = ta[iu * W + iv], dvv = tc[iu * W + iv];
          if (iu >= 1) { v -= 2 * LQ * c[(iu - 1) * W + iv]; dvv += -2 * c[(iu - 1) * W + iv] - 2 * LQ * dc[(iu - 1) * W + iv]; }
          o[iu * W + iv] = v / (2 * aQ);
          dobj[iu * W + iv] = dvv / (2 * aQ);
        }
      if (j >= 1)
        for (std::size_t k = 0; k < sz; ++k) {
          o[k] += Real(j) / (2 * aQ) * phi[id(i, j - 1, 0)][k];
          dobj[k] += Real(j) / (2 * aQ) * dphi[id(i, j - 1, 0)][k];
        }
    }
  for (int i = 0; i <= mP; ++i)
    for (int j = 0; j <= mQ; ++j)
      for (int k = 0; k < mS; ++k) {
        const auto &c = phi[id(i, j, k)];
        const auto &dc = dphi[id(i, j, k)];
        auto &o = phi[id(i, j, k + 1)];
        auto &dobj = dphi[id(i, j, k + 1)];
        dv(c, tb); dv(dc, td);
        for (int iu = 0; iu < W; ++iu)
          for (int iv = 0; iv < W; ++iv) {
            Real v = tb[iu * W + iv], dvv = td[iu * W + iv];
            if (iv >= 1) { v -= 2 * LS * c[iu * W + iv - 1]; dvv += -2 * LS * dc[iu * W + iv - 1]; }
            o[iu * W + iv] = v / (2 * aS);
            dobj[iu * W + iv] = dvv / (2 * aS);
          }
        if (k >= 1)
          for (std::size_t q = 0; q < sz; ++q) {
            o[q] += Real(k) / (2 * aS) * phi[id(i, j, k - 1)][q];
            dobj[q] += Real(k) / (2 * aS) * dphi[id(i, j, k - 1)][q];
          }
      }
}

/// Unnormalised three-electron integral with an r^2 moment on the 1-2 operator:
/// the 1-2 slot carries r12^2 exp(-t^2 r12^2) = -d/d(t^2) exp(-t^2 r12^2), so the
/// integrand is -d/du(M Theta), u = t^2. With Coulomb nodes on 1-2 this is the
/// linear operator r12 = r12^2 r12^{-1}; with a Gaussian-geminal node list
/// {(4 g_k g_l c_k c_l, sqrt(g_k+g_l))} it is (grad_1 f12).(grad_1 f12).
template <class Real>
Real three_electron_raw_moment12(const CartGauss<Real> &a, const CartGauss<Real> &b,
                                 const CartGauss<Real> &c, const CartGauss<Real> &d,
                                 const CartGauss<Real> &e, const CartGauss<Real> &f,
                                 const std::vector<OpNode<Real>> &nodes12,
                                 const std::vector<OpNode<Real>> &nodes13) {
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
  const int MP = std::max({mP[0], mP[1], mP[2]}), MQ = std::max({mQ[0], mQ[1], mQ[2]}),
            MS = std::max({mS[0], mS[1], mS[2]}), W = MP + MQ + MS + 1;
  const Real pi = pi_v<Real>(), pi92 = pi * pi * pi * pi * sqrt_(pi);
  std::vector<std::vector<Real>> phi, dphi;
  auto evalp = [&](const std::vector<Real> &p, int dir) {
    Real val = 0, up = 1;
    for (int iu = 0; iu < W; ++iu) {
      Real vp = 1;
      for (int iv = 0; iv < W; ++iv) {
        val += p[iu * W + iv] * up * vp;
        vp *= XiPS[dir];
      }
      up *= XiPQ[dir];
    }
    return val;
  };
  Real acc = 0;
  for (const auto &n12 : nodes12) {
    const Real u1 = n12.t * n12.t;
    const Real LQ = u1 * aQ / (u1 + aQ), LQp = aQ * aQ / ((u1 + aQ) * (u1 + aQ));
    for (const auto &n13 : nodes13) {
      const Real s2 = n13.t * n13.t, LS = s2 * aS / (s2 + aS);
      const Real sum = aP + LQ + LS, denom = sum * (aQ + u1) * (aS + s2);
      const Real M = pi92 * exp_(-LQ * RPQ2 - LS * RPS2) / (denom * sqrt_(denom));
      const Real dM = M * (-RPQ2 * LQp - Real(1.5) * (LQp / sum + 1 / (aQ + u1)));
      te_build_phi_dLQ(MP, MQ, MS, LQ, LS, aP, aQ, aS, phi, dphi);
      Real Th[3], dTh[3];
      for (int dir = 0; dir < 3; ++dir) {
        Real t = 0, dt = 0;
        for (int nP = 0; nP <= mP[dir]; ++nP)
          for (int nQ = 0; nQ <= mQ[dir]; ++nQ)
            for (int nS = 0; nS <= mS[dir]; ++nS) {
              const int q = (nP * (MQ + 1) + nQ) * (MS + 1) + nS;
              const Real coef = TP[dir][nP] * TQ[dir][nQ] * TS[dir][nS];
              t += coef * evalp(phi[q], dir);
              dt += coef * evalp(dphi[q], dir);
            }
        Th[dir] = t;
        dTh[dir] = dt;
      }
      const Real Theta = Th[0] * Th[1] * Th[2];
      const Real dTheta_dLQ =
          dTh[0] * Th[1] * Th[2] + Th[0] * dTh[1] * Th[2] + Th[0] * Th[1] * dTh[2];
      // integrand = -d/du1 (M Theta) = -(dM Theta + M LQ' dTheta/dLQ)
      const Real integ = -(dM * Theta + M * LQp * dTheta_dLQ);
      acc += n12.weight * n13.weight * integ;
    }
  }
  return Kad * Kbe * Kcf * acc;
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

} // namespace intti
