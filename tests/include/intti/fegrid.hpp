// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Tensorial finite-element (FE) real-space grid + the t-adapted DAGE Coulomb
// operator -- the second, operator-agnostic integral route (roadmap M-FE),
// sharing the t-quadrature kernel core (tgrid.hpp) with the GTO/STO engine.
//
// A 1D axis is split into elements; on each element a function is a local
// polynomial through the element's Gauss-Legendre nodes (barycentric Lagrange).
// Each element carries its OWN order (np), so the mesh is hp-adaptive: uniform
// meshes come from make_fegrid1d(ne, np, L); adapted meshes from
// make_fegrid1d_hp(gaussians, eps), which places elements/orders so every 1D
// Gaussian is resolved to eps (a cheap 1D problem, since a 1D Gaussian is entire
// and barycentric interpolation converges exponentially in the order). The 3D
// grid is the tensor product. The Coulomb (or any 1/r-family) potential of a
// density rho on the grid is the "direct approach" (DAGE):
//   V(r1) = int rho(r2)/|r1-r2| dr2 = sum_t w_t int rho(r2) e^{-t^2|r1-r2|^2} dr2,
// and since the Gaussian kernel factorizes per Cartesian axis, each t-node is
// three successive 1D convolutions of the density tensor (along z, y, x). Each
// 1D convolution is t-ADAPTED: substitute u2 = u1 - v/t so the kernel e^{-v^2}
// is flat for every t, and evaluate the (piecewise-polynomial) source at the
// off-grid points u1 - v/t by barycentric interpolation within the source
// element (far elements screen via the clamped v-range). This is machine-exact
// per axis for all t, so NO delta-tail is needed and general (non-separable)
// densities are handled directly. The t-grid selects the kernel: make_tgrid(
// coulomb()) for 1/r, make_tgrid(yukawa(kappa)) for e^{-kappa r}/r (the
// Helmholtz Green's function apply, up to 1/4pi).
//
// This first version is a host (serial) reference implementation; the API is
// kept tensor-at-a-time so a Kokkos/GPU port (team-scratch over lines) can slot
// in later. Matrix/grid-level only -- no per-quartet surface.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "intti/device.hpp"
#include "intti/kernel.hpp"
#include "intti/tgrid.hpp"

namespace intti {

namespace detail {
/// Gauss-Legendre nodes/weights on [a,b] (Newton on the Legendre polynomial).
template <class Real>
void fe_gauss_legendre(int n, Real a, Real b, std::vector<Real> &x, std::vector<Real> &w) {
  x.assign(n, Real(0));
  w.assign(n, Real(0));
  const Real pi = pi_v<Real>();
  for (int i = 0; i < n; ++i) {
    Real xi = std::cos(pi * (i + Real(0.75)) / (n + Real(0.5))), dp = 0;
    for (int it = 0; it < 100; ++it) {
      Real p0 = 1, p1 = xi;
      for (int k = 2; k <= n; ++k) {
        Real p2 = ((2 * k - 1) * xi * p1 - (k - 1) * p0) / k;
        p0 = p1;
        p1 = p2;
      }
      dp = n * (xi * p1 - p0) / (xi * xi - 1);
      Real dx = -p1 / dp;
      xi += dx;
      if (std::abs(dx) < Real(1e-15)) break;
    }
    x[i] = Real(0.5) * (a + b) + Real(0.5) * (b - a) * xi;
    w[i] = (b - a) / ((1 - xi * xi) * dp * dp);
  }
}
} // namespace detail

/// 1D tensorial finite-element grid: `ne` elements on the boundaries `be`, each
/// element e carrying `nps[e]` Gauss-Legendre nodes (a degree-(nps[e]-1) local
/// polynomial). Uniform meshes have all nps[e] equal; hp meshes vary them. The
/// 3D grid is the tensor product along x, y, z (N = sum_e nps[e] points/axis).
template <class Real> struct FEGrid1D {
  int ne{0}, N{0};
  Real L{0};                    // nominal half-width (uniform meshes); |domain|/2
  std::vector<Real> be;         // element boundaries (ne+1)
  std::vector<int> nps, noff;   // per-element order (ne) and node offset (ne+1)
  std::vector<Real> rn, bw;     // per-element ref nodes / bary weights (concat by noff)
  std::vector<Real> xnode, xw;  // global node coords / quadrature weights (N)
  Real cen(int e) const { return Real(0.5) * (be[e] + be[e + 1]); }
  Real hw(int e) const { return Real(0.5) * (be[e + 1] - be[e]); }
  int np(int e) const { return nps[e]; }
};

namespace detail {
/// Build a 1D FE grid from explicit element boundaries and per-element orders.
template <class Real>
FEGrid1D<Real> make_fegrid1d_elements(const std::vector<Real> &be,
                                      const std::vector<int> &orders) {
  FEGrid1D<Real> g;
  g.ne = static_cast<int>(orders.size());
  g.be = be;
  g.L = Real(0.5) * (be.back() - be.front());
  g.nps = orders;
  g.noff.assign(g.ne + 1, 0);
  for (int e = 0; e < g.ne; ++e) g.noff[e + 1] = g.noff[e] + orders[e];
  g.N = g.noff[g.ne];
  g.xnode.resize(g.N);
  g.xw.resize(g.N);
  for (int e = 0; e < g.ne; ++e) {
    const int p = orders[e];
    std::vector<Real> rn, rw;
    fe_gauss_legendre<Real>(p, Real(-1), Real(1), rn, rw);
    std::vector<Real> bw(p, Real(1));
    for (int k = 0; k < p; ++k)
      for (int j = 0; j < p; ++j)
        if (j != k) bw[k] /= (rn[k] - rn[j]);
    for (int k = 0; k < p; ++k) {
      g.xnode[g.noff[e] + k] = g.cen(e) + g.hw(e) * rn[k];
      g.xw[g.noff[e] + k] = g.hw(e) * rw[k];
      g.rn.push_back(rn[k]);
      g.bw.push_back(bw[k]);
    }
  }
  return g;
}
} // namespace detail

/// Uniform 1D FE grid: ne equal elements on [-L, L], each of order np.
template <class Real> FEGrid1D<Real> make_fegrid1d(int ne, int np, Real L) {
  std::vector<Real> be(ne + 1);
  for (int e = 0; e <= ne; ++e) be[e] = -L + 2 * L * e / ne;
  return detail::make_fegrid1d_elements(be, std::vector<int>(ne, np));
}

/// A 1D Gaussian factor exp(-alpha (x - center)^2) that must be resolved on an
/// axis (the per-axis projection of a primitive; see make_fegrid1d_hp).
template <class Real> struct FEGaussian1D {
  Real alpha, center;
};

namespace detail {
/// Barycentric Lagrange value of element e's local polynomial (nodal values f,
/// length nps[e]) at reference coordinate xi in [-1,1].
template <class Real> Real fe_lag(const FEGrid1D<Real> &g, int e, const Real *f, Real xi) {
  Real num = 0, den = 0;
  const int p = g.nps[e], o = g.noff[e];
  for (int k = 0; k < p; ++k) {
    Real d = xi - g.rn[o + k];
    if (std::abs(d) < Real(1e-13)) return f[k];
    Real t = g.bw[o + k] / d;
    num += t * f[k];
    den += t;
  }
  return num / den;
}

/// Worst RELATIVE interpolation error, over every Gaussian and every polynomial
/// degree 0..pdeg, of (x - centre)^d exp(-alpha (x-centre)^2) on [a,b] at order p
/// (barycentric through p Gauss-Legendre nodes, nprobe+1 probes). pdeg > 0
/// resolves the AO-PRODUCT factor (a degree-up-to-2l polynomial times the
/// Gaussian), not just the envelope; each degree's error is normalised by that
/// factor's global peak (d/(2 alpha))^{d/2} e^{-d/2}, so eps is a consistent
/// relative tolerance across degrees.
template <class Real>
Real fe_hp_elem_err(const std::vector<FEGaussian1D<Real>> &gs, Real a, Real b, int p,
                    int pdeg = 0, int nprobe = 64) {
  std::vector<Real> nodes, gw;
  fe_gauss_legendre<Real>(p, a, b, nodes, gw);
  std::vector<Real> w(p, Real(1));
  for (int k = 0; k < p; ++k)
    for (int j = 0; j < p; ++j)
      if (j != k) w[k] /= (nodes[k] - nodes[j]);
  auto ipow = [](Real x, int n) { Real r = 1; for (int i = 0; i < n; ++i) r *= x; return r; };
  Real worst = 0;
  std::vector<Real> vals(p);
  for (const auto &gg : gs) {
    for (int d = 0; d <= pdeg; ++d) {
      const Real peak = d == 0 ? Real(1)
                               : std::pow(d / (2 * gg.alpha), Real(d) / 2) * std::exp(-Real(d) / 2);
      for (int k = 0; k < p; ++k) {
        const Real dx = nodes[k] - gg.center;
        vals[k] = ipow(dx, d) * std::exp(-gg.alpha * dx * dx);
      }
      for (int m = 0; m <= nprobe; ++m) {
        const Real xt = a + (b - a) * m / nprobe, dx = xt - gg.center;
        const Real exact = ipow(dx, d) * std::exp(-gg.alpha * dx * dx);
        Real num = 0, den = 0, interp = 0;
        bool hit = false;
        for (int k = 0; k < p; ++k) {
          Real dd = xt - nodes[k];
          if (std::abs(dd) < Real(1e-14)) { interp = vals[k]; hit = true; break; }
          Real t = w[k] / dd;
          num += t * vals[k];
          den += t;
        }
        if (!hit) interp = num / den;
        worst = std::max(worst, std::abs(interp - exact) / peak);
      }
    }
  }
  return worst;
}

/// Recursive hp refinement of [a,b]: try increasing order up to pmax; bisect (h)
/// only when no order reaches eps. Appends (right boundary, order) per element.
/// `pdeg` is the max AO-product polynomial degree to resolve (0 = envelope only).
template <class Real>
void fe_hp_refine(const std::vector<FEGaussian1D<Real>> &gs, Real a, Real b, Real eps,
                  int pmin, int pmax, int pdeg, std::vector<Real> &bnd, std::vector<int> &ord) {
  for (int p = pmin; p <= pmax; ++p)
    if (fe_hp_elem_err(gs, a, b, p, pdeg) <= eps) {
      bnd.push_back(b);
      ord.push_back(p);
      return;
    }
  const Real m = Real(0.5) * (a + b);
  fe_hp_refine(gs, a, m, eps, pmin, pmax, pdeg, bnd, ord);
  fe_hp_refine(gs, m, b, eps, pmin, pmax, pdeg, bnd, ord);
}
} // namespace detail

/// hp-adaptive 1D FE grid resolving every 1D Gaussian in `gaussians` to absolute
/// interpolation error `eps`, with minimal degrees of freedom. The domain covers
/// all Gaussians down to amplitude eps; element boundaries are seeded at the
/// centres and then hp-refined (order up to pmax, bisecting only when needed).
/// This is the cheap per-axis construction of the tensorial grid (roadmap M-FE).
/// `pdeg` is the maximum AO-product polynomial degree per axis to resolve (0 =
/// the Gaussian envelope only; 2*lmax for a basis of max angular momentum lmax).
///
namespace detail {
/// Local feature scale at x: the width of the tightest Gaussian that is still
/// alive there, 1/sqrt(a) over the g with a (x-c)^2 <= -ln(eps). Near a nucleus a
/// tight core function keeps this small, so nothing merges; away from every
/// centre only diffuse functions survive the amplitude test and the scale is
/// large.
template <class Real>
Real fe_local_scale(const std::vector<FEGaussian1D<Real>> &gs, Real x, Real eps,
                    Real fallback) {
  const Real cut = -std::log(eps);
  Real h = fallback;
  bool any = false;
  for (const auto &g : gs) {
    const Real d = x - g.center;
    if (g.alpha * d * d <= cut) {
      const Real w = Real(1) / std::sqrt(g.alpha);
      if (!any || w < h) h = w;
      any = true;
    }
  }
  return h;
}

/// Relative defect of the grid's QUADRATURE on one 1D Gaussian factor:
///   int (x-x0)^{2d} e^{-2 a (x-x0)^2} dx  =  (2d-1)!! / (4a)^d  sqrt(pi/(2a)),
/// which is closed form, so the defect is a single number per function and per
/// degree. A grid that reproduces the norm of a function describes it; a grid
/// that does not, does not. This is exactly the criterion the analytic-potential
/// route needs, because there the grid is used ONLY as a quadrature.
template <class Real>
Real fe_norm_defect(const FEGrid1D<Real> &g, Real a, Real x0, int d) {
  Real q = 0;
  for (int i = 0; i < g.N; ++i) {
    const Real u = g.xnode[i] - x0;
    Real t = std::exp(-2 * a * u * u);
    for (int k = 0; k < 2 * d; ++k) t *= u;
    q += g.xw[i] * t;
  }
  Real exact = std::sqrt(pi_v<Real>() / (2 * a));
  for (int m = 1; m <= d; ++m) exact *= Real(2 * m - 1) / (4 * a);
  return std::abs(q - exact) / exact;
}

} // namespace detail

/// Greedy, norm-driven 1D mesh construction. MEASURED WORSE THAN THE BISECTION
/// BUILDER ONCE THAT IS SEEDED CORRECTLY, and kept only so the comparison is on
/// record. On one water in def2-SVP, asked for eps = 1e-2:
///
///     atom-seeded bisection   N = 155   worst norm defect 4.3e-5
///     this, greedy            N = 864   worst norm defect 9.3e-3
///
/// The greedy converges to the tolerance it was given; the bisection builder
/// overshoots it by four orders and still uses a fifth of the points. Refining
/// where the norm defect is worst is a sound idea, but it loses to trying every
/// order pmin..pmax on a candidate element before splitting it -- a Gaussian is
/// entire, so p-refinement converges exponentially and h-refinement does not,
/// and the per-element order trial exploits that far better than a global
/// greedy loop can.
///
/// The bisection builder (make_fegrid1d_hp) seeds an element boundary at every
/// Gaussian centre it is handed. A caller with one Gaussian per shell PAIR per
/// axis therefore gets 3 npair boundaries, and since a Gaussian product is
/// smooth there is nothing at those points to resolve: the mesh comes out as one
/// element per seed, every element at pmin, and N grows as nshell^2 with no
/// relation to the requested accuracy. Measured on one water molecule that was
/// N = 1452 per axis for def2-SVP and N = 17383 for def2-QZVPPD; since the cost
/// is N^3, that is three to four orders of magnitude of pure waste.
///
/// This builds the mesh from the accuracy instead. Each iteration measures the
/// norm defect of every function, takes the WORST, and inserts points at that
/// function's own natural length scale, x0 + k/sqrt(a) for k = -NSCALE..NSCALE;
/// when those are already present it raises the order of the element holding the
/// centre, and bisects only when the order is exhausted. Nothing is placed that
/// some function did not ask for, which is what keeps the mesh compact when the
/// x, y and z projections of a polyatomic molecule crowd together.
template <class Real>
FEGrid1D<Real> make_fegrid1d_greedy(const std::vector<FEGaussian1D<Real>> &gaussians,
                                    Real eps, int pmin = 4, int pmax = 16, int pdeg = 0,
                                    int maxit = 4000) {
  constexpr int NSCALE = 6;
  pmin = std::max(pmin, pdeg + 2);
  pmax = std::max(pmax, pmin);
  Real xlo = gaussians.front().center, xhi = xlo;
  for (const auto &g : gaussians) {
    const Real w = std::sqrt((-std::log(eps) + pdeg) / g.alpha);
    xlo = std::min(xlo, g.center - w);
    xhi = std::max(xhi, g.center + w);
  }
  // distinct (alpha, centre): duplicates only cost time in the scan
  std::vector<FEGaussian1D<Real>> gs;
  for (const auto &g : gaussians) {
    bool dup = false;
    for (const auto &h : gs)
      if (std::abs(h.alpha - g.alpha) <= Real(1e-12) * h.alpha &&
          std::abs(h.center - g.center) <= Real(1e-10)) {
        dup = true;
        break;
      }
    if (!dup) gs.push_back(g);
  }

  std::vector<Real> bnd{xlo, xhi};
  std::vector<int> ord{pmin};
  auto insert_point = [&](Real x) {
    if (!(x > bnd.front() && x < bnd.back())) return false;
    for (std::size_t i = 0; i < bnd.size(); ++i) {
      const Real scale = std::max<Real>(Real(1e-10), std::abs(bnd[i]));
      if (std::abs(bnd[i] - x) <= Real(1e-6) * scale) return false;
    }
    const std::size_t k =
        std::lower_bound(bnd.begin(), bnd.end(), x) - bnd.begin(); // 1 <= k <= ne
    bnd.insert(bnd.begin() + k, x);
    ord.insert(ord.begin() + (k - 1), ord[k - 1]);
    return true;
  };

  FEGrid1D<Real> g = detail::make_fegrid1d_elements(bnd, ord);
  for (int it = 0; it < maxit; ++it) {
    Real worst = 0;
    int iworst = -1;
    for (std::size_t i = 0; i < gs.size(); ++i)
      for (int d = 0; d <= pdeg; ++d) {
        const Real e = detail::fe_norm_defect(g, gs[i].alpha, gs[i].center, d);
        if (e > worst) {
          worst = e;
          iworst = static_cast<int>(i);
        }
      }
    if (worst <= eps || iworst < 0) break;
    const Real a = gs[iworst].alpha, c = gs[iworst].center;
    const Real h = Real(1) / std::sqrt(a);
    // ORDER FIRST. A Gaussian is entire, so barycentric interpolation on an
    // element converges exponentially in the order while h-refinement at fixed
    // order converges only algebraically: raising p is strictly the better move
    // until p is spent. Refining h first measured 3x worse (N = 474 against 155
    // on one water in def2-SVP), which is a factor of 27 in points.
    const Real reach = NSCALE * h;
    bool bumped = false;
    for (int e = 0; e + 1 < static_cast<int>(bnd.size()); ++e)
      if (bnd[e + 1] > c - reach && bnd[e] < c + reach && ord[e] < pmax) {
        ++ord[e];
        bumped = true;
      }
    if (!bumped) {
      // p is spent over the function's support: now split, at its own natural
      // length scale rather than blindly in half
      bool added = false;
      for (int k = -NSCALE; k <= NSCALE && !added; ++k) added = insert_point(c + k * h);
      if (!added) {
        int e = 0;
        while (e + 2 < static_cast<int>(bnd.size()) && bnd[e + 1] < c) ++e;
        if (!insert_point(Real(0.5) * (bnd[e] + bnd[e + 1]))) break;
      }
    }
    g = detail::make_fegrid1d_elements(bnd, ord);
  }
  return g;
}

/// `seeds_in`, when non-null, replaces the default seeding at every Gaussian
/// centre. Seeds only choose the INITIAL subdivision -- accuracy comes from
/// fe_hp_refine, which tests every Gaussian on every element regardless -- so
/// changing them cannot make the mesh wrong, only bigger or smaller.
///
/// The default is much bigger. A caller with one Gaussian per SHELL PAIR per
/// axis (grid_for_basis) seeds 3 npair boundaries, and since a Gaussian product
/// is smooth there is nothing at those points for a boundary to resolve: the
/// mesh comes out as one element per seed, all at pmin, and N grows as nshell^2
/// with no relation to the accuracy asked for. Seeding at the distinct function
/// centres instead lets the refinement decide, which is what it is for.
template <class Real>
FEGrid1D<Real> make_fegrid1d_hp(const std::vector<FEGaussian1D<Real>> &gaussians,
                                Real eps, int pmin, int pmax, int pdeg,
                                const std::vector<Real> *seeds_in,
                                Real seed_merge = Real(1));

template <class Real>
FEGrid1D<Real> make_fegrid1d_hp(const std::vector<FEGaussian1D<Real>> &gaussians,
                                Real eps, int pmin = 4, int pmax = 16, int pdeg = 0) {
  return make_fegrid1d_hp<Real>(gaussians, eps, pmin, pmax, pdeg, nullptr);
}

template <class Real>
FEGrid1D<Real> make_fegrid1d_hp(const std::vector<FEGaussian1D<Real>> &gaussians,
                                Real eps, int pmin, int pmax, int pdeg,
                                const std::vector<Real> *seeds_in, Real seed_merge) {
  Real xlo = gaussians.front().center, xhi = xlo;
  for (const auto &g : gaussians) {
    // the product factor (x-c)^pdeg exp(-a(x-c)^2) extends a little past the
    // pure-Gaussian eps radius; widen the domain by the polynomial's reach.
    const Real w = std::sqrt((-std::log(eps) + pdeg) / g.alpha);
    xlo = std::min(xlo, g.center - w);
    xhi = std::max(xhi, g.center + w);
  }
  pmin = std::max(pmin, pdeg + 2); // room to represent the degree-pdeg polynomial
  pmax = std::max(pmax, pmin);
  std::vector<Real> seeds{xlo, xhi};
  if (seeds_in) {
    for (Real c : *seeds_in)
      if (c > xlo && c < xhi) seeds.push_back(c);
  } else {
    for (const auto &g : gaussians)
      if (g.center > xlo && g.center < xhi) seeds.push_back(g.center);
  }
  std::sort(seeds.begin(), seeds.end());
  seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());
  // Collapse seeds closer together than the local feature scale: two boundaries
  // that close cannot hold anything between them the neighbouring elements do
  // not already resolve. The test is local, which is what makes it safe -- it
  // never merges across a nucleus, because a tight core function keeps the
  // scale there far below any atomic separation.
  //
  // THIS BUYS 1-3%, NOT THE FACTOR IT LOOKS LIKE IT SHOULD. One shared 1D grid
  // serves all three axes, so the seed list is the union of the x, y and z
  // projections of every centre -- 3 natom values for a molecule at general
  // coordinates -- and it is tempting to read the resulting mesh growth as a
  // seeding artefact. It is not. Measured on water clusters at general
  // orientation, merging moves N by 247 -> 245 and 667 -> 660. The refinement
  // puts those boundaries back, because each atom's core Gaussian really does
  // make a narrow feature at its projected coordinate ON EVERY AXIS, and a
  // tensor-product grid has to resolve all 3 natom of them. The crowding is
  // real; it is the discretisation, not the seeding.
  if (seed_merge > Real(0) && seeds.size() > 2) {
    const Real span = seeds.back() - seeds.front();
    std::vector<Real> keep{seeds.front()};
    for (std::size_t i = 1; i + 1 < seeds.size(); ++i) {
      const Real h = detail::fe_local_scale(gaussians, seeds[i], eps, span);
      if (seeds[i] - keep.back() >= seed_merge * h) keep.push_back(seeds[i]);
    }
    keep.push_back(seeds.back());
    seeds.swap(keep);
  }
  std::vector<Real> bnd{xlo};
  std::vector<int> ord;
  for (std::size_t s = 0; s + 1 < seeds.size(); ++s)
    detail::fe_hp_refine(gaussians, seeds[s], seeds[s + 1], eps, pmin, pmax, pdeg, bnd, ord);
  return detail::make_fegrid1d_elements(bnd, ord);
}

namespace detail {
/// t-adapted 1D convolution of an FE line (nodal values `in`, length N) with the
/// Gaussian kernel exp(-t^2 (u1-u2)^2): out(u1) = int in(u2) exp(-t^2(u1-u2)^2)
/// du2. vg/vw are reference GL nodes/weights on [-1,1]; vmax clamps the flat-
/// kernel window. Handles per-element variable order via g.noff / g.nps.
template <class Real>
void fe_conv1d(const FEGrid1D<Real> &g, const std::vector<Real> &vg,
               const std::vector<Real> &vw, Real vmax, const Real *in, Real t, Real *out) {
  const int nv = static_cast<int>(vg.size());
  for (int jo = 0; jo < g.N; ++jo) {
    const Real u1 = g.xnode[jo];
    Real acc = 0;
    for (int eB = 0; eB < g.ne; ++eB) {
      Real lo = t * (u1 - g.be[eB + 1]), hi = t * (u1 - g.be[eB]);
      lo = std::max(lo, -vmax); hi = std::min(hi, vmax);
      if (hi <= lo) continue; // far element: outside the exp(-v^2) window
      const Real *fB = in + g.noff[eB];
      const Real vc = Real(0.5) * (lo + hi), vh = Real(0.5) * (hi - lo);
      Real s = 0;
      for (int gi = 0; gi < nv; ++gi) {
        const Real v = vc + vh * vg[gi];
        const Real xi = ((u1 - v / t) - g.cen(eB)) / g.hw(eB);
        s += vh * vw[gi] * fe_lag(g, eB, fB, xi) * std::exp(-v * v);
      }
      acc += s / t;
    }
    out[jo] = acc;
  }
}
} // namespace detail

/// DAGE: Coulomb-family potential of a density rho on the 3D FE grid,
/// V(r1) = sum_t w_t int rho(r2) kernel_t(|r1-r2|) dr2, via three t-adapted 1D
/// convolutions per t-node. rho and the returned V are N^3 tensors in
/// row-major (ix,iy,iz) order (N = grid.N). The kernel is set by `tgrid`
/// (coulomb -> 1/r; yukawa(kappa) -> e^{-kappa r}/r). `nv` is the inner
/// (flat-kernel) quadrature order. Handles general non-separable densities and
/// hp (variable-order) grids.
///
/// Kokkos-parallel over the N^2 independent lines of each axis sweep (double-
/// buffered A<->B), so it runs on the active Kokkos backend (OpenMP/CUDA/HIP);
/// the per-node math is identical to the serial reference. In the tensor-product
/// grid every line shares the same 1D mesh, so all work-items do identical work
/// (see the roadmap M-FE GPU note).
namespace detail {
/// Device core of fe_dage3d: the density R and the returned potential V are
/// device Views (the grid is staged internally). Lets a caller chain the whole
/// grid-RI pipeline on the device without a host round-trip.
template <class Real>
Kokkos::View<Real *> fe_dage3d_dev(const FEGrid1D<Real> &grid, const TGrid<Real> &tgrid,
                                   const Kokkos::View<Real *> &R, int nv, Real vmax) {
  const int N = grid.N, ne = grid.ne;
  const std::size_t N3 = static_cast<std::size_t>(N) * N * N;
  std::vector<Real> vgh, vwh;
  fe_gauss_legendre<Real>(nv, Real(-1), Real(1), vgh, vwh);
  auto xnode = to_device(grid.xnode, "fe::xnode");
  auto be = to_device(grid.be, "fe::be");
  auto nps = to_device(grid.nps, "fe::nps");
  auto noff = to_device(grid.noff, "fe::noff");
  auto rn = to_device(grid.rn, "fe::rn");
  auto bw = to_device(grid.bw, "fe::bw");
  auto vg = to_device(vgh, "fe::vg");
  auto vw = to_device(vwh, "fe::vw");
  Kokkos::View<Real *> A("fe::A", N3), B("fe::B", N3), V("fe::V", N3);
  Kokkos::deep_copy(V, Real(0));
  const int nline = N * N;
  // one axis sweep: transform each of the N^2 lines (in -> out) along `axis`.
  auto sweep = [&](Kokkos::View<Real *> in, Kokkos::View<Real *> out, int axis, Real t) {
    Kokkos::parallel_for(
        "fe::dage::conv", nline, KOKKOS_LAMBDA(int line) {
          const int i = line / N, j = line % N;
          std::size_t base;
          std::size_t stride;
          if (axis == 0) { base = (static_cast<std::size_t>(i) * N + j) * N; stride = 1; }
          else if (axis == 1) { base = static_cast<std::size_t>(i) * N * N + j; stride = N; }
          else { base = static_cast<std::size_t>(i) * N + j; stride = static_cast<std::size_t>(N) * N; }
          for (int jo = 0; jo < N; ++jo) {
            const Real u1 = xnode(jo);
            Real acc = 0;
            for (int eB = 0; eB < ne; ++eB) {
              Real lo = t * (u1 - be(eB + 1)), hi = t * (u1 - be(eB));
              lo = lo > -vmax ? lo : -vmax;
              hi = hi < vmax ? hi : vmax;
              if (hi <= lo) continue;
              const int o = noff(eB), p = nps(eB);
              const Real cen = Real(0.5) * (be(eB) + be(eB + 1));
              const Real hw = Real(0.5) * (be(eB + 1) - be(eB));
              const Real vc = Real(0.5) * (lo + hi), vh = Real(0.5) * (hi - lo);
              Real s = 0;
              for (int gi = 0; gi < nv; ++gi) {
                const Real v = vc + vh * vg(gi);
                const Real xi = ((u1 - v / t) - cen) / hw;
                Real num = 0, den = 0, lag = 0;
                bool hit = false;
                for (int k = 0; k < p; ++k) {
                  const Real dd = xi - rn(o + k);
                  const Real val = in(base + static_cast<std::size_t>(o + k) * stride);
                  if (dd < Real(1e-13) && dd > Real(-1e-13)) { lag = val; hit = true; break; }
                  const Real tt = bw(o + k) / dd;
                  num += tt * val;
                  den += tt;
                }
                if (!hit) lag = num / den;
                s += vh * vw(gi) * lag * Kokkos::exp(-v * v);
              }
              acc += s / t;
            }
            out(base + static_cast<std::size_t>(jo) * stride) = acc;
          }
        });
  };
  for (int it = 0; it < tgrid.n(); ++it) {
    const Real t = tgrid.t[it], wt = tgrid.w[it];
    Kokkos::deep_copy(A, R);
    sweep(A, B, 0, t); // z: A -> B
    sweep(B, A, 1, t); // y: B -> A
    sweep(A, B, 2, t); // x: A -> B (result in B)
    Kokkos::parallel_for(
        "fe::dage::acc", N3, KOKKOS_LAMBDA(std::size_t k) { V(k) += wt * B(k); });
  }
  return V;
}
} // namespace detail

/// DAGE Coulomb-family potential (host wrapper around detail::fe_dage3d_dev):
/// density vector in, potential vector out.
template <class Real>
std::vector<Real> fe_dage3d(const FEGrid1D<Real> &grid, const TGrid<Real> &tgrid,
                            const std::vector<Real> &rho, int nv = 32, Real vmax = Real(8)) {
  auto R = detail::to_device(rho, "fe::rho");
  auto V = detail::fe_dage3d_dev(grid, tgrid, R, nv, vmax);
  const std::size_t N3 = static_cast<std::size_t>(grid.N) * grid.N * grid.N;
  auto hV = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, V);
  std::vector<Real> Vout(N3);
  for (std::size_t k = 0; k < N3; ++k) Vout[k] = hV(k);
  return Vout;
}

/// Grid inner product int f g dr over the 3D FE grid (product of 1D weights).
template <class Real>
Real fe_inner(const FEGrid1D<Real> &grid, const std::vector<Real> &f,
              const std::vector<Real> &g) {
  const int N = grid.N;
  Real s = 0;
  for (int ix = 0; ix < N; ++ix)
    for (int iy = 0; iy < N; ++iy) {
      const Real wxy = grid.xw[ix] * grid.xw[iy];
      const std::size_t base = (static_cast<std::size_t>(ix) * N + iy) * N;
      for (int iz = 0; iz < N; ++iz) {
        const std::size_t k = base + iz;
        s += wxy * grid.xw[iz] * f[k] * g[k];
      }
    }
  return s;
}

} // namespace intti
