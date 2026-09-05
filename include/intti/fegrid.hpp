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

#include "kernel.hpp"
#include "tgrid.hpp"

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
template <class Real>
FEGrid1D<Real> make_fegrid1d_hp(const std::vector<FEGaussian1D<Real>> &gaussians,
                                Real eps, int pmin = 4, int pmax = 16, int pdeg = 0) {
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
  for (const auto &g : gaussians)
    if (g.center > xlo && g.center < xhi) seeds.push_back(g.center);
  std::sort(seeds.begin(), seeds.end());
  seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());
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
template <class Real>
std::vector<Real> fe_dage3d(const FEGrid1D<Real> &grid, const TGrid<Real> &tgrid,
                            const std::vector<Real> &rho, int nv = 32, Real vmax = Real(8)) {
  const int N = grid.N;
  const std::size_t N3 = static_cast<std::size_t>(N) * N * N;
  std::vector<Real> vg, vw;
  detail::fe_gauss_legendre<Real>(nv, Real(-1), Real(1), vg, vw);
  std::vector<Real> V(N3, Real(0)), work(N3), line(N), out(N);
  for (int it = 0; it < tgrid.n(); ++it) {
    const Real t = tgrid.t[it];
    work = rho;
    // conv along z (contiguous lines)
    for (int ix = 0; ix < N; ++ix)
      for (int iy = 0; iy < N; ++iy) {
        Real *ln = &work[(static_cast<std::size_t>(ix) * N + iy) * N];
        detail::fe_conv1d(grid, vg, vw, vmax, ln, t, out.data());
        for (int iz = 0; iz < N; ++iz) ln[iz] = out[iz];
      }
    // conv along y
    for (int ix = 0; ix < N; ++ix)
      for (int iz = 0; iz < N; ++iz) {
        for (int iy = 0; iy < N; ++iy) line[iy] = work[(static_cast<std::size_t>(ix) * N + iy) * N + iz];
        detail::fe_conv1d(grid, vg, vw, vmax, line.data(), t, out.data());
        for (int iy = 0; iy < N; ++iy) work[(static_cast<std::size_t>(ix) * N + iy) * N + iz] = out[iy];
      }
    // conv along x
    for (int iy = 0; iy < N; ++iy)
      for (int iz = 0; iz < N; ++iz) {
        for (int ix = 0; ix < N; ++ix) line[ix] = work[(static_cast<std::size_t>(ix) * N + iy) * N + iz];
        detail::fe_conv1d(grid, vg, vw, vmax, line.data(), t, out.data());
        for (int ix = 0; ix < N; ++ix) work[(static_cast<std::size_t>(ix) * N + iy) * N + iz] = out[ix];
      }
    const Real wt = tgrid.w[it];
    for (std::size_t i = 0; i < N3; ++i) V[i] += wt * work[i];
  }
  return V;
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
