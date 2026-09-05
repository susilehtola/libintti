// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Tensorial finite-element (FE) real-space grid + the t-adapted DAGE Coulomb
// operator -- the second, operator-agnostic integral route (roadmap M-FE),
// sharing the t-quadrature kernel core (tgrid.hpp) with the GTO/STO engine.
//
// A 1D axis is split into elements; on each element a function is a local
// polynomial through the element's Gauss-Legendre nodes (barycentric Lagrange).
// The 3D grid is the tensor product. The Coulomb (or any 1/r-family) potential
// of a density rho on the grid is the "direct approach" (DAGE):
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

/// 1D tensorial finite-element grid: ne equal elements on [-L,L], each carrying
/// np Gauss-Legendre nodes (a degree-(np-1) local polynomial). The 3D grid is
/// the tensor product of this along x, y, z (N = ne*np points per axis).
template <class Real> struct FEGrid1D {
  int ne{0}, np{0}, N{0};
  Real L{0};
  std::vector<Real> be;                 // element boundaries (ne+1)
  std::vector<Real> rn, rw, bw;         // reference nodes/weights/bary weights (np, on [-1,1])
  std::vector<Real> xnode, xw;          // global node coords / quadrature weights (N)
  Real cen(int e) const { return Real(0.5) * (be[e] + be[e + 1]); }
  Real hw(int e) const { return Real(0.5) * (be[e + 1] - be[e]); }
};

/// Build a 1D FE grid of ne elements x np nodes on [-L, L].
template <class Real> FEGrid1D<Real> make_fegrid1d(int ne, int np, Real L) {
  FEGrid1D<Real> g;
  g.ne = ne; g.np = np; g.N = ne * np; g.L = L;
  g.be.resize(ne + 1);
  for (int e = 0; e <= ne; ++e) g.be[e] = -L + 2 * L * e / ne;
  detail::fe_gauss_legendre<Real>(np, Real(-1), Real(1), g.rn, g.rw);
  g.bw.assign(np, Real(1));
  for (int k = 0; k < np; ++k)
    for (int j = 0; j < np; ++j)
      if (j != k) g.bw[k] /= (g.rn[k] - g.rn[j]);
  g.xnode.resize(g.N); g.xw.resize(g.N);
  for (int e = 0; e < ne; ++e)
    for (int k = 0; k < np; ++k) {
      g.xnode[e * np + k] = g.cen(e) + g.hw(e) * g.rn[k];
      g.xw[e * np + k] = g.hw(e) * g.rw[k];
    }
  return g;
}

namespace detail {
/// Barycentric Lagrange value of the element-local polynomial (nodal values f,
/// length np) at reference coordinate xi in [-1,1].
template <class Real> Real fe_lag(const FEGrid1D<Real> &g, const Real *f, Real xi) {
  Real num = 0, den = 0;
  for (int k = 0; k < g.np; ++k) {
    Real d = xi - g.rn[k];
    if (std::abs(d) < Real(1e-13)) return f[k];
    Real t = g.bw[k] / d;
    num += t * f[k];
    den += t;
  }
  return num / den;
}

/// t-adapted 1D convolution of an FE line (nodal values `in`, length N) with the
/// Gaussian kernel exp(-t^2 (u1-u2)^2): out(u1) = int in(u2) exp(-t^2(u1-u2)^2)
/// du2. vg/vw are reference GL nodes/weights on [-1,1]; vmax clamps the flat-
/// kernel window.
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
      const Real *fB = in + static_cast<std::size_t>(eB) * g.np;
      const Real vc = Real(0.5) * (lo + hi), vh = Real(0.5) * (hi - lo);
      Real s = 0;
      for (int gi = 0; gi < nv; ++gi) {
        const Real v = vc + vh * vg[gi];
        const Real xi = ((u1 - v / t) - g.cen(eB)) / g.hw(eB);
        s += vh * vw[gi] * fe_lag(g, fB, xi) * std::exp(-v * v);
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
/// (flat-kernel) quadrature order. Handles general non-separable densities.
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
