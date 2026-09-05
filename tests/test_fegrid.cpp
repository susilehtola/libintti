// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/fegrid.hpp"
#include "intti/kernel.hpp"
#include "intti/tgrid.hpp"

// Fast CI tests of the FE-grid machinery: grid quadrature exactness and the
// t-adapted 1D convolution (the core operator) against 1D analytic references,
// plus one tiny 3D DAGE smoke test. The heavy high-accuracy 3D DAGE validation
// and its spectral convergence live in prototype/fe_dage3d.cpp (2.4e-5 @ N=40
// -> 5.1e-8 @ N=63) and prototype/fe_yukawa_dage.cpp (2.1e-7), which are too
// slow (host-serial, tens of seconds) for the unit-test suite.

namespace {

// FE grid quadrature of a Gaussian is spectrally exact: int e^{-2a r^2} dr =
// (pi/2a)^{3/2}. Cheap (pure quadrature), so use a well-resolved grid.
TEST(FEGrid, GaussianQuadratureExact) {
  auto g = intti::make_fegrid1d(8, 12, 8.0); // N=96/axis, quadrature only
  const double a = 1.0;
  const int N = g.N;
  std::vector<double> rho((std::size_t)N * N * N);
  for (int ix = 0; ix < N; ++ix)
    for (int iy = 0; iy < N; ++iy)
      for (int iz = 0; iz < N; ++iz) {
        const double x = g.xnode[ix], y = g.xnode[iy], z = g.xnode[iz];
        rho[((std::size_t)ix * N + iy) * N + iz] = std::exp(-a * (x * x + y * y + z * z));
      }
  const double num = intti::fe_inner(g, rho, rho);      // int e^{-2a r^2}
  const double ref = std::pow(M_PI / (2 * a), 1.5);
  EXPECT_LT(std::abs(num - ref), 1e-9 * ref);
}

// The t-adapted 1D convolution (the core DAGE operator) vs the analytic
// Gaussian-Gaussian convolution, for a wide range of t including large t (the
// knife's-edge regime). Closed form:
//   int e^{-a u'^2} e^{-t^2 (u-u')^2} du' = sqrt(pi/(a+t^2)) e^{-a t^2 u^2/(a+t^2)}.
TEST(FEGrid, TadaptConv1dVsAnalytic) {
  auto g = intti::make_fegrid1d(8, 12, 8.0);
  const int N = g.N, nv = 40;
  const double a = 1.0;
  std::vector<double> vg, vw;
  intti::detail::fe_gauss_legendre<double>(nv, -1.0, 1.0, vg, vw);
  std::vector<double> f(N), out(N);
  for (int i = 0; i < N; ++i) f[i] = std::exp(-a * g.xnode[i] * g.xnode[i]);
  for (double t : {0.5, 2.0, 8.0, 32.0, 128.0}) {
    intti::detail::fe_conv1d(g, vg, vw, 8.0, f.data(), t, out.data());
    const double pre = std::sqrt(M_PI / (a + t * t)), c = a * t * t / (a + t * t);
    double worst = 0, scale = 0;
    for (int i = 0; i < N; ++i) {
      const double u = g.xnode[i], ref = pre * std::exp(-c * u * u);
      worst = std::max(worst, std::abs(out[i] - ref));
      scale = std::max(scale, std::abs(ref));
    }
    EXPECT_LT(worst, 1e-6 * scale) << "t=" << t; // t-uniform to grid representation
  }
}

// Tiny 3D DAGE smoke test: Coulomb self-energy of a Gaussian vs analytic
//   (g_a|1/r|g_a) = 2 pi^{5/2} / (a^2 sqrt(2a)). Small grid + coarse t-grid keep
// it fast; accuracy is grid-limited (tight accuracy is the prototype's job).
TEST(FEGrid, DageCoulombSmoke) {
  auto g = intti::make_fegrid1d(3, 7, 8.0); // N=21/axis (small, fast)
  const double a = 1.0;
  const int N = g.N;
  std::vector<double> rho((std::size_t)N * N * N);
  for (int ix = 0; ix < N; ++ix)
    for (int iy = 0; iy < N; ++iy)
      for (int iz = 0; iz < N; ++iz) {
        const double x = g.xnode[ix], y = g.xnode[iy], z = g.xnode[iz];
        rho[((std::size_t)ix * N + iy) * N + iz] = std::exp(-a * (x * x + y * y + z * z));
      }
  intti::TGridSpec<double> spec;
  spec.n = 24; // coarse Mobius t-grid for a fast smoke test
  auto V = intti::fe_dage3d(g, intti::make_tgrid(intti::coulomb(), spec), rho, 16);
  const double J = intti::fe_inner(g, rho, V);
  const double ref = 2 * std::pow(M_PI, 2.5) / (a * a * std::sqrt(2 * a));
  EXPECT_LT(std::abs(J - ref), 5e-2 * ref) << "grid " << J << " vs analytic " << ref;
}

} // namespace
