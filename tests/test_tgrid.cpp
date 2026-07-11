// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola

#include <cmath>

#include <gtest/gtest.h>

#include "intti/tgrid.hpp"

namespace {

// sum_i w_i exp(-t_i^2 r^2) on the host copy of the grid
double kernel_quad(const intti::TGrid &grid, double r) {
  auto th = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grid.t);
  auto wh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grid.w);
  double s = 0.0;
  for (int i = 0; i < grid.n(); ++i)
    s += wh(i) * std::exp(-th(i) * th(i) * r * r);
  return s;
}

TEST(GaussLegendre, PolynomialExactness) {
  // n-point rule integrates polynomials up to degree 2n-1 exactly
  const int n = 6;
  double x[n], w[n];
  intti::gauss_legendre(n, -1.0, 3.0, x, w);
  for (int deg = 0; deg <= 2 * n - 1; ++deg) {
    double s = 0.0;
    for (int i = 0; i < n; ++i)
      s += w[i] * std::pow(x[i], deg);
    const double exact = (std::pow(3.0, deg + 1) - std::pow(-1.0, deg + 1)) / (deg + 1);
    EXPECT_NEAR(s, exact, 1e-12 * std::abs(exact)) << "degree " << deg;
  }
}

TEST(TGrid, CoulombPointwise) {
  // (2/sqrt(pi)) int_0^tc exp(-t^2 r^2) dt = erf(tc r)/r ~= 1/r for tc r >> 1
  intti::TGridSpec spec;
  spec.t_lin = 4.0;
  spec.n_lin = 60;
  spec.n_log = 100;
  spec.t_c = 1000.0;
  auto grid = intti::make_tgrid(intti::Kernel::coulomb(), spec);
  EXPECT_GT(grid.tail_coeff, 0.0);
  for (double r : {0.05, 0.2, 1.0, 2.0})
    EXPECT_NEAR(kernel_quad(grid, r) * r, 1.0, 1e-10) << "r=" << r;
}

TEST(TGrid, ErfPointwise) {
  const double omega = 0.7;
  auto grid = intti::make_tgrid(intti::Kernel::erf_rs(omega));
  EXPECT_EQ(grid.tail_coeff, 0.0);
  for (double r : {0.3, 1.0, 3.0})
    EXPECT_NEAR(kernel_quad(grid, r), std::erf(omega * r) / r, 1e-10) << "r=" << r;
}

TEST(TGrid, ErfcPointwise) {
  const double omega = 0.7;
  intti::TGridSpec spec;
  spec.n_log = 120;
  spec.t_c = 1000.0;
  auto grid = intti::make_tgrid(intti::Kernel::erfc_rs(omega), spec);
  for (double r : {0.5, 1.0, 2.0})
    EXPECT_NEAR(kernel_quad(grid, r), std::erfc(omega * r) / r, 1e-10) << "r=" << r;
}

TEST(TGrid, YukawaPointwise) {
  const double kappa = 1.3;
  intti::TGridSpec spec;
  spec.n_lin = 60;
  spec.n_log = 120;
  spec.t_c = 1000.0;
  auto grid = intti::make_tgrid(intti::Kernel::yukawa(kappa), spec);
  for (double r : {0.5, 1.0, 2.0})
    EXPECT_NEAR(kernel_quad(grid, r), std::exp(-kappa * r) / r, 1e-9) << "r=" << r;
}

TEST(TGrid, ErfPlusErfcIsCoulomb) {
  const double omega = 0.6;
  auto erf_grid = intti::make_tgrid(intti::Kernel::erf_rs(omega));
  auto erfc_grid = intti::make_tgrid(intti::Kernel::erfc_rs(omega));
  auto coul_grid = intti::make_tgrid(intti::Kernel::coulomb());
  for (double r : {0.5, 1.0, 2.0}) {
    const double split = kernel_quad(erf_grid, r) + kernel_quad(erfc_grid, r);
    EXPECT_NEAR(split, kernel_quad(coul_grid, r), 1e-11) << "r=" << r;
  }
}

} // namespace
