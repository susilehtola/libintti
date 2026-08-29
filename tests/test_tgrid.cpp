// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>

#include <gtest/gtest.h>

#include "intti/tgrid.hpp"

namespace {

// sum_i w_i exp(-t_i^2 r^2)
double kernel_quad(const intti::TGrid<double> &grid, double r) {
  double s = 0.0;
  for (int i = 0; i < grid.n(); ++i)
    s += grid.w[i] * std::exp(-grid.t[i] * grid.t[i] * r * r);
  return s;
}

intti::TGridSpec<double> linlog_spec() {
  intti::TGridSpec<double> spec;
  spec.mapping = intti::TMapping::LinLog;
  return spec;
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

TEST(TGrid, LinLogCoulombPointwise) {
  // (2/sqrt(pi)) int_0^tc exp(-t^2 r^2) dt = erf(tc r)/r ~= 1/r for tc r >> 1
  auto spec = linlog_spec();
  spec.t_lin = 4.0;
  spec.n_lin = 60;
  spec.n_log = 100;
  spec.t_c = 1000.0;
  auto grid = intti::make_tgrid(intti::coulomb(), spec);
  EXPECT_GT(grid.tail_coeff, 0.0);
  for (double r : {0.05, 0.2, 1.0, 2.0})
    EXPECT_NEAR(kernel_quad(grid, r) * r, 1.0, 1e-10) << "r=" << r;
}

TEST(TGrid, ErfPointwise) {
  // finite range: default (Mobius) spec uses plain Gauss-Legendre on [0, omega]
  const double omega = 0.7;
  auto grid = intti::make_tgrid(intti::erf_rs(omega));
  EXPECT_EQ(grid.tail_coeff, 0.0);
  for (double r : {0.3, 1.0, 3.0})
    EXPECT_NEAR(kernel_quad(grid, r), std::erf(omega * r) / r, 1e-10) << "r=" << r;
}

TEST(TGrid, LinLogErfcPointwise) {
  const double omega = 0.7;
  auto spec = linlog_spec();
  spec.n_log = 120;
  spec.t_c = 1000.0;
  auto grid = intti::make_tgrid(intti::erfc_rs(omega), spec);
  for (double r : {0.5, 1.0, 2.0})
    EXPECT_NEAR(kernel_quad(grid, r), std::erfc(omega * r) / r, 1e-10) << "r=" << r;
}

TEST(TGrid, LinLogYukawaPointwise) {
  const double kappa = 1.3;
  auto spec = linlog_spec();
  spec.n_lin = 60;
  spec.n_log = 120;
  spec.t_c = 1000.0;
  auto grid = intti::make_tgrid(intti::yukawa(kappa), spec);
  for (double r : {0.5, 1.0, 2.0})
    EXPECT_NEAR(kernel_quad(grid, r), std::exp(-kappa * r) / r, 1e-9) << "r=" << r;
}

TEST(TGrid, LinLogErfPlusErfcIsCoulomb) {
  const double omega = 0.6;
  auto erf_grid = intti::make_tgrid(intti::erf_rs(omega), linlog_spec());
  auto erfc_grid = intti::make_tgrid(intti::erfc_rs(omega), linlog_spec());
  auto coul_grid = intti::make_tgrid(intti::coulomb(), linlog_spec());
  for (double r : {0.5, 1.0, 2.0}) {
    const double split = kernel_quad(erf_grid, r) + kernel_quad(erfc_grid, r);
    EXPECT_NEAR(split, kernel_quad(coul_grid, r), 1e-11) << "r=" << r;
  }
}

TEST(TGrid, MobiusIsUntruncated) {
  auto grid = intti::make_tgrid(intti::coulomb());
  EXPECT_EQ(grid.tail_coeff, 0.0);
  EXPECT_EQ(grid.n(), 64);
  // nodes extend far beyond any LinLog default truncation
  EXPECT_GT(grid.t.back(), 1e3);
}

} // namespace
