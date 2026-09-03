// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>

#include <vector>

#include <gtest/gtest.h>

#include "intti/quartet.hpp"
#include "intti/tgrid.hpp"

namespace {

// Unnormalised analytic (ss|ss) via the Boys function F0.
double ssss_analytic(double a, const double A[3], double b, const double B[3], double c,
                     const double C[3], double d, const double D[3]) {
  const double p = a + b, q = c + d;
  double AB2 = 0, CD2 = 0, PQ2 = 0, P[3], Q[3];
  for (int k = 0; k < 3; ++k) {
    AB2 += (A[k] - B[k]) * (A[k] - B[k]);
    CD2 += (C[k] - D[k]) * (C[k] - D[k]);
    P[k] = (a * A[k] + b * B[k]) / p;
    Q[k] = (c * C[k] + d * D[k]) / q;
  }
  for (int k = 0; k < 3; ++k) PQ2 += (P[k] - Q[k]) * (P[k] - Q[k]);
  const double rho = p * q / (p + q), x = rho * PQ2;
  const double F0 = x < 1e-13 ? 1.0 : 0.5 * std::sqrt(M_PI / x) * std::erf(std::sqrt(x));
  const double Kab = std::exp(-a * b / p * AB2), Kcd = std::exp(-c * d / q * CD2);
  return Kab * Kcd * 2 * std::pow(M_PI, 2.5) / (p * q * std::sqrt(p + q)) * F0;
}

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

TEST(TGrid, AdaptiveRangeWideExponents) {
  // A single Mobius grid sized to a wide exponent span [1e-3, 1e7] (10 decades,
  // beyond the default's [1e-2, 1e6]) must reproduce analytic (ss|ss) for every
  // exponent combination -- and the default grid should be visibly worse.
  using Shell = intti::PrimitiveShell<double>;
  auto adapt = intti::make_tgrid(intti::coulomb(), intti::mobius_spec_for_range(1e-5, 1e9));
  auto def = intti::make_tgrid(intti::coulomb());
  const std::vector<double> exps = {1e-5, 1e-2, 1e1, 1e4, 1e7, 1e9};
  const double A[3] = {0, 0, 0}, B[3] = {0.3, 0.0, 0.0};
  const double C[3] = {0.0, 0.4, 0.7}, D[3] = {0.1, 0.0, 0.0};
  double err_adapt = 0, err_def = 0;
  for (double a : exps)
    for (double b : exps)
      for (double c : exps)
        for (double d : exps) {
          Shell sa{a, {A[0], A[1], A[2]}, 0}, sb{b, {B[0], B[1], B[2]}, 0};
          Shell sc{c, {C[0], C[1], C[2]}, 0}, sd{d, {D[0], D[1], D[2]}, 0};
          const double ref = ssss_analytic(a, A, b, B, c, C, d, D);
          if (std::abs(ref) < 1e-290) continue;
          double va = 0, vd = 0;
          intti::eri_quartet(intti::make_pair(sa, sb), intti::make_pair(sc, sd), adapt, &va);
          intti::eri_quartet(intti::make_pair(sa, sb), intti::make_pair(sc, sd), def, &vd);
          err_adapt = std::max(err_adapt, std::abs((va - ref) / ref));
          err_def = std::max(err_def, std::abs((vd - ref) / ref));
        }
  EXPECT_LT(err_adapt, 1e-7) << "adaptive grid inaccurate over the wide range";
  EXPECT_GT(err_def, 1e-4) << "default grid should not cover this range (gap not exercised)";
}

TEST(TGrid, BeylkinMonzonFewerNodesThanMobius) {
  // Over the same wide span, the Beylkin-Monzon (ExpSum) grid must be as
  // accurate as the Mobius grid but with substantially fewer nodes (its node
  // count grows logarithmically, not superlinearly, with the range).
  using Shell = intti::PrimitiveShell<double>;
  auto bm = intti::make_tgrid(intti::coulomb(), intti::exp_sum_spec_for_range(1e-5, 1e9));
  auto mob = intti::make_tgrid(intti::coulomb(), intti::mobius_spec_for_range(1e-5, 1e9));
  const std::vector<double> exps = {1e-5, 1e-2, 1e1, 1e4, 1e7, 1e9};
  const double A[3] = {0, 0, 0}, B[3] = {0.3, 0.0, 0.0};
  const double C[3] = {0.0, 0.4, 0.7}, D[3] = {0.1, 0.0, 0.0};
  double err_bm = 0, err_mob = 0;
  for (double a : exps)
    for (double b : exps)
      for (double c : exps)
        for (double d : exps) {
          Shell sa{a, {A[0], A[1], A[2]}, 0}, sb{b, {B[0], B[1], B[2]}, 0};
          Shell sc{c, {C[0], C[1], C[2]}, 0}, sd{d, {D[0], D[1], D[2]}, 0};
          const double ref = ssss_analytic(a, A, b, B, c, C, d, D);
          if (std::abs(ref) < 1e-290) continue;
          double vb = 0, vm = 0;
          intti::eri_quartet(intti::make_pair(sa, sb), intti::make_pair(sc, sd), bm, &vb);
          intti::eri_quartet(intti::make_pair(sa, sb), intti::make_pair(sc, sd), mob, &vm);
          err_bm = std::max(err_bm, std::abs((vb - ref) / ref));
          err_mob = std::max(err_mob, std::abs((vm - ref) / ref));
        }
  EXPECT_LT(err_bm, 1e-7) << "BM grid inaccurate (n=" << bm.n() << ")";
  EXPECT_LT(err_mob, 1e-7) << "Mobius control inaccurate (n=" << mob.n() << ")";
  EXPECT_LT(bm.n(), mob.n()) << "BM should need fewer nodes: bm=" << bm.n() << " mob=" << mob.n();
}

TEST(TGrid, ExpSumYukawaPointwise) {
  // Yukawa is the natural range-separated fit for the sinc grid: the
  // e^{-kappa^2/4t^2} factor kills the small-t tail, so both ends decay and the
  // trapezoidal rule is exponentially convergent.
  const double kappa = 1.3;
  intti::TGridSpec<double> spec;
  spec.mapping = intti::TMapping::ExpSum;
  spec.es_tmin = 0.05;
  spec.es_tmax = 1e2;
  spec.es_h = 0.2;
  auto grid = intti::make_tgrid(intti::yukawa(kappa), spec);
  for (double r : {0.1, 0.3, 1.0, 2.0, 3.0})
    EXPECT_NEAR(kernel_quad(grid, r), std::exp(-kappa * r) / r, 1e-9)
        << "r=" << r << " n=" << grid.n();
}

TEST(TGrid, ExpSumRejectsRangeSeparated) {
  // erf/erfc have a hard boundary at omega (only O(h^2) for the sinc rule), so
  // the ExpSum mapping refuses them -- Mobius (Gauss-Legendre) is the tool there.
  intti::TGridSpec<double> spec;
  spec.mapping = intti::TMapping::ExpSum;
  EXPECT_THROW(intti::make_tgrid(intti::erf_rs(0.7), spec), std::invalid_argument);
  EXPECT_THROW(intti::make_tgrid(intti::erfc_rs(0.7), spec), std::invalid_argument);
}

} // namespace
