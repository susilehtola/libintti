// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/nuclear.hpp"

namespace {

using Shell = intti::PrimitiveShell<double>;

double F0(double x) {
  return x < 1e-14 ? 1.0 : 0.5 * std::sqrt(M_PI / x) * std::erf(std::sqrt(x));
}

intti::ShellBasis<double> spd_basis() {
  return intti::make_basis<double>({
      {1.2, {0.0, 0.0, 0.0}, 0},
      {0.8, {0.0, 0.0, 0.0}, 1},
      {0.5, {0.3, -0.4, 0.9}, 0},
      {0.6, {0.3, -0.4, 0.9}, 2},
  });
}

TEST(Nuclear, SSAnalytic) {
  const double a = 0.8, b = 1.3;
  const double A[3] = {0.0, 0.1, -0.3}, B[3] = {0.5, -0.2, 0.4}, C[3] = {1.0, 0.8, 0.0};
  auto bas = intti::make_basis<double>({{a, {A[0], A[1], A[2]}, 0}, {b, {B[0], B[1], B[2]}, 0}});
  auto grid = intti::make_tgrid(intti::coulomb());
  // one unit charge at C, weight -1 (nuclear attraction sign)
  std::vector<intti::PointCharge<double>> ch{{-1.0, {C[0], C[1], C[2]}}};
  auto V = intti::nuclear_matrix(bas, ch, grid);
  const double p = a + b;
  double P[3], R2 = 0, mu = a * b / p, RAB2 = 0;
  for (int d = 0; d < 3; ++d) {
    P[d] = (a * A[d] + b * B[d]) / p;
    R2 += (P[d] - C[d]) * (P[d] - C[d]);
    RAB2 += (A[d] - B[d]) * (A[d] - B[d]);
  }
  const double exact = -(2 * M_PI / p) * std::exp(-mu * RAB2) * F0(p * R2);
  EXPECT_NEAR(V[0 * 2 + 1], exact, 1e-13 * std::abs(exact));
  EXPECT_NEAR(V[1 * 2 + 0], exact, 1e-13 * std::abs(exact)); // symmetric
}

TEST(Nuclear, ScreeningControlled) {
  auto bas = spd_basis();
  auto grid = intti::make_tgrid(intti::coulomb());
  std::vector<intti::PointCharge<double>> ch{{-1.0, {0.0, 0.0, 0.0}},
                                             {-6.0, {0.3, -0.4, 0.9}},
                                             {-1.0, {5.0, 5.0, 5.0}}}; // far
  auto V0 = intti::nuclear_matrix(bas, ch, grid, 0.0);
  auto V1 = intti::nuclear_matrix(bas, ch, grid, 1e-10);
  double md = 0, mx = 0;
  for (std::size_t i = 0; i < V0.size(); ++i) {
    md = std::max(md, std::abs(V0[i] - V1[i]));
    mx = std::max(mx, std::abs(V0[i]));
  }
  EXPECT_LT(md, 1e-8 * mx);
}

TEST(Nuclear, CollocationMatchesUnitCharge) {
  auto bas = spd_basis();
  auto grid = intti::make_tgrid(intti::coulomb());
  const std::array<double, 3> pt = {0.2, -0.1, 0.9};
  auto Vp = intti::potential_matrices(bas, {pt}, grid);
  std::vector<intti::PointCharge<double>> ch{{1.0, {pt[0], pt[1], pt[2]}}};
  auto Vn = intti::nuclear_matrix(bas, ch, grid);
  for (std::size_t i = 0; i < Vn.size(); ++i)
    EXPECT_NEAR(Vp[0][i], Vn[i], 1e-14);
  // symmetry
  const int n = bas.nao;
  double as = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      as = std::max(as, std::abs(Vp[0][i * n + j] - Vp[0][j * n + i]));
  EXPECT_LT(as, 1e-13);
}

} // namespace
