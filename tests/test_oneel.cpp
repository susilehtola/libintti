// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/oneel.hpp"

namespace {

using Shell = intti::PrimitiveShell<double>;

intti::ShellBasis<double> basis() {
  return intti::make_basis<double>({
      {1.2, {0.0, 0.0, 0.0}, 0},
      {0.8, {0.0, 0.0, 0.0}, 1},
      {0.5, {0.3, -0.4, 0.9}, 0},
      {0.6, {0.3, -0.4, 0.9}, 2},
  });
}

TEST(OneEl, SSOverlapAnalytic) {
  // <exp(-a|r-A|^2) | exp(-b|r-B|^2)> = (pi/p)^{3/2} exp(-mu |A-B|^2)
  const double a = 1.2, b = 0.5;
  const double A[3] = {0, 0, 0}, B[3] = {0.3, -0.4, 0.9};
  auto bas = intti::make_basis<double>({{a, {A[0], A[1], A[2]}, 0}, {b, {B[0], B[1], B[2]}, 0}});
  auto S = intti::overlap_matrix(bas);
  const double p = a + b, mu = a * b / p;
  double R2 = 0;
  for (int d = 0; d < 3; ++d) R2 += (A[d] - B[d]) * (A[d] - B[d]);
  const double exact = std::pow(M_PI / p, 1.5) * std::exp(-mu * R2);
  EXPECT_NEAR(S[0 * 2 + 1], exact, 1e-14 * exact);
  EXPECT_NEAR(S[1 * 2 + 0], exact, 1e-14 * exact); // symmetric
  EXPECT_NEAR(S[0], std::pow(M_PI / (2 * a), 1.5), 1e-13); // self-overlap
}

TEST(OneEl, DipoleOfSymmetricGaussianIsCenter) {
  // <s_A | (r - O) | s_A> / <s_A|s_A> = A - O  (moment of a centered Gaussian)
  const double A[3] = {0.3, -0.4, 0.9};
  auto bas = intti::make_basis<double>({{0.7, {A[0], A[1], A[2]}, 0}});
  auto S = intti::overlap_matrix(bas);
  const double origin[3] = {0, 0, 0};
  auto M = intti::multipole_matrices(bas, 1, origin); // 0:s, 1:x,2:y,3:z
  for (int d = 0; d < 3; ++d)
    EXPECT_NEAR(M[d + 1][0] / S[0], A[d], 1e-13);
}

TEST(OneEl, MatricesSymmetric) {
  auto bas = basis();
  auto S = intti::overlap_matrix(bas);
  auto T = intti::kinetic_matrix(bas);
  const int n = bas.nao;
  double asS = 0, asT = 0, mn = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      asS = std::max(asS, std::abs(S[i * n + j] - S[j * n + i]));
      asT = std::max(asT, std::abs(T[i * n + j] - T[j * n + i]));
      mn = std::max(mn, std::abs(S[i * n + j]));
    }
  EXPECT_LT(asS, 1e-14 * mn);
  EXPECT_LT(asT, 1e-13);
  for (int i = 0; i < n; ++i)
    EXPECT_GT(T[i * n + i], 0.0) << "kinetic diagonal must be positive";
}

TEST(OneEl, AngularMomentumAntisymmetric) {
  // r x nabla is anti-Hermitian -> the AO matrix is real antisymmetric with
  // zero diagonal (convention-independent structural check)
  auto bas = basis();
  const double O[3] = {0.1, 0.0, -0.2};
  auto L = intti::angular_momentum(bas, O);
  const int n = bas.nao;
  for (int d = 0; d < 3; ++d) {
    double asym = 0, mx = 0, diag = 0;
    for (int i = 0; i < n; ++i) {
      diag = std::max(diag, std::abs(L[d][i * n + i]));
      for (int j = 0; j < n; ++j) {
        asym = std::max(asym, std::abs(L[d][i * n + j] + L[d][j * n + i]));
        mx = std::max(mx, std::abs(L[d][i * n + j]));
      }
    }
    EXPECT_LT(asym, 1e-13 * (mx + 1)) << "component " << d << " not antisymmetric";
    EXPECT_LT(diag, 1e-13) << "component " << d << " diagonal nonzero";
    EXPECT_GT(mx, 1e-3) << "component " << d << " vanished";
  }
}

TEST(OneEl, PropertyValueContraction) {
  auto bas = basis();
  auto S = intti::overlap_matrix(bas);
  const int n = bas.nao;
  std::vector<double> D(static_cast<std::size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i) D[i * n + i] = 1.0; // identity density
  // Tr(D S) = Tr(S) for identity
  double tr = 0;
  for (int i = 0; i < n; ++i) tr += S[i * n + i];
  EXPECT_NEAR(intti::property_value(D, S), tr, 1e-13);
}

} // namespace
