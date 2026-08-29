// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/hermite1d.hpp"
#include "intti/tgrid.hpp"

namespace {

TEST(ECoeffs, OverlapClosure) {
  // int (x-A)^i exp(-a(x-A)^2) (x-B)^j exp(-b(x-B)^2) dx = E_0^{ij} sqrt(pi/p)
  const double a = 0.9, b = 1.7, A = -0.3, B = 0.6;
  const double p = a + b, P = (a * A + b * B) / p;
  const double K = std::exp(-a * b / p * (A - B) * (A - B));
  const int la = 3, lb = 3;
  std::vector<double> E((la + 1) * (lb + 1) * (la + lb + 1));
  intti::e_coeffs(la, lb, p, P - A, P - B, K, E.data());

  const int nq = 400;
  std::vector<double> x(nq), w(nq);
  intti::gauss_legendre(nq, -12.0, 12.0, x.data(), w.data());
  for (int i = 0; i <= la; ++i)
    for (int j = 0; j <= lb; ++j) {
      double ref = 0.0;
      for (int k = 0; k < nq; ++k)
        ref += w[k] * std::pow(x[k] - A, i) * std::exp(-a * (x[k] - A) * (x[k] - A)) *
               std::pow(x[k] - B, j) * std::exp(-b * (x[k] - B) * (x[k] - B));
      const double e0 = E[(i * (lb + 1) + j) * (la + lb + 1)];
      EXPECT_NEAR(e0 * std::sqrt(M_PI / p), ref, 1e-12 * std::abs(ref))
          << "i=" << i << " j=" << j;
    }
}

TEST(HermiteB, LowOrdersAnalytic) {
  const double theta = 0.8, X = 1.3;
  double B[4];
  intti::hermite_b(3, theta, X, B);
  const double g = std::exp(-theta * X * X);
  EXPECT_DOUBLE_EQ(B[0], g);
  EXPECT_NEAR(B[1], -2.0 * theta * X * g, 1e-15);
  EXPECT_NEAR(B[2], (4.0 * theta * theta * X * X - 2.0 * theta) * g, 1e-14);
  EXPECT_NEAR(B[3], (12.0 * theta * theta * X - 8.0 * std::pow(theta, 3) * std::pow(X, 3)) * g,
              1e-13);
}

} // namespace
