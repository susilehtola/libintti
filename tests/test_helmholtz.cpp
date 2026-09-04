// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
//
// M-HK: the Yukawa / bound-state Helmholtz attraction primitive. The screened
// kernel e^{-kappa r}/r drives the same t-quadrature machinery as Coulomb (only
// the grid weights change), so nuclear_matrix / yukawa_attraction_matrix with a
// Yukawa grid give the Yukawa attraction. Validated against an INDEPENDENT
// real-space oracle: the screened-Poisson potential of a spherical Gaussian,
//   Phi_p(R) = (2 pi/(kappa R)) int_0^inf e^{-p rho^2} rho
//              [e^{-kappa|R-rho|} - e^{-kappa(R+rho)}] d rho,
// evaluated by Gauss-Legendre (split at the |R-rho| kink) -- a completely
// different computational path from the t-quadrature.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/helmholtz.hpp"

namespace {

// n-point Gauss-Legendre nodes/weights on [-1,1] (Newton on Legendre roots).
void gauss_legendre(int n, std::vector<double> &x, std::vector<double> &w) {
  x.resize(n);
  w.resize(n);
  for (int i = 0; i < n; ++i) {
    double z = std::cos(M_PI * (i + 0.75) / (n + 0.5)), z1;
    int it = 0;
    double pp = 0;
    do {
      double p0 = 1, p1 = 0;
      for (int j = 0; j < n; ++j) {
        double p2 = p1;
        p1 = p0;
        p0 = ((2 * j + 1) * z * p1 - j * p2) / (j + 1);
      }
      pp = n * (z * p0 - p1) / (z * z - 1);
      z1 = z;
      z = z1 - p0 / pp;
    } while (std::fabs(z - z1) > 1e-15 && ++it < 100);
    x[i] = z;
    w[i] = 2 / ((1 - z * z) * pp * pp);
  }
}

double sub_integral(double p, double kappa, double R, double lo, double hi) {
  static std::vector<double> x, w;
  if (x.empty()) gauss_legendre(400, x, w);
  const double c = 0.5 * (hi - lo), m = 0.5 * (hi + lo);
  double s = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double rho = m + c * x[i];
    const double f = std::exp(-p * rho * rho) * rho *
                     (std::exp(-kappa * std::fabs(R - rho)) -
                      std::exp(-kappa * (R + rho)));
    s += w[i] * f;
  }
  return c * s;
}

// Yukawa potential of a spherical Gaussian e^{-p rho^2} at distance R.
double phi_oracle(double p, double kappa, double R) {
  const double I = sub_integral(p, kappa, R, 0, R) +
                   sub_integral(p, kappa, R, R, R + 20 / std::sqrt(p));
  return 2 * M_PI / (kappa * R) * I;
}

TEST(Helmholtz, YukawaAttractionVsRadialOracle) {
  for (double kappa : {0.5, 1.3, 3.0}) {
    for (double a : {0.6, 2.0}) {
      // two s-shells at the origin (product exponent p = a + 0.7a, K = 1),
      // unit charge at distance R along z
      auto basis = intti::make_basis<double>({{a, {0, 0, 0}, 0}, {0.7 * a, {0, 0, 0}, 0}});
      const double p = a + 0.7 * a;
      for (double R : {0.8, 1.5, 3.0}) {
        std::vector<intti::PointCharge<double>> ch{{1.0, {0, 0, R}}};
        auto V = intti::yukawa_attraction_matrix(basis, ch, kappa);
        const double got = V[0 * 2 + 1];
        const double ref = phi_oracle(p, kappa, R);
        EXPECT_NEAR(got, ref, 1e-5 * std::abs(ref))
            << "kappa=" << kappa << " a=" << a << " R=" << R;
      }
    }
  }
}

// nuclear_matrix accepts a Yukawa grid directly (the primitive is not special):
// a well-ranged Yukawa grid reproduces the radial oracle through nuclear_matrix.
TEST(Helmholtz, NuclearMatrixWithYukawaGrid) {
  const double kappa = 1.0, a = 1.1, p = a + 0.9;
  auto basis = intti::make_basis<double>({{a, {0, 0, 0}, 0}, {0.9, {0, 0, 0}, 0}});
  auto grid = intti::yukawa_grid<double>(kappa, 1e-2, 1e3);
  std::vector<intti::PointCharge<double>> ch{{1.0, {0, 0, 1.2}}};
  auto V = intti::nuclear_matrix(basis, ch, grid);
  EXPECT_NEAR(V[1], phi_oracle(p, kappa, 1.2), 1e-5 * std::abs(V[1]));
}

} // namespace
