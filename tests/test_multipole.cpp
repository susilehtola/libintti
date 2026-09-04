// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
//
// M-MP: far-field (multipole) ERIs must converge to the exact eri_quartet as
// the bra/ket separation grows, with error ~ exp(-alpha |R_PQ|^2). For compact
// Gaussians the finite Hermite multipole expansion is exact, so well-separated
// pairs match to machine precision; the multipole tensor is also grid-free and
// precision-generic.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/multipole.hpp"
#include "intti/quartet.hpp"
#include "intti/tgrid.hpp"

namespace {

using intti::ncart;

// Far-field ERI vs exact eri_quartet at increasing separation, several classes.
TEST(Multipole, FarFieldConvergesToExact) {
  auto sh = [](double a, double x, double y, double z, int l) {
    return intti::PrimitiveShell<double>{a, {x, y, z}, l};
  };
  auto grid = intti::make_tgrid(intti::coulomb<double>());
  struct Cfg {
    int la, lb, lc, ld;
  };
  const Cfg cfgs[] = {{0, 0, 0, 0}, {1, 0, 0, 0}, {1, 1, 1, 1}, {2, 0, 2, 0}};
  for (const auto &cf : cfgs) {
    double prev = 1e99;
    for (double Rsep : {4.0, 8.0}) {
      auto a = sh(1.2, 0, 0, 0, cf.la);
      auto b = sh(0.9, 0.3, 0.1, -0.2, cf.lb);
      auto c = sh(1.0, Rsep, 0, 0, cf.lc);
      auto d = sh(0.8, Rsep + 0.2, -0.1, 0.3, cf.ld);
      auto bra = intti::make_pair(a, b), ket = intti::make_pair(c, d);
      const int nb = ncart(cf.la) * ncart(cf.lb);
      const int nk = ncart(cf.lc) * ncart(cf.ld);
      std::vector<double> ex(nb * nk), ff(nb * nk);
      intti::eri_quartet(bra, ket, grid, ex.data());
      intti::eri_quartet_farfield(bra, ket, ff.data());
      double maxrel = 0;
      for (int i = 0; i < nb * nk; ++i)
        maxrel = std::max(maxrel, std::fabs((ff[i] - ex[i]) / ex[i]));
      // error must fall as R grows and reach machine precision by R=8
      EXPECT_LT(maxrel, prev);
      if (Rsep >= 8.0) EXPECT_LT(maxrel, 1e-12);
      prev = maxrel;
    }
  }
}

// The exponent-free tensor is a plain derivative of 1/R: check low orders by
// hand. R = (X,Y,Z), T_000 = 1/R, T_100 = -X/R^3, T_200 = (3X^2-R^2)/R^5,
// T_110 = 3XY/R^5.
TEST(Multipole, TensorMatchesClosedForm) {
  const double R[3] = {1.3, -0.7, 2.1};
  const double R2 = R[0] * R[0] + R[1] * R[1] + R[2] * R[2];
  const double Rn = std::sqrt(R2);
  const int L = 2, D = L + 1;
  std::vector<double> T(static_cast<std::size_t>(D) * D * D);
  intti::multipole_tensor(L, R, T.data());
  auto Ti = [&](int t, int u, int v) { return T[(t * D + u) * D + v]; };
  EXPECT_NEAR(Ti(0, 0, 0), 1.0 / Rn, 1e-14);
  EXPECT_NEAR(Ti(1, 0, 0), -R[0] / (R2 * Rn), 1e-14);
  EXPECT_NEAR(Ti(0, 1, 0), -R[1] / (R2 * Rn), 1e-14);
  EXPECT_NEAR(Ti(2, 0, 0), (3 * R[0] * R[0] - R2) / (R2 * R2 * Rn), 1e-14);
  EXPECT_NEAR(Ti(1, 1, 0), 3 * R[0] * R[1] / (R2 * R2 * Rn), 1e-14);
}

// Precision-generic: the far-field path uses no grid, so it runs unchanged at
// long double and matches the double result (the reference has no grid error to
// diverge from here since we compare the two far-field evaluations).
TEST(Multipole, PrecisionGeneric) {
  auto shd = [](double a, double x, double y, double z, int l) {
    return intti::PrimitiveShell<double>{a, {x, y, z}, l};
  };
  auto shl = [](long double a, long double x, long double y, long double z,
                int l) {
    return intti::PrimitiveShell<long double>{a, {x, y, z}, l};
  };
  auto bd = intti::make_pair(shd(1.1, 0, 0, 0, 1), shd(0.7, 0.2, 0, 0, 1));
  auto kd = intti::make_pair(shd(0.9, 10, 0, 0, 1), shd(1.3, 10.1, 0.2, 0, 0));
  auto bl = intti::make_pair(shl(1.1L, 0, 0, 0, 1), shl(0.7L, 0.2L, 0, 0, 1));
  auto kl =
      intti::make_pair(shl(0.9L, 10, 0, 0, 1), shl(1.3L, 10.1L, 0.2L, 0, 0));
  const int n = ncart(1) * ncart(1) * ncart(1) * ncart(0);
  std::vector<double> vd(n);
  std::vector<long double> vl(n);
  intti::eri_quartet_farfield(bd, kd, vd.data());
  intti::eri_quartet_farfield(bl, kl, vl.data());
  for (int i = 0; i < n; ++i)
    EXPECT_NEAR(vd[i], static_cast<double>(vl[i]), 1e-12 * std::fabs(vd[i]));
}

} // namespace
