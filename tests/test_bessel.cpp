// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>

#include <gtest/gtest.h>

#include "intti/bessel.hpp"

namespace {

struct IveRef {
  int m;
  double x;
  double val;
};

// 30-digit mpmath reference values of e^{-x} I_m(x)
const IveRef kRef[] = {
    {0, 0.001, 0.999000749583515559395},
    {0, 0.1, 0.907100925782301096436},
    {0, 1.0, 0.465759607593640436502},
    {0, 10.0, 0.127833337163428607323},
    {0, 100.0, 0.0399443792990966826476},
    {0, 10000.0, 0.00398947267460473210636},
    {1, 0.001, 0.000499500312354221336984},
    {1, 0.1, 0.0452984468088093250071},
    {1, 1.0, 0.207910415349708448869},
    {1, 10.0, 0.121262681384455518719},
    {1, 100.0, 0.0397441530251302526736},
    {1, 10000.0, 0.00398927319598366226448},
    {3, 0.001, 2.08125117139772456707e-11},
    {3, 0.1, 0.0000188625642254732624594},
    {3, 1.0, 0.00815530777281429381659},
    {3, 10.0, 0.0798303610298405172873},
    {3, 100.0, 0.0381781731755864895699},
    {3, 10000.0, 0.00398767772605567605033},
    {8, 0.001, 9.67843703093406287593e-32},
    {8, 0.1, 8.76860887493351974329e-16},
    {8, 1.0, 3.66430880311277831669e-8},
    {8, 10.0, 0.00526940789100638988988},
    {8, 100.0, 0.0289637757890192445111},
    {8, 10000.0, 0.00397672613070939531693},
};

TEST(Bessel, ReferenceValuesDouble) {
  double v[9];
  for (const auto &r : kRef) {
    intti::ive_ladder(r.m, r.x, v);
    EXPECT_NEAR(v[r.m], r.val, 5e-15 * r.val) << "m=" << r.m << " x=" << r.x;
  }
}

TEST(Bessel, ReferenceValuesLongDouble) {
  // floor: the double-rounded constant (~1e-16 rel) plus the sensitivity of
  // I_m to the binary rounding of x, d ln I_m/dx ~ m/x (up to ~5e-16 for
  // m=8 at x=0.1); long double evaluation must sit at that combined floor
  long double v[9];
  for (const auto &r : kRef) {
    intti::ive_ladder(r.m, static_cast<long double>(r.x), v);
    EXPECT_LT(std::abs(v[r.m] - static_cast<long double>(r.val)),
              1e-15L * static_cast<long double>(r.val))
        << "m=" << r.m << " x=" << r.x;
  }
}

TEST(Bessel, LadderIdentity) {
  // I_{m-1}(x) - I_{m+1}(x) = (2m/x) I_m(x), scaled form included
  for (double x : {0.5, 7.0, 50.0, 400.0, 5e4}) {
    double v[11];
    intti::ive_ladder(10, x, v);
    for (int m = 1; m < 10; ++m)
      EXPECT_NEAR(v[m - 1] - v[m + 1], 2 * m / x * v[m], 1e-14 * v[0])
          << "m=" << m << " x=" << x;
  }
}

TEST(Bessel, ZeroArgument) {
  double v[5];
  intti::ive_ladder(4, 0.0, v);
  EXPECT_DOUBLE_EQ(v[0], 1.0);
  for (int m = 1; m <= 4; ++m)
    EXPECT_DOUBLE_EQ(v[m], 0.0);
}

} // namespace
