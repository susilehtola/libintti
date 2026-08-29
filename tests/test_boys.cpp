// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <gtest/gtest.h>

#include "intti/reference/boys.hpp"

namespace {

struct BoysRef {
  int m;
  double x;
  double val;
};

// 30-digit mpmath reference values
const BoysRef kRef[] = {
    {0, 0.1, 0.967643312635591831},
    {0, 1.0, 0.746824132812427025},
    {0, 10.0, 0.280247390506642741},
    {0, 30.0, 0.16180215937964007},
    {0, 40.0, 0.140124780409948217},
    {0, 100.0, 0.0886226925452758014},
    {3, 0.1, 0.132188029635738948},
    {3, 1.0, 0.0667322747768222568},
    {3, 10.0, 0.000522541236721495176},
    {3, 30.0, 0.0000112362610663344912},
    {3, 40.0, 4.10521817607264513e-6},
    {3, 100.0, 1.66167548522392128e-7},
    {10, 0.1, 0.0434651897241010828},
    {10, 1.0, 0.0191729360913146307},
    {10, 10.0, 8.57837826217344578e-6},
    {10, 30.0, 1.75197494140663688e-10},
    {10, 40.0, 8.54430414005834711e-12},
    {10, 100.0, 5.66639194474392784e-16},
};

TEST(Boys, ReferenceValues) {
  double F[11];
  for (const auto &r : kRef) {
    intti::ref::boys(r.m, r.x, F);
    EXPECT_NEAR(F[r.m], r.val, 5e-14 * r.val) << "m=" << r.m << " x=" << r.x;
  }
}

TEST(Boys, SmallXLimit) {
  double F[6];
  intti::ref::boys(5, 0.0, F);
  for (int m = 0; m <= 5; ++m)
    EXPECT_DOUBLE_EQ(F[m], 1.0 / (2 * m + 1));
}

} // namespace
