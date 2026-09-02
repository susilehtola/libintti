// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>

#include <gtest/gtest.h>

#include "intti/threeel.hpp"

namespace {

using S = intti::PrimitiveShell<double>;

TEST(ThreeEl, OneCentreAnalytic) {
  // one-centre three-electron Coulomb of six identical s-GTOs: G_aaaaaa = 4z/3
  auto grid = intti::make_tgrid(intti::coulomb());
  for (double z : {0.5, 1.0, 1.3, 4.0}) {
    S a{z, {0.0, 0.0, 0.0}, 0};
    const double g = intti::three_electron_coulomb(a, a, a, a, a, a, grid);
    EXPECT_NEAR(g, 4 * z / 3, 1e-12 * (4 * z / 3)) << "zeta=" << z;
  }
}

TEST(ThreeEl, GeneralVsReference) {
  // arbitrary exponents/centres, checked against the analytic-Boys-free
  // reference (prototype), electron 1 = (a,d), 2 = (b,e), 3 = (c,f)
  auto grid = intti::make_tgrid(intti::coulomb());
  S a{0.9, {0.0, 0.0, 0.0}, 0}, b{1.1, {0.4, 0.0, 0.0}, 0}, c{0.7, {0.0, 0.3, 0.5}, 0};
  S d{1.3, {0.1, 0.0, 0.0}, 0}, e{0.8, {0.4, 0.1, 0.0}, 0}, f{1.0, {0.0, 0.3, 0.6}, 0};
  const double g = intti::three_electron_coulomb(a, b, c, d, e, f, grid);
  EXPECT_NEAR(g, 0.9594692137444143, 1e-11) << "general 3-electron integral";
}

TEST(ThreeEl, Electron23SwapSymmetry) {
  // r12^{-1} r13^{-1} is symmetric under exchanging electrons 2 and 3, i.e.
  // (b,e) <-> (c,f): G_abcdef == G_acbdfe
  auto grid = intti::make_tgrid(intti::coulomb());
  S a{0.9, {0.0, 0.0, 0.0}, 0}, b{1.1, {0.4, 0.0, 0.0}, 0}, c{0.7, {0.0, 0.3, 0.5}, 0};
  S d{1.3, {0.1, 0.0, 0.0}, 0}, e{0.8, {0.4, 0.1, 0.0}, 0}, f{1.0, {0.0, 0.3, 0.6}, 0};
  const double g1 = intti::three_electron_coulomb(a, b, c, d, e, f, grid);
  const double g2 = intti::three_electron_coulomb(a, c, b, d, f, e, grid);
  EXPECT_NEAR(g1, g2, 1e-13 * std::abs(g1));
}

} // namespace
