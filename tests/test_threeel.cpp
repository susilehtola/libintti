// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>

#include <gtest/gtest.h>

#include "intti/threeel.hpp"

namespace {

using G = intti::CartGauss<double>;

TEST(ThreeEl, OneCentreAnalytic) {
  // one-centre three-electron Coulomb of six identical s-GTOs: G_aaaaaa = 4z/3
  auto grid = intti::make_tgrid(intti::coulomb());
  for (double z : {0.5, 1.0, 1.3, 4.0}) {
    G a{z, {0.0, 0.0, 0.0}, {0, 0, 0}};
    const double g = intti::three_electron_coulomb(a, a, a, a, a, a, grid);
    EXPECT_NEAR(g, 4 * z / 3, 1e-12 * (4 * z / 3)) << "zeta=" << z;
  }
}

TEST(ThreeEl, GeneralSTypeVsReference) {
  auto grid = intti::make_tgrid(intti::coulomb());
  G a{0.9, {0.0, 0.0, 0.0}, {0, 0, 0}}, b{1.1, {0.4, 0.0, 0.0}, {0, 0, 0}},
      c{0.7, {0.0, 0.3, 0.5}, {0, 0, 0}}, d{1.3, {0.1, 0.0, 0.0}, {0, 0, 0}},
      e{0.8, {0.4, 0.1, 0.0}, {0, 0, 0}}, f{1.0, {0.0, 0.3, 0.6}, {0, 0, 0}};
  const double g = intti::three_electron_coulomb(a, b, c, d, e, f, grid);
  EXPECT_NEAR(g, 0.9594692137444143, 1e-11) << "general s-type 3-electron integral";
}

TEST(ThreeEl, Electron23SwapSymmetry) {
  // r12^{-1} r13^{-1} symmetric under exchanging electrons 2 and 3: (b,e)<->(c,f)
  auto grid = intti::make_tgrid(intti::coulomb());
  G a{0.9, {0.0, 0.0, 0.0}, {1, 0, 0}}, b{1.1, {0.4, 0.0, 0.0}, {0, 1, 0}},
      c{0.7, {0.0, 0.3, 0.5}, {0, 0, 1}}, d{1.3, {0.1, 0.0, 0.0}, {0, 0, 0}},
      e{0.8, {0.4, 0.1, 0.0}, {0, 0, 0}}, f{1.0, {0.0, 0.3, 0.6}, {0, 0, 0}};
  const double g1 = intti::three_electron_coulomb(a, b, c, d, e, f, grid);
  const double g2 = intti::three_electron_coulomb(a, c, b, d, f, e, grid);
  EXPECT_NEAR(g1, g2, 1e-13 * std::abs(g1));
}

// p-function integrals via the identity chi_{p_x} = (1/2 alpha) d/dA_x chi_s:
// the raw (unnormalised) l>0 integral must equal the finite difference of the
// s-integral w.r.t. that function's centre -- an independent check of the
// angular-momentum recursion.
TEST(ThreeEl, PFunctionVsCentreDerivative) {
  auto grid = intti::make_tgrid(intti::coulomb());
  const double za = 0.9, zb = 1.1, zc = 0.7, zd = 1.3, ze = 0.8, zf = 1.0;
  const double Ra[3] = {0.0, 0.0, 0.0}, Rb[3] = {0.4, 0.0, 0.0}, Rc[3] = {0.0, 0.3, 0.5};
  const double Rd[3] = {0.1, 0.0, 0.0}, Re[3] = {0.4, 0.1, 0.0}, Rf[3] = {0.0, 0.3, 0.6};
  auto sh = [&](const double *R, double z) { return G{z, {R[0], R[1], R[2]}, {0, 0, 0}}; };
  const double h = 1e-4;
  // helper: raw s-integral with function `which` (0=a..5=f) displaced in dir e
  struct FN {
    double z;
    const double *R;
  };
  FN fn[6] = {{za, Ra}, {zb, Rb}, {zc, Rc}, {zd, Rd}, {ze, Re}, {zf, Rf}};
  auto raw_s_disp = [&](int which, int dir, double delta) {
    G gg[6];
    for (int i = 0; i < 6; ++i) gg[i] = sh(fn[i].R, fn[i].z);
    gg[which].center[dir] += delta;
    return intti::detail::three_electron_raw(gg[0], gg[1], gg[2], gg[3], gg[4], gg[5], grid);
  };
  auto p_raw = [&](int which, int dir) {
    G gg[6];
    for (int i = 0; i < 6; ++i) gg[i] = sh(fn[i].R, fn[i].z);
    gg[which].l[dir] = 1;
    return intti::detail::three_electron_raw(gg[0], gg[1], gg[2], gg[3], gg[4], gg[5], grid);
  };
  // check p on the P density (function a, dir x), Q density (b, y), S density (c, z)
  const int cases[3][2] = {{0, 0}, {1, 1}, {2, 2}};
  const double zof[6] = {za, zb, zc, zd, ze, zf};
  for (auto &cs : cases) {
    const int which = cs[0], dir = cs[1];
    const double fd =
        (raw_s_disp(which, dir, h) - raw_s_disp(which, dir, -h)) / (2 * h) / (2 * zof[which]);
    const double p = p_raw(which, dir);
    EXPECT_NEAR(p, fd, 1e-6 * (std::abs(fd) + 1)) << "function " << which << " dir " << dir;
  }
}

} // namespace
