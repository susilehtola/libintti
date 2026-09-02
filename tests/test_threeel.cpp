// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/threeel.hpp"

namespace {

using G = intti::CartGauss<double>;

TEST(ThreeEl, OneCentreAnalytic) {
  // one-centre three-electron Coulomb of six identical s-GTOs: G_aaaaaa = 4z/3
  // (derived symbolically in references/sympy_three_electron.py)
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

TEST(ThreeEl, GaussianGeminalOneCentreAnalytic) {
  // both operators Gaussian geminals e^{-gamma r^2}, e^{-delta r^2}; one centre,
  // identical s-GTOs exponent z. No quadrature -- the analytic Gaussian result
  // G = N pi^{9/2}/[(a+LQ+LS)(a+g)(a+d)]^{3/2}, a = 2z, LQ = a g/(a+g), etc.
  const double z = 1.1, gamma = 0.7, delta = 1.3;
  G a{z, {0.0, 0.0, 0.0}, {0, 0, 0}};
  auto op12 = intti::detail::gaussian_nodes<double>({1.0}, {gamma});
  auto op13 = intti::detail::gaussian_nodes<double>({1.0}, {delta});
  const double got = intti::three_electron(a, a, a, a, a, a, op12, op13);
  const double al = 2 * z, LQ = al * gamma / (al + gamma), LS = al * delta / (al + delta);
  const double N0 = std::pow(2 * z / M_PI, 0.75); // s-type norm per function
  const double ana = std::pow(N0, 6) * std::pow(M_PI, 4.5) /
                     std::pow((al + LQ + LS) * (al + gamma) * (al + delta), 1.5);
  EXPECT_NEAR(got, ana, 1e-13 * ana);
}

TEST(ThreeEl, MixedCoulombGaussianPFunctionVsFD) {
  // r12^{-1} on the 1-2 pair, a Gaussian geminal e^{-gamma r13^2} on the 1-3
  // pair. A p-function on any density is (1/2 alpha) d/dcentre of the s-integral,
  // for ANY operator -- so this validates the mixed operator and l>0 at once.
  auto grid = intti::make_tgrid(intti::coulomb());
  const double gamma = 0.6;
  auto op12 = intti::detail::coulomb_nodes(grid);
  auto op13 = intti::detail::gaussian_nodes<double>({1.0}, {gamma});
  const double za = 0.9, zb = 1.1, zc = 0.7, zd = 1.3, ze = 0.8, zf = 1.0;
  const double R[6][3] = {{0, 0, 0},   {0.4, 0, 0}, {0, 0.3, 0.5},
                          {0.1, 0, 0}, {0.4, 0.1, 0}, {0, 0.3, 0.6}};
  const double zof[6] = {za, zb, zc, zd, ze, zf};
  auto build = [&](int which, int dir, int power, double delta) {
    G g[6];
    for (int i = 0; i < 6; ++i) g[i] = G{zof[i], {R[i][0], R[i][1], R[i][2]}, {0, 0, 0}};
    if (power) g[which].l[dir] = 1;
    g[which].center[dir] += delta;
    return intti::detail::three_electron_raw_nodes(g[0], g[1], g[2], g[3], g[4], g[5], op12, op13);
  };
  const double h = 1e-4;
  for (int which : {0, 1, 2}) { // P, Q, S densities
    const int dir = which;
    const double fd = (build(which, dir, 0, h) - build(which, dir, 0, -h)) / (2 * h) /
                      (2 * zof[which]);
    const double p = build(which, dir, 1, 0.0);
    EXPECT_NEAR(p, fd, 1e-6 * (std::abs(fd) + 1)) << "mixed op, function " << which;
  }
}

TEST(ThreeEl, GeminalOverRReducesToCoulomb) {
  // (f/r) with f = e^{-0 r^2} = 1 is exactly the Coulomb node list, so
  // f12/r12 * r13^{-1} must equal r12^{-1} r13^{-1}.
  auto grid = intti::make_tgrid(intti::coulomb());
  G a{0.9, {0.0, 0.0, 0.0}, {1, 0, 0}}, b{1.1, {0.4, 0.0, 0.0}, {0, 0, 0}},
      c{0.7, {0.0, 0.3, 0.5}, {0, 1, 0}}, d{1.3, {0.1, 0.0, 0.0}, {0, 0, 0}},
      e{0.8, {0.4, 0.1, 0.0}, {0, 0, 0}}, f{1.0, {0.0, 0.3, 0.6}, {0, 0, 1}};
  auto fr = intti::detail::geminal_over_r_nodes<double>({1.0}, {0.0}, grid);
  auto co = intti::detail::coulomb_nodes(grid);
  const double g1 = intti::three_electron(a, b, c, d, e, f, fr, co);
  const double g2 = intti::three_electron_coulomb(a, b, c, d, e, f, grid);
  EXPECT_NEAR(g1, g2, 1e-12 * std::abs(g2));
}

TEST(ThreeEl, GeminalOverRPFunctionVsFD) {
  // f12/r12 (Gaussian geminal over r) on the 1-2 pair, Coulomb on 1-3;
  // validate l>0 on all three densities via the centre-derivative identity.
  auto grid = intti::make_tgrid(intti::coulomb());
  auto op12 = intti::detail::geminal_over_r_nodes<double>({0.6, 0.3}, {0.4, 1.5}, grid);
  auto op13 = intti::detail::coulomb_nodes(grid);
  const double zof[6] = {0.9, 1.1, 0.7, 1.3, 0.8, 1.0};
  const double R[6][3] = {{0, 0, 0},   {0.4, 0, 0},   {0, 0.3, 0.5},
                          {0.1, 0, 0}, {0.4, 0.1, 0}, {0, 0.3, 0.6}};
  auto build = [&](int which, int dir, int power, double delta) {
    G g[6];
    for (int i = 0; i < 6; ++i) g[i] = G{zof[i], {R[i][0], R[i][1], R[i][2]}, {0, 0, 0}};
    if (power) g[which].l[dir] = 1;
    g[which].center[dir] += delta;
    return intti::detail::three_electron_raw_nodes(g[0], g[1], g[2], g[3], g[4], g[5], op12,
                                                   op13);
  };
  const double h = 1e-4;
  for (int which : {0, 1, 2}) {
    const int dir = which;
    const double fd =
        (build(which, dir, 0, h) - build(which, dir, 0, -h)) / (2 * h) / (2 * zof[which]);
    EXPECT_NEAR(build(which, dir, 1, 0.0), fd, 1e-6 * (std::abs(fd) + 1)) << "which " << which;
  }
}

TEST(ThreeEl, R2MomentVsExponentDerivative) {
  // r12^2 e^{-g r12^2} = -d/dg e^{-g r12^2}, so the moment integral must equal
  // -d/dg of the Gaussian-geminal integral. Checked s-type and every l>0
  // p-function (P, Q, S densities), with Coulomb on the 1-3 pair.
  auto grid = intti::make_tgrid(intti::coulomb());
  auto op13 = intti::detail::coulomb_nodes(grid);
  const double gam = 0.7;
  const double zof[6] = {0.9, 1.1, 0.7, 1.3, 0.8, 1.0};
  const double R[6][3] = {{0, 0, 0},   {0.4, 0, 0},   {0, 0.3, 0.5},
                          {0.1, 0, 0}, {0.4, 0.1, 0}, {0, 0.3, 0.6}};
  auto shells = [&](int which, int dir, int power) {
    std::array<G, 6> g;
    for (int i = 0; i < 6; ++i) g[i] = G{zof[i], {R[i][0], R[i][1], R[i][2]}, {0, 0, 0}};
    if (power) g[which].l[dir] = 1;
    return g;
  };
  const double h = 1e-5;
  for (int which : {-1, 0, 1, 2}) { // -1 = pure s-type; 0,1,2 = p on P,Q,S
    auto g = shells(which < 0 ? 0 : which, which < 0 ? 0 : which, which < 0 ? 0 : 1);
    auto mom = intti::detail::three_electron_raw_moment12(
        g[0], g[1], g[2], g[3], g[4], g[5], intti::detail::gaussian_nodes<double>({1.0}, {gam}),
        op13);
    auto ig = [&](double gg) {
      return intti::detail::three_electron_raw_nodes(
          g[0], g[1], g[2], g[3], g[4], g[5],
          intti::detail::gaussian_nodes<double>({1.0}, {gg}), op13);
    };
    const double fd = -(ig(gam + h) - ig(gam - h)) / (2 * h);
    EXPECT_NEAR(mom, fd, 1e-5 * (std::abs(fd) + 1)) << "case " << which;
  }
}

TEST(ThreeEl, ThreeBodyEnergyContraction) {
  // matrix-level: E = sum_{abcdef} G_{abcdef} D_ad D_be D_cf.
  auto grid = intti::make_tgrid(intti::coulomb());
  auto op = intti::detail::coulomb_nodes(grid);
  // single s-function, D = [[c]]: E = c^3 G_aaaaaa = c^3 * 4 zeta/3
  {
    const double z = 1.3, c = 1.5;
    std::vector<G> basis = {G{z, {0.0, 0.0, 0.0}, {0, 0, 0}}};
    std::vector<double> D = {c};
    const double E = intti::three_electron_energy(basis, D, op, op);
    EXPECT_NEAR(E, c * c * c * 4 * z / 3, 1e-11 * std::abs(E));
  }
  // two s-functions vs independent reference
  {
    std::vector<G> basis = {G{1.0, {0.0, 0.0, 0.0}, {0, 0, 0}},
                            G{0.8, {0.5, 0.0, 0.0}, {0, 0, 0}}};
    std::vector<double> D = {1.0, 0.2, 0.2, 0.7};
    const double E = intti::three_electron_energy(basis, D, op, op);
    EXPECT_NEAR(E, 9.674661513439341, 1e-9);
  }
}

} // namespace
