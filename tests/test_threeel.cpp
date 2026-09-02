// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "intti/threeel.hpp"
#include "threeel_reference.hpp"

namespace {

using G = intti::CartGauss<double>;

// Build a CartGauss from an independent-reference Fn record.
G from_ref(const te_ref::Fn &r) {
  return G{r.alpha, {r.c[0], r.c[1], r.c[2]}, {r.l[0], r.l[1], r.l[2]}};
}

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
  // Independent reference: reduce electrons 2,3 to their Gaussian (erf) Coulomb
  // potentials, leaving a single 3D integral over r1, brute-forced with scipy
  // (uses none of the (t,s)-quadrature machinery). See references/README.md.
  EXPECT_NEAR(g, 1.009834732922867, 1e-10) << "general s-type 3-electron integral";
}

TEST(ThreeEl, Electron23SwapSymmetry) {
  // r12^{-1} r13^{-1} symmetric under exchanging electrons 2 and 3: (b,e)<->(c,f)
  auto grid = intti::make_tgrid(intti::coulomb());
  G a{0.9, {0.0, 0.0, 0.0}, {1, 0, 0}}, b{1.1, {0.4, 0.0, 0.0}, {0, 1, 0}},
      c{0.7, {0.0, 0.3, 0.5}, {0, 0, 1}}, d{1.3, {0.1, 0.0, 0.0}, {0, 0, 0}},
      e{0.8, {0.4, 0.1, 0.0}, {0, 0, 0}}, f{1.0, {0.0, 0.3, 0.6}, {0, 0, 0}};
  const double g1 = intti::three_electron_coulomb(a, b, c, d, e, f, grid);
  const double g2 = intti::three_electron_coulomb(a, c, b, d, f, e, grid);
  EXPECT_NEAR(g1, g2, 1e-11 * (std::abs(g1) + 1)); // 3x3 solve breaks exact bit symmetry
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
    // 4th-order central difference (h from the enclosing scope is 1e-4 -> too
    // coarse for 1e-6; a 5-point stencil reaches ~1e-11 with the same step).
    const double fd = (raw_s_disp(which, dir, -2 * h) - 8 * raw_s_disp(which, dir, -h) +
                       8 * raw_s_disp(which, dir, h) - raw_s_disp(which, dir, 2 * h)) /
                      (12 * h) / (2 * zof[which]);
    const double p = p_raw(which, dir);
    // Coarse cross-check: finite-differencing the grid-based Coulomb integral in
    // double precision has a ~1e-6 noise floor. The tight, independent l>0 and
    // derivative validation lives in ThreeEl.IndependentReference* below.
    EXPECT_NEAR(p, fd, 2e-5 * (std::abs(fd) + 1)) << "function " << which << " dir " << dir;
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

TEST(ThreeEl, HigherLGaussianGeminalVsSympy) {
  // Independent l>0 oracle: the raw (unnormalised) Gaussian-geminal integral
  // with a p_x on the P density (a), p_y on Q (b), p_z on S (c), general centres
  // and exponents, is a polynomial x Gaussian moment computed exactly in
  // references/sympy_three_electron.py (closed-form multivariate-normal moment).
  // This does not rely on the finite-difference centre-derivative identity.
  G a{0.9, {0.1, 0.0, 0.0}, {1, 0, 0}};
  G b{1.1, {0.3, 0.0, 0.0}, {0, 1, 0}};
  G c{1.0, {0.0, 0.4, 0.0}, {0, 0, 1}};
  G d{0.7, {0.0, 0.2, 0.0}, {0, 0, 0}};
  G e{0.8, {0.0, 0.0, 0.1}, {0, 0, 0}};
  G f{0.6, {0.2, 0.0, 0.0}, {0, 0, 0}};
  auto op12 = intti::detail::gaussian_nodes<double>({1.0}, {0.5});
  auto op13 = intti::detail::gaussian_nodes<double>({1.0}, {0.7});
  const double got = intti::detail::three_electron_raw_nodes(a, b, c, d, e, f, op12, op13);
  EXPECT_NEAR(got, -2.7750197635878001e-6, 1e-11 * 2.78e-6);
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
    const double fd = (build(which, dir, 0, -2 * h) - 8 * build(which, dir, 0, -h) +
                       8 * build(which, dir, 0, h) - build(which, dir, 0, 2 * h)) /
                      (12 * h) / (2 * zof[which]);
    const double p = build(which, dir, 1, 0.0);
    EXPECT_NEAR(p, fd, 2e-5 * (std::abs(fd) + 1)) << "mixed op, function " << which; // grid-FD floor
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
    const double fd = (build(which, dir, 0, -2 * h) - 8 * build(which, dir, 0, -h) +
                       8 * build(which, dir, 0, h) - build(which, dir, 0, 2 * h)) /
                      (12 * h) / (2 * zof[which]);
    EXPECT_NEAR(build(which, dir, 1, 0.0), fd, 2e-5 * (std::abs(fd) + 1)) << "which " << which;
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
  const double h = 1e-3; // 4th-order in the exponent; the grid sum is noisy, so
                         // a wider step + higher-order stencil beats a tiny h.
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
    const double fd = -(ig(gam - 2 * h) - 8 * ig(gam - h) + 8 * ig(gam + h) - ig(gam + 2 * h)) /
                      (12 * h);
    EXPECT_NEAR(mom, fd, 5e-6 * (std::abs(fd) + 1)) << "case " << which; // grid-FD floor
  }
}

TEST(ThreeEl, IndependentReferenceIntegralsAndMoments) {
  // Data-driven: every l>0 / many-centre Gaussian-geminal integral and r12^2
  // moment in tests/threeel_reference.hpp, generated by an independent
  // Gauss-Hermite quadrature (references/te_reference.py, anchored to scipy),
  // must be reproduced by the engine.
  auto op12 = intti::detail::gaussian_nodes<double>({1.0}, {te_ref::gamma_op});
  auto op13 = intti::detail::gaussian_nodes<double>({1.0}, {te_ref::delta_op});
  for (const auto &cs : te_ref::cases) {
    G g[6];
    for (int i = 0; i < 6; ++i) g[i] = from_ref(cs.f[i]);
    const double gi = intti::three_electron(g[0], g[1], g[2], g[3], g[4], g[5], op12, op13);
    EXPECT_NEAR(gi, cs.integral, 1e-10 * (std::abs(cs.integral) + 1)) << "integral " << cs.name;
    const double gm =
        intti::three_electron_moment12(g[0], g[1], g[2], g[3], g[4], g[5], op12, op13);
    EXPECT_NEAR(gm, cs.moment, 1e-10 * (std::abs(cs.moment) + 1)) << "moment " << cs.name;
  }
}

TEST(ThreeEl, IndependentReferenceDerivatives) {
  // Data-driven centre derivatives: the engine forms d/dA of the raw integral
  // analytically via the McMurchie-Davidson shift 2 alpha raw(l+1) - l raw(l-1),
  // checked against the independent finite-difference value in the header.
  auto op12 = intti::detail::gaussian_nodes<double>({1.0}, {te_ref::gamma_op});
  auto op13 = intti::detail::gaussian_nodes<double>({1.0}, {te_ref::delta_op});
  for (const auto &dc : te_ref::dcases) {
    const te_ref::Case *base = nullptr;
    for (const auto &cs : te_ref::cases)
      if (std::strcmp(cs.name, dc.name) == 0) base = &cs;
    ASSERT_NE(base, nullptr) << dc.name;
    G g[6];
    for (int i = 0; i < 6; ++i) g[i] = from_ref(base->f[i]);
    const int w = dc.which, dim = dc.dim, l = g[w].l[dim];
    const double alpha = g[w].alpha;
    auto raw_with = [&](int lv) {
      G gg[6];
      for (int i = 0; i < 6; ++i) gg[i] = g[i];
      gg[w].l[dim] = lv;
      return intti::detail::three_electron_raw_nodes(gg[0], gg[1], gg[2], gg[3], gg[4], gg[5], op12,
                                                     op13);
    };
    const double d = 2 * alpha * raw_with(l + 1) - (l > 0 ? l * raw_with(l - 1) : 0.0);
    EXPECT_NEAR(d, dc.value, 1e-7 * (std::abs(dc.value) + 1)) << "deriv " << dc.name;
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
  // two s-functions vs an independent reference: the same sextet sum evaluated
  // with each Coulomb integral reduced to the erf-potential 3D form and a
  // converged Gauss-Hermite quadrature (references/, independent of the engine).
  {
    std::vector<G> basis = {G{1.0, {0.0, 0.0, 0.0}, {0, 0, 0}},
                            G{0.8, {0.5, 0.0, 0.0}, {0, 0, 0}}};
    std::vector<double> D = {1.0, 0.2, 0.2, 0.7};
    const double E = intti::three_electron_energy(basis, D, op, op);
    EXPECT_NEAR(E, 9.94368873264104, 1e-8);
  }
}

TEST(ThreeEl, ThreeBodyFockIsEnergyGradient) {
  // The effective one-body matrix F = dE/dD. Two independent checks: the cubic
  // homogeneity identity sum_pq F_pq D_pq = 3 E (exact), and F_pq vs a finite
  // difference of the energy in that density entry.
  auto grid = intti::make_tgrid(intti::coulomb());
  auto op = intti::detail::coulomb_nodes(grid);
  const int n = 2;
  std::vector<G> basis = {G{1.0, {0.0, 0.0, 0.0}, {0, 0, 0}},
                          G{0.8, {0.5, 0.0, 0.0}, {0, 0, 0}}};
  std::vector<double> D = {1.0, 0.2, 0.3, 0.7}; // general (non-symmetric)
  const auto F = intti::three_electron_fock(basis, D, op, op);
  const double E = intti::three_electron_energy(basis, D, op, op);
  double trace = 0;
  for (int i = 0; i < n * n; ++i) trace += F[i] * D[i];
  EXPECT_NEAR(trace, 3 * E, 1e-9 * std::abs(E)) << "Euler identity sum F D = 3E";
  const double h = 1e-3;
  auto energy = [&](const std::vector<double> &Dm) {
    return intti::three_electron_energy(basis, Dm, op, op);
  };
  for (int p = 0; p < n; ++p)
    for (int q = 0; q < n; ++q) {
      auto shift = [&](double s) {
        std::vector<double> Dm = D;
        Dm[p * n + q] += s;
        return energy(Dm);
      };
      const double fd =
          (shift(-2 * h) - 8 * shift(-h) + 8 * shift(h) - shift(2 * h)) / (12 * h);
      EXPECT_NEAR(F[p * n + q], fd, 1e-7 * (std::abs(fd) + 1)) << "F[" << p << "," << q << "]";
    }
}

} // namespace
