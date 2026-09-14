// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/nuclear.hpp"

namespace {

using Shell = intti::PrimitiveShell<double>;

double F0(double x) {
  return x < 1e-14 ? 1.0 : 0.5 * std::sqrt(M_PI / x) * std::erf(std::sqrt(x));
}

intti::ShellBasis<double> spd_basis() {
  return intti::make_basis<double>({
      {1.2, {0.0, 0.0, 0.0}, 0},
      {0.8, {0.0, 0.0, 0.0}, 1},
      {0.5, {0.3, -0.4, 0.9}, 0},
      {0.6, {0.3, -0.4, 0.9}, 2},
  });
}

TEST(Nuclear, SSAnalytic) {
  const double a = 0.8, b = 1.3;
  const double A[3] = {0.0, 0.1, -0.3}, B[3] = {0.5, -0.2, 0.4}, C[3] = {1.0, 0.8, 0.0};
  auto bas = intti::make_basis<double>({{a, {A[0], A[1], A[2]}, 0}, {b, {B[0], B[1], B[2]}, 0}});
  auto grid = intti::make_tgrid(intti::coulomb());
  // one unit charge at C, weight -1 (nuclear attraction sign)
  std::vector<intti::PointCharge<double>> ch{{-1.0, {C[0], C[1], C[2]}}};
  auto V = intti::nuclear_matrix(bas, ch, grid);
  const double p = a + b;
  double P[3], R2 = 0, mu = a * b / p, RAB2 = 0;
  for (int d = 0; d < 3; ++d) {
    P[d] = (a * A[d] + b * B[d]) / p;
    R2 += (P[d] - C[d]) * (P[d] - C[d]);
    RAB2 += (A[d] - B[d]) * (A[d] - B[d]);
  }
  const double exact = -(2 * M_PI / p) * std::exp(-mu * RAB2) * F0(p * R2);
  EXPECT_NEAR(V[0 * 2 + 1], exact, 1e-13 * std::abs(exact));
  EXPECT_NEAR(V[1 * 2 + 0], exact, 1e-13 * std::abs(exact)); // symmetric
}

TEST(Nuclear, ScreeningControlled) {
  auto bas = spd_basis();
  auto grid = intti::make_tgrid(intti::coulomb());
  std::vector<intti::PointCharge<double>> ch{{-1.0, {0.0, 0.0, 0.0}},
                                             {-6.0, {0.3, -0.4, 0.9}},
                                             {-1.0, {5.0, 5.0, 5.0}}}; // far
  auto V0 = intti::nuclear_matrix(bas, ch, grid, 0.0);
  auto V1 = intti::nuclear_matrix(bas, ch, grid, 1e-10);
  double md = 0, mx = 0;
  for (std::size_t i = 0; i < V0.size(); ++i) {
    md = std::max(md, std::abs(V0[i] - V1[i]));
    mx = std::max(mx, std::abs(V0[i]));
  }
  EXPECT_LT(md, 1e-8 * mx);
}

TEST(Nuclear, CollocationMatchesUnitCharge) {
  auto bas = spd_basis();
  auto grid = intti::make_tgrid(intti::coulomb());
  const std::array<double, 3> pt = {0.2, -0.1, 0.9};
  auto Vp = intti::potential_matrices(bas, {pt}, grid);
  std::vector<intti::PointCharge<double>> ch{{1.0, {pt[0], pt[1], pt[2]}}};
  auto Vn = intti::nuclear_matrix(bas, ch, grid);
  for (std::size_t i = 0; i < Vn.size(); ++i)
    EXPECT_NEAR(Vp[0][i], Vn[i], 1e-14);
  // symmetry
  const int n = bas.nao;
  double as = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      as = std::max(as, std::abs(Vp[0][i * n + j] - Vp[0][j * n + i]));
  EXPECT_LT(as, 1e-13);
}

// M-MP: the nuclear-attraction far-field (far_tau > 0) must reproduce the pure
// t-quadrature build (far_tau = 0) to ~far_tau when charges are well separated
// from the pairs. Mix of near (bonded) and far charges, with an spd basis.
TEST(Nuclear, FarFieldMatchesExact) {
  auto bas = spd_basis();
  auto grid = intti::make_tgrid(intti::coulomb());
  std::vector<intti::PointCharge<double>> ch{
      {-1.0, {0.0, 0.0, 1.1}},   // near
      {-6.0, {0.0, 0.0, 20.0}},  // far
      {-8.0, {25.0, -5.0, 0.0}}, // far
  };
  auto Vexact = intti::nuclear_matrix(bas, ch, grid, 0.0, /*far_tau=*/0.0);
  auto Vfar = intti::nuclear_matrix(bas, ch, grid, 0.0, /*far_tau=*/1e-13);
  double scale = 0;
  for (double v : Vexact) scale = std::max(scale, std::abs(v));
  for (std::size_t i = 0; i < Vexact.size(); ++i)
    EXPECT_LT(std::abs(Vfar[i] - Vexact[i]), 1e-10 * scale) << "elem " << i;

  // collocation far-field too (a distant grid point)
  std::array<double, 3> pt{18.0, 0.0, 0.0};
  auto Vp0 = intti::potential_matrices(bas, {pt}, grid, 0.0, 0.0);
  auto Vpf = intti::potential_matrices(bas, {pt}, grid, 0.0, 1e-13);
  double s2 = 0;
  for (double v : Vp0[0]) s2 = std::max(s2, std::abs(v));
  for (std::size_t i = 0; i < Vp0[0].size(); ++i)
    EXPECT_LT(std::abs(Vpf[0][i] - Vp0[0][i]), 1e-10 * s2) << "coll " << i;
}

} // namespace

// potential_on_points is the density-contracted form of potential_matrices:
// V(g) = sum_ab D_ab <a|1/|r-r_g||b>. Validating it AGAINST that routine is the
// point -- potential_matrices is already tested, and the new one exists only
// because its shape (nao^2 per point) cannot be put on a grid.
TEST(Nuclear, PotentialOnPointsMatchesTheCollocationMatrices) {
  auto basis = intti::make_basis<double>({{1.7, {0.0, 0.0, 0.0}, 0},
                                          {0.6, {0.0, 0.0, 0.0}, 1},
                                          {1.1, {0.3, -0.4, 1.2}, 0},
                                          {0.8, {0.3, -0.4, 1.2}, 1}});
  const int n = basis.nao;
  std::vector<double> D((std::size_t)n * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) D[i * n + j] = 0.2 + 0.4 * std::sin(0.9 * i + 1.7 * j);
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) {
      const double a = 0.5 * (D[i * n + j] + D[j * n + i]);
      D[i * n + j] = D[j * n + i] = a;
    }
  std::vector<std::array<double, 3>> pts;
  for (double x : {-2.5, -0.3, 0.0, 1.4, 6.0})
    for (double y : {-1.1, 0.2, 3.3})
      for (double z : {-0.7, 0.5, 4.8}) pts.push_back({x, y, z});
  auto grid = intti::make_tgrid(intti::coulomb());

  const auto ref = intti::potential_matrices(basis, pts, grid);
  std::vector<double> want(pts.size(), 0.0);
  for (std::size_t g = 0; g < pts.size(); ++g)
    for (int a = 0; a < n; ++a)
      for (int b = 0; b < n; ++b)
        want[g] += D[a * n + b] * ref[g][(std::size_t)a * n + b];

  double scale = 0;
  for (double v : want) scale = std::max(scale, std::abs(v));
  ASSERT_GT(scale, 1e-3);

  // exact branch, and the multipole far branch, must both reproduce it
  for (double far : {0.0, 1e-12, 1e-8}) {
    const auto got = intti::potential_on_points(basis, D.data(), pts, grid, 0.0, far);
    double worst = 0;
    for (std::size_t g = 0; g < pts.size(); ++g)
      worst = std::max(worst, std::abs(got[g] - want[g]));
    // the far branch is exact up to exp(-p R^2); 1e-8 is a loose tolerance and
    // still lands far inside it
    EXPECT_LT(worst, (far > 0 ? 1e-7 : 1e-12) * scale) << "far_tau=" << far;
  }
}

// The boxed far field expands the potential once per box instead of once per
// point. It measured no faster than the per-point far field and is not on the
// shipped path (see its documentation), but it is correct and is the M2L half
// of a hierarchical method, so its accuracy is pinned here: it must converge to
// the exact potential as the tolerance tightens.
TEST(Nuclear, BoxedFarFieldConvergesToTheExactPotential) {
  std::vector<intti::PrimitiveShell<double>> sh;
  for (int i = 0; i < 3; ++i) {
    sh.push_back({1.8, {2.6 * i, 0.0, 0.0}, 0});
    sh.push_back({0.6, {2.6 * i, 0.0, 0.0}, 1});
  }
  auto basis = intti::make_basis(sh);
  const int n = basis.nao;
  std::vector<double> D((std::size_t)n * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) D[i * n + j] = 0.2 + 0.3 * std::cos(0.7 * i + 1.3 * j);
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) {
      const double a = 0.5 * (D[i * n + j] + D[j * n + i]);
      D[i * n + j] = D[j * n + i] = a;
    }

  const int nb = 4, per = 4;
  const double lo = -5.0, hi = 10.0, bw = (hi - lo) / nb;
  std::vector<std::array<double, 3>> pts;
  intti::PointBoxes<double> boxes;
  boxes.start.push_back(0);
  for (int bx = 0; bx < nb; ++bx)
    for (int by = 0; by < nb; ++by)
      for (int bz = 0; bz < nb; ++bz) {
        const double cx = lo + (bx + 0.5) * bw, cy = lo + (by + 0.5) * bw,
                     cz = lo + (bz + 0.5) * bw;
        double rad = 0;
        for (int i = 0; i < per; ++i)
          for (int j = 0; j < per; ++j)
            for (int k = 0; k < per; ++k) {
              const double x = lo + bx * bw + (i + 0.5) * bw / per;
              const double y = lo + by * bw + (j + 0.5) * bw / per;
              const double z = lo + bz * bw + (k + 0.5) * bw / per;
              pts.push_back({x, y, z});
              rad = std::max(rad, std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) +
                                            (z - cz) * (z - cz)));
            }
        boxes.center.push_back({cx, cy, cz});
        boxes.radius.push_back(rad);
        boxes.start.push_back((int)pts.size());
      }

  auto tg = intti::make_tgrid(intti::coulomb());
  const auto exact = intti::potential_on_points(basis, D.data(), pts, tg, 0.0, 0.0);
  double scale = 0;
  for (double v : exact) scale = std::max(scale, std::abs(v));
  ASSERT_GT(scale, 1e-3);

  double prev = 1.0;
  for (double ft : {1e-6, 1e-9, 1e-12}) {
    const auto got =
        intti::potential_on_points_boxed(basis, D.data(), pts, boxes, tg, 0.0, ft, 10);
    double worst = 0;
    for (std::size_t g = 0; g < pts.size(); ++g)
      worst = std::max(worst, std::abs(got[g] - exact[g]));
    EXPECT_LT(worst, 30 * ft * scale) << "boxed far field beyond its tolerance, far_tau=" << ft;
    EXPECT_LE(worst, prev * 1.001) << "tightening far_tau made it worse at " << ft;
    prev = std::max(worst, 1e-16);
  }
}
