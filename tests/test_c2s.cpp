// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/c2s.hpp"
#include "intti/quartet.hpp"

namespace {

TEST(C2S, RowsAreSphereOrthonormal) {
  // int S_m S_m' dOmega / (4 pi) = delta_mm' by construction
  auto sphere = [](int a, int b, int c) -> double {
    if (a % 2 || b % 2 || c % 2) return 0.0;
    auto dfact = [](int n) {
      double r = 1;
      for (int k = n; k > 1; k -= 2)
        r *= k;
      return r;
    };
    return dfact(a - 1) * dfact(b - 1) * dfact(c - 1) / dfact(a + b + c + 1);
  };
  for (int l = 0; l <= 4; ++l) {
    auto C = intti::c2s_matrix<double>(l);
    const int nc = intti::ncart(l), nm = 2 * l + 1;
    for (int m = 0; m < nm; ++m)
      for (int m2 = 0; m2 <= m; ++m2) {
        double s = 0;
        for (int k = 0; k < nc; ++k)
          for (int k2 = 0; k2 < nc; ++k2) {
            int lx, ly, lz, lx2, ly2, lz2;
            intti::cart_comp(l, k, lx, ly, lz);
            intti::cart_comp(l, k2, lx2, ly2, lz2);
            s += C[m * nc + k] * C[m2 * nc + k2] *
                 sphere(lx + lx2, ly + ly2, lz + lz2);
          }
        EXPECT_NEAR(s, m == m2 ? 1.0 : 0.0, 1e-13) << "l=" << l << " m=" << m
                                                   << " m'=" << m2;
      }
  }
}

TEST(C2S, KnownShapes) {
  // l = 1: rows m = -1, 0, +1 are y, z, x
  auto C1 = intti::c2s_matrix<double>(1);
  EXPECT_NEAR(std::abs(C1[0 * 3 + 1]), std::sqrt(3.0), 1e-14); // m=-1: y
  EXPECT_NEAR(std::abs(C1[1 * 3 + 2]), std::sqrt(3.0), 1e-14); // m=0:  z
  EXPECT_NEAR(std::abs(C1[2 * 3 + 0]), std::sqrt(3.0), 1e-14); // m=+1: x
  // l = 2, m = 0: ratio xx : yy : zz = -1 : -1 : 2, no cross terms
  auto C2 = intti::c2s_matrix<double>(2);
  const int nc = 6; // xx xy xz yy yz zz
  const double *m0 = &C2[2 * nc];
  EXPECT_NEAR(m0[0] / m0[5], -0.5, 1e-14);
  EXPECT_NEAR(m0[3] / m0[5], -0.5, 1e-14);
  EXPECT_NEAR(m0[1], 0.0, 1e-14);
  EXPECT_NEAR(m0[2], 0.0, 1e-14);
  EXPECT_NEAR(m0[4], 0.0, 1e-14);
  // l = 2, m = +2: xx - yy; m = -2: xy only
  const double *p2 = &C2[4 * nc];
  EXPECT_NEAR(p2[0] + p2[3], 0.0, 1e-14);
  EXPECT_NEAR(p2[1], 0.0, 1e-14);
  const double *mm2 = &C2[0 * nc];
  EXPECT_NEAR(mm2[0], 0.0, 1e-14);
  EXPECT_NEAR(mm2[3], 0.0, 1e-14);
  EXPECT_GT(std::abs(mm2[1]), 0.1);
}

// m-summed spherical invariant of a (dd|dd) block: T = sum_{mm'} (mm|m'm')
double dd_invariant(const double *A, const double *B) {
  using Shell = intti::PrimitiveShell<double>;
  Shell a{0.9, {A[0], A[1], A[2]}, 2}, b{1.3, {B[0], B[1], B[2]}, 2};
  auto pair = intti::make_pair(a, b);
  auto grid = intti::make_tgrid(intti::coulomb());
  const int nc = 6, nm = 5;
  std::vector<double> cart(nc * nc * nc * nc);
  intti::eri_quartet(pair, pair, grid, cart.data());
  auto C = intti::c2s_matrix<double>(2);
  // transform all four indices
  // transform index idx: earlier indices are already spherical (nm), later
  // ones still Cartesian (nc); reshape as (pre, nc, post)
  auto trans = [&](std::vector<double> &v, int idx) {
    int pre = 1, post = 1;
    for (int i = 0; i < idx; ++i)
      pre *= nm; // already-transformed leading indices
    for (int i = idx + 1; i < 4; ++i)
      post *= nc;
    std::vector<double> r(static_cast<std::size_t>(pre) * nm * post);
    for (int p = 0; p < pre; ++p)
      for (int m = 0; m < nm; ++m)
        for (int q = 0; q < post; ++q) {
          double s = 0;
          for (int k = 0; k < nc; ++k)
            s += C[m * nc + k] * v[(p * nc + k) * static_cast<std::size_t>(post) + q];
          r[(p * nm + m) * static_cast<std::size_t>(post) + q] = s;
        }
    v = std::move(r);
  };
  trans(cart, 0);
  trans(cart, 1);
  trans(cart, 2);
  trans(cart, 3);
  double T = 0;
  for (int m = 0; m < nm; ++m)
    for (int m2 = 0; m2 < nm; ++m2)
      T += cart[((m * nm + m) * nm + m2) * nm + m2];
  return T;
}

TEST(C2S, RotationalInvariance) {
  const double A[3] = {0.0, 0.0, 0.0}, B[3] = {0.9, -0.4, 1.1};
  // rotate by 40 degrees about a skew axis
  const double ang = 40.0 * M_PI / 180.0;
  const double c = std::cos(ang), s = std::sin(ang);
  // rotation about the (1,1,1)/sqrt(3) axis (Rodrigues)
  const double u = 1.0 / std::sqrt(3.0);
  double R[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      R[i][j] = c * (i == j) + (1 - c) * u * u +
                s * ((i - j + 3) % 3 == 1 ? u : ((j - i + 3) % 3 == 1 ? -u : 0));
  double Br[3] = {0, 0, 0}, Ar[3] = {0, 0, 0};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      Br[i] += R[i][j] * B[j];
      Ar[i] += R[i][j] * A[j];
    }
  const double T0 = dd_invariant(A, B);
  const double T1 = dd_invariant(Ar, Br);
  EXPECT_NEAR(T1, T0, 1e-12 * std::abs(T0));
}

} // namespace
