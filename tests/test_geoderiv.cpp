// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola

#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/deriv.hpp"
#include "intti/geoderiv.hpp"
#include "intti/nuclear.hpp"
#include "intti/tgrid.hpp"

namespace {

using Shell = intti::PrimitiveShell<double>;

std::vector<Shell> shells(const double A[3], const double B[3]) {
  return {{0.9, {A[0], A[1], A[2]}, 1},
          {1.3, {B[0], B[1], B[2]}, 0},
          {0.7, {B[0], B[1], B[2]}, 2}};
}

const double A0[3] = {0.1, -0.2, 0.3}, B0[3] = {0.5, 0.4, -0.6};

// overlap matrix with bra centre A (shell 0) and ket centre B (shells 1,2)
std::vector<double> Smat(const double A[3], const double B[3]) {
  return intti::overlap_matrix(intti::make_basis(shells(A, B)));
}
std::vector<double> Tmat(const double A[3], const double B[3]) {
  return intti::kinetic_matrix(intti::make_basis(shells(A, B)));
}

TEST(GeoDeriv, Order1MatchesGradient) {
  // <nabla_x mu|nu> (the shift) must equal M16a overlap_deriv
  auto bas = intti::make_basis(shells(A0, B0));
  auto Gref = intti::overlap_deriv(bas);
  for (int d = 0; d < 3; ++d) {
    std::array<int, 3> na{0, 0, 0}, nb{0, 0, 0};
    na[d] = 1;
    auto G = intti::overlap_geoderiv(bas, na, nb);
    const int n = bas.nao;
    double md = 0, mx = 0;
    for (std::size_t i = 0; i < G.size(); ++i) {
      md = std::max(md, std::abs(G[i] - Gref[d][i]));
      mx = std::max(mx, std::abs(Gref[d][i]));
    }
    EXPECT_LT(md, 1e-12 * mx) << "d=" << d;
  }
}

// second mixed difference of S moving centre-1 comp d1 and centre-2 comp d2
template <class Fmat>
double fd2(Fmat Sof, int idx, int nao, int d1, bool bra1, int d2, bool bra2, double h) {
  auto shift = [&](double A[3], double B[3], int d, bool bra, double s) {
    (bra ? A : B)[d] += s;
  };
  double Ap[3], Bp[3];
  auto ev = [&](double s1, double s2) {
    for (int k = 0; k < 3; ++k) { Ap[k] = A0[k]; Bp[k] = B0[k]; }
    shift(Ap, Bp, d1, bra1, s1);
    shift(Ap, Bp, d2, bra2, s2);
    return Sof(Ap, Bp)[idx];
  };
  return (ev(h, h) - ev(h, -h) - ev(-h, h) + ev(-h, -h)) / (4 * h * h);
}

TEST(GeoDeriv, HessianBraKetVsFiniteDifference) {
  // d/dA_x d/dB_y S = nabla^A_x nabla^B_y S = geoderiv(na=(1,0,0), nb=(0,1,0))
  auto bas = intti::make_basis(shells(A0, B0));
  const int nao = bas.nao;
  auto G = intti::overlap_geoderiv(bas, {1, 0, 0}, {0, 1, 0});
  const double h = 1e-4;
  for (int mu = 0; mu < 3; ++mu)          // bra shell 0 (p): AOs 0..2
    for (int nu = 3; nu < nao; ++nu) {    // ket shells 1,2
      const int idx = mu * nao + nu;
      const double fd = fd2(Smat, idx, nao, 0, true, 1, false, h); // A_x, B_y
      EXPECT_NEAR(G[idx], fd, 1e-5 * (std::abs(fd) + 1.0)) << "mu=" << mu << " nu=" << nu;
    }
}

TEST(GeoDeriv, HessianBraBraVsFiniteDifference) {
  // d/dA_x d/dA_y S = geoderiv(na=(1,1,0), nb=0)
  auto bas = intti::make_basis(shells(A0, B0));
  const int nao = bas.nao;
  auto G = intti::overlap_geoderiv(bas, {1, 1, 0}, {0, 0, 0});
  const double h = 1e-4;
  for (int mu = 0; mu < 3; ++mu)
    for (int nu = 3; nu < nao; ++nu) {
      const int idx = mu * nao + nu;
      const double fd = fd2(Smat, idx, nao, 0, true, 1, true, h); // A_x, A_y
      EXPECT_NEAR(G[idx], fd, 1e-5 * (std::abs(fd) + 1.0)) << "mu=" << mu << " nu=" << nu;
    }
}

TEST(GeoDeriv, KineticOrder1MatchesGradient) {
  auto bas = intti::make_basis(shells(A0, B0));
  auto Gref = intti::kinetic_deriv(bas);
  for (int d = 0; d < 3; ++d) {
    std::array<int, 3> na{0, 0, 0}, nb{0, 0, 0};
    na[d] = 1;
    auto G = intti::kinetic_geoderiv(bas, na, nb);
    double md = 0, mx = 0;
    for (std::size_t i = 0; i < G.size(); ++i) {
      md = std::max(md, std::abs(G[i] - Gref[d][i]));
      mx = std::max(mx, std::abs(Gref[d][i]));
    }
    EXPECT_LT(md, 1e-12 * mx) << "d=" << d;
  }
}

TEST(GeoDeriv, KineticHessianVsFiniteDifference) {
  auto bas = intti::make_basis(shells(A0, B0));
  const int nao = bas.nao;
  auto Gbk = intti::kinetic_geoderiv(bas, {1, 0, 0}, {0, 1, 0}); // A_x B_y
  auto Gbb = intti::kinetic_geoderiv(bas, {1, 1, 0}, {0, 0, 0}); // A_x A_y
  const double h = 1e-4;
  for (int mu = 0; mu < 3; ++mu)
    for (int nu = 3; nu < nao; ++nu) {
      const int idx = mu * nao + nu;
      const double fbk = fd2(Tmat, idx, nao, 0, true, 1, false, h);
      const double fbb = fd2(Tmat, idx, nao, 0, true, 1, true, h);
      EXPECT_NEAR(Gbk[idx], fbk, 1e-5 * (std::abs(fbk) + 1.0)) << "bk mu=" << mu;
      EXPECT_NEAR(Gbb[idx], fbb, 1e-5 * (std::abs(fbb) + 1.0)) << "bb mu=" << mu;
    }
}

TEST(GeoDeriv, NuclearOrder1AndHessian) {
  auto grid = intti::make_tgrid(intti::coulomb());
  std::vector<intti::PointCharge<double>> ch{{-1.0, {0.0, 0.0, 0.9}}};
  auto bas = intti::make_basis(shells(A0, B0));
  const int nao = bas.nao;
  // order 1 == M16b nuclear_deriv
  auto Gref = intti::nuclear_deriv(bas, ch, grid);
  for (int d = 0; d < 3; ++d) {
    std::array<int, 3> na{0, 0, 0}, nb{0, 0, 0};
    na[d] = 1;
    auto G = intti::nuclear_geoderiv(bas, ch, grid, na, nb);
    double md = 0, mx = 0;
    for (std::size_t i = 0; i < G.size(); ++i) {
      md = std::max(md, std::abs(G[i] - Gref[d][i]));
      mx = std::max(mx, std::abs(Gref[d][i]));
    }
    EXPECT_LT(md, 1e-11 * mx) << "d=" << d;
  }
  // Hessian bra-ket and bra-bra vs finite difference of nuclear_matrix
  auto Vof = [&](const double A[3], const double B[3]) {
    return intti::nuclear_matrix(intti::make_basis(shells(A, B)), ch, grid);
  };
  auto Gbk = intti::nuclear_geoderiv(bas, ch, grid, {1, 0, 0}, {0, 1, 0});
  auto Gbb = intti::nuclear_geoderiv(bas, ch, grid, {1, 1, 0}, {0, 0, 0});
  const double h = 1e-4;
  for (int mu = 0; mu < 3; ++mu)
    for (int nu = 3; nu < nao; ++nu) {
      const int idx = mu * nao + nu;
      EXPECT_NEAR(Gbk[idx], fd2(Vof, idx, nao, 0, true, 1, false, h),
                  1e-5 * (std::abs(Gbk[idx]) + 1.0)) << "bk mu=" << mu;
      EXPECT_NEAR(Gbb[idx], fd2(Vof, idx, nao, 0, true, 1, true, h),
                  1e-5 * (std::abs(Gbb[idx]) + 1.0)) << "bb mu=" << mu;
    }
}

TEST(GeoDeriv, ThirdOrderRunsAndIsFiniteDiffConsistent) {
  // demonstrate arbitrary order: d^3/dA_x^2 dA_y  vs finite diff of the
  // order-1 (nabla_y) derivative in A_x twice
  auto bas = intti::make_basis(shells(A0, B0));
  const int nao = bas.nao;
  auto G3 = intti::overlap_geoderiv(bas, {2, 1, 0}, {0, 0, 0});
  const double h = 2e-3;
  auto dy = [&](const double A[3], const double B[3]) {
    return intti::overlap_geoderiv(intti::make_basis(shells(A, B)), {0, 1, 0}, {0, 0, 0});
  };
  for (int mu = 0; mu < 3; ++mu)
    for (int nu = 3; nu < nao; ++nu) {
      const int idx = mu * nao + nu;
      double Ap[3], Am[3], A2[3];
      for (int k = 0; k < 3; ++k) { Ap[k] = Am[k] = A2[k] = A0[k]; }
      Ap[0] += h; Am[0] -= h;
      // d^2/dA_x^2 of (nabla_y S): but geoderiv applies nabla (=-d/dA), so
      // d^2/dA_x^2 [nabla_y] = nabla_x^2 nabla_y = geoderiv(2,1,0); FD in A_x
      // of nabla_y uses nabla_x = -d/dA_x, so second FD gives (+)nabla_x^2.
      const double fd = (dy(Ap, B0)[idx] - 2 * dy(A2, B0)[idx] + dy(Am, B0)[idx]) / (h * h);
      EXPECT_NEAR(G3[idx], fd, 1e-3 * (std::abs(fd) + 1.0)) << "mu=" << mu << " nu=" << nu;
    }
}

} // namespace
