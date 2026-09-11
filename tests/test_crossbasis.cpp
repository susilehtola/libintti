// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/crossbasis.hpp"
#include "intti/fock.hpp"
#include "intti/tgrid.hpp"

// Cross-basis Coulomb/exchange. The decisive oracle is that with the SAME basis
// on both sides they must reproduce the ordinary coulomb_build/exchange_build
// exactly -- an independent check against the established real device path.

namespace {

using Shell = intti::PrimitiveShell<double>;

intti::ShellBasis<double> basisA() {
  return intti::make_basis<double>(
      {{1.2, {0.0, 0.0, 0.0}, 0}, {0.7, {0.4, -0.2, 0.5}, 1}});
}

// a genuinely different basis (different exponents, centres and l)
intti::ShellBasis<double> basisB() {
  return intti::make_basis<double>(
      {{0.9, {0.3, 0.1, -0.4}, 0}, {1.6, {-0.5, 0.6, 0.2}, 0}, {0.5, {0.1, 0.0, 0.9}, 1}});
}

std::vector<double> sym_density(int n, double seed) {
  std::vector<double> D(static_cast<std::size_t>(n) * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) D[i * n + j] = 0.15 + seed * 0.03 * (i + 1) * (j + 1);
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) D[j * n + i] = D[i * n + j];
  return D;
}

TEST(CrossBasis, CoulombSameBasisMatchesCoulombBuild) {
  auto bas = basisA();
  const int n = bas.nao;
  auto grid = intti::make_tgrid(intti::coulomb());
  auto D = sym_density(n, 1.0);
  auto V = intti::coulomb_cross(bas, bas, D.data(), grid);
  std::vector<double> J(static_cast<std::size_t>(n) * n);
  intti::coulomb_build(bas, D.data(), grid, J.data());
  double mx = 0, err = 0;
  for (std::size_t i = 0; i < J.size(); ++i) {
    mx = std::max(mx, std::abs(J[i]));
    err = std::max(err, std::abs(V[i] - J[i]));
  }
  ASSERT_GT(mx, 1e-6);
  EXPECT_LT(err, 1e-12 * mx) << "cross-basis J with one basis != coulomb_build";
}

TEST(CrossBasis, ExchangeSameBasisMatchesExchangeBuild) {
  auto bas = basisA();
  const int n = bas.nao;
  auto grid = intti::make_tgrid(intti::coulomb());
  auto D = sym_density(n, 1.0);
  auto Kx = intti::exchange_cross(bas, bas, D.data(), grid);
  std::vector<double> K(static_cast<std::size_t>(n) * n);
  intti::exchange_build(bas, D.data(), grid, K.data(), 0.0);
  double mx = 0, err = 0;
  for (std::size_t i = 0; i < K.size(); ++i) {
    mx = std::max(mx, std::abs(K[i]));
    err = std::max(err, std::abs(Kx[i] - K[i]));
  }
  ASSERT_GT(mx, 1e-6);
  EXPECT_LT(err, 1e-12 * mx) << "cross-basis K with one basis != exchange_build";
}

// With two genuinely different bases the results must be nontrivial, the right
// shape, and linear in the density (the property every Fock-like build has).
TEST(CrossBasis, TwoBasesShapeAndLinearity) {
  auto A = basisA(), B = basisB();
  const int nA = A.nao, nB = B.nao;
  ASSERT_NE(nA, nB);
  auto grid = intti::make_tgrid(intti::coulomb());
  auto D1 = sym_density(nB, 1.0), D2 = sym_density(nB, 2.0);
  std::vector<double> Dsum(D1.size());
  for (std::size_t i = 0; i < D1.size(); ++i) Dsum[i] = D1[i] + 2.5 * D2[i];

  auto V1 = intti::coulomb_cross(A, B, D1.data(), grid);
  auto V2 = intti::coulomb_cross(A, B, D2.data(), grid);
  auto Vs = intti::coulomb_cross(A, B, Dsum.data(), grid);
  auto K1 = intti::exchange_cross(A, B, D1.data(), grid);
  auto K2 = intti::exchange_cross(A, B, D2.data(), grid);
  auto Ks = intti::exchange_cross(A, B, Dsum.data(), grid);

  ASSERT_EQ(V1.size(), static_cast<std::size_t>(nA) * nA);
  ASSERT_EQ(K1.size(), static_cast<std::size_t>(nA) * nA);
  double mx = 0, lin = 0;
  for (std::size_t i = 0; i < V1.size(); ++i) {
    mx = std::max(mx, std::abs(V1[i]));
    lin = std::max(lin, std::abs(Vs[i] - (V1[i] + 2.5 * V2[i])));
    lin = std::max(lin, std::abs(Ks[i] - (K1[i] + 2.5 * K2[i])));
  }
  EXPECT_GT(mx, 1e-6) << "cross-basis V must be nontrivial";
  EXPECT_LT(lin, 1e-12 * (mx + 1)) << "cross-basis builds must be linear in D";
}

// The cross-basis Coulomb is symmetric in its output indices (the operator is
// multiplicative and D is symmetric), while K need not be.
TEST(CrossBasis, CoulombOutputSymmetric) {
  auto A = basisA(), B = basisB();
  const int nA = A.nao;
  auto grid = intti::make_tgrid(intti::coulomb());
  auto D = sym_density(B.nao, 1.0);
  auto V = intti::coulomb_cross(A, B, D.data(), grid);
  double mx = 0, asym = 0;
  for (int i = 0; i < nA; ++i)
    for (int j = 0; j < nA; ++j) {
      mx = std::max(mx, std::abs(V[i * nA + j]));
      asym = std::max(asym, std::abs(V[i * nA + j] - V[j * nA + i]));
    }
  ASSERT_GT(mx, 1e-6);
  EXPECT_LT(asym, 1e-12 * mx) << "cross-basis V must be symmetric";
}

} // namespace

// Cross-basis J and K over a chain, with the t-resolved node truncation on.
// Their digest weights each quartet by a density element, so the per-quartet
// budget is the tolerance divided by max|D|; the error must stay inside the
// tolerance the caller asked for, not inside that intermediate.
TEST(CrossBasis, ScreenedMatchesExact) {
  auto grid = intti::make_tgrid(intti::coulomb());
  std::vector<intti::PrimitiveShell<double>> ash, bsh;
  for (int i = 0; i < 5; ++i) {
    ash.push_back({1.0 + 0.3 * i, {4.0 * i, 0.0, 0.0}, i % 2});
    bsh.push_back({0.9 + 0.25 * i, {4.0 * i, 0.6, 0.0}, (i + 1) % 2});
  }
  auto A = intti::make_basis(ash), B = intti::make_basis(bsh);
  const int nB = B.nao;
  std::vector<double> D(static_cast<std::size_t>(nB) * nB);
  for (int i = 0; i < nB; ++i)
    for (int j = 0; j < nB; ++j) D[i * nB + j] = (i == j ? 1.4 : 0.2 / (1 + std::abs(i - j)));

  const auto J0 = intti::coulomb_cross(A, B, D.data(), grid);
  const auto K0 = intti::exchange_cross(A, B, D.data(), grid);
  for (double tau : {1e-12, 1e-10}) {
    const auto J = intti::coulomb_cross(A, B, D.data(), grid, tau);
    const auto K = intti::exchange_cross(A, B, D.data(), grid, tau);
    double dj = 0, dk = 0;
    for (std::size_t i = 0; i < J0.size(); ++i) dj = std::max(dj, std::abs(J[i] - J0[i]));
    for (std::size_t i = 0; i < K0.size(); ++i) dk = std::max(dk, std::abs(K[i] - K0[i]));
    EXPECT_LT(dj, tau) << "cross J screened beyond its budget at tau=" << tau;
    EXPECT_LT(dk, tau) << "cross K screened beyond its budget at tau=" << tau;
  }
}
