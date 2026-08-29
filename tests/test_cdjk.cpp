// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "intti/cdjk.hpp"

namespace {

using Shell = intti::PrimitiveShell<double>;

intti::ShellBasis<double> test_basis() {
  return intti::make_basis<double>({
      {1.2, {0.0, 0.0, 0.0}, 0},
      {0.3, {0.0, 0.0, 0.0}, 0},
      {0.8, {0.0, 0.0, 0.0}, 1},
      {1.5, {0.0, 0.0, 1.4}, 0},
      {0.5, {0.0, 0.0, 1.4}, 1},
      {0.9, {0.0, 0.0, 1.4}, 2},
  });
}

std::vector<double> random_symmetric(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> D(static_cast<std::size_t>(n) * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j <= i; ++j)
      D[i * n + j] = D[j * n + i] = u(rng);
  return D;
}

double max_abs_diff(const std::vector<double> &a, const std::vector<double> &b) {
  double m = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

double max_abs(const std::vector<double> &a) {
  double m = 0;
  for (double v : a)
    m = std::max(m, std::abs(v));
  return m;
}

TEST(CDJK, ThresholdControlledAgainstExact) {
  auto b = test_basis();
  auto D = random_symmetric(b.nao, 23);
  auto grid = intti::make_tgrid(intti::coulomb());
  const std::size_t n2 = static_cast<std::size_t>(b.nao) * b.nao;
  std::vector<double> Jx(n2), Kx(n2);
  intti::coulomb_build(b, D.data(), grid, Jx.data());
  intti::exchange_build(b, D.data(), grid, Kx.data(), 0.0);

  double prevJ = 1e100, prevK = 1e100;
  for (double tau : {1e-4, 1e-6, 1e-8}) {
    intti::CholeskyOptions<double> opt;
    opt.tau = tau;
    auto cb = intti::two_step_cholesky(b, grid, opt);
    std::vector<double> J(n2), K(n2);
    intti::cholesky_jk(b, cb, D.data(), J.data(), K.data());
    const double eJ = max_abs_diff(J, Jx), eK = max_abs_diff(K, Kx);
    // the CD residual bound |R_pq| <= tau propagates linearly into J/K
    EXPECT_LT(eJ, 50 * tau * max_abs(D) * b.nao) << "tau=" << tau;
    EXPECT_LT(eK, 50 * tau * max_abs(D) * b.nao) << "tau=" << tau;
    EXPECT_LE(eJ, prevJ * 1.5) << "tau=" << tau; // tightening tau must not hurt
    EXPECT_LE(eK, prevK * 1.5) << "tau=" << tau;
    prevJ = eJ;
    prevK = eK;
  }
}

TEST(CDJK, NullOutputsAllowed) {
  auto b = test_basis();
  auto D = random_symmetric(b.nao, 29);
  auto grid = intti::make_tgrid(intti::coulomb());
  intti::CholeskyOptions<double> opt;
  opt.tau = 1e-6;
  auto cb = intti::two_step_cholesky(b, grid, opt);
  const std::size_t n2 = static_cast<std::size_t>(b.nao) * b.nao;
  std::vector<double> J(n2), K(n2), Jonly(n2), Konly(n2);
  intti::cholesky_jk(b, cb, D.data(), J.data(), K.data());
  intti::cholesky_jk(b, cb, D.data(), Jonly.data(), static_cast<double *>(nullptr));
  intti::cholesky_jk(b, cb, D.data(), static_cast<double *>(nullptr), Konly.data());
  EXPECT_LT(max_abs_diff(J, Jonly), 1e-15);
  EXPECT_LT(max_abs_diff(K, Konly), 1e-15);
}

} // namespace
