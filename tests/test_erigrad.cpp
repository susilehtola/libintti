// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <array>
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "intti/erigrad.hpp"
#include "intti/fock.hpp"

namespace {

using Shell = intti::PrimitiveShell<double>;

// three shells: s and p on centre 0, s on centre 1 (perturbable centres)
std::vector<Shell> make_shells(const double c0[3], const double c1[3]) {
  return {{1.1, {c0[0], c0[1], c0[2]}, 0},
          {0.7, {c0[0], c0[1], c0[2]}, 1},
          {0.9, {c1[0], c1[1], c1[2]}, 0}};
}

double E2(const intti::ShellBasis<double> &bas, const std::vector<double> &D,
          const intti::TGrid<double> &grid) {
  const std::size_t n2 = static_cast<std::size_t>(bas.nao) * bas.nao;
  std::vector<double> J(n2), K(n2);
  intti::coulomb_build(bas, D.data(), grid, J.data());
  intti::exchange_build(bas, D.data(), grid, K.data(), 0.0);
  double e = 0;
  for (std::size_t i = 0; i < n2; ++i) e += 0.5 * D[i] * J[i] - 0.25 * D[i] * K[i];
  return e;
}

TEST(EriGrad, TwoElectronForceVsFiniteDifference) {
  const double c0[3] = {0.0, 0.0, 0.0}, c1[3] = {0.2, -0.3, 1.3};
  auto bas = intti::make_basis(make_shells(c0, c1));
  const int nao = bas.nao;
  std::mt19937 rng(5);
  std::normal_distribution<double> nd;
  std::vector<double> D(static_cast<std::size_t>(nao) * nao, 0.0);
  for (int k = 0; k < 2; ++k) { // PSD density
    std::vector<double> v(nao);
    for (auto &x : v) x = nd(rng);
    for (int i = 0; i < nao; ++i)
      for (int j = 0; j < nao; ++j) D[i * nao + j] += v[i] * v[j];
  }
  auto grid = intti::make_tgrid(intti::coulomb());
  auto grad = intti::two_electron_gradient(bas, D.data(), grid);

  const double h = 1e-4;
  // shell 0,1 on centre 0; shell 2 on centre 1
  const int shell_center[3] = {0, 0, 1};
  double C[2][3] = {{c0[0], c0[1], c0[2]}, {c1[0], c1[1], c1[2]}};
  for (int s = 0; s < 3; ++s)
    for (int e = 0; e < 3; ++e) {
      const int ctr = shell_center[s];
      auto ev = [&](double sign) {
        double cc[2][3];
        for (int a = 0; a < 2; ++a)
          for (int d = 0; d < 3; ++d) cc[a][d] = C[a][d];
        // move ONLY shell s (not the whole centre): give it its own centre
        auto sh = make_shells(cc[0], cc[1]);
        sh[s].center[e] += sign * h;
        return E2(intti::make_basis(sh), D, grid);
      };
      const double fd = (ev(1) - ev(-1)) / (2 * h);
      EXPECT_NEAR(grad[s][e], fd, 1e-5 * (std::abs(fd) + 1.0))
          << "shell " << s << " dir " << e;
    }
}

// Screening at a tiny threshold must reproduce the exact (unscreened) gradient;
// exercises the 8-fold permutational orbit on an s/p/d basis.
TEST(EriGrad, ScreeningMatchesExact) {
  std::vector<Shell> shells = {
      {1.3, {0.0, 0.0, 0.0}, 0}, {0.8, {0.0, 0.0, 0.0}, 1}, {0.5, {0.0, 0.0, 0.0}, 2},
      {1.1, {1.6, -0.4, 0.2}, 0}, {0.6, {1.6, -0.4, 0.2}, 1}};
  auto bas = intti::make_basis(shells);
  const int nao = bas.nao;
  std::mt19937 rng(9);
  std::normal_distribution<double> nd;
  std::vector<double> D(static_cast<std::size_t>(nao) * nao, 0.0);
  for (int k = 0; k < 3; ++k) {
    std::vector<double> v(nao);
    for (auto &x : v) x = nd(rng);
    for (int i = 0; i < nao; ++i)
      for (int j = 0; j < nao; ++j) D[i * nao + j] += v[i] * v[j];
  }
  auto grid = intti::make_tgrid(intti::coulomb());
  auto g0 = intti::two_electron_gradient(bas, D.data(), grid);
  auto gs = intti::two_electron_gradient(bas, D.data(), grid, 1e-14);
  double worst = 0, scale = 0;
  for (std::size_t s = 0; s < g0.size(); ++s)
    for (int e = 0; e < 3; ++e) {
      worst = std::max(worst, std::abs(g0[s][e] - gs[s][e]));
      scale = std::max(scale, std::abs(g0[s][e]));
    }
  EXPECT_GT(scale, 1e-2);
  EXPECT_LT(worst, 1e-10 * (scale + 1)) << "tiny-tau screening must match exact";
}

} // namespace
