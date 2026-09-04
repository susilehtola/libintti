// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "intti/erigrad.hpp"
#include "intti/erihess.hpp"
#include "intti/fock.hpp"

namespace {

using Shell = intti::PrimitiveShell<double>;

// four primitive shells on three distinct centres (per-shell perturbable)
std::vector<Shell> base_shells() {
  return {{1.1, {0.0, 0.0, 0.0}, 0},
          {0.7, {0.0, 0.0, 0.0}, 1},
          {0.9, {0.2, -0.3, 1.3}, 0},
          {0.6, {-0.4, 0.5, 0.2}, 1}};
}

TEST(EriHess, TwoElectronHessianVsGradientFiniteDifference) {
  auto shells = base_shells();
  const int ns = static_cast<int>(shells.size());
  auto bas = intti::make_basis(shells);
  const int nao = bas.nao;
  std::mt19937 rng(11);
  std::normal_distribution<double> nd;
  std::vector<double> D(static_cast<std::size_t>(nao) * nao, 0.0);
  for (int k = 0; k < 2; ++k) { // symmetric PSD density
    std::vector<double> v(nao);
    for (auto &x : v) x = nd(rng);
    for (int i = 0; i < nao; ++i)
      for (int j = 0; j < nao; ++j) D[i * nao + j] += v[i] * v[j];
  }
  auto grid = intti::make_tgrid(intti::coulomb());
  auto Hana = intti::two_electron_hessian(bas, D.data(), grid);
  const int dim = 3 * ns;

  // finite difference of the (validated) 2e gradient w.r.t. each shell centre
  auto grad_at = [&](int qshell, int f, double delta) {
    auto sh = shells;
    sh[qshell].center[f] += delta;
    auto b = intti::make_basis(sh);
    return intti::two_electron_gradient(b, D.data(), grid);
  };
  const double h = 2e-4;
  double worst = 0, scale = 0;
  for (int q = 0; q < ns; ++q)
    for (int f = 0; f < 3; ++f) {
      auto gp = grad_at(q, f, h), gm = grad_at(q, f, -h);
      for (int p = 0; p < ns; ++p)
        for (int e = 0; e < 3; ++e) {
          const double fd = (gp[p][e] - gm[p][e]) / (2 * h);
          const double an = Hana[(3 * p + e) * static_cast<std::size_t>(dim) + 3 * q + f];
          worst = std::max(worst, std::abs(fd - an));
          scale = std::max(scale, std::abs(an));
        }
    }
  EXPECT_GT(scale, 1e-2) << "hessian must be nonzero";
  EXPECT_LT(worst, 5e-6 * (scale + 1)) << "analytic Hessian != FD of gradient";
}

TEST(EriHess, Symmetric) {
  auto bas = intti::make_basis(base_shells());
  const int nao = bas.nao, ns = static_cast<int>(bas.shells.size()), dim = 3 * ns;
  std::vector<double> D(static_cast<std::size_t>(nao) * nao, 0.0);
  for (int i = 0; i < nao; ++i) D[i * nao + i] = 1.0;
  auto grid = intti::make_tgrid(intti::coulomb());
  auto H = intti::two_electron_hessian(bas, D.data(), grid);
  double asym = 0, mx = 0;
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j) {
      asym = std::max(asym, std::abs(H[i * dim + j] - H[j * dim + i]));
      mx = std::max(mx, std::abs(H[i * dim + j]));
    }
  EXPECT_LT(asym, 1e-12 * (mx + 1)) << "Hessian must be symmetric";
}

// translational invariance: summing the Hessian over all shell centres (for a
// fixed differentiated centre) must vanish -- rigid translation is a null mode.
TEST(EriHess, TranslationalInvariance) {
  auto bas = intti::make_basis(base_shells());
  const int nao = bas.nao, ns = static_cast<int>(bas.shells.size()), dim = 3 * ns;
  std::mt19937 rng(3);
  std::normal_distribution<double> nd;
  std::vector<double> D(static_cast<std::size_t>(nao) * nao, 0.0);
  for (int k = 0; k < 2; ++k) {
    std::vector<double> v(nao);
    for (auto &x : v) x = nd(rng);
    for (int i = 0; i < nao; ++i)
      for (int j = 0; j < nao; ++j) D[i * nao + j] += v[i] * v[j];
  }
  auto grid = intti::make_tgrid(intti::coulomb());
  auto H = intti::two_electron_hessian(bas, D.data(), grid);
  double worst = 0, mx = 0;
  for (int q = 0; q < ns; ++q)
    for (int e = 0; e < 3; ++e)
      for (int f = 0; f < 3; ++f) {
        double s = 0;
        for (int p = 0; p < ns; ++p)
          s += H[(3 * p + e) * static_cast<std::size_t>(dim) + 3 * q + f];
        worst = std::max(worst, std::abs(s));
      }
  for (double x : H) mx = std::max(mx, std::abs(x));
  EXPECT_LT(worst, 1e-9 * (mx + 1)) << "sum over differentiated centre must vanish";
}

// Tiny-tau screening must reproduce the exact Hessian; exercises the 8-fold
// permutational orbit including a d shell.
TEST(EriHess, ScreeningMatchesExact) {
  std::vector<Shell> shells = {{1.3, {0.0, 0.0, 0.0}, 0},
                               {0.8, {0.0, 0.0, 0.0}, 1},
                               {0.5, {0.0, 0.0, 0.0}, 2},
                               {1.1, {1.5, -0.4, 0.2}, 1}};
  auto bas = intti::make_basis(shells);
  const int nao = bas.nao, ns = static_cast<int>(bas.shells.size()), dim = 3 * ns;
  std::mt19937 rng(21);
  std::normal_distribution<double> nd;
  std::vector<double> D(static_cast<std::size_t>(nao) * nao, 0.0);
  for (int k = 0; k < 3; ++k) {
    std::vector<double> v(nao);
    for (auto &x : v) x = nd(rng);
    for (int i = 0; i < nao; ++i)
      for (int j = 0; j < nao; ++j) D[i * nao + j] += v[i] * v[j];
  }
  auto grid = intti::make_tgrid(intti::coulomb());
  auto H0 = intti::two_electron_hessian(bas, D.data(), grid);
  auto Hs = intti::two_electron_hessian(bas, D.data(), grid, 1e-14);
  double worst = 0, scale = 0;
  for (std::size_t i = 0; i < H0.size(); ++i) {
    worst = std::max(worst, std::abs(H0[i] - Hs[i]));
    scale = std::max(scale, std::abs(H0[i]));
  }
  (void)dim;
  EXPECT_GT(scale, 1e-2);
  EXPECT_LT(worst, 1e-10 * (scale + 1)) << "tiny-tau screening must match exact";
}

} // namespace
