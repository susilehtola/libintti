// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "intti/jbuild.hpp"
#include "intti/quartet.hpp"

namespace {

template <class Real> struct System {
  std::vector<intti::ShellPair<Real>> pairs;
  std::vector<int> off; ///< product offsets per pair
  int nprod{0};
};

template <class Real> System<Real> test_system() {
  using Shell = intti::PrimitiveShell<Real>;
  const std::vector<Shell> shells = {
      {Real(1.2), {Real(0), Real(0), Real(0)}, 0},
      {Real(0.3), {Real(0), Real(0), Real(0)}, 0},
      {Real(0.8), {Real(0), Real(0), Real(0)}, 1},
      {Real(1.5), {Real(0), Real(0), Real(1.4)}, 0},
      {Real(0.5), {Real(0), Real(0), Real(1.4)}, 1},
      {Real(0.9), {Real(0), Real(0), Real(1.4)}, 2},
  };
  System<Real> s;
  for (std::size_t i = 0; i < shells.size(); ++i)
    for (std::size_t j = i; j < shells.size(); ++j)
      s.pairs.push_back(intti::make_pair(shells[i], shells[j]));
  s.off.resize(s.pairs.size() + 1, 0);
  for (std::size_t p = 0; p < s.pairs.size(); ++p)
    s.off[p + 1] = s.off[p] + intti::ncart(s.pairs[p].la) * intti::ncart(s.pairs[p].lb);
  s.nprod = s.off.back();
  return s;
}

// dense reference: J_p = sum_q D_q (p|q) via explicit quartets on the SAME grid
template <class Real>
std::vector<Real> dense_j(const System<Real> &s, const std::vector<Real> &D,
                          const intti::TGrid<Real> &grid) {
  std::vector<Real> J(s.nprod, Real(0));
  for (std::size_t p = 0; p < s.pairs.size(); ++p)
    for (std::size_t q = 0; q < s.pairs.size(); ++q) {
      const int np = s.off[p + 1] - s.off[p];
      const int nq = s.off[q + 1] - s.off[q];
      std::vector<Real> block(np * nq);
      intti::eri_quartet(s.pairs[p], s.pairs[q], grid, block.data());
      for (int i = 0; i < np; ++i)
        for (int j = 0; j < nq; ++j)
          J[s.off[p] + i] += D[s.off[q] + j] * block[i * nq + j];
    }
  return J;
}

std::vector<double> random_density(int nprod) {
  std::mt19937 rng(17);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> D(nprod);
  for (auto &d : D)
    d = u(rng);
  return D;
}

TEST(JBuild, MatchesDenseContractionMobius) {
  auto s = test_system<double>();
  auto tab = intti::make_pair_table(s.pairs);
  auto D = random_density(s.nprod);
  auto grid = intti::make_tgrid(intti::coulomb());
  std::vector<double> J(s.nprod);
  intti::coulomb_build(tab, D.data(), grid, J.data());
  auto Jref = dense_j(s, D, grid);
  double scale = 0.0;
  for (int i = 0; i < s.nprod; ++i)
    scale = std::max(scale, std::abs(Jref[i]));
  for (int i = 0; i < s.nprod; ++i)
    EXPECT_NEAR(J[i], Jref[i], 1e-12 * scale) << "product " << i;
}

TEST(JBuild, MatchesDenseContractionLinLogWithTail) {
  auto s = test_system<double>();
  auto tab = intti::make_pair_table(s.pairs);
  auto D = random_density(s.nprod);
  intti::TGridSpec<double> spec;
  spec.mapping = intti::TMapping::LinLog;
  auto grid = intti::make_tgrid(intti::coulomb(), spec);
  ASSERT_GT(grid.tail_coeff, 0.0);
  std::vector<double> J(s.nprod);
  intti::coulomb_build(tab, D.data(), grid, J.data());
  auto Jref = dense_j(s, D, grid);
  double scale = 0.0;
  for (int i = 0; i < s.nprod; ++i)
    scale = std::max(scale, std::abs(Jref[i]));
  for (int i = 0; i < s.nprod; ++i)
    EXPECT_NEAR(J[i], Jref[i], 1e-12 * scale) << "product " << i;
}

TEST(JBuild, LongDouble) {
  auto s = test_system<long double>();
  auto tab = intti::make_pair_table(s.pairs);
  std::vector<long double> D(s.nprod);
  auto Dd = random_density(s.nprod);
  for (int i = 0; i < s.nprod; ++i)
    D[i] = Dd[i];
  auto grid = intti::make_tgrid(intti::coulomb<long double>());
  std::vector<long double> J(s.nprod);
  intti::coulomb_build(tab, D.data(), grid, J.data());
  auto Jref = dense_j(s, D, grid);
  long double scale = 0;
  for (int i = 0; i < s.nprod; ++i)
    scale = std::max(scale, std::abs(Jref[i]));
  for (int i = 0; i < s.nprod; ++i)
    EXPECT_LT(std::abs(J[i] - Jref[i]), 1e-15L * scale) << "product " << i;
}

} // namespace
