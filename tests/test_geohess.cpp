// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "intti/geohess.hpp"
#include "intti/nuclear.hpp"
#include "intti/oneel.hpp"

namespace {

using Shell = intti::PrimitiveShell<double>;

std::vector<Shell> base_shells() {
  return {{1.1, {0.0, 0.0, 0.0}, 0},
          {0.7, {0.0, 0.0, 0.0}, 1},
          {0.9, {0.2, -0.3, 1.3}, 0}};
}

std::vector<double> rand_sym(int nao, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> nd;
  std::vector<double> W(static_cast<std::size_t>(nao) * nao, 0.0);
  for (int i = 0; i < nao; ++i)
    for (int j = i; j < nao; ++j) {
      const double v = nd(rng);
      W[i * nao + j] = v;
      W[j * nao + i] = v;
    }
  return W;
}

// FD check: analytic Hessian of sum W_mn M_mn against a 4-point second
// difference using only the undifferentiated matrix builder `mat`.
void check_1e(const std::vector<double> &Han, const std::vector<Shell> &shells,
              const std::vector<double> &W,
              std::function<std::vector<double>(const std::vector<Shell> &)> mat) {
  const int ns = static_cast<int>(shells.size()), dim = 3 * ns;
  const int nao = intti::make_basis(shells).nao;
  auto E = [&](int p, int e, double sp, int q, int f, double sq) {
    auto sh = shells;
    sh[p].center[e] += sp;
    sh[q].center[f] += sq;
    auto M = mat(sh);
    double s = 0;
    for (std::size_t i = 0; i < W.size(); ++i) s += W[i] * M[i];
    return s;
  };
  const double h = 2e-3;
  double worst = 0, scale = 0;
  for (int p = 0; p < ns; ++p)
    for (int e = 0; e < 3; ++e)
      for (int q = 0; q < ns; ++q)
        for (int f = 0; f < 3; ++f) {
          const double fd = (E(p, e, h, q, f, h) - E(p, e, h, q, f, -h) -
                             E(p, e, -h, q, f, h) + E(p, e, -h, q, f, -h)) /
                            (4 * h * h);
          const double an = Han[(3 * p + e) * static_cast<std::size_t>(dim) + 3 * q + f];
          worst = std::max(worst, std::abs(fd - an));
          scale = std::max(scale, std::abs(an));
        }
  (void)nao;
  EXPECT_GT(scale, 1e-2) << "hessian must be nonzero";
  EXPECT_LT(worst, 2e-5 * (scale + 1)) << "analytic 1e Hessian != finite difference";
}

TEST(GeoHess, OverlapHessianVsFiniteDiff) {
  auto shells = base_shells();
  auto bas = intti::make_basis(shells);
  auto W = rand_sym(bas.nao, 7);
  auto H = intti::overlap_hessian(bas, W.data());
  check_1e(H, shells, W,
           [](const std::vector<Shell> &sh) { return intti::overlap_matrix(intti::make_basis(sh)); });
}

TEST(GeoHess, KineticHessianVsFiniteDiff) {
  auto shells = base_shells();
  auto bas = intti::make_basis(shells);
  auto D = rand_sym(bas.nao, 13);
  auto H = intti::kinetic_hessian(bas, D.data());
  check_1e(H, shells, D,
           [](const std::vector<Shell> &sh) { return intti::kinetic_matrix(intti::make_basis(sh)); });
}

TEST(GeoHess, OneElectronHessianSymmetric) {
  auto bas = intti::make_basis(base_shells());
  auto D = rand_sym(bas.nao, 21);
  const int dim = 3 * static_cast<int>(bas.shells.size());
  for (auto &H : {intti::overlap_hessian(bas, D.data()), intti::kinetic_hessian(bas, D.data())}) {
    double asym = 0, mx = 0;
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j) {
        asym = std::max(asym, std::abs(H[i * dim + j] - H[j * dim + i]));
        mx = std::max(mx, std::abs(H[i * dim + j]));
      }
    EXPECT_LT(asym, 1e-11 * (mx + 1));
  }
}

TEST(GeoHess, NuclearAttractionHessianVsFiniteDiff) {
  // three shells, each on its own centre carrying a point charge (an "atom");
  // moving an atom displaces its shell centre and its nucleus together.
  std::vector<Shell> shells = {{1.1, {0.0, 0.0, 0.0}, 0},
                               {0.7, {0.3, -0.2, 1.1}, 1},
                               {0.9, {-0.5, 0.4, 0.6}, 0}};
  const int ns = static_cast<int>(shells.size());
  auto bas0 = intti::make_basis(shells);
  const int nao = bas0.nao, dim = 3 * ns;
  const double Z[3] = {1.0, 6.0, 1.0};
  std::vector<int> charge_shell = {0, 1, 2};
  auto make_charges = [&](const std::vector<Shell> &sh) {
    std::vector<intti::PointCharge<double>> c(ns);
    for (int i = 0; i < ns; ++i)
      c[i] = {-Z[i], {sh[i].center[0], sh[i].center[1], sh[i].center[2]}};
    return c;
  };
  auto grid = intti::make_tgrid(intti::coulomb());
  auto W = rand_sym(nao, 31);
  auto H = intti::nuclear_attraction_hessian(bas0, make_charges(shells), grid, charge_shell,
                                             W.data());
  auto Ene = [&](const std::vector<Shell> &sh) {
    auto b = intti::make_basis(sh);
    auto V = intti::nuclear_matrix(b, make_charges(sh), grid);
    double e = 0;
    for (std::size_t i = 0; i < W.size(); ++i) e += W[i] * V[i];
    return e;
  };
  auto E = [&](int p, int e, double sp, int q, int f, double sq) {
    auto sh = shells;
    sh[p].center[e] += sp; // moves shell centre and (via make_charges) the nucleus
    sh[q].center[f] += sq;
    return Ene(sh);
  };
  const double h = 2e-3;
  double worst = 0, scale = 0;
  for (int p = 0; p < ns; ++p)
    for (int e = 0; e < 3; ++e)
      for (int q = 0; q < ns; ++q)
        for (int f = 0; f < 3; ++f) {
          const double fd = (E(p, e, h, q, f, h) - E(p, e, h, q, f, -h) -
                             E(p, e, -h, q, f, h) + E(p, e, -h, q, f, -h)) /
                            (4 * h * h);
          const double an = H[(3 * p + e) * static_cast<std::size_t>(dim) + 3 * q + f];
          worst = std::max(worst, std::abs(fd - an));
          scale = std::max(scale, std::abs(an));
        }
  EXPECT_GT(scale, 1e-2) << "hessian must be nonzero";
  EXPECT_LT(worst, 5e-5 * (scale + 1)) << "analytic V Hessian != finite difference";
}

TEST(GeoHess, NuclearRepulsionHessianVsFiniteDiff) {
  std::vector<intti::PointCharge<double>> chg{
      {1.0, {0.0, 0.0, 0.0}}, {8.0, {0.0, 0.0, 1.8}}, {1.0, {1.4, 0.0, 2.2}}};
  const int N = static_cast<int>(chg.size()), dim = 3 * N;
  auto H = intti::nuclear_repulsion_hessian(chg);
  auto Enn = [&](std::vector<intti::PointCharge<double>> c) {
    double e = 0;
    for (int I = 0; I < N; ++I)
      for (int J = I + 1; J < N; ++J) {
        double r2 = 0;
        for (int d = 0; d < 3; ++d) r2 += (c[I].R[d] - c[J].R[d]) * (c[I].R[d] - c[J].R[d]);
        e += c[I].weight * c[J].weight / std::sqrt(r2);
      }
    return e;
  };
  const double h = 1e-3;
  double worst = 0, scale = 0;
  for (int p = 0; p < N; ++p)
    for (int e = 0; e < 3; ++e)
      for (int q = 0; q < N; ++q)
        for (int f = 0; f < 3; ++f) {
          auto shift = [&](double sp, double sq) {
            auto c = chg;
            c[p].R[e] += sp;
            c[q].R[f] += sq;
            return Enn(c);
          };
          const double fd =
              (shift(h, h) - shift(h, -h) - shift(-h, h) + shift(-h, -h)) / (4 * h * h);
          const double an = H[(3 * p + e) * static_cast<std::size_t>(dim) + 3 * q + f];
          worst = std::max(worst, std::abs(fd - an));
          scale = std::max(scale, std::abs(an));
        }
  EXPECT_LT(worst, 1e-5 * (scale + 1)) << "analytic E_nn Hessian != finite difference";
}

} // namespace
