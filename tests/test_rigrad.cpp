// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "intti/rigrad.hpp"

namespace {

using Shell = intti::PrimitiveShell<double>;

std::vector<Shell> orb_shells() {
  return {{1.3, {0.0, 0.0, 0.0}, 0},
          {0.8, {0.0, 0.0, 0.0}, 1},
          {1.0, {0.1, -0.2, 1.2}, 0}};
}
// a well-conditioned auxiliary basis on the same two centres (no metric
// truncation, so the fit is M^{-1} and the energy is exactly reproduced)
std::vector<Shell> aux_shells() {
  return {{2.6, {0.0, 0.0, 0.0}, 0}, {0.9, {0.0, 0.0, 0.0}, 0},
          {1.4, {0.0, 0.0, 0.0}, 1}, {2.2, {0.1, -0.2, 1.2}, 0},
          {0.7, {0.1, -0.2, 1.2}, 0}, {1.1, {0.1, -0.2, 1.2}, 1}};
}

// The density is built as a sum of outer products, so it is ALREADY factorised:
// D = C C^T with C nao x 3. The orbital-driven RI-K builders take those factors
// directly, which is why they can be used here without any decomposition.
constexpr int kDensityRank = 3;

std::vector<double> density_factors(int nao) {
  std::mt19937 rng(19);
  std::normal_distribution<double> nd;
  std::vector<double> C(static_cast<std::size_t>(nao) * kDensityRank, 0.0);
  for (int k = 0; k < kDensityRank; ++k)
    for (int i = 0; i < nao; ++i) C[i * kDensityRank + k] = nd(rng);
  return C;
}

std::vector<double> density(int nao) {
  const auto C = density_factors(nao);
  std::vector<double> D(static_cast<std::size_t>(nao) * nao, 0.0);
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j) {
      double acc = 0;
      for (int k = 0; k < kDensityRank; ++k)
        acc += C[i * kDensityRank + k] * C[j * kDensityRank + k];
      D[i * nao + j] = acc;
    }
  return D;
}

double ri_j_energy(const std::vector<Shell> &os, const std::vector<Shell> &as,
                   const std::vector<double> &D, const intti::TGrid<double> &grid) {
  auto orb = intti::make_basis(os);
  auto aux = intti::make_basis(as);
  auto fit = intti::ri_fit(orb, aux, grid, 1e-12);
  std::vector<double> J(static_cast<std::size_t>(orb.nao) * orb.nao, 0.0);
  intti::ri_jk(fit, D.data(), J.data(), static_cast<double *>(nullptr));
  double e = 0;
  for (std::size_t i = 0; i < J.size(); ++i) e += 0.5 * D[i] * J[i];
  return e;
}

double ri_k_energy(const std::vector<Shell> &os, const std::vector<Shell> &as,
                   const std::vector<double> &D, const intti::TGrid<double> &grid) {
  auto orb = intti::make_basis(os);
  auto aux = intti::make_basis(as);
  auto fit = intti::ri_fit(orb, aux, grid, 1e-12);
  std::vector<double> K(static_cast<std::size_t>(orb.nao) * orb.nao, 0.0);
  intti::ri_jk(fit, D.data(), static_cast<double *>(nullptr), K.data());
  double e = 0;
  for (std::size_t i = 0; i < K.size(); ++i) e += -0.25 * D[i] * K[i];
  return e;
}

TEST(RIGrad, ExchangeGradientVsFiniteDifference) {
  auto os = orb_shells();
  auto as = aux_shells();
  auto orb = intti::make_basis(os);
  auto D = density(orb.nao);
  auto grid = intti::make_tgrid(intti::coulomb());
  const auto C = density_factors(orb.nao);
  auto g = intti::ri_k_gradient_occ(orb, intti::make_basis(as), C.data(), C.data(),
                                    kDensityRank, grid, 1e-12);
  const double h = 1e-4;
  double worst = 0, scale = 0;
  auto fd = [&](std::vector<Shell> &shells, int s, int e) {
    const double c0 = shells[s].center[e];
    shells[s].center[e] = c0 + h;
    const double ep = ri_k_energy(os, as, D, grid);
    shells[s].center[e] = c0 - h;
    const double em = ri_k_energy(os, as, D, grid);
    shells[s].center[e] = c0;
    return (ep - em) / (2 * h);
  };
  for (int s = 0; s < static_cast<int>(os.size()); ++s)
    for (int e = 0; e < 3; ++e) {
      worst = std::max(worst, std::abs(fd(os, s, e) - g.forb[s][e]));
      scale = std::max(scale, std::abs(g.forb[s][e]));
    }
  for (int s = 0; s < static_cast<int>(as.size()); ++s)
    for (int e = 0; e < 3; ++e) {
      worst = std::max(worst, std::abs(fd(as, s, e) - g.faux[s][e]));
      scale = std::max(scale, std::abs(g.faux[s][e]));
    }
  EXPECT_GT(scale, 1e-2) << "gradient must be nonzero";
  EXPECT_LT(worst, 1e-6 * (scale + 1)) << "analytic RI-K gradient != finite difference";
}

TEST(RIGrad, CoulombGradientVsFiniteDifference) {
  auto os = orb_shells();
  auto as = aux_shells();
  auto orb = intti::make_basis(os);
  auto D = density(orb.nao);
  auto grid = intti::make_tgrid(intti::coulomb());
  auto g = intti::ri_j_gradient(orb, intti::make_basis(as), D.data(), grid, 1e-12);

  const double h = 1e-4;
  double worst = 0, scale = 0;
  auto fd = [&](std::vector<Shell> &shells, int s, int e) {
    const double c0 = shells[s].center[e];
    shells[s].center[e] = c0 + h;
    const double ep = ri_j_energy(os, as, D, grid);
    shells[s].center[e] = c0 - h;
    const double em = ri_j_energy(os, as, D, grid);
    shells[s].center[e] = c0;
    return (ep - em) / (2 * h);
  };
  for (int s = 0; s < static_cast<int>(os.size()); ++s)
    for (int e = 0; e < 3; ++e) {
      const double num = fd(os, s, e);
      worst = std::max(worst, std::abs(num - g.forb[s][e]));
      scale = std::max(scale, std::abs(g.forb[s][e]));
    }
  for (int s = 0; s < static_cast<int>(as.size()); ++s)
    for (int e = 0; e < 3; ++e) {
      const double num = fd(as, s, e);
      worst = std::max(worst, std::abs(num - g.faux[s][e]));
      scale = std::max(scale, std::abs(g.faux[s][e]));
    }
  EXPECT_GT(scale, 1e-2) << "gradient must be nonzero";
  EXPECT_LT(worst, 1e-6 * (scale + 1)) << "analytic RI-J gradient != finite difference";
}

// combined per-shell RI-J gradient vector (orbital shells then auxiliary)
std::vector<double> ri_j_grad_vec(const std::vector<Shell> &os, const std::vector<Shell> &as,
                                  const std::vector<double> &D,
                                  const intti::TGrid<double> &grid) {
  auto orb = intti::make_basis(os);
  auto g = intti::ri_j_gradient(orb, intti::make_basis(as), D.data(), grid, 1e-12);
  std::vector<double> v;
  for (auto &f : g.forb)
    for (int e = 0; e < 3; ++e) v.push_back(f[e]);
  for (auto &f : g.faux)
    for (int e = 0; e < 3; ++e) v.push_back(f[e]);
  return v;
}

TEST(RIGrad, CoulombHessianVsGradientFiniteDifference) {
  auto os = orb_shells();
  auto as = aux_shells();
  auto orb = intti::make_basis(os);
  auto D = density(orb.nao);
  auto grid = intti::make_tgrid(intti::coulomb());
  auto H = intti::ri_j_hessian(orb, intti::make_basis(as), D.data(), grid, 1e-12);
  const int nso = static_cast<int>(os.size()), nsa = static_cast<int>(as.size());
  const int ncen = nso + nsa, dim = 3 * ncen;
  const double h = 1e-4;
  double worst = 0, scale = 0;
  for (int y = 0; y < ncen; ++y)
    for (int f = 0; f < 3; ++f) {
      auto disp = [&](double s) {
        auto o2 = os, a2 = as;
        (y < nso ? o2[y] : a2[y - nso]).center[f] += s;
        return ri_j_grad_vec(o2, a2, D, grid);
      };
      auto gp = disp(h), gm = disp(-h);
      for (int x = 0; x < ncen; ++x)
        for (int e = 0; e < 3; ++e) {
          const double fd = (gp[3 * x + e] - gm[3 * x + e]) / (2 * h);
          const double an = H[(3 * x + e) * static_cast<std::size_t>(dim) + 3 * y + f];
          worst = std::max(worst, std::abs(fd - an));
          scale = std::max(scale, std::abs(an));
        }
    }
  EXPECT_GT(scale, 1e-2) << "hessian must be nonzero";
  EXPECT_LT(worst, 5e-5 * (scale + 1)) << "analytic RI-J Hessian != FD of gradient";
}

TEST(RIGrad, CoulombHessianSymmetric) {
  auto os = orb_shells();
  auto as = aux_shells();
  auto orb = intti::make_basis(os);
  auto D = density(orb.nao);
  auto grid = intti::make_tgrid(intti::coulomb());
  auto H = intti::ri_j_hessian(orb, intti::make_basis(as), D.data(), grid, 1e-12);
  const int dim = 3 * (static_cast<int>(os.size()) + static_cast<int>(as.size()));
  double asym = 0, mx = 0;
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j) {
      asym = std::max(asym, std::abs(H[i * dim + j] - H[j * dim + i]));
      mx = std::max(mx, std::abs(H[i * dim + j]));
    }
  EXPECT_LT(asym, 1e-10 * (mx + 1));
}

std::vector<double> ri_k_grad_vec(const std::vector<Shell> &os, const std::vector<Shell> &as,
                                  const std::vector<double> &D,
                                  const intti::TGrid<double> &grid) {
  auto orb = intti::make_basis(os);
  const auto C = density_factors(orb.nao);
  auto g = intti::ri_k_gradient_occ(orb, intti::make_basis(as), C.data(), C.data(),
                                    kDensityRank, grid, 1e-12);
  std::vector<double> v;
  for (auto &f : g.forb)
    for (int e = 0; e < 3; ++e) v.push_back(f[e]);
  for (auto &f : g.faux)
    for (int e = 0; e < 3; ++e) v.push_back(f[e]);
  return v;
}

TEST(RIGrad, ExchangeHessianVsGradientFiniteDifference) {
  auto os = orb_shells();
  auto as = aux_shells();
  auto orb = intti::make_basis(os);
  auto D = density(orb.nao);
  auto grid = intti::make_tgrid(intti::coulomb());
  const auto Cf = density_factors(orb.nao);
  auto H = intti::ri_k_hessian_occ(orb, intti::make_basis(as), Cf.data(), Cf.data(),
                                   kDensityRank, grid, 1e-12);
  const int nso = static_cast<int>(os.size()), nsa = static_cast<int>(as.size());
  const int ncen = nso + nsa, dim = 3 * ncen;
  const double h = 1e-4;
  double worst = 0, scale = 0, asym = 0;
  for (int y = 0; y < ncen; ++y)
    for (int f = 0; f < 3; ++f) {
      auto disp = [&](double s) {
        auto o2 = os, a2 = as;
        (y < nso ? o2[y] : a2[y - nso]).center[f] += s;
        return ri_k_grad_vec(o2, a2, D, grid);
      };
      auto gp = disp(h), gm = disp(-h);
      for (int x = 0; x < ncen; ++x)
        for (int e = 0; e < 3; ++e) {
          const double fd = (gp[3 * x + e] - gm[3 * x + e]) / (2 * h);
          const double an = H[(3 * x + e) * static_cast<std::size_t>(dim) + 3 * y + f];
          worst = std::max(worst, std::abs(fd - an));
          scale = std::max(scale, std::abs(an));
        }
    }
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j)
      asym = std::max(asym, std::abs(H[i * dim + j] - H[j * dim + i]));
  EXPECT_GT(scale, 1e-2) << "hessian must be nonzero";
  EXPECT_LT(asym, 1e-10 * (scale + 1)) << "Hessian must be symmetric";
  EXPECT_LT(worst, 5e-5 * (scale + 1)) << "analytic RI-K Hessian != FD of gradient";
}

// translational invariance: summing the force over every shell centre (orbital
// and auxiliary) must vanish -- rigid translation leaves E_J unchanged.
TEST(RIGrad, TranslationalInvariance) {
  auto os = orb_shells();
  auto as = aux_shells();
  auto orb = intti::make_basis(os);
  auto D = density(orb.nao);
  auto grid = intti::make_tgrid(intti::coulomb());
  auto g = intti::ri_j_gradient(orb, intti::make_basis(as), D.data(), grid, 1e-12);
  double sum[3] = {0, 0, 0}, mx = 0;
  for (auto &f : g.forb)
    for (int e = 0; e < 3; ++e) {
      sum[e] += f[e];
      mx = std::max(mx, std::abs(f[e]));
    }
  for (auto &f : g.faux)
    for (int e = 0; e < 3; ++e) {
      sum[e] += f[e];
      mx = std::max(mx, std::abs(f[e]));
    }
  for (int e = 0; e < 3; ++e) EXPECT_LT(std::abs(sum[e]), 1e-9 * (mx + 1));
}

} // namespace
