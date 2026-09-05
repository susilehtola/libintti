// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/contracted.hpp"
#include "intti/fock.hpp"
#include "intti/gridri.hpp"
#include "intti/gto.hpp"
#include "intti/kernel.hpp"
#include "intti/tgrid.hpp"

// Grid-RI Fock builders (roadmap M-FE grid-RI): J via density-on-grid + DAGE and
// K via co-densities, validated against the exact GTO coulomb_build /
// exchange_build. Coarse grid + t-grid keep CI fast; accuracy is grid-limited
// (the prototypes fe_gridri_j/jp/k show the tighter convergence).

namespace {

std::vector<double> sym(int n, double s) {
  std::vector<double> D((std::size_t)n * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) D[i * n + j] = 0.15 + 0.25 * std::cos(s * i + 1.2 * j);
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) { double a = 0.5 * (D[i * n + j] + D[j * n + i]); D[i * n + j] = D[j * n + i] = a; }
  return D;
}

// grid-RI J with an s+p basis matches the exact GTO coulomb_build.
TEST(GridRI, CoulombMatchesGTO) {
  std::vector<intti::PrimitiveShell<double>> shells = {
      {2.0, {0.0, 0.0, 0.0}, 0}, {1.3, {0.0, 0.0, 0.0}, 1}, {1.1, {0.0, 0.0, 1.3}, 0}};
  auto sb = intti::make_basis(shells);
  const int n = sb.nao;
  auto D = sym(n, 0.8);
  auto grid = intti::grid_for_basis(sb, 1e-2);
  intti::TGridSpec<double> spec; spec.n = 16;
  auto Jg = intti::grid_coulomb_build(sb, D.data(), grid,
                                      intti::make_tgrid(intti::coulomb(), spec), 14);
  std::vector<double> Jref((std::size_t)n * n, 0.0);
  intti::coulomb_build(sb, D.data(), intti::make_tgrid(intti::coulomb()), Jref.data());
  double worst = 0, scale = 0;
  for (std::size_t k = 0; k < Jref.size(); ++k) {
    worst = std::max(worst, std::abs(Jg[k] - Jref[k]));
    scale = std::max(scale, std::abs(Jref[k]));
  }
  EXPECT_LT(worst, 3e-2 * scale) << "grid-RI J vs GTO J, grid N=" << grid.N;
}

// grid-RI K (co-densities, rank-1 occupied set) matches the exact GTO
// exchange_build on the corresponding density D = Cocc Cocc^T.
TEST(GridRI, ExchangeMatchesGTO) {
  std::vector<intti::PrimitiveShell<double>> shells = {
      {2.0, {0.0, 0.0, 0.0}, 0}, {0.9, {0.0, 0.0, 0.0}, 0}, {1.1, {0.0, 0.0, 1.3}, 0}};
  auto sb = intti::make_basis(shells);
  const int n = sb.nao, nocc = 1;
  std::vector<double> C((std::size_t)n * nocc);
  for (int u = 0; u < n; ++u) C[u] = 0.3 + 0.2 * std::sin(1.3 * u);
  std::vector<double> D((std::size_t)n * n, 0.0);
  for (int u = 0; u < n; ++u)
    for (int v = 0; v < n; ++v) D[u * n + v] = C[u] * C[v];
  auto grid = intti::grid_for_basis(sb, 2e-2);
  intti::TGridSpec<double> spec; spec.n = 14;
  auto Kg = intti::grid_exchange_build(sb, C.data(), nocc, grid,
                                       intti::make_tgrid(intti::coulomb(), spec), 12);
  std::vector<double> Kref((std::size_t)n * n, 0.0);
  intti::exchange_build(sb, D.data(), intti::make_tgrid(intti::coulomb()), Kref.data(), 0.0);
  double worst = 0, scale = 0;
  for (std::size_t k = 0; k < Kref.size(); ++k) {
    worst = std::max(worst, std::abs(Kg[k] - Kref[k]));
    scale = std::max(scale, std::abs(Kref[k]));
  }
  EXPECT_LT(worst, 8e-2 * scale) << "grid-RI K vs GTO K, grid N=" << grid.N;
}

// A small generally-contracted s+p basis.
intti::ContractedBasis<double> contracted_sp() {
  intti::ContractedShell<double> s;
  s.center[0] = 0; s.center[1] = 0; s.center[2] = 0;
  s.l = 0; s.alpha = {2.0, 0.9}; s.coeff = {0.6, 0.5};
  intti::ContractedShell<double> p;
  p.center[0] = 0; p.center[1] = 0; p.center[2] = 1.2;
  p.l = 1; p.alpha = {1.5, 0.8}; p.coeff = {0.55, 0.5};
  return intti::make_contracted_basis<double>({s, p});
}

// grid-RI J over a generally-contracted basis matches the analytic contracted
// coulomb_build -- contraction is absorbed into the pointwise AO evaluation.
TEST(GridRI, ContractedCoulombMatchesGTO) {
  auto cb = contracted_sp();
  const int n = cb.nao; // 1 + 3 = 4
  auto D = sym(n, 0.9);
  auto grid = intti::grid_for_basis(cb, 5e-2);
  intti::TGridSpec<double> spec; spec.n = 16;
  auto Jg = intti::grid_coulomb_build(cb, D.data(), grid,
                                      intti::make_tgrid(intti::coulomb(), spec), 14);
  std::vector<double> Jref((std::size_t)n * n, 0.0);
  intti::coulomb_build(cb, D.data(), intti::make_tgrid(intti::coulomb()), Jref.data());
  double worst = 0, scale = 0;
  for (std::size_t k = 0; k < Jref.size(); ++k) {
    worst = std::max(worst, std::abs(Jg[k] - Jref[k]));
    scale = std::max(scale, std::abs(Jref[k]));
  }
  EXPECT_LT(worst, 6e-2 * scale) << "contracted grid-RI J vs analytic, grid N=" << grid.N;
}

// grid-RI K over a generally-contracted basis matches the analytic contracted
// exchange_build on D = Cocc Cocc^T.
TEST(GridRI, ContractedExchangeMatchesGTO) {
  auto cb = contracted_sp();
  const int n = cb.nao, nocc = 1;
  std::vector<double> C((std::size_t)n * nocc);
  for (int u = 0; u < n; ++u) C[u] = 0.3 + 0.2 * std::sin(1.1 * u);
  std::vector<double> D((std::size_t)n * n, 0.0);
  for (int u = 0; u < n; ++u)
    for (int v = 0; v < n; ++v) D[u * n + v] = C[u] * C[v];
  auto grid = intti::grid_for_basis(cb, 5e-2);
  intti::TGridSpec<double> spec; spec.n = 14;
  auto Kg = intti::grid_exchange_build(cb, C.data(), nocc, grid,
                                       intti::make_tgrid(intti::coulomb(), spec), 12);
  std::vector<double> Kref((std::size_t)n * n, 0.0);
  intti::exchange_build(cb, D.data(), intti::make_tgrid(intti::coulomb()), Kref.data(), 0.0);
  double worst = 0, scale = 0;
  for (std::size_t k = 0; k < Kref.size(); ++k) {
    worst = std::max(worst, std::abs(Kg[k] - Kref[k]));
    scale = std::max(scale, std::abs(Kref[k]));
  }
  EXPECT_LT(worst, 1.5e-1 * scale) << "contracted grid-RI K vs analytic, grid N=" << grid.N;
}

} // namespace
