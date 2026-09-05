// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/c2s.hpp"
#include "intti/contracted.hpp"
#include "intti/fegrid.hpp"
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
  auto grid = intti::grid_for_basis(sb, 3e-2);
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

// grid-RI J with a d shell matches the exact GTO coulomb_build -- the direct
// high-l accuracy check enabled by the degree-aware grid (grid_for_basis
// resolves the degree-2l AO-product polynomial x Gaussian, not just the
// envelope). Cartesian d (6 components).
TEST(GridRI, HighLCoulombMatchesGTO) {
  std::vector<intti::PrimitiveShell<double>> shells = {
      {1.8, {0.0, 0.0, 0.0}, 0}, {1.2, {0.0, 0.0, 0.0}, 2}}; // s + d
  auto sb = intti::make_basis(shells);
  const int n = sb.nao; // 1 + 6 = 7
  auto D = sym(n, 0.7);
  auto grid = intti::grid_for_basis(sb, 1.5e-2);
  intti::TGridSpec<double> spec; spec.n = 16;
  auto Jg = intti::grid_coulomb_build(sb, D.data(), grid,
                                      intti::make_tgrid(intti::coulomb(), spec), 16);
  std::vector<double> Jref((std::size_t)n * n, 0.0);
  intti::coulomb_build(sb, D.data(), intti::make_tgrid(intti::coulomb()), Jref.data());
  double worst = 0, scale = 0;
  for (std::size_t k = 0; k < Jref.size(); ++k) {
    worst = std::max(worst, std::abs(Jg[k] - Jref[k]));
    scale = std::max(scale, std::abs(Jref[k]));
  }
  EXPECT_LT(worst, 4e-2 * scale) << "d-shell grid-RI J vs exact GTO, grid N=" << grid.N;
}

// A small generally-contracted s+p basis.
intti::ContractedBasis<double> contracted_sp() {
  intti::ContractedShell<double> s;
  s.center[0] = 0; s.center[1] = 0; s.center[2] = 0;
  s.l = 0; s.alpha = {3.0, 1.3}; s.coeff = {0.6, 0.5};
  intti::ContractedShell<double> p;
  p.center[0] = 0; p.center[1] = 0; p.center[2] = 1.2;
  p.l = 1; p.alpha = {2.0, 1.1}; p.coeff = {0.55, 0.5};
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

// The spherical (real solid harmonic) AOs evaluated on the grid are orthogonal:
// an isolated d shell has a diagonal grid overlap. This validates the c2s
// combination + indexing produce genuine harmonics (convention-light).
TEST(GridRI, SphericalOrthogonality) {
  std::vector<intti::PrimitiveShell<double>> shells = {{1.0, {0.0, 0.0, 0.0}, 2}};
  auto sb = intti::make_basis(shells);
  auto grid = intti::grid_for_basis(sb, 1e-6);
  auto sph = intti::ao_values_on_grid_spherical(sb, grid);
  const int nm = intti::nao_spherical(sb); // 5 for a d shell
  ASSERT_EQ(nm, 5);
  double diag = 0, offd = 0;
  for (int m = 0; m < nm; ++m)
    for (int m2 = 0; m2 < nm; ++m2) {
      const double s = intti::fe_inner(grid, sph[m], sph[m2]);
      if (m == m2) diag = std::max(diag, std::abs(s));
      else offd = std::max(offd, std::abs(s));
    }
  EXPECT_LT(offd, 1e-4 * diag) << "spherical AOs not orthogonal on the grid";
}

// The spherical grid-RI J is the c2s-transform of the Cartesian grid-RI J:
// J_sph(D_sph) == C J_cart(C^T D_sph C) C^T on the same grid (exact identity,
// grid error cancels). Validates the spherical builder applies c2s consistently.
TEST(GridRI, SphericalCoulombConsistentWithCartesian) {
  std::vector<intti::PrimitiveShell<double>> shells = {
      {1.6, {0.0, 0.0, 0.0}, 0}, {1.1, {0.0, 0.0, 0.0}, 2}}; // s + d
  auto sb = intti::make_basis(shells);
  const int nc = sb.nao, nsph = intti::nao_spherical(sb);
  // block-diagonal C (nsph x nc)
  std::vector<double> C((std::size_t)nsph * nc, 0.0);
  {
    int so = 0;
    for (std::size_t s = 0; s < shells.size(); ++s) {
      const int l = shells[s].l, ncc = intti::ncart(l), nm = 2 * l + 1;
      auto Cl = intti::c2s_matrix<double>(l);
      const int co = sb.ao_off[s];
      for (int m = 0; m < nm; ++m)
        for (int k = 0; k < ncc; ++k)
          C[(std::size_t)(so + m) * nc + co + k] = Cl[(std::size_t)m * ncc + k];
      so += nm;
    }
  }
  // spherical density D_sph, then D_cart = C^T D_sph C
  auto Dsph = sym(nsph, 0.7);
  std::vector<double> DC((std::size_t)nsph * nc, 0.0), Dcart((std::size_t)nc * nc, 0.0);
  for (int m = 0; m < nsph; ++m)
    for (int k = 0; k < nc; ++k) {
      double t = 0;
      for (int m2 = 0; m2 < nsph; ++m2) t += Dsph[m * nsph + m2] * C[(std::size_t)m2 * nc + k];
      DC[(std::size_t)m * nc + k] = t;
    }
  for (int a = 0; a < nc; ++a)
    for (int b = 0; b < nc; ++b) {
      double t = 0;
      for (int m = 0; m < nsph; ++m) t += C[(std::size_t)m * nc + a] * DC[(std::size_t)m * nc + b];
      Dcart[a * nc + b] = t;
    }
  // Exactness identity on the SAME grid: J_sph_grid == C (J_cart_grid) C^T. The
  // grid error cancels, so this validates that the spherical builder applies c2s
  // consistently in both the AO evaluation and the contraction (machine
  // precision), independent of grid resolution.
  auto grid = intti::grid_for_basis(sb, 3e-2);
  intti::TGridSpec<double> spec; spec.n = 14;
  auto tg = intti::make_tgrid(intti::coulomb(), spec);
  auto Jcart = intti::grid_coulomb_build(sb, Dcart.data(), grid, tg, 12);
  std::vector<double> CJ((std::size_t)nsph * nc, 0.0), Jref((std::size_t)nsph * nsph, 0.0);
  for (int m = 0; m < nsph; ++m)
    for (int b = 0; b < nc; ++b) {
      double t = 0;
      for (int a = 0; a < nc; ++a) t += C[(std::size_t)m * nc + a] * Jcart[a * nc + b];
      CJ[(std::size_t)m * nc + b] = t;
    }
  for (int m = 0; m < nsph; ++m)
    for (int m2 = 0; m2 < nsph; ++m2) {
      double t = 0;
      for (int b = 0; b < nc; ++b) t += CJ[(std::size_t)m * nc + b] * C[(std::size_t)m2 * nc + b];
      Jref[m * nsph + m2] = t;
    }
  auto Jg = intti::grid_coulomb_build_spherical(sb, Dsph.data(), grid, tg, 12);
  double worst = 0, scale = 0;
  for (std::size_t k = 0; k < Jref.size(); ++k) {
    worst = std::max(worst, std::abs(Jg[k] - Jref[k]));
    scale = std::max(scale, std::abs(Jref[k]));
  }
  EXPECT_LT(worst, 1e-9 * (scale + 1e-12)) << "spherical J != c2s(cartesian grid J)";
}

} // namespace
