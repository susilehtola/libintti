// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/threeel_ri.hpp"

namespace {

using intti::CartGauss;

// Exact O(nao^6) reference: E3 = sum_{abcdef} G_abcdef D_ad D_be D_cf over the
// unnormalised primitive orbitals (the same convention the RI path uses).
double exact_e3(const std::vector<CartGauss<double>> &g, const std::vector<double> &D, int n,
                const intti::TGrid<double> &grid) {
  double E = 0;
  for (int a = 0; a < n; ++a)
    for (int d = 0; d < n; ++d)
      for (int b = 0; b < n; ++b)
        for (int e = 0; e < n; ++e)
          for (int c = 0; c < n; ++c)
            for (int f = 0; f < n; ++f)
              E += D[a * n + d] * D[b * n + e] * D[c * n + f] *
                   intti::detail::three_electron_raw(g[a], g[b], g[c], g[d], g[e], g[f], grid);
  return E;
}

TEST(ThreeElRI, ExactWhenAuxSpansProducts) {
  // Two s orbitals -> three distinct products (s-Gaussians at the product
  // centres). With the auxiliary basis chosen to be exactly those products, the
  // density lies in span(aux), the fit is exact, and the RI-folded energy must
  // reproduce the direct sextet sum to numerical precision.
  auto grid = intti::make_tgrid(intti::coulomb());
  auto orb = intti::make_basis<double>({{1.0, {0.0, 0.0, 0.0}, 0}, {0.8, {0.5, 0.0, 0.0}, 0}});
  const std::vector<double> D = {1.0, 0.3, 0.3, 0.7};
  const double p01 = (1.0 * 0.0 + 0.8 * 0.5) / 1.8; // product centre of the 0-1 pair
  auto aux = intti::make_basis<double>({{2.0, {0.0, 0.0, 0.0}, 0},
                                        {1.6, {0.5, 0.0, 0.0}, 0},
                                        {1.8, {p01, 0.0, 0.0}, 0}});
  const double eri = intti::three_electron_energy_ri(orb, D, aux, grid);
  const auto og = intti::detail::shellbasis_to_cartgauss(orb);
  const double eex = exact_e3(og, D, 2, grid);
  EXPECT_NEAR(eri, eex, 1e-9 * std::abs(eex)) << "RI must be exact when aux spans the products";
}

TEST(ThreeElRI, ApproximatesWithIncompleteAux) {
  // A smaller auxiliary basis that does NOT span the products: RI now
  // approximates. It should be close but measurably different from exact.
  auto grid = intti::make_tgrid(intti::coulomb());
  auto orb = intti::make_basis<double>({{1.0, {0.0, 0.0, 0.0}, 0}, {0.8, {0.5, 0.0, 0.0}, 0}});
  const std::vector<double> D = {1.0, 0.3, 0.3, 0.7};
  // two diffuse-ish auxiliaries only
  auto aux = intti::make_basis<double>({{1.9, {0.0, 0.0, 0.0}, 0}, {1.7, {0.5, 0.0, 0.0}, 0}});
  const double eri = intti::three_electron_energy_ri(orb, D, aux, grid);
  const auto og = intti::detail::shellbasis_to_cartgauss(orb);
  const double eex = exact_e3(og, D, 2, grid);
  EXPECT_GT(std::abs(eri - eex), 1e-9 * std::abs(eex)) << "incomplete aux should not be exact";
  EXPECT_NEAR(eri, eex, 0.2 * std::abs(eex)) << "but the RI fit should still be reasonable";
}

TEST(ThreeElRI, ExactForHigherLProducts) {
  // A single p-shell's pairwise products chi_pi chi_pj = (r-A)_i (r-A)_j
  // e^{-2a(r-A)^2} span exactly one d-shell at the same centre with exponent 2a.
  // So RI is again exact -- and this exercises the cart_comp AO ordering for
  // both the orbital p-shell and the auxiliary d-shell.
  auto grid = intti::make_tgrid(intti::coulomb());
  const double a = 0.9;
  auto orb = intti::make_basis<double>({{a, {0.1, 0.2, 0.3}, 1}});          // p-shell, nao=3
  auto aux = intti::make_basis<double>({{2 * a, {0.1, 0.2, 0.3}, 2}});      // d-shell, naux=6
  // symmetric 3x3 density over the p components
  const std::vector<double> D = {0.7, 0.2, -0.1, 0.2, 0.5, 0.15, -0.1, 0.15, 0.9};
  const double eri = intti::three_electron_energy_ri(orb, D, aux, grid);
  const auto og = intti::detail::shellbasis_to_cartgauss(orb);
  const double eex = exact_e3(og, D, 3, grid);
  EXPECT_NEAR(eri, eex, 1e-9 * std::abs(eex)) << "RI must be exact for the p x p -> d products";
}

} // namespace
