// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/fock.hpp"
#include "intti/oneel.hpp"
#include "intti/sto.hpp"

namespace {

const double PI = 3.14159265358979323846;

TEST(STO, ExpansionReproducesExponential) {
  const double zeta = 1.3;
  std::vector<double> s, c;
  intti::sto_gaussians(zeta, 96, s, c);
  double worst = 0;
  for (double r = 0.02; r < 10.0; r *= 1.1) {
    double approx = 0;
    for (std::size_t k = 0; k < s.size(); ++k) approx += c[k] * std::exp(-s[k] * r * r);
    worst = std::max(worst, std::abs(approx - std::exp(-zeta * r)));
  }
  EXPECT_LT(worst, 1e-9) << "s-expansion does not reproduce exp(-zeta r)";
}

TEST(STO, OverlapMatchesAnalytic1s2p) {
  const double zeta = 1.1;
  // 1s (l=0) and 2p (l=1) minimal STOs at the same centre
  std::vector<intti::StoShell<double>> shells = {
      {zeta, {0.0, 0.0, 0.0}, 0, 96}, {zeta, {0.0, 0.0, 0.0}, 1, 96}};
  auto ex = intti::expand_sto(shells);
  auto Sprim = intti::overlap_matrix(ex.prim);
  auto S = intti::contract_to_sto(Sprim, ex);
  const int na = ex.nsto_ao; // 1 (s) + 3 (p)
  // unnormalized Slater self-overlaps: <e^{-zr}|e^{-zr}> = pi/z^3 (1s);
  // <x e^{-zr}|x e^{-zr}> = pi/z^5 (each 2p Cartesian component)
  const double s1s = S[0 * na + 0];
  EXPECT_NEAR(s1s, PI / std::pow(zeta, 3), 1e-8 * PI / std::pow(zeta, 3));
  for (int j = 1; j <= 3; ++j)
    EXPECT_NEAR(S[j * na + j], PI / std::pow(zeta, 5), 1e-8 * PI / std::pow(zeta, 5))
        << "2p component " << j;
}

TEST(STO, SelfRepulsionMatchesSlater) {
  // 1s Slater self-repulsion int int rho rho / r12 = 5 zeta / 8, rho = phi^2,
  // phi = sqrt(zeta^3/pi) e^{-zeta r} normalized. Reproduced by contracting
  // the primitive same-centre (ss|ss) Coulomb integrals over the s-expansion.
  const double zeta = 1.3;
  std::vector<double> s, c;
  intti::sto_gaussians(zeta, 56, s, c); // enough nodes for ~1e-6; keeps the
                                        // primitive (ss|ss) build cheap
  std::vector<intti::PrimitiveShell<double>> prims;
  for (double sk : s) prims.push_back({sk, {0.0, 0.0, 0.0}, 0});
  auto bas = intti::make_basis(prims);
  const int n = bas.nao;
  std::vector<double> D(static_cast<std::size_t>(n) * n);
  const double norm = zeta * zeta * zeta / PI; // (zeta^3/pi)
  for (int k = 0; k < n; ++k)
    for (int l = 0; l < n; ++l) D[k * n + l] = norm * c[k] * c[l];
  auto grid = intti::make_tgrid(intti::coulomb());
  std::vector<double> J(static_cast<std::size_t>(n) * n, 0.0);
  intti::coulomb_build(bas, D.data(), grid, J.data());
  double selfrep = 0;
  for (std::size_t i = 0; i < J.size(); ++i) selfrep += D[i] * J[i];
  EXPECT_NEAR(selfrep, 5 * zeta / 8, 1e-6 * (5 * zeta / 8))
      << "STO self-repulsion != 5 zeta / 8";
}

} // namespace
