// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/ncenter.hpp"
#include "intti/quartet.hpp"

namespace {

using Shell = intti::PrimitiveShell<double>;

intti::ShellBasis<double> orb_basis() {
  return intti::make_basis<double>({{1.2, {0.0, 0.0, 0.0}, 0},
                                    {0.8, {0.0, 0.0, 0.0}, 1},
                                    {0.5, {0.4, -0.3, 0.8}, 0}});
}
intti::ShellBasis<double> aux_basis() {
  return intti::make_basis<double>({{2.6, {0.0, 0.0, 0.0}, 0},
                                    {1.1, {0.0, 0.0, 0.0}, 2},
                                    {1.7, {0.4, -0.3, 0.8}, 1}});
}

TEST(NCenter, TwoCenterSymmetricAndMatchesGhost) {
  auto aux = aux_basis();
  auto grid = intti::make_tgrid(intti::coulomb());
  auto M = intti::coulomb_2c(aux, grid);
  const int n = aux.nao;
  double as = 0, mx = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      as = std::max(as, std::abs(M[i * n + j] - M[j * n + i]));
      mx = std::max(mx, std::abs(M[i * n + j]));
    }
  EXPECT_LT(as, 1e-13 * mx);
  // element (shell0 P, shell2 Q) vs a direct ghost quartet
  auto braP = intti::detail::ghost_pair(aux.shells[0]);
  auto ketQ = intti::detail::ghost_pair(aux.shells[2]);
  std::vector<double> blk(1 * 3);
  intti::eri_quartet(braP, ketQ, grid, blk.data());
  for (int kQ = 0; kQ < 3; ++kQ)
    EXPECT_NEAR(M[aux.ao_off[0] * n + aux.ao_off[2] + kQ], blk[kQ], 1e-13);
}

TEST(NCenter, ThreeCenterMatchesGhostQuartet) {
  auto orb = orb_basis();
  auto aux = aux_basis();
  auto grid = intti::make_tgrid(intti::coulomb());
  auto T = intti::coulomb_3c(orb, aux, grid);
  const int nao = orb.nao, naux = aux.nao;
  // orb shells (m=1 p, n=0 s) x aux shell (a=2, p) block
  auto bra = intti::make_pair(orb.shells[1], orb.shells[0]);
  auto ket = intti::detail::ghost_pair(aux.shells[2]);
  const int nm = 3, nn = 1, nP = 3;
  std::vector<double> blk(nm * nn * nP);
  intti::eri_quartet(bra, ket, grid, blk.data());
  for (int km = 0; km < nm; ++km)
    for (int kP = 0; kP < nP; ++kP) {
      const int mu = orb.ao_off[1] + km, nu = orb.ao_off[0], P = aux.ao_off[2] + kP;
      EXPECT_NEAR(T[(static_cast<std::size_t>(mu) * nao + nu) * naux + P],
                  blk[(km * nn + 0) * nP + kP], 1e-13);
    }
}

// M-MP: the RI 3-center far-field (far_tau > 0) must reproduce the exact
// t-quadrature (mu nu|P) tensor to ~far_tau when the orbital pairs and auxiliary
// functions are well separated. Orbitals at the origin, aux far away (and one
// aux near, exercising both branches).
TEST(NCenter, ThreeCenterFarFieldMatchesExact) {
  auto orb = intti::make_basis<double>({{1.2, {0, 0, 0}, 0},
                                        {0.8, {0, 0, 0}, 1},
                                        {0.5, {0.3, 0, 0}, 0}});
  auto aux = intti::make_basis<double>({{2.6, {0.1, 0, 0}, 0},   // near
                                        {1.1, {0, 0, 18.0}, 2},  // far
                                        {1.7, {20.0, -3.0, 0}, 1}}); // far
  auto grid = intti::make_tgrid(intti::coulomb());
  auto Texact = intti::coulomb_3c(orb, aux, grid, /*far_tau=*/0.0);
  auto Tfar = intti::coulomb_3c(orb, aux, grid, /*far_tau=*/1e-13);
  double scale = 0;
  for (double v : Texact) scale = std::max(scale, std::abs(v));
  for (std::size_t i = 0; i < Texact.size(); ++i)
    EXPECT_LT(std::abs(Tfar[i] - Texact[i]), 1e-10 * scale) << "elem " << i;
}

} // namespace
