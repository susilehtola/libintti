// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/contracted.hpp"
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

// The 2- and 3-centre builds are the RI hot path, and their quartets are plain
// Coulomb integrals over real pairs -- exactly what the t-resolved node
// truncation (screening.hpp) bounds. They previously took no screening
// tolerance at all, so an extended system evaluated the full t-grid for every
// aux function no matter how far away it was.
TEST(NCenter, ScreenedTwoAndThreeCentreMatchTheExactBuild) {
  auto grid = intti::make_tgrid(intti::coulomb());
  // a chain, so most orbital-pair/aux-function distances are large
  std::vector<Shell> osh, ash;
  for (int i = 0; i < 6; ++i) {
    osh.push_back({1.1 + 0.2 * i, {4.5 * i, 0.0, 0.0}, i % 2});
    ash.push_back({2.2 + 0.4 * i, {4.5 * i, 0.0, 0.0}, i % 3});
  }
  auto orb = intti::make_basis(osh), aux = intti::make_basis(ash);

  const auto M0 = intti::coulomb_2c(aux, grid);
  const auto T0 = intti::coulomb_3c(orb, aux, grid);
  double sm = 0, st = 0;
  for (double v : M0) sm = std::max(sm, std::abs(v));
  for (double v : T0) st = std::max(st, std::abs(v));
  ASSERT_GT(sm, 1e-3);
  ASSERT_GT(st, 1e-3);

  for (double tau : {1e-13, 1e-11}) {
    const auto M = intti::coulomb_2c(aux, grid, tau);
    const auto T = intti::coulomb_3c(orb, aux, grid, 0.0, tau);
    double dm = 0, dt = 0;
    for (std::size_t i = 0; i < M0.size(); ++i) dm = std::max(dm, std::abs(M[i] - M0[i]));
    for (std::size_t i = 0; i < T0.size(); ++i) dt = std::max(dt, std::abs(T[i] - T0[i]));
    // eps is an ABSOLUTE per-quartet budget: the error may not exceed it
    EXPECT_LT(dm, tau) << "2c screened beyond its budget at tau=" << tau;
    EXPECT_LT(dt, tau) << "3c screened beyond its budget at tau=" << tau;
  }
  // and it must actually be doing something at a loose tolerance
  const auto Mloose = intti::coulomb_2c(aux, grid, 1e-8);
  double dl = 0;
  for (std::size_t i = 0; i < M0.size(); ++i)
    dl = std::max(dl, std::abs(Mloose[i] - M0[i]));
  EXPECT_GT(dl, 0.0) << "screening changed nothing -- it is not reaching this path";
  EXPECT_LT(dl, 1e-8);
}

TEST(NCenter, ScreenedContractedTwoAndThreeCentreMatchTheExactBuild) {
  // Contracted: the digest multiplies each primitive quartet by an effective
  // coefficient, so the per-quartet budget is divided by the largest one. If
  // that factor were dropped the errors below would exceed the bound badly.
  //
  // WHAT tau ACTUALLY PROMISES. It is a PER-QUARTET budget -- the same
  // convention as the Schwarz tolerance everywhere else in the library -- and a
  // contracted output element is a sum over nprim_M * nprim_N * nprim_A
  // primitive quartets, each entitled to its own. So the guarantee on the
  // OUTPUT carries that count, and this test checks it at the right size rather
  // than pretending tau bounds the element directly. The earlier form of this
  // test asserted the element bound and passed only because the estimate was
  // loose enough not to spend its budget; tightening the estimate exposed that,
  // which is the honest reading, not a regression.
  auto grid = intti::make_tgrid(intti::coulomb());
  std::vector<intti::ContractedShell<double>> osh, ash;
  for (int i = 0; i < 4; ++i) {
    osh.push_back({{5.0 * i, 0.0, 0.0}, i % 2, {2.9, 0.8, 0.24}, {0.15, 0.49, 0.61}});
    ash.push_back({{5.0 * i, 0.0, 0.0}, i % 2, {4.4, 1.3}, {0.6, 0.55}});
  }
  intti::ContractedBasis<double> orb = intti::make_contracted_basis(osh);
  intti::ContractedBasis<double> aux = intti::make_contracted_basis(ash);

  // largest number of primitive quartets behind one output element
  int npmax_o = 0, npmax_a = 0;
  for (const auto &sh : orb.shells) npmax_o = std::max(npmax_o, sh.nprim());
  for (const auto &sh : aux.shells) npmax_a = std::max(npmax_a, sh.nprim());
  const double n2c = double(npmax_a) * npmax_a;
  const double n3c = double(npmax_o) * npmax_o * npmax_a;

  const auto M0 = intti::coulomb_2c(aux, grid);
  const auto T0 = intti::coulomb_3c(orb, aux, grid);
  double prev_m = 0, prev_t = 0;
  for (double tau : {1e-11, 1e-13}) {
    const auto M = intti::coulomb_2c(aux, grid, tau);
    const auto T = intti::coulomb_3c(orb, aux, grid, tau);
    double dm = 0, dt = 0;
    for (std::size_t i = 0; i < M0.size(); ++i) dm = std::max(dm, std::abs(M[i] - M0[i]));
    for (std::size_t i = 0; i < T0.size(); ++i) dt = std::max(dt, std::abs(T[i] - T0[i]));
    EXPECT_LT(dm, tau * n2c) << "contracted 2c screened beyond its budget at tau=" << tau;
    EXPECT_LT(dt, tau * n3c) << "contracted 3c screened beyond its budget at tau=" << tau;
    // and the knob has to work: a tighter tolerance must not give a worse result
    if (prev_m > 0) {
      EXPECT_LE(dm, prev_m) << "tightening tau made 2c worse";
      EXPECT_LE(dt, prev_t) << "tightening tau made 3c worse";
    }
    prev_m = dm;
    prev_t = dt;
  }
}
