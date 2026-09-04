// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
//
// M-TC: the non-Hermitian transcorrelated two-body integral
//   <pq| nabla_1 u_12 . nabla_1 |rs>,  u = sum_k c_k e^{-g_k r12^2}.
// tc_gradu_grad_quartet reduces it (integration by parts) to plain
// Gaussian-geminal integrals with derivative-shifted orbitals. Reference values
// come from the independent symbolic oracle references/sympy_tc.py (direct 6D
// Gaussian integration of the operator -- no by-parts, no libintti).

#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "intti/tc.hpp"

namespace {

using intti::PrimitiveShell;

// Fixed geometry shared with references/sympy_tc.py:
//   p: a=1.2 @ (0,0,0)   q: b=0.9 @ (0,1,0)
//   r: c=0.8 @ (1/2,0,0) s: d=1.1 @ (1/5,9/10,3/10)
static PrimitiveShell<double> P(int l) { return {1.2, {0, 0, 0}, l}; }
static PrimitiveShell<double> Q(int l) { return {0.9, {0, 1, 0}, l}; }
static PrimitiveShell<double> R(int l) { return {0.8, {0.5, 0, 0}, l}; }
static PrimitiveShell<double> S(int l) { return {1.1, {0.2, 0.9, 0.3}, l}; }

TEST(TC, NonHermitianSSSS) {
  auto gem = intti::gaussian_geminal<double>({0.7}, {1.0});
  double out[1];
  intti::tc_gradu_grad_quartet(P(0), Q(0), R(0), S(0), gem, out);
  // oracle: full 6D + factorised 2D symbolic (references/sympy_tc.py)
  EXPECT_NEAR(out[0], 0.6919329300223959, 1e-12);
}

TEST(TC, NonHermitianGeminalSum) {
  auto gem = intti::gaussian_geminal<double>({0.2, 0.8, 3.0}, {0.5, 0.3, 0.2});
  double out[1];
  intti::tc_gradu_grad_quartet(P(0), Q(0), R(0), S(0), gem, out);
  EXPECT_NEAR(out[0], 0.5896308883697698, 1e-12);
}

// Ket orbital r carries angular momentum (the operator's derivative acts on it).
TEST(TC, NonHermitianKetPtype) {
  auto gem = intti::gaussian_geminal<double>({0.7}, {1.0});
  double out[3]; // r is l=1: components px, py, pz
  intti::tc_gradu_grad_quartet(P(0), Q(0), R(1), S(0), gem, out);
  EXPECT_NEAR(out[0], -0.373931732962435, 1e-12); // r = p_x
}

// Both electron-1 orbitals carry angular momentum.
TEST(TC, NonHermitianBraKetPtype) {
  auto gem = intti::gaussian_geminal<double>({0.7}, {1.0});
  double out[9]; // p is l=1 (3), r is l=1 (3); layout (kp*1+kq)*3+kr, kq=ks=0
  intti::tc_gradu_grad_quartet(P(1), Q(0), R(1), S(0), gem, out);
  // p = p_y (kp=1), r = p_x (kr=0)
  EXPECT_NEAR(out[(1 * 1 + 0) * 3 + 0], -0.015163762479314398, 1e-12);
}

// Matrix-level non-Hermitian TC Fock build vs a dense reference contraction of
// the validated quartet. Confirms the density-in/matrix-out driver's indexing,
// and that F is non-symmetric (the operator is non-Hermitian).
TEST(TC, NonHermitianFockBuild) {
  auto basis = intti::make_basis<double>({{1.3, {0, 0, 0}, 0},
                                          {0.7, {0.4, 0, 0}, 1},
                                          {1.0, {0, 0.5, -0.3}, 2}, // d: exercises c>=2
                                          {0.9, {0, 0.5, -0.3}, 0}});
  auto gem = intti::gaussian_geminal<double>({0.5, 1.4}, {0.6, 0.3});
  const int nao = basis.nao;
  std::vector<double> Dmat(static_cast<std::size_t>(nao) * nao);
  std::mt19937 rng(31);
  std::uniform_real_distribution<double> u(-1, 1);
  for (auto &d : Dmat) d = u(rng);

  auto F = intti::tc_gradu_grad_build(basis, Dmat.data(), gem);

  // dense reference: loop shell quartets, contract the validated quartet
  std::vector<double> Fref(static_cast<std::size_t>(nao) * nao, 0.0);
  const int ns = static_cast<int>(basis.shells.size());
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b)
      for (int c = 0; c < ns; ++c)
        for (int d = 0; d < ns; ++d) {
          const auto &P = basis.shells[a], &Q = basis.shells[b];
          const auto &R = basis.shells[c], &S = basis.shells[d];
          const int np = intti::ncart(P.l), nq = intti::ncart(Q.l);
          const int nr = intti::ncart(R.l), nsc = intti::ncart(S.l);
          std::vector<double> blk(static_cast<std::size_t>(np) * nq * nr * nsc);
          intti::tc_gradu_grad_quartet(P, Q, R, S, gem, blk.data());
          for (int kp = 0; kp < np; ++kp)
            for (int kq = 0; kq < nq; ++kq)
              for (int kr = 0; kr < nr; ++kr)
                for (int ks = 0; ks < nsc; ++ks)
                  Fref[(basis.ao_off[a] + kp) * nao + basis.ao_off[c] + kr] +=
                      Dmat[(basis.ao_off[b] + kq) * nao + basis.ao_off[d] + ks] *
                      blk[((kp * nq + kq) * nr + kr) * nsc + ks];
        }
  double scale = 0, asym = 0;
  for (double v : Fref) scale = std::max(scale, std::abs(v));
  for (std::size_t i = 0; i < F.size(); ++i)
    EXPECT_NEAR(F[i], Fref[i], 1e-12 * scale) << "elem " << i;
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j)
      asym = std::max(asym, std::abs(F[i * nao + j] - F[j * nao + i]));
  EXPECT_GT(asym, 1e-3 * scale); // genuinely non-Hermitian
}

// Screening (tau > 0) must reproduce the exact (tau = 0) non-Hermitian Fock
// build to ~tau: the geminal is short-ranged, so pairs across a wide gap are
// safely pruned. Two clusters far apart exercise the screen.
TEST(TC, NonHermitianFockScreened) {
  auto basis = intti::make_basis<double>({{1.3, {0, 0, 0}, 0},
                                          {0.7, {0.3, 0, 0}, 1},
                                          {1.1, {12.0, 0, 0}, 0},
                                          {0.6, {12.3, 0, 0}, 1}});
  auto gem = intti::gaussian_geminal<double>({0.8, 2.0}, {0.5, 0.4});
  const int nao = basis.nao;
  std::vector<double> Dmat(static_cast<std::size_t>(nao) * nao);
  std::mt19937 rng(9);
  std::uniform_real_distribution<double> u(-1, 1);
  for (auto &d : Dmat) d = u(rng);
  auto Fexact = intti::tc_gradu_grad_build(basis, Dmat.data(), gem, 0.0);
  auto Fscr = intti::tc_gradu_grad_build(basis, Dmat.data(), gem, 1e-11);
  double scale = 0;
  for (double v : Fexact) scale = std::max(scale, std::abs(v));
  for (std::size_t i = 0; i < Fexact.size(); ++i)
    EXPECT_NEAR(Fscr[i], Fexact[i], 1e-8 * scale) << "elem " << i;
}

} // namespace
