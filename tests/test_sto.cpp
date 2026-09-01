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
      {zeta, {0.0, 0.0, 0.0}, 0, 0, 96}, {zeta, {0.0, 0.0, 0.0}, 1, 0, 96}};
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

TEST(STO, HigherNRadialExpansion) {
  // r^m e^{-zeta r} = sum_k c_k e^{-s_k r^2} via the zeta-derivative coefficients
  const double zeta = 1.2;
  for (int m = 0; m <= 3; ++m) {
    std::vector<double> s, c;
    intti::sto_gaussians(zeta, 200, s, c, m);
    double worst = 0, scaleref = 0;
    for (double r = 0.05; r < 9.0; r *= 1.15) {
      double approx = 0;
      for (std::size_t k = 0; k < s.size(); ++k) approx += c[k] * std::exp(-s[k] * r * r);
      const double exact = std::pow(r, m) * std::exp(-zeta * r);
      worst = std::max(worst, std::abs(approx - exact));
      scaleref = std::max(scaleref, std::abs(exact));
    }
    EXPECT_LT(worst, 1e-6 * (scaleref + 1)) << "radial power m=" << m;
  }
}

TEST(STO, HigherNOverlapMatchesAnalytic) {
  // s-type STOs of principal quantum number n: <r^{n-1}e^{-zr}|r^{n-1}e^{-zr}>
  // = 4 pi (2n)! / (2z)^{2n+1}
  const double zeta = 1.1;
  auto analytic = [&](int n) {
    double fac = 1;
    for (int i = 2; i <= 2 * n; ++i) fac *= i;
    return 4 * PI * fac / std::pow(2 * zeta, 2 * n + 1);
  };
  for (int n = 1; n <= 3; ++n) {
    std::vector<intti::StoShell<double>> shells = {{zeta, {0.0, 0.0, 0.0}, 0, n, 200}};
    auto ex = intti::expand_sto(shells);
    auto S = intti::contract_to_sto(intti::overlap_matrix(ex.prim), ex);
    EXPECT_NEAR(S[0], analytic(n), 1e-6 * analytic(n)) << "n=" << n;
  }
}

TEST(STO, DeltaTailCorrectionRecoversTruncatedIntegral) {
  // int e^{-zeta r} d^3r = 8 pi / zeta^3. Truncate the s-grid at s_c with few
  // nodes: sum_k c_k (pi/s_k)^{3/2} misses the delta-like tail; adding the
  // analytic delta-tail weight recovers the exact value.
  const double zeta = 1.3;
  const double s_c = 1.0; // aggressive truncation -> sizeable tail
  std::vector<double> s, c;
  intti::sto_gaussians(zeta, 24, s, c, 0, 1e-4, s_c);
  double trunc = 0;
  for (std::size_t k = 0; k < s.size(); ++k) trunc += c[k] * std::pow(PI / s[k], 1.5);
  const double W = intti::sto_delta_tail_weight(zeta, s_c);
  const double exact = 8 * PI / (zeta * zeta * zeta);
  EXPECT_GT(std::abs(exact - trunc), 0.01 * exact) << "tail should be sizeable at s_c=1";
  // residual is the 24-node low-s quadrature error, not the delta correction
  EXPECT_NEAR(trunc + W, exact, 1e-5 * exact) << "delta-tail correction incomplete";
}

TEST(STO, DeltaTailOverlapBuilderMatchesFullGrid) {
  // 1s STOs on three centres. The delta-tail overlap builder on a coarse,
  // truncated s-grid (s_c) must reproduce the full-grid overlap.
  std::vector<intti::StoShell<double>> shells = {
      {1.2, {0.0, 0.0, 0.0}, 0, 0, 0}, {1.0, {1.6, 0.0, 0.0}, 0, 0, 0},
      {1.4, {0.3, 1.5, -0.4}, 0, 0, 0}};
  const int na = static_cast<int>(shells.size());
  // full-grid reference overlap (dense s-grid, no truncation)
  auto ref_shells = shells;
  for (auto &s : ref_shells) {
    s.ns = 200;
    s.smax = 1e6;
  }
  auto exref = intti::expand_sto(ref_shells);
  auto Sref = intti::contract_to_sto(intti::overlap_matrix(exref.prim), exref);
  // delta-tail builder: coarse truncated grid
  auto S = intti::sto_overlap_delta(shells, 30.0, 32);
  double worst = 0, scale = 0;
  for (int i = 0; i < na; ++i)
    for (int j = 0; j < na; ++j) {
      worst = std::max(worst, std::abs(S[i * na + j] - Sref[i * na + j]));
      scale = std::max(scale, std::abs(Sref[i * na + j]));
    }
  EXPECT_LT(worst, 1e-5 * scale) << "delta-tail overlap != full-grid overlap";
}

TEST(STO, DeltaTailOverlapLGreaterZero) {
  // mixed s and p minimal STOs on three centres: the l>0 tail acts as
  // derivatives of delta, so the builder must still match the full grid.
  std::vector<intti::StoShell<double>> shells = {
      {1.2, {0.0, 0.0, 0.0}, 0, 0, 0}, // 1s
      {1.0, {1.7, 0.0, 0.0}, 1, 0, 0}, // 2p
      {1.3, {0.2, 1.5, -0.3}, 1, 0, 0} // 2p
  };
  const int nao = 1 + 3 + 3;
  auto ref_shells = shells;
  for (auto &s : ref_shells) {
    s.ns = 220;
    s.smax = 1e6;
  }
  auto exref = intti::expand_sto(ref_shells);
  auto Sref = intti::contract_to_sto(intti::overlap_matrix(exref.prim), exref);
  auto S = intti::sto_overlap_delta(shells, 30.0, 40, 160);
  double worst = 0, scale = 0;
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j) {
      worst = std::max(worst, std::abs(S[i * nao + j] - Sref[i * nao + j]));
      scale = std::max(scale, std::abs(Sref[i * nao + j]));
    }
  EXPECT_LT(worst, 1e-4 * scale) << "l>0 delta-tail overlap != full-grid overlap";
}

TEST(STO, TwoCenterCoulombMatchesAnalytic) {
  // (rho_A|rho_B) between two equal-exponent 1s Slater densities at distance R
  // has the closed form J = 1/R - e^{-2zR}(1/R + 11z/8 + 3z^2 R/4 + z^3 R^2/6).
  const double z = 1.0, R = 2.0;
  const double A[3] = {0, 0, 0}, B[3] = {0, 0, R};
  const double analytic =
      1 / R - std::exp(-2 * z * R) * (1 / R + 11 * z / 8 + 3 * z * z * R / 4 +
                                      z * z * z * R * R / 6);
  const double full = intti::sto_coulomb_2c(z, A, z, B, 128);
  EXPECT_NEAR(full, analytic, 1e-6 * analytic) << "two-centre STO Coulomb wrong";
}

TEST(STO, TwoCenterCoulombDeltaTailMatchesFull) {
  // the two-electron delta-tail: rho_A on a coarse truncated grid + the tail
  // charge times rho_B's potential at A must reproduce the full-grid value.
  const double zA = 1.2, zB = 0.9, R = 2.3;
  const double A[3] = {0, 0, 0}, B[3] = {0.4, 0.0, R};
  const double full = intti::sto_coulomb_2c(zA, A, zB, B, 160);
  const double delta = intti::sto_coulomb_2c_delta(zA, A, zB, B, 4.0, 20, 160);
  // the truncated-only value (no correction) is visibly short
  const double trunc_only = delta - (zA * zA * zA / M_PI) *
                                        intti::sto_delta_tail_weight(2 * zA, 4.0) *
                                        intti::sto_slater_potential(zB, R);
  EXPECT_GT(std::abs(full - trunc_only), 1e-3 * full) << "tail should matter";
  // delta-tail is an approximation; its accuracy improves as t_c grows (tail
  // narrower). At this aggressive t_c=4 it recovers the full value to ~4e-4.
  EXPECT_NEAR(delta, full, 1e-3 * full) << "two-electron delta-tail != full grid";
}

TEST(STO, JKBuildSingleSelfERI) {
  // single 1s STO, D = [[1]] (unnormalized AO e^{-zeta r}): J[0][0] = K[0][0]
  // = (00|00) = 5 pi^2 / (8 zeta^5) (the normalized 5 zeta/8 scaled by
  // (pi/zeta^3)^2 for the two unnormalized densities).
  const double zeta = 1.3;
  std::vector<intti::StoShell<double>> shells = {{zeta, {0.0, 0.0, 0.0}, 0, 0, 40}};
  std::vector<double> D = {1.0};
  auto grid = intti::make_tgrid(intti::coulomb());
  auto jk = intti::sto_jk_build(shells, D, grid);
  const double exact = 5 * M_PI * M_PI / (8 * std::pow(zeta, 5));
  EXPECT_NEAR(jk.J[0], exact, 2e-3 * exact) << "STO J self-ERI wrong";
  EXPECT_NEAR(jk.K[0], exact, 2e-3 * exact) << "STO K self-ERI wrong";
}

TEST(STO, JKBuildTwoCenterCoulomb) {
  // two 1s STOs; with only B occupied (D = diag(0,1)), J[0][0] = (00|11) =
  // (pi^2/(zA^3 zB^3)) (rho_A|rho_B) -- cross-check vs sto_coulomb_2c.
  const double zA = 1.2, zB = 0.9;
  const double A[3] = {0, 0, 0}, B[3] = {0.3, 0.0, 2.2};
  // inter-centre Coulomb is not cusp-sensitive, so a capped smax lets 20 nodes
  // resolve the s-range well; the analytic reference uses the same density model
  std::vector<intti::StoShell<double>> shells = {{zA, {A[0], A[1], A[2]}, 0, 0, 20, 1e-4, 500.0},
                                                 {zB, {B[0], B[1], B[2]}, 0, 0, 20, 1e-4, 500.0}};
  std::vector<double> D = {0, 0, 0, 1}; // only B occupied
  auto grid = intti::make_tgrid(intti::coulomb());
  auto jk = intti::sto_jk_build(shells, D, grid);
  const double ref = M_PI * M_PI / (std::pow(zA, 3) * std::pow(zB, 3)) *
                     intti::sto_coulomb_2c(zA, A, zB, B, 160);
  EXPECT_NEAR(jk.J[0], ref, 5e-3 * ref) << "STO two-centre J wrong";
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
