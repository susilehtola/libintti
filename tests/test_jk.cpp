// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

// General multi-density J/K builder (jk.hpp). The oracle here is a dense loop
// over explicit shell quartets contracting the DEFINITIONS
//   J_mn = sum_ls (mn|ls) D_ls,   K_mn = sum_ls (ml|ns) D_ls,
// which assumes nothing about the density -- so it is a genuine reference for
// the general and antisymmetric cases the fused builders cannot express.

#include <cmath>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "intti/erigrad.hpp" // detail::eri_block4
#include "intti/fock.hpp"
#include "intti/jk.hpp"
#include "intti/ri.hpp"
#include "intti/tgrid.hpp"

namespace {

using S = intti::PrimitiveShell<double>;

std::vector<S> jk_shells() {
  return {{1.2, {0.0, 0.0, 0.0}, 0},
          {0.7, {0.5, 0.1, -0.2}, 1},
          {0.9, {-0.3, 0.4, 0.2}, 0},
          {0.5, {0.2, -0.4, 0.6}, 1}};
}

/// Reference J and K straight from the definitions, over every ordered quartet.
void ref_jk(const std::vector<S> &sh, const intti::ShellBasis<double> &bas,
            const std::vector<double> &D, const intti::TGrid<double> &grid,
            std::vector<double> &J, std::vector<double> &K) {
  const int n = bas.nao, ns = static_cast<int>(sh.size());
  J.assign(static_cast<std::size_t>(n) * n, 0.0);
  K.assign(static_cast<std::size_t>(n) * n, 0.0);
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b)
      for (int c = 0; c < ns; ++c)
        for (int d = 0; d < ns; ++d) {
          const auto blk = intti::detail::eri_block4(sh[a], sh[b], sh[c], sh[d], grid);
          const int na = intti::ncart(sh[a].l), nb = intti::ncart(sh[b].l);
          const int nc = intti::ncart(sh[c].l), nd = intti::ncart(sh[d].l);
          for (int ka = 0; ka < na; ++ka)
            for (int kb = 0; kb < nb; ++kb)
              for (int kc = 0; kc < nc; ++kc)
                for (int kd = 0; kd < nd; ++kd) {
                  const double v = blk[((ka * nb + kb) * nc + kc) * nd + kd];
                  const int A = bas.ao_off[a] + ka, B = bas.ao_off[b] + kb;
                  const int C = bas.ao_off[c] + kc, Dd = bas.ao_off[d] + kd;
                  J[A * n + B] += v * D[C * n + Dd]; // J_mn = (mn|ls) D_ls
                  K[A * n + C] += v * D[B * n + Dd]; // K_mn = (ml|ns) D_ls
                }
        }
}

std::vector<double> sym_density(int n) {
  std::vector<double> D(static_cast<std::size_t>(n) * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) D[i * n + j] = 0.3 / (1 + std::abs(i - j)) + 0.01 * (i + j);
  return D;
}

std::vector<double> antisym_density(int n) {
  std::vector<double> D(static_cast<std::size_t>(n) * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) D[i * n + j] = 0.1 * (i - j) / (1.0 + i + j);
  return D;
}

std::vector<double> general_density(int n) {
  std::vector<double> D(static_cast<std::size_t>(n) * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) D[i * n + j] = 0.2 / (1 + i + 2 * j) + 0.05 * i;
  return D;
}

double maxdiff(const std::vector<double> &a, const std::vector<double> &b) {
  double m = 0;
  for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

double maxabs(const std::vector<double> &a) {
  double m = 0;
  for (double v : a) m = std::max(m, std::abs(v));
  return m;
}

TEST(JK, GeneralAndAntisymmetricDensitiesMatchDefinition) {
  auto sh = jk_shells();
  auto bas = intti::make_basis(sh);
  auto grid = intti::make_tgrid(intti::coulomb());
  const int n = bas.nao;
  const auto Ds = sym_density(n), Da = antisym_density(n), Dg = general_density(n);
  std::vector<intti::JKRequest<double>> reqs = {
      {Ds.data(), intti::DensitySymmetry::General, intti::FockTerms::CoulombExchange},
      {Da.data(), intti::DensitySymmetry::Antisymmetric, intti::FockTerms::CoulombExchange},
      {Dg.data(), intti::DensitySymmetry::General, intti::FockTerms::CoulombExchange}};
  const auto got = intti::jk_build(bas, reqs, grid);
  ASSERT_EQ(got.J.size(), 3u);
  const std::vector<const std::vector<double> *> Ds3 = {&Ds, &Da, &Dg};
  for (int r = 0; r < 3; ++r) {
    std::vector<double> Jr, Kr;
    ref_jk(sh, bas, *Ds3[r], grid, Jr, Kr);
    const double scale = std::max(maxabs(Jr), maxabs(Kr));
    ASSERT_GT(scale, 1e-3) << "reference is trivially zero for request " << r;
    EXPECT_LT(maxdiff(got.J[r], Jr), 1e-12 * scale) << "J, request " << r;
    EXPECT_LT(maxdiff(got.K[r], Kr), 1e-12 * scale) << "K, request " << r;
  }
}

TEST(JK, StructuralSymmetryOfJAndK) {
  // The properties that make the symmetry tag meaningful, verified on the
  // reference itself rather than assumed: J sees only the symmetric part of D
  // (so it vanishes identically for antisymmetric D and is always symmetric),
  // while K(D)^T = K(D^T) -- so K inherits the density's symmetry.
  auto sh = jk_shells();
  auto bas = intti::make_basis(sh);
  auto grid = intti::make_tgrid(intti::coulomb());
  const int n = bas.nao;
  auto trans = [&](const std::vector<double> &M) {
    std::vector<double> T(M.size());
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) T[i * n + j] = M[j * n + i];
    return T;
  };
  std::vector<double> Js, Ks, Ja, Ka;
  const auto Ds = sym_density(n), Da = antisym_density(n);
  ref_jk(sh, bas, Ds, grid, Js, Ks);
  ref_jk(sh, bas, Da, grid, Ja, Ka);
  EXPECT_GT(maxabs(Js), 1e-3);
  EXPECT_LT(maxdiff(Js, trans(Js)), 1e-12 * maxabs(Js)) << "J is always symmetric";
  EXPECT_LT(maxdiff(Ks, trans(Ks)), 1e-12 * maxabs(Ks)) << "K(symmetric D) is symmetric";
  // J(antisymmetric D) = 0 exactly: (mn|ls) is symmetric in l<->s
  EXPECT_LT(maxabs(Ja), 1e-12 * maxabs(Js)) << "J must vanish for antisymmetric D";
  // K(antisymmetric D) is ANTISYMMETRIC -- what exchange_build cannot represent
  ASSERT_GT(maxabs(Ka), 1e-3);
  std::vector<double> sum(Ka.size());
  for (std::size_t i = 0; i < Ka.size(); ++i) sum[i] = Ka[i] + trans(Ka)[i];
  EXPECT_LT(maxabs(sum), 1e-12 * maxabs(Ka)) << "K(antisymmetric D) must be antisymmetric";
}

TEST(JK, SymmetricRequestsAgreeWithFusedBuilders) {
  // For symmetric densities jk_build delegates to the tuned fused engines; the
  // general path must reproduce them, so the two routes are cross-checked.
  auto sh = jk_shells();
  auto bas = intti::make_basis(sh);
  auto grid = intti::make_tgrid(intti::coulomb());
  const int n = bas.nao;
  const auto D = sym_density(n);
  std::vector<intti::JKRequest<double>> sym = {
      {D.data(), intti::DensitySymmetry::Symmetric, intti::FockTerms::CoulombExchange}};
  std::vector<intti::JKRequest<double>> gen = {
      {D.data(), intti::DensitySymmetry::General, intti::FockTerms::CoulombExchange}};
  const auto a = intti::jk_build(bas, sym, grid);
  const auto b = intti::jk_build(bas, gen, grid);
  const double scale = std::max(maxabs(a.J[0]), maxabs(a.K[0]));
  ASSERT_GT(scale, 1e-3);
  EXPECT_LT(maxdiff(a.J[0], b.J[0]), 1e-12 * scale) << "fused vs general J";
  EXPECT_LT(maxdiff(a.K[0], b.K[0]), 1e-12 * scale) << "fused vs general K";
}

TEST(JK, TermSelectionAndScreening) {
  auto sh = jk_shells();
  auto bas = intti::make_basis(sh);
  auto grid = intti::make_tgrid(intti::coulomb());
  const int n = bas.nao;
  const auto D = general_density(n);
  std::vector<intti::JKRequest<double>> reqs = {
      {D.data(), intti::DensitySymmetry::General, intti::FockTerms::Coulomb},
      {D.data(), intti::DensitySymmetry::General, intti::FockTerms::Exchange},
      {D.data(), intti::DensitySymmetry::General, intti::FockTerms::CoulombExchange}};
  const auto got = intti::jk_build(bas, reqs, grid);
  EXPECT_FALSE(got.J[0].empty());
  EXPECT_TRUE(got.K[0].empty()) << "Coulomb-only request must not build K";
  EXPECT_TRUE(got.J[1].empty()) << "Exchange-only request must not build J";
  EXPECT_FALSE(got.K[1].empty());
  // the selected terms must equal the corresponding halves of the both request
  EXPECT_LT(maxdiff(got.J[0], got.J[2]), 1e-13 * maxabs(got.J[2]));
  EXPECT_LT(maxdiff(got.K[1], got.K[2]), 1e-13 * maxabs(got.K[2]));
  // a tight screening threshold must not move the result
  const auto scr = intti::jk_build(bas, reqs, grid, 1e-14);
  EXPECT_LT(maxdiff(scr.J[2], got.J[2]), 1e-11 * maxabs(got.J[2]));
  EXPECT_LT(maxdiff(scr.K[2], got.K[2]), 1e-11 * maxabs(got.K[2]));
}


std::vector<S> jk_aux_shells() {
  return {{2.4, {0.0, 0.0, 0.0}, 0},      {1.1, {0.0, 0.0, 0.0}, 1},
          {0.6, {0.5, 0.1, -0.2}, 0},     {1.8, {0.5, 0.1, -0.2}, 1},
          {0.9, {-0.3, 0.4, 0.2}, 0},     {1.4, {0.2, -0.4, 0.6}, 0},
          {0.5, {0.2, -0.4, 0.6}, 1}};
}

TEST(JK, RiPathIsAlreadyGeneralInTheDensity) {
  // ri_jk contracts K = sum_P B^P D B^P as two GEMMs and never exploits a
  // triangle, so unlike exchange_build it is correct for a general density
  // without any change. Pinned here so a future "optimisation" that assumes
  // symmetry has to fail this test.
  auto osh = jk_shells();
  auto orb = intti::make_basis(osh);
  auto aux = intti::make_basis(jk_aux_shells());
  auto grid = intti::make_tgrid(intti::coulomb());
  const auto fit = intti::ri_fit(orb, aux, grid);
  const int n = orb.nao;
  const auto Da = antisym_density(n), Dg = general_density(n);
  auto trans = [&](const std::vector<double> &M) {
    std::vector<double> T(M.size());
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) T[i * n + j] = M[j * n + i];
    return T;
  };
  // K(D)^T = K(D^T): check on a genuinely non-symmetric density
  std::vector<double> Kg(static_cast<std::size_t>(n) * n), KgT(Kg.size());
  intti::ri_jk(fit, Dg.data(), static_cast<double *>(nullptr), Kg.data());
  const auto DgT = trans(Dg);
  intti::ri_jk(fit, DgT.data(), static_cast<double *>(nullptr), KgT.data());
  ASSERT_GT(maxabs(Kg), 1e-3);
  EXPECT_LT(maxdiff(trans(Kg), KgT), 1e-12 * maxabs(Kg)) << "K(D)^T must equal K(D^T)";
  // antisymmetric density: J vanishes, K is antisymmetric
  std::vector<double> Ja(Kg.size()), Ka(Kg.size());
  intti::ri_jk(fit, Da.data(), Ja.data(), Ka.data());
  ASSERT_GT(maxabs(Ka), 1e-3);
  EXPECT_LT(maxabs(Ja), 1e-12 * maxabs(Kg)) << "RI J must vanish for antisymmetric D";
  std::vector<double> sum(Ka.size());
  const auto KaT = trans(Ka);
  for (std::size_t i = 0; i < Ka.size(); ++i) sum[i] = Ka[i] + KaT[i];
  EXPECT_LT(maxabs(sum), 1e-12 * maxabs(Ka)) << "RI K(antisymmetric D) must be antisymmetric";
}

TEST(JK, RiMultiDensityMatchesPerDensityCalls) {
  auto orb = intti::make_basis(jk_shells());
  auto aux = intti::make_basis(jk_aux_shells());
  auto grid = intti::make_tgrid(intti::coulomb());
  const auto fit = intti::ri_fit(orb, aux, grid);
  const int n = orb.nao;
  const auto Ds = sym_density(n), Da = antisym_density(n), Dg = general_density(n);
  std::vector<intti::JKRequest<double>> reqs = {
      {Ds.data(), intti::DensitySymmetry::Symmetric, intti::FockTerms::CoulombExchange},
      {Da.data(), intti::DensitySymmetry::Antisymmetric, intti::FockTerms::Exchange},
      {Dg.data(), intti::DensitySymmetry::General, intti::FockTerms::Coulomb}};
  const auto got = intti::ri_jk_build(fit, reqs);
  EXPECT_TRUE(got.J[1].empty()) << "exchange-only request must not build J";
  EXPECT_TRUE(got.K[2].empty()) << "Coulomb-only request must not build K";
  const std::vector<const std::vector<double> *> Dv = {&Ds, &Da, &Dg};
  for (int r = 0; r < 3; ++r) {
    std::vector<double> J(static_cast<std::size_t>(n) * n), K(J.size());
    intti::ri_jk(fit, Dv[r]->data(), J.data(), K.data());
    if (!got.J[r].empty())
      EXPECT_LT(maxdiff(got.J[r], J), 1e-12 * (maxabs(J) + 1)) << "J, request " << r;
    if (!got.K[r].empty())
      EXPECT_LT(maxdiff(got.K[r], K), 1e-12 * (maxabs(K) + 1)) << "K, request " << r;
  }
}

TEST(JK, TwoSidedOrbitalExchangeMatchesDensityDriven) {
  // ri_k_occ2 is the form a CPHF perturbed density actually needs: not
  // idempotent, but a product of two DIFFERENT orbital sets, D = C_L C_R^T.
  // Using distinct C_L and C_R makes D non-symmetric, which the one-sided
  // ri_k_occ cannot represent at all.
  auto orb = intti::make_basis(jk_shells());
  auto aux = intti::make_basis(jk_aux_shells());
  auto grid = intti::make_tgrid(intti::coulomb());
  const auto fit = intti::ri_fit(orb, aux, grid);
  const int n = orb.nao, nvec = 3;
  std::vector<double> CL(static_cast<std::size_t>(n) * nvec),
      CR(static_cast<std::size_t>(n) * nvec);
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < nvec; ++k) {
      CL[i * nvec + k] = 0.3 * std::cos(0.7 * i + k) / (1.0 + i);
      CR[i * nvec + k] = 0.2 * std::sin(0.4 * i - 2.0 * k) + 0.05 * k;
    }
  std::vector<double> D(static_cast<std::size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double s = 0;
      for (int k = 0; k < nvec; ++k) s += CL[i * nvec + k] * CR[j * nvec + k];
      D[i * n + j] = s;
    }
  // D is genuinely non-symmetric
  double asym = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) asym = std::max(asym, std::abs(D[i * n + j] - D[j * n + i]));
  ASSERT_GT(asym, 1e-3) << "test density is symmetric, defeating the point";
  std::vector<double> Kref(D.size()), Korb(D.size());
  intti::ri_jk(fit, D.data(), static_cast<double *>(nullptr), Kref.data());
  intti::ri_k_occ2(fit, CL.data(), CR.data(), nvec, Korb.data());
  ASSERT_GT(maxabs(Kref), 1e-4);
  EXPECT_LT(maxdiff(Korb, Kref), 1e-12 * maxabs(Kref)) << "two-sided orbital vs density form";
  // and C_L = C_R must reproduce the existing one-sided routine exactly
  std::vector<double> K1(D.size()), K2(D.size());
  intti::ri_k_occ(fit, CL.data(), nvec, K1.data());
  intti::ri_k_occ2(fit, CL.data(), CL.data(), nvec, K2.data());
  EXPECT_LT(maxdiff(K1, K2), 1e-12 * (maxabs(K1) + 1)) << "C_L = C_R must recover ri_k_occ";
}

} // namespace
