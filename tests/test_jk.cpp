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

#include "intti/erigrad.hpp" // detail::eri_block4, two_electron_gradient
#include "intti/fock.hpp"
#include "intti/jk.hpp"
#include "intti/ri.hpp"
#include "intti/rigrad.hpp"
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

TEST(JK, DerivativeMatricesContractToTheKnownGradient) {
  // E_2e = 1/2 sum D J(D) - 1/4 sum D K(D), so at fixed D
  //   dE_2e/dR = 1/2 sum D dJ/dR - 1/4 sum D dK/dR
  // exactly. two_electron_gradient computes the left side and is already
  // validated, so this pins the new matrices against known-good code.
  auto sh = jk_shells();
  auto bas = intti::make_basis(sh);
  auto grid = intti::make_tgrid(intti::coulomb());
  const int n = bas.nao, ns = static_cast<int>(sh.size());
  const auto D = sym_density(n);
  const std::vector<intti::JKRequest<double>> reqs = {
      {D.data(), intti::DensitySymmetry::Symmetric, intti::FockTerms::CoulombExchange}};
  const auto dv = intti::jk_deriv_build(bas, reqs, grid);
  ASSERT_EQ(dv.nshell, ns);
  const auto ref = intti::two_electron_gradient(bas, D.data(), grid);
  ASSERT_EQ(static_cast<int>(ref.size()), ns);
  double scale = 0;
  for (const auto &g : ref)
    for (int e = 0; e < 3; ++e) scale = std::max(scale, std::abs(g[e]));
  ASSERT_GT(scale, 1e-3) << "reference gradient is trivially zero";
  const std::size_t n2 = static_cast<std::size_t>(n) * n;
  for (int s = 0; s < ns; ++s)
    for (int e = 0; e < 3; ++e) {
      const std::size_t o = (static_cast<std::size_t>(3 * s + e)) * n2;
      double g = 0;
      for (std::size_t i = 0; i < n2; ++i)
        g += 0.5 * D[i] * dv.J[0][o + i] - 0.25 * D[i] * dv.K[0][o + i];
      EXPECT_NEAR(g, ref[s][e], 1e-10 * (scale + 1)) << "shell " << s << " comp " << e;
    }
}

TEST(JK, DerivativeMatricesMatchFiniteDifference) {
  // The contraction above only pins a trace of each matrix. This differences
  // jk_build itself with respect to a shell centre, at FIXED density, so every
  // element is checked -- and with a GENERAL density, which the gradient
  // identity above cannot exercise.
  auto sh = jk_shells();
  auto grid = intti::make_tgrid(intti::coulomb());
  auto bas0 = intti::make_basis(sh);
  const int n = bas0.nao;
  const auto D = general_density(n);
  const std::vector<intti::JKRequest<double>> reqs = {
      {D.data(), intti::DensitySymmetry::General, intti::FockTerms::CoulombExchange}};
  const auto dv = intti::jk_deriv_build(bas0, reqs, grid);
  const std::size_t n2 = static_cast<std::size_t>(n) * n;
  const double h = 1e-4;
  for (int s : {0, 1, 3})
    for (int e = 0; e < 3; ++e) {
      auto shifted = [&](double delta) {
        auto sh2 = sh;
        sh2[s].center[e] += delta;
        auto b2 = intti::make_basis(sh2);
        return intti::jk_build(b2, reqs, grid);
      };
      const auto p = shifted(h), m = shifted(-h);
      const std::size_t o = (static_cast<std::size_t>(3 * s + e)) * n2;
      double worst = 0, sc = 0;
      for (std::size_t i = 0; i < n2; ++i) {
        const double fdJ = (p.J[0][i] - m.J[0][i]) / (2 * h);
        const double fdK = (p.K[0][i] - m.K[0][i]) / (2 * h);
        worst = std::max(worst, std::abs(fdJ - dv.J[0][o + i]));
        worst = std::max(worst, std::abs(fdK - dv.K[0][o + i]));
        sc = std::max(sc, std::max(std::abs(fdJ), std::abs(fdK)));
      }
      ASSERT_GT(sc, 1e-3) << "shell " << s << " comp " << e << " derivative is zero";
      // second-order central difference: ~h^2 relative accuracy floor
      EXPECT_LT(worst, 2e-6 * (sc + 1)) << "shell " << s << " comp " << e;
    }
}

TEST(JK, DerivativeMatricesSumToZeroOverShells) {
  // Translational invariance: moving every centre together cannot change an
  // integral. The builder gets the fourth slot from exactly this identity, so
  // what this really checks is that the other three slots are accumulated into
  // the right targets.
  auto sh = jk_shells();
  auto bas = intti::make_basis(sh);
  auto grid = intti::make_tgrid(intti::coulomb());
  const int n = bas.nao, ns = static_cast<int>(sh.size());
  const auto D = general_density(n);
  const std::vector<intti::JKRequest<double>> reqs = {
      {D.data(), intti::DensitySymmetry::General, intti::FockTerms::CoulombExchange}};
  const auto dv = intti::jk_deriv_build(bas, reqs, grid);
  const std::size_t n2 = static_cast<std::size_t>(n) * n;
  double scale = 0;
  for (std::size_t i = 0; i < dv.J[0].size(); ++i) scale = std::max(scale, std::abs(dv.J[0][i]));
  ASSERT_GT(scale, 1e-3);
  for (int e = 0; e < 3; ++e)
    for (std::size_t i = 0; i < n2; ++i) {
      double sj = 0, sk = 0;
      for (int s = 0; s < ns; ++s) {
        const std::size_t o = (static_cast<std::size_t>(3 * s + e)) * n2;
        sj += dv.J[0][o + i];
        sk += dv.K[0][o + i];
      }
      ASSERT_LT(std::abs(sj), 1e-11 * scale) << "sum of dJ over shells, comp " << e;
      ASSERT_LT(std::abs(sk), 1e-11 * scale) << "sum of dK over shells, comp " << e;
    }
}

TEST(JK, RiCoulombDerivativeMatricesMatchFiniteDifference) {
  // The RI derivative MATRIX needs gamma_x, which the RI gradient never does:
  // gamma is the stationary point of the fitting functional, so by the 2n+1 rule
  // the first-order ENERGY is complete without it, but the matrix is not a
  // stationary quantity. Differencing ri_jk's J at fixed density therefore tests
  // exactly the term that is new -- and displacing AUXILIARY shells tests it in
  // the place the plain gradient can hide it.
  auto osh = jk_shells();
  auto ash = jk_aux_shells();
  auto grid = intti::make_tgrid(intti::coulomb());
  auto orb = intti::make_basis(osh);
  auto aux = intti::make_basis(ash);
  const int n = orb.nao;
  const int nso = static_cast<int>(osh.size()), nsa = static_cast<int>(ash.size());
  const auto D = general_density(n);
  const std::vector<intti::JKRequest<double>> reqs = {
      {D.data(), intti::DensitySymmetry::General, intti::FockTerms::Coulomb}};
  const auto dv = intti::ri_j_deriv_build(orb, aux, reqs, grid);
  ASSERT_EQ(dv.nshell, nso + nsa);
  const std::size_t n2 = static_cast<std::size_t>(n) * n;
  const double h = 1e-4;
  // one orbital shell and one auxiliary shell, all three directions
  const std::vector<int> probe = {1, nso + 2};
  for (int s : probe)
    for (int e = 0; e < 3; ++e) {
      auto shifted = [&](double delta) {
        auto o2 = osh;
        auto a2 = ash;
        if (s < nso)
          o2[s].center[e] += delta;
        else
          a2[s - nso].center[e] += delta;
        auto ob = intti::make_basis(o2);
        auto ab = intti::make_basis(a2);
        const auto fit = intti::ri_fit(ob, ab, grid);
        std::vector<double> J(n2), K(n2);
        intti::ri_jk(fit, D.data(), J.data(), K.data());
        return J;
      };
      const auto Jp = shifted(h), Jm = shifted(-h);
      const std::size_t o = (static_cast<std::size_t>(3 * s + e)) * n2;
      double worst = 0, sc = 0;
      for (std::size_t i = 0; i < n2; ++i) {
        const double fd = (Jp[i] - Jm[i]) / (2 * h);
        worst = std::max(worst, std::abs(fd - dv.J[0][o + i]));
        sc = std::max(sc, std::abs(fd));
      }
      ASSERT_GT(sc, 1e-4) << "shell " << s << " comp " << e << " derivative is zero";
      EXPECT_LT(worst, 5e-6 * (sc + 1)) << "shell " << s << " comp " << e;
    }
}

TEST(JK, RiCoulombDerivativeMatricesContractToTheKnownGradient) {
  // E_J = 1/2 sum D J(D), so at fixed D the weighted trace of the derivative
  // matrices must reproduce ri_j_gradient, which is already validated. Note the
  // gradient itself is gamma_x-free, so agreement here also confirms the gamma_x
  // contribution cancels out of the trace exactly as the 2n+1 rule says.
  auto orb = intti::make_basis(jk_shells());
  auto aux = intti::make_basis(jk_aux_shells());
  auto grid = intti::make_tgrid(intti::coulomb());
  const int n = orb.nao;
  const auto D = sym_density(n);
  const std::vector<intti::JKRequest<double>> reqs = {
      {D.data(), intti::DensitySymmetry::Symmetric, intti::FockTerms::Coulomb}};
  const auto dv = intti::ri_j_deriv_build(orb, aux, reqs, grid);
  const auto ref = intti::ri_j_gradient(orb, aux, D.data(), grid);
  const int nso = static_cast<int>(orb.shells.size());
  const std::size_t n2 = static_cast<std::size_t>(n) * n;
  double scale = 0;
  for (const auto &g : ref.forb)
    for (int e = 0; e < 3; ++e) scale = std::max(scale, std::abs(g[e]));
  for (const auto &g : ref.faux)
    for (int e = 0; e < 3; ++e) scale = std::max(scale, std::abs(g[e]));
  ASSERT_GT(scale, 1e-4);
  for (int s = 0; s < dv.nshell; ++s)
    for (int e = 0; e < 3; ++e) {
      const std::size_t o = (static_cast<std::size_t>(3 * s + e)) * n2;
      double g = 0;
      for (std::size_t i = 0; i < n2; ++i) g += 0.5 * D[i] * dv.J[0][o + i];
      const double want = (s < nso) ? ref.forb[s][e] : ref.faux[s - nso][e];
      EXPECT_NEAR(g, want, 1e-9 * (scale + 1)) << "shell " << s << " comp " << e;
    }
}




TEST(JK, TiledOrbitalExchangeIsTileIndependentAndBounded) {
  // ri_k_occ_tiled never forms the nao^2 x naux fit vectors. The two tilings do
  // different jobs: the AUXILIARY tile bounds the three-centre block, while the
  // VECTOR tile bounds everything downstream -- the metric solve couples all
  // auxiliary functions so P cannot be blocked across it, but K is a plain sum
  // over k, so vector tiles accumulate. Every tiling must give the same K, and
  // the full-extent tiling must reproduce the dense ri_k_occ2: the dense case is
  // one block.
  auto orb = intti::make_basis(jk_shells());
  auto aux = intti::make_basis(jk_aux_shells());
  auto grid = intti::make_tgrid(intti::coulomb());
  const int n = orb.nao, nvec = 4;
  const int nsa = static_cast<int>(aux.shells.size());
  std::vector<double> CL(static_cast<std::size_t>(n) * nvec),
      CR(static_cast<std::size_t>(n) * nvec);
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < nvec; ++k) {
      CL[i * nvec + k] = 0.3 * std::cos(0.7 * i + k) / (1.0 + i);
      CR[i * nvec + k] = 0.2 * std::sin(0.4 * i - 2.0 * k) + 0.05 * k;
    }
  const std::size_t n2 = static_cast<std::size_t>(n) * n;
  // dense reference through the existing fit-vector route
  const auto fit = intti::ri_fit(orb, aux, grid);
  std::vector<double> Kref(n2);
  intti::ri_k_occ2(fit, CL.data(), CR.data(), nvec, Kref.data());
  ASSERT_GT(maxabs(Kref), 1e-4);
  // every combination of tilings, including the degenerate "one block" case
  for (int at : {0, 1, 2, nsa})
    for (int vt : {0, 1, 3, nvec}) {
      std::vector<double> K(n2);
      intti::ri_k_occ_tiled(orb, aux, grid, CL.data(), CR.data(), nvec, K.data(), at, vt);
      EXPECT_LT(maxdiff(K, Kref), 1e-11 * maxabs(Kref))
          << "aux_tile=" << at << " vec_tile=" << vt;
    }
  // and it must handle a non-symmetric density, which is the case it exists for
  std::vector<double> D(n2, 0.0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double sacc = 0;
      for (int k = 0; k < nvec; ++k) sacc += CL[i * nvec + k] * CR[j * nvec + k];
      D[i * n + j] = sacc;
    }
  double asym = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) asym = std::max(asym, std::abs(D[i * n + j] - D[j * n + i]));
  ASSERT_GT(asym, 1e-3) << "test density is symmetric, defeating the point";
  std::vector<double> Kd(n2);
  intti::ri_jk(fit, D.data(), static_cast<double *>(nullptr), Kd.data());
  std::vector<double> Kt(n2);
  intti::ri_k_occ_tiled(orb, aux, grid, CL.data(), CR.data(), nvec, Kt.data(), 2, 2);
  EXPECT_LT(maxdiff(Kt, Kd), 1e-11 * maxabs(Kd)) << "tiled orbital vs dense density-driven";
}


TEST(JK, RiCoulombDerivativeIsAuxiliaryTileIndependent) {
  // ri_j_deriv_build no longer materialises the nao^2 x naux three-centre
  // tensor; it consumes it in two auxiliary-tiled passes. The second pass had to
  // be hoisted so the tile loop sits OUTSIDE the perturbation loop -- tiling
  // inside it would have rebuilt the whole tensor nreq x npert times, trading a
  // memory problem for a worse time one. Every tiling must give the same answer,
  // and the full-extent tiling is the dense computation.
  auto orb = intti::make_basis(jk_shells());
  auto aux = intti::make_basis(jk_aux_shells());
  auto grid = intti::make_tgrid(intti::coulomb());
  const int n = orb.nao;
  const int nsa = static_cast<int>(aux.shells.size());
  const auto D = general_density(n);
  const std::vector<intti::JKRequest<double>> reqs = {
      {D.data(), intti::DensitySymmetry::General, intti::FockTerms::Coulomb}};
  const auto ref = intti::ri_j_deriv_build(orb, aux, reqs, grid, 1e-10, nsa);
  double scale = 0;
  for (double v : ref.J[0]) scale = std::max(scale, std::abs(v));
  ASSERT_GT(scale, 1e-3);
  for (int at : {1, 2, 3, nsa, 0}) {
    const auto got = intti::ri_j_deriv_build(orb, aux, reqs, grid, 1e-10, at);
    ASSERT_EQ(got.J[0].size(), ref.J[0].size());
    EXPECT_LT(maxdiff(got.J[0], ref.J[0]), 1e-11 * scale) << "aux_tile=" << at;
  }
}


TEST(JK, RiCoulombGradientAndHessianAreAuxiliaryTileIndependent) {
  // ri_j_gradient and ri_j_hessian each consumed the nao^2 x naux three-centre
  // tensor exactly once, as d = T^T D, so tiling it costs nothing and removes
  // the last dense allocation from the RI COULOMB path. Every tiling must agree,
  // full extent being the dense computation.
  auto orb = intti::make_basis(jk_shells());
  auto aux = intti::make_basis(jk_aux_shells());
  auto grid = intti::make_tgrid(intti::coulomb());
  const int n = orb.nao;
  const int nsa = static_cast<int>(aux.shells.size());
  const auto D = sym_density(n);
  const auto g0 = intti::ri_j_gradient(orb, aux, D.data(), grid, 1e-10, nsa);
  const auto h0 = intti::ri_j_hessian(orb, aux, D.data(), grid, 1e-10, nsa);
  double gs = 0, hs = 0;
  for (const auto &v : g0.forb)
    for (int e = 0; e < 3; ++e) gs = std::max(gs, std::abs(v[e]));
  for (double v : h0) hs = std::max(hs, std::abs(v));
  ASSERT_GT(gs, 1e-4);
  ASSERT_GT(hs, 1e-4);
  for (int at : {1, 2, 3, 0}) {
    const auto g = intti::ri_j_gradient(orb, aux, D.data(), grid, 1e-10, at);
    const auto h = intti::ri_j_hessian(orb, aux, D.data(), grid, 1e-10, at);
    for (std::size_t i = 0; i < g0.forb.size(); ++i)
      for (int e = 0; e < 3; ++e)
        EXPECT_NEAR(g.forb[i][e], g0.forb[i][e], 1e-11 * gs) << "grad orb aux_tile=" << at;
    for (std::size_t i = 0; i < g0.faux.size(); ++i)
      for (int e = 0; e < 3; ++e)
        EXPECT_NEAR(g.faux[i][e], g0.faux[i][e], 1e-11 * gs) << "grad aux aux_tile=" << at;
    EXPECT_LT(maxdiff(h, h0), 1e-11 * hs) << "hessian aux_tile=" << at;
  }
}

} // namespace
