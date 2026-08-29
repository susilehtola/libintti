// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "intti/fock.hpp"
#include "intti/ncenter.hpp"
#include "intti/ri.hpp"

namespace {

using Shell = intti::PrimitiveShell<double>;

intti::ShellBasis<double> orb_basis() {
  return intti::make_basis<double>({{1.2, {0.0, 0.0, 0.0}, 0},
                                    {0.35, {0.0, 0.0, 0.0}, 0},
                                    {0.8, {0.0, 0.0, 0.0}, 1},
                                    {0.5, {0.0, 0.0, 1.3}, 0},
                                    {0.6, {0.0, 0.0, 1.3}, 1}});
}
// a rich, well-separated (well-conditioned) auxiliary basis on both centres
intti::ShellBasis<double> aux_basis() {
  std::vector<Shell> s;
  for (double c : {0.0, 1.3}) {
    for (double a : {12.0, 3.0, 0.75, 0.2}) s.push_back({a, {0.0, 0.0, c}, 0});
    for (double a : {6.0, 1.5, 0.4}) s.push_back({a, {0.0, 0.0, c}, 1});
    for (double a : {2.0, 0.6}) s.push_back({a, {0.0, 0.0, c}, 2});
  }
  return intti::make_basis(s);
}

// a physical (positive semidefinite) density D = sum_k v_k v_k^T
std::vector<double> psd_density(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> nd;
  std::vector<double> D(static_cast<std::size_t>(n) * n, 0.0);
  for (int k = 0; k < 3; ++k) {
    std::vector<double> v(n);
    for (auto &x : v) x = nd(rng);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) D[i * n + j] += v[i] * v[j];
  }
  return D;
}

std::vector<double> rand_sym(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1, 1);
  std::vector<double> D(static_cast<std::size_t>(n) * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j <= i; ++j) D[i * n + j] = D[j * n + i] = u(rng);
  return D;
}

TEST(RI, FitVectorsReproduceMetricContraction) {
  // B B^T must equal T M^{-1} T^T  (built from the same validated 2c/3c)
  auto orb = orb_basis(), aux = aux_basis();
  auto grid = intti::make_tgrid(intti::coulomb());
  auto fit = intti::ri_fit(orb, aux, grid);
  auto M = intti::coulomb_2c(aux, grid);
  auto T = intti::coulomb_3c(orb, aux, grid);
  const int nao = orb.nao, naux = aux.nao;
  // Minv via eigen-decomposition
  std::vector<double> V = M, ev(naux);
  intti::detail::syevd(naux, V.data(), ev.data());
  double emax = 0;
  for (double e : ev) emax = std::max(emax, e);
  std::vector<double> Minv(static_cast<std::size_t>(naux) * naux, 0.0);
  for (int k = 0; k < naux; ++k) { // eigenvector k, component i is V[k*naux+i]
    if (ev[k] <= 1e-10 * emax) continue;
    for (int i = 0; i < naux; ++i)
      for (int j = 0; j < naux; ++j)
        Minv[i * naux + j] += V[k * naux + i] * (1.0 / ev[k]) * V[k * naux + j];
  }
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  double maxdiff = 0, scale = 0;
  for (std::size_t x = 0; x < N; ++x)
    for (std::size_t y = 0; y < N; ++y) {
      double bb = 0;
      for (int P = 0; P < naux; ++P) bb += fit.B[x * naux + P] * fit.B[y * naux + P];
      double ref = 0;
      for (int P = 0; P < naux; ++P)
        for (int Q = 0; Q < naux; ++Q)
          ref += T[x * naux + P] * Minv[P * naux + Q] * T[y * naux + Q];
      maxdiff = std::max(maxdiff, std::abs(bb - ref));
      scale = std::max(scale, std::abs(ref));
    }
  EXPECT_LT(maxdiff, 1e-10 * scale);
}

TEST(RI, JKAssemblyMatchesTensorContraction) {
  auto orb = orb_basis(), aux = aux_basis();
  auto grid = intti::make_tgrid(intti::coulomb());
  auto fit = intti::ri_fit(orb, aux, grid);
  const int nao = orb.nao, naux = fit.naux;
  auto D = rand_sym(nao, 7);
  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
  std::vector<double> J(n2), K(n2);
  intti::ri_jk(fit, D.data(), J.data(), K.data());
  // reference from RI 4-center V_{mu nu la si} = sum_P B_munu^P B_lasi^P
  auto B = [&](int a, int b, int P) {
    return fit.B[(static_cast<std::size_t>(a) * nao + b) * naux + P];
  };
  double eJ = 0, eK = 0, sc = 0;
  for (int mu = 0; mu < nao; ++mu)
    for (int nu = 0; nu < nao; ++nu) {
      double j = 0, k = 0;
      for (int la = 0; la < nao; ++la)
        for (int si = 0; si < nao; ++si) {
          double vJ = 0, vK = 0;
          for (int P = 0; P < naux; ++P) {
            vJ += B(mu, nu, P) * B(la, si, P);
            vK += B(mu, la, P) * B(nu, si, P);
          }
          j += vJ * D[la * nao + si];
          k += vK * D[la * nao + si];
        }
      eJ = std::max(eJ, std::abs(j - J[mu * nao + nu]));
      eK = std::max(eK, std::abs(k - K[mu * nao + nu]));
      sc = std::max(sc, std::abs(j));
    }
  EXPECT_LT(eJ, 1e-11 * sc);
  EXPECT_LT(eK, 1e-11 * sc);
}

TEST(RI, ApproximatesExactJK) {
  // with a rich aux, RI-J/K should be close to the exact builds
  auto orb = orb_basis(), aux = aux_basis();
  auto grid = intti::make_tgrid(intti::coulomb());
  auto fit = intti::ri_fit(orb, aux, grid);
  const int nao = orb.nao;
  auto D = psd_density(nao, 11);
  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
  std::vector<double> Jri(n2), Kri(n2), Jex(n2), Kex(n2);
  intti::ri_jk(fit, D.data(), Jri.data(), Kri.data());
  intti::coulomb_build(orb, D.data(), grid, Jex.data());
  intti::exchange_build(orb, D.data(), grid, Kex.data(), 0.0);
  double eJ = 0, eK = 0, sJ = 0, sK = 0;
  for (std::size_t i = 0; i < n2; ++i) {
    eJ = std::max(eJ, std::abs(Jri[i] - Jex[i]));
    eK = std::max(eK, std::abs(Kri[i] - Kex[i]));
    sJ = std::max(sJ, std::abs(Jex[i]));
    sK = std::max(sK, std::abs(Kex[i]));
  }
  // density-fitting accuracy on a physical density with a rich aux
  EXPECT_LT(eJ, 1e-3 * sJ) << "RI-J should approximate exact J";
  EXPECT_LT(eK, 2e-3 * sK) << "RI-K should approximate exact K";
}

TEST(RI, OccDrivenExchangeMatchesDensityDriven) {
  // ri_k_occ (orbital-driven) must equal ri_jk's K for D = sum_i C_i C_i^T
  auto orb = orb_basis(), aux = aux_basis();
  auto grid = intti::make_tgrid(intti::coulomb());
  auto fit = intti::ri_fit(orb, aux, grid);
  const int nao = orb.nao, nocc = 3;
  std::mt19937 rng(19);
  std::normal_distribution<double> nd;
  std::vector<double> C(static_cast<std::size_t>(nao) * nocc);
  for (auto &x : C) x = nd(rng);
  std::vector<double> D(static_cast<std::size_t>(nao) * nao, 0.0);
  for (int mu = 0; mu < nao; ++mu)
    for (int nu = 0; nu < nao; ++nu)
      for (int i = 0; i < nocc; ++i)
        D[mu * nao + nu] += C[mu * nocc + i] * C[nu * nocc + i];
  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
  std::vector<double> Kd(n2), Ko(n2);
  intti::ri_jk(fit, D.data(), static_cast<double *>(nullptr), Kd.data());
  intti::ri_k_occ(fit, C.data(), nocc, Ko.data());
  double md = 0, mx = 0;
  for (std::size_t i = 0; i < n2; ++i) {
    md = std::max(md, std::abs(Kd[i] - Ko[i]));
    mx = std::max(mx, std::abs(Kd[i]));
  }
  EXPECT_LT(md, 1e-12 * mx);
}

} // namespace
