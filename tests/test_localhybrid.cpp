// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/fock.hpp"
#include "intti/localhybrid.hpp"
#include "intti/oneel.hpp"
#include "intti/tgrid.hpp"

namespace {

using Shell = intti::PrimitiveShell<double>;

// H2-like: two s primitives on two centres
intti::ShellBasis<double> basis() {
  return intti::make_basis<double>({{0.8, {0.0, 0.0, -0.7}, 0}, {1.1, {0.0, 0.0, 0.7}, 0}});
}

// Gauss-Legendre product grid over [-L,L]^3 centred at 0.
void gl_box(double L, int n, std::vector<std::array<double, 3>> &pts,
            std::vector<double> &w) {
  std::vector<double> x(n), gw(n);
  intti::gauss_legendre(n, -L, L, x.data(), gw.data());
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        pts.push_back({x[i], x[j], x[k]});
        w.push_back(gw[i] * gw[j] * gw[k]);
      }
}

TEST(LocalHybrid, IntegratesToExactExchange) {
  auto bas = basis();
  const int nao = bas.nao;
  auto grid = intti::make_tgrid(intti::coulomb());
  // occupied orbital C (nao x 1): equal-weight combination, normalized so
  // that C^T S C = 1
  std::vector<double> C(nao, 1.0);
  auto Sm = intti::overlap_matrix(bas);
  double norm = 0;
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j) norm += C[i] * Sm[i * nao + j] * C[j];
  for (auto &c : C) c /= std::sqrt(norm);
  // density D = C C^T
  std::vector<double> D(static_cast<std::size_t>(nao) * nao);
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j) D[i * nao + j] = C[i] * C[j];
  std::vector<double> K(static_cast<std::size_t>(nao) * nao);
  intti::exchange_build(bas, D.data(), grid, K.data(), 0.0);
  double Ex = 0;
  for (std::size_t i = 0; i < D.size(); ++i) Ex += D[i] * K[i];
  Ex *= -0.5;

  // grid-integrated local exchange density (C as nao x 1)
  std::vector<std::array<double, 3>> pts;
  std::vector<double> w;
  gl_box(7.0, 48, pts, w);
  auto le = intti::local_exchange(bas, C.data(), 1, pts, w, grid);
  EXPECT_NEAR(le.energy, Ex, 1e-3 * std::abs(Ex))
      << "grid integral " << le.energy << " vs exact " << Ex;
}

TEST(LocalHybrid, RangeSeparatedIntegratesToShortRangeExchange) {
  // the LRSH mechanism: an erf(omega r)/r t grid gives the short-range
  // exchange density, which integrates to -1/2 Tr(D K_SR)
  auto bas = basis();
  const int nao = bas.nao;
  const double omega = 0.7;
  auto sr = intti::make_tgrid(intti::erf_rs(omega));
  std::vector<double> C(nao, 1.0);
  auto Sm = intti::overlap_matrix(bas);
  double norm = 0;
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j) norm += C[i] * Sm[i * nao + j] * C[j];
  for (auto &c : C) c /= std::sqrt(norm);
  std::vector<double> D(static_cast<std::size_t>(nao) * nao);
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j) D[i * nao + j] = C[i] * C[j];
  std::vector<double> Ksr(static_cast<std::size_t>(nao) * nao);
  intti::exchange_build(bas, D.data(), sr, Ksr.data(), 0.0);
  double Ex = 0;
  for (std::size_t i = 0; i < D.size(); ++i) Ex += D[i] * Ksr[i];
  Ex *= -0.5;

  std::vector<std::array<double, 3>> pts;
  std::vector<double> w;
  gl_box(7.0, 48, pts, w);
  auto le = intti::local_exchange(bas, C.data(), 1, pts, w, sr);
  EXPECT_NEAR(le.energy, Ex, 2e-3 * std::abs(Ex));
  EXPECT_LT(std::abs(le.energy), std::abs(Ex) + 1.0); // sanity
}

// LRSH with a per-point omega that includes every t-node reduces exactly to the
// full-range local_exchange on the same base grid (validates the per-point
// truncation plumbing).
TEST(LocalHybrid, LrshFullOmegaMatchesGlobal) {
  auto bas = basis();
  const int nao = bas.nao;
  auto base = intti::make_tgrid(intti::coulomb());
  std::vector<double> C(nao, 1.0);
  auto Sm = intti::overlap_matrix(bas);
  double norm = 0;
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j) norm += C[i] * Sm[i * nao + j] * C[j];
  for (auto &c : C) c /= std::sqrt(norm);
  std::vector<std::array<double, 3>> pts;
  std::vector<double> w;
  gl_box(7.0, 20, pts, w);
  auto full = intti::local_exchange(bas, C.data(), 1, pts, w, base);
  double tmax = 0;
  for (double t : base.t) tmax = std::max(tmax, t);
  std::vector<double> om(pts.size(), tmax + 1.0); // every node kept
  auto lrsh = intti::local_exchange_lrsh(bas, C.data(), 1, pts, w, om, base);
  EXPECT_NEAR(lrsh.energy, full.energy, 1e-10 * std::abs(full.energy));
}

// The exchange magnitude grows monotonically as the per-point omega admits more
// of the t-range (short-range -> full): |E(omega small)| < |E(mid)| < |E(full)|.
TEST(LocalHybrid, LrshOmegaMonotone) {
  auto bas = basis();
  const int nao = bas.nao;
  auto base = intti::make_tgrid(intti::coulomb());
  std::vector<double> C(nao, 1.0);
  auto Sm = intti::overlap_matrix(bas);
  double norm = 0;
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j) norm += C[i] * Sm[i * nao + j] * C[j];
  for (auto &c : C) c /= std::sqrt(norm);
  std::vector<std::array<double, 3>> pts;
  std::vector<double> w;
  gl_box(7.0, 20, pts, w);
  double tmax = 0;
  for (double t : base.t) tmax = std::max(tmax, t);
  auto en = [&](double omega) {
    std::vector<double> o(pts.size(), omega);
    return intti::local_exchange_lrsh(bas, C.data(), 1, pts, w, o, base).energy;
  };
  const double sr = en(0.5), mid = en(1.5), full = en(tmax + 1.0);
  EXPECT_LT(std::abs(sr), std::abs(mid)) << "sr=" << sr << " mid=" << mid;
  EXPECT_LT(std::abs(mid), std::abs(full)) << "mid=" << mid << " full=" << full;
}

TEST(LocalHybrid, AOValuesMatchDefinition) {
  auto bas = basis();
  std::vector<std::array<double, 3>> pts{{0.1, -0.2, 0.3}};
  auto v = intti::ao_values(bas, pts);
  // phi_0 = exp(-0.8 |r-A|^2), A=(0,0,-0.7)
  double d2 = 0.1 * 0.1 + 0.2 * 0.2 + (0.3 + 0.7) * (0.3 + 0.7);
  EXPECT_NEAR(v[0], std::exp(-0.8 * d2), 1e-14);
}

} // namespace
