// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "intti/cdjk.hpp"
#include "intti/quartet.hpp"

#ifdef INTTI_HAVE_QUADMATH
#include <quadmath.h>
#endif

namespace {

// Exact J/K from four-index ERIs: J_mu nu = sum (mu nu|la si) D_la si,
// K_mu la = sum (mu nu|la si) D_nu si. Templated -> any precision (serial
// eri_quartet path for __float128).
template <class Real>
void exact_jk(const intti::ShellBasis<Real> &b, const std::vector<Real> &D,
              const intti::TGrid<intti::real_t<Real>> &grid, std::vector<Real> &J,
              std::vector<Real> &K) {
  const int nao = b.nao;
  J.assign(static_cast<std::size_t>(nao) * nao, Real(0));
  K.assign(static_cast<std::size_t>(nao) * nao, Real(0));
  const int ns = static_cast<int>(b.shells.size());
  for (int si = 0; si < ns; ++si)
    for (int sj = 0; sj < ns; ++sj)
      for (int sk = 0; sk < ns; ++sk)
        for (int sl = 0; sl < ns; ++sl) {
          const int na = intti::ncart(b.shells[si].l), nb = intti::ncart(b.shells[sj].l);
          const int nc = intti::ncart(b.shells[sk].l), nd = intti::ncart(b.shells[sl].l);
          std::vector<Real> blk(static_cast<std::size_t>(na) * nb * nc * nd);
          intti::eri_quartet(intti::make_pair(b.shells[si], b.shells[sj]),
                             intti::make_pair(b.shells[sk], b.shells[sl]), grid, blk.data());
          for (int a = 0; a < na; ++a)
            for (int bb = 0; bb < nb; ++bb)
              for (int c = 0; c < nc; ++c)
                for (int d = 0; d < nd; ++d) {
                  const Real v = blk[((a * nb + bb) * nc + c) * nd + d];
                  const int mu = b.ao_off[si] + a, nu = b.ao_off[sj] + bb;
                  const int la = b.ao_off[sk] + c, sg = b.ao_off[sl] + d;
                  J[static_cast<std::size_t>(mu) * nao + nu] += v * D[static_cast<std::size_t>(la) * nao + sg];
                  K[static_cast<std::size_t>(mu) * nao + la] += v * D[static_cast<std::size_t>(nu) * nao + sg];
                }
        }
}

using Shell = intti::PrimitiveShell<double>;

intti::ShellBasis<double> test_basis() {
  return intti::make_basis<double>({
      {1.2, {0.0, 0.0, 0.0}, 0},
      {0.3, {0.0, 0.0, 0.0}, 0},
      {0.8, {0.0, 0.0, 0.0}, 1},
      {1.5, {0.0, 0.0, 1.4}, 0},
      {0.5, {0.0, 0.0, 1.4}, 1},
      {0.9, {0.0, 0.0, 1.4}, 2},
  });
}

std::vector<double> random_symmetric(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> D(static_cast<std::size_t>(n) * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j <= i; ++j)
      D[i * n + j] = D[j * n + i] = u(rng);
  return D;
}

double max_abs_diff(const std::vector<double> &a, const std::vector<double> &b) {
  double m = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

double max_abs(const std::vector<double> &a) {
  double m = 0;
  for (double v : a)
    m = std::max(m, std::abs(v));
  return m;
}

// The low-rank exchange path cholesky_jk_occ (D = occ_scale C C^T) must
// reproduce the generic cholesky_jk called with that same reconstructed D.
TEST(CDJK, LowRankOccMatchesGeneric) {
  auto b = test_basis();
  const int nao = b.nao, nocc = 3;
  const double occ_scale = 2.0;
  std::mt19937 rng(71);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> C(static_cast<std::size_t>(nao) * nocc);
  for (auto &x : C) x = u(rng);
  // reconstruct the equivalent dense density D = occ_scale C C^T
  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
  std::vector<double> D(n2, 0.0);
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j) {
      double s = 0;
      for (int k = 0; k < nocc; ++k) s += C[i * nocc + k] * C[j * nocc + k];
      D[static_cast<std::size_t>(i) * nao + j] = occ_scale * s;
    }
  auto grid = intti::make_tgrid(intti::coulomb());
  intti::CholeskyOptions<double> opt;
  opt.tau = 1e-8;
  auto cb = intti::two_step_cholesky(b, grid, opt);
  std::vector<double> Jg(n2), Kg(n2), Jo(n2), Ko(n2);
  intti::cholesky_jk(b, cb, D.data(), Jg.data(), Kg.data());
  intti::cholesky_jk_occ(b, cb, C.data(), nocc, occ_scale, Jo.data(), Ko.data());
  const double sc = max_abs(Kg) + max_abs(Jg) + 1.0;
  EXPECT_LT(max_abs_diff(Jg, Jo), 1e-11 * sc) << "low-rank J != generic J";
  EXPECT_LT(max_abs_diff(Kg, Ko), 1e-11 * sc) << "low-rank K != generic K";
  // null-output branches must also work
  intti::cholesky_jk_occ(b, cb, C.data(), nocc, occ_scale, static_cast<double *>(nullptr),
                         Ko.data());
  intti::cholesky_jk_occ(b, cb, C.data(), nocc, occ_scale, Jo.data(),
                         static_cast<double *>(nullptr));
  EXPECT_LT(max_abs_diff(Kg, Ko), 1e-11 * sc);
  EXPECT_LT(max_abs_diff(Jg, Jo), 1e-11 * sc);
}

TEST(CDJK, ThresholdControlledAgainstExact) {
  auto b = test_basis();
  auto D = random_symmetric(b.nao, 23);
  auto grid = intti::make_tgrid(intti::coulomb());
  const std::size_t n2 = static_cast<std::size_t>(b.nao) * b.nao;
  std::vector<double> Jx(n2), Kx(n2);
  intti::coulomb_build(b, D.data(), grid, Jx.data());
  intti::exchange_build(b, D.data(), grid, Kx.data(), 0.0);

  double prevJ = 1e100, prevK = 1e100;
  for (double tau : {1e-4, 1e-6, 1e-8}) {
    intti::CholeskyOptions<double> opt;
    opt.tau = tau;
    auto cb = intti::two_step_cholesky(b, grid, opt);
    std::vector<double> J(n2), K(n2);
    intti::cholesky_jk(b, cb, D.data(), J.data(), K.data());
    const double eJ = max_abs_diff(J, Jx), eK = max_abs_diff(K, Kx);
    // the CD residual bound |R_pq| <= tau propagates linearly into J/K
    EXPECT_LT(eJ, 50 * tau * max_abs(D) * b.nao) << "tau=" << tau;
    EXPECT_LT(eK, 50 * tau * max_abs(D) * b.nao) << "tau=" << tau;
    EXPECT_LE(eJ, prevJ * 1.5) << "tau=" << tau; // tightening tau must not hurt
    EXPECT_LE(eK, prevK * 1.5) << "tau=" << tau;
    prevJ = eJ;
    prevK = eK;
  }
}

TEST(CDJK, PrecisionGenericLongDouble) {
  // long double exercises cholesky_jk's LAPACK-free matmul path; the CD-J/K must
  // match the exact four-index J/K to the Cholesky threshold.
  using LD = long double;
  const std::vector<intti::PrimitiveShell<LD>> sh = {
      {1.2L, {0.0L, 0.0L, 0.0L}, 0}, {0.8L, {0.0L, 0.0L, 0.0L}, 1},
      {0.5L, {0.0L, 0.0L, 1.4L}, 0}, {0.9L, {0.0L, 0.0L, 1.4L}, 1}};
  auto b = intti::make_basis<LD>(sh);
  const int nao = b.nao;
  std::vector<LD> D(static_cast<std::size_t>(nao) * nao);
  { auto Dd = random_symmetric(nao, 41); for (std::size_t i = 0; i < D.size(); ++i) D[i] = Dd[i]; }
  auto grid = intti::make_tgrid(intti::coulomb<LD>());
  std::vector<LD> Jx, Kx;
  exact_jk(b, D, grid, Jx, Kx);
  intti::CholeskyOptions<LD> opt;
  opt.tau = 1e-12L;
  auto cb = intti::pivoted_cholesky(b, grid, opt);
  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
  std::vector<LD> J(n2), K(n2);
  intti::cholesky_jk(b, cb, D.data(), J.data(), K.data());
  LD eJ = 0, eK = 0;
  for (std::size_t i = 0; i < n2; ++i) {
    eJ = std::max(eJ, std::abs(J[i] - Jx[i]));
    eK = std::max(eK, std::abs(K[i] - Kx[i]));
  }
  EXPECT_LT(static_cast<double>(eJ), 1e-8) << "long double CD-J";
  EXPECT_LT(static_cast<double>(eK), 1e-8) << "long double CD-K";
}

#ifdef INTTI_HAVE_QUADMATH
TEST(CDJK, PrecisionGenericQuad) {
  // End-to-end __float128 CD -> J/K: serial pivoted Cholesky + precision-generic
  // cholesky_jk, matched to the exact quad J/K far below the double floor.
  using Q = __float128;
  auto q = [](const char *s) { return strtoflt128(s, nullptr); };
  const std::vector<intti::PrimitiveShell<Q>> sh = {
      {q("1.2"), {q("0"), q("0"), q("0")}, 0}, {q("0.8"), {q("0"), q("0"), q("0")}, 1},
      {q("0.5"), {q("0"), q("0"), q("1.4")}, 0}, {q("0.9"), {q("0"), q("0"), q("1.4")}, 1}};
  auto b = intti::make_basis<Q>(sh);
  const int nao = b.nao;
  std::vector<Q> D(static_cast<std::size_t>(nao) * nao);
  { auto Dd = random_symmetric(nao, 41); for (std::size_t i = 0; i < D.size(); ++i) D[i] = static_cast<Q>(Dd[i]); }
  auto grid = intti::make_tgrid(intti::coulomb<Q>());
  std::vector<Q> Jx, Kx;
  exact_jk(b, D, grid, Jx, Kx);
  intti::CholeskyOptions<Q> opt;
  opt.tau = q("1e-26");
  auto cb = intti::pivoted_cholesky(b, grid, opt); // serial
  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
  std::vector<Q> J(n2), K(n2);
  intti::cholesky_jk(b, cb, D.data(), J.data(), K.data());
  Q eJ = 0, eK = 0;
  for (std::size_t i = 0; i < n2; ++i) {
    eJ = std::max(eJ, static_cast<Q>(fabsq(J[i] - Jx[i])));
    eK = std::max(eK, static_cast<Q>(fabsq(K[i] - Kx[i])));
  }
  EXPECT_LT(static_cast<double>(eJ), 1e-20) << "quad CD-J below double floor";
  EXPECT_LT(static_cast<double>(eK), 1e-20) << "quad CD-K below double floor";
}
#endif

TEST(CDJK, NullOutputsAllowed) {
  auto b = test_basis();
  auto D = random_symmetric(b.nao, 29);
  auto grid = intti::make_tgrid(intti::coulomb());
  intti::CholeskyOptions<double> opt;
  opt.tau = 1e-6;
  auto cb = intti::two_step_cholesky(b, grid, opt);
  const std::size_t n2 = static_cast<std::size_t>(b.nao) * b.nao;
  std::vector<double> J(n2), K(n2), Jonly(n2), Konly(n2);
  intti::cholesky_jk(b, cb, D.data(), J.data(), K.data());
  intti::cholesky_jk(b, cb, D.data(), Jonly.data(), static_cast<double *>(nullptr));
  intti::cholesky_jk(b, cb, D.data(), static_cast<double *>(nullptr), Konly.data());
  EXPECT_LT(max_abs_diff(J, Jonly), 1e-15);
  EXPECT_LT(max_abs_diff(K, Konly), 1e-15);
}

} // namespace
