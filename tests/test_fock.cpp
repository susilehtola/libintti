// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "intti/fock.hpp"
#include "intti/quartet.hpp"

namespace {

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

// dense references from explicit quartets on the same grid
void dense_jk(const intti::ShellBasis<double> &b, const std::vector<double> &D,
              const intti::TGrid<double> &grid, std::vector<double> &J,
              std::vector<double> &K) {
  const int ns = static_cast<int>(b.shells.size()), nao = b.nao;
  J.assign(static_cast<std::size_t>(nao) * nao, 0.0);
  K.assign(static_cast<std::size_t>(nao) * nao, 0.0);
  for (int i = 0; i < ns; ++i)
    for (int j = 0; j < ns; ++j)
      for (int k = 0; k < ns; ++k)
        for (int l = 0; l < ns; ++l) {
          const auto bra = intti::make_pair(b.shells[i], b.shells[j]);
          const auto ket = intti::make_pair(b.shells[k], b.shells[l]);
          const int na = intti::ncart(b.shells[i].l), nb = intti::ncart(b.shells[j].l);
          const int nc = intti::ncart(b.shells[k].l), nd = intti::ncart(b.shells[l].l);
          std::vector<double> block(na * nb * nc * nd);
          intti::eri_quartet(bra, ket, grid, block.data());
          for (int ka = 0; ka < na; ++ka)
            for (int kb = 0; kb < nb; ++kb)
              for (int kc = 0; kc < nc; ++kc)
                for (int kd = 0; kd < nd; ++kd) {
                  const double v =
                      block[((ka * nb + kb) * nc + kc) * nd + kd];
                  const int A = b.ao_off[i] + ka, B = b.ao_off[j] + kb;
                  const int C = b.ao_off[k] + kc, E = b.ao_off[l] + kd;
                  // J_AB += D_CE (AB|CE); for K relabel (ac|bd) = (AB|CE)
                  // with a=A, c=B, b=C, d=E: K_AC += D_BE (AB|CE)
                  J[A * b.nao + B] += D[C * b.nao + E] * v;
                  K[A * b.nao + C] += D[B * b.nao + E] * v;
                }
        }
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

TEST(Fock, CoulombMatchesDense) {
  auto b = test_basis();
  auto D = random_symmetric(b.nao, 5);
  for (bool linlog : {false, true}) {
    intti::TGridSpec<double> spec;
    if (linlog) spec.mapping = intti::TMapping::LinLog;
    auto grid = intti::make_tgrid(intti::coulomb(), spec);
    std::vector<double> J(static_cast<std::size_t>(b.nao) * b.nao);
    intti::coulomb_build(b, D.data(), grid, J.data());
    std::vector<double> Jref, Kref;
    dense_jk(b, D, grid, Jref, Kref);
    EXPECT_LT(max_abs_diff(J, Jref), 1e-12 * max_abs(Jref)) << "linlog=" << linlog;
  }
}

TEST(Fock, ExchangeMatchesDense) {
  auto b = test_basis();
  auto D = random_symmetric(b.nao, 7);
  for (bool linlog : {false, true}) {
    intti::TGridSpec<double> spec;
    if (linlog) spec.mapping = intti::TMapping::LinLog;
    auto grid = intti::make_tgrid(intti::coulomb(), spec);
    std::vector<double> K(static_cast<std::size_t>(b.nao) * b.nao);
    intti::exchange_build(b, D.data(), grid, K.data(), 0.0);
    std::vector<double> Jref, Kref;
    dense_jk(b, D, grid, Jref, Kref);
    EXPECT_LT(max_abs_diff(K, Kref), 1e-12 * max_abs(Kref)) << "linlog=" << linlog;
  }
}

TEST(Fock, ScreeningIsControlled) {
  auto b = test_basis();
  auto D = random_symmetric(b.nao, 11);
  auto grid = intti::make_tgrid(intti::coulomb());
  const std::size_t n2 = static_cast<std::size_t>(b.nao) * b.nao;
  std::vector<double> J0(n2), J1(n2), K0(n2), K1(n2);
  intti::coulomb_build(b, D.data(), grid, J0.data());
  intti::coulomb_build(b, D.data(), grid, J1.data(), 1e-10);
  intti::exchange_build(b, D.data(), grid, K0.data(), 0.0);
  intti::exchange_build(b, D.data(), grid, K1.data(), 1e-10);
  EXPECT_LT(max_abs_diff(J0, J1), 1e-8 * max_abs(J0));
  EXPECT_LT(max_abs_diff(K0, K1), 1e-8 * max_abs(K0));
}

TEST(Fock, SymmetricDensityGivesSymmetricJK) {
  auto b = test_basis();
  auto D = random_symmetric(b.nao, 13);
  auto grid = intti::make_tgrid(intti::coulomb());
  const std::size_t n2 = static_cast<std::size_t>(b.nao) * b.nao;
  std::vector<double> J(n2), K(n2);
  intti::coulomb_build(b, D.data(), grid, J.data());
  intti::exchange_build(b, D.data(), grid, K.data(), 0.0);
  double asymJ = 0, asymK = 0;
  for (int i = 0; i < b.nao; ++i)
    for (int j = 0; j < b.nao; ++j) {
      asymJ = std::max(asymJ, std::abs(J[i * b.nao + j] - J[j * b.nao + i]));
      asymK = std::max(asymK, std::abs(K[i * b.nao + j] - K[j * b.nao + i]));
    }
  EXPECT_LT(asymJ, 1e-12 * max_abs(J));
  EXPECT_LT(asymK, 1e-12 * max_abs(K));
}

// The opt-in Symmetry::Full exchange build (atomic 8-fold scatter) must match
// both the exact dense K and the default deterministic build (to rounding).
TEST(Fock, ExchangeSym8MatchesDenseAndDeterministic) {
  auto b = test_basis();
  auto D = random_symmetric(b.nao, 19);
  const std::size_t n2 = static_cast<std::size_t>(b.nao) * b.nao;
  for (bool linlog : {false, true}) {
    intti::TGridSpec<double> spec;
    if (linlog) spec.mapping = intti::TMapping::LinLog;
    auto grid = intti::make_tgrid(intti::coulomb(), spec);
    std::vector<double> Kdet(n2), Ksym(n2), Jref, Kref;
    intti::exchange_build(b, D.data(), grid, Kdet.data(), 0.0);
    intti::exchange_build(b, D.data(), grid, Ksym.data(), 0.0, 0, 1,
                          intti::Symmetry::Full);
    dense_jk(b, D, grid, Jref, Kref);
    EXPECT_LT(max_abs_diff(Ksym, Kref), 1e-11 * max_abs(Kref)) << "linlog=" << linlog;
    EXPECT_LT(max_abs_diff(Ksym, Kdet), 1e-11 * max_abs(Kdet)) << "linlog=" << linlog;
  }
}

// Sym8 must remain correct with screening on (compared to exact dense, within
// the screening tolerance).
TEST(Fock, ExchangeSym8Screened) {
  auto b = test_basis();
  auto D = random_symmetric(b.nao, 23);
  auto grid = intti::make_tgrid(intti::coulomb());
  const std::size_t n2 = static_cast<std::size_t>(b.nao) * b.nao;
  std::vector<double> Ksym(n2), Jref, Kref;
  intti::exchange_build(b, D.data(), grid, Ksym.data(), 1e-10, 0, 1, intti::Symmetry::Full);
  dense_jk(b, D, grid, Jref, Kref);
  EXPECT_LT(max_abs_diff(Ksym, Kref), 1e-7 * max_abs(Kref));
}

// Memory tiling is lossless: the tiled exchange build gives a bit-identical
// result regardless of tile size (each element is computed independently), and
// matches the untiled build to rounding.
TEST(Fock, ExchangeTilingIsLossless) {
  auto b = test_basis();
  auto D = random_symmetric(b.nao, 29);
  auto grid = intti::make_tgrid(intti::coulomb());
  const std::size_t n2 = static_cast<std::size_t>(b.nao) * b.nao;
  const int ns = static_cast<int>(b.shells.size());
  std::vector<double> Kfull(n2), K1(n2), Kt(n2);
  intti::exchange_build(b, D.data(), grid, Kfull.data(), 0.0);       // untiled
  intti::exchange_build_tiled(b, D.data(), grid, K1.data(), ns, 0.0); // 1 tile
  // bit-identical across every tile granularity
  for (int ts : {1, 2, 3, 5}) {
    intti::exchange_build_tiled(b, D.data(), grid, Kt.data(), ts, 0.0);
    EXPECT_EQ(0.0, max_abs_diff(Kt, K1)) << "tile size " << ts << " not bit-identical";
  }
  // and equal to the untiled build to rounding (untiled mirrors the lower
  // triangle, so this is a different FP path -> ~rounding, not bit-exact)
  EXPECT_LT(max_abs_diff(K1, Kfull), 1e-11 * max_abs(Kfull));
}

// Coulomb tiling is lossless AND bit-exact vs the untiled build: each output
// pair's j^p is the full ket sum regardless of tiling, so no summation is
// regrouped (unlike RI-J).
TEST(Fock, CoulombTilingIsBitExact) {
  auto b = test_basis();
  auto D = random_symmetric(b.nao, 31);
  auto grid = intti::make_tgrid(intti::coulomb());
  const std::size_t n2 = static_cast<std::size_t>(b.nao) * b.nao;
  std::vector<double> Jfull(n2), Jt(n2);
  intti::coulomb_build(b, D.data(), grid, Jfull.data());
  for (int pt : {1, 2, 3, 7, 1000}) {
    intti::coulomb_build_tiled(b, D.data(), grid, Jt.data(), pt);
    EXPECT_EQ(0.0, max_abs_diff(Jt, Jfull)) << "pair_tile " << pt << " not bit-exact";
  }
}

} // namespace
