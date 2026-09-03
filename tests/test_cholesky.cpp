// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/cholesky.hpp"
#include "intti/quartet.hpp"

#ifdef INTTI_HAVE_QUADMATH
#include <quadmath.h>
#endif

namespace {

using Shell = intti::PrimitiveShell<double>;

// two-center mixed s/p/d system
std::vector<intti::ShellPair<double>> test_pairs() {
  const std::vector<Shell> shells = {
      {1.2, {0.0, 0.0, 0.0}, 0}, {0.3, {0.0, 0.0, 0.0}, 0},
      {0.8, {0.0, 0.0, 0.0}, 1}, {1.5, {0.0, 0.0, 1.4}, 0},
      {0.5, {0.0, 0.0, 1.4}, 1}, {0.9, {0.0, 0.0, 1.4}, 2},
  };
  std::vector<intti::ShellPair<double>> pairs;
  for (std::size_t i = 0; i < shells.size(); ++i)
    for (std::size_t j = i; j < shells.size(); ++j)
      pairs.push_back(intti::make_pair(shells[i], shells[j]));
  return pairs;
}

// dense ERI matrix (nprod x nprod) in the pair-product basis
std::vector<double> dense_eri(const std::vector<intti::ShellPair<double>> &pairs,
                              const std::vector<int> &off, int nprod,
                              const intti::TGrid<double> &grid) {
  std::vector<double> V(static_cast<std::size_t>(nprod) * nprod);
  for (std::size_t ib = 0; ib < pairs.size(); ++ib)
    for (std::size_t ik = 0; ik < pairs.size(); ++ik) {
      const int nci = intti::ncart(pairs[ib].la) * intti::ncart(pairs[ib].lb);
      const int nck = intti::ncart(pairs[ik].la) * intti::ncart(pairs[ik].lb);
      std::vector<double> block(nci * nck);
      intti::eri_quartet(pairs[ib], pairs[ik], grid, block.data());
      for (int ci = 0; ci < nci; ++ci)
        for (int ck = 0; ck < nck; ++ck)
          V[(off[ib] + ci) + static_cast<std::size_t>(nprod) * (off[ik] + ck)] =
              block[ci * nck + ck];
    }
  return V;
}

double max_reconstruction_error(const intti::CholeskyBasis<double> &basis,
                                const std::vector<double> &V, int nprod) {
  double maxerr = 0.0;
  for (int i = 0; i < nprod; ++i)
    for (int j = 0; j < nprod; ++j) {
      double lij = 0.0;
      for (int J = 0; J < basis.naux; ++J)
        lij += basis.L(i, J) * basis.L(j, J);
      maxerr = std::max(maxerr,
                        std::abs(lij - V[i + static_cast<std::size_t>(nprod) * j]));
    }
  return maxerr;
}

TEST(Cholesky, ReconstructionWithinThreshold) {
  auto pairs = test_pairs();
  auto tab = intti::make_pair_table(pairs);
  auto grid = intti::make_tgrid(intti::coulomb());
  intti::QuartetWorkspace<double> ws;

  int prev_naux = 0;
  for (double tau : {1e-4, 1e-6, 1e-8}) {
    intti::CholeskyOptions<double> opt;
    opt.tau = tau;
    auto basis = intti::two_step_cholesky(tab, grid, ws, opt);
    ASSERT_GT(basis.naux, 0);
    auto V = dense_eri(pairs, basis.prod_offset, basis.nprod, grid);
    const double err = max_reconstruction_error(basis, V, basis.nprod);
    // pivoted CD bounds the residual diagonal by tau; off-diagonals by
    // Cauchy-Schwarz on the residual
    EXPECT_LT(err, 2.0 * tau) << "tau=" << tau;
    EXPECT_GE(basis.naux, prev_naux); // naux grows as tau tightens
    prev_naux = basis.naux;
    // compression: naux well below nprod at loose thresholds
    if (tau == 1e-4) EXPECT_LT(basis.naux, basis.nprod);
  }
}

TEST(Cholesky, TwoStepMatchesOneStep) {
  auto pairs = test_pairs();
  auto tab = intti::make_pair_table(pairs);
  auto grid = intti::make_tgrid(intti::coulomb());
  intti::QuartetWorkspace<double> ws;
  intti::CholeskyOptions<double> opt;
  opt.tau = 1e-6;

  auto two = intti::two_step_cholesky(tab, grid, ws, opt);
  opt.one_step = true;
  auto one = intti::two_step_cholesky(tab, grid, ws, opt);
  ASSERT_EQ(two.naux, one.naux);
  // the vectors differ by an orthogonal transform; the reconstruction is
  // the invariant
  const int nprod = two.nprod;
  double maxdiff = 0.0;
  for (int i = 0; i < nprod; ++i)
    for (int j = 0; j <= i; ++j) {
      double vij2 = 0.0, vij1 = 0.0;
      for (int J = 0; J < two.naux; ++J) {
        vij2 += two.L(i, J) * two.L(j, J);
        vij1 += one.L(i, J) * one.L(j, J);
      }
      maxdiff = std::max(maxdiff, std::abs(vij2 - vij1));
    }
  EXPECT_LT(maxdiff, 1e-10);
}

TEST(Cholesky, PivotedLongDoublePrecisionGeneric) {
  // The one-step pivoted Cholesky is LAPACK-free, so it runs at long double
  // (unavailable to the two-step S^{-1/2} path). It must reconstruct the ERG
  // matrix to the threshold.
  using LD = long double;
  using ShellLD = intti::PrimitiveShell<LD>;
  const std::vector<ShellLD> shells = {
      {1.2L, {0.0L, 0.0L, 0.0L}, 0}, {0.3L, {0.0L, 0.0L, 0.0L}, 0},
      {0.8L, {0.0L, 0.0L, 0.0L}, 1}, {1.5L, {0.0L, 0.0L, 1.4L}, 0},
      {0.5L, {0.0L, 0.0L, 1.4L}, 1}, {0.9L, {0.0L, 0.0L, 1.4L}, 2},
  };
  std::vector<intti::ShellPair<LD>> pairs;
  for (std::size_t i = 0; i < shells.size(); ++i)
    for (std::size_t j = i; j < shells.size(); ++j)
      pairs.push_back(intti::make_pair(shells[i], shells[j]));
  auto tab = intti::make_pair_table(pairs);
  auto grid = intti::make_tgrid(intti::coulomb<LD>());
  intti::QuartetWorkspace<LD> ws;
  intti::CholeskyOptions<LD> opt;
  opt.tau = 1e-10L;
  auto basis = intti::pivoted_cholesky(tab, grid, ws, opt);
  ASSERT_GT(basis.naux, 0);
  const int nprod = basis.nprod;
  std::vector<LD> V(static_cast<std::size_t>(nprod) * nprod);
  for (std::size_t ib = 0; ib < pairs.size(); ++ib)
    for (std::size_t ik = 0; ik < pairs.size(); ++ik) {
      const int nci = intti::ncart(pairs[ib].la) * intti::ncart(pairs[ib].lb);
      const int nck = intti::ncart(pairs[ik].la) * intti::ncart(pairs[ik].lb);
      std::vector<LD> block(nci * nck);
      intti::eri_quartet(pairs[ib], pairs[ik], grid, block.data());
      for (int ci = 0; ci < nci; ++ci)
        for (int ck = 0; ck < nck; ++ck)
          V[(basis.prod_offset[ib] + ci) + static_cast<std::size_t>(nprod) * (basis.prod_offset[ik] + ck)] =
              block[ci * nck + ck];
    }
  LD maxerr = 0;
  for (int i = 0; i < nprod; ++i)
    for (int j = 0; j < nprod; ++j) {
      LD lij = 0;
      for (int J = 0; J < basis.naux; ++J) lij += basis.L(i, J) * basis.L(j, J);
      maxerr = std::max(maxerr, std::abs(lij - V[i + static_cast<std::size_t>(nprod) * j]));
    }
  EXPECT_LT(static_cast<double>(maxerr), 1e-9) << "long double pivoted CD reconstruction";
}

TEST(Cholesky, PivotedSerialMatchesBatch) {
  // The serial (eri_quartet) pivoted Cholesky must reconstruct the ERI to the
  // threshold and agree with the batched path at double precision.
  auto pairs = test_pairs();
  auto grid = intti::make_tgrid(intti::coulomb());
  intti::CholeskyOptions<double> opt;
  opt.tau = 1e-8;
  auto ser = intti::detail::pivoted_cholesky_serial(pairs, grid, opt);
  ASSERT_GT(ser.naux, 0);
  auto V = dense_eri(pairs, ser.prod_offset, ser.nprod, grid);
  EXPECT_LT(max_reconstruction_error(ser, V, ser.nprod), 2.0 * opt.tau);
  auto tab = intti::make_pair_table(pairs);
  intti::QuartetWorkspace<double> ws;
  auto bat = intti::pivoted_cholesky(tab, grid, ws, opt);
  ASSERT_EQ(ser.naux, bat.naux);
  double md = 0;
  for (int i = 0; i < ser.nprod; ++i)
    for (int j = 0; j <= i; ++j) {
      double s = 0, b = 0;
      for (int J = 0; J < ser.naux; ++J) {
        s += ser.L(i, J) * ser.L(j, J);
        b += bat.L(i, J) * bat.L(j, J);
      }
      md = std::max(md, std::abs(s - b));
    }
  EXPECT_LT(md, 1e-10) << "serial vs batched reconstruction";
}

#ifdef INTTI_HAVE_QUADMATH
TEST(Cholesky, PivotedQuadruplePrecision) {
  // The serial pivoted Cholesky runs at __float128 (the batch path cannot), and
  // reconstructs the ERI far below the double floor.
  using Q = __float128;
  auto q = [](const char *s) { return strtoflt128(s, nullptr); };
  const std::vector<intti::PrimitiveShell<Q>> shells = {
      {q("1.2"), {q("0.0"), q("0.0"), q("0.0")}, 0},
      {q("0.5"), {q("0.0"), q("0.0"), q("0.0")}, 1},
      {q("0.9"), {q("0.0"), q("0.0"), q("1.3")}, 0},
      {q("0.7"), {q("0.0"), q("0.0"), q("1.3")}, 1},
  };
  std::vector<intti::ShellPair<Q>> pairs;
  for (std::size_t i = 0; i < shells.size(); ++i)
    for (std::size_t j = i; j < shells.size(); ++j)
      pairs.push_back(intti::make_pair(shells[i], shells[j]));
  auto grid = intti::make_tgrid(intti::coulomb<Q>());
  intti::CholeskyOptions<Q> opt;
  opt.tau = q("1e-24");
  auto basis = intti::pivoted_cholesky(pairs, grid, opt); // dispatches to serial
  ASSERT_GT(basis.naux, 0);
  const int nprod = basis.nprod;
  std::vector<Q> V(static_cast<std::size_t>(nprod) * nprod);
  for (std::size_t ib = 0; ib < pairs.size(); ++ib)
    for (std::size_t ik = 0; ik < pairs.size(); ++ik) {
      const int nci = intti::ncart(pairs[ib].la) * intti::ncart(pairs[ib].lb);
      const int nck = intti::ncart(pairs[ik].la) * intti::ncart(pairs[ik].lb);
      std::vector<Q> block(nci * nck);
      intti::eri_quartet(pairs[ib], pairs[ik], grid, block.data());
      for (int ci = 0; ci < nci; ++ci)
        for (int ck = 0; ck < nck; ++ck)
          V[(basis.prod_offset[ib] + ci) + static_cast<std::size_t>(nprod) * (basis.prod_offset[ik] + ck)] =
              block[ci * nck + ck];
    }
  Q maxerr = 0;
  for (int i = 0; i < nprod; ++i)
    for (int j = 0; j < nprod; ++j) {
      Q lij = 0;
      for (int J = 0; J < basis.naux; ++J) lij += basis.L(i, J) * basis.L(j, J);
      const Q e = fabsq(lij - V[i + static_cast<std::size_t>(nprod) * j]);
      if (e > maxerr) maxerr = e;
    }
  EXPECT_LT(static_cast<double>(maxerr), 1e-20) << "quad pivoted CD below the double floor";
}
#endif

} // namespace
