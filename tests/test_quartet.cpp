// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/quartet.hpp"
#include "intti/reference/eri_analytic.hpp"

namespace {

using intti::PrimitiveShell;

const double kA[3] = {0.0, 0.1, -0.3};
const double kB[3] = {0.5, -0.2, 0.4};
const double kC[3] = {1.0, 0.8, 0.0};
const double kD[3] = {-0.4, 0.3, 1.1};

PrimitiveShell shell(double alpha, const double *center, int l) {
  return {alpha, {center[0], center[1], center[2]}, l};
}

// max |quad - ref| over components, relative to max |ref|
double max_rel_error(const PrimitiveShell &a, const PrimitiveShell &b,
                     const PrimitiveShell &c, const PrimitiveShell &d,
                     const intti::TGrid &grid) {
  const int nout = intti::ncart(a.l) * intti::ncart(b.l) * intti::ncart(c.l) *
                   intti::ncart(d.l);
  std::vector<double> quad(nout), ref(nout);
  intti::eri_quartet(intti::make_pair(a, b), intti::make_pair(c, d), grid, quad.data());
  intti::ref::eri_analytic(a, b, c, d, ref.data());
  double maxref = 0.0, maxdiff = 0.0;
  for (int k = 0; k < nout; ++k) {
    maxref = std::max(maxref, std::abs(ref[k]));
    maxdiff = std::max(maxdiff, std::abs(quad[k] - ref[k]));
  }
  return maxdiff / maxref;
}

TEST(Quartet, GoldenSSSS) {
  // (ss|ss) case verified against the analytic Boys expression
  auto grid = intti::make_tgrid(intti::Kernel::coulomb());
  double val;
  intti::eri_quartet(intti::make_pair(shell(0.8, kA, 0), shell(1.3, kB, 0)),
                     intti::make_pair(shell(2.1, kC, 0), shell(0.35, kD, 0)), grid,
                     &val);
  EXPECT_NEAR(val, 0.5625558912295, 2e-8);
}

TEST(Quartet, ShellsVsAnalytic) {
  auto grid = intti::make_tgrid(intti::Kernel::coulomb());
  const int cases[][4] = {{0, 0, 0, 0}, {1, 0, 0, 0}, {1, 1, 0, 0},
                          {1, 1, 1, 1}, {2, 1, 1, 0}, {2, 2, 2, 2}};
  for (const auto &lc : cases) {
    const double err = max_rel_error(shell(0.8, kA, lc[0]), shell(1.3, kB, lc[1]),
                                     shell(2.1, kC, lc[2]), shell(0.35, kD, lc[3]), grid);
    EXPECT_LT(err, 5e-7) << "shells " << lc[0] << lc[1] << lc[2] << lc[3];
  }
}

TEST(Quartet, ShellsVsAnalyticTightGrid) {
  intti::TGridSpec spec;
  spec.t_c = 200.0;
  spec.n_log = 120;
  auto grid = intti::make_tgrid(intti::Kernel::coulomb(), spec);
  const int cases[][4] = {{1, 1, 1, 1}, {2, 1, 1, 0}, {2, 2, 2, 2}};
  for (const auto &lc : cases) {
    const double err = max_rel_error(shell(0.8, kA, lc[0]), shell(1.3, kB, lc[1]),
                                     shell(2.1, kC, lc[2]), shell(0.35, kD, lc[3]), grid);
    EXPECT_LT(err, 5e-9) << "shells " << lc[0] << lc[1] << lc[2] << lc[3];
  }
}

TEST(Quartet, TailCorrectionConvergenceLaw) {
  // corrected error ~ 1/t_c^4, raw truncation error ~ 1/t_c^2
  const auto bra = intti::make_pair(shell(0.8, kA, 0), shell(1.3, kB, 0));
  const auto ket = intti::make_pair(shell(2.1, kC, 0), shell(0.35, kD, 0));
  double ref;
  {
    const PrimitiveShell a = shell(0.8, kA, 0), b = shell(1.3, kB, 0),
                         c = shell(2.1, kC, 0), d = shell(0.35, kD, 0);
    intti::ref::eri_analytic(a, b, c, d, &ref);
  }
  std::vector<double> err_corr, err_raw;
  for (double tc : {5.0, 10.0, 20.0, 40.0}) {
    intti::TGridSpec spec;
    spec.t_lin = 2.0;
    spec.n_lin = 40;
    spec.n_log = 60;
    spec.t_c = tc;
    auto grid = intti::make_tgrid(intti::Kernel::coulomb(), spec);
    double corr;
    intti::eri_quartet(bra, ket, grid, &corr);
    auto raw_grid = grid;
    raw_grid.tail_coeff = 0.0;
    double raw;
    intti::eri_quartet(bra, ket, raw_grid, &raw);
    err_corr.push_back(std::abs(corr - ref));
    err_raw.push_back(std::abs(raw - ref));
  }
  for (std::size_t i = 0; i + 1 < err_corr.size(); ++i) {
    const double ratio_corr = err_corr[i] / err_corr[i + 1];
    const double ratio_raw = err_raw[i] / err_raw[i + 1];
    EXPECT_GT(ratio_corr, 10.0) << "corrected error should decay ~1/t_c^4";
    EXPECT_LT(ratio_corr, 22.0);
    EXPECT_GT(ratio_raw, 3.0) << "raw error should decay ~1/t_c^2";
    EXPECT_LT(ratio_raw, 5.5);
  }
}

TEST(Quartet, ErfPlusErfcIsCoulomb) {
  const double omega = 0.6;
  const auto bra = intti::make_pair(shell(0.8, kA, 1), shell(1.3, kB, 1));
  const auto ket = intti::make_pair(shell(2.1, kC, 0), shell(0.35, kD, 0));
  const int nout = 9;
  std::vector<double> coul(nout), erf(nout), erfc(nout);
  intti::eri_quartet(bra, ket, intti::make_tgrid(intti::Kernel::coulomb()), coul.data());
  intti::eri_quartet(bra, ket, intti::make_tgrid(intti::Kernel::erf_rs(omega)), erf.data());
  intti::eri_quartet(bra, ket, intti::make_tgrid(intti::Kernel::erfc_rs(omega)),
                     erfc.data());
  double maxc = 0.0;
  for (int k = 0; k < nout; ++k)
    maxc = std::max(maxc, std::abs(coul[k]));
  for (int k = 0; k < nout; ++k)
    EXPECT_NEAR(erf[k] + erfc[k], coul[k], 1e-8 * maxc) << "component " << k;
}

TEST(Quartet, YukawaGridConvergence) {
  // formulation is validated pointwise in test_tgrid; here check the quartet
  // value is converged with respect to the grid resolution
  const auto bra = intti::make_pair(shell(0.8, kA, 1), shell(1.3, kB, 0));
  const auto ket = intti::make_pair(shell(2.1, kC, 1), shell(0.35, kD, 0));
  double coarse[9], dense[9];
  intti::eri_quartet(bra, ket, intti::make_tgrid(intti::Kernel::yukawa(1.2)), coarse);
  intti::TGridSpec spec;
  spec.n_lin = 100;
  spec.n_log = 160;
  intti::eri_quartet(bra, ket, intti::make_tgrid(intti::Kernel::yukawa(1.2), spec), dense);
  double maxd = 0.0;
  for (int k = 0; k < 9; ++k)
    maxd = std::max(maxd, std::abs(dense[k]));
  for (int k = 0; k < 9; ++k)
    EXPECT_NEAR(coarse[k], dense[k], 5e-9 * maxd) << "component " << k;
}

TEST(Quartet, CoincidentCenters) {
  // all centers equal: (ss|ss) = 2 pi^(5/2)/(pq sqrt(p+q))
  const double p = 0.8 + 1.3, q = 2.1 + 0.35;
  const double exact = 2.0 * std::pow(M_PI, 2.5) / (p * q * std::sqrt(p + q));
  auto grid = intti::make_tgrid(intti::Kernel::coulomb());
  double val;
  intti::eri_quartet(intti::make_pair(shell(0.8, kA, 0), shell(1.3, kA, 0)),
                     intti::make_pair(shell(2.1, kA, 0), shell(0.35, kA, 0)), grid, &val);
  // coincident centers maximize the four-orbital overlap, so the residual
  // O(1/t_c^4) tail error is at its largest here (~4e-8 at t_c = 60)
  EXPECT_NEAR(val, exact, 2e-7 * exact);
}

TEST(Quartet, ExtremeExponents) {
  // tight (1e4) x diffuse (1e-2) pairing
  auto def = intti::make_tgrid(intti::Kernel::coulomb());
  EXPECT_LT(max_rel_error(shell(1.0e4, kA, 1), shell(0.7, kB, 0), shell(1.0e-2, kC, 1),
                          shell(0.5, kD, 0), def),
            5e-8);
  intti::TGridSpec spec;
  spec.t_c = 2000.0;
  spec.n_log = 160;
  auto tight = intti::make_tgrid(intti::Kernel::coulomb(), spec);
  EXPECT_LT(max_rel_error(shell(1.0e4, kA, 1), shell(0.7, kB, 0), shell(1.0e-2, kC, 1),
                          shell(0.5, kD, 0), tight),
            1e-12);
}

} // namespace
