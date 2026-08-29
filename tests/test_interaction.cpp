// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/product.hpp"

namespace {

using Shell = intti::PrimitiveShell<double>;
using GTO = intti::GTOProduct<double>;
using PF = intti::ProductFunction<double>;

// two diatomic segments on the z axis
const double kA[3] = {0.0, 0.0, -0.7};
const double kB[3] = {0.0, 0.0, 0.7};
const double kC[3] = {0.0, 0.0, 2.5};
const double kD[3] = {0.0, 0.0, 3.5};

intti::ShellPair<double> pair1(int la = 0, int lb = 0) {
  return intti::make_pair(Shell{0.9, {kA[0], kA[1], kA[2]}, la},
                          Shell{1.3, {kB[0], kB[1], kB[2]}, lb});
}
intti::ShellPair<double> pair2(int lc = 0, int ld = 0) {
  return intti::make_pair(Shell{1.1, {kC[0], kC[1], kC[2]}, lc},
                          Shell{0.6, {kD[0], kD[1], kD[2]}, ld});
}

// reference: analytic-path interaction on the (exact) Mobius grid
double reference(const GTO &a, const GTO &b) {
  auto grid = intti::make_tgrid(intti::coulomb());
  return intti::interaction<double>(PF{a}, PF{b}, grid);
}

intti::TGridSpec<double> linlog20(int n_lin = 30, int n_log = 45) {
  intti::TGridSpec<double> s;
  s.mapping = intti::TMapping::LinLog;
  s.t_lin = 2.0;
  s.n_lin = n_lin;
  s.n_log = n_log;
  s.t_c = 20.0;
  return s;
}

TEST(Interaction, GTOxGTOMatchesQuartet) {
  const auto bra = pair1(1, 0), ket = pair2(1, 0);
  auto grid = intti::make_tgrid(intti::coulomb());
  double buf[9];
  intti::eri_quartet(bra, ket, grid, buf);
  for (int ka = 0; ka < 3; ++ka)
    for (int kc = 0; kc < 3; ++kc) {
      const double v = intti::interaction<double>(PF{GTO{bra, ka, 0}},
                                                  PF{GTO{ket, kc, 0}}, grid);
      EXPECT_DOUBLE_EQ(v, buf[ka * 3 + kc]) << ka << " " << kc;
    }
}

TEST(Interaction, CoaxialPSCxPSC_ss) {
  const auto bra = pair1(), ket = pair2();
  const double exact = reference(GTO{bra, 0, 0}, GTO{ket, 0, 0});
  auto g1 = intti::make_psc_grid(bra.A, bra.B, intti::psc_xi_max(0.9, 1.4), 32, 32);
  auto g2 = intti::make_psc_grid(ket.A, ket.B, intti::psc_xi_max(0.6, 1.0), 32, 32);
  auto f1 = intti::psc_product(bra, 0, 0, g1);
  auto f2 = intti::psc_product(ket, 0, 0, g2);
  auto grid = intti::make_tgrid(intti::coulomb(), linlog20());
  const double v = intti::interaction<double>(PF{f1}, PF{f2}, grid);
  EXPECT_NEAR(v, exact, 5e-6 * std::abs(exact));
}

TEST(Interaction, CoaxialPSCxPSC_pxChannels) {
  // px-containing products exercise the |m| = 1 Bessel channel
  const auto bra = pair1(1, 0), ket = pair2(1, 0);
  const double exact = reference(GTO{bra, 0, 0}, GTO{ket, 0, 0}); // px comps
  auto g1 = intti::make_psc_grid(bra.A, bra.B, intti::psc_xi_max(0.9, 1.4), 32, 32);
  auto g2 = intti::make_psc_grid(ket.A, ket.B, intti::psc_xi_max(0.6, 1.0), 32, 32);
  auto f1 = intti::psc_product(bra, 0, 0, g1);
  auto f2 = intti::psc_product(ket, 0, 0, g2);
  EXPECT_EQ(f1.mmax, 1);
  auto grid = intti::make_tgrid(intti::coulomb(), linlog20());
  const double v = intti::interaction<double>(PF{f1}, PF{f2}, grid);
  // the |m| = 1 channel converges slightly slower than ss on equal grids
  EXPECT_NEAR(v, exact, 5e-5 * std::abs(exact));
}

TEST(Interaction, TailCorrectionMatters) {
  // Overlapping products. t_c must stay within what the spatial grids
  // genuinely resolve: beyond that, the discrete double sum degenerates to
  // its diagonal and spuriously captures the delta contribution (see
  // docs/psc.md), which would double-count with the correction. At t_c = 6
  // the grids resolve the kernel, the raw truncation error is ~pi S/t_c^2,
  // and the tail correction repairs it.
  const auto bra = pair1(), ket = intti::make_pair(Shell{1.1, {0.0, 0.0, -0.5}, 0},
                                                   Shell{0.6, {0.0, 0.0, 0.9}, 0});
  const double exact = reference(GTO{bra, 0, 0}, GTO{ket, 0, 0});
  auto g1 = intti::make_psc_grid(bra.A, bra.B, intti::psc_xi_max(0.9, 1.4), 32, 32);
  auto g2 = intti::make_psc_grid(ket.A, ket.B, intti::psc_xi_max(0.6, 1.4), 32, 32);
  auto f1 = intti::psc_product(bra, 0, 0, g1);
  auto f2 = intti::psc_product(ket, 0, 0, g2);
  auto spec = linlog20();
  spec.t_c = 6.0;
  auto grid = intti::make_tgrid(intti::coulomb(), spec);
  const double with_tail = intti::interaction<double>(PF{f1}, PF{f2}, grid);
  auto raw = grid;
  raw.tail_coeff = 0.0;
  const double without = intti::interaction<double>(PF{f1}, PF{f2}, raw);
  EXPECT_LT(std::abs(with_tail - exact), 0.05 * std::abs(without - exact));
  EXPECT_NEAR(with_tail, exact, 2e-3 * std::abs(exact));
}

TEST(Interaction, NonCoaxialClouds) {
  // rotate the second segment by 60 degrees about y around its center
  const double ang = M_PI / 3.0, cz = 3.0, hR = 0.5;
  const double Cr[3] = {-std::sin(ang) * hR, 0.0, cz - std::cos(ang) * hR};
  const double Dr[3] = {std::sin(ang) * hR, 0.0, cz + std::cos(ang) * hR};
  const auto bra = pair1();
  const auto ket = intti::make_pair(Shell{1.1, {Cr[0], Cr[1], Cr[2]}, 0},
                                    Shell{0.6, {Dr[0], Dr[1], Dr[2]}, 0});
  const double exact = reference(GTO{bra, 0, 0}, GTO{ket, 0, 0});
  auto g1 = intti::make_psc_grid(bra.A, bra.B, intti::psc_xi_max(0.9, 1.4), 16, 16);
  auto g2 = intti::make_psc_grid(ket.A, ket.B, intti::psc_xi_max(0.6, 1.0), 16, 16);
  auto f1 = intti::psc_product(bra, 0, 0, g1);
  auto f2 = intti::psc_product(ket, 0, 0, g2);
  auto grid = intti::make_tgrid(intti::coulomb(), linlog20());
  const double v = intti::interaction<double>(PF{f1}, PF{f2}, grid);
  EXPECT_NEAR(v, exact, 5e-3 * std::abs(exact));
}

TEST(Interaction, MixedGTOxPSC) {
  const auto bra = pair1(1, 0), ket = pair2();
  const double exact = reference(GTO{bra, 2, 0}, GTO{ket, 0, 0}); // pz x ss
  auto g2 = intti::make_psc_grid(ket.A, ket.B, intti::psc_xi_max(0.6, 1.0), 32, 32);
  auto f2 = intti::psc_product(ket, 0, 0, g2);
  auto grid = intti::make_tgrid(intti::coulomb(), linlog20());
  const double v =
      intti::interaction<double>(PF{GTO{bra, 2, 0}}, PF{f2}, grid);
  EXPECT_NEAR(v, exact, 1e-5 * std::abs(exact));
}

TEST(Interaction, ResolutionTcHeuristic) {
  // the automatic t_c must land in the safe regime for the overlapping
  // configuration of TailCorrectionMatters without hand tuning
  const auto bra = pair1(), ket = intti::make_pair(Shell{1.1, {0.0, 0.0, -0.5}, 0},
                                                   Shell{0.6, {0.0, 0.0, 0.9}, 0});
  const double exact = reference(GTO{bra, 0, 0}, GTO{ket, 0, 0});
  auto g1 = intti::make_psc_grid(bra.A, bra.B, intti::psc_xi_max(0.9, 1.4), 32, 32);
  auto g2 = intti::make_psc_grid(ket.A, ket.B, intti::psc_xi_max(0.6, 1.4), 32, 32);
  auto f1 = intti::psc_product(bra, 0, 0, g1);
  auto f2 = intti::psc_product(ket, 0, 0, g2);
  const double tc = std::min(intti::resolution_tc(g1), intti::resolution_tc(g2));
  EXPECT_GT(tc, 2.0);
  EXPECT_LT(tc, 20.0); // must stay below the demonstrated failure point
  auto grid = intti::make_tgrid(intti::coulomb(), intti::linlog_for(tc));
  const double v = intti::interaction<double>(PF{f1}, PF{f2}, grid);
  EXPECT_NEAR(v, exact, 5e-3 * std::abs(exact));
}

TEST(Interaction, MobiusWithGridProductThrows) {
  const auto ket = pair2();
  auto g2 = intti::make_psc_grid(ket.A, ket.B, intti::psc_xi_max(0.6, 1.0), 16, 16);
  auto f2 = intti::psc_product(ket, 0, 0, g2);
  auto mobius = intti::make_tgrid(intti::coulomb());
  EXPECT_THROW(intti::interaction<double>(PF{f2}, PF{f2}, mobius),
               std::invalid_argument);
}

} // namespace
