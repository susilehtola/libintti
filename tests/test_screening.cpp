// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
//
// MBIE-1 / QQR distance-including screening (screening.hpp): the estimate must
// (1) always be a valid upper bound on the true |(ab|cd)|, and (2) beat plain
// Schwarz once the bra/ket clouds separate (it decays ~1/R where Schwarz is
// flat). Validated against exact eri_quartet across s/p/d pairs and separations.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/quartet.hpp"
#include "intti/screening.hpp"
#include "intti/tgrid.hpp"

namespace {

using Shell = intti::PrimitiveShell<double>;

// Schwarz factor Q_pair = sqrt(max_component |(pair|pair)|).
double schwarz_Q(const intti::ShellPair<double> &sp,
                 const intti::TGrid<double> &grid) {
  const int n = intti::ncart(sp.la) * intti::ncart(sp.lb);
  std::vector<double> blk(static_cast<std::size_t>(n) * n);
  intti::eri_quartet(sp, sp, grid, blk.data());
  double m = 0;
  for (int i = 0; i < n; ++i) m = std::max(m, std::abs(blk[i * n + i]));
  return std::sqrt(m);
}

double true_max(const intti::ShellPair<double> &bra,
                const intti::ShellPair<double> &ket,
                const intti::TGrid<double> &grid) {
  const int nb = intti::ncart(bra.la) * intti::ncart(bra.lb);
  const int nk = intti::ncart(ket.la) * intti::ncart(ket.lb);
  std::vector<double> blk(static_cast<std::size_t>(nb) * nk);
  intti::eri_quartet(bra, ket, grid, blk.data());
  double m = 0;
  for (double v : blk) m = std::max(m, std::abs(v));
  return m;
}

TEST(Screening, MbieUpperBoundsAndBeatsSchwarz) {
  auto grid = intti::make_tgrid(intti::coulomb());
  struct Cfg {
    int la, lb, lc, ld;
  };
  const Cfg cfgs[] = {{0, 0, 0, 0}, {1, 0, 1, 0}, {1, 1, 0, 0}, {2, 0, 2, 0}};
  for (const auto &cf : cfgs) {
    // bra cloud near the origin, ket cloud pushed out along z
    auto a = Shell{1.1, {0, 0, 0}, cf.la};
    auto b = Shell{0.8, {0.2, 0.1, 0}, cf.lb};
    auto bra = intti::make_pair(a, b);
    const double Qbra = schwarz_Q(bra, grid);
    bool beaten = false;
    // MBIE-1 is the monopole level: it beats Schwarz early for ss, and only at
    // larger R for p/d (their dipole-governed decay needs MBIE-2). The sweep
    // reaches far enough that every class is meaningfully screened.
    for (double z : {1.0, 3.0, 6.0, 12.0, 24.0}) {
      auto c = Shell{0.9, {0, 0, z}, cf.lc};
      auto d = Shell{1.3, {0.1, 0, z + 0.2}, cf.ld};
      auto ket = intti::make_pair(c, d);
      const double Qket = schwarz_Q(ket, grid);
      const double est = intti::mbie_estimate(bra, ket, Qbra, Qket, 1e-12);
      const double tru = true_max(bra, ket, grid);
      const double schwarz = Qbra * Qket;
      // (1) valid upper bound (small slack for the coarse polynomial margin)
      EXPECT_GE(est * (1 + 1e-10), tru)
          << "cfg " << cf.la << cf.lb << cf.lc << cf.ld << " z=" << z;
      // (2) never looser than Schwarz
      EXPECT_LE(est, schwarz * (1 + 1e-12));
      if (est < schwarz * 0.5) beaten = true;
    }
    // at the far separations MBIE must be meaningfully tighter than Schwarz
    EXPECT_TRUE(beaten) << "cfg " << cf.la << cf.lb << cf.lc << cf.ld;
  }
}

} // namespace
