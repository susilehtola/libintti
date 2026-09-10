// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
//
// MBIE-1 / QQR distance-including screening (screening.hpp): the estimate must
// (1) always be a valid upper bound on the true |(ab|cd)|, and (2) beat plain
// Schwarz once the bra/ket clouds separate (it decays ~1/R where Schwarz is
// flat). Validated against exact eri_quartet across s/p/d pairs and separations.

#include <cmath>
#include <algorithm>
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

// t-RESOLVED screening: the quadrature is truncated per quartet, and the two
// things that must hold are that the discarded tail really is below eps, and
// that anything is discarded at all.
//
// The first is the correctness property and is checked against the EXACT
// quartet: recompute on the truncated grid and compare. The second guards
// against a bound so loose it never fires, which would pass the first
// trivially.
TEST(Screening, TResolvedTruncationRespectsEpsAndFires) {
  auto full = intti::make_tgrid(intti::coulomb());
  const int nt = full.n();
  struct Cfg { int la, lb, lc, ld; };
  const Cfg cfgs[] = {{0, 0, 0, 0}, {1, 0, 1, 0}, {1, 1, 1, 1}, {2, 0, 2, 0}};
  const double eps = 1e-10;

  for (const auto &cf : cfgs) {
    auto a = Shell{1.1, {0, 0, 0}, cf.la};
    auto b = Shell{0.8, {0.2, 0.1, 0}, cf.lb};
    auto bra = intti::make_pair(a, b);
    bool fired = false;
    for (double R : {1.0, 2.0, 4.0, 8.0, 16.0}) {
      auto c = Shell{0.9, {0, 0, R}, cf.lc};
      auto d = Shell{0.7, {0.1, 0, R + 0.3}, cf.ld};
      auto ket = intti::make_pair(c, d);

      const int keep = intti::t_screen_keep(bra, ket, full, eps);
      ASSERT_GE(keep, 0);
      ASSERT_LE(keep, nt);
      if (keep < nt) fired = true;

      // exact quartet on the full grid
      const int nb = intti::ncart(cf.la) * intti::ncart(cf.lb);
      const int nk = intti::ncart(cf.lc) * intti::ncart(cf.ld);
      std::vector<double> ref(static_cast<std::size_t>(nb) * nk);
      intti::eri_quartet(bra, ket, full, ref.data());

      // ...and on the truncated prefix
      intti::TGrid<double> cut = full;
      cut.t.resize(keep);
      cut.w.resize(keep);
      cut.tail_coeff = 0.0; // the tail term IS the discarded large-t region
      std::vector<double> got(static_cast<std::size_t>(nb) * nk);
      intti::eri_quartet(bra, ket, cut, got.data());

      double worst = 0;
      for (std::size_t i = 0; i < ref.size(); ++i)
        worst = std::max(worst, std::abs(got[i] - ref[i]));
      EXPECT_LT(worst, eps * 10)
          << "truncating to " << keep << "/" << nt << " nodes at R=" << R
          << " lost more than eps for " << cf.la << cf.lb << cf.lc << cf.ld;
    }
    EXPECT_TRUE(fired) << "t-screening never truncated for " << cf.la << cf.lb << cf.lc
                       << cf.ld << ": the bound is too loose to be useful";
  }
}
// How much it removes, and -- the point -- that it removes nearly as much for
// p and d pairs as for s. MBIE-1 is monopole-level, so it screens p/d only at
// much larger separation than s; here the decay is exact at every node and the
// angular class barely enters. Asserting that keeps the advantage from silently
// regressing to monopole behaviour.
TEST(Screening, TResolvedTruncationIsAngularMomentumBlind) {
  auto grid = intti::make_tgrid(intti::coulomb());
  const int nt = grid.n();
  struct Cfg { int la, lb, lc, ld; };
  const Cfg cfgs[] = {{0, 0, 0, 0}, {1, 0, 1, 0}, {1, 1, 1, 1}, {2, 0, 2, 0}};
  int keep_far[4] = {0, 0, 0, 0};
  for (int ci = 0; ci < 4; ++ci) {
    const auto &cf = cfgs[ci];
    auto a = Shell{1.1, {0, 0, 0}, cf.la};
    auto b = Shell{0.8, {0.2, 0.1, 0}, cf.lb};
    auto bra = intti::make_pair(a, b);

    // close pairs must NOT be truncated: the clouds still overlap and the
    // large-t nodes carry the short-range part
    auto cn = Shell{0.9, {0, 0, 1.0}, cf.lc};
    auto dn = Shell{0.7, {0.1, 0, 1.3}, cf.ld};
    EXPECT_EQ(intti::t_screen_keep(bra, intti::make_pair(cn, dn), grid, 1e-10), nt)
        << "a close quartet was truncated";

    auto cf2 = Shell{0.9, {0, 0, 32.0}, cf.lc};
    auto df2 = Shell{0.7, {0.1, 0, 32.3}, cf.ld};
    keep_far[ci] = intti::t_screen_keep(bra, intti::make_pair(cf2, df2), grid, 1e-10);
    EXPECT_LT(keep_far[ci], nt / 3) << "far quartet barely truncated";
    EXPECT_GT(keep_far[ci], 0) << "the small-t nodes carry the 1/R tail and must survive";
  }
  const int lo = *std::min_element(keep_far, keep_far + 4);
  const int hi = *std::max_element(keep_far, keep_far + 4);
  EXPECT_LE(hi - lo, 4) << "truncation has become angular-momentum dependent, i.e. it has "
                           "degraded towards monopole (MBIE-1) behaviour";
}

// Screening wired into the BATCHED engine: a screened batch must reproduce an
// unscreened one, and must actually evaluate fewer nodes. This is the property
// that matters in production -- the earlier tests check the estimate in
// isolation, this one checks the plumbing that acts on it.
TEST(Screening, ScreenedBatchMatchesUnscreened) {
  auto grid = intti::make_tgrid(intti::coulomb());
  const int nt = grid.n();
  // two well-separated clusters, so a good fraction of the quartets are distant
  std::vector<Shell> sh;
  for (double z : {0.0, 0.9}) {
    sh.push_back(Shell{1.3, {0, 0, z}, 0});
    sh.push_back(Shell{0.6, {0.3, 0, z}, 1});
  }
  for (double z : {24.0, 24.9}) {
    sh.push_back(Shell{1.1, {0, 0, z}, 0});
    sh.push_back(Shell{0.5, {0.2, 0, z}, 1});
  }
  std::vector<intti::ShellPair<double>> plist;
  for (std::size_t i = 0; i < sh.size(); ++i)
    for (std::size_t j = 0; j < sh.size(); ++j)
      plist.push_back(intti::make_pair(sh[i], sh[j]));
  auto tab = intti::make_pair_table(plist);
  const int npair = static_cast<int>(plist.size());
  std::vector<std::pair<int, int>> quartets;
  for (int a = 0; a < npair; ++a)
    for (int b = a; b < npair; ++b) quartets.push_back({a, b});

  intti::QuartetWorkspace<double> ws;
  auto full = intti::make_batch(tab, quartets);
  Kokkos::View<double *> ref("ref", full.nout_total);
  intti::eri_quartets(tab, full, grid, ref, ws);

  auto cut = intti::make_batch(tab, quartets);
  const double eps = 1e-11;
  intti::t_screen_batch(cut, plist, grid, eps);
  Kokkos::View<double *> got("got", cut.nout_total);
  intti::eri_quartets(tab, cut, grid, got, ws);

  const auto [before, after] = intti::t_screen_nodes(cut, nt);
  EXPECT_LT(after, before * 9 / 10) << "screening removed almost nothing on a "
                                       "two-cluster system: " << after << "/" << before;

  auto hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ref);
  auto hg = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, got);
  double worst = 0, scale = 0;
  for (std::size_t i = 0; i < ref.extent(0); ++i) {
    worst = std::max(worst, std::abs(hg(i) - hr(i)));
    scale = std::max(scale, std::abs(hr(i)));
  }
  EXPECT_GT(scale, 1e-3) << "integrals trivially zero";
  EXPECT_LT(worst, eps * 100) << "screened batch differs from unscreened by more than eps";
  printf("t-screening: %zu of %zu nodes evaluated (%.1f%%), max deviation %.2e\n", after,
         before, 100.0 * after / before, worst);
}
