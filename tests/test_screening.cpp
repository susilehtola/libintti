// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
//
// MBIE-1 / QQR distance-including screening (screening.hpp): the estimate must
// (1) always be a valid upper bound on the true |(ab|cd)|, and (2) beat plain
// Schwarz once the bra/ket clouds separate (it decays ~1/R where Schwarz is
// flat). Validated against exact eri_quartet across s/p/d pairs and separations.

#include <cmath>
#include <algorithm>
#include <array>
#include <chrono>
#include <vector>

#include <gtest/gtest.h>

#include "intti/quartet.hpp"
#include "intti/rigrad.hpp"
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
      EXPECT_LT(worst, eps)
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
  EXPECT_LT(worst, eps) << "screened batch differs from unscreened by more than eps";
  printf("t-screening: %zu of %zu nodes evaluated (%.1f%%), max deviation %.2e\n", after,
         before, 100.0 * after / before, worst);
}
// WHERE t-screening actually pays. This is an extended-system optimisation,
// not a general one, and the numbers are lopsided enough that shipping it on by
// default would be wrong:
//
//     compact 9-atom (ethanol-like)   94.6% of nodes kept   1.06x
//     two clusters, 30 bohr apart     15.1%                 6.6x
//     linear chain, 5.5 bohr spacing   4.1%                24x
//
// The reason is structural, not a tuning failure: theta(t) R^2 only becomes
// large when R exceeds the pair extents, so a molecule whose diameter is
// comparable to its basis extent has no distant quartets to truncate. The
// chain is the regime distance-including screening exists for.
//
// Asserted as a PATTERN rather than exact fractions, which depend on the basis.
TEST(Screening, PaysOnExtendedSystemsNotCompactOnes) {
  auto grid = intti::make_tgrid(intti::coulomb());
  const int nt = grid.n();
  auto fraction = [&](const std::vector<std::array<double, 3>> &at, double tau) {
    std::vector<Shell> sh;
    for (const auto &c : at) {
      sh.push_back(Shell{1.3, {c[0], c[1], c[2]}, 0});
      sh.push_back(Shell{0.35, {c[0], c[1], c[2]}, 0});
      sh.push_back(Shell{0.6, {c[0], c[1], c[2]}, 1});
    }
    std::vector<intti::ShellPair<double>> plist;
    for (std::size_t i = 0; i < sh.size(); ++i)
      for (std::size_t j = 0; j < sh.size(); ++j)
        plist.push_back(intti::make_pair(sh[i], sh[j]));
    auto tab = intti::make_pair_table(plist);
    const int npair = static_cast<int>(plist.size());
    std::vector<std::pair<int, int>> qs;
    for (int a = 0; a < npair; ++a)
      for (int b = a; b < npair; ++b) qs.push_back({a, b});
    auto batch = intti::make_batch(tab, qs);
    intti::t_screen_batch(batch, plist, grid, tau);
    const auto [before, after] = intti::t_screen_nodes(batch, nt);
    return double(after) / double(before);
  };
  const std::vector<std::array<double, 3>> compact = {
      {0, 0, 0},      {2.9, 0, 0},       {3.8, 2.3, 0},     {-0.7, 1.9, 0},
      {-0.7, -1, 1.6}, {-0.7, -1, -1.6}, {3.6, -1, 1.6},    {3.6, -1, -1.6},
      {5.6, 2.2, 0}};
  std::vector<std::array<double, 3>> chain;
  for (int i = 0; i < 9; ++i) chain.push_back({i * 5.5, 0, 0});

  const double fc = fraction(compact, 1e-10), fe = fraction(chain, 1e-10);
  EXPECT_GT(fc, 0.8) << "a compact molecule suddenly screens well -- verify the bound is "
                        "still an upper bound before believing it";
  EXPECT_LT(fe, 0.2) << "the extended case no longer screens: the estimate has regressed";
  EXPECT_LT(fe, fc / 3) << "screening no longer discriminates by system extent";
}
// Screening a DERIVATIVE batch must not amplify. The quartets there are
// promoted/demoted pairs combined with md_grad_terms coefficients of -2*alpha
// and l, so a per-quartet bound of eps could in principle come out of the
// digest multiplied by 2*alpha -- which reaches 1e6 for a tight basis function.
//
// It does not: the output error stays at or below eps for exponents spanning
// six decades, and in practice sits at the Hessian's own round-off floor.
//
// Honest about what this shows: it is a GUARD, not a demonstration. The error
// is independent of eps across four decades, which means screening is only
// removing quartets that contribute nothing here -- the desired behaviour, but
// it leaves the amplification path itself unexercised. If a case is found where
// t-screening materially changes an RI derivative, this test should be
// retargeted at it.
TEST(Screening, DerivativeScreeningDoesNotAmplify) {
  auto grid = intti::make_tgrid(intti::coulomb<double>());
  auto probe = [&](double atight, double aloose) {
    std::vector<intti::PrimitiveShell<double>> osh, ash;
    for (int i = 0; i < 4; ++i) {
      const double z = i * 20.0;
      osh.push_back({atight, {0, 0, z}, 0});
      osh.push_back({aloose, {0, 0, z}, 1});
      ash.push_back({atight * 2, {0, 0, z}, 0});
      ash.push_back({aloose * 2, {0, 0, z}, 1});
    }
    auto orb = intti::make_basis(osh);
    auto aux = intti::make_basis(ash);
    const int n = orb.nao;
    std::vector<double> D(static_cast<std::size_t>(n) * n);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) D[i * n + j] = 0.1 + 0.3 * std::sin(0.7 * i + 1.3 * j);
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j) D[i * n + j] = D[j * n + i];
    const auto ref = intti::ri_j_hessian(orb, aux, D.data(), grid, 1e-12, 0, 0.0);
    double scale = 0;
    for (double v : ref) scale = std::max(scale, std::abs(v));
    EXPECT_GT(scale, 1e-2) << "Hessian trivially zero";
    for (double eps : {1e-12, 1e-10, 1e-8}) {
      const auto got = intti::ri_j_hessian(orb, aux, D.data(), grid, 1e-12, 0, eps);
      double worst = 0;
      for (std::size_t i = 0; i < ref.size(); ++i)
        worst = std::max(worst, std::abs(got[i] - ref[i]));
      // a factor of 10 of headroom over eps; 2*alpha would be up to 1e6
      EXPECT_LT(worst, std::max(eps * 10, 1e-11 * scale))
          << "derivative screening amplified: alpha_max=" << atight << " eps=" << eps;
    }
  };
  probe(2.2, 0.45);
  probe(11720.0, 0.0737); // cc-pVDZ oxygen span
  probe(1.0e6, 0.05);
}

// Filtering BEFORE the batch is built, rather than zeroing node counts after.
// A fully-screened quartet left in the batch still pays the f-phase and still
// occupies output slots, so nout_total -- which sets the largest allocation in
// the engine -- does not shrink at all. This checks that dropping them changes
// nothing numerically and does shrink it.
TEST(Screening, PreFilterShrinksTheBatchWithoutChangingResults) {
  auto grid = intti::make_tgrid(intti::coulomb());
  const int nt = grid.n();
  std::vector<Shell> sh;
  for (int i = 0; i < 6; ++i) {
    const double z = i * 9.0;
    sh.push_back(Shell{1.3, {0, 0, z}, 0});
    sh.push_back(Shell{0.4, {0, 0, z}, 1});
  }
  std::vector<intti::ShellPair<double>> plist;
  for (std::size_t i = 0; i < sh.size(); ++i)
    for (std::size_t j = 0; j < sh.size(); ++j)
      plist.push_back(intti::make_pair(sh[i], sh[j]));
  auto tab = intti::make_pair_table(plist);
  const int npair = static_cast<int>(plist.size());
  std::vector<std::pair<int, int>> qs;
  for (int a = 0; a < npair; ++a)
    for (int b = a; b < npair; ++b) qs.push_back({a, b});

  intti::QuartetWorkspace<double> ws;
  auto full = intti::make_batch(tab, qs);
  Kokkos::View<double *> ref("ref", full.nout_total);
  intti::eri_quartets(tab, full, grid, ref, ws);

  const double eps = 1e-11;
  auto [live, livekeep] = intti::t_screen_filter(qs, plist, grid, eps);
  auto lean = intti::make_batch(tab, live);
  intti::t_screen_apply(lean, livekeep, nt);
  Kokkos::View<double *> got("got", lean.nout_total);
  intti::eri_quartets(tab, lean, grid, got, ws);

  EXPECT_LT(live.size(), qs.size()) << "nothing was dropped: the filter is inert";
  EXPECT_LT(lean.nout_total, full.nout_total) << "nout_total did not shrink";

  // every surviving quartet must match the unfiltered result, and every dropped
  // one must have been genuinely negligible
  auto hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ref);
  auto hg = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, got);
  double worst_live = 0, worst_dropped = 0, scale = 0;
  std::size_t li = 0;
  for (std::size_t q = 0; q < qs.size(); ++q) {
    const std::size_t o0 = full.h_offset[q], o1 = full.h_offset[q + 1];
    const bool kept = li < live.size() && live[li] == qs[q];
    for (std::size_t k = o0; k < o1; ++k) scale = std::max(scale, std::abs(hr(k)));
    if (kept) {
      const std::size_t p0 = lean.h_offset[li];
      for (std::size_t k = 0; k < o1 - o0; ++k)
        worst_live = std::max(worst_live, std::abs(hg(p0 + k) - hr(o0 + k)));
      ++li;
    } else {
      for (std::size_t k = o0; k < o1; ++k)
        worst_dropped = std::max(worst_dropped, std::abs(hr(k)));
    }
  }
  EXPECT_EQ(li, live.size()) << "survivor bookkeeping is out of step";
  EXPECT_GT(scale, 1e-3) << "integrals trivially zero";
  EXPECT_LT(worst_live, eps) << "a surviving quartet changed";
  EXPECT_LT(worst_dropped, eps) << "a DROPPED quartet was not negligible";
  printf("pre-filter: %zu of %zu quartets kept, nout %zu -> %zu (%.1f%%); "
         "worst dropped %.2e\n", live.size(), qs.size(), full.nout_total,
         lean.nout_total, 100.0 * lean.nout_total / full.nout_total, worst_dropped);
}

// Both bounds, swept hard. The t-screening estimate was wrong for eight commits
// because the tests asserted at eps*10 and eps*100 -- slack that large tests an
// order of magnitude, not a bound. This sweeps angular momenta, exponent ratios
// and separations together and asserts each estimate is >= the true value with
// only round-off allowed, which is the only assertion that actually tests the
// claim.
TEST(Screening, BoundsHoldUnderASweep) {
  auto grid = intti::make_tgrid(intti::coulomb());
  const int nt = grid.n();
  const double alphas[] = {0.05, 0.4, 3.0, 40.0, 900.0};
  int mbie_checked = 0, tscr_checked = 0, tscr_truncated = 0;
  double worst_mbie_ratio = 0, worst_tscr = 0;

  for (int la = 0; la <= 2; ++la)
    for (int lb = 0; lb <= 2 - la; ++lb)
      for (int lc = 0; lc <= 2; ++lc)
        for (int ld = 0; ld <= 2 - lc; ++ld)
          for (double aa : alphas)
            for (double ac : alphas)
              for (double R : {0.5, 1.5, 4.0, 10.0, 25.0}) {
                auto a = Shell{aa, {0, 0, 0}, la};
                auto b = Shell{aa * 0.35, {0.21, 0.13, 0}, lb};
                auto c = Shell{ac, {0, 0, R}, lc};
                auto d = Shell{ac * 0.6, {0.11, 0, R + 0.17}, ld};
                auto bra = intti::make_pair(a, b);
                auto ket = intti::make_pair(c, d);

                const int nb = intti::ncart(la) * intti::ncart(lb);
                const int nk = intti::ncart(lc) * intti::ncart(ld);
                std::vector<double> blk(static_cast<std::size_t>(nb) * nk);
                intti::eri_quartet(bra, ket, grid, blk.data());
                double tru = 0;
                for (double v : blk) tru = std::max(tru, std::abs(v));

                // (1) MBIE must be an upper bound on the true maximum
                const double Qb = schwarz_Q(bra, grid), Qk = schwarz_Q(ket, grid);
                const double est = intti::mbie_estimate(bra, ket, Qb, Qk, 1e-12);
                ++mbie_checked;
                if (tru > 0) worst_mbie_ratio = std::max(worst_mbie_ratio, tru / est);
                EXPECT_GE(est * (1 + 1e-10), tru)
                    << "MBIE under-estimates: l=" << la << lb << lc << ld
                    << " alpha=" << aa << "," << ac << " R=" << R;

                // (2) truncating to t_screen_keep must lose less than eps
                const double eps = 1e-11;
                const int keep = intti::t_screen_keep(bra, ket, grid, eps);
                ++tscr_checked;
                if (keep < nt) ++tscr_truncated;
                intti::TGrid<double> cut = grid;
                cut.t.resize(keep);
                cut.w.resize(keep);
                cut.tail_coeff = 0.0;
                std::vector<double> got(static_cast<std::size_t>(nb) * nk);
                intti::eri_quartet(bra, ket, cut, got.data());
                double lost = 0;
                for (std::size_t i = 0; i < blk.size(); ++i)
                  lost = std::max(lost, std::abs(got[i] - blk[i]));
                worst_tscr = std::max(worst_tscr, lost);
                EXPECT_LT(lost, eps)
                    << "t-screening lost more than eps: l=" << la << lb << lc << ld
                    << " alpha=" << aa << "," << ac << " R=" << R << " keep=" << keep;
              }
  printf("sweep: %d MBIE checks (worst true/est = %.3f), %d t-screen checks "
         "(%d truncated, worst loss %.2e)\n",
         mbie_checked, worst_mbie_ratio, tscr_checked, tscr_truncated, worst_tscr);
  EXPECT_GT(tscr_truncated, tscr_checked / 10)
      << "the sweep barely exercised truncation";
  EXPECT_LE(worst_mbie_ratio, 1.0 + 1e-9) << "MBIE was exceeded somewhere in the sweep";
}

// High total angular momentum, which the sweep above does not reach (it tops out
// at L = la+lb+lc+ld = 4). The Cramer factor carries sqrt(L!), so if the bound
// degrades anywhere it is here -- a real basis reaches L = 8 for (dd|dd) and 12
// for (ff|ff).
TEST(Screening, BoundsHoldAtHighAngularMomentum) {
  auto grid = intti::make_tgrid(intti::coulomb());
  const int nt = grid.n();
  const double eps = 1e-11;
  int truncated = 0, checked = 0;
  double worst_loss = 0, worst_mbie = 0;
  for (int l = 1; l <= 3; ++l)
    for (double aa : {0.3, 2.0, 30.0})
      for (double R : {1.0, 3.0, 8.0, 20.0}) {
        auto a = Shell{aa, {0, 0, 0}, l};
        auto b = Shell{aa * 0.4, {0.19, 0.07, 0}, l};
        auto c = Shell{aa * 0.8, {0, 0, R}, l};
        auto d = Shell{aa * 0.3, {0.13, 0, R + 0.21}, l};
        auto bra = intti::make_pair(a, b);
        auto ket = intti::make_pair(c, d);
        const int nb = intti::ncart(l) * intti::ncart(l);
        std::vector<double> blk(static_cast<std::size_t>(nb) * nb);
        intti::eri_quartet(bra, ket, grid, blk.data());
        double tru = 0;
        for (double v : blk) tru = std::max(tru, std::abs(v));

        const double est = intti::mbie_estimate(bra, ket, schwarz_Q(bra, grid),
                                                schwarz_Q(ket, grid), 1e-12);
        if (tru > 0) worst_mbie = std::max(worst_mbie, tru / est);
        EXPECT_GE(est * (1 + 1e-10), tru) << "MBIE under-estimates at L=" << 4 * l;

        const int keep = intti::t_screen_keep(bra, ket, grid, eps);
        ++checked;
        if (keep < nt) ++truncated;
        intti::TGrid<double> cut = grid;
        cut.t.resize(keep);
        cut.w.resize(keep);
        cut.tail_coeff = 0.0;
        std::vector<double> got(blk.size());
        intti::eri_quartet(bra, ket, cut, got.data());
        double lost = 0;
        for (std::size_t i = 0; i < blk.size(); ++i)
          lost = std::max(lost, std::abs(got[i] - blk[i]));
        worst_loss = std::max(worst_loss, lost);
        EXPECT_LT(lost, eps) << "t-screening lost more than eps at L=" << 4 * l
                             << " alpha=" << aa << " R=" << R << " keep=" << keep;
      }
  printf("high-L: %d checks up to L=12, %d truncated, worst loss %.2e, "
         "worst MBIE true/est %.3f\n", checked, truncated, worst_loss, worst_mbie);
  EXPECT_GT(truncated, 0) << "no truncation at high L: the sweep proves nothing there";
}
