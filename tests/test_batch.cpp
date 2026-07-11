// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/batch.hpp"
#include "intti/quartet.hpp"

namespace {

using Shell = intti::PrimitiveShell<double>;

std::vector<intti::ShellPair<double>> test_pairs() {
  const std::vector<Shell> shells = {
      {0.8, {0.0, 0.1, -0.3}, 0}, {1.3, {0.5, -0.2, 0.4}, 1},
      {2.1, {1.0, 0.8, 0.0}, 2},  {0.35, {-0.4, 0.3, 1.1}, 0},
      {5.0, {0.2, 0.0, 0.6}, 1},
  };
  std::vector<intti::ShellPair<double>> pairs;
  for (std::size_t i = 0; i < shells.size(); ++i)
    for (std::size_t j = i; j < shells.size(); ++j)
      pairs.push_back(intti::make_pair(shells[i], shells[j]));
  return pairs;
}

TEST(Batch, MatchesSingleQuartetDriver) {
  auto pairs = test_pairs();
  auto tab = intti::make_pair_table(pairs);
  const int npair = static_cast<int>(pairs.size());

  // every (bra, ket) pair combination, mixed classes, class-sorted
  std::vector<std::pair<int, int>> quartets;
  for (int ib = 0; ib < npair; ++ib)
    for (int ik = 0; ik < npair; ++ik)
      quartets.push_back({ib, ik});
  intti::sort_by_class(tab, quartets);

  auto grid = intti::make_tgrid(intti::coulomb());
  auto batch = intti::make_batch(tab, quartets);
  Kokkos::View<double *> out("out", batch.nout_total);
  intti::QuartetWorkspace<double> ws;
  ws.chunk = 32; // force several chunks
  intti::eri_quartets(tab, batch, grid, out, ws);
  auto oh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);

  for (std::size_t iq = 0; iq < quartets.size(); ++iq) {
    const auto &[ib, ik] = quartets[iq];
    const int nout = intti::ncart(pairs[ib].la) * intti::ncart(pairs[ib].lb) *
                     intti::ncart(pairs[ik].la) * intti::ncart(pairs[ik].lb);
    std::vector<double> ref(nout);
    intti::eri_quartet(pairs[ib], pairs[ik], grid, ref.data());
    double scale = 0.0;
    for (int k = 0; k < nout; ++k)
      scale = std::max(scale, std::abs(ref[k]));
    for (int k = 0; k < nout; ++k)
      EXPECT_NEAR(oh(batch.h_offset[iq] + k), ref[k], 1e-14 * scale + 1e-300)
          << "quartet " << iq << " component " << k;
  }
}

TEST(Batch, WorkspaceIsReused) {
  auto pairs = test_pairs();
  auto tab = intti::make_pair_table(pairs);
  std::vector<std::pair<int, int>> quartets;
  for (int ib = 0; ib < tab.npair; ++ib)
    quartets.push_back({ib, ib});
  auto grid = intti::make_tgrid(intti::coulomb());
  auto batch = intti::make_batch(tab, quartets);
  Kokkos::View<double *> out("out", batch.nout_total);
  intti::QuartetWorkspace<double> ws;
  intti::eri_quartets(tab, batch, grid, out, ws);
  const double *fp = ws.f.data();
  const double *gp = ws.g.data();
  // repeated identical calls must not reallocate
  for (int rep = 0; rep < 3; ++rep) {
    intti::eri_quartets(tab, batch, grid, out, ws);
    EXPECT_EQ(ws.f.data(), fp);
    EXPECT_EQ(ws.g.data(), gp);
  }
}

TEST(Batch, LinLogGridWithTail) {
  // batched driver must apply the delta-function tail exactly like the
  // single-quartet driver on truncated grids
  auto pairs = test_pairs();
  auto tab = intti::make_pair_table(pairs);
  std::vector<std::pair<int, int>> quartets = {{0, 2}, {1, 4}, {3, 3}};
  intti::TGridSpec<double> spec;
  spec.mapping = intti::TMapping::LinLog;
  auto grid = intti::make_tgrid(intti::coulomb(), spec);
  auto batch = intti::make_batch(tab, quartets);
  Kokkos::View<double *> out("out", batch.nout_total);
  intti::QuartetWorkspace<double> ws;
  intti::eri_quartets(tab, batch, grid, out, ws);
  auto oh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  for (std::size_t iq = 0; iq < quartets.size(); ++iq) {
    const auto &[ib, ik] = quartets[iq];
    const int nout = intti::ncart(pairs[ib].la) * intti::ncart(pairs[ib].lb) *
                     intti::ncart(pairs[ik].la) * intti::ncart(pairs[ik].lb);
    std::vector<double> ref(nout);
    intti::eri_quartet(pairs[ib], pairs[ik], grid, ref.data());
    for (int k = 0; k < nout; ++k)
      EXPECT_NEAR(oh(batch.h_offset[iq] + k), ref[k], 1e-14 * (std::abs(ref[k]) + 1e-3));
  }
}

} // namespace
