// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <cstdint>
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

TEST(Batch, AccumulateContractedBlock) {
  // 2-primitive contracted s and p shells: the contracted (AB|CD) block is
  // one accumulating batch over the 16 primitive quartets
  const double cA[3] = {0.0, 0.1, -0.3}, cB[3] = {0.5, -0.2, 0.4};
  const double cC[3] = {1.0, 0.8, 0.0}, cD[3] = {-0.4, 0.3, 1.1};
  const double za[2] = {0.8, 2.4}, ca[2] = {0.6, 0.5};
  const double zb[2] = {1.3, 0.4}, cb[2] = {0.7, 0.4};
  const double zc[2] = {2.1, 0.9}, cc[2] = {0.5, 0.6};
  const double zd[2] = {0.35, 1.7}, cd[2] = {0.8, 0.3};
  const int la = 1, lb = 0, lc = 1, ld = 0; // (pp|ss)-like block, 9 values
  std::vector<intti::ShellPair<double>> bra_pairs, ket_pairs;
  std::vector<double> bra_c, ket_c;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      bra_pairs.push_back(intti::make_pair(Shell{za[i], {cA[0], cA[1], cA[2]}, la},
                                           Shell{zb[j], {cB[0], cB[1], cB[2]}, lb}));
      bra_c.push_back(ca[i] * cb[j]);
      ket_pairs.push_back(intti::make_pair(Shell{zc[i], {cC[0], cC[1], cC[2]}, lc},
                                           Shell{zd[j], {cD[0], cD[1], cD[2]}, ld}));
      ket_c.push_back(cc[i] * cd[j]);
    }
  auto all_pairs = bra_pairs;
  all_pairs.insert(all_pairs.end(), ket_pairs.begin(), ket_pairs.end());
  auto tab = intti::make_pair_table(all_pairs);
  std::vector<std::pair<int, int>> quartets;
  std::vector<double> coeff;
  for (int ib = 0; ib < 4; ++ib)
    for (int ik = 0; ik < 4; ++ik) {
      quartets.push_back({ib, 4 + ik});
      coeff.push_back(bra_c[ib] * ket_c[ik]);
    }
  auto grid = intti::make_tgrid(intti::coulomb());
  auto batch = intti::make_batch(tab, quartets);
  const int nout = 9;
  Kokkos::View<double *> out("out", nout);
  Kokkos::View<double *> cv("coeff", quartets.size());
  Kokkos::View<std::int64_t *> seg("seg", quartets.size());
  {
    auto hc = Kokkos::create_mirror_view(cv);
    auto hs = Kokkos::create_mirror_view(seg);
    for (std::size_t iq = 0; iq < quartets.size(); ++iq) {
      hc(iq) = coeff[iq];
      hs(iq) = 0; // all primitives accumulate into the single contracted block
    }
    Kokkos::deep_copy(cv, hc);
    Kokkos::deep_copy(seg, hs);
  }
  intti::QuartetWorkspace<double> ws;
  intti::eri_quartets_accumulate<double>(tab, batch, cv, seg, grid, out, ws);
  auto oh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  // reference: explicit weighted primitive sum
  std::vector<double> ref(nout, 0.0);
  for (std::size_t iq = 0; iq < quartets.size(); ++iq) {
    double buf[nout];
    intti::eri_quartet(all_pairs[quartets[iq].first], all_pairs[quartets[iq].second],
                       grid, buf);
    for (int k = 0; k < nout; ++k)
      ref[k] += coeff[iq] * buf[k];
  }
  double scale = 0.0;
  for (int k = 0; k < nout; ++k)
    scale = std::max(scale, std::abs(ref[k]));
  for (int k = 0; k < nout; ++k)
    EXPECT_NEAR(oh(k), ref[k], 1e-13 * scale) << "component " << k;
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


TEST(Batch, OutputScanIsSixtyFourBit) {
  // The per-quartet output scan is the one quantity in the batch planner that
  // grows without bound: with d functions a quartet already materialises 6^4 =
  // 1296 integrals, so a batch of a few million quartets passes 2^31 and an int
  // scan would wrap to a negative offset -- a silent out-of-bounds write rather
  // than a diagnosable failure. Plan such a batch (the quartet list and the scan
  // only; the output itself is never allocated) and check the arithmetic holds.
  const std::vector<Shell> shells = {{2.1, {1.0, 0.8, 0.0}, 2}, {1.7, {0.0, 0.2, 0.3}, 2}};
  std::vector<intti::ShellPair<double>> pairs = {intti::make_pair(shells[0], shells[1])};
  auto tab = intti::make_pair_table(pairs);
  const std::size_t per = 6 * 6 * 6 * 6; // (dd|dd)
  const std::size_t nq = (std::size_t(1) << 31) / per + 1000;
  std::vector<std::pair<int, int>> quartets(nq, {0, 0});
  auto batch = intti::make_batch(tab, quartets);
  ASSERT_EQ(batch.h_offset.size(), nq + 1);
  EXPECT_GT(batch.nout_total, std::size_t(1) << 31) << "batch must cross the int boundary";
  EXPECT_EQ(batch.nout_total, nq * per);
  // the scan stays monotone across the boundary (an int scan turns negative here)
  const std::size_t at = (std::size_t(1) << 31) / per;
  EXPECT_LT(batch.h_offset[at], batch.h_offset[at + 1]);
  EXPECT_EQ(batch.h_offset[at + 1], (at + 1) * per);
  // and the device copy carries the same value, not a truncation of it
  auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, batch.out_offset);
  EXPECT_EQ(static_cast<std::size_t>(ho(nq)), batch.nout_total);
  EXPECT_GT(ho(nq), std::int64_t(0));
}

} // namespace
