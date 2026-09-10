// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

// What the t-quadrature node count buys, pinned in code.
//
// TGridSpec defaults to n = 64 Mobius-mapped Gauss-Legendre nodes. That number
// is an ACCURACY calibration for an 8-decade exponent span (1e-2 .. 1e6), not a
// hardware choice -- nothing about it is tied to a warp or thread-block width.
// Since the node count is the innermost loop of every integral, it multiplies
// the cost of everything, so the calibration deserves to be asserted rather
// than remembered.
//
// The measurement below also records the headroom: a REAL basis spans far less
// than 8 decades (cc-pVDZ oxygen is ~0.07 .. 1.2e4, about 5.4) and reaches
// round-off at n = 40. mobius_spec_for_range, the adaptive sizer, returns 64
// for that case anyway -- so the ~1.6x is real but currently unclaimed. This
// test fails if that stops being true in either direction.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/fock.hpp"
#include "intti/kernel.hpp"
#include "intti/tgrid.hpp"

namespace {

intti::ShellBasis<double> span_basis(const std::vector<double> &alphas) {
  std::vector<intti::PrimitiveShell<double>> sh;
  for (double a : alphas) {
    sh.push_back({a, {0.0, 0.0, 0.0}, 0});
    sh.push_back({a * 0.7, {0.0, 0.0, 1.4}, 1});
  }
  return intti::make_basis(sh);
}

std::vector<double> sym_density(int n) {
  std::vector<double> D(static_cast<std::size_t>(n) * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) D[i * n + j] = 0.1 + 0.3 * std::sin(0.7 * i + 1.3 * j);
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) D[i * n + j] = D[j * n + i];
  return D;
}

/// relative error of J (and K) at `nodes` against a converged reference
std::pair<double, double> err_at(const intti::ShellBasis<double> &b,
                                 const std::vector<double> &D, int nodes) {
  const int n = b.nao;
  auto build = [&](int nn) {
    intti::TGridSpec<double> spec;
    spec.n = nn;
    auto grid = intti::make_tgrid(intti::coulomb<double>(), spec);
    std::vector<double> J(static_cast<std::size_t>(n) * n, 0.0);
    std::vector<double> K(static_cast<std::size_t>(n) * n, 0.0);
    intti::coulomb_build(b, D.data(), grid, J.data());
    intti::exchange_build(b, D.data(), grid, K.data(), 0.0);
    return std::make_pair(J, K);
  };
  const auto ref = build(256), got = build(nodes);
  double dj = 0, dk = 0, sj = 0, sk = 0;
  for (std::size_t i = 0; i < ref.first.size(); ++i) {
    dj = std::max(dj, std::abs(got.first[i] - ref.first[i]));
    dk = std::max(dk, std::abs(got.second[i] - ref.second[i]));
    sj = std::max(sj, std::abs(ref.first[i]));
    sk = std::max(sk, std::abs(ref.second[i]));
  }
  return {dj / sj, dk / sk};
}

} // namespace

// The default n = 64 delivers what it was calibrated for across 8 decades, and
// a materially smaller grid does NOT -- so the default is not simply padded.
TEST(TGridNodes, DefaultResolvesEightDecades) {
  auto b = span_basis({1e6, 1e4, 1e2, 1.0, 1e-2});
  auto D = sym_density(b.nao);
  const auto e64 = err_at(b, D, 64);
  EXPECT_LT(e64.first, 1e-12) << "default node count does not resolve 8 decades (J)";
  EXPECT_LT(e64.second, 1e-12) << "default node count does not resolve 8 decades (K)";
  const auto e40 = err_at(b, D, 40);
  EXPECT_GT(e40.first, 1e-10)
      << "40 nodes now suffice over 8 decades: the default is padding, re-calibrate";
}

// A REAL basis spans ~5.4 decades and is converged well below the default, which
// is the headroom a basis-sized grid would claim.
TEST(TGridNodes, RealBasisConvergesWellBelowTheDefault) {
  auto b = span_basis({11720.0, 273.2, 34.46, 4.756, 0.3782, 0.0737}); // cc-pVDZ O
  auto D = sym_density(b.nao);
  const auto e40 = err_at(b, D, 40);
  EXPECT_LT(e40.first, 1e-12) << "40 nodes no longer converge a 5-decade span (J)";
  EXPECT_LT(e40.second, 1e-12) << "40 nodes no longer converge a 5-decade span (K)";
}

// ...and the adaptive sizer does not currently claim it: it returns the default
// for a span needing far fewer nodes. Asserted so that fixing it is a visible,
// deliberate change to this expectation rather than a silent one.
TEST(TGridNodes, AdaptiveSizerDoesNotYetShrinkForNarrowSpans) {
  auto spec = intti::mobius_spec_for_range<double>(0.0516, 1.17e4);
  EXPECT_GE(spec.n, 60) << "mobius_spec_for_range now shrinks narrow spans -- if this is "
                           "intended, default_grid() should start using it";
}
