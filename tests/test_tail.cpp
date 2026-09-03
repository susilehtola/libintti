// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
//
// Higher-order Losilla delta-tail corrections for the truncated Coulomb
// t-quadrature. Including tail orders 0..K leaves a residual O(1/t_c^{2K+4}),
// so a harder (cheaper) truncation reaches the same accuracy. Oracle values are
// derived in references/sympy_tail.py.
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#ifdef INTTI_HAVE_QUADMATH
#include <quadmath.h>
#endif

#include "intti/fock.hpp"
#include "intti/quartet.hpp"

namespace {

using intti::PrimitiveShell;

// exact (ss|ss) with p=1.3, q=0.8, |P-Q|=1.1 (references/sympy_tail.py).
constexpr double kExactSSSS = 19.304872806918379;

// (ss|ss) built as two coincident s primitives per centre so the pair-product
// exponent is p (resp. q); |A-B| = R along x. Unnormalised, matching the oracle.
template <class Real>
Real ssss(double p, double q, double R, const intti::TGrid<Real> &grid) {
  PrimitiveShell<Real> a{Real(p / 2), {Real(0), Real(0), Real(0)}, 0};
  PrimitiveShell<Real> c{Real(q / 2), {Real(R), Real(0), Real(0)}, 0};
  Real v;
  intti::eri_quartet(intti::make_pair(a, a), intti::make_pair(c, c), grid, &v);
  return v;
}

// truncated LinLog grid at t_c with tail order K (well-resolved explicit part).
template <class Real> intti::TGrid<Real> tail_grid(double t_c, int K) {
  auto spec = intti::linlog_for<Real>(Real(t_c), Real(1e-15));
  spec.tail_order = K;
  return intti::make_tgrid(intti::coulomb<Real>(), spec);
}

// Untruncated Mobius reference (no tail); converges to the scalar floor.
template <class Real> intti::TGrid<Real> ref_grid() {
  intti::TGridSpec<Real> ms;
  ms.mapping = intti::TMapping::Mobius;
  ms.n = 200;
  return intti::make_tgrid(intti::coulomb<Real>(), ms);
}

TEST(Tail, ReferenceMatchesOracle) {
  const double ref = ssss<double>(1.3, 0.8, 1.1, ref_grid<double>());
  EXPECT_NEAR(ref, kExactSSSS, 1e-11);
}

TEST(Tail, HigherOrderMonotonicallyBetter) {
  // At a fixed hard truncation, each retained order strictly improves accuracy.
  const double t_c = 6.0;
  double prev = 1.0;
  for (int K = 0; K <= 3; ++K) {
    const double v = ssss<double>(1.3, 0.8, 1.1, tail_grid<double>(t_c, K));
    const double err = std::abs((v - kExactSSSS) / kExactSSSS);
    EXPECT_LT(err, prev * 0.2) << "order K=" << K << " err=" << err;
    prev = err;
  }
  // K=3 is dramatically better than the leading-only tail.
  const double e0 = std::abs(ssss<double>(1.3, 0.8, 1.1, tail_grid<double>(t_c, 0)) - kExactSSSS);
  const double e3 = std::abs(ssss<double>(1.3, 0.8, 1.1, tail_grid<double>(t_c, 3)) - kExactSSSS);
  EXPECT_LT(e3, e0 * 1e-5);
}

TEST(Tail, ResidualScalesAsExpectedOrder) {
  // Retaining orders 0..K leaves a residual ~ 1/t_c^{2K+4}: doubling t_c drops
  // the error by ~4^{K+2}. Check K=1 (expect ~64x) and K=2 (~256x).
  for (auto [K, factor] : {std::pair<int, double>{1, 40.0}, {2, 150.0}}) {
    const double e_lo = std::abs(ssss<double>(1.3, 0.8, 1.1, tail_grid<double>(6.0, K)) - kExactSSSS);
    const double e_hi = std::abs(ssss<double>(1.3, 0.8, 1.1, tail_grid<double>(12.0, K)) - kExactSSSS);
    EXPECT_GT(e_lo / e_hi, factor) << "K=" << K << " ratio " << e_lo / e_hi;
  }
}

TEST(Tail, HigherAngularMomentumProducts) {
  // l>0 quartet: every component of the truncated+tail result matches the
  // untruncated reference, and the higher-order tail is far tighter.
  const double kA[3] = {0.0, 0.0, 0.0}, kB[3] = {0.3, -0.1, 0.2};
  const double kC[3] = {1.1, 0.0, 0.0}, kD[3] = {0.9, 0.4, -0.2};
  PrimitiveShell<double> a{0.8, {kA[0], kA[1], kA[2]}, 1};
  PrimitiveShell<double> b{1.3, {kB[0], kB[1], kB[2]}, 0};
  PrimitiveShell<double> c{0.7, {kC[0], kC[1], kC[2]}, 1};
  PrimitiveShell<double> d{0.35, {kD[0], kD[1], kD[2]}, 0};
  const auto bra = intti::make_pair(a, b);
  const auto ket = intti::make_pair(c, d);
  const int nq = 3 * 1 * 3 * 1; // ncart(1)^2

  std::vector<double> ref(nq), k0(nq), k2(nq);
  intti::eri_quartet(bra, ket, ref_grid<double>(), ref.data());
  {
    auto s0 = intti::linlog_for<double>(10.0, 1e-15);
    s0.tail_order = 0;
    intti::eri_quartet(bra, ket, intti::make_tgrid(intti::coulomb(), s0), k0.data());
    auto s2 = intti::linlog_for<double>(10.0, 1e-15);
    s2.tail_order = 2;
    intti::eri_quartet(bra, ket, intti::make_tgrid(intti::coulomb(), s2), k2.data());
  }
  double emax0 = 0, emax2 = 0, refmax = 0;
  for (int i = 0; i < nq; ++i) {
    refmax = std::max(refmax, std::abs(ref[i]));
    emax0 = std::max(emax0, std::abs(k0[i] - ref[i]));
    emax2 = std::max(emax2, std::abs(k2[i] - ref[i]));
  }
  EXPECT_LT(emax2 / refmax, 2e-8) << "order-2 tail l>0 max rel err " << emax2 / refmax;
  EXPECT_LT(emax2, emax0 * 1e-3);
}

TEST(Tail, OrderClampedToKMax) {
  auto spec = intti::linlog_for<double>(8.0);
  spec.tail_order = 100;
  auto grid = intti::make_tgrid(intti::coulomb(), spec);
  EXPECT_EQ(grid.tail_order, intti::TAIL_KMAX);
}

TEST(Tail, CoulombExchangeBuildHigherOrder) {
  // The higher-order tail flows through the matrix-level J and K builders:
  // both converge to the untruncated reference far faster than order 0.
  auto b = intti::make_basis<double>({{1.2, {0, 0, 0}, 0},
                                      {0.35, {0, 0, 0}, 0},
                                      {0.8, {0, 0, 0}, 1},
                                      {1.5, {0, 0, 1.3}, 0},
                                      {0.5, {0, 0, 1.3}, 1}});
  const int nao = b.nao, n2 = nao * nao;
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(-1, 1);
  std::vector<double> D(n2);
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j <= i; ++j)
      D[i * nao + j] = D[j * nao + i] = u(rng);

  intti::TGridSpec<double> ms;
  ms.mapping = intti::TMapping::Mobius;
  ms.n = 200;
  const auto ref = intti::make_tgrid(intti::coulomb(), ms);
  std::vector<double> Jref(n2), Kref(n2);
  intti::coulomb_build(b, D.data(), ref, Jref.data());
  intti::exchange_build(b, D.data(), ref, Kref.data(), 0.0, 0, 1);

  auto build = [&](int K, std::vector<double> &J, std::vector<double> &Kk) {
    auto sp = intti::linlog_for<double>(10.0, 1e-15);
    sp.tail_order = K;
    const auto g = intti::make_tgrid(intti::coulomb(), sp);
    J.assign(n2, 0.0);
    Kk.assign(n2, 0.0);
    intti::coulomb_build(b, D.data(), g, J.data());
    intti::exchange_build(b, D.data(), g, Kk.data(), 0.0, 0, 1);
  };
  std::vector<double> J0, K0, J2, K2;
  build(0, J0, K0);
  build(2, J2, K2);
  double jr = 0, je0 = 0, je2 = 0, kr = 0, ke0 = 0, ke2 = 0;
  for (int i = 0; i < n2; ++i) {
    jr = std::max(jr, std::abs(Jref[i]));
    je0 = std::max(je0, std::abs(J0[i] - Jref[i]));
    je2 = std::max(je2, std::abs(J2[i] - Jref[i]));
    kr = std::max(kr, std::abs(Kref[i]));
    ke0 = std::max(ke0, std::abs(K0[i] - Kref[i]));
    ke2 = std::max(ke2, std::abs(K2[i] - Kref[i]));
  }
  EXPECT_LT(je2 / jr, 1e-7) << "J order-2 rel err " << je2 / jr;
  EXPECT_LT(ke2 / kr, 1e-7) << "K order-2 rel err " << ke2 / kr;
  EXPECT_LT(je2, je0 * 1e-2) << "J: order-2 not better than order-0";
  EXPECT_LT(ke2, ke0 * 1e-2) << "K: order-2 not better than order-0";
}

TEST(Tail, AdaptiveSpecMeetsTargetAtFloor) {
  // adaptive_linlog_tail_spec picks (t_c, tail_order) to hit eps for the
  // tightest pair, whose reduced exponent rho = p q/(p+q) = alpha_max sets the
  // convergence floor t_c > sqrt(alpha_max). Worst-case quartet: all four
  // exponents = alpha_max (p = q = 2 alpha_max, so rho = alpha_max).
  intti::TGridSpec<double> ms;
  ms.mapping = intti::TMapping::Mobius;
  ms.n = 400;
  const auto ref = intti::make_tgrid(intti::coulomb(), ms);
  for (double amax : {1.5, 8.0, 40.0}) {
    for (double eps : {1e-6, 1e-10}) {
      auto sp = intti::adaptive_linlog_tail_spec<double>(0.1, amax, eps);
      EXPECT_GT(sp.t_c, std::sqrt(amax)) << "t_c below convergence floor";
      const auto g = intti::make_tgrid(intti::coulomb(), sp);
      double worst = 0;
      for (double R : {0.0, 0.4, 1.2, 2.5}) {
        intti::PrimitiveShell<double> a{amax, {0, 0, 0}, 0};
        intti::PrimitiveShell<double> c{amax, {R, 0, 0}, 0};
        double v, r;
        intti::eri_quartet(intti::make_pair(a, a), intti::make_pair(c, c), g, &v);
        intti::eri_quartet(intti::make_pair(a, a), intti::make_pair(c, c), ref, &r);
        worst = std::max(worst, std::abs((v - r) / r));
      }
      EXPECT_LT(worst, eps) << "amax=" << amax << " eps=" << eps
                            << " t_c=" << sp.t_c << " K=" << sp.tail_order
                            << " worst=" << worst;
    }
  }
}

#ifdef INTTI_HAVE_QUADMATH
TEST(Tail, QuadPrecisionDeepAccuracy) {
  // In quad the tail residual is visible far below the double floor: at a hard
  // t_c=6 the order-3 tail reaches ~1e-11 where the leading tail is ~3e-5.
  using Q = __float128;
  const Q ref = ssss<Q>(1.3, 0.8, 1.1, ref_grid<Q>());
  const double e0 = (double)fabsq((ssss<Q>(1.3, 0.8, 1.1, tail_grid<Q>(6.0, 0)) - ref) / ref);
  const double e3 = (double)fabsq((ssss<Q>(1.3, 0.8, 1.1, tail_grid<Q>(6.0, 3)) - ref) / ref);
  EXPECT_LT(e3, 1e-10) << "quad order-3 tail err " << e3;
  EXPECT_LT(e3, e0 * 1e-6);
}
#endif

} // namespace
