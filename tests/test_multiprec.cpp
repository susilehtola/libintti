// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

// The core is templated on the scalar type; exercise it beyond double.

#include <cmath>
#include <vector>

#ifdef INTTI_HAVE_QUADMATH
#include <quadmath.h>
#endif

#include <gtest/gtest.h>

#include "intti/quartet.hpp"
#include "intti/reference/boys.hpp"
#include "intti/reference/eri_analytic.hpp"

namespace {

using LD = long double;

TEST(MultiPrec, GaussLegendreLongDouble) {
  const int n = 8;
  LD x[n], w[n];
  intti::gauss_legendre<LD>(n, -1.0L, 1.0L, x, w);
  // degree-14 monomial integrated exactly, at long double precision
  LD s = 0;
  for (int i = 0; i < n; ++i)
    s += w[i] * std::pow(x[i], 14);
  const LD exact = 2.0L / 15.0L;
  EXPECT_LT(std::abs(s - exact) / exact, 1e-17L);
}

TEST(MultiPrec, BoysLongDouble) {
  // 30-digit mpmath references; long double (80-bit) eps is 1.1e-19
  struct Ref {
    int m;
    LD x, val;
  };
  const Ref refs[] = {
      {0, 0.1L, 0.967643312635591831L},
      {0, 10.0L, 0.280247390506642741L},
      {3, 1.0L, 0.0667322747768222568L},
      {3, 40.0L, 4.10521817607264513e-6L},
      {10, 30.0L, 1.75197494140663688e-10L},
      {10, 100.0L, 5.66639194474392784e-16L},
  };
  LD F[11];
  for (const auto &r : refs) {
    intti::ref::boys(r.m, r.x, F);
    EXPECT_LT(std::abs(F[r.m] - r.val) / r.val, 5e-18L) << "m=" << r.m;
  }
}

TEST(MultiPrec, QuartetBeyondDoublePrecision) {
  // 40-digit arithmetic shows the default Mobius grid converges to the
  // working-precision floor of the scalar type; in long double the golden
  // (ss|ss) quartet must therefore come out well below double precision
  const LD golden = 0.5625558912294552353350707L;
  intti::PrimitiveShell<LD> a{0.8L, {0.0L, 0.1L, -0.3L}, 0};
  intti::PrimitiveShell<LD> b{1.3L, {0.5L, -0.2L, 0.4L}, 0};
  intti::PrimitiveShell<LD> c{2.1L, {1.0L, 0.8L, 0.0L}, 0};
  intti::PrimitiveShell<LD> d{0.35L, {-0.4L, 0.3L, 1.1L}, 0};
  auto grid = intti::make_tgrid(intti::coulomb<LD>());
  LD val;
  intti::eri_quartet(intti::make_pair(a, b), intti::make_pair(c, d), grid, &val);
  EXPECT_LT(std::abs(val - golden) / golden, 5e-18L);
}

TEST(MultiPrec, AnalyticReferenceLongDouble) {
  // McMurchie-Davidson reference agrees with the quadrature at long double
  // precision for a full (pd|ps) quartet
  intti::PrimitiveShell<LD> a{0.8L, {0.0L, 0.1L, -0.3L}, 1};
  intti::PrimitiveShell<LD> b{1.3L, {0.5L, -0.2L, 0.4L}, 2};
  intti::PrimitiveShell<LD> c{2.1L, {1.0L, 0.8L, 0.0L}, 1};
  intti::PrimitiveShell<LD> d{0.35L, {-0.4L, 0.3L, 1.1L}, 0};
  const int nout = 3 * 6 * 3 * 1;
  std::vector<LD> quad(nout), ref(nout);
  auto grid = intti::make_tgrid(intti::coulomb<LD>());
  intti::eri_quartet(intti::make_pair(a, b), intti::make_pair(c, d), grid, quad.data());
  intti::ref::eri_analytic(a, b, c, d, ref.data());
  LD maxref = 0, maxdiff = 0;
  for (int k = 0; k < nout; ++k) {
    maxref = std::max(maxref, std::abs(ref[k]));
    maxdiff = std::max(maxdiff, std::abs(quad[k] - ref[k]));
  }
  EXPECT_LT(maxdiff / maxref, 1e-16L);
}

TEST(MultiPrec, DoubleMatchesLongDouble) {
  intti::PrimitiveShell<double> ad{0.8, {0.0, 0.1, -0.3}, 1};
  intti::PrimitiveShell<double> bd{1.3, {0.5, -0.2, 0.4}, 1};
  intti::PrimitiveShell<double> cd{2.1, {1.0, 0.8, 0.0}, 0};
  intti::PrimitiveShell<double> dd{0.35, {-0.4, 0.3, 1.1}, 0};
  intti::PrimitiveShell<LD> al{0.8L, {0.0L, 0.1L, -0.3L}, 1};
  intti::PrimitiveShell<LD> bl{1.3L, {0.5L, -0.2L, 0.4L}, 1};
  intti::PrimitiveShell<LD> cl{2.1L, {1.0L, 0.8L, 0.0L}, 0};
  intti::PrimitiveShell<LD> dl{0.35L, {-0.4L, 0.3L, 1.1L}, 0};
  double qd[9];
  LD ql[9];
  intti::eri_quartet(intti::make_pair(ad, bd), intti::make_pair(cd, dd),
                     intti::make_tgrid(intti::coulomb()), qd);
  intti::eri_quartet(intti::make_pair(al, bl), intti::make_pair(cl, dl),
                     intti::make_tgrid(intti::coulomb<LD>()), ql);
  for (int k = 0; k < 9; ++k)
    EXPECT_NEAR(qd[k], static_cast<double>(ql[k]), 1e-13 * std::abs(static_cast<double>(ql[0])) + 1e-15);
}

#ifdef INTTI_HAVE_QUADMATH
// Quadruple precision (__float128, 33 decimal digits). Kokkos cannot handle
// this type, so traits.hpp routes it to the serial host path.
//
// NOTE the input parsing: writing Q(0.8) would round 0.8 to a *double* first
// and then widen it, silently capping the whole calculation at double
// precision (this actually happened -- the first version of this test came
// out at 2e-16). Decimal inputs must be parsed straight into quad with
// strtoflt128.
TEST(MultiPrec, QuadruplePrecisionQuartet) {
  using Q = __float128;
  auto q = [](const char *s) { return strtoflt128(s, nullptr); };

  intti::PrimitiveShell<Q> a{q("0.8"), {q("0.0"), q("0.1"), q("-0.3")}, 0};
  intti::PrimitiveShell<Q> b{q("1.3"), {q("0.5"), q("-0.2"), q("0.4")}, 0};
  intti::PrimitiveShell<Q> c{q("2.1"), {q("1.0"), q("0.8"), q("0.0")}, 0};
  intti::PrimitiveShell<Q> d{q("0.35"), {q("-0.4"), q("0.3"), q("1.1")}, 0};
  // 40-digit mpmath reference of the golden (ss|ss) quartet
  const Q golden = q("0.5625558912294552353350706513510382177662");

  auto grid = intti::make_tgrid(intti::coulomb<Q>());
  Q val;
  intti::eri_quartet(intti::make_pair(a, b), intti::make_pair(c, d), grid, &val);
  const double err = static_cast<double>(fabsq((val - golden) / golden));

  // the Mobius grid converges to the working-precision floor of the scalar
  // type; in quad that is ~1e-33, so demand far beyond double AND beyond
  // long double (whose epsilon is 1.1e-19)
  EXPECT_LT(err, 1e-25) << "quad precision achieved only " << err;
}
#endif

} // namespace
