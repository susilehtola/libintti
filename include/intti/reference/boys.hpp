// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola
#pragma once

#include <cmath>
#include <limits>

#include "../math.hpp"

namespace intti::ref {

/// Boys function F_m(x) = int_0^1 u^{2m} exp(-x u^2) du for m = 0..mmax,
/// in the precision of Real. Reference implementation for tests and grid
/// generation; not performance critical.
template <class Real> void boys(int mmax, Real x, Real *F) {
  using std::erf;
  using std::exp;
  using std::log;
  using std::sqrt;
  if (x == Real(0)) {
    for (int m = 0; m <= mmax; ++m)
      F[m] = Real(1) / (2 * m + 1);
    return;
  }
  const Real pi = pi_v<Real>();
  const Real emx = exp(-x);
  // erf(sqrt(x)) = 1 - O(exp(-x)): the asymptotic branch is exact to eps
  // once exp(-x) < eps, i.e. x > -log(eps)
  const Real x_asym = -log(std::numeric_limits<Real>::epsilon()) + Real(2);
  if (x > x_asym) {
    F[0] = sqrt(pi / x) / 2; // erf(sqrt(x)) == 1 to working precision
    for (int m = 0; m < mmax; ++m)
      F[m + 1] = ((2 * m + 1) * F[m] - emx) / (2 * x);
    return;
  }
  // series at mmax: F_m = e^{-x} sum_k (2x)^k / ((2m+1)(2m+3)...(2m+2k+1));
  // all terms positive, no cancellation, converges for any x
  Real term = Real(1) / (2 * mmax + 1);
  Real sum = term;
  const Real cut = std::numeric_limits<Real>::epsilon() / 100;
  for (int k = 1; k < 100000; ++k) {
    term *= 2 * x / (2 * mmax + 2 * k + 1);
    sum += term;
    if (term < cut * sum) break;
  }
  F[mmax] = emx * sum;
  // downward recursion (stable)
  for (int m = mmax; m > 0; --m)
    F[m - 1] = (2 * x * F[m] + emx) / (2 * m - 1);
}

} // namespace intti::ref
