// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola
#pragma once

#include <cmath>

namespace intti::ref {

/// Boys function F_m(x) = int_0^1 u^{2m} exp(-x u^2) du for m = 0..mmax.
/// Reference implementation for tests; not performance critical.
inline void boys(int mmax, double x, double *F) {
  if (x < 1e-13) {
    for (int m = 0; m <= mmax; ++m)
      F[m] = 1.0 / (2 * m + 1);
    return;
  }
  const double emx = std::exp(-x);
  if (x > 35.0) {
    // asymptotic value + upward recursion (stable for large x)
    F[0] = 0.5 * std::sqrt(M_PI / x) * std::erf(std::sqrt(x));
    for (int m = 0; m < mmax; ++m)
      F[m + 1] = ((2 * m + 1) * F[m] - emx) / (2.0 * x);
    return;
  }
  // series at mmax: F_m = e^{-x} sum_k (2x)^k / ((2m+1)(2m+3)...(2m+2k+1))
  double term = 1.0 / (2 * mmax + 1);
  double sum = term;
  for (int k = 1; k < 10000; ++k) {
    term *= 2.0 * x / (2 * mmax + 2 * k + 1);
    sum += term;
    if (term < 1e-17 * sum) break;
  }
  F[mmax] = emx * sum;
  // downward recursion (stable)
  for (int m = mmax; m > 0; --m)
    F[m - 1] = (2.0 * x * F[m] + emx) / (2 * m - 1);
}

} // namespace intti::ref
