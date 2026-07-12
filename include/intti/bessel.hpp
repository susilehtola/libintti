// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola
#pragma once

#include <limits>

#include <Kokkos_Core.hpp>

#include "gto.hpp" // LMAX
#include "math.hpp"

namespace intti {

/// Exponentially scaled modified Bessel functions of the first kind,
///   out[m] = ive(m, x) = e^{-x} I_m(x),  m = 0..mmax,  x >= 0,
/// in the precision of Real. These carry the azimuthal coupling of coaxial
/// products at each t node: the phi integrals of the Gaussian kernel give
/// (2 pi)^2 delta_{m,-m'} I_m(2 t^2 rho1 rho2), evaluated stably in scaled
/// form with exp(-t^2 (rho1 - rho2)^2) absorbed by the caller.
///
/// mmax is bounded by the exact azimuthal truncation of orbital products
/// (m <= l_a + l_b <= 2 LMAX). For x below the series limit the positive-term
/// power series is summed per m (no cancellation); for large x the uniform
/// asymptotic expansion seeds m = 0, 1 and the ladder goes upward, which is
/// stable here because the contamination growth K_m/K_0 ~ e^{m^2/2x} stays
/// O(1) when m^2 << x.
template <class Real>
KOKKOS_INLINE_FUNCTION void ive_ladder(int mmax, Real x, Real *out) {
  if (x == Real(0)) {
    out[0] = 1;
    for (int m = 1; m <= mmax; ++m)
      out[m] = 0;
    return;
  }
  const Real eps = std::numeric_limits<Real>::epsilon();
  // series limit: the unscaled positive-term sum peaks near e^x; keep it
  // far from overflow for the widest supported type
  const Real series_limit = Real(200);
  if (x <= series_limit) {
    const Real emx = exp_(-x);
    const Real x2 = x / 2;
    const Real q = x2 * x2;
    for (int m = 0; m <= mmax; ++m) {
      // (x/2)^m / m!
      Real term = emx;
      for (int j = 1; j <= m; ++j)
        term *= x2 / j;
      Real sum = term;
      for (int k = 1; k < 100000; ++k) {
        term *= q / (Real(k) * Real(k + m));
        sum += term;
        if (term < eps * sum) break;
      }
      out[m] = sum;
    }
    return;
  }
  // large x: asymptotic ive(m, x) ~ (2 pi x)^{-1/2} sum_k (-)^k a_k(mu)/(8x)^k,
  // mu = 4 m^2, a_k = (mu-1)(mu-9)...(mu-(2k-1)^2)/k!; sum to the smallest term
  const Real pi = Real(3.14159265358979323846264338327950288L);
  auto asym = [&](int m) {
    const Real mu = Real(4 * m * m);
    Real term = 1, sum = 1;
    Real prev = std::numeric_limits<Real>::max();
    for (int k = 1; k < 60; ++k) {
      const Real f = (mu - Real((2 * k - 1) * (2 * k - 1))) / (Real(8 * k) * x);
      term *= -f;
      const Real mag = term < 0 ? -term : term;
      if (mag >= prev) break; // asymptotic series: stop at the smallest term
      sum += term;
      if (mag < eps * (sum < 0 ? -sum : sum)) break;
      prev = mag;
    }
    return sum / sqrt_(2 * pi * x);
  };
  out[0] = asym(0);
  if (mmax >= 1) out[1] = asym(1);
  for (int m = 1; m < mmax; ++m)
    out[m + 1] = out[m - 1] - (Real(2 * m) / x) * out[m];
  return;
}

} // namespace intti
