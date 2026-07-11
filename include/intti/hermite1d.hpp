// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola
#pragma once

#include <Kokkos_Core.hpp>

namespace intti {

/// McMurchie-Davidson expansion coefficients E_t^{ij} of a 1D Gaussian pair
/// in Hermite Gaussians Lambda_t(x; p, P) = (d/dP)^t exp(-p (x-P)^2):
///   (x-A)^i exp(-a(x-A)^2) (x-B)^j exp(-b(x-B)^2) = sum_t E_t^{ij} Lambda_t.
///
/// E must hold (la+1)*(lb+1)*(la+lb+1) doubles, indexed
/// E[(i*(lb+1)+j)*(la+lb+1)+t] with t = 0..i+j. The Gaussian-product
/// prefactor K = exp(-a b/p (A-B)^2) is folded into E_0^{00}.
KOKKOS_INLINE_FUNCTION
void e_coeffs(int la, int lb, double p, double PA, double PB, double K, double *E) {
  const int nt = la + lb + 1;
  for (int i = 0; i < (la + 1) * (lb + 1) * nt; ++i)
    E[i] = 0.0;
  auto at = [=](double *buf, int i, int j, int t) -> double & {
    return buf[(i * (lb + 1) + j) * nt + t];
  };
  at(E, 0, 0, 0) = K;
  const double o2p = 1.0 / (2.0 * p);
  for (int i = 0; i < la; ++i)
    for (int t = 0; t <= i + 1; ++t) {
      double v = PA * at(E, i, 0, t);
      if (t > 0) v += o2p * at(E, i, 0, t - 1);
      if (t + 1 <= i) v += (t + 1) * at(E, i, 0, t + 1);
      at(E, i + 1, 0, t) = v;
    }
  for (int i = 0; i <= la; ++i)
    for (int j = 0; j < lb; ++j)
      for (int t = 0; t <= i + j + 1; ++t) {
        double v = PB * at(E, i, j, t);
        if (t > 0) v += o2p * at(E, i, j, t - 1);
        if (t + 1 <= i + j) v += (t + 1) * at(E, i, j, t + 1);
        at(E, i, j + 1, t) = v;
      }
}

/// Derivatives of a Gaussian: B_n = (d/dX)^n exp(-theta X^2), n = 0..nmax.
/// These carry the Hermite-index dependence of the two-pair interaction at
/// fixed quadrature node t: with X = P - Q, D = pq + t^2(p+q) and
/// theta = t^2 p q / D, the 1D bra(t)-ket(tau) matrix element is
///   J_{t,tau} = (-1)^tau (pi/sqrt(D)) B_{t+tau}(theta, X).
KOKKOS_INLINE_FUNCTION
void hermite_b(int nmax, double theta, double X, double *B) {
  B[0] = Kokkos::exp(-theta * X * X);
  if (nmax >= 1) B[1] = -2.0 * theta * X * B[0];
  for (int n = 1; n < nmax; ++n)
    B[n + 1] = -2.0 * theta * (X * B[n] + n * B[n - 1]);
}

} // namespace intti
