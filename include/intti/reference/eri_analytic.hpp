// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola
#pragma once

#include <cmath>
#include <vector>

#include "../gto.hpp"
#include "../hermite1d.hpp"
#include "boys.hpp"

namespace intti::ref {

/// Analytic Coulomb ERI quartet (ab|cd) for unnormalized primitive Cartesian
/// shells, by the McMurchie-Davidson scheme. Reference for tests only; no
/// performance considerations. out is ordered like eri_quartet().
inline void eri_analytic(const PrimitiveShell &a, const PrimitiveShell &b,
                         const PrimitiveShell &c, const PrimitiveShell &d,
                         double *out) {
  const int la = a.l, lb = b.l, lc = c.l, ld = d.l;
  const double p = a.alpha + b.alpha, q = c.alpha + d.alpha;
  const double rho = p * q / (p + q);
  double P[3], Q[3], PQ[3];
  double T = 0.0;
  for (int i = 0; i < 3; ++i) {
    P[i] = (a.alpha * a.center[i] + b.alpha * b.center[i]) / p;
    Q[i] = (c.alpha * c.center[i] + d.alpha * d.center[i]) / q;
    PQ[i] = P[i] - Q[i];
    T += PQ[i] * PQ[i];
  }
  T *= rho;

  const int N = la + lb + lc + ld;
  std::vector<double> F(N + 1);
  boys(N, T, F.data());

  // Hermite Coulomb integrals R^{(n)}_{tuv}
  const int M = N + 1;
  std::vector<double> R(static_cast<std::size_t>(M) * M * M * M, 0.0);
  auto r = [&](int n, int t, int u, int v) -> double & {
    return R[((static_cast<std::size_t>(n) * M + t) * M + u) * M + v];
  };
  for (int n = 0; n <= N; ++n)
    r(n, 0, 0, 0) = std::pow(-2.0 * rho, n) * F[n];
  for (int n = N - 1; n >= 0; --n)
    for (int t = 0; t <= N - n; ++t)
      for (int u = 0; u <= N - n - t; ++u)
        for (int v = 0; v <= N - n - t - u; ++v) {
          if (t + 1 <= N - n)
            r(n, t + 1, u, v) = (t > 0 ? t * r(n + 1, t - 1, u, v) : 0.0) +
                                PQ[0] * r(n + 1, t, u, v);
          if (u + 1 <= N - n)
            r(n, t, u + 1, v) = (u > 0 ? u * r(n + 1, t, u - 1, v) : 0.0) +
                                PQ[1] * r(n + 1, t, u, v);
          if (v + 1 <= N - n)
            r(n, t, u, v + 1) = (v > 0 ? v * r(n + 1, t, u, v - 1) : 0.0) +
                                PQ[2] * r(n + 1, t, u, v);
        }

  // E coefficients per direction
  std::vector<double> Eb[3], Ek[3];
  for (int dd = 0; dd < 3; ++dd) {
    Eb[dd].resize((la + 1) * (lb + 1) * (la + lb + 1));
    Ek[dd].resize((lc + 1) * (ld + 1) * (lc + ld + 1));
    const double ABd = a.center[dd] - b.center[dd];
    const double CDd = c.center[dd] - d.center[dd];
    e_coeffs(la, lb, p, P[dd] - a.center[dd], P[dd] - b.center[dd],
             std::exp(-a.alpha * b.alpha / p * ABd * ABd), Eb[dd].data());
    e_coeffs(lc, ld, q, Q[dd] - c.center[dd], Q[dd] - d.center[dd],
             std::exp(-c.alpha * d.alpha / q * CDd * CDd), Ek[dd].data());
  }
  auto eb = [&](int dd, int i, int j, int t) {
    return Eb[dd][(i * (lb + 1) + j) * (la + lb + 1) + t];
  };
  auto ek = [&](int dd, int i, int j, int t) {
    return Ek[dd][(i * (ld + 1) + j) * (lc + ld + 1) + t];
  };

  const double pref = 2.0 * std::pow(M_PI, 2.5) / (p * q * std::sqrt(p + q));
  for (int ka = 0; ka < ncart(la); ++ka) {
    int a3[3];
    cart_comp(la, ka, a3[0], a3[1], a3[2]);
    for (int kb = 0; kb < ncart(lb); ++kb) {
      int b3[3];
      cart_comp(lb, kb, b3[0], b3[1], b3[2]);
      for (int kc = 0; kc < ncart(lc); ++kc) {
        int c3[3];
        cart_comp(lc, kc, c3[0], c3[1], c3[2]);
        for (int kd = 0; kd < ncart(ld); ++kd) {
          int d3[3];
          cart_comp(ld, kd, d3[0], d3[1], d3[2]);
          double val = 0.0;
          for (int t = 0; t <= a3[0] + b3[0]; ++t)
            for (int u = 0; u <= a3[1] + b3[1]; ++u)
              for (int v = 0; v <= a3[2] + b3[2]; ++v) {
                const double ebp = eb(0, a3[0], b3[0], t) * eb(1, a3[1], b3[1], u) *
                                   eb(2, a3[2], b3[2], v);
                if (ebp == 0.0) continue;
                for (int tt = 0; tt <= c3[0] + d3[0]; ++tt)
                  for (int uu = 0; uu <= c3[1] + d3[1]; ++uu)
                    for (int vv = 0; vv <= c3[2] + d3[2]; ++vv) {
                      const double ekp = ek(0, c3[0], d3[0], tt) *
                                         ek(1, c3[1], d3[1], uu) *
                                         ek(2, c3[2], d3[2], vv);
                      if (ekp == 0.0) continue;
                      val += ebp * ekp * ((tt + uu + vv) % 2 ? -1.0 : 1.0) *
                             r(0, t + tt, u + uu, v + vv);
                    }
              }
          const int k = ((ka * ncart(lb) + kb) * ncart(lc) + kc) * ncart(ld) + kd;
          out[k] = pref * val;
        }
      }
    }
  }
}

} // namespace intti::ref
