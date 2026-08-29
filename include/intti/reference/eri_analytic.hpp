// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "../gto.hpp"
#include "../hermite1d.hpp"
#include "boys.hpp"

namespace intti::ref {

/// Analytic Coulomb ERI quartet (ab|cd) for unnormalized primitive Cartesian
/// shells, by the McMurchie-Davidson scheme, in the precision of Real.
/// Reference for tests only; no performance considerations. out is ordered
/// like eri_quartet().
template <class Real>
void eri_analytic(const PrimitiveShell<Real> &a, const PrimitiveShell<Real> &b,
                  const PrimitiveShell<Real> &c, const PrimitiveShell<Real> &d,
                  Real *out) {
  using std::exp;
  using std::sqrt;
  const int la = a.l, lb = b.l, lc = c.l, ld = d.l;
  const Real p = a.alpha + b.alpha, q = c.alpha + d.alpha;
  const Real rho = p * q / (p + q);
  const Real pi = pi_v<Real>();
  Real P[3], Q[3], PQ[3];
  Real T{};
  for (int i = 0; i < 3; ++i) {
    P[i] = (a.alpha * a.center[i] + b.alpha * b.center[i]) / p;
    Q[i] = (c.alpha * c.center[i] + d.alpha * d.center[i]) / q;
    PQ[i] = P[i] - Q[i];
    T += PQ[i] * PQ[i];
  }
  T *= rho;

  const int N = la + lb + lc + ld;
  std::vector<Real> F(N + 1);
  boys(N, T, F.data());

  // Hermite Coulomb integrals R^{(n)}_{tuv}
  const int M = N + 1;
  std::vector<Real> R(static_cast<std::size_t>(M) * M * M * M, Real(0));
  auto r = [&](int n, int t, int u, int v) -> Real & {
    return R[((static_cast<std::size_t>(n) * M + t) * M + u) * M + v];
  };
  Real m2rho_n = 1; // (-2 rho)^n
  for (int n = 0; n <= N; ++n) {
    r(n, 0, 0, 0) = m2rho_n * F[n];
    m2rho_n *= -2 * rho;
  }
  for (int n = N - 1; n >= 0; --n)
    for (int t = 0; t <= N - n; ++t)
      for (int u = 0; u <= N - n - t; ++u)
        for (int v = 0; v <= N - n - t - u; ++v) {
          if (t + 1 <= N - n)
            r(n, t + 1, u, v) = (t > 0 ? Real(t) * r(n + 1, t - 1, u, v) : Real(0)) +
                                PQ[0] * r(n + 1, t, u, v);
          if (u + 1 <= N - n)
            r(n, t, u + 1, v) = (u > 0 ? Real(u) * r(n + 1, t, u - 1, v) : Real(0)) +
                                PQ[1] * r(n + 1, t, u, v);
          if (v + 1 <= N - n)
            r(n, t, u, v + 1) = (v > 0 ? Real(v) * r(n + 1, t, u, v - 1) : Real(0)) +
                                PQ[2] * r(n + 1, t, u, v);
        }

  // E coefficients per direction
  std::vector<Real> Eb[3], Ek[3];
  for (int dd = 0; dd < 3; ++dd) {
    Eb[dd].resize((la + 1) * (lb + 1) * (la + lb + 1));
    Ek[dd].resize((lc + 1) * (ld + 1) * (lc + ld + 1));
    const Real ABd = a.center[dd] - b.center[dd];
    const Real CDd = c.center[dd] - d.center[dd];
    e_coeffs(la, lb, p, P[dd] - a.center[dd], P[dd] - b.center[dd],
             exp(-a.alpha * b.alpha / p * ABd * ABd), Eb[dd].data());
    e_coeffs(lc, ld, q, Q[dd] - c.center[dd], Q[dd] - d.center[dd],
             exp(-c.alpha * d.alpha / q * CDd * CDd), Ek[dd].data());
  }
  auto eb = [&](int dd, int i, int j, int t) {
    return Eb[dd][(i * (lb + 1) + j) * (la + lb + 1) + t];
  };
  auto ek = [&](int dd, int i, int j, int t) {
    return Ek[dd][(i * (ld + 1) + j) * (lc + ld + 1) + t];
  };

  const Real pref = 2 * pi * pi * sqrt(pi) / (p * q * sqrt(p + q));
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
          Real val{};
          for (int t = 0; t <= a3[0] + b3[0]; ++t)
            for (int u = 0; u <= a3[1] + b3[1]; ++u)
              for (int v = 0; v <= a3[2] + b3[2]; ++v) {
                const Real ebp = eb(0, a3[0], b3[0], t) * eb(1, a3[1], b3[1], u) *
                                 eb(2, a3[2], b3[2], v);
                if (ebp == Real(0)) continue;
                for (int tt = 0; tt <= c3[0] + d3[0]; ++tt)
                  for (int uu = 0; uu <= c3[1] + d3[1]; ++uu)
                    for (int vv = 0; vv <= c3[2] + d3[2]; ++vv) {
                      const Real ekp = ek(0, c3[0], d3[0], tt) *
                                       ek(1, c3[1], d3[1], uu) *
                                       ek(2, c3[2], d3[2], vv);
                      if (ekp == Real(0)) continue;
                      const Real contrib = ebp * ekp * r(0, t + tt, u + uu, v + vv);
                      val += (tt + uu + vv) % 2 ? -contrib : contrib;
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
