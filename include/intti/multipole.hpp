// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Far-field (multipole) two-electron integrals -- the FMM far-field of the
// Coulomb build (M-MP). The exact McMurchie-Davidson ERI contracts the pair
// Hermite coefficients with the Hermite-Coulomb tensor R_{tuv}(alpha, R_PQ);
// for WELL-SEPARATED pairs (alpha |R_PQ|^2 large) that tensor tends to its
// exponent-free multipole limit
//     T_{tuv}(R) = d_x^t d_y^u d_z^v (1/|R|),
// built by the SAME Cartesian recursion seeded with the asymptotic Boys value
//     S_n = (-1)^n (2n-1)!! / |R|^{2n+1}.
// Because a Gaussian product has a FINITE, EXACT Hermite multipole expansion
// (t+u+v <= la+lb), the ONLY approximation is the Boys asymptotics, whose error
// decays as exp(-alpha |R_PQ|^2): machine precision by alpha R^2 ~ 32, and at
// large R more accurate than the t-grid Coulomb reference itself. The tensor is
// exponent-independent (shared across all primitives of a contracted pair) and
// needs no quadrature grid, so this path is precision-generic by construction.
//
// Like eri_quartet, the per-quartet routines here are INTERNAL / test-only; the
// matrix-level consumers (a near/far-split coulomb_build / nuclear build) are the
// public surface.

#include <cstddef>
#include <vector>

#include "gto.hpp"
#include "hermite1d.hpp"
#include "math.hpp"
#include "traits.hpp"

namespace intti {

/// Exponent-free Hermite multipole tensor T_{tuv} = d^t_x d^u_y d^v_z (1/|R|)
/// for t+u+v <= L, written row-major into T with stride (L+1) per index:
/// T[(t*(L+1)+u)*(L+1)+v]. R is the (real) separation P_bra - P_ket. Uses the
/// standard MD downward recursion seeded with S_n = (-1)^n (2n-1)!!/|R|^{2n+1}.
template <class Real>
void multipole_tensor(int L, const Real R[3], Real *T, Real *W) {
  const int D = L + 1;
  auto id4 = [=](int n, int t, int u, int v) {
    return ((n * D + t) * D + u) * D + v;
  };
  // scratch over the auxiliary order n (n + (t+u+v) <= L), caller-provided:
  // the boxed far field rebuilds this tensor once per (box, pair), so an
  // allocation here would dominate it.
  for (std::size_t i = 0; i < static_cast<std::size_t>(D) * D * D * D; ++i) W[i] = Real(0);
  const Real R2 = R[0] * R[0] + R[1] * R[1] + R[2] * R[2];
  Real dfac = 1, sign = 1, Rpow = sqrt_(R2); // R^{2n+1}
  for (int n = 0; n <= L; ++n) {
    W[id4(n, 0, 0, 0)] = sign * dfac / Rpow;
    sign = -sign;
    dfac *= Real(2 * n + 1);
    Rpow *= R2;
  }
  for (int o = 1; o <= L; ++o)
    for (int n = 0; n <= L - o; ++n)
      for (int t = 0; t <= o; ++t)
        for (int u = 0; u <= o - t; ++u) {
          const int v = o - t - u;
          Real val;
          if (t > 0) {
            val = R[0] * W[id4(n + 1, t - 1, u, v)];
            if (t > 1) val += Real(t - 1) * W[id4(n + 1, t - 2, u, v)];
          } else if (u > 0) {
            val = R[1] * W[id4(n + 1, t, u - 1, v)];
            if (u > 1) val += Real(u - 1) * W[id4(n + 1, t, u - 2, v)];
          } else {
            val = R[2] * W[id4(n + 1, t, u, v - 1)];
            if (v > 1) val += Real(v - 1) * W[id4(n + 1, t, u, v - 2)];
          }
          W[id4(n, t, u, v)] = val;
        }
  const int D3 = D * D * D;
  for (int i = 0; i < D3; ++i) T[i] = W[i]; // n = 0 block
}

/// multipole_tensor with the scratch allocated internally.
template <class Real> void multipole_tensor(int L, const Real R[3], Real *T) {
  const int D = L + 1;
  std::vector<Real> W(static_cast<std::size_t>(D) * D * D * D, Real(0));
  multipole_tensor(L, R, T, W.data());
}

namespace detail {

/// Hermite multipole moments of a shell pair: E[(ka*nb+kb), tx, ty, tz] =
/// product of the three 1D e_coeffs (which already carry the pair prefactor K).
/// nt = la+lb+1 is the per-axis Hermite extent. Real-centre pairs only.
template <class Scalar>
void pair_hermite_moments(const ShellPair<Scalar> &sp, std::vector<Scalar> &E,
                          int &nt) {
  nt = sp.la + sp.lb + 1;
  std::vector<Scalar> Ed[3];
  for (int d = 0; d < 3; ++d) {
    Ed[d].assign((sp.la + 1) * (sp.lb + 1) * nt, Scalar(0));
    e_coeffs(sp.la, sp.lb, sp.p, Scalar(sp.P[d] - sp.A[d]),
             Scalar(sp.P[d] - sp.B[d]), sp.K[d], Ed[d].data());
  }
  const int na = ncart(sp.la), nb = ncart(sp.lb);
  E.assign(static_cast<std::size_t>(na) * nb * nt * nt * nt, Scalar(0));
  for (int ka = 0; ka < na; ++ka) {
    int ax, ay, az;
    cart_comp(sp.la, ka, ax, ay, az);
    for (int kb = 0; kb < nb; ++kb) {
      int bx, by, bz;
      cart_comp(sp.lb, kb, bx, by, bz);
      auto ex = [&](int d, int i, int j, int t) {
        return Ed[d][(i * (sp.lb + 1) + j) * nt + t];
      };
      for (int tx = 0; tx < nt; ++tx)
        for (int ty = 0; ty < nt; ++ty)
          for (int tz = 0; tz < nt; ++tz)
            E[(((static_cast<std::size_t>(ka * nb + kb)) * nt + tx) * nt + ty) *
                  nt +
              tz] = ex(0, ax, bx, tx) * ex(1, ay, by, ty) * ex(2, az, bz, tz);
    }
  }
}

} // namespace detail

/// Far-field ERI (ab|cd) for well-separated bra/ket pairs, exponent-free and
/// grid-free. out is ncart(la)*ncart(lb)*ncart(lc)*ncart(ld) values, laid out
/// as eri_quartet's. INTERNAL / test-only -- validated to converge to
/// eri_quartet as the separation grows (error ~ exp(-alpha |R_PQ|^2)). Real
/// centres only (the multipole tensor is real); GIAO/complex deferred.
template <class Scalar>
void eri_quartet_farfield(const ShellPair<Scalar> &bra,
                          const ShellPair<Scalar> &ket, Scalar *out) {
  using Real = real_t<Scalar>;
  static_assert(!is_complex_v<Scalar>, "eri_quartet_farfield: real centres only");
  const int la = bra.la, lb = bra.lb, lc = ket.la, ld = ket.lb;
  const int L = la + lb + lc + ld;
  std::vector<Scalar> Eb, Ek;
  int ntb, ntk;
  detail::pair_hermite_moments(bra, Eb, ntb);
  detail::pair_hermite_moments(ket, Ek, ntk);
  Real R[3];
  for (int d = 0; d < 3; ++d) R[d] = Real(bra.P[d]) - Real(ket.P[d]);
  std::vector<Real> T;
  T.assign(static_cast<std::size_t>((L + 1)) * (L + 1) * (L + 1), Real(0));
  multipole_tensor(L, R, T.data());
  const int Dt = L + 1;
  auto Ti = [&](int t, int u, int v) { return T[(t * Dt + u) * Dt + v]; };
  const Real pi = pi_v<Real>();
  const Real pq = bra.p * ket.p;
  const Scalar pref = (pi * pi * pi) / (pq * sqrt_(pq)); // pi^3 / (p q)^{3/2}
  const int nba = ncart(la), nbb = ncart(lb), nkc = ncart(lc), nkd = ncart(ld);
  for (int ka = 0; ka < nba; ++ka)
    for (int kb = 0; kb < nbb; ++kb)
      for (int kc = 0; kc < nkc; ++kc)
        for (int kd = 0; kd < nkd; ++kd) {
          Scalar ff = 0;
          for (int tx = 0; tx < ntb; ++tx)
            for (int ty = 0; ty < ntb; ++ty)
              for (int tz = 0; tz < ntb; ++tz) {
                const Scalar qb =
                    Eb[(((static_cast<std::size_t>(ka * nbb + kb)) * ntb + tx) *
                            ntb +
                        ty) *
                           ntb +
                       tz];
                if (qb == Scalar(0)) continue;
                for (int sx = 0; sx < ntk; ++sx)
                  for (int sy = 0; sy < ntk; ++sy)
                    for (int sz = 0; sz < ntk; ++sz) {
                      const Scalar qk =
                          Ek[(((static_cast<std::size_t>(kc * nkd + kd)) * ntk +
                               sx) *
                                  ntk +
                              sy) *
                                 ntk +
                             sz];
                      if (qk == Scalar(0)) continue;
                      const Real sgn = ((sx + sy + sz) & 1) ? Real(-1) : Real(1);
                      ff += qb * qk * sgn * Ti(tx + sx, ty + sy, tz + sz);
                    }
              }
          out[((ka * nbb + kb) * nkc + kc) * nkd + kd] = pref * ff;
        }
}

} // namespace intti
