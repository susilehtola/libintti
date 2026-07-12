// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola
#pragma once

// GIAOs / London orbitals: integrals in a finite magnetic field.
//
// A London orbital carries a field-dependent plane-wave phase,
//
//   omega_a(r) = exp(-i/2 [B x (R_a - O)] . r) chi_a(r),
//
// with O the gauge origin. The bra-side product of two of them,
// omega_a^*(r) omega_b(r), therefore carries the phase exp(-i K_ab . r) with
//
//   K_ab = (1/2) B x (R_b - R_a),
//
// which depends only on the *difference* of the centres: the gauge origin
// cancels, so GIAO integrals are gauge-origin independent by construction
// (test_giao.cpp checks exactly this).
//
// Multiplying the Gaussian product exp(-p (r-P)^2) by that plane wave gives
//
//   exp(-p (r - Ptilde)^2) * exp(-i K.P - K^2/(4p)),   Ptilde = P - i K/(2p),
//
// i.e. a Gaussian with the SAME REAL EXPONENT p and a COMPLEX CENTRE. Every
// step of libintti's machinery -- the Gaussian product theorem, the
// McMurchie-Davidson E coefficients, the Hermite B_n recursion, the
// t quadrature -- is pure algebra in the centres, so it continues to complex
// centres unchanged: a GIAO ERI is just eri_quartet() instantiated on
// std::complex.
//
// This is a structural advantage of the quadrature route. The analytic route
// needs the Boys function of a *complex* argument, which is awkward; the
// t quadrature never forms a Boys function at all. Measured: the default
// 64-node Mobius grid reproduces analytic complex-Boys GIAO ERIs to 3e-15 at
// fields up to B = 5 a.u., with no extra nodes.

#include <cmath>
#include <complex>

#include "gto.hpp"
#include "math.hpp"
#include "traits.hpp"

namespace intti {

/// K_ab = (1/2) B x (R_b - R_a): the plane-wave vector of the London pair
/// product omega_a^* omega_b. Gauge-origin independent.
template <class Real>
void giao_k(const PrimitiveShell<Real> &a, const PrimitiveShell<Real> &b,
            const Real B[3], Real K[3]) {
  const Real d[3] = {b.center[0] - a.center[0], b.center[1] - a.center[1],
                     b.center[2] - a.center[2]};
  K[0] = Real(0.5) * (B[1] * d[2] - B[2] * d[1]);
  K[1] = Real(0.5) * (B[2] * d[0] - B[0] * d[2]);
  K[2] = Real(0.5) * (B[0] * d[1] - B[1] * d[0]);
}

/// London (GIAO) shell pair omega_a^*(r) omega_b(r) in magnetic field B.
/// The result is an ordinary ShellPair with a complex product centre, so it
/// can be handed straight to eri_quartet() and the one-electron drivers.
///
/// Note the conjugation convention: the FIRST shell is the conjugated (bra)
/// one. The ket pair of a two-electron integral uses the same function.
template <class Real>
ShellPair<std::complex<Real>> make_giao_pair(const PrimitiveShell<Real> &a,
                                             const PrimitiveShell<Real> &b,
                                             const Real B[3]) {
  using C = std::complex<Real>;
  static_assert(!is_complex_v<Real>, "make_giao_pair takes real shells");
  ShellPair<C> sp;
  sp.p = a.alpha + b.alpha;
  const Real mu = a.alpha * b.alpha / sp.p;
  Real K[3];
  giao_k(a, b, B, K);

  // real product centre and the field-free prefactor
  Real P[3], K0[3];
  for (int d = 0; d < 3; ++d) {
    P[d] = (a.alpha * a.center[d] + b.alpha * b.center[d]) / sp.p;
    const Real AB = a.center[d] - b.center[d];
    K0[d] = exp_(-mu * AB * AB);
  }

  // exp(-i K.r) shifts the centre into the complex plane and contributes
  //   exp(-i K.P - K^2/(4p)),
  // which factorises over the Cartesian directions.
  for (int d = 0; d < 3; ++d) {
    sp.P[d] = C(P[d], -K[d] / (2 * sp.p));
    const Real damp = exp_(-K[d] * K[d] / (4 * sp.p));
    const Real phase = -K[d] * P[d];
    sp.K[d] = C(K0[d] * damp) * C(std::cos(phase), std::sin(phase)); // exp(-i K_d P_d)
    sp.A[d] = a.center[d];
    sp.B[d] = b.center[d];
  }
  sp.la = a.l;
  sp.lb = b.l;
  return sp;
}

} // namespace intti
