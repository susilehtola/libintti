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

#include <array>
#include <cmath>
#include <complex>
#include <vector>

#include "gto.hpp"
#include "hermite1d.hpp"
#include "math.hpp"
#include "oneel.hpp"
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

/// Complex GIAO overlap matrix S(B) = <omega_mu | omega_nu> in a finite
/// magnetic field B (nao x nao, row-major, over the primitive Cartesian AOs).
/// London orbitals shift the product Gaussian centre into the complex plane
/// (make_giao_pair), so the ordinary MD E-coefficient overlap runs unchanged
/// on complex scalars. At B = 0 this reduces to the real overlap_matrix.
template <class Real>
std::vector<std::complex<Real>> giao_overlap(const ShellBasis<Real> &basis,
                                             const Real B[3]) {
  using C = std::complex<Real>;
  static_assert(!is_complex_v<Real>, "giao_overlap takes a real basis");
  const int nao = basis.nao;
  std::vector<C> S(static_cast<std::size_t>(nao) * nao, C(0));
  const int ns = static_cast<int>(basis.shells.size());
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const auto sp = make_giao_pair(sa, sb, B); // complex centre + prefactor
      const int la = sa.l, lb = sb.l, lb1 = lb + 1, nt = la + lb + 1;
      std::vector<C> sf[3];
      for (int d = 0; d < 3; ++d) {
        std::vector<C> E(static_cast<std::size_t>(la + 1) * (lb + 1) * nt);
        const C PA = sp.P[d] - C(sa.center[d]);
        const C PB = sp.P[d] - C(sb.center[d]);
        e_coeffs(la, lb, sp.p, PA, PB, sp.K[d], E.data());
        const Real pref = sqrt_(pi_v<Real>() / sp.p);
        sf[d].assign(static_cast<std::size_t>(la + 1) * lb1, C(0));
        for (int i = 0; i <= la; ++i)
          for (int j = 0; j <= lb; ++j)
            sf[d][i * lb1 + j] = C(pref) * E[(i * (lb + 1) + j) * nt + 0];
      }
      for (int ka = 0; ka < ncart(la); ++ka) {
        int a3[3];
        cart_comp(la, ka, a3[0], a3[1], a3[2]);
        for (int kb = 0; kb < ncart(lb); ++kb) {
          int b3[3];
          cart_comp(lb, kb, b3[0], b3[1], b3[2]);
          S[(basis.ao_off[a] + ka) * static_cast<std::size_t>(nao) + basis.ao_off[b] + kb] =
              sf[0][a3[0] * lb1 + b3[0]] * sf[1][a3[1] * lb1 + b3[1]] *
              sf[2][a3[2] * lb1 + b3[2]];
        }
      }
    }
  return S;
}

/// Analytic magnetic-field derivative dS/dB_k of the GIAO overlap at B = 0.
///
/// The London pair omega_mu^* omega_nu carries the phase
/// exp(i/2 [B x (R_mu - R_nu)].r), so
///   dS_munu/dB_k |_{B=0} = (i/2) [e_k x (R_mu - R_nu)] . <mu| r |nu>,
/// a purely imaginary combination of the dipole moment matrices about the
/// coordinate origin (the gauge origin cancels between bra and ket). Returns
/// the three components {dS/dB_x, dS/dB_y, dS/dB_z}, each nao x nao row-major.
template <class Real>
std::array<std::vector<std::complex<Real>>, 3>
giao_overlap_dB(const ShellBasis<Real> &basis) {
  using C = std::complex<Real>;
  static_assert(!is_complex_v<Real>, "giao_overlap_dB takes a real basis");
  const int nao = basis.nao;
  std::array<std::vector<C>, 3> dS;
  for (auto &m : dS) m.assign(static_cast<std::size_t>(nao) * nao, C(0));
  const int ns = static_cast<int>(basis.shells.size());
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const int la = sa.l, lb = sb.l, lb1 = lb + 1;
      std::vector<Real> Sf[3], Df[3]; // 1D overlap and dipole <i|x_d|j> about 0
      for (int d = 0; d < 3; ++d) {
        std::vector<Real> s;
        int lbx;
        detail::overlap_1d(sa.alpha, sa.center[d], sb.alpha, sb.center[d], la, lb, 1, 0,
                           s, lbx);
        Sf[d].assign(static_cast<std::size_t>(la + 1) * lb1, Real(0));
        for (int i = 0; i <= la; ++i)
          for (int j = 0; j <= lb; ++j) Sf[d][i * lb1 + j] = s[i * (lbx + 1) + j];
        std::vector<Real> m1;
        detail::multipole_1d(s, lbx, la, lb, sa.center[d], Real(0), 1, m1);
        Df[d].assign(static_cast<std::size_t>(la + 1) * lb1, Real(0));
        for (int i = 0; i <= la; ++i)
          for (int j = 0; j <= lb; ++j)
            Df[d][i * lb1 + j] = m1[(1 * (la + 1) + i) * lb1 + j]; // e=1 block
      }
      const Real dv[3] = {sa.center[0] - sb.center[0], sa.center[1] - sb.center[1],
                          sa.center[2] - sb.center[2]};
      for (int ka = 0; ka < ncart(la); ++ka) {
        int a3[3];
        cart_comp(la, ka, a3[0], a3[1], a3[2]);
        for (int kb = 0; kb < ncart(lb); ++kb) {
          int b3[3];
          cart_comp(lb, kb, b3[0], b3[1], b3[2]);
          auto S = [&](int d) { return Sf[d][a3[d] * lb1 + b3[d]]; };
          // <mu| x_d |nu> = dipole in direction d times overlap in the others
          const Real D[3] = {Df[0][a3[0] * lb1 + b3[0]] * S(1) * S(2),
                             S(0) * Df[1][a3[1] * lb1 + b3[1]] * S(2),
                             S(0) * S(1) * Df[2][a3[2] * lb1 + b3[2]]};
          const std::size_t idx =
              (basis.ao_off[a] + ka) * static_cast<std::size_t>(nao) + basis.ao_off[b] + kb;
          const C half_i(Real(0), Real(0.5));
          dS[0][idx] = half_i * (-dv[2] * D[1] + dv[1] * D[2]);
          dS[1][idx] = half_i * (dv[2] * D[0] - dv[0] * D[2]);
          dS[2][idx] = half_i * (-dv[1] * D[0] + dv[0] * D[1]);
        }
      }
    }
  return dS;
}

} // namespace intti
