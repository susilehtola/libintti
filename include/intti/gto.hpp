// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola
#pragma once

#include <Kokkos_Core.hpp>

#include "math.hpp"
#include "traits.hpp"

namespace intti {

/// Maximum angular momentum per shell supported by fixed-size buffers.
inline constexpr int LMAX = 8;

/// Number of Cartesian components of angular momentum l.
KOKKOS_INLINE_FUNCTION constexpr int ncart(int l) { return (l + 1) * (l + 2) / 2; }

/// k-th Cartesian component (lx, ly, lz) of shell l; lx runs from l down to 0,
/// then ly from l-lx down to 0 (libcint ordering: xx, xy, xz, yy, yz, zz).
KOKKOS_INLINE_FUNCTION void cart_comp(int l, int k, int &lx, int &ly, int &lz) {
  int idx = 0;
  for (int ax = l; ax >= 0; --ax)
    for (int ay = l - ax; ay >= 0; --ay) {
      if (idx == k) {
        lx = ax;
        ly = ay;
        lz = l - ax - ay;
        return;
      }
      ++idx;
    }
  lx = ly = lz = 0;
}

/// Unnormalized primitive Cartesian Gaussian shell:
///   (x-Ax)^lx (y-Ay)^ly (z-Az)^lz exp(-alpha |r-A|^2)  for all lx+ly+lz = l.
template <class Real = double> struct PrimitiveShell {
  Real alpha;     ///< exponent
  Real center[3]; ///< A
  int l;          ///< angular momentum
};

/// Gaussian-product-theorem data for a shell pair. This is the reuse unit of
/// the library: per-(pair, t) intermediates are shared across all quartets in
/// later J/K builds, so all integral drivers consume ShellPairs, never shells.
///
/// The exponent p and the shell centres A, B are always REAL. The product
/// centre P and the prefactor K carry the scalar type, which is complex for
/// GIAOs (giao.hpp): multiplying a Gaussian by the London plane wave shifts
/// its centre into the complex plane but leaves its exponent alone. Every
/// downstream quantity built from exponents only (D, theta, rho, the
/// pi/sqrt(D) prefactors) therefore stays real.
template <class Scalar = double> struct ShellPair {
  using R = real_t<Scalar>;
  R p;         ///< alpha + beta
  Scalar P[3]; ///< product centre (alpha A + beta B)/p, complex for GIAOs
  Scalar K[3]; ///< per-direction prefactor, complex for GIAOs
  R A[3], B[3];
  int la, lb;
};

/// Value of one Cartesian component (ka, kb) of a pair product at point r,
/// via the Gaussian product theorem (only ShellPair data needed):
///   prod_d (r_d-A_d)^{a_d} (r_d-B_d)^{b_d} K_d exp(-p (r_d-P_d)^2).
template <class Scalar>
KOKKOS_INLINE_FUNCTION Scalar pair_component_value(const ShellPair<Scalar> &sp, int ka,
                                                   int kb, const real_t<Scalar> *r) {
  using R = real_t<Scalar>;
  int a3[3], b3[3];
  cart_comp(sp.la, ka, a3[0], a3[1], a3[2]);
  cart_comp(sp.lb, kb, b3[0], b3[1], b3[2]);
  Scalar val = Scalar(1);
  for (int d = 0; d < 3; ++d) {
    const R dA = r[d] - sp.A[d], dB = r[d] - sp.B[d];
    const Scalar dP = r[d] - sp.P[d];
    Scalar f = sp.K[d] * exp_(Scalar(-sp.p * dP * dP));
    for (int j = 0; j < a3[d]; ++j)
      f *= dA;
    for (int j = 0; j < b3[d]; ++j)
      f *= dB;
    val *= f;
  }
  return val;
}

template <class Real>
ShellPair<Real> make_pair(const PrimitiveShell<Real> &a, const PrimitiveShell<Real> &b) {
  static_assert(!is_complex_v<Real>, "make_pair takes real shells; see make_giao_pair");
  ShellPair<Real> sp;
  sp.p = a.alpha + b.alpha;
  const Real mu = a.alpha * b.alpha / sp.p;
  for (int d = 0; d < 3; ++d) {
    sp.P[d] = (a.alpha * a.center[d] + b.alpha * b.center[d]) / sp.p;
    const Real AB = a.center[d] - b.center[d];
    sp.K[d] = exp_(-mu * AB * AB);
    sp.A[d] = a.center[d];
    sp.B[d] = b.center[d];
  }
  sp.la = a.l;
  sp.lb = b.l;
  return sp;
}

} // namespace intti
