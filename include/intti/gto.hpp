// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola
#pragma once

#include <cmath>

#include <Kokkos_Core.hpp>

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
struct PrimitiveShell {
  double alpha;     ///< exponent
  double center[3]; ///< A
  int l;            ///< angular momentum
};

/// Gaussian-product-theorem data for a shell pair. This is the reuse unit of
/// the library: per-(pair, t) intermediates are shared across all quartets in
/// later J/K builds, so all integral drivers consume ShellPairs, never shells.
struct ShellPair {
  double p;    ///< alpha + beta
  double P[3]; ///< product center (alpha A + beta B)/p
  double K[3]; ///< per-direction prefactor exp(-alpha beta/p (A_d - B_d)^2)
  double A[3], B[3];
  int la, lb;
};

inline ShellPair make_pair(const PrimitiveShell &a, const PrimitiveShell &b) {
  ShellPair sp;
  sp.p = a.alpha + b.alpha;
  const double mu = a.alpha * b.alpha / sp.p;
  for (int d = 0; d < 3; ++d) {
    sp.P[d] = (a.alpha * a.center[d] + b.alpha * b.center[d]) / sp.p;
    const double AB = a.center[d] - b.center[d];
    sp.K[d] = std::exp(-mu * AB * AB);
    sp.A[d] = a.center[d];
    sp.B[d] = b.center[d];
  }
  sp.la = a.l;
  sp.lb = b.l;
  return sp;
}

} // namespace intti
