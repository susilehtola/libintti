// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola
#pragma once

#include <cmath>
#include <type_traits>

#include <Kokkos_Core.hpp>

namespace intti {

/// exp() usable in device code for builtin floating-point types and on the
/// host for class-type scalars (e.g. MPFR wrappers) via ADL.
template <class Real>
KOKKOS_INLINE_FUNCTION Real exp_(Real x) {
  if constexpr (std::is_floating_point_v<Real>) {
    return Kokkos::exp(x);
  } else {
    using std::exp;
    return exp(x);
  }
}

/// sqrt() with the same dispatch as exp_().
template <class Real>
KOKKOS_INLINE_FUNCTION Real sqrt_(Real x) {
  if constexpr (std::is_floating_point_v<Real>) {
    return Kokkos::sqrt(x);
  } else {
    using std::sqrt;
    return sqrt(x);
  }
}

/// pi in the precision of Real (host only).
template <class Real> Real pi_v() {
  using std::acos;
  return acos(Real(-1));
}

} // namespace intti
