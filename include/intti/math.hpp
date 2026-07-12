// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola
#pragma once

#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "traits.hpp"

namespace intti {

/// exp(): Kokkos math for the device-capable scalars (float, double), and
/// <cmath>/ADL otherwise -- long double, __float128 (quadmath), std::complex
/// (std::exp), or class-type scalars such as MPFR wrappers.
template <class Scalar>
KOKKOS_INLINE_FUNCTION Scalar exp_(Scalar x) {
  if constexpr (kokkos_scalar_v<Scalar>) {
    return Kokkos::exp(x);
  } else if constexpr (is_quad_v<Scalar>) {
#ifdef INTTI_HAVE_QUADMATH
    return expq(x);
#else
    return Scalar(0);
#endif
  } else {
    using std::exp;
    return exp(x);
  }
}

/// sqrt() with the same dispatch. For complex scalars this is the principal
/// branch; libintti only ever takes square roots of quantities with positive
/// real part (D = pq + t^2(p+q) and p+q are real and positive even for
/// GIAOs), so no branch cut is ever crossed.
template <class Scalar>
KOKKOS_INLINE_FUNCTION Scalar sqrt_(Scalar x) {
  if constexpr (kokkos_scalar_v<Scalar>) {
    return Kokkos::sqrt(x);
  } else if constexpr (is_quad_v<Scalar>) {
#ifdef INTTI_HAVE_QUADMATH
    return sqrtq(x);
#else
    return Scalar(0);
#endif
  } else {
    using std::sqrt;
    return sqrt(x);
  }
}

/// |x| in the real type of Scalar.
template <class Scalar>
KOKKOS_INLINE_FUNCTION real_t<Scalar> abs_(Scalar x) {
  if constexpr (is_complex_v<Scalar>) {
    return std::abs(x);
  } else if constexpr (is_quad_v<Scalar>) {
#ifdef INTTI_HAVE_QUADMATH
    return fabsq(x);
#else
    return real_t<Scalar>(0);
#endif
  } else if constexpr (kokkos_scalar_v<Scalar>) {
    return Kokkos::abs(x);
  } else {
    using std::abs;
    return abs(x);
  }
}

/// pi in the precision of Real (host only). Real must be a real type.
template <class Real> Real pi_v() {
  static_assert(!is_complex_v<Real>, "pi_v requires a real type");
  if constexpr (is_quad_v<Real>) {
#ifdef INTTI_HAVE_QUADMATH
    return M_PIq;
#else
    return Real(0);
#endif
  } else {
    using std::acos;
    return acos(Real(-1));
  }
}

/// machine epsilon of Real.
template <class Real> Real eps_() {
  return std::numeric_limits<Real>::epsilon();
}

} // namespace intti
