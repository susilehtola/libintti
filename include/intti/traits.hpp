// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Scalar traits. libintti is templated on a *scalar* type, which may be
// complex (GIAOs / London orbitals in a magnetic field, see giao.hpp) or of
// extended precision (long double, __float128, MPFR wrappers). Two notions
// must be kept apart:
//
//   * the scalar type      -- orbital coefficients, product centres, integrals;
//   * its real type        -- exponents, the t grid, tolerances, comparisons.
//
// For GIAOs only the *centres* become complex: the exponents, and hence
// D = pq + t^2(p+q), theta, rho and the pi/sqrt(D) prefactors, stay real.
// That is why the t grid (tgrid.hpp) is always built in real_t<Scalar>.

#include <complex>
#include <limits>
#include <type_traits>

#include <Kokkos_Core.hpp> // Kokkos::complex (device-runnable complex scalar)

#ifdef INTTI_HAVE_QUADMATH
#include <quadmath.h>
#endif

namespace intti {

/// real_t<T>: the underlying real type of a (possibly complex) scalar.
/// Two complex types are supported: std::complex (host, and the reference
/// implementation) and Kokkos::complex (the same algebra, but usable inside
/// device kernels -- std::complex is not).
template <class T> struct real_type {
  using type = T;
};
template <class T> struct real_type<std::complex<T>> {
  using type = T;
};
template <class T> struct real_type<Kokkos::complex<T>> {
  using type = T;
};
template <class T> using real_t = typename real_type<T>::type;

template <class T> inline constexpr bool is_complex_v = false;
template <class T> inline constexpr bool is_complex_v<std::complex<T>> = true;
template <class T> inline constexpr bool is_complex_v<Kokkos::complex<T>> = true;

/// Scalars that Kokkos can hold in a View and compute with. float and double
/// run on any backend; long double works on the host backends (and is only
/// ever used for reference values, never on a device). Kokkos::complex<float>
/// and <double> are device-runnable too -- that is the finite-magnetic-field
/// (London/GIAO) path on the GPU. Everything else -- __float128, std::complex,
/// class-type scalars such as MPFR wrappers -- takes the serial host path of
/// eri_quartet().
///
/// std::is_floating_point_v is NOT a usable test here: it is *true* for
/// __float128, which Kokkos cannot handle, and *false* for std::complex.
template <class T>
inline constexpr bool kokkos_scalar_v =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, long double> || std::is_same_v<T, Kokkos::complex<float>> ||
    std::is_same_v<T, Kokkos::complex<double>>;

/// Types for which the host math functions come from <cmath> / ADL.
template <class T>
inline constexpr bool is_quad_v =
#ifdef INTTI_HAVE_QUADMATH
    std::is_same_v<T, __float128>;
#else
    false;
#endif

} // namespace intti
