// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

namespace intti {

/// Two-electron interaction kernels supported by the Gaussian resolution
///   1/r = (2/sqrt(pi)) \int_0^\infty exp(-t^2 r^2) dt
/// and its variants; see Jusélius and Sundholm, J. Chem. Phys. 126, 094101
/// (2007) and Losilla, Mehine and Sundholm, Mol. Phys. 110, 2569 (2012).
enum class KernelType {
  Coulomb, ///< 1/r: t in [0, inf)
  Erf,     ///< erf(omega r)/r: t in [0, omega]
  Erfc,    ///< erfc(omega r)/r: t in [omega, inf)
  Yukawa   ///< exp(-kappa r)/r: t in [0, inf) with weight exp(-kappa^2/(4t^2))
};

/// Kernel descriptor: shapes the t quadrature grid and the tail correction.
/// The hot loops are kernel-agnostic.
template <class Real = double> struct Kernel {
  KernelType type{KernelType::Coulomb};
  Real omega{0}; ///< range-separation parameter (Erf, Erfc)
  Real kappa{0}; ///< screening exponent (Yukawa)
};

template <class Real = double> Kernel<Real> coulomb() {
  return {KernelType::Coulomb, Real(0), Real(0)};
}
template <class Real> Kernel<Real> erf_rs(Real omega) {
  return {KernelType::Erf, omega, Real(0)};
}
template <class Real> Kernel<Real> erfc_rs(Real omega) {
  return {KernelType::Erfc, omega, Real(0)};
}
template <class Real> Kernel<Real> yukawa(Real kappa) {
  return {KernelType::Yukawa, Real(0), kappa};
}

} // namespace intti
