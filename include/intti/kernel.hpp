// SPDX-License-Identifier: MPL-2.0
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
struct Kernel {
  KernelType type{KernelType::Coulomb};
  double omega{0.0}; ///< range-separation parameter (Erf, Erfc)
  double kappa{0.0}; ///< screening exponent (Yukawa)

  static Kernel coulomb() { return {KernelType::Coulomb, 0.0, 0.0}; }
  static Kernel erf_rs(double omega) { return {KernelType::Erf, omega, 0.0}; }
  static Kernel erfc_rs(double omega) { return {KernelType::Erfc, omega, 0.0}; }
  static Kernel yukawa(double kappa) { return {KernelType::Yukawa, 0.0, kappa}; }
};

} // namespace intti
