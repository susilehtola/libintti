// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

namespace intti {

/// Two-electron interaction kernels supported by the Gaussian resolution
///   1/r = (2/sqrt(pi)) \int_0^\infty exp(-t^2 r^2) dt
///       = (2/sqrt(pi)) \int_0^\infty w^{-2} exp(-r^2/w^2) dw     (w = 1/t)
/// and its variants. This Gaussian/Laplace ("proper-time") transform of 1/r --
/// the same identity the Boys function rests on -- is decades old and not
/// original to any one method: it is the plain Laplace transform; the
/// Gaussian-transform molecular-integral method of Shavitt & Karplus (1960s);
/// the finite-element form of White, Wilkins & Teter, Phys. Rev. B 39, 5819
/// (1989), eqn 25; and the tensorial-grid form of Losilla & Sundholm et al.
/// (DAGE, J. Chem. Phys. 132, 024102 (2010)). The specific t-quadrature /
/// delta-tail machinery we use follows Jusélius and Sundholm, J. Chem. Phys.
/// 126, 094101 (2007) and Losilla, Mehine and Sundholm, Mol. Phys. 110, 2569
/// (2012) -- realizations of the identity, which they did not originate.
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
