// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Interfaces for orbital-product ("fit") functions across basis-set families.
// M2 fixes the API; the implementations land in M3. The mathematical basis is
// validated in prototype/psc_validation.py and documented in docs/psc.md.

#include <variant>
#include <vector>

#include "gto.hpp"
#include "tgrid.hpp"

namespace intti {

/// 2D prolate spheroidal grid for a two-center orbital product with foci at
/// the two centers. A product function is chi = sum_m f_m(xi, eta) e^{i m phi}
/// with the m sum *exactly* finite (|m| <= l_a + l_b) about the interfocal
/// axis. Nodes: Gauss-Legendre in eta on [-1, 1]; Gauss-Legendre in xi on
/// [1, xi_max] with xi_max set by the exponential decay of the product.
template <class Real> struct PSCGrid {
  Real focus_a[3], focus_b[3];
  int n_xi{48}, n_eta{48};
  std::vector<Real> xi, eta; ///< nodes
  std::vector<Real> w2d;     ///< combined weights incl. the (R/2)^3 (xi^2-eta^2) Jacobian
};

/// A two-center product on a PSC grid: Fourier components f_m(xi, eta),
/// m = -mmax..mmax (exact truncation for AO products).
template <class Real> struct PSCProduct {
  const PSCGrid<Real> *grid;
  int mmax;
  std::vector<Real> f_re, f_im; ///< (n_xi*n_eta) x (2 mmax + 1), m-major
};

/// A product resolved on a Cartesian tensor grid (FEM / bubbles-and-cube path).
template <class Real> struct TensorGridProduct {
  std::vector<Real> x, y, z; ///< per-axis nodes
  std::vector<Real> w;       ///< per-axis weights (separable)
  std::vector<Real> values;  ///< (nx, ny, nz), x fastest
};

/// Orbital-product function in one of the supported representations:
/// - ShellPair: analytic GTO pair (Hermite recursions at every t; Mobius grids)
/// - PSCProduct: exact diatomic representation (LinLog + delta tail; the grid
///   cannot resolve arbitrarily large t)
/// - TensorGridProduct: 3D grid (LinLog + delta tail)
template <class Real>
using ProductFunction =
    std::variant<ShellPair<Real>, PSCProduct<Real>, TensorGridProduct<Real>>;

/// Kernel-mediated interaction (f | kernel | g) evaluated on the shared t
/// grid. Coaxial PSC pairs couple diagonally in m through scaled Bessel
/// factors ive(m, 2 t^2 rho1 rho2); general geometries go through uniform-phi
/// point clouds; every pairing of representations reduces to per-t GEMMs.
/// Implemented in M3.
template <class Real>
Real interaction(const ProductFunction<Real> &f, const ProductFunction<Real> &g,
                 const TGrid<Real> &grid);

} // namespace intti
