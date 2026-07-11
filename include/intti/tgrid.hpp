// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola
#pragma once

#include <Kokkos_Core.hpp>

#include "kernel.hpp"

namespace intti {

/// Parameters of the composite linear + logarithmic t quadrature
/// (Jusélius and Sundholm, J. Chem. Phys. 126, 094101 (2007)).
struct TGridSpec {
  double t_lin{3.0}; ///< switch point between the linear and logarithmic panels
  int n_lin{50};     ///< Gauss-Legendre nodes on the linear panel
  int n_log{80};     ///< Gauss-Legendre nodes on the logarithmic panel
  double t_c{60.0};  ///< truncation point of infinite-range kernels
};

/// Quadrature grid in t. The weights include the overall 2/sqrt(pi) factor
/// and the kernel t-weight (e.g. exp(-kappa^2/(4t^2)) for Yukawa), so
///   kernel(r) ~= sum_i w_i exp(-t_i^2 r^2)  (+ delta-function tail).
///
/// The truncated tail t > t_c is rectified with the delta-function trick of
/// Losilla, Mehine and Sundholm, Mol. Phys. 110, 2569 (2012):
/// exp(-t^2 r^2) -> (pi^(3/2)/t^3) delta^3(r), so the tail contributes
/// tail_coeff = pi/t_c^2 times the corresponding delta-type integral
/// (the four-orbital overlap for an ERI quartet). This turns the O(1/t_c^2)
/// truncation error into O(1/t_c^4).
struct TGrid {
  Kokkos::View<double *> t; ///< quadrature nodes
  Kokkos::View<double *> w; ///< quadrature weights (see above)
  double t_c{0.0};          ///< truncation point
  double tail_coeff{0.0};   ///< pi/t_c^2, or 0 for finite-range kernels (Erf)

  int n() const { return static_cast<int>(t.extent(0)); }
};

/// Build the composite t grid for the given kernel.
TGrid make_tgrid(const Kernel &kernel, const TGridSpec &spec = {});

/// Gauss-Legendre nodes and weights on [a, b]; exposed for tests.
void gauss_legendre(int n, double a, double b, double *x, double *w);

} // namespace intti
