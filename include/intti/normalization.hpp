// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Primitive Gaussian normalization conventions. The library core works with
// unnormalized primitives; these factors map onto external conventions
// (pinned against PySCF/libcint by prototype/pyscf_validation.py; see
// docs/conventions.md).

#include <cmath>

#include "gto.hpp"
#include "math.hpp"

namespace intti {

namespace detail {
inline double dfact(int n) { // (n)!! with (-1)!! = 1
  double r = 1;
  for (int k = n; k > 1; k -= 2)
    r *= k;
  return r;
}
} // namespace detail

/// Unit L2 normalization of the Cartesian primitive x^lx y^ly z^lz e^{-a r^2}.
template <class Real> Real cart_norm_component(int lx, int ly, int lz, Real alpha) {
  using std::pow;
  using std::sqrt;
  const Real pi = pi_v<Real>();
  const int l = lx + ly + lz;
  return sqrt(pow(2 * alpha / pi, Real(1.5)) * pow(4 * alpha, Real(l)) /
              Real(detail::dfact(2 * lx - 1) * detail::dfact(2 * ly - 1) *
                   detail::dfact(2 * lz - 1)));
}

/// CCA Cartesian convention: every component of the shell carries the
/// normalization of the axial x^l component.
template <class Real> Real cart_norm_cca(int l, Real alpha) {
  return cart_norm_component<Real>(l, 0, 0, alpha);
}

/// PySCF/libcint cart=True convention (pinned by
/// prototype/pyscf_validation.py): s and p shells are unit-normalized per
/// component; l >= 2 shells carry the common spherical normalization,
/// which is the axial factor times sqrt(4 pi / (2l+1)).
template <class Real> Real cart_norm_pyscf(int l, Real alpha) {
  using std::sqrt;
  if (l <= 1) return cart_norm_component<Real>(l, 0, 0, alpha);
  const Real pi = pi_v<Real>();
  return cart_norm_cca<Real>(l, alpha) * sqrt(4 * pi / (2 * l + 1));
}

/// Rescale from intti's c2s_matrix(l) rows (include/intti/c2s.hpp;
/// sphere-orthonormal, m = -l..+l) to libcint's int2e_sph row convention,
/// pinned empirically by prototype/pyscf_validation.py (see
/// docs/conventions.md): for l <= 1, libcint's spherical AOs coincide
/// exactly with the (cart_norm_pyscf-normalized) Cartesian ones -- same
/// functions, same order (x, y, z for l = 1), so a c2s row permutation
/// would be needed on top of a scale factor and the facade bypasses
/// c2s_matrix entirely for these shells (factor 1, unused). For l >= 2 the
/// row order already matches m = -l..+l and the rescale is a single
/// l-independent constant, 1/sqrt(4 pi), confirmed to 1e-15 relative
/// deviation against PySCF's int2e_sph for l = 2, 3.
template <class Real> Real sph_rescale(int l) {
  using std::sqrt;
  if (l <= 1) return Real(1);
  return 1 / sqrt(4 * pi_v<Real>());
}

} // namespace intti
