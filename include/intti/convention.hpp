// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Runtime selection of AO conventions.
//
// Codes disagree about how the functions of a shell are ORDERED and about their
// PHASES -- libcint keeps p spherical functions Cartesian, Molden and Gaussian
// order spherical harmonics 0, +1, -1, +2, -2 rather than -l..+l, Cartesian
// monomial orders differ, and several codes flip signs of particular m. None of
// that is a different integral. Within a shell every one of these differences is
// a PERMUTATION COMPOSED WITH A DIAGONAL of +-1, so a single small object per l
// expresses all of them, and they compose: converting between two foreign
// conventions is the product of two of these through intti's own.
//
// That is why this is a boundary layer and not a parameter threaded through the
// engine, exactly as with real/complex harmonics (harmonics.hpp). The
// quadrature, the McMurchie-Davidson recursion and the J/K builders never see
// it.
//
// WHAT THIS DELIBERATELY DOES NOT COVER: normalisation. Ordering and phase
// depend only on l, but a normalisation convention can depend on the primitive
// EXPONENT (cart_norm_pyscf does), so it is not a per-l object and does not
// belong here. It is already handled where the basis set is converted --
// coeff_rescale and cart_norm_pyscf in the facade. Folding it in here would
// make the object look more general than it is.
//
// ADDING A CONVENTION. A wrong table here is SILENT: it produces a perfectly
// self-consistent transform of the wrong basis, which is precisely how the
// libcint p ordering was missed until L_z failed to come out diagonal. So a new
// convention needs an oracle, not a table transcribed from documentation. The
// two below are validated: intti's own by construction, libcint's against
// pyscf.gto.cart2sph and against L_z = i m S on PySCF's own integrals.

#include <cmath>
#include <cstddef>
#include <vector>

#include "gto.hpp"

namespace intti {

/// How one shell's functions map onto intti's own ordering:
///     foreign_i = scale[i] * intti[perm[i]]
/// with scale +-1 (phase only; see the note on normalisation above).
struct ShellReindex {
  std::vector<int> perm;
  std::vector<double> scale;
  bool identity() const {
    for (std::size_t i = 0; i < perm.size(); ++i)
      if (perm[i] != static_cast<int>(i) || scale[i] != 1.0) return false;
    return true;
  }
};

inline ShellReindex identity_reindex(int n) {
  ShellReindex r;
  r.perm.resize(n);
  r.scale.assign(n, 1.0);
  for (int i = 0; i < n; ++i) r.perm[i] = i;
  return r;
}

/// The AO conventions intti can convert to and from.
enum class AoConvention {
  Intti,  ///< Cartesian in cart_comp order; spherical m = -l..+l, standard phases
  Libcint ///< as used by libcint and PySCF
};

/// Reindex for a SPHERICAL shell of angular momentum l.
inline ShellReindex sph_reindex(AoConvention c, int l) {
  ShellReindex r = identity_reindex(2 * l + 1);
  if (c == AoConvention::Libcint && l == 1) {
    // libcint slot (0,1,2) = (x,y,z); intti's m = (-1,0,+1) = (y,z,x).
    // So slot 0 <- intti 2, slot 1 <- intti 0, slot 2 <- intti 1.
    r.perm = {2, 0, 1};
  }
  return r;
}

/// Reindex for a CARTESIAN shell of angular momentum l. intti and libcint agree
/// here -- the facade's Cartesian path reproduces PySCF to 1e-14 at l up to 2 --
/// so this is the identity today and exists so that a convention which does
/// differ has somewhere to go.
inline ShellReindex cart_reindex(AoConvention c, int l) {
  (void)c;
  return identity_reindex(ncart(l));
}

/// Rewrite a whole nao x nao AO matrix from intti's convention into `to`.
/// `nfunc[s]` is the number of functions in shell s and `reindex(s)` its map.
template <class Real, class Fn>
std::vector<Real> convert_matrix(const std::vector<int> &ao_off,
                                 const std::vector<int> &nfunc, int nao, const Real *M,
                                 Fn reindex) {
  const int ns = static_cast<int>(nfunc.size());
  std::vector<int> perm(nao);
  std::vector<Real> scale(nao, Real(1));
  for (int s = 0; s < ns; ++s) {
    const ShellReindex r = reindex(s);
    for (int i = 0; i < nfunc[s]; ++i) {
      perm[ao_off[s] + i] = ao_off[s] + r.perm[i];
      scale[ao_off[s] + i] = static_cast<Real>(r.scale[i]);
    }
  }
  std::vector<Real> out(static_cast<std::size_t>(nao) * nao, Real(0));
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j)
      out[static_cast<std::size_t>(i) * nao + j] =
          scale[i] * scale[j] *
          M[static_cast<std::size_t>(perm[i]) * nao + perm[j]];
  return out;
}

} // namespace intti
