// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Real <-> complex spherical harmonics, in the Condon-Shortley phase
// convention.
//
// This is a BOUNDARY transform, not a second integral path, and that is the
// whole point. Within a shell the two conventions are related by a fixed
// unitary that depends only on l, so it sits at exactly the seam c2s.hpp
// occupies: the engine stays real Cartesian, and only the AO basis the caller
// sees changes. Nothing in the quadrature, the McMurchie-Davidson recursion or
// the J/K builders is aware of it.
//
// Condon-Shortley (the (-1)^m carried by Y_l^m itself), with the real harmonics
// ordered m = -l..+l as c2s_matrix produces them:
//
//   Y_l^0    = R_0
//   Y_l^{+m} = (-1)^m (R_{+m} + i R_{-m}) / sqrt(2)      m > 0
//   Y_l^{-m} = (R_{+m} - i R_{-m}) / sqrt(2)             m > 0
//
// where R_{+m} is the cosine-like real harmonic and R_{-m} the sine-like one.
//
// VALIDATION. The phase convention is the one thing here that cannot be
// asserted into correctness -- codes genuinely differ, and a wrong relative
// sign inside a (+m, -m) pair is invisible to any norm or unitarity check. The
// oracle is physical and independent of this file: L_z must be DIAGONAL in the
// complex basis with eigenvalue m. intti computes <mu|r x nabla|nu> by an
// unrelated route (oneel.hpp), and r x nabla is i L, so the z component must
// come out diagonal with entries i*m. That pins the phases rather than assuming
// them; see tests/test_harmonics.cpp.

#include <complex>
#include <utility>
#include <map>
#include <vector>

#include "c2s.hpp"
#include "convention.hpp"
#include "gto.hpp"

namespace intti {

/// Which spherical-harmonic convention an AO basis is expressed in.
enum class Harmonics { Real, Complex };

/// How the REAL spherical harmonics of the input are ordered within a shell.
///
/// Standard is m = -l..+l with the usual sine/cosine assignment, which is what
/// c2s_matrix produces. Libcint agrees at every l EXCEPT l = 1, where it keeps p
/// functions in Cartesian order (x, y, z) rather than (y, z, x) -- a historical
/// quirk, and the only structural difference. Established by comparing
/// c2s_matrix against pyscf.gto.cart2sph at l = 1, 2, 3, not assumed: at l = 2
/// and 3 the two agree row for row up to each shell's overall normalisation,
/// which is uniform within a shell and therefore commutes with the transform.
///
/// Getting this wrong is silent -- it produces a perfectly unitary transform of
/// the wrong basis -- so it is a parameter rather than a guess.
enum class RealOrder { Standard, Libcint };

/// (2l+1) x (2l+1) row-major unitary U with chi^complex_m = sum_m' U[m][m']
/// chi^real_m', both indices ordered m = -l..+l.
/// The general form: `order` is the reindex of the REAL basis this matrix is to
/// consume, so a consumer with a convention intti does not ship supplies its own
/// ShellReindex and needs no change here. The enum overload below is the
/// convenience wrapper over the two conventions that are validated.
template <class Real = double>
std::vector<std::complex<Real>> r2c_matrix(int l, const ShellReindex &order) {
  using C = std::complex<Real>;
  const int nm = 2 * l + 1;
  std::vector<C> U(static_cast<std::size_t>(nm) * nm, C(0));
  const Real inv = Real(1) / std::sqrt(Real(2));
  U[static_cast<std::size_t>(l) * nm + l] = C(1); // m = 0
  for (int m = 1; m <= l; ++m) {
    const Real sgn = (m % 2) ? Real(-1) : Real(1); // Condon-Shortley (-1)^m
    // Y_l^{+m} = (-1)^m (R_{+m} + i R_{-m}) / sqrt(2)
    U[static_cast<std::size_t>(m + l) * nm + (m + l)] = C(sgn * inv, 0);
    U[static_cast<std::size_t>(m + l) * nm + (-m + l)] = C(0, sgn * inv);
    // Y_l^{-m} = (R_{+m} - i R_{-m}) / sqrt(2)
    U[static_cast<std::size_t>(-m + l) * nm + (m + l)] = C(inv, 0);
    U[static_cast<std::size_t>(-m + l) * nm + (-m + l)] = C(0, -inv);
  }
  // The real-basis index is the COLUMN, so a foreign ordering enters as a column
  // permutation and phase.
  if (!order.identity()) {
    std::vector<C> P(U.size(), C(0));
    for (int a = 0; a < nm; ++a)
      for (int slot = 0; slot < nm; ++slot)
        P[static_cast<std::size_t>(a) * nm + slot] =
            static_cast<Real>(order.scale[slot]) *
            U[static_cast<std::size_t>(a) * nm + order.perm[slot]];
    U.swap(P);
  }
  return U;
}

/// Convenience overload for the shipped conventions, via the shared table in
/// convention.hpp -- so a convention is defined in one place and validated in
/// one place.
template <class Real = double>
std::vector<std::complex<Real>> r2c_matrix(int l, RealOrder order = RealOrder::Standard) {
  return r2c_matrix<Real>(l, sph_reindex(order == RealOrder::Libcint
                                             ? AoConvention::Libcint
                                             : AoConvention::Intti,
                                         l));
}

/// Transform a whole nao x nao matrix from the real to the complex spherical
/// basis, given the angular momentum of each shell and its AO offset.
///
/// With chi^c_i = sum_m U_im chi^r_m and M_ij = <chi_i|O|chi_j>, the bra is
/// conjugated, so M^c = conj(U) M^r U^T -- NOT U M U^dagger. The distinction
/// matters: the two differ by an overall conjugation, which is exactly the
/// sign of m and therefore the thing the L_z check is testing.
/// General form: `reindex(l)` gives the ShellReindex of the real basis being
/// consumed, so an unshipped convention plugs in here without touching this
/// file. See the enum overload below for the validated presets.
template <class Real, class Fn>
std::vector<std::complex<Real>>
real_to_complex_with(const std::vector<int> &shell_l, const std::vector<int> &ao_off,
                     int nao, const Real *M, Fn reindex) {
  using C = std::complex<Real>;
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  std::vector<C> tmp(N, C(0)), out(N, C(0));
  // No memo here: the reindex is caller-supplied, so there is no key to cache
  // on. The matrices are (2l+1)^2 and built once per shell, which is nothing
  // beside the integrals.
  std::map<int, std::vector<C>> cache;
  auto Umat = [&](int l) -> const std::vector<C> & {
    auto it = cache.find(l);
    if (it == cache.end()) it = cache.emplace(l, r2c_matrix<Real>(l, reindex(l))).first;
    return it->second;
  };
  // left index: tmp = conj(U) M
  for (std::size_t s = 0; s < shell_l.size(); ++s) {
    const int l = shell_l[s], nm = 2 * l + 1, o = ao_off[s];
    const auto &U = Umat(l);
    for (int a = 0; a < nm; ++a)
      for (int j = 0; j < nao; ++j) {
        C acc(0);
        for (int b = 0; b < nm; ++b)
          acc += std::conj(U[static_cast<std::size_t>(a) * nm + b]) *
                 C(M[static_cast<std::size_t>(o + b) * nao + j], Real(0));
        tmp[static_cast<std::size_t>(o + a) * nao + j] = acc;
      }
  }
  // right index: out = tmp U^T
  for (std::size_t s = 0; s < shell_l.size(); ++s) {
    const int l = shell_l[s], nm = 2 * l + 1, o = ao_off[s];
    const auto &U = Umat(l);
    for (int i = 0; i < nao; ++i)
      for (int a = 0; a < nm; ++a) {
        C acc(0);
        for (int b = 0; b < nm; ++b)
          acc += tmp[static_cast<std::size_t>(i) * nao + o + b] *
                 U[static_cast<std::size_t>(a) * nm + b];
        out[static_cast<std::size_t>(i) * nao + o + a] = acc;
      }
  }
  return out;
}

/// Convenience overload for the shipped conventions.
template <class Real>
std::vector<std::complex<Real>>
real_to_complex(const std::vector<int> &shell_l, const std::vector<int> &ao_off, int nao,
                const Real *M, RealOrder order = RealOrder::Standard) {
  const auto conv =
      order == RealOrder::Libcint ? AoConvention::Libcint : AoConvention::Intti;
  return real_to_complex_with(shell_l, ao_off, nao, M,
                              [&](int l) { return sph_reindex(conv, l); });
}

/// The switch itself: return an AO matrix in the requested harmonic
/// convention. Harmonics::Real hands back the input with zero imaginary part,
/// so a caller that wants to be convention-agnostic writes one code path and
/// pays only a copy for the real case.
template <class Real>
std::vector<std::complex<Real>>
to_harmonics(Harmonics kind, const std::vector<int> &shell_l,
             const std::vector<int> &ao_off, int nao, const Real *M,
             RealOrder order = RealOrder::Standard) {
  if (kind == Harmonics::Complex) return real_to_complex(shell_l, ao_off, nao, M, order);
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  std::vector<std::complex<Real>> out(N);
  for (std::size_t i = 0; i < N; ++i) out[i] = std::complex<Real>(M[i], Real(0));
  return out;
}

} // namespace intti
