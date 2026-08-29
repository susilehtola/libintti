// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola
#pragma once

// One-electron property integral matrices in the McMurchie-Davidson basis:
// overlap, kinetic energy, and Cartesian multipole moments. These are all
// "moment" integrals -- Gaussian products times a polynomial (x-C)^e, with no
// interaction kernel -- so they come straight off the E coefficients
// (hermite1d.hpp) with no t quadrature.
//
// The library's native API is matrix-level (a whole nao x nao AO matrix per
// call), never individual shell blocks: overlap_matrix(basis), etc. Matrices
// are built over the primitive Cartesian AOs of a ShellBasis, row-major,
// unnormalized primitives -- the caller (or the libcint facade) applies the
// normalization/contraction, exactly as for the ERIs.

#include <array>
#include <cstddef>
#include <vector>

#include "fock.hpp"
#include "gto.hpp"
#include "hermite1d.hpp"
#include "math.hpp"

namespace intti {

namespace detail {

/// Per-direction 1D overlap table for a primitive shell pair:
///   s1[i*(lbx+1)+j] = int (x-A)^i e^{-alpha (x-A)^2} (x-B)^j e^{-beta (x-B)^2} dx,
/// for i = 0..la+exa, j = 0..lb+exb. Built from the MD E coefficients
/// (E_0^{ij} sqrt(pi/p)). exa/exb request extra angular momentum so kinetic
/// (ket +2) and multipoles (bra +order) can raise indices.
template <class Real>
void overlap_1d(Real alpha, Real A, Real beta, Real B, int la, int lb, int exa,
                int exb, std::vector<Real> &s1, int &lbx) {
  const int lax = la + exa;
  lbx = lb + exb;
  const Real p = alpha + beta;
  const Real P = (alpha * A + beta * B) / p;
  const Real mu = alpha * beta / p;
  const Real ab = A - B;
  const Real K = exp_(-mu * ab * ab);
  std::vector<Real> E(static_cast<std::size_t>(lax + 1) * (lbx + 1) * (lax + lbx + 1));
  e_coeffs(lax, lbx, p, P - A, P - B, K, E.data());
  const Real pref = sqrt_(pi_v<Real>() / p);
  s1.assign(static_cast<std::size_t>(lax + 1) * (lbx + 1), Real(0));
  const int nt = lax + lbx + 1;
  for (int i = 0; i <= lax; ++i)
    for (int j = 0; j <= lbx; ++j)
      s1[i * (lbx + 1) + j] = pref * E[(i * (lbx + 1) + j) * nt + 0];
}

/// 1D kinetic table T[i*(lbx+1)+j] from the overlap table and the ket
/// exponent beta: T_{ij} = -2 b^2 S_{i,j+2} + b(2j+1) S_{i,j} - j(j-1)/2 S_{i,j-2}.
template <class Real>
void kinetic_1d(const std::vector<Real> &s1, int lbx, int la, int lb, Real beta,
                std::vector<Real> &t1) {
  t1.assign(static_cast<std::size_t>(la + 1) * (lb + 1), Real(0));
  auto S = [&](int i, int j) { return s1[i * (lbx + 1) + j]; };
  for (int i = 0; i <= la; ++i)
    for (int j = 0; j <= lb; ++j) {
      Real v = -2 * beta * beta * S(i, j + 2) + beta * (2 * j + 1) * S(i, j);
      if (j >= 2) v -= Real(0.5) * j * (j - 1) * S(i, j - 2);
      t1[i * (lb + 1) + j] = v;
    }
}

/// 1D multipole table m1[(e*(la+1)+i)*(lb+1)+j] = <i|(x-O)^e|j>, e = 0..emax,
/// from (x-O)^e = sum_k C(e,k)(x-A)^k (A-O)^{e-k} and <i|(x-A)^k|j> = S_{i+k,j}.
template <class Real>
void multipole_1d(const std::vector<Real> &s1, int lbx, int la, int lb, Real A,
                  Real O, int emax, std::vector<Real> &m1) {
  m1.assign(static_cast<std::size_t>(emax + 1) * (la + 1) * (lb + 1), Real(0));
  auto S = [&](int i, int j) { return s1[i * (lbx + 1) + j]; };
  const Real dAO = A - O;
  for (int e = 0; e <= emax; ++e) {
    Real binom = 1; // C(e,k)
    for (int k = 0; k <= e; ++k) {
      Real pw = 1; // (A-O)^{e-k}
      for (int r = 0; r < e - k; ++r)
        pw *= dAO;
      const Real c = binom * pw;
      for (int i = 0; i <= la; ++i)
        for (int j = 0; j <= lb; ++j)
          m1[(static_cast<std::size_t>(e) * (la + 1) + i) * (lb + 1) + j] +=
              c * S(i + k, j);
      binom = binom * (e - k) / (k + 1);
    }
  }
}

} // namespace detail

/// Overlap matrix S (nao x nao, row-major) over the primitive Cartesian AOs.
template <class Real>
std::vector<Real> overlap_matrix(const ShellBasis<Real> &basis) {
  const int nao = basis.nao;
  std::vector<Real> S(static_cast<std::size_t>(nao) * nao, Real(0));
  const int ns = static_cast<int>(basis.shells.size());
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      std::vector<Real> sx, sy, sz;
      int lbx;
      detail::overlap_1d(sa.alpha, sa.center[0], sb.alpha, sb.center[0], sa.l, sb.l, 0, 0, sx, lbx);
      detail::overlap_1d(sa.alpha, sa.center[1], sb.alpha, sb.center[1], sa.l, sb.l, 0, 0, sy, lbx);
      detail::overlap_1d(sa.alpha, sa.center[2], sb.alpha, sb.center[2], sa.l, sb.l, 0, 0, sz, lbx);
      for (int ka = 0; ka < ncart(sa.l); ++ka) {
        int a3[3];
        cart_comp(sa.l, ka, a3[0], a3[1], a3[2]);
        for (int kb = 0; kb < ncart(sb.l); ++kb) {
          int b3[3];
          cart_comp(sb.l, kb, b3[0], b3[1], b3[2]);
          const Real v = sx[a3[0] * (lbx + 1) + b3[0]] * sy[a3[1] * (lbx + 1) + b3[1]] *
                         sz[a3[2] * (lbx + 1) + b3[2]];
          S[(basis.ao_off[a] + ka) * nao + basis.ao_off[b] + kb] = v;
        }
      }
    }
  return S;
}

/// Kinetic energy matrix T = -1/2 <a| nabla^2 |b>.
template <class Real>
std::vector<Real> kinetic_matrix(const ShellBasis<Real> &basis) {
  const int nao = basis.nao;
  std::vector<Real> T(static_cast<std::size_t>(nao) * nao, Real(0));
  const int ns = static_cast<int>(basis.shells.size());
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      std::vector<Real> sx, sy, sz, tx, ty, tz;
      int lbx;
      detail::overlap_1d(sa.alpha, sa.center[0], sb.alpha, sb.center[0], sa.l, sb.l, 0, 2, sx, lbx);
      detail::kinetic_1d(sx, lbx, sa.l, sb.l, sb.alpha, tx);
      detail::overlap_1d(sa.alpha, sa.center[1], sb.alpha, sb.center[1], sa.l, sb.l, 0, 2, sy, lbx);
      detail::kinetic_1d(sy, lbx, sa.l, sb.l, sb.alpha, ty);
      detail::overlap_1d(sa.alpha, sa.center[2], sb.alpha, sb.center[2], sa.l, sb.l, 0, 2, sz, lbx);
      detail::kinetic_1d(sz, lbx, sa.l, sb.l, sb.alpha, tz);
      const int lb1 = sb.l + 1;
      auto Sx = [&](int i, int j) { return sx[i * (lbx + 1) + j]; };
      auto Sy = [&](int i, int j) { return sy[i * (lbx + 1) + j]; };
      auto Sz = [&](int i, int j) { return sz[i * (lbx + 1) + j]; };
      for (int ka = 0; ka < ncart(sa.l); ++ka) {
        int a3[3];
        cart_comp(sa.l, ka, a3[0], a3[1], a3[2]);
        for (int kb = 0; kb < ncart(sb.l); ++kb) {
          int b3[3];
          cart_comp(sb.l, kb, b3[0], b3[1], b3[2]);
          const Real v =
              tx[a3[0] * lb1 + b3[0]] * Sy(a3[1], b3[1]) * Sz(a3[2], b3[2]) +
              Sx(a3[0], b3[0]) * ty[a3[1] * lb1 + b3[1]] * Sz(a3[2], b3[2]) +
              Sx(a3[0], b3[0]) * Sy(a3[1], b3[1]) * tz[a3[2] * lb1 + b3[2]];
          T[(basis.ao_off[a] + ka) * nao + basis.ao_off[b] + kb] = v;
        }
      }
    }
  return T;
}

/// Cartesian multipole matrices <a|(x-O)^ex (y-O)^ey (z-O)^ez|b> for every
/// (ex,ey,ez) with ex+ey+ez <= max_order, returned as
/// out[component] where component runs in cart_comp order per total order,
/// concatenated order 0,1,2,...; see multipole_labels() for the mapping.
/// Order 0 is the overlap, order 1 the dipole, order 2 the quadrupole, ...
template <class Real>
std::vector<std::vector<Real>> multipole_matrices(const ShellBasis<Real> &basis,
                                                  int max_order,
                                                  const Real origin[3]) {
  const int nao = basis.nao;
  // component list: (ex,ey,ez), total order 0..max_order, cart_comp ordering
  std::vector<std::array<int, 3>> comps;
  for (int L = 0; L <= max_order; ++L)
    for (int k = 0; k < ncart(L); ++k) {
      int e[3];
      cart_comp(L, k, e[0], e[1], e[2]);
      comps.push_back({e[0], e[1], e[2]});
    }
  std::vector<std::vector<Real>> out(comps.size(),
                                     std::vector<Real>(static_cast<std::size_t>(nao) * nao, Real(0)));
  const int ns = static_cast<int>(basis.shells.size());
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      std::vector<Real> s[3], m[3];
      int lbx;
      for (int d = 0; d < 3; ++d) {
        detail::overlap_1d(sa.alpha, sa.center[d], sb.alpha, sb.center[d], sa.l, sb.l,
                           max_order, 0, s[d], lbx);
        detail::multipole_1d(s[d], lbx, sa.l, sb.l, sa.center[d], origin[d], max_order, m[d]);
      }
      const int lb1 = sb.l + 1;
      auto M = [&](int d, int e, int i, int j) {
        return m[d][(static_cast<std::size_t>(e) * (sa.l + 1) + i) * lb1 + j];
      };
      for (std::size_t ci = 0; ci < comps.size(); ++ci) {
        const auto &e = comps[ci];
        for (int ka = 0; ka < ncart(sa.l); ++ka) {
          int a3[3];
          cart_comp(sa.l, ka, a3[0], a3[1], a3[2]);
          for (int kb = 0; kb < ncart(sb.l); ++kb) {
            int b3[3];
            cart_comp(sb.l, kb, b3[0], b3[1], b3[2]);
            const Real v = M(0, e[0], a3[0], b3[0]) * M(1, e[1], a3[1], b3[1]) *
                           M(2, e[2], a3[2], b3[2]);
            out[ci][(basis.ao_off[a] + ka) * nao + basis.ao_off[b] + kb] = v;
          }
        }
      }
    }
  return out;
}

/// (ex,ey,ez) labels matching the component order of multipole_matrices().
inline std::vector<std::array<int, 3>> multipole_labels(int max_order) {
  std::vector<std::array<int, 3>> comps;
  for (int L = 0; L <= max_order; ++L)
    for (int k = 0; k < ncart(L); ++k) {
      int e[3];
      cart_comp(L, k, e[0], e[1], e[2]);
      comps.push_back({e[0], e[1], e[2]});
    }
  return comps;
}

/// Trace(D^T M) -- contract a density (or any nao x nao matrix) with a
/// property matrix to get the property value.
template <class Real>
Real property_value(const std::vector<Real> &D, const std::vector<Real> &M) {
  Real s{};
  for (std::size_t i = 0; i < D.size(); ++i)
    s += D[i] * M[i];
  return s;
}

} // namespace intti
