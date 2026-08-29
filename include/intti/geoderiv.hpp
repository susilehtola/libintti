// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Arbitrary-order geometric derivatives, done the quadrature way.
//
// Because the t grid is perturbation-independent, differentiation commutes
// with the quadrature: the derivative of an integral, to any order, is the
// same quadrature sum of the analytically differentiated integrand. Every
// integrand factors per Cartesian direction, and d/d(centre) of a Gaussian
// factor is the McMurchie-Davidson shift operator
//   d/dA_x : F(i) -> i F(i-1) - 2 alpha F(i+1)   (bra centre A)
//   d/dB_x : F(j) -> j F(j-1) - 2 beta  F(j+1)   (ket centre B)
// applied to the 1D factor. Arbitrary order is just repetition; a mixed
// derivative d^na_A d^nb_B is na_d shifts on the bra index and nb_d on the ket
// index, per direction. This one shift core drives every derivative (of every
// operator, 1e or 2e) -- gen1int's N-ary-tree program, but the differentiation
// is analytic on the fixed grid rather than a special recursion per operator.
//
// This header provides the shift core and the geometric derivatives of the
// overlap matrix as the reference demonstrator (gradient = order 1, Hessian =
// order 2). The same shift applied to the kinetic / nuclear / ERI 1D factors
// gives their derivatives identically.

#include <array>
#include <cstddef>
#include <vector>

#include "fock.hpp"
#include "gto.hpp"
#include "oneel.hpp" // detail::overlap_1d

namespace intti {

namespace detail {

/// Apply the bra-centre derivative shift once to a 1D factor table stored as
/// (nbra+1) x (nket+1) row-major over the CURRENT valid ranges [0,ib]x[0,jb]:
///   out(i,j) = i F(i-1,j) - 2 alpha F(i+1,j),   i in [0,ib-1], j in [0,jb].
/// The bra range shrinks by one (needs F up to ib).
template <class Real>
void shift_bra(const std::vector<Real> &F, int ldF, int ib, int jb, Real alpha,
               std::vector<Real> &out) {
  out.assign(static_cast<std::size_t>(ib) * (jb + 1), Real(0)); // new bra range [0,ib-1]
  for (int i = 0; i < ib; ++i)
    for (int j = 0; j <= jb; ++j) {
      Real v = -2 * alpha * F[(i + 1) * ldF + j];
      if (i >= 1) v += Real(i) * F[(i - 1) * ldF + j];
      out[i * (jb + 1) + j] = v;
    }
}

/// Apply the ket-centre derivative shift once: out(i,j) = j F(i,j-1) - 2b F(i,j+1).
template <class Real>
void shift_ket(const std::vector<Real> &F, int ldF, int ib, int jb, Real beta,
               std::vector<Real> &out) {
  out.assign(static_cast<std::size_t>(ib + 1) * jb, Real(0)); // new ket range [0,jb-1]
  for (int i = 0; i <= ib; ++i)
    for (int j = 0; j < jb; ++j) {
      Real v = -2 * beta * F[i * ldF + (j + 1)];
      if (j >= 1) v += Real(j) * F[i * ldF + (j - 1)];
      out[i * jb + j] = v;
    }
}

/// 1D overlap factor differentiated na times on the bra centre and nb times on
/// the ket centre; returns a (la+1) x (lb+1) table (row-major).
template <class Real>
std::vector<Real> overlap_1d_deriv(Real alpha, Real A, Real beta, Real B, int la,
                                   int lb, int na, int nb) {
  std::vector<Real> s;
  int lbx;
  overlap_1d(alpha, A, beta, B, la, lb, na, nb, s, lbx); // s: (la+na+1)x(lbx+1), lbx=lb+nb
  int ib = la + na, jb = lb + nb, ld = lbx + 1;
  std::vector<Real> cur = s, nxt;
  for (int k = 0; k < na; ++k) {
    shift_bra(cur, ld, ib, jb, alpha, nxt);
    --ib;
    ld = jb + 1;
    cur = nxt;
  }
  for (int k = 0; k < nb; ++k) {
    shift_ket(cur, ld, ib, jb, beta, nxt);
    --jb;
    ld = jb + 1;
    cur = nxt;
  }
  return cur; // (la+1) x (lb+1)
}

} // namespace detail

/// Geometric derivative of the overlap matrix: d^na_A d^nb_B S, where
/// na = (nax,nay,naz) is the derivative multi-index on the bra centre and nb
/// on the ket centre. na=nb=0 is S; |na|=1 is a gradient component; |na|+|nb|=2
/// a Hessian component. Whole nao x nao matrix (row-major).
template <class Real>
std::vector<Real> overlap_geoderiv(const ShellBasis<Real> &basis,
                                   const std::array<int, 3> &na,
                                   const std::array<int, 3> &nb) {
  const int nao = basis.nao;
  std::vector<Real> G(static_cast<std::size_t>(nao) * nao, Real(0));
  const int ns = static_cast<int>(basis.shells.size());
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const int la = sa.l, lb = sb.l, lb1 = lb + 1;
      std::vector<Real> f[3];
      for (int d = 0; d < 3; ++d)
        f[d] = detail::overlap_1d_deriv(sa.alpha, sa.center[d], sb.alpha, sb.center[d],
                                        la, lb, na[d], nb[d]);
      for (int ka = 0; ka < ncart(la); ++ka) {
        int a3[3];
        cart_comp(la, ka, a3[0], a3[1], a3[2]);
        for (int kb = 0; kb < ncart(lb); ++kb) {
          int b3[3];
          cart_comp(lb, kb, b3[0], b3[1], b3[2]);
          G[(basis.ao_off[a] + ka) * static_cast<std::size_t>(nao) + basis.ao_off[b] + kb] =
              f[0][a3[0] * lb1 + b3[0]] * f[1][a3[1] * lb1 + b3[1]] *
              f[2][a3[2] * lb1 + b3[2]];
        }
      }
    }
  return G;
}

} // namespace intti
