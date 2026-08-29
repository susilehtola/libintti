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

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#include "fock.hpp"
#include "gto.hpp"
#include "hermite1d.hpp"
#include "math.hpp"
#include "nuclear.hpp" // PointCharge
#include "oneel.hpp"   // detail::overlap_1d
#include "tgrid.hpp"

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

/// Apply na bra-shifts then nb ket-shifts to a 1D factor table F stored as
/// (ib0+1) x (jb0+1) row-major; returns the reduced (ib0-na+1) x (jb0-nb+1)
/// table. This is the shift core reused by every operator.
template <class Real>
std::vector<Real> apply_shifts(std::vector<Real> F, int ib0, int jb0, int na,
                               int nb, Real alpha, Real beta) {
  int ib = ib0, jb = jb0, ld = jb0 + 1;
  std::vector<Real> nxt;
  for (int k = 0; k < na; ++k) {
    shift_bra(F, ld, ib, jb, alpha, nxt);
    --ib;
    ld = jb + 1;
    F = nxt;
  }
  for (int k = 0; k < nb; ++k) {
    shift_ket(F, ld, ib, jb, beta, nxt);
    --jb;
    ld = jb + 1;
    F = nxt;
  }
  return F;
}

/// 1D overlap factor differentiated na times on the bra centre and nb times on
/// the ket centre; returns a (la+1) x (lb+1) table (row-major).
template <class Real>
std::vector<Real> overlap_1d_deriv(Real alpha, Real A, Real beta, Real B, int la,
                                   int lb, int na, int nb) {
  std::vector<Real> s;
  int lbx;
  overlap_1d(alpha, A, beta, B, la, lb, na, nb, s, lbx); // s: (la+na+1)x(lbx+1)
  return apply_shifts(std::move(s), la + na, lb + nb, na, nb, alpha, beta);
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

/// Geometric derivative of the kinetic-energy matrix, d^na_A d^nb_B T. The
/// same shift core applied to the overlap and kinetic 1D factors of
/// T = Tx Sy Sz + Sx Ty Sz + Sx Sy Tz.
template <class Real>
std::vector<Real> kinetic_geoderiv(const ShellBasis<Real> &basis,
                                   const std::array<int, 3> &na,
                                   const std::array<int, 3> &nb) {
  const int nao = basis.nao;
  std::vector<Real> G(static_cast<std::size_t>(nao) * nao, Real(0));
  const int ns = static_cast<int>(basis.shells.size());
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const int la = sa.l, lb = sb.l, lb1 = lb + 1;
      std::vector<Real> Sf[3], Tf[3]; // shifted 1D overlap and kinetic factors
      for (int d = 0; d < 3; ++d) {
        std::vector<Real> s2, sS;
        int lbx2, lbxS;
        // overlap for the kinetic recursion (ket +2) and for the S factors
        detail::overlap_1d(sa.alpha, sa.center[d], sb.alpha, sb.center[d], la, lb,
                           na[d], nb[d] + 2, s2, lbx2);
        detail::overlap_1d(sa.alpha, sa.center[d], sb.alpha, sb.center[d], la, lb,
                           na[d], nb[d], sS, lbxS);
        std::vector<Real> t1;
        detail::kinetic_1d(s2, lbx2, la + na[d], lb + nb[d], sb.alpha, t1);
        Sf[d] = detail::apply_shifts(std::move(sS), la + na[d], lb + nb[d], na[d],
                                     nb[d], sa.alpha, sb.alpha);
        Tf[d] = detail::apply_shifts(std::move(t1), la + na[d], lb + nb[d], na[d],
                                     nb[d], sa.alpha, sb.alpha);
      }
      for (int ka = 0; ka < ncart(la); ++ka) {
        int a3[3];
        cart_comp(la, ka, a3[0], a3[1], a3[2]);
        for (int kb = 0; kb < ncart(lb); ++kb) {
          int b3[3];
          cart_comp(lb, kb, b3[0], b3[1], b3[2]);
          auto S = [&](int d) { return Sf[d][a3[d] * lb1 + b3[d]]; };
          auto T = [&](int d) { return Tf[d][a3[d] * lb1 + b3[d]]; };
          G[(basis.ao_off[a] + ka) * static_cast<std::size_t>(nao) + basis.ao_off[b] + kb] =
              T(0) * S(1) * S(2) + S(0) * T(1) * S(2) + S(0) * S(1) * T(2);
        }
      }
    }
  return G;
}

/// Geometric derivative of the nuclear-attraction matrix,
/// d^na_A d^nb_B <a| sum_C w_C/|r-R_C| |b>. The kernel-based (t-quadrature)
/// operator: the shift core is applied to the nuclear 1D factor g_d(t) at
/// each t node (differentiation analytic per node), then summed over t.
template <class Real>
std::vector<Real> nuclear_geoderiv(const ShellBasis<Real> &basis,
                                   const std::vector<PointCharge<Real>> &charges,
                                   const TGrid<Real> &grid,
                                   const std::array<int, 3> &na,
                                   const std::array<int, 3> &nb) {
  const int nao = basis.nao;
  std::vector<Real> G(static_cast<std::size_t>(nao) * nao, Real(0));
  const Real pi = pi_v<Real>();
  const int ns = static_cast<int>(basis.shells.size());
  const int nt = grid.n();
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const int la = sa.l, lb = sb.l, lb1 = lb + 1;
      const int lae[3] = {la + na[0], la + na[1], la + na[2]};
      const int lbe[3] = {lb + nb[0], lb + nb[1], lb + nb[2]};
      const Real p = sa.alpha + sb.alpha, mu = sa.alpha * sb.alpha / p;
      Real Pd[3];
      std::vector<Real> E[3];
      for (int d = 0; d < 3; ++d) {
        Pd[d] = (sa.alpha * sa.center[d] + sb.alpha * sb.center[d]) / p;
        const Real ab = sa.center[d] - sb.center[d];
        E[d].assign(static_cast<std::size_t>(lae[d] + 1) * (lbe[d] + 1) * (lae[d] + lbe[d] + 1),
                    Real(0));
        e_coeffs(lae[d], lbe[d], p, Pd[d] - sa.center[d], Pd[d] - sb.center[d],
                 exp_(-mu * ab * ab), E[d].data());
      }
      const int nca = ncart(la), ncb = ncart(lb);
      std::vector<Real> acc(static_cast<std::size_t>(nca) * ncb, Real(0));
      std::vector<Real> Bv(la + lb + std::max({na[0], na[1], na[2], nb[0], nb[1], nb[2]}) * 2 + 1);
      for (const auto &c : charges)
        for (int it = 0; it < nt; ++it) {
          const Real t = grid.t[it], denom = p + t * t;
          const Real theta = p * t * t / denom, pref = sqrt_(pi / denom);
          const Real wt = grid.w[it] * c.weight;
          // per direction: build g_d table then apply the derivative shifts
          std::vector<Real> gsh[3];
          for (int d = 0; d < 3; ++d) {
            const int L = lae[d] + lbe[d];
            hermite_b(L, theta, Pd[d] - c.R[d], Bv.data());
            std::vector<Real> g(static_cast<std::size_t>(lae[d] + 1) * (lbe[d] + 1));
            for (int i = 0; i <= lae[d]; ++i)
              for (int j = 0; j <= lbe[d]; ++j) {
                Real s = 0;
                for (int tau = 0; tau <= i + j; ++tau)
                  s += E[d][(i * (lbe[d] + 1) + j) * (L + 1) + tau] * Bv[tau];
                g[i * (lbe[d] + 1) + j] = pref * s;
              }
            gsh[d] = detail::apply_shifts(std::move(g), lae[d], lbe[d], na[d], nb[d],
                                          sa.alpha, sb.alpha);
          }
          for (int ka = 0; ka < nca; ++ka) {
            int a3[3];
            cart_comp(la, ka, a3[0], a3[1], a3[2]);
            for (int kb = 0; kb < ncb; ++kb) {
              int b3[3];
              cart_comp(lb, kb, b3[0], b3[1], b3[2]);
              acc[ka * ncb + kb] += wt * gsh[0][a3[0] * lb1 + b3[0]] *
                                    gsh[1][a3[1] * lb1 + b3[1]] *
                                    gsh[2][a3[2] * lb1 + b3[2]];
            }
          }
        }
      for (int ka = 0; ka < nca; ++ka)
        for (int kb = 0; kb < ncb; ++kb)
          G[(basis.ao_off[a] + ka) * static_cast<std::size_t>(nao) + basis.ao_off[b] + kb] =
              acc[ka * ncb + kb];
    }
  return G;
}

} // namespace intti
