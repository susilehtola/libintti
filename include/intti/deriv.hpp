// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Derivative integrals: the black-box enabler (analytic forces, Hessians,
// magnetic response). The McMurchie-Davidson centre-shift relation makes
// every derivative a linear combination of shifted-angular-momentum
// integrals from the existing engine. Differentiating an AO w.r.t. the
// electron coordinate raises/lowers its Cartesian power:
//   d/dx [ (x-A)^l e^{-a(x-A)^2} ] = l (x-A)^{l-1} e^{...} - 2a (x-A)^{l+1} e^{...}
// so <nabla_x mu|nu> = l_x S(mu[l_x-1], nu) - 2 a_mu S(mu[l_x+1], nu), etc.
// These are the libcint int1e_ip* integrals (gradient on the bra).
//
// Matrix-level API: whole nao x nao gradient-component matrices.

#include <array>
#include <cstddef>
#include <vector>

#include "fock.hpp"
#include "gto.hpp"
#include "oneel.hpp" // detail::overlap_1d, kinetic_1d

namespace intti {

namespace detail {
/// Shift operator D[F](i,j) = i*F(i-1,j) - 2 alpha * F(i+1,j) on the bra
/// index i, i.e. d/dx acting on the bra 1D factor. F is a flat (la+2)x(lbx+1)
/// table (bra extended by 1). Returns the derivative 1D table (la+1)x(lb+1).
template <class Real>
void bra_shift(const std::vector<Real> &F, int lbx, int la, int lb, Real alpha,
               std::vector<Real> &dF) {
  dF.assign(static_cast<std::size_t>(la + 1) * (lb + 1), Real(0));
  auto at = [&](int i, int j) { return F[i * (lbx + 1) + j]; };
  for (int i = 0; i <= la; ++i)
    for (int j = 0; j <= lb; ++j) {
      Real v = -2 * alpha * at(i + 1, j);
      if (i >= 1) v += Real(i) * at(i - 1, j);
      dF[i * (lb + 1) + j] = v;
    }
}
} // namespace detail

/// Overlap gradient <nabla mu | nu>: three nao x nao matrices (x, y, z),
/// matching PySCF int1e_ipovlp.
template <class Real>
std::array<std::vector<Real>, 3> overlap_deriv(const ShellBasis<Real> &basis) {
  const int nao = basis.nao;
  std::array<std::vector<Real>, 3> G;
  for (auto &g : G) g.assign(static_cast<std::size_t>(nao) * nao, Real(0));
  const int ns = static_cast<int>(basis.shells.size());
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const int la = sa.l, lb = sb.l, lb1 = lb + 1;
      std::vector<Real> s[3], ds[3];
      int lbx;
      for (int d = 0; d < 3; ++d) {
        detail::overlap_1d(sa.alpha, sa.center[d], sb.alpha, sb.center[d], la, lb, 1, 0,
                           s[d], lbx);
        detail::bra_shift(s[d], lbx, la, lb, sa.alpha, ds[d]);
      }
      auto S = [&](int d, int i, int j) { return s[d][i * (lbx + 1) + j]; };
      auto DS = [&](int d, int i, int j) { return ds[d][i * lb1 + j]; };
      for (int ka = 0; ka < ncart(la); ++ka) {
        int a3[3];
        cart_comp(la, ka, a3[0], a3[1], a3[2]);
        for (int kb = 0; kb < ncart(lb); ++kb) {
          int b3[3];
          cart_comp(lb, kb, b3[0], b3[1], b3[2]);
          const std::size_t idx =
              (basis.ao_off[a] + ka) * static_cast<std::size_t>(nao) + basis.ao_off[b] + kb;
          G[0][idx] = DS(0, a3[0], b3[0]) * S(1, a3[1], b3[1]) * S(2, a3[2], b3[2]);
          G[1][idx] = S(0, a3[0], b3[0]) * DS(1, a3[1], b3[1]) * S(2, a3[2], b3[2]);
          G[2][idx] = S(0, a3[0], b3[0]) * S(1, a3[1], b3[1]) * DS(2, a3[2], b3[2]);
        }
      }
    }
  return G;
}

/// Kinetic-energy gradient <nabla mu | T | nu>, matching PySCF int1e_ipkin.
template <class Real>
std::array<std::vector<Real>, 3> kinetic_deriv(const ShellBasis<Real> &basis) {
  const int nao = basis.nao;
  std::array<std::vector<Real>, 3> G;
  for (auto &g : G) g.assign(static_cast<std::size_t>(nao) * nao, Real(0));
  const int ns = static_cast<int>(basis.shells.size());
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const int la = sa.l, lb = sb.l, lb1 = lb + 1;
      std::vector<Real> s[3], t[3], ds[3], dt[3];
      int lbx;
      for (int d = 0; d < 3; ++d) {
        // overlap up to bra la+1, ket lb+2 (kinetic needs ket +2)
        detail::overlap_1d(sa.alpha, sa.center[d], sb.alpha, sb.center[d], la, lb, 1, 2,
                           s[d], lbx);
        detail::kinetic_1d(s[d], lbx, la + 1, lb, sb.alpha, t[d]); // t: (la+2)x(lb+1)
        detail::bra_shift(s[d], lbx, la, lb, sa.alpha, ds[d]);     // dS: (la+1)x(lb+1)
        // dT from t (bra index up to la+1): reuse bra_shift with stride lb1
        dt[d].assign(static_cast<std::size_t>(la + 1) * (lb + 1), Real(0));
        auto T = [&](int i, int j) { return t[d][i * lb1 + j]; };
        for (int i = 0; i <= la; ++i)
          for (int j = 0; j <= lb; ++j) {
            Real v = -2 * sa.alpha * T(i + 1, j);
            if (i >= 1) v += Real(i) * T(i - 1, j);
            dt[d][i * lb1 + j] = v;
          }
      }
      auto S = [&](int d, int i, int j) { return s[d][i * (lbx + 1) + j]; };
      auto Tk = [&](int d, int i, int j) { return t[d][i * lb1 + j]; };
      auto DS = [&](int d, int i, int j) { return ds[d][i * lb1 + j]; };
      auto DT = [&](int d, int i, int j) { return dt[d][i * lb1 + j]; };
      for (int ka = 0; ka < ncart(la); ++ka) {
        int a3[3];
        cart_comp(la, ka, a3[0], a3[1], a3[2]);
        for (int kb = 0; kb < ncart(lb); ++kb) {
          int b3[3];
          cart_comp(lb, kb, b3[0], b3[1], b3[2]);
          const std::size_t idx =
              (basis.ao_off[a] + ka) * static_cast<std::size_t>(nao) + basis.ao_off[b] + kb;
          const int ax = a3[0], ay = a3[1], az = a3[2];
          const int bx = b3[0], by = b3[1], bz = b3[2];
          // T = Tx Sy Sz + Sx Ty Sz + Sx Sy Tz; apply nabla to the bra per dir
          G[0][idx] = DT(0, ax, bx) * S(1, ay, by) * S(2, az, bz) +
                      DS(0, ax, bx) * Tk(1, ay, by) * S(2, az, bz) +
                      DS(0, ax, bx) * S(1, ay, by) * Tk(2, az, bz);
          G[1][idx] = Tk(0, ax, bx) * DS(1, ay, by) * S(2, az, bz) +
                      S(0, ax, bx) * DT(1, ay, by) * S(2, az, bz) +
                      S(0, ax, bx) * DS(1, ay, by) * Tk(2, az, bz);
          G[2][idx] = Tk(0, ax, bx) * S(1, ay, by) * DS(2, az, bz) +
                      S(0, ax, bx) * Tk(1, ay, by) * DS(2, az, bz) +
                      S(0, ax, bx) * S(1, ay, by) * DT(2, az, bz);
        }
      }
    }
  return G;
}

} // namespace intti
