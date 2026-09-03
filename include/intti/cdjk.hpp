// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Cholesky-accelerated Coulomb and exchange builds: with
// (ab|cd) ~= sum_J L_ab^J L_cd^J and each vector viewed as a symmetric AO
// matrix, J = sum_J (D : L^J) L^J and K = sum_J L^J D L^J — the standard
// CD/RI Fock construction. This replaces the quartic screened exchange
// build (kbuild.hpp) whenever a decomposition is on hand.

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cholesky.hpp"
#include "fock.hpp"

namespace intti {

/// Two-step Cholesky decomposition over the triangular shell-pair set of a
/// basis; records the pair -> shell bookkeeping needed by cholesky_jk.
template <class Real>
CholeskyBasis<Real> two_step_cholesky(const ShellBasis<Real> &basis,
                                      const TGrid<Real> &grid,
                                      const CholeskyOptions<Real> &opt = {}) {
  std::vector<ShellPair<Real>> plist;
  std::vector<std::pair<int, int>> pshell;
  make_shell_pairs(basis, plist, pshell);
  auto tab = make_pair_table(plist);
  QuartetWorkspace<Real> ws;
  auto cb = two_step_cholesky(tab, grid, ws, opt);
  cb.pair_shells = pshell;
  return cb;
}

/// ShellBasis-level pivoted Cholesky that records the pair -> shell bookkeeping
/// cholesky_jk needs. Precision-generic: dispatches to the batched path for
/// kokkos scalars and the serial eri_quartet path otherwise (__float128 / MPFR).
template <class Real>
CholeskyBasis<Real> pivoted_cholesky(const ShellBasis<Real> &basis,
                                     const TGrid<real_t<Real>> &grid,
                                     const CholeskyOptions<Real> &opt = {}) {
  std::vector<ShellPair<Real>> plist;
  std::vector<std::pair<int, int>> pshell;
  make_shell_pairs(basis, plist, pshell);
  auto cb = pivoted_cholesky(plist, grid, opt);
  cb.pair_shells = pshell;
  return cb;
}

/// J and/or K from Cholesky vectors; D must be symmetric (nao x nao,
/// row-major); either output pointer may be null. J costs O(naux nao^2),
/// K costs O(naux nao^3) via BLAS.
template <class Real>
void cholesky_jk(const ShellBasis<Real> &basis, const CholeskyBasis<Real> &cb,
                 const Real *D, Real *J, Real *K) {
  // Precision-generic: BLAS for float/double, a triple-loop matmul otherwise
  // (long double / __float128 / MPFR), via detail::matmul_nn.
  if (cb.pair_shells.empty())
    throw std::invalid_argument(
        "cholesky_jk: CholeskyBasis lacks pair bookkeeping; use the "
        "ShellBasis-level two_step_cholesky overload");
  const int nao = basis.nao;
  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
  if (J)
    for (std::size_t i = 0; i < n2; ++i)
      J[i] = 0;
  if (K)
    for (std::size_t i = 0; i < n2; ++i)
      K[i] = 0;
  std::vector<Real> LJ(n2), X(n2);
  const int npair = static_cast<int>(cb.pair_shells.size());
  for (int Jx = 0; Jx < cb.naux; ++Jx) {
    // unpack the vector into a symmetric AO matrix
    for (std::size_t i = 0; i < n2; ++i)
      LJ[i] = 0;
    for (int p = 0; p < npair; ++p) {
      const auto [i, j] = cb.pair_shells[p];
      const int la = basis.shells[i].l, lb = basis.shells[j].l;
      const int ncb2 = ncart(lb);
      for (int ka = 0; ka < ncart(la); ++ka)
        for (int kb = 0; kb < ncb2; ++kb) {
          const Real v = cb.L(cb.prod_offset[p] + ka * ncb2 + kb, Jx);
          const int r = basis.ao_off[i] + ka, c = basis.ao_off[j] + kb;
          LJ[static_cast<std::size_t>(r) * nao + c] = v;
          LJ[static_cast<std::size_t>(c) * nao + r] = v;
        }
    }
    if (J) {
      Real cJ = 0;
      for (std::size_t i = 0; i < n2; ++i)
        cJ += D[i] * LJ[i];
      for (std::size_t i = 0; i < n2; ++i)
        J[i] += cJ * LJ[i];
    }
    if (K) {
      // X = LJ * D, K += X * LJ; all matrices symmetric, so the row/column
      // major distinction cancels in the final product
      detail::matmul_nn(nao, LJ.data(), D, X.data(), Real(0));
      detail::matmul_nn(nao, X.data(), LJ.data(), K, Real(1));
    }
  }
}

} // namespace intti
