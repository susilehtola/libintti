// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Density-fitting (RI) J and K builds. With the 2-center metric (P|Q) and the
// 3-center integrals (mu nu|P) from ncenter.hpp, form the fitted vectors
//   B_{mu nu}^P = sum_Q (mu nu|Q) (M^{-1/2})_{QP},   M_{PQ} = (P|Q),
// so that (mu nu|la si) ~= sum_P B_{mu nu}^P B_{la si}^P. Then
//   J_{mu nu} = sum_P B_{mu nu}^P (sum_{la si} B_{la si}^P D_{la si}),
//   K_{mu nu} = sum_P (B^P D B^P)_{mu nu},
// exactly the Cholesky-vector contraction pattern of cdjk.hpp. Matrix-level
// API: density in, J/K out; the fit vectors are an opaque intermediate.
//
// The auxiliary basis may be an external RI set or one generated in-library by
// two_step_cholesky (cholesky.hpp) -- both are just ShellBasis inputs here.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "blas.hpp"     // detail::gemm
#include "cholesky.hpp" // detail::syevd
#include "fock.hpp"
#include "ncenter.hpp"
#include "tgrid.hpp"

namespace intti {

/// Fitted RI vectors B (nao*nao x naux, row-major B[(mu*nao+nu)*naux+P]).
template <class Real> struct RIFit {
  int nao{0}, naux{0};
  std::vector<Real> B;
};

/// Build the RI fit for an orbital basis against an auxiliary basis.
/// tau_lin drops metric eigenvalues below tau_lin*max (linear-dependence).
template <class Real>
RIFit<Real> ri_fit(const ShellBasis<Real> &orb, const ShellBasis<Real> &aux,
                   const TGrid<Real> &grid, Real tau_lin = Real(1e-10)) {
  static_assert(std::is_same_v<Real, double> || std::is_same_v<Real, float>,
                "ri_fit requires float or double (LAPACK)");
  const int nao = orb.nao, naux = aux.nao;
  auto M = coulomb_2c(aux, grid);                    // naux x naux
  auto T = coulomb_3c(orb, aux, grid);               // nao*nao x naux
  // M^{-1/2} with eigenvalue cutoff
  std::vector<Real> eval(naux);
  detail::syevd(naux, M.data(), eval.data());        // M now holds eigenvectors V
  Real emax = 0;
  for (Real e : eval) emax = std::max(emax, e);
  std::vector<Real> Mhalf(static_cast<std::size_t>(naux) * naux, Real(0));
  // syevd returns eigenvectors as columns in column-major storage, so
  // eigenvector k, component P is M[k*naux + P] -- i.e. row-major M holds the
  // eigenvectors as rows. Scale each eigenvector row by 1/√e (0 if dropped),
  // then Mhalf = V^T diag(1/√e) V = M^T . Vs.
  std::vector<Real> Vs(M.begin(), M.end());
  for (int k = 0; k < naux; ++k) {
    const Real s = (eval[k] <= tau_lin * emax) ? Real(0)
                                               : Real(1) / std::sqrt(eval[k]);
    for (int P = 0; P < naux; ++P) Vs[k * naux + P] *= s;
  }
  detail::gemm('T', 'N', naux, naux, naux, Real(1), M.data(), naux, Vs.data(),
               naux, Real(0), Mhalf.data(), naux);
  // B = T . Mhalf  (contract over Q)
  RIFit<Real> fit;
  fit.nao = nao;
  fit.naux = naux;
  fit.B.assign(static_cast<std::size_t>(nao) * nao * naux, Real(0));
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  // B[N x naux] = T[N x naux] . Mhalf[naux x naux]
  detail::gemm('N', 'N', static_cast<int>(N), naux, naux, Real(1), T.data(),
               naux, Mhalf.data(), naux, Real(0), fit.B.data(), naux);
  return fit;
}

/// Memory-lean tiled RI-J that never materialises the nao^2 x naux 3-index
/// tensor (nor the fit vectors B): the largest RI object, and the real
/// "too big for GPU/device memory" case. RI-J is a sum over the auxiliary index,
///   J_{mu nu} = sum_P (mu nu|P) d_P,   d = M^{-1} gamma,
///   gamma_Q = sum_{mu nu} (mu nu|Q) D_{mu nu},   M_{PQ} = (P|Q),
/// so it factors into two auxiliary passes that each need only ONE 3-center
/// block (mu nu|P-tile) live at a time (aux_tile_shells auxiliary shells per
/// tile), plus O(naux) intermediates and the naux x naux metric. Peak 3-center
/// memory is nao^2 x (tile aux AOs) instead of nao^2 x naux. Mathematically
/// identical to ri_jk's J (B M^{-1/2} contraction gives the same M^{-1});
/// the per-tile GEMM grouping changes the P-summation grouping, so the result
/// matches the untiled build and is tile-size-independent to rounding (not
/// bit-exact, unlike the exchange row-tiling). tau_lin drops metric eigenvalues
/// below tau_lin*max (linear dependence), as in ri_fit.
template <class Real>
void ri_j_tiled(const ShellBasis<Real> &orb, const ShellBasis<Real> &aux,
                const TGrid<Real> &grid, const Real *D, Real *J,
                int aux_tile_shells, Real tau_lin = Real(1e-10)) {
  static_assert(std::is_same_v<Real, double> || std::is_same_v<Real, float>,
                "ri_j_tiled requires float or double (LAPACK)");
  const int nao = orb.nao, naux = aux.nao;
  const int nsa = static_cast<int>(aux.shells.size());
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  if (aux_tile_shells < 1) aux_tile_shells = nsa;
  // metric pseudo-inverse M^{-1} = V^T diag(1/e) V with an eigenvalue cutoff
  auto M = coulomb_2c(aux, grid);
  std::vector<Real> eval(naux);
  detail::syevd(naux, M.data(), eval.data()); // M now holds eigenvectors (rows)
  Real emax = 0;
  for (Real e : eval) emax = std::max(emax, e);
  std::vector<Real> Vs(M.begin(), M.end());
  for (int k = 0; k < naux; ++k) {
    const Real s = (eval[k] <= tau_lin * emax) ? Real(0) : Real(1) / eval[k];
    for (int P = 0; P < naux; ++P) Vs[k * naux + P] *= s;
  }
  std::vector<Real> Minv(static_cast<std::size_t>(naux) * naux, Real(0));
  detail::gemm('T', 'N', naux, naux, naux, Real(1), M.data(), naux, Vs.data(), naux, Real(0),
               Minv.data(), naux);

  // pass 1: gamma_P = sum_{mu nu} (mu nu|P) D_{mu nu}, one aux tile at a time
  std::vector<Real> gamma(naux, Real(0));
  for (int A0 = 0; A0 < nsa; A0 += aux_tile_shells) {
    const int A1 = std::min(A0 + aux_tile_shells, nsa);
    const int p0 = aux.ao_off[A0], blk = aux.ao_off[A1] - p0;
    auto Tblk = coulomb_3c_auxblock(orb, aux, grid, A0, A1); // N x blk
    // gamma_blk[blk] = Tblk^T . D
    detail::gemm('T', 'N', blk, 1, static_cast<int>(N), Real(1), Tblk.data(), blk, D, 1,
                 Real(0), gamma.data() + p0, 1);
  }
  // d = M^{-1} gamma
  std::vector<Real> d(naux, Real(0));
  detail::gemm('N', 'N', naux, 1, naux, Real(1), Minv.data(), naux, gamma.data(), 1, Real(0),
               d.data(), 1);
  // pass 2: J_{mu nu} = sum_P (mu nu|P) d_P, one aux tile at a time
  for (std::size_t i = 0; i < N; ++i) J[i] = 0;
  for (int A0 = 0; A0 < nsa; A0 += aux_tile_shells) {
    const int A1 = std::min(A0 + aux_tile_shells, nsa);
    const int p0 = aux.ao_off[A0], blk = aux.ao_off[A1] - p0;
    auto Tblk = coulomb_3c_auxblock(orb, aux, grid, A0, A1); // N x blk
    // J += Tblk . d_blk
    detail::gemm('N', 'N', static_cast<int>(N), 1, blk, Real(1), Tblk.data(), blk, d.data() + p0,
                 1, Real(1), J, 1);
  }
}

/// RI Coulomb and/or exchange from a symmetric density D (nao x nao,
/// row-major). Either output pointer may be null.
template <class Real>
void ri_jk(const RIFit<Real> &fit, const Real *D, Real *J, Real *K) {
  const int nao = fit.nao, naux = fit.naux;
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  if (J) {
    // c_P = sum_{mn} B_{mn}^P D_{mn};  J_{mn} = sum_P B_{mn}^P c_P
    std::vector<Real> c(naux, Real(0));
    for (std::size_t mn = 0; mn < N; ++mn)
      for (int P = 0; P < naux; ++P)
        c[P] += fit.B[mn * naux + P] * D[mn];
    for (std::size_t mn = 0; mn < N; ++mn) {
      Real s = 0;
      for (int P = 0; P < naux; ++P)
        s += fit.B[mn * naux + P] * c[P];
      J[mn] = s;
    }
  }
  if (K) {
    for (std::size_t i = 0; i < N; ++i)
      K[i] = 0;
    // K_{mu nu} = sum_P (B^P D B^P)_{mu nu}; B^P is nao x nao at stride naux
    std::vector<Real> BP(N), BD(N);
    for (int P = 0; P < naux; ++P) {
      // Gather the strided slice B^P into a contiguous nao x nao scratch.
      for (std::size_t mn = 0; mn < N; ++mn)
        BP[mn] = fit.B[mn * naux + P];
      // BD = B^P D
      detail::gemm('N', 'N', nao, nao, nao, Real(1), BP.data(), nao, D, nao,
                   Real(0), BD.data(), nao);
      // K += BD B^P
      detail::gemm('N', 'N', nao, nao, nao, Real(1), BD.data(), nao, BP.data(),
                   nao, Real(1), K, nao);
    }
  }
}

/// Occupation-driven RI exchange: the "optimal" RI-K contraction. Given
/// occupied MO coefficients C (nao x nocc, row-major C[mu*nocc+i]), contract
/// the fit vectors with the orbitals early (half-transform W_{i mu}^P =
/// sum_nu B_{mu nu}^P C_{nu i}) so the cost scales with nocc, not nao:
///   K_{mu nu} = sum_P sum_i W_{i mu}^P W_{i nu}^P.
/// Identical to ri_jk's K for D = sum_i C_i C_i^T, but cheaper when
/// nocc < nao and the natural form for orbital-driven methods.
template <class Real>
void ri_k_occ(const RIFit<Real> &fit, const Real *C, int nocc, Real *K) {
  const int nao = fit.nao, naux = fit.naux;
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  for (std::size_t i = 0; i < N; ++i)
    K[i] = 0;
  std::vector<Real> BP(N);                                    // gathered B^P
  std::vector<Real> X(static_cast<std::size_t>(nao) * nocc);  // half-transform
  for (int P = 0; P < naux; ++P) {
    // Gather the strided slice B^P into a contiguous nao x nao scratch.
    for (std::size_t mn = 0; mn < N; ++mn)
      BP[mn] = fit.B[mn * naux + P];
    // X_{mu i} = sum_nu B^P_{mu nu} C_{nu i}  (nao x nocc)
    detail::gemm('N', 'N', nao, nocc, nao, Real(1), BP.data(), nao, C, nocc,
                 Real(0), X.data(), nocc);
    // K += X X^T  (K_{mu nu} += sum_i X_{mu i} X_{nu i})
    detail::gemm('N', 'T', nao, nao, nocc, Real(1), X.data(), nocc, X.data(),
                 nocc, Real(1), K, nao);
  }
}

} // namespace intti
