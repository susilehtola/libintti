// SPDX-License-Identifier: MPL-2.0
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

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

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
  // eigenvector k, component P is M[k*naux + P].
  for (int k = 0; k < naux; ++k) {
    if (eval[k] <= tau_lin * emax) continue;
    const Real s = Real(1) / std::sqrt(eval[k]);
    for (int P = 0; P < naux; ++P)
      for (int Q = 0; Q < naux; ++Q)
        Mhalf[P * naux + Q] += M[k * naux + P] * s * M[k * naux + Q]; // V diag(1/√e) V^T
  }
  // B = T . Mhalf  (contract over Q)
  RIFit<Real> fit;
  fit.nao = nao;
  fit.naux = naux;
  fit.B.assign(static_cast<std::size_t>(nao) * nao * naux, Real(0));
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  for (std::size_t mn = 0; mn < N; ++mn)
    for (int P = 0; P < naux; ++P) {
      Real s = 0;
      for (int Q = 0; Q < naux; ++Q)
        s += T[mn * naux + Q] * Mhalf[Q * naux + P];
      fit.B[mn * naux + P] = s;
    }
  return fit;
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
    std::vector<Real> BD(N);
    for (int P = 0; P < naux; ++P) {
      // BD = B^P D
      for (int mu = 0; mu < nao; ++mu)
        for (int la = 0; la < nao; ++la) {
          Real s = 0;
          for (int nu = 0; nu < nao; ++nu)
            s += fit.B[(static_cast<std::size_t>(mu) * nao + nu) * naux + P] *
                 D[static_cast<std::size_t>(nu) * nao + la];
          BD[static_cast<std::size_t>(mu) * nao + la] = s;
        }
      // K += BD B^P
      for (int mu = 0; mu < nao; ++mu)
        for (int nu = 0; nu < nao; ++nu) {
          Real s = 0;
          for (int la = 0; la < nao; ++la)
            s += BD[static_cast<std::size_t>(mu) * nao + la] *
                 fit.B[(static_cast<std::size_t>(la) * nao + nu) * naux + P];
          K[static_cast<std::size_t>(mu) * nao + nu] += s;
        }
    }
  }
}

} // namespace intti
