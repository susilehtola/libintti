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
#include "jk.hpp"
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

/// RI Coulomb and/or exchange from a density D (nao x nao, row-major). Either
/// output pointer may be null.
///
/// D need NOT be symmetric. Unlike the fused exchange_build, which computes the
/// upper triangle and mirrors it, the contraction here is K = sum_P B^P D B^P
/// as two GEMMs and assumes nothing about D -- so it is already correct for the
/// general and antisymmetric densities that response theory produces, and gives
/// K(D)^T = K(D^T) as it must. J likewise picks up only the symmetric part of D,
/// because B^P is symmetric in (mu,nu), so J(antisymmetric D) is exactly zero.
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

/// Multi-density RI J/K: the same interface as jk_build (jk.hpp), one pass over
/// the fit vectors for all requests. The saving over calling ri_jk per density
/// is the gather: B is stored strided by naux, so each B^P slice costs an
/// O(nao^2) gather that ri_jk repeats for every density. Here it is gathered
/// once per P and applied to all of them, and the Coulomb half collapses to two
/// GEMMs over the whole request set.
///
/// DensitySymmetry is accepted for interface uniformity but not needed: the RI
/// contraction is general in D already (see ri_jk).
template <class Real>
JKResult<Real> ri_jk_build(const RIFit<Real> &fit,
                           const std::vector<JKRequest<Real>> &reqs) {
  const int nao = fit.nao, naux = fit.naux;
  const int nreq = static_cast<int>(reqs.size());
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  JKResult<Real> res;
  res.J.resize(nreq);
  res.K.resize(nreq);
  std::vector<int> wantJ(nreq), wantK(nreq);
  for (int r = 0; r < nreq; ++r) {
    wantJ[r] = reqs[r].terms != FockTerms::Exchange;
    wantK[r] = reqs[r].terms != FockTerms::Coulomb;
    if (wantJ[r]) res.J[r].assign(N, Real(0));
    if (wantK[r]) res.K[r].assign(N, Real(0));
  }
  // Coulomb: c(P, r) = sum_mn B(mn, P) D_r(mn); J_r(mn) = sum_P B(mn, P) c(P, r)
  int njr = 0;
  for (int r = 0; r < nreq; ++r) njr += wantJ[r];
  if (njr > 0) {
    std::vector<Real> Dm(N * njr), c(static_cast<std::size_t>(naux) * njr), Jm(N * njr);
    int col = 0;
    for (int r = 0; r < nreq; ++r) {
      if (!wantJ[r]) continue;
      for (std::size_t i = 0; i < N; ++i) Dm[i * njr + col] = reqs[r].D[i];
      ++col;
    }
    detail::gemm('T', 'N', naux, njr, static_cast<int>(N), Real(1), fit.B.data(), naux,
                 Dm.data(), njr, Real(0), c.data(), njr);
    detail::gemm('N', 'N', static_cast<int>(N), njr, naux, Real(1), fit.B.data(), naux,
                 c.data(), njr, Real(0), Jm.data(), njr);
    col = 0;
    for (int r = 0; r < nreq; ++r) {
      if (!wantJ[r]) continue;
      for (std::size_t i = 0; i < N; ++i) res.J[r][i] = Jm[i * njr + col];
      ++col;
    }
  }
  // Exchange: K_r += B^P D_r B^P, gathering B^P once for the whole request set
  int nkr = 0;
  for (int r = 0; r < nreq; ++r) nkr += wantK[r];
  if (nkr > 0) {
    std::vector<Real> BP(N), Dstack(static_cast<std::size_t>(nao) * nao * nkr);
    std::vector<Real> BD(static_cast<std::size_t>(nao) * nao * nkr);
    // densities side by side: Dstack is nao x (nkr*nao)
    {
      int col = 0;
      for (int r = 0; r < nreq; ++r) {
        if (!wantK[r]) continue;
        for (int i = 0; i < nao; ++i)
          for (int j = 0; j < nao; ++j)
            Dstack[static_cast<std::size_t>(i) * nkr * nao + col * nao + j] =
                reqs[r].D[static_cast<std::size_t>(i) * nao + j];
        ++col;
      }
    }
    for (int P = 0; P < naux; ++P) {
      for (std::size_t mn = 0; mn < N; ++mn) BP[mn] = fit.B[mn * naux + P];
      detail::gemm('N', 'N', nao, nkr * nao, nao, Real(1), BP.data(), nao, Dstack.data(),
                   nkr * nao, Real(0), BD.data(), nkr * nao);
      int col = 0;
      for (int r = 0; r < nreq; ++r) {
        if (!wantK[r]) continue;
        detail::gemm('N', 'N', nao, nao, nao, Real(1), BD.data() + col * nao, nkr * nao,
                     BP.data(), nao, Real(1), res.K[r].data(), nao);
        ++col;
      }
    }
  }
  return res;
}

/// Two-sided occupation-driven RI exchange, the general form of ri_k_occ: for a
/// density factorised as D = C_L C_R^T (nao x nvec each, row-major),
///   K = sum_P (B^P C_L)(B^P C_R)^T,
/// which is B^P D B^P because B^P is symmetric. C_L = C_R recovers ri_k_occ.
///
/// This is the case where the orbital-driven form genuinely matters: a CPHF
/// perturbed density is NOT idempotent and is not a single orbital product, but
/// it IS a product of two different orbital sets -- which is exactly why psi4's
/// JK takes C_left and C_right. Cost scales with nvec rather than nao.
template <class Real>
void ri_k_occ2(const RIFit<Real> &fit, const Real *CL, const Real *CR, int nvec, Real *K) {
  const int nao = fit.nao, naux = fit.naux;
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  for (std::size_t i = 0; i < N; ++i) K[i] = Real(0);
  std::vector<Real> BP(N);
  std::vector<Real> XL(static_cast<std::size_t>(nao) * nvec);
  std::vector<Real> XR(static_cast<std::size_t>(nao) * nvec);
  for (int P = 0; P < naux; ++P) {
    for (std::size_t mn = 0; mn < N; ++mn) BP[mn] = fit.B[mn * naux + P];
    detail::gemm('N', 'N', nao, nvec, nao, Real(1), BP.data(), nao, CL, nvec, Real(0),
                 XL.data(), nvec);
    detail::gemm('N', 'N', nao, nvec, nao, Real(1), BP.data(), nao, CR, nvec, Real(0),
                 XR.data(), nvec);
    detail::gemm('N', 'T', nao, nao, nvec, Real(1), XL.data(), nvec, XR.data(), nvec,
                 Real(1), K, nao);
  }
}

/// Orbital-driven RI exchange with BOUNDED memory: never forms the nao^2 x naux
/// fit vectors. For a density factorised as D = C_L C_R^T,
///   X^P_{mk} = sum_l (ml|P) C_L[l,k],   Y^P_{nk} = sum_s (ns|P) C_R[s,k],
///   K_mn     = sum_{P,k} [M^{-1} X]^P_{mk} Y^P_{nk},
/// which is B^P D B^P written with the metric inverse instead of its square
/// root, so it agrees with ri_k_occ2 to rounding.
///
/// TWO independent tilings, and which axis does what is the whole point:
///   * the AUXILIARY axis tiles the three-centre integrals, exactly as in
///     ri_j_tiled -- one (mu nu|P-tile) block live at a time;
///   * the VECTOR axis tiles everything else. The metric solve couples all
///     auxiliary functions, so P cannot be blocked across it, but K is a plain
///     sum over k, so vector tiles simply accumulate.
/// Peak storage is 3 x naux x nao x vec_tile plus one three-centre tile, versus
/// nao^2 x naux for the fit vectors. Setting both tiles to their full extent
/// recovers the dense computation exactly -- the dense case is one block.
///
/// Cost of the trade, stated: the three-centre integrals are recomputed once per
/// vector tile, so the integral work scales as ceil(nvec/vec_tile). Choose the
/// largest vector tile that fits.
template <class Real>
void ri_k_occ_tiled(const ShellBasis<Real> &orb, const ShellBasis<Real> &aux,
                    const TGrid<Real> &grid, const Real *CL, const Real *CR, int nvec,
                    Real *K, int aux_tile_shells = 0, int vec_tile = 0,
                    Real tau_lin = Real(1e-10)) {
  static_assert(std::is_same_v<Real, double> || std::is_same_v<Real, float>,
                "ri_k_occ_tiled requires float or double (LAPACK)");
  const int nao = orb.nao, naux = aux.nao;
  const int nsa = static_cast<int>(aux.shells.size());
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  if (aux_tile_shells < 1) aux_tile_shells = nsa;
  if (vec_tile < 1) vec_tile = nvec;
  for (std::size_t i = 0; i < N; ++i) K[i] = Real(0);
  if (nvec == 0) return;

  // metric pseudo-inverse, as in ri_j_tiled
  auto M = coulomb_2c(aux, grid);
  std::vector<Real> eval(naux);
  detail::syevd(naux, M.data(), eval.data()); // M now holds eigenvectors (rows)
  Real emax = 0;
  for (Real e : eval) emax = std::max(emax, e);
  std::vector<Real> Vs(M.begin(), M.end());
  for (int k = 0; k < naux; ++k) {
    const Real sc = (eval[k] <= tau_lin * emax) ? Real(0) : Real(1) / eval[k];
    for (int P = 0; P < naux; ++P) Vs[k * naux + P] *= sc;
  }
  std::vector<Real> Minv(static_cast<std::size_t>(naux) * naux, Real(0));
  detail::gemm('T', 'N', naux, naux, naux, Real(1), M.data(), naux, Vs.data(), naux, Real(0),
               Minv.data(), naux);

  std::vector<Real> Tp(N); // one gathered (mu nu|P) slab
  for (int k0 = 0; k0 < nvec; k0 += vec_tile) {
    const int k1 = std::min(k0 + vec_tile, nvec);
    const int kb = k1 - k0;
    const std::size_t slab = static_cast<std::size_t>(nao) * kb;
    // X, Y are naux x (nao x kb), P-major so the metric solve is one GEMM
    std::vector<Real> X(static_cast<std::size_t>(naux) * slab, Real(0));
    std::vector<Real> Y(static_cast<std::size_t>(naux) * slab, Real(0));
    // slices of the coefficient blocks for this vector tile
    std::vector<Real> CLb(static_cast<std::size_t>(nao) * kb),
        CRb(static_cast<std::size_t>(nao) * kb);
    for (int i = 0; i < nao; ++i)
      for (int k = 0; k < kb; ++k) {
        CLb[i * kb + k] = CL[i * nvec + k0 + k];
        CRb[i * kb + k] = CR[i * nvec + k0 + k];
      }
    for (int A0 = 0; A0 < nsa; A0 += aux_tile_shells) {
      const int A1 = std::min(A0 + aux_tile_shells, nsa);
      const int p0 = aux.ao_off[A0], blk = aux.ao_off[A1] - p0;
      const auto Tblk = coulomb_3c_auxblock(orb, aux, grid, A0, A1); // N x blk
      for (int P = 0; P < blk; ++P) {
        // gather the strided (mu nu|P) slab, then two half transforms
        for (std::size_t mn = 0; mn < N; ++mn) Tp[mn] = Tblk[mn * blk + P];
        detail::gemm('N', 'N', nao, kb, nao, Real(1), Tp.data(), nao, CLb.data(), kb, Real(0),
                     X.data() + static_cast<std::size_t>(p0 + P) * slab, kb);
        detail::gemm('N', 'N', nao, kb, nao, Real(1), Tp.data(), nao, CRb.data(), kb, Real(0),
                     Y.data() + static_cast<std::size_t>(p0 + P) * slab, kb);
      }
    }
    // Xhat = M^{-1} X over the auxiliary index: one GEMM for the whole tile
    std::vector<Real> Xh(X.size(), Real(0));
    detail::gemm('N', 'N', naux, static_cast<int>(slab), naux, Real(1), Minv.data(), naux,
                 X.data(), static_cast<int>(slab), Real(0), Xh.data(),
                 static_cast<int>(slab));
    // K += sum_P Xhat_P (nao x kb) . Y_P^T (kb x nao)
    for (int P = 0; P < naux; ++P)
      detail::gemm('N', 'T', nao, nao, kb, Real(1), Xh.data() + static_cast<std::size_t>(P) * slab,
                   kb, Y.data() + static_cast<std::size_t>(P) * slab, kb, Real(1), K, nao);
  }
}

} // namespace intti
