// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "batch.hpp"

extern "C" {
void dsyevd_(const char *jobz, const char *uplo, const int *n, double *a,
             const int *lda, double *w, double *work, const int *lwork,
             int *iwork, const int *liwork, int *info);
void ssyevd_(const char *jobz, const char *uplo, const int *n, float *a,
             const int *lda, float *w, float *work, const int *lwork,
             int *iwork, const int *liwork, int *info);
void dgemm_(const char *, const char *, const int *, const int *, const int *,
            const double *, const double *, const int *, const double *,
            const int *, const double *, double *, const int *);
void sgemm_(const char *, const char *, const int *, const int *, const int *,
            const float *, const float *, const int *, const float *,
            const int *, const float *, float *, const int *);
}

namespace intti {

/// Cholesky decomposition of the ERI matrix: (ab|cd) ~= sum_J L_ab^J L_cd^J.
/// Pivots are Cartesian components of shell-pair products.
template <class Real> struct CholeskyBasis {
  std::vector<std::pair<int, int>> pivots; ///< (pair index, component in pair)
  std::vector<int> prod_offset;            ///< pair -> first global product index
  int nprod{0}, naux{0};
  /// Cholesky vectors, column-major (nprod x naux), host
  Kokkos::View<Real **, Kokkos::LayoutLeft, Kokkos::HostSpace> L;
  /// shell indices (i, j) of each pair; set by the ShellBasis-level overload
  /// (cdjk.hpp) and required by cholesky_jk
  std::vector<std::pair<int, int>> pair_shells;
};

template <class Real> struct CholeskyOptions {
  Real tau{1e-8};      ///< decomposition threshold (diagonal residual)
  Real tau_lin{1e-12}; ///< eigenvalue cutoff for S^{-1/2} in step 2
  bool one_step{false}; ///< return the step-1 vectors (reference/debugging)
};

namespace detail {

template <class Real>
void syevd(int n, Real *a, Real *w) {
  int info = 0, lwork = -1, liwork = -1;
  Real wq;
  int iq;
  if constexpr (std::is_same_v<Real, double>)
    dsyevd_("V", "L", &n, a, &n, w, &wq, &lwork, &iq, &liwork, &info);
  else
    ssyevd_("V", "L", &n, a, &n, w, &wq, &lwork, &iq, &liwork, &info);
  lwork = static_cast<int>(wq);
  liwork = iq;
  std::vector<Real> work(lwork);
  std::vector<int> iwork(liwork);
  if constexpr (std::is_same_v<Real, double>)
    dsyevd_("V", "L", &n, a, &n, w, work.data(), &lwork, iwork.data(), &liwork, &info);
  else
    ssyevd_("V", "L", &n, a, &n, w, work.data(), &lwork, iwork.data(), &liwork, &info);
  if (info != 0) throw std::runtime_error("intti: syevd failed");
}

template <class Real>
void gemm_nn(int m, int n, int k, const Real *A, int lda, const Real *B, int ldb,
             Real *C, int ldc) {
  const Real one = 1, zero = 0;
  if constexpr (std::is_same_v<Real, double>)
    dgemm_("N", "N", &m, &n, &k, &one, A, &lda, B, &ldb, &zero, C, &ldc);
  else
    sgemm_("N", "N", &m, &n, &k, &one, A, &lda, B, &ldb, &zero, C, &ldc);
}

/// Run a quartet batch and return the host copy of the output.
template <class Real>
std::vector<Real> run_batch(const PairTable<Real> &pairs,
                            const std::vector<std::pair<int, int>> &quartets,
                            const TGrid<Real> &grid, QuartetWorkspace<Real> &ws) {
  auto batch = make_batch(pairs, quartets);
  Kokkos::View<Real *> out("intti::chol::out", batch.nout_total);
  eri_quartets(pairs, batch, grid, out, ws);
  auto oh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  std::vector<Real> res(batch.nout_total);
  for (int i = 0; i < batch.nout_total; ++i)
    res[i] = oh(i);
  return res;
}

} // namespace detail

/// Two-step pivoted Cholesky decomposition (Folkestad-Koch style).
/// Step 1 selects the pivot products by pivoted Cholesky of the diagonal;
/// step 2 rebuilds the vectors RI-style, L = (ab|J) S^{-1/2}, which is one
/// batched integral pass plus dense linear algebra. Requires float/double
/// (host LAPACK); arbitrary-precision Cholesky is deferred.
template <class Real>
CholeskyBasis<Real> two_step_cholesky(const PairTable<Real> &pairs,
                                      const TGrid<Real> &grid,
                                      QuartetWorkspace<Real> &ws,
                                      const CholeskyOptions<Real> &opt = {}) {
  // Step 1 (the pivoted Cholesky, the one_step result) is LAPACK-free and works
  // for any batch-integral scalar; only step 2's dense S^{-1/2} needs host
  // LAPACK, so long double is supported via one_step (see below).
  static_assert(kokkos_scalar_v<Real>,
                "two_step_cholesky needs a batched-integral scalar (float/double/long double)");
  CholeskyBasis<Real> basis;
  const int npair = pairs.npair;
  basis.prod_offset.resize(npair + 1);
  basis.prod_offset[0] = 0;
  auto pair_nprod = [&](int ip) {
    return ncart(pairs.h_la[ip]) * ncart(pairs.h_lb[ip]);
  };
  for (int ip = 0; ip < npair; ++ip)
    basis.prod_offset[ip + 1] = basis.prod_offset[ip] + pair_nprod(ip);
  const int nprod = basis.prod_offset[npair];
  basis.nprod = nprod;

  // diagonal D_ab = (ab|ab)
  std::vector<Real> D(nprod);
  {
    std::vector<std::pair<int, int>> diag_q(npair);
    for (int ip = 0; ip < npair; ++ip)
      diag_q[ip] = {ip, ip};
    auto vals = detail::run_batch(pairs, diag_q, grid, ws);
    int off = 0;
    for (int ip = 0; ip < npair; ++ip) {
      const int nc = pair_nprod(ip);
      for (int c = 0; c < nc; ++c)
        D[basis.prod_offset[ip] + c] = vals[off + c * nc + c];
      off += nc * nc;
    }
  }

  // step 1: pivoted Cholesky driven by the updated diagonal; one batched
  // column sweep per selected pivot pair (all its components at once)
  std::vector<std::vector<Real>> Lcols; // step-1 vectors, each length nprod
  std::vector<std::pair<int, int>> pivots;
  std::vector<std::pair<int, int>> col_q(npair);
  while (true) {
    int jglob = -1;
    Real dmax = opt.tau;
    for (int i = 0; i < nprod; ++i)
      if (D[i] > dmax) {
        dmax = D[i];
        jglob = i;
      }
    if (jglob < 0) break;
    int jp = 0;
    while (basis.prod_offset[jp + 1] <= jglob)
      ++jp;
    // columns (ab|jp,*) for every pair ab
    for (int ip = 0; ip < npair; ++ip)
      col_q[ip] = {ip, jp};
    auto vals = detail::run_batch(pairs, col_q, grid, ws);
    const int ncj = pair_nprod(jp);
    // process this pivot pair's components in decreasing-diagonal order
    std::vector<int> order(ncj);
    for (int c = 0; c < ncj; ++c)
      order[c] = c;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return D[basis.prod_offset[jp] + a] > D[basis.prod_offset[jp] + b];
    });
    for (int jc : order) {
      const int j = basis.prod_offset[jp] + jc;
      if (D[j] <= opt.tau) continue;
      // assemble the residual column
      std::vector<Real> col(nprod);
      {
        int off = 0;
        for (int ip = 0; ip < npair; ++ip) {
          const int nci = pair_nprod(ip);
          for (int c = 0; c < nci; ++c)
            col[basis.prod_offset[ip] + c] = vals[off + c * ncj + jc];
          off += nci * ncj;
        }
      }
      for (const auto &Lk : Lcols) {
        const Real ljk = Lk[j];
        for (int i = 0; i < nprod; ++i)
          col[i] -= Lk[i] * ljk;
      }
      const Real diag = col[j];
      if (diag <= opt.tau) {
        D[j] = 0;
        continue;
      }
      const Real inv = 1 / std::sqrt(diag);
      for (int i = 0; i < nprod; ++i)
        col[i] *= inv;
      for (int i = 0; i < nprod; ++i) {
        D[i] -= col[i] * col[i];
        if (D[i] < Real(0)) D[i] = 0;
      }
      pivots.push_back({jp, jc});
      Lcols.push_back(std::move(col));
    }
  }
  basis.pivots = pivots;
  basis.naux = static_cast<int>(pivots.size());
  basis.L = Kokkos::View<Real **, Kokkos::LayoutLeft, Kokkos::HostSpace>(
      "intti::chol::L", nprod, basis.naux);

  if (opt.one_step) {
    for (int J = 0; J < basis.naux; ++J)
      for (int i = 0; i < nprod; ++i)
        basis.L(i, J) = Lcols[J][i];
    return basis;
  }

  // step 2: RI-style vector construction over the fixed pivot set. Needs a
  // dense S^{-1/2} (host LAPACK eigensolver), so it is float/double only; long
  // double uses the LAPACK-free one_step path above.
  if constexpr (std::is_same_v<Real, double> || std::is_same_v<Real, float>) {
  const int naux = basis.naux;
  if (naux == 0) return basis;
  // unique pivot pairs
  std::vector<int> ppairs;
  for (const auto &[jp, jc] : pivots)
    if (ppairs.empty() || ppairs.back() != jp) ppairs.push_back(jp);
  // M_ab,J = (ab|J): batches (ip, jp) over unique pivot pairs
  std::vector<Real> M(static_cast<std::size_t>(nprod) * naux);
  {
    std::vector<std::pair<int, int>> q;
    q.reserve(npair * ppairs.size());
    for (int jp : ppairs)
      for (int ip = 0; ip < npair; ++ip)
        q.push_back({ip, jp});
    auto vals = detail::run_batch(pairs, q, grid, ws);
    // scatter the pivot components' columns
    std::size_t off = 0;
    std::size_t qi = 0;
    for (int jp : ppairs) {
      const int ncj = pair_nprod(jp);
      for (int ip = 0; ip < npair; ++ip, ++qi) {
        const int nci = pair_nprod(ip);
        for (int J = 0; J < naux; ++J)
          if (pivots[J].first == jp)
            for (int c = 0; c < nci; ++c)
              M[basis.prod_offset[ip] + c + static_cast<std::size_t>(nprod) * J] =
                  vals[off + c * ncj + pivots[J].second];
        off += static_cast<std::size_t>(nci) * ncj;
      }
    }
  }
  // S_JK = (J|K) extracted from M at the pivot rows
  std::vector<Real> S(static_cast<std::size_t>(naux) * naux);
  for (int J = 0; J < naux; ++J)
    for (int K = 0; K < naux; ++K)
      S[J + static_cast<std::size_t>(naux) * K] =
          M[basis.prod_offset[pivots[J].first] + pivots[J].second +
            static_cast<std::size_t>(nprod) * K];
  // S^{-1/2} with eigenvalue cutoff
  std::vector<Real> eval(naux);
  detail::syevd(naux, S.data(), eval.data()); // S now holds eigenvectors V
  std::vector<Real> Shalf(static_cast<std::size_t>(naux) * naux, Real(0));
  {
    std::vector<Real> Vs(static_cast<std::size_t>(naux) * naux);
    for (int k = 0; k < naux; ++k) {
      const Real s = eval[k] > opt.tau_lin ? 1 / std::sqrt(eval[k]) : Real(0);
      for (int J = 0; J < naux; ++J)
        Vs[J + static_cast<std::size_t>(naux) * k] =
            S[J + static_cast<std::size_t>(naux) * k] * s;
    }
    // Shalf = Vs * V^T
    const Real one = 1, zero = 0;
    const int n = naux;
    if constexpr (std::is_same_v<Real, double>)
      dgemm_("N", "T", &n, &n, &n, &one, Vs.data(), &n, S.data(), &n, &zero,
             Shalf.data(), &n);
    else
      sgemm_("N", "T", &n, &n, &n, &one, Vs.data(), &n, S.data(), &n, &zero,
             Shalf.data(), &n);
  }
  // L = M * S^{-1/2}
  detail::gemm_nn(nprod, naux, naux, M.data(), nprod, Shalf.data(), naux,
                  basis.L.data(), nprod);
  return basis;
  } else {
    throw std::runtime_error("two_step_cholesky: the two-step S^{-1/2} path needs "
                             "float/double LAPACK; pass one_step=true for long double");
  }
}

/// Precision-generic pivoted Cholesky of the ERI over a shell-pair set: the
/// LAPACK-free one-step decomposition (ab|cd) ~= sum_J L_ab^J L_cd^J. Works for
/// float/double/long double (the batched-integral scalars); no host eigensolver.
template <class Real>
CholeskyBasis<Real> pivoted_cholesky(const PairTable<Real> &pairs, const TGrid<Real> &grid,
                                     QuartetWorkspace<Real> &ws, CholeskyOptions<Real> opt = {}) {
  opt.one_step = true;
  return two_step_cholesky(pairs, grid, ws, opt);
}

namespace detail {
/// Serial pivoted Cholesky driven by the single-quartet eri_quartet() host
/// path -- no batch, no LAPACK -- so it runs at ANY real scalar, including
/// __float128 / MPFR wrappers that the Kokkos batch path (PairTable) rejects.
/// Same pivoted recurrence as two_step_cholesky's step 1.
template <class Real>
CholeskyBasis<Real> pivoted_cholesky_serial(const std::vector<ShellPair<Real>> &pairs,
                                            const TGrid<real_t<Real>> &grid,
                                            const CholeskyOptions<Real> &opt) {
  CholeskyBasis<Real> basis;
  const int npair = static_cast<int>(pairs.size());
  auto pn = [&](int ip) { return ncart(pairs[ip].la) * ncart(pairs[ip].lb); };
  basis.prod_offset.assign(npair + 1, 0);
  for (int ip = 0; ip < npair; ++ip) basis.prod_offset[ip + 1] = basis.prod_offset[ip] + pn(ip);
  const int nprod = basis.prod_offset[npair];
  basis.nprod = nprod;
  // diagonal D_ab = (ab|ab)
  std::vector<Real> D(nprod);
  for (int ip = 0; ip < npair; ++ip) {
    const int nc = pn(ip);
    std::vector<Real> blk(static_cast<std::size_t>(nc) * nc);
    eri_quartet(pairs[ip], pairs[ip], grid, blk.data());
    for (int c = 0; c < nc; ++c) D[basis.prod_offset[ip] + c] = blk[c * nc + c];
  }
  std::vector<std::vector<Real>> Lcols;
  std::vector<std::pair<int, int>> pivots;
  while (true) {
    int jglob = -1;
    Real dmax = opt.tau;
    for (int i = 0; i < nprod; ++i)
      if (D[i] > dmax) { dmax = D[i]; jglob = i; }
    if (jglob < 0) break;
    int jp = 0;
    while (basis.prod_offset[jp + 1] <= jglob) ++jp;
    const int ncj = pn(jp);
    // (ab | jp,*) blocks for every pair ab, once for this pivot pair
    std::vector<std::vector<Real>> colblk(npair);
    for (int ip = 0; ip < npair; ++ip) {
      colblk[ip].resize(static_cast<std::size_t>(pn(ip)) * ncj);
      eri_quartet(pairs[ip], pairs[jp], grid, colblk[ip].data());
    }
    std::vector<int> order(ncj);
    for (int c = 0; c < ncj; ++c) order[c] = c;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return D[basis.prod_offset[jp] + a] > D[basis.prod_offset[jp] + b];
    });
    for (int jc : order) {
      const int j = basis.prod_offset[jp] + jc;
      if (D[j] <= opt.tau) continue;
      std::vector<Real> col(nprod);
      for (int ip = 0; ip < npair; ++ip) {
        const int nci = pn(ip);
        for (int c = 0; c < nci; ++c) col[basis.prod_offset[ip] + c] = colblk[ip][c * ncj + jc];
      }
      for (const auto &Lk : Lcols) {
        const Real ljk = Lk[j];
        for (int i = 0; i < nprod; ++i) col[i] -= Lk[i] * ljk;
      }
      const Real diag = col[j];
      if (diag <= opt.tau) { D[j] = 0; continue; }
      const Real inv = Real(1) / sqrt_(diag);
      for (int i = 0; i < nprod; ++i) col[i] *= inv;
      for (int i = 0; i < nprod; ++i) {
        D[i] -= col[i] * col[i];
        if (D[i] < Real(0)) D[i] = 0;
      }
      pivots.push_back({jp, jc});
      Lcols.push_back(std::move(col));
    }
  }
  basis.pivots = pivots;
  basis.naux = static_cast<int>(pivots.size());
  basis.L = Kokkos::View<Real **, Kokkos::LayoutLeft, Kokkos::HostSpace>("intti::chol::Lserial",
                                                                         nprod, basis.naux);
  for (int J = 0; J < basis.naux; ++J)
    for (int i = 0; i < nprod; ++i) basis.L(i, J) = Lcols[J][i];
  return basis;
}
} // namespace detail

/// Precision-generic pivoted Cholesky over a shell-pair list: dispatches to the
/// batched path for kokkos scalars (float/double/long double) and to the serial
/// eri_quartet path otherwise (__float128 / MPFR). Single entry, all precisions.
template <class Real>
CholeskyBasis<Real> pivoted_cholesky(const std::vector<ShellPair<Real>> &pairs,
                                     const TGrid<real_t<Real>> &grid,
                                     const CholeskyOptions<Real> &opt = {}) {
  if constexpr (kokkos_scalar_v<Real>) {
    auto tab = make_pair_table(pairs);
    QuartetWorkspace<Real> ws;
    return pivoted_cholesky(tab, grid, ws, opt);
  } else {
    return detail::pivoted_cholesky_serial(pairs, grid, opt);
  }
}

} // namespace intti
