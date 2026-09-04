// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Precision-generic dense linear algebra. The RI / derivative builders contract
// large tensors that are really GEMMs; dispatch them here so they get BLAS
// throughput at float/double while staying correct at extended precision
// (long double / __float128 / MPFR, which have no BLAS) via a triple-loop
// fallback. Row-major throughout to match the library's tensor storage.

#include <cstddef>
#include <type_traits>

namespace intti {
namespace detail {

extern "C" {
void dgemm_(const char *, const char *, const int *, const int *, const int *,
            const double *, const double *, const int *, const double *,
            const int *, const double *, double *, const int *);
void sgemm_(const char *, const char *, const int *, const int *, const int *,
            const float *, const float *, const int *, const float *,
            const int *, const float *, float *, const int *);
}

/// Row-major GEMM: C(m x n) = alpha * op(A) * op(B) + beta * C, where op(X) is
/// X (trans='N') or X^T (trans='T'). op(A) is m x k, op(B) is k x n. lda/ldb/ldc
/// are the row strides (>= the number of columns of the STORED A/B/C). BLAS for
/// float/double (via the row-major<->col-major swap), triple loop otherwise.
template <class Real>
void gemm(char transA, char transB, int m, int n, int k, Real alpha,
          const Real *A, int lda, const Real *B, int ldb, Real beta, Real *C,
          int ldc) {
  if constexpr (std::is_same_v<Real, double> || std::is_same_v<Real, float>) {
    // row-major C = op(A) op(B)  <=>  col-major C = op(B) op(A) with n,m swapped
    if constexpr (std::is_same_v<Real, double>)
      dgemm_(&transB, &transA, &n, &m, &k, &alpha, B, &ldb, A, &lda, &beta, C, &ldc);
    else
      sgemm_(&transB, &transA, &n, &m, &k, &alpha, B, &ldb, A, &lda, &beta, C, &ldc);
  } else {
    const bool ta = transA == 'T' || transA == 't';
    const bool tb = transB == 'T' || transB == 't';
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
        Real s = 0;
        for (int l = 0; l < k; ++l) {
          const Real a = ta ? A[static_cast<std::size_t>(l) * lda + i]
                            : A[static_cast<std::size_t>(i) * lda + l];
          const Real b = tb ? B[static_cast<std::size_t>(j) * ldb + l]
                            : B[static_cast<std::size_t>(l) * ldb + j];
          s += a * b;
        }
        Real &c = C[static_cast<std::size_t>(i) * ldc + j];
        c = alpha * s + beta * c;
      }
  }
}

} // namespace detail
} // namespace intti
