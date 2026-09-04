// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
//
// Precision-generic row-major GEMM (blas.hpp): the BLAS (double) path and the
// extended-precision triple-loop path must both match a naive reference for all
// transpose combinations and non-square shapes.

#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "intti/blas.hpp"

namespace {

template <class R>
void ref_gemm(char tA, char tB, int m, int n, int k, R al, const R *A, int lda,
              const R *B, int ldb, R be, R *C, int ldc) {
  const bool ta = tA == 'T', tb = tB == 'T';
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      R s = 0;
      for (int l = 0; l < k; ++l) {
        const R a = ta ? A[l * lda + i] : A[i * lda + l];
        const R b = tb ? B[j * ldb + l] : B[l * ldb + j];
        s += a * b;
      }
      C[i * ldc + j] = al * s + be * C[i * ldc + j];
    }
}

template <class R> void run() {
  const int m = 3, n = 4, k = 5;
  std::mt19937 rng(1);
  auto fill = [&](std::vector<R> &v) {
    for (auto &x : v) x = R((rng() % 1000) / 500.0 - 1.0);
  };
  for (char tA : {'N', 'T'})
    for (char tB : {'N', 'T'}) {
      const int lda = tA == 'N' ? k : m, ldb = tB == 'N' ? n : k, ldc = n;
      std::vector<R> A((tA == 'N' ? m : k) * lda), B((tB == 'N' ? k : n) * ldb);
      std::vector<R> C0(m * ldc), C1;
      fill(A);
      fill(B);
      fill(C0);
      C1 = C0;
      intti::detail::gemm<R>(tA, tB, m, n, k, R(2), A.data(), lda, B.data(), ldb,
                             R(0.5), C1.data(), ldc);
      ref_gemm<R>(tA, tB, m, n, k, R(2), A.data(), lda, B.data(), ldb, R(0.5),
                  C0.data(), ldc);
      for (int i = 0; i < m * n; ++i)
        EXPECT_NEAR(double(C0[i]), double(C1[i]), 1e-12)
            << "tA=" << tA << " tB=" << tB << " i=" << i;
    }
}

TEST(Blas, GemmDouble) { run<double>(); }
TEST(Blas, GemmLongDouble) { run<long double>(); }

} // namespace
