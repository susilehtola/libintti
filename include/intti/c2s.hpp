// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Cartesian -> real solid harmonic transform. The harmonics are built
// exactly from the Racah recursion
//   (l-m) C_l^m = (2l-1) z C_{l-1}^m - (l+m-1) r^2 C_{l-2}^m,
// seeded by C_m^m = (x+iy)^m, as homogeneous polynomials in the Cartesian
// monomial basis, then row-normalized to unit L2 norm on the unit sphere
// (so the transform rows are orthonormal and Wigner rotations are
// orthogonal; any external convention, e.g. libcint's, is a per-row
// rescaling applied at the facade).

#include <cmath>
#include <vector>

#include "gto.hpp"
#include "math.hpp"

namespace intti {

/// index of the Cartesian monomial (lx, ly, lz) in the cart_comp ordering
inline int cart_index(int l, int lx, int ly) {
  return (l - lx) * (l - lx + 1) / 2 + (l - lx - ly);
}

/// (2l+1) x ncart(l) row-major transform matrix; rows ordered m = -l..+l.
template <class Real = double> std::vector<Real> c2s_matrix(int l) {
  using std::sqrt;
  const int nc = ncart(l);
  std::vector<Real> out(static_cast<std::size_t>(2 * l + 1) * nc, Real(0));
  // sphere integral of x^a y^b z^c over the unit sphere / (4 pi)
  auto sphere = [](int a, int b, int c) -> double {
    if (a % 2 || b % 2 || c % 2) return 0.0;
    auto dfact = [](int n) {
      double r = 1;
      for (int k = n; k > 1; k -= 2)
        r *= k;
      return r;
    };
    return dfact(a - 1) * dfact(b - 1) * dfact(c - 1) / dfact(a + b + c + 1);
  };
  for (int m = 0; m <= l; ++m) {
    // seed C_m^m = (x + i y)^m at degree m, then recur up to degree l;
    // polynomials stored as dense coefficient vectors over the monomials
    std::vector<double> re(ncart(m), 0.0), im(ncart(m), 0.0);
    {
      double binom = 1;
      for (int j = 0; j <= m; ++j) {
        // x^{m-j} y^j, coefficient binom * i^j
        const int idx = cart_index(m, m - j, j);
        if (j % 4 == 0) re[idx] += binom;
        else if (j % 4 == 1) im[idx] += binom;
        else if (j % 4 == 2) re[idx] -= binom;
        else im[idx] -= binom;
        binom = binom * (m - j) / (j + 1);
      }
    }
    std::vector<double> re1, im1; // C_{k-1}^m while building degree k
    for (int k = m + 1; k <= l; ++k) {
      std::vector<double> ren(ncart(k), 0.0), imn(ncart(k), 0.0);
      // (k - m) C_k = (2k-1) z C_{k-1} - (k+m-1) r^2 C_{k-2}
      auto add_z = [&](const std::vector<double> &src, std::vector<double> &dst,
                       double fac, int kk) {
        // src has degree kk-1; multiply by z
        for (int lx = kk - 1; lx >= 0; --lx)
          for (int ly = kk - 1 - lx; ly >= 0; --ly)
            dst[cart_index(kk, lx, ly)] += fac * src[cart_index(kk - 1, lx, ly)];
      };
      auto add_r2 = [&](const std::vector<double> &src, std::vector<double> &dst,
                        double fac, int kk) {
        // src has degree kk-2; multiply by x^2 + y^2 + z^2
        for (int lx = kk - 2; lx >= 0; --lx)
          for (int ly = kk - 2 - lx; ly >= 0; --ly) {
            const double v = fac * src[cart_index(kk - 2, lx, ly)];
            dst[cart_index(kk, lx + 2, ly)] += v;
            dst[cart_index(kk, lx, ly + 2)] += v;
            dst[cart_index(kk, lx, ly)] += v;
          }
      };
      const double f1 = double(2 * k - 1) / (k - m);
      add_z(re, ren, f1, k);
      add_z(im, imn, f1, k);
      if (k - 2 >= m) {
        const double f2 = -double(k + m - 1) / (k - m);
        add_r2(re1, ren, f2, k);
        add_r2(im1, imn, f2, k);
      }
      re1 = std::move(re);
      im1 = std::move(im);
      re = std::move(ren);
      im = std::move(imn);
    }
    // real forms: m = 0 -> Re; +m -> Re, -m -> Im; normalize on the sphere
    auto emit = [&](const std::vector<double> &poly, int mrow) {
      double n2 = 0;
      for (int lx = l; lx >= 0; --lx)
        for (int ly = l - lx; ly >= 0; --ly)
          for (int lx2 = l; lx2 >= 0; --lx2)
            for (int ly2 = l - lx2; ly2 >= 0; --ly2)
              n2 += poly[cart_index(l, lx, ly)] * poly[cart_index(l, lx2, ly2)] *
                    sphere(lx + lx2, ly + ly2, (l - lx - ly) + (l - lx2 - ly2));
      const double inv = 1.0 / std::sqrt(n2);
      for (int k = 0; k < nc; ++k)
        out[static_cast<std::size_t>(mrow + l) * nc + k] = Real(poly[k] * inv);
    };
    emit(re, m);
    if (m > 0) emit(im, -m);
  }
  return out;
}

/// transform the leading index of a block: out[(2l+1) x ncols] =
/// C(l) * in[ncart(l) x ncols]
template <class Real>
void apply_c2s(int l, int ncols, const Real *in, Real *out) {
  auto C = c2s_matrix<Real>(l);
  const int nc = ncart(l), nm = 2 * l + 1;
  for (int m = 0; m < nm; ++m)
    for (int j = 0; j < ncols; ++j) {
      Real s = 0;
      for (int k = 0; k < nc; ++k)
        s += C[static_cast<std::size_t>(m) * nc + k] * in[k * ncols + j];
      out[m * ncols + j] = s;
    }
}

} // namespace intti
