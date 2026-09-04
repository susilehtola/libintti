// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
//
// M-HK: the Yukawa / bound-state Helmholtz attraction primitive. The screened
// kernel e^{-kappa r}/r drives the same t-quadrature machinery as Coulomb (only
// the grid weights change), so nuclear_matrix / yukawa_attraction_matrix with a
// Yukawa grid give the Yukawa attraction. Validated against an INDEPENDENT
// real-space oracle: the screened-Poisson potential of a spherical Gaussian,
//   Phi_p(R) = (2 pi/(kappa R)) int_0^inf e^{-p rho^2} rho
//              [e^{-kappa|R-rho|} - e^{-kappa(R+rho)}] d rho,
// evaluated by Gauss-Legendre (split at the |R-rho| kink) -- a completely
// different computational path from the t-quadrature.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/cholesky.hpp" // detail::jacobi_eigh (LAPACK-free generalized solve)
#include "intti/helmholtz.hpp"
#include "intti/oneel.hpp"

namespace {

// n-point Gauss-Legendre nodes/weights on [-1,1] (Newton on Legendre roots).
void gauss_legendre(int n, std::vector<double> &x, std::vector<double> &w) {
  x.resize(n);
  w.resize(n);
  for (int i = 0; i < n; ++i) {
    double z = std::cos(M_PI * (i + 0.75) / (n + 0.5)), z1;
    int it = 0;
    double pp = 0;
    do {
      double p0 = 1, p1 = 0;
      for (int j = 0; j < n; ++j) {
        double p2 = p1;
        p1 = p0;
        p0 = ((2 * j + 1) * z * p1 - j * p2) / (j + 1);
      }
      pp = n * (z * p0 - p1) / (z * z - 1);
      z1 = z;
      z = z1 - p0 / pp;
    } while (std::fabs(z - z1) > 1e-15 && ++it < 100);
    x[i] = z;
    w[i] = 2 / ((1 - z * z) * pp * pp);
  }
}

double sub_integral(double p, double kappa, double R, double lo, double hi) {
  static std::vector<double> x, w;
  if (x.empty()) gauss_legendre(400, x, w);
  const double c = 0.5 * (hi - lo), m = 0.5 * (hi + lo);
  double s = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double rho = m + c * x[i];
    const double f = std::exp(-p * rho * rho) * rho *
                     (std::exp(-kappa * std::fabs(R - rho)) -
                      std::exp(-kappa * (R + rho)));
    s += w[i] * f;
  }
  return c * s;
}

// Yukawa potential of a spherical Gaussian e^{-p rho^2} at distance R.
double phi_oracle(double p, double kappa, double R) {
  const double I = sub_integral(p, kappa, R, 0, R) +
                   sub_integral(p, kappa, R, R, R + 20 / std::sqrt(p));
  return 2 * M_PI / (kappa * R) * I;
}

TEST(Helmholtz, YukawaAttractionVsRadialOracle) {
  for (double kappa : {0.5, 1.3, 3.0}) {
    for (double a : {0.6, 2.0}) {
      // two s-shells at the origin (product exponent p = a + 0.7a, K = 1),
      // unit charge at distance R along z
      auto basis = intti::make_basis<double>({{a, {0, 0, 0}, 0}, {0.7 * a, {0, 0, 0}, 0}});
      const double p = a + 0.7 * a;
      for (double R : {0.8, 1.5, 3.0}) {
        std::vector<intti::PointCharge<double>> ch{{1.0, {0, 0, R}}};
        auto V = intti::yukawa_attraction_matrix(basis, ch, kappa);
        const double got = V[0 * 2 + 1];
        const double ref = phi_oracle(p, kappa, R);
        EXPECT_NEAR(got, ref, 1e-5 * std::abs(ref))
            << "kappa=" << kappa << " a=" << a << " R=" << R;
      }
    }
  }
}

// nuclear_matrix accepts a Yukawa grid directly (the primitive is not special):
// a well-ranged Yukawa grid reproduces the radial oracle through nuclear_matrix.
TEST(Helmholtz, NuclearMatrixWithYukawaGrid) {
  const double kappa = 1.0, a = 1.1, p = a + 0.9;
  auto basis = intti::make_basis<double>({{a, {0, 0, 0}, 0}, {0.9, {0, 0, 0}, 0}});
  auto grid = intti::yukawa_grid<double>(kappa, 1e-2, 1e3);
  std::vector<intti::PointCharge<double>> ch{{1.0, {0, 0, 1.2}}};
  auto V = intti::nuclear_matrix(basis, ch, grid);
  EXPECT_NEAR(V[1], phi_oracle(p, kappa, 1.2), 1e-5 * std::abs(V[1]));
}

// M-HK step 2: the two-kernel M(kappa)=<mu|G_kappa V|nu> must satisfy the
// Green's-function fixed point at the H_core ground state: M(kappa0) c0 =
// -1/2 S c0, kappa0=sqrt(-2 eps0). The residual is basis-limited (the integral
// form uses the exact kinetic, Galerkin projects it), so it shrinks toward 0
// with basis completeness; a 12-function even-tempered s-basis reaches < 1e-4.
TEST(Helmholtz, TwoKernelMkappaFixedPoint) {
  std::vector<intti::PrimitiveShell<double>> sh;
  for (int i = 0; i < 12; ++i) sh.push_back({0.04 * std::pow(2.6, i), {0, 0, 0}, 0});
  auto basis = intti::make_basis(sh);
  const int n = basis.nao;
  auto S = intti::overlap_matrix(basis);
  auto T = intti::kinetic_matrix(basis);
  std::vector<intti::PointCharge<double>> ch{{-1.0, {0, 0, 0}}}; // -Z, Z=1
  auto V = intti::nuclear_matrix(basis, ch, intti::make_tgrid(intti::coulomb()));
  std::vector<double> H(static_cast<std::size_t>(n) * n);
  for (int i = 0; i < n * n; ++i) H[i] = T[i] + V[i];

  // Lowdin S^{-1/2} = U diag(1/sqrt(s)) U^T (jacobi_eigh: evec column-major)
  std::vector<double> sw, U;
  intti::detail::jacobi_eigh(n, S, sw, U);
  auto ev = [&](int i, int k) { return U[static_cast<std::size_t>(k) * n + i]; };
  std::vector<double> Sinv(static_cast<std::size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double s = 0;
      for (int k = 0; k < n; ++k) s += ev(i, k) * (1.0 / std::sqrt(sw[k])) * ev(j, k);
      Sinv[i * n + j] = s;
    }
  auto mul = [&](const std::vector<double> &A, const std::vector<double> &B) {
    std::vector<double> C(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        double s = 0;
        for (int k = 0; k < n; ++k) s += A[i * n + k] * B[k * n + j];
        C[i * n + j] = s;
      }
    return C;
  };
  auto Hp = mul(mul(Sinv, H), Sinv);
  std::vector<double> hw, Uh;
  intti::detail::jacobi_eigh(n, Hp, hw, Uh);
  const double eps0 = hw[0]; // lowest
  ASSERT_LT(eps0, 0.0);
  EXPECT_NEAR(eps0, -0.5, 1e-3); // basis-limit H ground state
  std::vector<double> c0(n, 0.0); // c0 = Sinv v0, v0 = column 0 of Uh
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) c0[i] += Sinv[i * n + j] * Uh[0 * n + j];

  const double kappa0 = std::sqrt(-2 * eps0);
  auto M = intti::helmholtz_nuclear_matrix(basis, ch, kappa0);
  double num = 0, den = 0;
  for (int i = 0; i < n; ++i) {
    double Mc = 0, Sc = 0;
    for (int k = 0; k < n; ++k) {
      Mc += M[i * n + k] * c0[k];
      Sc += S[i * n + k] * c0[k];
    }
    num += (Mc + 0.5 * Sc) * (Mc + 0.5 * Sc);
    den += (0.5 * Sc) * (0.5 * Sc);
  }
  EXPECT_LT(std::sqrt(num / den), 1e-4); // fixed-point holds to the basis limit
}

// M-HK many-electron: the full effective potential M_eff = M_nuc + M_J - 1/2 M_K
// (J/K from helmholtz_jk_matrices, general-L three-kernel builds over threeel)
// must satisfy the RHF-like fixed point M_eff(kappa0) c0 = -1/2 S c0, with
// (eps0,c0) from F = T + V_nuc + J[D] - 1/2 K[D], D = 2 c c^T (fixed density).
TEST(Helmholtz, ManyElectronJKFixedPoint) {
  std::vector<intti::PrimitiveShell<double>> sh;
  for (int i = 0; i < 5; ++i) sh.push_back({0.14 * std::pow(2.5, i), {0, 0, 0}, 0});
  auto basis = intti::make_basis(sh);
  const int n = basis.nao;
  auto S = intti::overlap_matrix(basis);
  auto T = intti::kinetic_matrix(basis);
  auto cgrid = intti::make_tgrid(intti::coulomb());
  std::vector<intti::PointCharge<double>> ch{{-2.0, {0, 0, 0}}}; // He, Z=2
  auto Vn = intti::nuclear_matrix(basis, ch, cgrid);
  // ERI tensor (s-basis: one component per shell)
  std::vector<double> ERI((size_t)n * n * n * n);
  for (int a = 0; a < n; ++a)
    for (int b = 0; b < n; ++b)
      for (int d = 0; d < n; ++d)
        for (int e = 0; e < n; ++e) {
          double o;
          intti::eri_quartet(intti::make_pair(sh[a], sh[b]),
                             intti::make_pair(sh[d], sh[e]), cgrid, &o);
          ERI[((size_t)a * n + b) * n * n + ((size_t)d * n + e)] = o;
        }
  // Lowdin generalized solve  F c = eps S c  ->  lowest (eps0, c0)
  std::vector<double> sw, U;
  intti::detail::jacobi_eigh(n, S, sw, U);
  auto ev = [&](int i, int k) { return U[(size_t)k * n + i]; };
  std::vector<double> Si((size_t)n * n, 0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double s = 0;
      for (int k = 0; k < n; ++k) s += ev(i, k) * (1.0 / std::sqrt(sw[k])) * ev(j, k);
      Si[i * n + j] = s;
    }
  auto mul = [&](const std::vector<double> &A, const std::vector<double> &B) {
    std::vector<double> C((size_t)n * n, 0);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        double s = 0;
        for (int k = 0; k < n; ++k) s += A[i * n + k] * B[k * n + j];
        C[i * n + j] = s;
      }
    return C;
  };
  auto ground = [&](const std::vector<double> &F, double &eps, std::vector<double> &c) {
    auto Hp = mul(mul(Si, F), Si);
    std::vector<double> hw, Uh;
    intti::detail::jacobi_eigh(n, Hp, hw, Uh);
    eps = hw[0];
    c.assign(n, 0);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) c[i] += Si[i * n + j] * Uh[0 * n + j];
  };
  std::vector<double> H0((size_t)n * n), chc, c0;
  for (int i = 0; i < n * n; ++i) H0[i] = T[i] + Vn[i];
  double e;
  ground(H0, e, chc);
  std::vector<double> D((size_t)n * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) D[i * n + j] = 2 * chc[i] * chc[j]; // RHF density
  std::vector<double> Jm((size_t)n * n, 0), Km((size_t)n * n, 0);
  for (int mu = 0; mu < n; ++mu)
    for (int nu = 0; nu < n; ++nu) {
      double sj = 0, sk = 0;
      for (int r = 0; r < n; ++r)
        for (int sg = 0; sg < n; ++sg) {
          sj += D[r * n + sg] * ERI[((size_t)mu * n + nu) * n * n + ((size_t)r * n + sg)];
          sk += D[r * n + sg] * ERI[((size_t)mu * n + r) * n * n + ((size_t)nu * n + sg)];
        }
      Jm[mu * n + nu] = sj;
      Km[mu * n + nu] = sk;
    }
  std::vector<double> F((size_t)n * n);
  for (int i = 0; i < n * n; ++i) F[i] = T[i] + Vn[i] + Jm[i] - 0.5 * Km[i];
  double eps0;
  ground(F, eps0, c0);
  ASSERT_LT(eps0, 0.0);
  const double kappa0 = std::sqrt(-2 * eps0);
  auto Mn = intti::helmholtz_nuclear_matrix(basis, ch, kappa0);
  std::vector<double> MK;
  auto MJ = intti::helmholtz_jk_matrices(basis, D.data(), kappa0, MK);
  double num = 0, den = 0;
  for (int i = 0; i < n; ++i) {
    double Mc = 0, Sc = 0;
    for (int k = 0; k < n; ++k) {
      Mc += (Mn[i * n + k] + MJ[i * n + k] - 0.5 * MK[i * n + k]) * c0[k];
      Sc += S[i * n + k] * c0[k];
    }
    num += (Mc + 0.5 * Sc) * (Mc + 0.5 * Sc);
    den += (0.5 * Sc) * (0.5 * Sc);
  }
  EXPECT_LT(std::sqrt(num / den), 1e-3); // basis-limited fixed point
}

} // namespace
