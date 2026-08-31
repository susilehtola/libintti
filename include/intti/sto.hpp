// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Slater-type orbitals as quadrature-contracted Gaussians. A Slater radial
// exponential is the exact integral transform of a Gaussian,
//   e^{-zeta r} = (zeta / 2 sqrt(pi)) int_0^inf s^{-3/2} e^{-zeta^2/4s} e^{-s r^2} ds,
// so discretizing the s-integral turns e^{-zeta r} into a contracted s-GTO
//   e^{-zeta r} ~ sum_k c_k e^{-s_k r^2}.
// A minimal STO (principal quantum number n = l+1), r^l Y_lm e^{-zeta r}, is
// then a contracted CARTESIAN GTO shell of angular momentum l with those same
// exponents and coefficients (the r^l Y_lm is the ordinary solid-harmonic
// Cartesian polynomial the GTO shell already carries). The s-integral is the
// extra "radial" quadrature dimension; nesting it with the t-quadrature Coulomb
// kernel gives STO integrals as a plain contraction of Gaussian integrals --
// so every existing matrix builder works once the primitive shells are
// contracted to the STO basis.
//
// This header provides the s-expansion, the expansion of STO shells into a
// primitive ShellBasis with a contraction map, and the contraction of any
// primitive AO matrix down to the STO basis. Higher n (extra even radial
// powers r^{2k}) is a later extension; the minimal-STO case is exact here.

#include <cmath>
#include <cstddef>
#include <vector>

#include "fock.hpp"
#include "gto.hpp"
#include "math.hpp"

namespace intti {

/// A Slater-type-orbital shell: r^l Y_lm exp(-zeta r) on `center`, expanded
/// with `ns` Gaussian nodes over the radial transform variable s.
template <class Real> struct StoShell {
  Real zeta;
  Real center[3];
  int l;
  int ns{48};
};

/// s-expansion of exp(-zeta r) = sum_k c_k exp(-s_k r^2): log-space
/// Gauss-Legendre nodes s_k in [smin, smax] with the transform weight folded
/// into c_k. Returns {exponents, coefficients}.
template <class Real>
void sto_gaussians(Real zeta, int ns, std::vector<Real> &s, std::vector<Real> &c,
                   Real smin = Real(1e-4), Real smax = Real(1e6)) {
  s.assign(ns, Real(0));
  c.assign(ns, Real(0));
  // Gauss-Legendre nodes/weights on [-1,1] (Newton on Legendre P_ns)
  const Real pi = pi_v<Real>();
  const int m = (ns + 1) / 2;
  std::vector<Real> x(ns), w(ns);
  for (int i = 0; i < m; ++i) {
    Real z = std::cos(pi * (i + Real(0.75)) / (ns + Real(0.5)));
    Real z1, pp;
    do {
      Real p1 = 1, p2 = 0;
      for (int j = 0; j < ns; ++j) {
        const Real p3 = p2;
        p2 = p1;
        p1 = ((2 * j + 1) * z * p2 - j * p3) / (j + 1);
      }
      pp = ns * (z * p1 - p2) / (z * z - 1);
      z1 = z;
      z = z1 - p1 / pp;
    } while (std::abs(z - z1) > Real(1e-15));
    x[i] = -z;
    x[ns - 1 - i] = z;
    w[i] = 2 / ((1 - z * z) * pp * pp);
    w[ns - 1 - i] = w[i];
  }
  const Real lo = std::log(smin), hi = std::log(smax);
  for (int k = 0; k < ns; ++k) {
    const Real u = Real(0.5) * (hi - lo) * (x[k] + 1) + lo;
    const Real sk = std::exp(u);
    const Real ws = Real(0.5) * (hi - lo) * w[k] * sk; // ds = s du
    s[k] = sk;
    c[k] = zeta / (2 * std::sqrt(pi)) * ws * std::pow(sk, Real(-1.5)) *
           std::exp(-zeta * zeta / (4 * sk));
  }
}

/// Expansion of a set of STO shells into primitive Gaussian shells plus the
/// contraction map: sto_ao (shell i, Cartesian comp j) = sum_k c_{i,k} *
/// primitive (shell i, node k, comp j).
template <class Real> struct StoExpansion {
  ShellBasis<Real> prim;              ///< primitive Gaussian basis
  std::vector<Real> coeff;            ///< contraction matrix, nsto_ao x nprim_ao
  int nsto_ao{0}, nprim_ao{0};
};

template <class Real>
StoExpansion<Real> expand_sto(const std::vector<StoShell<Real>> &shells) {
  StoExpansion<Real> ex;
  std::vector<PrimitiveShell<Real>> prims;
  // count STO AOs and primitive AOs
  std::vector<int> sto_off(shells.size() + 1, 0), prim_off;
  for (std::size_t i = 0; i < shells.size(); ++i)
    sto_off[i + 1] = sto_off[i] + ncart(shells[i].l);
  ex.nsto_ao = sto_off.back();
  int poff = 0;
  std::vector<std::vector<Real>> ck(shells.size());
  std::vector<std::vector<int>> node_primoff(shells.size());
  for (std::size_t i = 0; i < shells.size(); ++i) {
    std::vector<Real> sk;
    sto_gaussians(shells[i].zeta, shells[i].ns, sk, ck[i]);
    for (int k = 0; k < shells[i].ns; ++k) {
      prims.push_back({sk[k],
                       {shells[i].center[0], shells[i].center[1], shells[i].center[2]},
                       shells[i].l});
      node_primoff[i].push_back(poff);
      poff += ncart(shells[i].l);
    }
  }
  ex.nprim_ao = poff;
  ex.prim = make_basis(prims);
  ex.coeff.assign(static_cast<std::size_t>(ex.nsto_ao) * ex.nprim_ao, Real(0));
  for (std::size_t i = 0; i < shells.size(); ++i) {
    const int nc = ncart(shells[i].l);
    for (int k = 0; k < shells[i].ns; ++k)
      for (int j = 0; j < nc; ++j)
        ex.coeff[static_cast<std::size_t>(sto_off[i] + j) * ex.nprim_ao +
                 node_primoff[i][k] + j] = ck[i][k];
  }
  return ex;
}

/// Contract a primitive AO matrix (nprim_ao x nprim_ao) to the STO basis:
/// M_sto = C M_prim C^T (nsto_ao x nsto_ao).
template <class Real>
std::vector<Real> contract_to_sto(const std::vector<Real> &Mprim, const StoExpansion<Real> &ex) {
  const int np = ex.nprim_ao, na = ex.nsto_ao;
  auto C = [&](int a, int p) { return ex.coeff[static_cast<std::size_t>(a) * np + p]; };
  std::vector<Real> tmp(static_cast<std::size_t>(na) * np, Real(0)); // C M
  for (int a = 0; a < na; ++a)
    for (int q = 0; q < np; ++q) {
      Real s = 0;
      for (int p = 0; p < np; ++p) s += C(a, p) * Mprim[static_cast<std::size_t>(p) * np + q];
      tmp[static_cast<std::size_t>(a) * np + q] = s;
    }
  std::vector<Real> Msto(static_cast<std::size_t>(na) * na, Real(0));
  for (int a = 0; a < na; ++a)
    for (int b = 0; b < na; ++b) {
      Real s = 0;
      for (int q = 0; q < np; ++q) s += tmp[static_cast<std::size_t>(a) * np + q] * C(b, q);
      Msto[static_cast<std::size_t>(a) * na + b] = s;
    }
  return Msto;
}

} // namespace intti
