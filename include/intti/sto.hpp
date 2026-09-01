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

/// A Slater-type-orbital shell: r^{n-1} Y_lm exp(-zeta r) on `center` (principal
/// quantum number n >= l+1), expanded with `ns` Gaussian nodes over the radial
/// transform variable s. The angular r^l Y_lm is the shell's Cartesian
/// solid harmonic; the extra radial power r^{n-1-l} is carried by the
/// zeta-differentiated expansion coefficients.
template <class Real> struct StoShell {
  Real zeta;
  Real center[3];
  int l;
  int n{0};   ///< principal quantum number; 0 (or l+1) means the minimal STO
  int ns{48};
};

/// Value of P_m(zeta), where f^{(m)}(zeta) = P_m(zeta) e^{-a zeta^2} with
/// f = zeta e^{-a zeta^2}: P_0 = zeta, P_{m+1} = P_m' - 2 a zeta P_m. Used for
/// the radial power: r^m e^{-zeta r} = (-d/dzeta)^m e^{-zeta r}.
template <class Real> Real sto_Pm(Real zeta, Real a, int m) {
  std::vector<Real> P = {Real(0), Real(1)}; // zeta
  for (int j = 0; j < m; ++j) {
    std::vector<Real> Pp(P.size(), Real(0)); // derivative
    for (std::size_t i = 1; i < P.size(); ++i) Pp[i - 1] = Real(i) * P[i];
    std::vector<Real> Pn(P.size() + 1, Real(0));
    for (std::size_t i = 0; i < P.size(); ++i) {
      Pn[i] += Pp[i];               // P'
      Pn[i + 1] += -2 * a * P[i];   // -2 a zeta P
    }
    P = Pn;
  }
  Real v = 0, zp = 1;
  for (std::size_t i = 0; i < P.size(); ++i) {
    v += P[i] * zp;
    zp *= zeta;
  }
  return v;
}

/// s-expansion of r^m exp(-zeta r) = sum_k c_k exp(-s_k r^2) (m the extra radial
/// power beyond the angular r^l): log-space Gauss-Legendre nodes s_k in
/// [smin, smax] with the transform weight and the m zeta-derivatives folded into
/// c_k. m = 0 is the plain exponential exp(-zeta r).
template <class Real>
void sto_gaussians(Real zeta, int ns, std::vector<Real> &s, std::vector<Real> &c, int m = 0,
                   Real smin = Real(1e-4), Real smax = Real(1e6)) {
  s.assign(ns, Real(0));
  c.assign(ns, Real(0));
  // Gauss-Legendre nodes/weights on [-1,1] (Newton on Legendre P_ns)
  const Real pi = pi_v<Real>();
  const int mhalf = (ns + 1) / 2;
  std::vector<Real> x(ns), w(ns);
  for (int i = 0; i < mhalf; ++i) {
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
  const Real sign = (m % 2 == 0) ? Real(1) : Real(-1); // (-d/dzeta)^m
  for (int k = 0; k < ns; ++k) {
    const Real u = Real(0.5) * (hi - lo) * (x[k] + 1) + lo;
    const Real sk = std::exp(u);
    const Real ws = Real(0.5) * (hi - lo) * w[k] * sk; // ds = s du
    const Real a = Real(1) / (4 * sk);
    s[k] = sk;
    // (-d/dzeta)^m [ zeta e^{-zeta^2/4s} ] = (-1)^m P_m(zeta) e^{-zeta^2/4s}
    c[k] = sign * sto_Pm(zeta, a, m) / (2 * std::sqrt(pi)) * ws * std::pow(sk, Real(-1.5)) *
           std::exp(-zeta * zeta / (4 * sk));
  }
}

/// Delta-function tail weight of the s-quadrature truncated at s_c:
///   W = int_{s_c}^inf g(s,zeta) (pi/s)^{3/2} ds = (8 pi / zeta^3)[1 - (1+u)e^{-u}],
/// u = zeta^2/(4 s_c). For a smooth kernel V, the tail of the STO's s-expansion
/// (the tight, delta-like Gaussians beyond s_c) contributes W * V(centre) to
/// int phi_STO(r) V(r) d^3r -- the same delta-function tail correction used for
/// the Coulomb t-quadrature (Losilla et al.), now on the s-quadrature, so far
/// fewer s-nodes are needed to resolve the r=0 cusp. (m = 0, 1s radial.)
template <class Real> Real sto_delta_tail_weight(Real zeta, Real s_c) {
  const Real pi = pi_v<Real>();
  const Real u = zeta * zeta / (4 * s_c);
  return (8 * pi / (zeta * zeta * zeta)) * (Real(1) - (1 + u) * std::exp(-u));
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
    // radial power beyond r^l: m = (n-1) - l; n<=0 defaults to the minimal STO
    const int nn = shells[i].n > 0 ? shells[i].n : shells[i].l + 1;
    const int m = nn - 1 - shells[i].l;
    sto_gaussians(shells[i].zeta, shells[i].ns, sk, ck[i], m);
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
