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
  Real smin{1e-4}, smax{1e6}; ///< s-grid range (smax = s_c truncates the tail)
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
    sto_gaussians(shells[i].zeta, shells[i].ns, sk, ck[i], m, shells[i].smin, shells[i].smax);
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

namespace detail {
/// Lower incomplete gamma gamma(k, x) for integer k >= 1:
///   gamma(k,x) = (k-1)! [1 - e^{-x} sum_{j=0}^{k-1} x^j/j!].
template <class Real> Real lower_gamma_int(int k, Real x) {
  Real fact = 1;
  for (int i = 2; i < k; ++i) fact *= i; // (k-1)!
  Real sum = 0, term = 1; // x^j/j!
  for (int j = 0; j < k; ++j) {
    sum += term;
    term *= x / (j + 1);
  }
  return fact * (Real(1) - std::exp(-x) * sum);
}

/// Delta-tail weight of a Cartesian component with powers a[3] (minimal STO,
/// radial e^{-zeta r}): the tight tail Gaussian x^a e^{-s r^2} acts as
/// derivatives of delta, so
///   W = C (pi zeta/2) (4/zeta^2)^{N+2} gamma(N+2, u_c),
/// C = prod_d (2 n_d - 1)!! / 2^{n_d}, n_d = (a_d + a_d%2)/2, N = sum n_d,
/// u_c = zeta^2/(4 s_c). (a = 0 -> the 1s weight 8 pi/zeta^3[1-(1+u)e^{-u}].)
template <class Real> Real sto_tail_weight_comp(Real zeta, const int a[3], Real s_c) {
  int N = 0;
  Real C = 1;
  for (int d = 0; d < 3; ++d) {
    const int nd = (a[d] + (a[d] & 1)) / 2;
    N += nd;
    Real df = 1; // (2 nd - 1)!!
    for (int t = 2 * nd - 1; t > 0; t -= 2) df *= t;
    Real p2 = 1;
    for (int t = 0; t < nd; ++t) p2 *= 2;
    C *= df / p2;
  }
  const Real u = zeta * zeta / (4 * s_c);
  const Real pi = pi_v<Real>();
  Real pw = 1;
  for (int t = 0; t < N + 2; ++t) pw *= 4 / (zeta * zeta);
  return C * (pi * zeta / 2) * pw * lower_gamma_int(N + 2, u);
}

/// m-th derivative d_u^m [u^a e^{-s u^2}] at u, obtained by iterating the
/// coefficient map Poly(u) -> Poly'(u) - 2 s u Poly(u) on Poly_0 = u^a. delta in
/// {0,1} recovers the old parity-derivative g1d.
template <class Real> Real g1d(int a, int m, Real s, Real u) {
  std::vector<Real> c(a + 1, Real(0));
  c[a] = 1;
  for (int step = 0; step < m; ++step) {
    std::vector<Real> nc(c.size() + 1, Real(0));
    for (int n = 0; n < static_cast<int>(c.size()); ++n) {
      if (n >= 1) nc[n - 1] += Real(n) * c[n];
      nc[n + 1] += Real(-2 * s) * c[n];
    }
    c.swap(nc);
  }
  Real v = 0;
  for (int n = static_cast<int>(c.size()) - 1; n >= 0; --n) v = v * u + c[n];
  return v * std::exp(-s * u * u);
}

/// Higher-order delta-tail weight for a Cartesian component with powers a[3] and
/// per-axis extra Laplacian orders p[3]. The tail of the STO samples the
/// partner's derivatives of order dB_d + 2 p_d (dB_d = a_d % 2 the leading
/// parity), weighted by
///   W^{(p)} = [prod_d (2(N_d+p_d)-1)!! / (dB_d+2p_d)!]
///             * 8 pi * 2^{N+k} gamma(N+k+2, u_c) / zeta^{3+2(N+k)},
/// N_d = (a_d + a_d%2)/2, N = sum N_d, k = sum p_d, u_c = zeta^2/(4 s_c). p = 0
/// recovers sto_tail_weight_comp; a = 0 gives the 1s radial weight
/// W_k (nabla^2)^k after the multinomial sum over p. (references/sympy_slater.py.)
template <class Real>
Real sto_tail_weight_comp_p(Real zeta, const int a[3], const int p[3], Real s_c) {
  int N = 0, k = 0;
  Real coef = 1;
  for (int d = 0; d < 3; ++d) {
    const int Nd = (a[d] + (a[d] & 1)) / 2;
    const int nd = Nd + p[d];
    N += Nd;
    k += p[d];
    Real df = 1; // (2 nd - 1)!!
    for (int t = 2 * nd - 1; t > 0; t -= 2) df *= t;
    Real mf = 1; // (dB_d + 2 p_d)!
    for (int t = 2; t <= (a[d] & 1) + 2 * p[d]; ++t) mf *= t;
    coef *= df / mf;
  }
  const Real pi = pi_v<Real>();
  const Real u = zeta * zeta / (4 * s_c);
  const int Nk = N + k;
  Real two = 1;
  for (int t = 0; t < Nk; ++t) two *= 2; // 2^{N+k}
  Real zpow = 1;
  for (int t = 0; t < 3 + 2 * Nk; ++t) zpow *= zeta; // zeta^{3+2(N+k)}
  return coef * 8 * pi * two * lower_gamma_int(Nk + 2, u) / zpow;
}
} // namespace detail

/// STO overlap matrix built with the delta-tail acceleration, for minimal STOs
/// of any angular momentum. Each orbital's s-integral is split at s_c: the
/// low-s part (s <= s_c) is a contracted GTO handled by the ordinary overlap
/// machinery, and the delta-like high-s tail is added as a centre correction.
/// For l>0 the angular polynomial vanishes at the centre, so the tail acts as
/// *derivatives* of delta: an off-centre partner contributes W_partner^{comp}
/// times the parity-order derivatives (order a_d mod 2 per direction) of the
/// other orbital's low-s part at the partner's centre. The tail-tail term
/// vanishes for distinct centres; the diagonal block is taken from a dense
/// single-centre grid. Reproduces the full-grid overlap from a coarse,
/// truncated s-grid.
/// radial_order > 0 adds the higher-order (Gaussian-smoothing) delta-tail terms
/// -- the tight tail Gaussians sample the partner's derivatives of order
/// (parity + 2p) summed over p, sum_{1<=|p|<=radial_order} W^{(p)} d^{dB+2p}
/// phi_partner(centre) -- for any angular momentum (1s reduces to the pure
/// (nabla^2)^k radial series). A coarser truncated s-grid then reaches the same
/// accuracy. Capped at TAIL_KMAX.
template <class Real>
std::vector<Real> sto_overlap_delta(const std::vector<StoShell<Real>> &shells, Real s_c,
                                    int ns_low, int ns_dense = 128, int radial_order = 0) {
  const int nsh = static_cast<int>(shells.size());
  const int Krad = radial_order < TAIL_KMAX ? radial_order : TAIL_KMAX;
  // STO AO offsets
  std::vector<int> off(nsh + 1, 0);
  for (int i = 0; i < nsh; ++i) off[i + 1] = off[i] + ncart(shells[i].l);
  const int nao = off.back();
  // low-s truncated contracted overlap over all shells
  std::vector<StoShell<Real>> tsh = shells;
  for (auto &s : tsh) {
    s.smax = s_c;
    s.ns = ns_low;
  }
  auto ex = expand_sto(tsh);
  auto LL = contract_to_sto(overlap_matrix(ex.prim), ex);
  // per-shell low-s nodes/coeffs
  std::vector<std::vector<Real>> sk(nsh), ck(nsh);
  for (int i = 0; i < nsh; ++i) {
    const int nn = shells[i].n > 0 ? shells[i].n : shells[i].l + 1;
    sto_gaussians(shells[i].zeta, ns_low, sk[i], ck[i], nn - 1 - shells[i].l, shells[i].smin,
                  s_c);
  }
  // D^{delta}[phi_A^{compA,low}](p): parity derivative of A's low-s component
  auto dval = [&](int A, const int aA[3], const int delta[3], const Real p[3]) {
    const Real u[3] = {p[0] - shells[A].center[0], p[1] - shells[A].center[1],
                       p[2] - shells[A].center[2]};
    Real v = 0;
    for (std::size_t k = 0; k < sk[A].size(); ++k) {
      Real prod = ck[A][k];
      for (int d = 0; d < 3; ++d) prod *= detail::g1d(aA[d], delta[d], sk[A][k], u[d]);
      v += prod;
    }
    return v;
  };
  std::vector<Real> S(static_cast<std::size_t>(nao) * nao, Real(0));
  for (int A = 0; A < nsh; ++A)
    for (int B = 0; B < nsh; ++B) {
      const int lA = shells[A].l, lB = shells[B].l;
      if (A == B) {
        // dense single-centre self-overlap block
        StoShell<Real> one = shells[A];
        one.smax = Real(1e6);
        one.ns = ns_dense;
        auto exd = expand_sto(std::vector<StoShell<Real>>{one});
        auto blk = contract_to_sto(overlap_matrix(exd.prim), exd);
        const int nc = ncart(lA);
        for (int i = 0; i < nc; ++i)
          for (int j = 0; j < nc; ++j)
            S[static_cast<std::size_t>(off[A] + i) * nao + off[A] + j] = blk[i * nc + j];
        continue;
      }
      for (int cA = 0; cA < ncart(lA); ++cA) {
        int aA[3];
        cart_comp(lA, cA, aA[0], aA[1], aA[2]);
        for (int cB = 0; cB < ncart(lB); ++cB) {
          int aB[3];
          cart_comp(lB, cB, aB[0], aB[1], aB[2]);
          const int dB[3] = {aB[0] & 1, aB[1] & 1, aB[2] & 1};
          const int dA[3] = {aA[0] & 1, aA[1] & 1, aA[2] & 1};
          const std::size_t idx =
              static_cast<std::size_t>(off[A] + cA) * nao + off[B] + cB;
          Real v = LL[idx];
          v += detail::sto_tail_weight_comp(shells[B].zeta, aB, s_c) *
               dval(A, aA, dB, shells[B].center);
          v += detail::sto_tail_weight_comp(shells[A].zeta, aA, s_c) *
               dval(B, aB, dA, shells[A].center);
          // higher-order tail: the tight tail Gaussians sample the partner's
          // derivatives of order dB + 2p (parity + radial pairs), summed over
          // p with 1 <= |p| <= Krad (|p|=0 is the leading term above). Works for
          // any l (1s -> the (nabla^2)^k series after the multinomial over p).
          for (int px = 0; px <= Krad; ++px)
            for (int py = 0; py <= Krad - px; ++py)
              for (int pz = 0; pz <= Krad - px - py; ++pz) {
                if (px + py + pz == 0) continue;
                const int pp[3] = {px, py, pz};
                const int mB[3] = {dB[0] + 2 * px, dB[1] + 2 * py, dB[2] + 2 * pz};
                v += detail::sto_tail_weight_comp_p(shells[B].zeta, aB, pp, s_c) *
                     dval(A, aA, mB, shells[B].center);
                const int mA[3] = {dA[0] + 2 * px, dA[1] + 2 * py, dA[2] + 2 * pz};
                v += detail::sto_tail_weight_comp_p(shells[A].zeta, aA, pp, s_c) *
                     dval(B, aB, mA, shells[A].center);
              }
          S[idx] = v;
        }
      }
    }
  return S;
}

/// Transform an STO-basis density to the primitive GTO basis: D_prim = C^T D C
/// (nprim_ao x nprim_ao), the reverse of contract_to_sto.
template <class Real>
std::vector<Real> expand_density_to_prim(const std::vector<Real> &Dsto,
                                         const StoExpansion<Real> &ex) {
  const int na = ex.nsto_ao, np = ex.nprim_ao;
  auto C = [&](int a, int p) { return ex.coeff[static_cast<std::size_t>(a) * np + p]; };
  std::vector<Real> DC(static_cast<std::size_t>(na) * np, Real(0)); // D C
  for (int a = 0; a < na; ++a)
    for (int q = 0; q < np; ++q) {
      Real s = 0;
      for (int b = 0; b < na; ++b) s += Dsto[static_cast<std::size_t>(a) * na + b] * C(b, q);
      DC[static_cast<std::size_t>(a) * np + q] = s;
    }
  std::vector<Real> Dp(static_cast<std::size_t>(np) * np, Real(0)); // C^T (D C)
  for (int p = 0; p < np; ++p)
    for (int q = 0; q < np; ++q) {
      Real s = 0;
      for (int a = 0; a < na; ++a) s += C(a, p) * DC[static_cast<std::size_t>(a) * np + q];
      Dp[static_cast<std::size_t>(p) * np + q] = s;
    }
  return Dp;
}

/// STO-basis Coulomb (J) and exchange (K) from an STO density: every STO AO is
/// a contracted GTO, so the STO ERI is a contraction of primitive ERIs. Push
/// the density to the primitives (C^T D C), run the ordinary J/K builds, and
/// contract the results back (C J C^T). Matrix-level; tau screens the
/// primitive builds.
template <class Real> struct StoJK {
  std::vector<Real> J, K;
};

template <class Real>
StoJK<Real> sto_jk_build(const std::vector<StoShell<Real>> &shells,
                         const std::vector<Real> &Dsto, const TGrid<Real> &grid,
                         Real tau = Real(0)) {
  auto ex = expand_sto(shells);
  auto Dp = expand_density_to_prim(Dsto, ex);
  const int np = ex.nprim_ao;
  std::vector<Real> Jp(static_cast<std::size_t>(np) * np, Real(0));
  std::vector<Real> Kp(static_cast<std::size_t>(np) * np, Real(0));
  coulomb_build(ex.prim, Dp.data(), grid, Jp.data(), tau);
  exchange_build(ex.prim, Dp.data(), grid, Kp.data(), tau, 0, 1);
  StoJK<Real> out;
  out.J = contract_to_sto(Jp, ex);
  out.K = contract_to_sto(Kp, ex);
  return out;
}

namespace detail {
/// Boys F0(x) = int_0^1 e^{-x t^2} dt = (1/2) sqrt(pi/x) erf(sqrt x), F0(0)=1.
template <class Real> Real boys0(Real x) {
  if (x < Real(1e-13)) return Real(1) - x / 3;
  return Real(0.5) * std::sqrt(pi_v<Real>() / x) * std::erf(std::sqrt(x));
}
/// Two-centre (ss|ss) Coulomb of unnormalized Gaussians e^{-p r_A^2}, e^{-q r_B^2}:
///   2 pi^{5/2} / (p q sqrt(p+q)) F0(pq/(p+q) R^2).
template <class Real> Real ss_coulomb(Real p, Real q, Real R2) {
  const Real pi = pi_v<Real>();
  return 2 * std::pow(pi, Real(2.5)) / (p * q * std::sqrt(p + q)) *
         boys0(p * q / (p + q) * R2);
}
} // namespace detail

/// Coulomb potential at distance R of a normalized 1s Slater density
/// rho = (zeta^3/pi) e^{-2 zeta r} (unit charge):
///   V(R) = (1/R)[1 - (1 + zeta R) e^{-2 zeta R}],  V(0) = zeta.
template <class Real> Real sto_slater_potential(Real zeta, Real R) {
  if (R < Real(1e-12)) return zeta;
  return (Real(1) / R) * (Real(1) - (1 + zeta * R) * std::exp(-2 * zeta * R));
}

/// Two-centre Coulomb repulsion (rho_A|rho_B) between the 1s Slater densities of
/// orbitals of exponents zeta_A, zeta_B (rho = phi^2), via the s-expansion of
/// each density (exponent 2 zeta) contracted through the analytic ss Coulomb.
template <class Real>
Real sto_coulomb_2c(Real zA, const Real A[3], Real zB, const Real B[3], int ns = 96) {
  std::vector<Real> tA, dA, tB, dB;
  sto_gaussians(2 * zA, ns, tA, dA);
  sto_gaussians(2 * zB, ns, tB, dB);
  const Real pi = pi_v<Real>();
  Real R2 = 0;
  for (int d = 0; d < 3; ++d) R2 += (A[d] - B[d]) * (A[d] - B[d]);
  const Real CA = zA * zA * zA / pi, CB = zB * zB * zB / pi;
  Real J = 0;
  for (std::size_t k = 0; k < tA.size(); ++k)
    for (std::size_t m = 0; m < tB.size(); ++m)
      J += CA * dA[k] * CB * dB[m] * detail::ss_coulomb(tA[k], tB[m], R2);
  return J;
}

/// (rho_A|rho_B) with the two-electron delta-tail acceleration: rho_A's s-grid
/// is truncated at t_c, and its tight, delta-like tail charge contributes
/// Q_tail * V_B(A) -- the tail sits at A and samples the smooth Coulomb
/// potential of rho_B there (B != A). rho_B is kept on a full grid. Reproduces
/// sto_coulomb_2c from a coarse truncated rho_A grid.
template <class Real>
Real sto_coulomb_2c_delta(Real zA, const Real A[3], Real zB, const Real B[3], Real t_c,
                          int ns_low, int ns_B = 96) {
  std::vector<Real> tA, dA, tB, dB;
  sto_gaussians(2 * zA, ns_low, tA, dA, 0, Real(1e-4), t_c); // truncated at t_c
  sto_gaussians(2 * zB, ns_B, tB, dB);
  const Real pi = pi_v<Real>();
  Real R2 = 0;
  for (int d = 0; d < 3; ++d) R2 += (A[d] - B[d]) * (A[d] - B[d]);
  const Real R = std::sqrt(R2);
  const Real CA = zA * zA * zA / pi, CB = zB * zB * zB / pi;
  Real J = 0;
  for (std::size_t k = 0; k < tA.size(); ++k)
    for (std::size_t m = 0; m < tB.size(); ++m)
      J += CA * dA[k] * CB * dB[m] * detail::ss_coulomb(tA[k], tB[m], R2);
  // delta tail of rho_A: tail charge Q_tail times rho_B's potential at A
  const Real Qtail = CA * sto_delta_tail_weight(2 * zA, t_c);
  J += Qtail * sto_slater_potential(zB, R);
  return J;
}

namespace detail {
/// Radial delta-tail weight of a 1s density of decay zd (= 2 zeta):
///   W_k = (8 pi / (k! zd^{3+2k})) gamma(k+2, u_c),  u_c = zd^2/(4 t_c).
/// The tail of the density samples the partner potential's Laplacians:
/// (rho^tail | V) = sum_k W_k (nabla^2)^k V(centre) (times the density norm).
template <class Real> Real sto_radial_weight(Real zd, Real t_c, int k) {
  const Real pi = pi_v<Real>();
  const Real u = zd * zd / (4 * t_c);
  Real zpow = 1;
  for (int i = 0; i < 3 + 2 * k; ++i) zpow *= zd;
  Real kfact = 1;
  for (int i = 2; i <= k; ++i) kfact *= i;
  return 8 * pi / (kfact * zpow) * lower_gamma_int(k + 2, u);
}

/// Same-centre Coulomb of two 1s Slater densities rho = phi^2 (orbital exponents
/// zA, zB, density decays 2 zA, 2 zB), in closed form:
///   (rho_A|rho_B) = zA - zA^3/s^2 - zA^3 zB/s^3,  s = zA + zB.
/// Symmetric in A,B; reduces to 5 zeta/8 for zA=zB. (references/sympy_slater.py.)
/// Handles R=0 exactly, so the delta-tail is used only for R>0 (all smooth).
template <class Real> Real sto_coulomb_1c(Real zA, Real zB) {
  const Real s = zA + zB;
  return zA - zA * zA * zA / (s * s) - zA * zA * zA * zB / (s * s * s);
}
} // namespace detail

/// Symmetric two-electron delta-tail (rho_A|rho_B) of two 1s Slater densities
/// rho = phi^2 (the pair-density level -- see prototype/sto_delta_tail_j.py).
/// Same centre (R=0) is exact via the closed-form 1-centre Coulomb
/// (detail::sto_coulomb_1c), so the delta tail is used only for R>0, where every
/// term is smooth. There BOTH densities are truncated at t_c and, exact under the
/// point-charge tail approximation,
///   (rho_A|rho_B) = (low_A|low_B) + (tail_A|rho_B) + (rho_A|tail_B) - Q_A Q_B/R,
/// with the cross terms carried to order radial_order via the Gaussian-smoothing
/// series (tail_A|rho_B) = sum_k W_{A,k} (nabla^2)^k V_B(A). Poisson makes those
/// closed form: (nabla^2)^k V_B = -4 pi (nabla^2)^{k-1} rho_B, and for a Slater
/// density (nabla^2)^j e^{-kappa r} = (kappa^{2j} - 2j kappa^{2j-1}/r) e^{-kappa r}.
/// Order 0 recovers Q_A V_B(A) + Q_B V_A(B); each order removes a 1/t_c^2 factor.
template <class Real>
Real sto_coulomb_2c_delta_sym(Real zA, const Real A[3], Real zB, const Real B[3],
                              Real t_c, int ns_low, int radial_order = 0) {
  const Real pi = pi_v<Real>();
  Real R2 = 0;
  for (int d = 0; d < 3; ++d) R2 += (A[d] - B[d]) * (A[d] - B[d]);
  const Real R = std::sqrt(R2);
  if (R < Real(1e-9)) return detail::sto_coulomb_1c(zA, zB);
  const int K = radial_order < TAIL_KMAX ? radial_order : TAIL_KMAX;
  std::vector<Real> tA, dA, tB, dB;
  sto_gaussians(2 * zA, ns_low, tA, dA, 0, Real(1e-4), t_c);
  sto_gaussians(2 * zB, ns_low, tB, dB, 0, Real(1e-4), t_c);
  const Real CA = zA * zA * zA / pi, CB = zB * zB * zB / pi;
  Real J = 0;
  for (std::size_t k = 0; k < tA.size(); ++k)
    for (std::size_t m = 0; m < tB.size(); ++m)
      J += CA * dA[k] * CB * dB[m] * detail::ss_coulomb(tA[k], tB[m], R2);
  // (tail of density with norm CT, decay 2 zT | potential of the zP density at R)
  auto cross = [&](Real zT, Real CT, Real zP, Real CP) {
    Real v = CT * sto_delta_tail_weight(2 * zT, t_c) * sto_slater_potential(zP, R);
    const Real kappa = 2 * zP, eKR = std::exp(-kappa * R);
    for (int k = 1; k <= K; ++k) {
      // (nabla^2)^{k-1} rho_P(R) = CP (kappa^{2(k-1)} - 2(k-1) kappa^{2k-3}/R) e^{-kR}
      Real k2 = 1;
      for (int t = 0; t < 2 * (k - 1); ++t) k2 *= kappa;
      Real term2 = 0;
      if (k >= 2) {
        Real k3 = 1;
        for (int t = 0; t < 2 * k - 3; ++t) k3 *= kappa;
        term2 = 2 * (k - 1) * k3 / R;
      }
      const Real lapV = -4 * pi * CP * (k2 - term2) * eKR; // (nabla^2)^k V_P(R)
      v += CT * detail::sto_radial_weight(2 * zT, t_c, k) * lapV;
    }
    return v;
  };
  J += cross(zA, CA, zB, CB) + cross(zB, CB, zA, CA);
  const Real QA = CA * sto_delta_tail_weight(2 * zA, t_c);
  const Real QB = CB * sto_delta_tail_weight(2 * zB, t_c);
  J -= QA * QB / R;
  return J;
}

} // namespace intti
