// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// GIAOs / London orbitals: integrals in a finite magnetic field.
//
// A London orbital carries a field-dependent plane-wave phase,
//
//   omega_a(r) = exp(-i/2 [B x (R_a - O)] . r) chi_a(r),
//
// with O the gauge origin. The bra-side product of two of them,
// omega_a^*(r) omega_b(r), therefore carries the phase exp(-i K_ab . r) with
//
//   K_ab = (1/2) B x (R_b - R_a),
//
// which depends only on the *difference* of the centres: the gauge origin
// cancels, so GIAO integrals are gauge-origin independent by construction
// (test_giao.cpp checks exactly this).
//
// Multiplying the Gaussian product exp(-p (r-P)^2) by that plane wave gives
//
//   exp(-p (r - Ptilde)^2) * exp(-i K.P - K^2/(4p)),   Ptilde = P - i K/(2p),
//
// i.e. a Gaussian with the SAME REAL EXPONENT p and a COMPLEX CENTRE. Every
// step of libintti's machinery -- the Gaussian product theorem, the
// McMurchie-Davidson E coefficients, the Hermite B_n recursion, the
// t quadrature -- is pure algebra in the centres, so it continues to complex
// centres unchanged: a GIAO ERI is just eri_quartet() instantiated on
// std::complex.
//
// This is a structural advantage of the quadrature route. The analytic route
// needs the Boys function of a *complex* argument, which is awkward; the
// t quadrature never forms a Boys function at all. Measured: the default
// 64-node Mobius grid reproduces analytic complex-Boys GIAO ERIs to 3e-15 at
// fields up to B = 5 a.u., with no extra nodes.

#include <array>
#include <cmath>
#include <complex>
#include <vector>

#include "gto.hpp"
#include "hermite1d.hpp"
#include "math.hpp"
#include "nuclear.hpp"
#include "oneel.hpp"
#include "tgrid.hpp"
#include "traits.hpp"

namespace intti {

/// K_ab = (1/2) B x (R_b - R_a): the plane-wave vector of the London pair
/// product omega_a^* omega_b. Gauge-origin independent.
template <class Real>
void giao_k(const PrimitiveShell<Real> &a, const PrimitiveShell<Real> &b,
            const Real B[3], Real K[3]) {
  const Real d[3] = {b.center[0] - a.center[0], b.center[1] - a.center[1],
                     b.center[2] - a.center[2]};
  K[0] = Real(0.5) * (B[1] * d[2] - B[2] * d[1]);
  K[1] = Real(0.5) * (B[2] * d[0] - B[0] * d[2]);
  K[2] = Real(0.5) * (B[0] * d[1] - B[1] * d[0]);
}

/// London (GIAO) shell pair omega_a^*(r) omega_b(r) in magnetic field B.
/// The result is an ordinary ShellPair with a complex product centre, so it
/// can be handed straight to eri_quartet() and the one-electron drivers.
///
/// Note the conjugation convention: the FIRST shell is the conjugated (bra)
/// one. The ket pair of a two-electron integral uses the same function.
template <class Real>
ShellPair<std::complex<Real>> make_giao_pair(const PrimitiveShell<Real> &a,
                                             const PrimitiveShell<Real> &b,
                                             const Real B[3]) {
  using C = std::complex<Real>;
  static_assert(!is_complex_v<Real>, "make_giao_pair takes real shells");
  ShellPair<C> sp;
  sp.p = a.alpha + b.alpha;
  const Real mu = a.alpha * b.alpha / sp.p;
  Real K[3];
  giao_k(a, b, B, K);

  // real product centre and the field-free prefactor
  Real P[3], K0[3];
  for (int d = 0; d < 3; ++d) {
    P[d] = (a.alpha * a.center[d] + b.alpha * b.center[d]) / sp.p;
    const Real AB = a.center[d] - b.center[d];
    K0[d] = exp_(-mu * AB * AB);
  }

  // exp(-i K.r) shifts the centre into the complex plane and contributes
  //   exp(-i K.P - K^2/(4p)),
  // which factorises over the Cartesian directions.
  for (int d = 0; d < 3; ++d) {
    sp.P[d] = C(P[d], -K[d] / (2 * sp.p));
    const Real damp = exp_(-K[d] * K[d] / (4 * sp.p));
    const Real phase = -K[d] * P[d];
    sp.K[d] = C(K0[d] * damp) * C(std::cos(phase), std::sin(phase)); // exp(-i K_d P_d)
    sp.A[d] = a.center[d];
    sp.B[d] = b.center[d];
  }
  sp.la = a.l;
  sp.lb = b.l;
  return sp;
}

namespace detail {

/// std::complex ShellPair -> Kokkos::complex ShellPair (device-runnable).
/// std::complex cannot be used inside a device kernel; Kokkos::complex can, and
/// carries the same algebra, so this is the bridge from the host-built London
/// pairs to the device path.
template <class Real>
ShellPair<Kokkos::complex<Real>> to_kokkos_pair(const ShellPair<std::complex<Real>> &sp) {
  ShellPair<Kokkos::complex<Real>> o;
  o.p = sp.p;
  o.la = sp.la;
  o.lb = sp.lb;
  for (int d = 0; d < 3; ++d) {
    o.P[d] = Kokkos::complex<Real>(sp.P[d].real(), sp.P[d].imag());
    o.K[d] = Kokkos::complex<Real>(sp.K[d].real(), sp.K[d].imag());
    o.A[d] = sp.A[d];
    o.B[d] = sp.B[d];
  }
  return o;
}

/// London pairs for the finite-field 1e builders, as a device PairTable of
/// Kokkos::complex plus the side arrays the kernels need. Every ordered (a,b)
/// pair is stored (the London phases make S,T,V Hermitian, not symmetric, and
/// the host builders loop all pairs too). (exa,exb) extend the angular momenta
/// so kinetic (+2 ket) and the nuclear bra derivative have the indices they
/// need; la0/lb0 are the ORIGINAL momenta for the output loops.
template <class Real> struct Giao1ePairs {
  PairTable<Kokkos::complex<Real>> tab;
  Kokkos::View<int *> aoa, aob, la0, lb0;
  Kokkos::View<Real *> alpha, beta;
  int npair{0};
};

template <class Real>
Giao1ePairs<Real> make_giao_1e_pairs(const ShellBasis<Real> &basis, const Real B[3],
                                     int exa, int exb) {
  const int ns = static_cast<int>(basis.shells.size());
  std::vector<ShellPair<Kokkos::complex<Real>>> plist;
  std::vector<int> haoa, haob, hla0, hlb0;
  std::vector<Real> halpha, hbeta;
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      PrimitiveShell<Real> sae = basis.shells[a], sbe = basis.shells[b];
      sae.l += exa;
      sbe.l += exb;
      plist.push_back(to_kokkos_pair(make_giao_pair(sae, sbe, B)));
      haoa.push_back(basis.ao_off[a]);
      haob.push_back(basis.ao_off[b]);
      hla0.push_back(basis.shells[a].l);
      hlb0.push_back(basis.shells[b].l);
      halpha.push_back(basis.shells[a].alpha);
      hbeta.push_back(basis.shells[b].alpha);
    }
  Giao1ePairs<Real> gp;
  gp.tab = make_pair_table(plist);
  gp.npair = gp.tab.npair;
  gp.aoa = to_device(haoa, "intti::g1e::aoa");
  gp.aob = to_device(haob, "intti::g1e::aob");
  gp.la0 = to_device(hla0, "intti::g1e::la0");
  gp.lb0 = to_device(hlb0, "intti::g1e::lb0");
  gp.alpha = to_device(halpha, "intti::g1e::alpha");
  gp.beta = to_device(hbeta, "intti::g1e::beta");
  return gp;
}

/// Device finite-field overlap S(B). Only E is complex; the prefactor
/// sqrt(pi/p) is real because p = alpha+beta stays real for London pairs.
template <class Real>
std::vector<std::complex<Real>> giao_overlap_dev(const ShellBasis<Real> &basis,
                                                 const Real B[3]) {
  using KC = Kokkos::complex<Real>;
  const int nao = basis.nao;
  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
  auto gp = make_giao_1e_pairs(basis, B, 0, 0);
  Kokkos::View<Real *> Sr("intti::g1e::Sr", n2), Si("intti::g1e::Si", n2);
  auto pv = gp.tab.p, Ev = gp.tab.E;
  auto lav = gp.tab.la, lbv = gp.tab.lb, eoffv = gp.tab.e_off;
  auto aoa = gp.aoa, aob = gp.aob;
  const Real pi = pi_v<Real>();
  Kokkos::parallel_for(
      "intti::g1e::ovlp", Kokkos::RangePolicy<>(0, gp.npair), KOKKOS_LAMBDA(int p) {
        const int la = lav(p), lb = lbv(p), n1 = la + lb + 1;
        const int esz = (la + 1) * (lb + 1) * n1;
        const Real pref = sqrt_(pi / pv(p));
        const Real pref3 = pref * pref * pref;
        const int eo = eoffv(p), oa = aoa(p), ob = aob(p);
        for (int ka = 0; ka < ncart(la); ++ka) {
          int a3[3];
          cart_comp(la, ka, a3[0], a3[1], a3[2]);
          for (int kb = 0; kb < ncart(lb); ++kb) {
            int b3[3];
            cart_comp(lb, kb, b3[0], b3[1], b3[2]);
            const KC ex = Ev(eo + 0 * esz + (a3[0] * (lb + 1) + b3[0]) * n1);
            const KC ey = Ev(eo + 1 * esz + (a3[1] * (lb + 1) + b3[1]) * n1);
            const KC ez = Ev(eo + 2 * esz + (a3[2] * (lb + 1) + b3[2]) * n1);
            const KC v = ex * ey * ez * pref3;
            const std::size_t idx = static_cast<std::size_t>(oa + ka) * nao + ob + kb;
            Sr(idx) = v.real();
            Si(idx) = v.imag();
          }
        }
      });
  auto hr = to_host(Sr), hi = to_host(Si);
  std::vector<std::complex<Real>> S(n2);
  for (std::size_t i = 0; i < n2; ++i) S[i] = std::complex<Real>(hr[i], hi[i]);
  return S;
}

/// Device finite-field kinetic T(B), momentum form
/// 1/2 sum_c <d_c omega_mu | d_c omega_nu>. Each London-orbital gradient is
/// d_c omega = e^{i phi}(i a_c chi + d_c chi) with a = -1/2 B x (R-O), so the
/// 1D factor per direction mixes the phase term (imaginary, index unchanged)
/// with the ordinary MD shift (l+-1). Bra and ket are both extended by one.
template <class Real>
std::vector<std::complex<Real>> giao_kinetic_dev(const ShellBasis<Real> &basis,
                                                 const Real B[3], const Real O[3]) {
  using KC = Kokkos::complex<Real>;
  const int nao = basis.nao;
  const int ns = static_cast<int>(basis.shells.size());
  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
  auto gp = make_giao_1e_pairs(basis, B, 1, 1);
  // vector potential a = -1/2 B x (R - O) at the bra and ket centres of each
  // pair; pairs are ordered p = a*ns + b, matching make_giao_1e_pairs.
  auto avec = [&](const Real R[3], Real o[3]) {
    const Real w[3] = {R[0] - O[0], R[1] - O[1], R[2] - O[2]};
    o[0] = -Real(0.5) * (B[1] * w[2] - B[2] * w[1]);
    o[1] = -Real(0.5) * (B[2] * w[0] - B[0] * w[2]);
    o[2] = -Real(0.5) * (B[0] * w[1] - B[1] * w[0]);
  };
  Kokkos::View<Real *[3], Kokkos::LayoutLeft> amv("intti::g1e::am", gp.npair),
      anv("intti::g1e::an", gp.npair);
  {
    auto ha = Kokkos::create_mirror_view(amv), hb = Kokkos::create_mirror_view(anv);
    for (int a = 0; a < ns; ++a)
      for (int b = 0; b < ns; ++b) {
        Real am[3], an[3];
        avec(basis.shells[a].center, am);
        avec(basis.shells[b].center, an);
        const int p = a * ns + b;
        for (int d = 0; d < 3; ++d) {
          ha(p, d) = am[d];
          hb(p, d) = an[d];
        }
      }
    Kokkos::deep_copy(amv, ha);
    Kokkos::deep_copy(anv, hb);
  }
  Kokkos::View<Real *> Tr("intti::g1e::Tr", n2), Ti("intti::g1e::Ti", n2);
  auto pv = gp.tab.p, Ev = gp.tab.E;
  auto lav = gp.tab.la, lbv = gp.tab.lb, eoffv = gp.tab.e_off;
  auto aoa = gp.aoa, aob = gp.aob, alv = gp.alpha, bev = gp.beta;
  const Real pi = pi_v<Real>();
  Kokkos::parallel_for(
      "intti::g1e::kin", Kokkos::RangePolicy<>(0, gp.npair), KOKKOS_LAMBDA(int p) {
        const int lap = lav(p), lbp = lbv(p), la0 = lap - 1, lb0 = lbp - 1;
        const int n1 = lap + lbp + 1, esz = (lap + 1) * (lbp + 1) * n1;
        const Real pref = sqrt_(pi / pv(p)), al = alv(p), be = bev(p);
        const int eo = eoffv(p), oa = aoa(p), ob = aob(p);
        auto sd = [&](int d, int i, int j) {
          return pref * Ev(eo + d * esz + (i * (lbp + 1) + j) * n1);
        };
        for (int ka = 0; ka < ncart(la0); ++ka) {
          int a3[3];
          cart_comp(la0, ka, a3[0], a3[1], a3[2]);
          for (int kb = 0; kb < ncart(lb0); ++kb) {
            int b3[3];
            cart_comp(lb0, kb, b3[0], b3[1], b3[2]);
            auto facc = [&](int c) {
              KC bc[3];
              int bi[3];
              int nb = 0;
              bc[nb] = KC(Real(0), -amv(p, c)); bi[nb] = a3[c]; ++nb;
              bc[nb] = KC(-2 * al, Real(0));    bi[nb] = a3[c] + 1; ++nb;
              if (a3[c] >= 1) { bc[nb] = KC(Real(a3[c]), Real(0)); bi[nb] = a3[c] - 1; ++nb; }
              KC kc[3];
              int kj[3];
              int nk = 0;
              kc[nk] = KC(Real(0), anv(p, c)); kj[nk] = b3[c]; ++nk;
              kc[nk] = KC(-2 * be, Real(0));   kj[nk] = b3[c] + 1; ++nk;
              if (b3[c] >= 1) { kc[nk] = KC(Real(b3[c]), Real(0)); kj[nk] = b3[c] - 1; ++nk; }
              KC f(Real(0), Real(0));
              for (int q = 0; q < nb; ++q)
                for (int r = 0; r < nk; ++r) f += bc[q] * kc[r] * sd(c, bi[q], kj[r]);
              return f;
            };
            const KC bx = sd(0, a3[0], b3[0]), by = sd(1, a3[1], b3[1]), bz = sd(2, a3[2], b3[2]);
            const KC v = (facc(0) * by * bz + bx * facc(1) * bz + bx * by * facc(2)) * Real(0.5);
            const std::size_t idx = static_cast<std::size_t>(oa + ka) * nao + ob + kb;
            Tr(idx) = v.real();
            Ti(idx) = v.imag();
          }
        }
      });
  auto hr = to_host(Tr), hi = to_host(Ti);
  std::vector<std::complex<Real>> T(n2);
  for (std::size_t i = 0; i < n2; ++i) T[i] = std::complex<Real>(hr[i], hi[i]);
  return T;
}

/// Device finite-field nuclear attraction V(B). Identical in structure to the
/// real device attraction: only the product centre (and hence the hermite_b
/// argument and the E tables) is complex; theta, the prefactor and the weights
/// stay real because p = alpha+beta is real.
template <class Real>
std::vector<std::complex<Real>>
giao_nuclear_dev(const ShellBasis<Real> &basis, const std::vector<PointCharge<Real>> &charges,
                 const Real B[3], const TGrid<Real> &grid) {
  using KC = Kokkos::complex<Real>;
  const int nao = basis.nao;
  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
  auto gp = make_giao_1e_pairs(basis, B, 0, 0);
  const int nt = grid.n(), npc = static_cast<int>(charges.size());
  auto tv = to_device(grid.t, "intti::g1e::t"), wv = to_device(grid.w, "intti::g1e::w");
  std::vector<Real> hcw(npc);
  for (int c = 0; c < npc; ++c) hcw[c] = charges[c].weight;
  auto cw = to_device(hcw, "intti::g1e::cw");
  Kokkos::View<Real *[3], Kokkos::LayoutLeft> cR("intti::g1e::cR", npc);
  {
    auto h = Kokkos::create_mirror_view(cR);
    for (int c = 0; c < npc; ++c)
      for (int d = 0; d < 3; ++d) h(c, d) = charges[c].R[d];
    Kokkos::deep_copy(cR, h);
  }
  Kokkos::View<Real *> Vr("intti::g1e::Vr", n2), Vi("intti::g1e::Vi", n2);
  auto pv = gp.tab.p, Ev = gp.tab.E;
  auto Pv = gp.tab.P;
  auto lav = gp.tab.la, lbv = gp.tab.lb, eoffv = gp.tab.e_off;
  auto aoa = gp.aoa, aob = gp.aob;
  const Real pi = pi_v<Real>();
  Kokkos::parallel_for(
      "intti::g1e::nuc", Kokkos::RangePolicy<>(0, gp.npair), KOKKOS_LAMBDA(int p) {
        const int la = lav(p), lb = lbv(p), n1 = la + lb + 1;
        const int esz = (la + 1) * (lb + 1) * n1;
        const Real pp = pv(p);
        const int eo = eoffv(p), oa = aoa(p), ob = aob(p);
        const KC Px = Pv(p, 0), Py = Pv(p, 1), Pz = Pv(p, 2);
        KC Bx[2 * LMAX + 1], By[2 * LMAX + 1], Bz[2 * LMAX + 1];
        for (int c = 0; c < npc; ++c) {
          const KC Rx = Px - cR(c, 0), Ry = Py - cR(c, 1), Rz = Pz - cR(c, 2);
          const Real wc = cw(c);
          for (int it = 0; it < nt; ++it) {
            const Real t = tv(it), denom = pp + t * t, theta = pp * t * t / denom;
            const Real pref = sqrt_(pi / denom), wt = wv(it) * wc;
            hermite_b(la + lb, theta, Rx, Bx);
            hermite_b(la + lb, theta, Ry, By);
            hermite_b(la + lb, theta, Rz, Bz);
            for (int ka = 0; ka < ncart(la); ++ka) {
              int a3[3];
              cart_comp(la, ka, a3[0], a3[1], a3[2]);
              for (int kb = 0; kb < ncart(lb); ++kb) {
                int b3[3];
                cart_comp(lb, kb, b3[0], b3[1], b3[2]);
                KC g[3];
                for (int d = 0; d < 3; ++d) {
                  const int base = eo + d * esz + (a3[d] * (lb + 1) + b3[d]) * n1;
                  const KC *Bd = (d == 0 ? Bx : (d == 1 ? By : Bz));
                  KC s(Real(0), Real(0));
                  for (int tt = 0; tt <= a3[d] + b3[d]; ++tt) s += Ev(base + tt) * Bd[tt];
                  g[d] = s * pref;
                }
                const KC v = g[0] * g[1] * g[2] * wt;
                const std::size_t idx = static_cast<std::size_t>(oa + ka) * nao + ob + kb;
                Vr(idx) += v.real();
                Vi(idx) += v.imag();
              }
            }
          }
        }
      });
  auto hr = to_host(Vr), hi = to_host(Vi);
  std::vector<std::complex<Real>> V(n2);
  for (std::size_t i = 0; i < n2; ++i) V[i] = std::complex<Real>(hr[i], hi[i]);
  return V;
}

} // namespace detail

/// Complex GIAO overlap matrix S(B) = <omega_mu | omega_nu> in a finite
/// magnetic field B (nao x nao, row-major, over the primitive Cartesian AOs).
/// London orbitals shift the product Gaussian centre into the complex plane
/// (make_giao_pair), so the ordinary MD E-coefficient overlap runs unchanged
/// on complex scalars. At B = 0 this reduces to the real overlap_matrix.
template <class Real>
std::vector<std::complex<Real>> giao_overlap(const ShellBasis<Real> &basis,
                                             const Real B[3]) {
  using C = std::complex<Real>;
  static_assert(!is_complex_v<Real>, "giao_overlap takes a real basis");
  if constexpr (kokkos_scalar_v<Kokkos::complex<Real>>)
    return detail::giao_overlap_dev(basis, B);
  const int nao = basis.nao;
  std::vector<C> S(static_cast<std::size_t>(nao) * nao, C(0));
  const int ns = static_cast<int>(basis.shells.size());
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const auto sp = make_giao_pair(sa, sb, B); // complex centre + prefactor
      const int la = sa.l, lb = sb.l, lb1 = lb + 1, nt = la + lb + 1;
      std::vector<C> sf[3];
      for (int d = 0; d < 3; ++d) {
        std::vector<C> E(static_cast<std::size_t>(la + 1) * (lb + 1) * nt);
        const C PA = sp.P[d] - C(sa.center[d]);
        const C PB = sp.P[d] - C(sb.center[d]);
        e_coeffs(la, lb, sp.p, PA, PB, sp.K[d], E.data());
        const Real pref = sqrt_(pi_v<Real>() / sp.p);
        sf[d].assign(static_cast<std::size_t>(la + 1) * lb1, C(0));
        for (int i = 0; i <= la; ++i)
          for (int j = 0; j <= lb; ++j)
            sf[d][i * lb1 + j] = C(pref) * E[(i * (lb + 1) + j) * nt + 0];
      }
      for (int ka = 0; ka < ncart(la); ++ka) {
        int a3[3];
        cart_comp(la, ka, a3[0], a3[1], a3[2]);
        for (int kb = 0; kb < ncart(lb); ++kb) {
          int b3[3];
          cart_comp(lb, kb, b3[0], b3[1], b3[2]);
          S[(basis.ao_off[a] + ka) * static_cast<std::size_t>(nao) + basis.ao_off[b] + kb] =
              sf[0][a3[0] * lb1 + b3[0]] * sf[1][a3[1] * lb1 + b3[1]] *
              sf[2][a3[2] * lb1 + b3[2]];
        }
      }
    }
  return S;
}

/// Analytic magnetic-field derivative dS/dB_k of the GIAO overlap at B = 0.
///
/// The London pair omega_mu^* omega_nu carries the phase
/// exp(i/2 [B x (R_mu - R_nu)].r), so
///   dS_munu/dB_k |_{B=0} = (i/2) [e_k x (R_mu - R_nu)] . <mu| r |nu>,
/// a purely imaginary combination of the dipole moment matrices about the
/// coordinate origin (the gauge origin cancels between bra and ket). Returns
/// the three components {dS/dB_x, dS/dB_y, dS/dB_z}, each nao x nao row-major.
template <class Real>
std::array<std::vector<std::complex<Real>>, 3>
giao_overlap_dB(const ShellBasis<Real> &basis) {
  using C = std::complex<Real>;
  static_assert(!is_complex_v<Real>, "giao_overlap_dB takes a real basis");
  const int nao = basis.nao;
  std::array<std::vector<C>, 3> dS;
  for (auto &m : dS) m.assign(static_cast<std::size_t>(nao) * nao, C(0));
  const int ns = static_cast<int>(basis.shells.size());
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const int la = sa.l, lb = sb.l, lb1 = lb + 1;
      std::vector<Real> Sf[3], Df[3]; // 1D overlap and dipole <i|x_d|j> about 0
      for (int d = 0; d < 3; ++d) {
        std::vector<Real> s;
        int lbx;
        detail::overlap_1d(sa.alpha, sa.center[d], sb.alpha, sb.center[d], la, lb, 1, 0,
                           s, lbx);
        Sf[d].assign(static_cast<std::size_t>(la + 1) * lb1, Real(0));
        for (int i = 0; i <= la; ++i)
          for (int j = 0; j <= lb; ++j) Sf[d][i * lb1 + j] = s[i * (lbx + 1) + j];
        std::vector<Real> m1;
        detail::multipole_1d(s, lbx, la, lb, sa.center[d], Real(0), 1, m1);
        Df[d].assign(static_cast<std::size_t>(la + 1) * lb1, Real(0));
        for (int i = 0; i <= la; ++i)
          for (int j = 0; j <= lb; ++j)
            Df[d][i * lb1 + j] = m1[(1 * (la + 1) + i) * lb1 + j]; // e=1 block
      }
      const Real dv[3] = {sa.center[0] - sb.center[0], sa.center[1] - sb.center[1],
                          sa.center[2] - sb.center[2]};
      for (int ka = 0; ka < ncart(la); ++ka) {
        int a3[3];
        cart_comp(la, ka, a3[0], a3[1], a3[2]);
        for (int kb = 0; kb < ncart(lb); ++kb) {
          int b3[3];
          cart_comp(lb, kb, b3[0], b3[1], b3[2]);
          auto S = [&](int d) { return Sf[d][a3[d] * lb1 + b3[d]]; };
          // <mu| x_d |nu> = dipole in direction d times overlap in the others
          const Real D[3] = {Df[0][a3[0] * lb1 + b3[0]] * S(1) * S(2),
                             S(0) * Df[1][a3[1] * lb1 + b3[1]] * S(2),
                             S(0) * S(1) * Df[2][a3[2] * lb1 + b3[2]]};
          const std::size_t idx =
              (basis.ao_off[a] + ka) * static_cast<std::size_t>(nao) + basis.ao_off[b] + kb;
          const C half_i(Real(0), Real(0.5));
          dS[0][idx] = half_i * (-dv[2] * D[1] + dv[1] * D[2]);
          dS[1][idx] = half_i * (dv[2] * D[0] - dv[0] * D[2]);
          dS[2][idx] = half_i * (-dv[1] * D[0] + dv[0] * D[1]);
        }
      }
    }
  return dS;
}

/// Complex GIAO kinetic matrix T(B) = <omega_mu| -1/2 nabla^2 |omega_nu> in a
/// finite field B with gauge origin O. Unlike the overlap and nuclear cases,
/// the operator differentiates the London phase, so this is built from the
/// momentum form 1/2 sum_c <d_c omega_mu | d_c omega_nu>: each London-orbital
/// gradient d_c omega = e^{i phi}(i a_c chi + d_c chi) with a = -1/2 B x (R-O),
/// and every phase-weighted 1D overlap is the complex MD overlap table. Reduces
/// to the real kinetic_matrix at B = 0. Used to finite-difference giao_kinetic_dB.
template <class Real>
std::vector<std::complex<Real>> giao_kinetic(const ShellBasis<Real> &basis,
                                             const Real B[3], const Real O[3]) {
  using C = std::complex<Real>;
  static_assert(!is_complex_v<Real>, "giao_kinetic takes a real basis");
  if constexpr (kokkos_scalar_v<Kokkos::complex<Real>>)
    return detail::giao_kinetic_dev(basis, B, O);
  const int nao = basis.nao;
  std::vector<C> T(static_cast<std::size_t>(nao) * nao, C(0));
  const int ns = static_cast<int>(basis.shells.size());
  auto avec = [&](const Real R[3], Real out[3]) { // a = -1/2 B x (R - O)
    const Real w[3] = {R[0] - O[0], R[1] - O[1], R[2] - O[2]};
    out[0] = -Real(0.5) * (B[1] * w[2] - B[2] * w[1]);
    out[1] = -Real(0.5) * (B[2] * w[0] - B[0] * w[2]);
    out[2] = -Real(0.5) * (B[0] * w[1] - B[1] * w[0]);
  };
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const auto sp = make_giao_pair(sa, sb, B);
      const int la = sa.l, lb = sb.l, lap = la + 1, lbp = lb + 1, nt = lap + lbp + 1;
      std::vector<C> s[3]; // complex 1D overlap tables, bra..la+1, ket..lb+1
      for (int d = 0; d < 3; ++d) {
        std::vector<C> E(static_cast<std::size_t>(lap + 1) * (lbp + 1) * nt);
        e_coeffs(lap, lbp, sp.p, sp.P[d] - C(sa.center[d]), sp.P[d] - C(sb.center[d]),
                 sp.K[d], E.data());
        const Real pref = sqrt_(pi_v<Real>() / sp.p);
        s[d].assign(static_cast<std::size_t>(lap + 1) * (lbp + 1), C(0));
        for (int i = 0; i <= lap; ++i)
          for (int j = 0; j <= lbp; ++j)
            s[d][i * (lbp + 1) + j] = C(pref) * E[(i * (lbp + 1) + j) * nt + 0];
      }
      Real am[3], an[3];
      avec(sa.center, am);
      avec(sb.center, an);
      const Real al = sa.alpha, be = sb.alpha;
      auto sd = [&](int d, int i, int j) { return s[d][i * (lbp + 1) + j]; };
      for (int ka = 0; ka < ncart(la); ++ka) {
        int a3[3];
        cart_comp(la, ka, a3[0], a3[1], a3[2]);
        for (int kb = 0; kb < ncart(lb); ++kb) {
          int b3[3];
          cart_comp(lb, kb, b3[0], b3[1], b3[2]);
          // 1D derivative-overlap-derivative factor in direction c
          auto facc = [&](int c) {
            C bc[3];
            int bi[3];
            int nb = 0;
            bc[nb] = C(Real(0), -am[c]); bi[nb] = a3[c]; ++nb;      // conj phase deriv
            bc[nb] = C(-2 * al);        bi[nb] = a3[c] + 1; ++nb;  // -2a (x-A)^{+1}
            if (a3[c] >= 1) { bc[nb] = C(Real(a3[c])); bi[nb] = a3[c] - 1; ++nb; }
            C kc[3];
            int kj[3];
            int nk = 0;
            kc[nk] = C(Real(0), an[c]); kj[nk] = b3[c]; ++nk;
            kc[nk] = C(-2 * be);        kj[nk] = b3[c] + 1; ++nk;
            if (b3[c] >= 1) { kc[nk] = C(Real(b3[c])); kj[nk] = b3[c] - 1; ++nk; }
            C f(0);
            for (int p = 0; p < nb; ++p)
              for (int q = 0; q < nk; ++q) f += bc[p] * kc[q] * sd(c, bi[p], kj[q]);
            return f;
          };
          const C bx = sd(0, a3[0], b3[0]), by = sd(1, a3[1], b3[1]),
                  bz = sd(2, a3[2], b3[2]);
          T[(basis.ao_off[a] + ka) * static_cast<std::size_t>(nao) + basis.ao_off[b] + kb] =
              C(Real(0.5)) * (facc(0) * by * bz + bx * facc(1) * bz + bx * by * facc(2));
        }
      }
    }
  return T;
}

/// Analytic magnetic-field derivative dT/dB_k of the GIAO kinetic matrix at
/// B = 0, gauge origin O. Because -1/2 nabla^2 also acts on the London phase,
///   dT_munu/dB_k = (i/2) [e_k x (R_mu - R_nu)] . <mu| r (-1/2 nabla^2) |nu>
///                + (i/2) [e_k x (R_nu - O)]   . <mu| grad |nu>,
/// the first term the phase-weighted kinetic (kinetic_moment_matrices), the
/// second from the operator differentiating the ket phase (gradient_matrices).
/// Returns {dT/dB_x, dT/dB_y, dT/dB_z}.
template <class Real>
std::array<std::vector<std::complex<Real>>, 3>
giao_kinetic_dB(const ShellBasis<Real> &basis, const Real O[3]) {
  using C = std::complex<Real>;
  static_assert(!is_complex_v<Real>, "giao_kinetic_dB takes a real basis");
  const int nao = basis.nao;
  const auto KM = kinetic_moment_matrices(basis); // <mu| x_c (-1/2 lap) |nu>
  const auto G = gradient_matrices(basis);        // <mu| d/dr_c |nu>
  std::array<std::vector<C>, 3> dT;
  for (auto &m : dT) m.assign(static_cast<std::size_t>(nao) * nao, C(0));
  const int ns = static_cast<int>(basis.shells.size());
  const C hi(Real(0), Real(0.5));
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const Real u[3] = {sa.center[0] - sb.center[0], sa.center[1] - sb.center[1],
                         sa.center[2] - sb.center[2]};                        // R_mu - R_nu
      const Real v[3] = {sb.center[0] - O[0], sb.center[1] - O[1], sb.center[2] - O[2]}; // R_nu - O
      for (int ka = 0; ka < ncart(sa.l); ++ka)
        for (int kb = 0; kb < ncart(sb.l); ++kb) {
          const std::size_t idx =
              (basis.ao_off[a] + ka) * static_cast<std::size_t>(nao) + basis.ao_off[b] + kb;
          const Real Kx = KM[0][idx], Ky = KM[1][idx], Kz = KM[2][idx];
          const Real Gx = G[0][idx], Gy = G[1][idx], Gz = G[2][idx];
          dT[0][idx] = hi * ((-u[2] * Ky + u[1] * Kz) + (-v[2] * Gy + v[1] * Gz));
          dT[1][idx] = hi * ((u[2] * Kx - u[0] * Kz) + (v[2] * Gx - v[0] * Gz));
          dT[2][idx] = hi * ((-u[1] * Kx + u[0] * Ky) + (-v[1] * Gx + v[0] * Gy));
        }
    }
  return dT;
}

/// Complex GIAO nuclear-attraction matrix V(B) = <omega_mu| sum_C w_C/|r-R_C|
/// |omega_nu> in a finite field B. The London phase shifts the product centre
/// into the complex plane (make_giao_pair), so the ordinary MD + t-quadrature
/// nuclear machinery runs unchanged on complex scalars. Reduces to the real
/// nuclear_matrix at B = 0. Used to finite-difference giao_nuclear_dB.
template <class Real>
std::vector<std::complex<Real>>
giao_nuclear(const ShellBasis<Real> &basis,
             const std::vector<PointCharge<Real>> &charges,
             const TGrid<Real> &grid, const Real B[3]) {
  using C = std::complex<Real>;
  static_assert(!is_complex_v<Real>, "giao_nuclear takes a real basis");
  if constexpr (kokkos_scalar_v<Kokkos::complex<Real>>) {
    bool ok = true;
    for (const auto &s : basis.shells)
      if (s.l > LMAX) ok = false; // B stack arrays are 2*LMAX+1
    if (ok) return detail::giao_nuclear_dev(basis, charges, B, grid);
  }
  const int nao = basis.nao, nt = grid.n();
  const Real pi = pi_v<Real>();
  std::vector<C> V(static_cast<std::size_t>(nao) * nao, C(0));
  const int ns = static_cast<int>(basis.shells.size());
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const auto sp = make_giao_pair(sa, sb, B);
      const int la = sa.l, lb = sb.l, esz = (la + 1) * (lb + 1) * (la + lb + 1);
      const Real p = sp.p;
      std::vector<C> E(static_cast<std::size_t>(3) * esz);
      for (int d = 0; d < 3; ++d)
        e_coeffs(la, lb, p, sp.P[d] - C(sa.center[d]), sp.P[d] - C(sb.center[d]),
                 sp.K[d], E.data() + d * esz);
      const int nca = ncart(la), ncb = ncart(lb), ntab = la + lb + 1;
      std::vector<C> acc(static_cast<std::size_t>(nca) * ncb, C(0));
      std::vector<C> Bx(ntab), By(ntab), Bz(ntab);
      for (const auto &c : charges) {
        for (int it = 0; it < nt; ++it) {
          const Real t = grid.t[it], denom = p + t * t;
          const Real theta = p * t * t / denom, pref = sqrt_(pi / denom);
          hermite_b(la + lb, theta, sp.P[0] - C(c.R[0]), Bx.data());
          hermite_b(la + lb, theta, sp.P[1] - C(c.R[1]), By.data());
          hermite_b(la + lb, theta, sp.P[2] - C(c.R[2]), Bz.data());
          const Real wt = grid.w[it] * c.weight;
          const C *Bd[3] = {Bx.data(), By.data(), Bz.data()};
          for (int ka = 0; ka < nca; ++ka) {
            int a3[3];
            cart_comp(la, ka, a3[0], a3[1], a3[2]);
            for (int kb = 0; kb < ncb; ++kb) {
              int b3[3];
              cart_comp(lb, kb, b3[0], b3[1], b3[2]);
              auto gd = [&](int d) {
                const C *Ed = E.data() + d * esz + (a3[d] * (lb + 1) + b3[d]) * ntab;
                C s(0);
                for (int tt = 0; tt <= a3[d] + b3[d]; ++tt) s += Ed[tt] * Bd[d][tt];
                return C(pref) * s;
              };
              acc[ka * ncb + kb] += C(wt) * gd(0) * gd(1) * gd(2);
            }
          }
        }
      }
      for (int ka = 0; ka < nca; ++ka)
        for (int kb = 0; kb < ncb; ++kb)
          V[(basis.ao_off[a] + ka) * static_cast<std::size_t>(nao) + basis.ao_off[b] + kb] =
              acc[ka * ncb + kb];
    }
  return V;
}

/// Analytic magnetic-field derivative dV/dB_k of the GIAO nuclear-attraction
/// matrix at B = 0. The nuclear potential is a multiplicative operator, so the
/// London phase is differentiated exactly as for the overlap:
///   dV_munu/dB_k = (i/2) [e_k x (R_mu - R_nu)] . <mu| r V |nu>,
/// with <mu| r V |nu> the position-weighted nuclear attraction
/// (nuclear_moment_matrices). Returns {dV/dB_x, dV/dB_y, dV/dB_z}. `charges`
/// carry the operator weight (nuclei_as_charges gives -Z, matching int1e_nuc).
template <class Real>
std::array<std::vector<std::complex<Real>>, 3>
giao_nuclear_dB(const ShellBasis<Real> &basis,
                const std::vector<PointCharge<Real>> &charges,
                const TGrid<Real> &grid, Real tau = Real(0)) {
  using C = std::complex<Real>;
  static_assert(!is_complex_v<Real>, "giao_nuclear_dB takes a real basis");
  const int nao = basis.nao;
  const auto M = nuclear_moment_matrices(basis, charges, grid, tau); // <mu|x_c V|nu>
  std::array<std::vector<C>, 3> dV;
  for (auto &m : dV) m.assign(static_cast<std::size_t>(nao) * nao, C(0));
  const int ns = static_cast<int>(basis.shells.size());
  const C half_i(Real(0), Real(0.5));
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const Real dv[3] = {sa.center[0] - sb.center[0], sa.center[1] - sb.center[1],
                          sa.center[2] - sb.center[2]};
      for (int ka = 0; ka < ncart(sa.l); ++ka)
        for (int kb = 0; kb < ncart(sb.l); ++kb) {
          const std::size_t idx =
              (basis.ao_off[a] + ka) * static_cast<std::size_t>(nao) + basis.ao_off[b] + kb;
          const Real Mx = M[0][idx], My = M[1][idx], Mz = M[2][idx];
          dV[0][idx] = half_i * (-dv[2] * My + dv[1] * Mz);
          dV[1][idx] = half_i * (dv[2] * Mx - dv[0] * Mz);
          dV[2][idx] = half_i * (-dv[1] * Mx + dv[0] * My);
        }
    }
  return dV;
}

} // namespace intti
