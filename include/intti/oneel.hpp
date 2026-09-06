// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// One-electron property integral matrices in the McMurchie-Davidson basis:
// overlap, kinetic energy, and Cartesian multipole moments. These are all
// "moment" integrals -- Gaussian products times a polynomial (x-C)^e, with no
// interaction kernel -- so they come straight off the E coefficients
// (hermite1d.hpp) with no t quadrature.
//
// The library's native API is matrix-level (a whole nao x nao AO matrix per
// call), never individual shell blocks: overlap_matrix(basis), etc. Matrices
// are built over the primitive Cartesian AOs of a ShellBasis, row-major,
// unnormalized primitives -- the caller (or the libcint facade) applies the
// normalization/contraction, exactly as for the ERIs.

#include <array>
#include <cstddef>
#include <vector>

#include "batch.hpp"  // PairTable, make_pair_table (E precomputed on device)
#include "device.hpp" // detail::to_device / to_host
#include "fock.hpp"
#include "gto.hpp"
#include "hermite1d.hpp"
#include "lkc.hpp"
#include "math.hpp"

namespace intti {

namespace detail {

/// Gaussian pair prefactor exp(-mu |R_a - R_b|^2), mu = alpha beta / (alpha +
/// beta): the common factor of every element of the shell-pair block. When it
/// underflows to 0 (distant or very tight pairs) the whole block is exactly 0,
/// so a `pref <= tau` test screens pairs (tau = 0 skips only the exactly-zero
/// blocks -- exact; tau > 0 is an approximate distance screen).
template <class Real>
Real pair_gauss_prefactor(const PrimitiveShell<Real> &sa, const PrimitiveShell<Real> &sb) {
  const Real p = sa.alpha + sb.alpha, mu = sa.alpha * sb.alpha / p;
  Real r2 = 0;
  for (int d = 0; d < 3; ++d) {
    const Real ab = sa.center[d] - sb.center[d];
    r2 += ab * ab;
  }
  return exp_(-mu * r2);
}

/// Per-direction 1D overlap table for a primitive shell pair:
///   s1[i*(lbx+1)+j] = int (x-A)^i e^{-alpha (x-A)^2} (x-B)^j e^{-beta (x-B)^2} dx,
/// for i = 0..la+exa, j = 0..lb+exb. Built from the MD E coefficients
/// (E_0^{ij} sqrt(pi/p)). exa/exb request extra angular momentum so kinetic
/// (ket +2) and multipoles (bra +order) can raise indices.
template <class Real>
void overlap_1d(Real alpha, Real A, Real beta, Real B, int la, int lb, int exa,
                int exb, std::vector<Real> &s1, int &lbx) {
  const int lax = la + exa;
  lbx = lb + exb;
  const Real p = alpha + beta;
  const Real P = (alpha * A + beta * B) / p;
  const Real mu = alpha * beta / p;
  const Real ab = A - B;
  const Real K = exp_(-mu * ab * ab);
  std::vector<Real> E(static_cast<std::size_t>(lax + 1) * (lbx + 1) * (lax + lbx + 1));
  e_coeffs(lax, lbx, p, P - A, P - B, K, E.data());
  const Real pref = sqrt_(pi_v<Real>() / p);
  s1.assign(static_cast<std::size_t>(lax + 1) * (lbx + 1), Real(0));
  const int nt = lax + lbx + 1;
  for (int i = 0; i <= lax; ++i)
    for (int j = 0; j <= lbx; ++j)
      s1[i * (lbx + 1) + j] = pref * E[(i * (lbx + 1) + j) * nt + 0];
}

/// 1D kinetic table T[i*(lbx+1)+j] from the overlap table and the ket
/// exponent beta: T_{ij} = -2 b^2 S_{i,j+2} + b(2j+1) S_{i,j} - j(j-1)/2 S_{i,j-2}.
template <class Real>
void kinetic_1d(const std::vector<Real> &s1, int lbx, int la, int lb, Real beta,
                std::vector<Real> &t1) {
  t1.assign(static_cast<std::size_t>(la + 1) * (lb + 1), Real(0));
  auto S = [&](int i, int j) { return s1[i * (lbx + 1) + j]; };
  for (int i = 0; i <= la; ++i)
    for (int j = 0; j <= lb; ++j) {
      Real v = -2 * beta * beta * S(i, j + 2) + beta * (2 * j + 1) * S(i, j);
      if (j >= 2) v -= Real(0.5) * j * (j - 1) * S(i, j - 2);
      t1[i * (lb + 1) + j] = v;
    }
}

/// 1D multipole table m1[(e*(la+1)+i)*(lb+1)+j] = <i|(x-O)^e|j>, e = 0..emax,
/// from (x-O)^e = sum_k C(e,k)(x-A)^k (A-O)^{e-k} and <i|(x-A)^k|j> = S_{i+k,j}.
template <class Real>
void multipole_1d(const std::vector<Real> &s1, int lbx, int la, int lb, Real A,
                  Real O, int emax, std::vector<Real> &m1) {
  m1.assign(static_cast<std::size_t>(emax + 1) * (la + 1) * (lb + 1), Real(0));
  auto S = [&](int i, int j) { return s1[i * (lbx + 1) + j]; };
  const Real dAO = A - O;
  for (int e = 0; e <= emax; ++e) {
    Real binom = 1; // C(e,k)
    for (int k = 0; k <= e; ++k) {
      Real pw = 1; // (A-O)^{e-k}
      for (int r = 0; r < e - k; ++r)
        pw *= dAO;
      const Real c = binom * pw;
      for (int i = 0; i <= la; ++i)
        for (int j = 0; j <= lb; ++j)
          m1[(static_cast<std::size_t>(e) * (la + 1) + i) * (lb + 1) + j] +=
              c * S(i + k, j);
      binom = binom * (e - k) / (k + 1);
    }
  }
}

// ---- device (GPU) 1e path ---------------------------------------------------
// Same architecture as the J/K builds: e_coeffs are precomputed on the host into
// the PairTable's device E View (make_pair_table), and the Cartesian assembly +
// scatter run in a Kokkos parallel_for over shell pairs. Extending each pair's
// angular momentum by (exa,exb) makes the on-device E carry the higher indices
// kinetic (+2 on the ket) and multipoles (+order on the bra) need. The public
// builders dispatch here for float/double/long double and keep the serial host
// loop for __float128 / class-type scalars (which Kokkos cannot run).

template <class Real> struct OneEPairs {
  PairTable<Real> tab;          ///< E for (la+exa, lb+exb)
  Kokkos::View<int *> aoa, aob; ///< AO offsets of the canonical pair's shells
  Kokkos::View<int *> la0, lb0; ///< original momenta (output loop bounds)
  Kokkos::View<Real *> beta;    ///< ket exponent (kinetic recurrence)
  int npair{0};
};

/// Canonical (a<=b) shell pairs above the screening threshold, each extended by
/// (exa,exb), with the side arrays the device 1e kernels need.
template <class Real>
OneEPairs<Real> make_1e_pairs(const ShellBasis<Real> &basis, int exa, int exb, Real tau) {
  const int ns = static_cast<int>(basis.shells.size());
  std::vector<ShellPair<Real>> plist;
  std::vector<int> haoa, haob, hla0, hlb0;
  std::vector<Real> hbeta;
  for (int a = 0; a < ns; ++a)
    for (int b = a; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      if (pair_gauss_prefactor(sa, sb) <= tau) continue; // exact at tau = 0
      PrimitiveShell<Real> sae = sa, sbe = sb;
      sae.l += exa;
      sbe.l += exb;
      plist.push_back(make_pair(sae, sbe));
      haoa.push_back(basis.ao_off[a]);
      haob.push_back(basis.ao_off[b]);
      hla0.push_back(sa.l);
      hlb0.push_back(sb.l);
      hbeta.push_back(sb.alpha);
    }
  OneEPairs<Real> op;
  op.tab = make_pair_table(plist);
  op.npair = op.tab.npair;
  op.aoa = to_device(haoa, "intti::1e::aoa");
  op.aob = to_device(haob, "intti::1e::aob");
  op.la0 = to_device(hla0, "intti::1e::la0");
  op.lb0 = to_device(hlb0, "intti::1e::lb0");
  op.beta = to_device(hbeta, "intti::1e::beta");
  return op;
}

template <class Real>
std::vector<Real> overlap_matrix_dev(const ShellBasis<Real> &basis, Real tau) {
  const int nao = basis.nao;
  auto op = make_1e_pairs(basis, 0, 0, tau);
  const int npair = op.npair;
  Kokkos::View<Real *> Sd("intti::ovlp::S", static_cast<std::size_t>(nao) * nao);
  auto pv = op.tab.p, Ev = op.tab.E;
  auto lav = op.tab.la, lbv = op.tab.lb, eoffv = op.tab.e_off, aoa = op.aoa, aob = op.aob;
  const Real pi = pi_v<Real>();
  Kokkos::parallel_for(
      "intti::ovlp::asm", Kokkos::RangePolicy<>(0, npair), KOKKOS_LAMBDA(int p) {
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
            const Real ex = Ev(eo + 0 * esz + (a3[0] * (lb + 1) + b3[0]) * n1);
            const Real ey = Ev(eo + 1 * esz + (a3[1] * (lb + 1) + b3[1]) * n1);
            const Real ez = Ev(eo + 2 * esz + (a3[2] * (lb + 1) + b3[2]) * n1);
            const Real val = pref3 * ex * ey * ez;
            const int r = oa + ka, c = ob + kb;
            Sd(static_cast<std::size_t>(r) * nao + c) = val;
            if (r != c) Sd(static_cast<std::size_t>(c) * nao + r) = val;
          }
        }
      });
  return to_host(Sd);
}

template <class Real>
std::vector<Real> kinetic_matrix_dev(const ShellBasis<Real> &basis, Real tau) {
  const int nao = basis.nao;
  auto op = make_1e_pairs(basis, 0, 2, tau); // ket +2 for the T recurrence
  const int npair = op.npair;
  Kokkos::View<Real *> Td("intti::kin::T", static_cast<std::size_t>(nao) * nao);
  auto pv = op.tab.p, Ev = op.tab.E, betav = op.beta;
  auto lav = op.tab.la, lbv = op.tab.lb, eoffv = op.tab.e_off, aoa = op.aoa, aob = op.aob;
  const Real pi = pi_v<Real>();
  Kokkos::parallel_for(
      "intti::kin::asm", Kokkos::RangePolicy<>(0, npair), KOKKOS_LAMBDA(int p) {
        const int la = lav(p), lbx = lbv(p), lb0 = lbx - 2, n1 = la + lbx + 1;
        const int esz = (la + 1) * (lbx + 1) * n1;
        const Real pref = sqrt_(pi / pv(p)), bta = betav(p);
        const int eo = eoffv(p), oa = aoa(p), ob = aob(p);
        auto S1 = [&](int d, int i, int j) {
          return pref * Ev(eo + d * esz + (i * (lbx + 1) + j) * n1);
        };
        auto T1 = [&](int d, int i, int j) {
          Real v = -2 * bta * bta * S1(d, i, j + 2) + bta * (2 * j + 1) * S1(d, i, j);
          if (j >= 2) v -= Real(0.5) * j * (j - 1) * S1(d, i, j - 2);
          return v;
        };
        for (int ka = 0; ka < ncart(la); ++ka) {
          int a3[3];
          cart_comp(la, ka, a3[0], a3[1], a3[2]);
          for (int kb = 0; kb < ncart(lb0); ++kb) {
            int b3[3];
            cart_comp(lb0, kb, b3[0], b3[1], b3[2]);
            const Real sx = S1(0, a3[0], b3[0]), sy = S1(1, a3[1], b3[1]), sz = S1(2, a3[2], b3[2]);
            const Real val = T1(0, a3[0], b3[0]) * sy * sz + sx * T1(1, a3[1], b3[1]) * sz +
                             sx * sy * T1(2, a3[2], b3[2]);
            const int r = oa + ka, c = ob + kb;
            Td(static_cast<std::size_t>(r) * nao + c) = val;
            if (r != c) Td(static_cast<std::size_t>(c) * nao + r) = val;
          }
        }
      });
  return to_host(Td);
}

} // namespace detail

/// Overlap matrix S (nao x nao, row-major) over the primitive Cartesian AOs.
template <class Real>
std::vector<Real> overlap_matrix(const ShellBasis<Real> &basis, Real tau = Real(0)) {
  if constexpr (kokkos_scalar_v<Real>)
    return detail::overlap_matrix_dev(basis, tau);
  const int nao = basis.nao;
  std::vector<Real> S(static_cast<std::size_t>(nao) * nao, Real(0));
  const int ns = static_cast<int>(basis.shells.size());
  // S is symmetric: compute the upper triangle a <= b and mirror.
  for (int a = 0; a < ns; ++a)
    for (int b = a; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      if (detail::pair_gauss_prefactor(sa, sb) <= tau) continue; // exact at tau=0
      std::vector<Real> sx, sy, sz;
      int lbx;
      detail::overlap_1d(sa.alpha, sa.center[0], sb.alpha, sb.center[0], sa.l, sb.l, 0, 0, sx, lbx);
      detail::overlap_1d(sa.alpha, sa.center[1], sb.alpha, sb.center[1], sa.l, sb.l, 0, 0, sy, lbx);
      detail::overlap_1d(sa.alpha, sa.center[2], sb.alpha, sb.center[2], sa.l, sb.l, 0, 0, sz, lbx);
      detail::scatter_pair(S, basis, a, b, +1,
          [&](int, const int *a3, int, const int *b3) {
            return sx[a3[0] * (lbx + 1) + b3[0]] * sy[a3[1] * (lbx + 1) + b3[1]] *
                   sz[a3[2] * (lbx + 1) + b3[2]];
          });
    }
  return S;
}

/// Kinetic energy matrix T = -1/2 <a| nabla^2 |b>.
template <class Real>
std::vector<Real> kinetic_matrix(const ShellBasis<Real> &basis, Real tau = Real(0)) {
  if constexpr (kokkos_scalar_v<Real>)
    return detail::kinetic_matrix_dev(basis, tau);
  const int nao = basis.nao;
  std::vector<Real> T(static_cast<std::size_t>(nao) * nao, Real(0));
  const int ns = static_cast<int>(basis.shells.size());
  // T is symmetric: compute the upper triangle a <= b and mirror.
  for (int a = 0; a < ns; ++a)
    for (int b = a; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      if (detail::pair_gauss_prefactor(sa, sb) <= tau) continue; // exact at tau=0
      std::vector<Real> sx, sy, sz, tx, ty, tz;
      int lbx;
      detail::overlap_1d(sa.alpha, sa.center[0], sb.alpha, sb.center[0], sa.l, sb.l, 0, 2, sx, lbx);
      detail::kinetic_1d(sx, lbx, sa.l, sb.l, sb.alpha, tx);
      detail::overlap_1d(sa.alpha, sa.center[1], sb.alpha, sb.center[1], sa.l, sb.l, 0, 2, sy, lbx);
      detail::kinetic_1d(sy, lbx, sa.l, sb.l, sb.alpha, ty);
      detail::overlap_1d(sa.alpha, sa.center[2], sb.alpha, sb.center[2], sa.l, sb.l, 0, 2, sz, lbx);
      detail::kinetic_1d(sz, lbx, sa.l, sb.l, sb.alpha, tz);
      const int lb1 = sb.l + 1;
      auto Sx = [&](int i, int j) { return sx[i * (lbx + 1) + j]; };
      auto Sy = [&](int i, int j) { return sy[i * (lbx + 1) + j]; };
      auto Sz = [&](int i, int j) { return sz[i * (lbx + 1) + j]; };
      detail::scatter_pair(T, basis, a, b, +1,
          [&](int, const int *a3, int, const int *b3) {
            return tx[a3[0] * lb1 + b3[0]] * Sy(a3[1], b3[1]) * Sz(a3[2], b3[2]) +
                   Sx(a3[0], b3[0]) * ty[a3[1] * lb1 + b3[1]] * Sz(a3[2], b3[2]) +
                   Sx(a3[0], b3[0]) * Sy(a3[1], b3[1]) * tz[a3[2] * lb1 + b3[2]];
          });
    }
  return T;
}

/// Cartesian multipole matrices <a|(x-O)^ex (y-O)^ey (z-O)^ez|b> for every
/// (ex,ey,ez) with ex+ey+ez <= max_order, returned as
/// out[component] where component runs in cart_comp order per total order,
/// concatenated order 0,1,2,...; see multipole_labels() for the mapping.
/// Order 0 is the overlap, order 1 the dipole, order 2 the quadrupole, ...
template <class Real>
std::vector<std::vector<Real>> multipole_matrices(const ShellBasis<Real> &basis,
                                                  int max_order,
                                                  const Real origin[3],
                                                  Real tau = Real(0)) {
  const int nao = basis.nao;
  // component list: (ex,ey,ez), total order 0..max_order, cart_comp ordering
  std::vector<std::array<int, 3>> comps;
  for (int L = 0; L <= max_order; ++L)
    for (int k = 0; k < ncart(L); ++k) {
      int e[3];
      cart_comp(L, k, e[0], e[1], e[2]);
      comps.push_back({e[0], e[1], e[2]});
    }
  std::vector<std::vector<Real>> out(comps.size(),
                                     std::vector<Real>(static_cast<std::size_t>(nao) * nao, Real(0)));
  const int ns = static_cast<int>(basis.shells.size());
  // each multipole matrix is symmetric (multiplicative operator): a <= b + mirror.
  for (int a = 0; a < ns; ++a)
    for (int b = a; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      if (detail::pair_gauss_prefactor(sa, sb) <= tau) continue; // exact at tau=0
      std::vector<Real> s[3], m[3];
      int lbx;
      for (int d = 0; d < 3; ++d) {
        detail::overlap_1d(sa.alpha, sa.center[d], sb.alpha, sb.center[d], sa.l, sb.l,
                           max_order, 0, s[d], lbx);
        detail::multipole_1d(s[d], lbx, sa.l, sb.l, sa.center[d], origin[d], max_order, m[d]);
      }
      const int lb1 = sb.l + 1;
      auto M = [&](int d, int e, int i, int j) {
        return m[d][(static_cast<std::size_t>(e) * (sa.l + 1) + i) * lb1 + j];
      };
      for (std::size_t ci = 0; ci < comps.size(); ++ci) {
        const auto &e = comps[ci];
        detail::scatter_pair(out[ci], basis, a, b, +1,
            [&](int, const int *a3, int, const int *b3) {
              return M(0, e[0], a3[0], b3[0]) * M(1, e[1], a3[1], b3[1]) *
                     M(2, e[2], a3[2], b3[2]);
            });
      }
    }
  return out;
}

/// (ex,ey,ez) labels matching the component order of multipole_matrices().
inline std::vector<std::array<int, 3>> multipole_labels(int max_order) {
  std::vector<std::array<int, 3>> comps;
  for (int L = 0; L <= max_order; ++L)
    for (int k = 0; k < ncart(L); ++k) {
      int e[3];
      cart_comp(L, k, e[0], e[1], e[2]);
      comps.push_back({e[0], e[1], e[2]});
    }
  return comps;
}

/// Angular-momentum integrals <mu|(r-O) x nabla|nu>, three matrices
/// (Lx, Ly, Lz). r x nabla is anti-Hermitian, so each matrix is real and
/// antisymmetric. Per direction the integrand factors into an overlap S, a
/// moment M_d = <i|(x_d-O_d)|j>, and a ket derivative K_d = <i|d/dx_d|j>:
///   Lx = Sx (My Kz - Ky Mz),  Ly = Sy (Mz Kx - Mx Kz),  Lz = Sz (Mx Ky - My Kx).
/// This is <mu| r x p |nu> up to the -i in p (that factor is the caller's
/// convention; PySCF int1e_cg_irxp carries the i).
template <class Real>
std::array<std::vector<Real>, 3> angular_momentum(const ShellBasis<Real> &basis,
                                                  const Real origin[3]) {
  const int nao = basis.nao;
  std::array<std::vector<Real>, 3> L;
  for (auto &m : L) m.assign(static_cast<std::size_t>(nao) * nao, Real(0));
  const int ns = static_cast<int>(basis.shells.size());
  // L is antisymmetric: compute the upper triangle a <= b and mirror with -1.
  for (int a = 0; a < ns; ++a)
    for (int b = a; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const int la = sa.l, lb = sb.l, lb1 = lb + 1;
      std::vector<Real> Sf[3], Mf[3], Kf[3];
      for (int d = 0; d < 3; ++d) {
        std::vector<Real> s;
        int lbx;
        detail::overlap_1d(sa.alpha, sa.center[d], sb.alpha, sb.center[d], la, lb, 1, 1,
                           s, lbx);
        // overlap S (restricted to la x lb)
        Sf[d].assign(static_cast<std::size_t>(la + 1) * lb1, Real(0));
        Kf[d].assign(static_cast<std::size_t>(la + 1) * lb1, Real(0));
        for (int i = 0; i <= la; ++i)
          for (int j = 0; j <= lb; ++j) {
            Sf[d][i * lb1 + j] = s[i * (lbx + 1) + j];
            // ket derivative d/dx_d : j S[i][j-1] - 2 beta S[i][j+1]
            Real k = -2 * sb.alpha * s[i * (lbx + 1) + (j + 1)];
            if (j >= 1) k += Real(j) * s[i * (lbx + 1) + (j - 1)];
            Kf[d][i * lb1 + j] = k;
          }
        // moment M_d = <i|(x_d - O_d)|j> = S_{i+1,j} + (A_d - O_d) S_{i,j}
        std::vector<Real> m1;
        detail::multipole_1d(s, lbx, la, lb, sa.center[d], origin[d], 1, m1);
        Mf[d].assign(static_cast<std::size_t>(la + 1) * lb1, Real(0));
        for (int i = 0; i <= la; ++i)
          for (int j = 0; j <= lb; ++j)
            Mf[d][i * lb1 + j] = m1[(1 * (la + 1) + i) * lb1 + j]; // e=1 block
      }
      detail::scatter_pair3(L, basis, a, b, -1,
          [&](int, const int *a3, int, const int *b3) -> std::array<Real, 3> {
            auto S = [&](int d) { return Sf[d][a3[d] * lb1 + b3[d]]; };
            auto M = [&](int d) { return Mf[d][a3[d] * lb1 + b3[d]]; };
            auto K = [&](int d) { return Kf[d][a3[d] * lb1 + b3[d]]; };
            return {S(0) * (M(1) * K(2) - K(1) * M(2)),
                    S(1) * (M(2) * K(0) - M(0) * K(2)),
                    S(2) * (M(0) * K(1) - M(1) * K(0))};
          });
    }
  return L;
}

/// Electron-gradient matrices G_c = <mu| d/dr_c |nu> (c = x, y, z), the ket
/// derivative acting to the right. Real; equals -int1e_ipovlp by parts. Built
/// directly from the ket-derivative 1D factor j S[i][j-1] - 2 beta S[i][j+1].
template <class Real>
std::array<std::vector<Real>, 3> gradient_matrices(const ShellBasis<Real> &basis) {
  const int nao = basis.nao;
  std::array<std::vector<Real>, 3> G;
  for (auto &m : G) m.assign(static_cast<std::size_t>(nao) * nao, Real(0));
  const int ns = static_cast<int>(basis.shells.size());
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const int la = sa.l, lb = sb.l, lb1 = lb + 1;
      std::vector<Real> Sf[3], Kf[3];
      for (int d = 0; d < 3; ++d) {
        std::vector<Real> s;
        int lbx;
        detail::overlap_1d(sa.alpha, sa.center[d], sb.alpha, sb.center[d], la, lb, 0, 1,
                           s, lbx);
        Sf[d].assign(static_cast<std::size_t>(la + 1) * lb1, Real(0));
        Kf[d].assign(static_cast<std::size_t>(la + 1) * lb1, Real(0));
        for (int i = 0; i <= la; ++i)
          for (int j = 0; j <= lb; ++j) {
            Sf[d][i * lb1 + j] = s[i * (lbx + 1) + j];
            Real k = -2 * sb.alpha * s[i * (lbx + 1) + (j + 1)];
            if (j >= 1) k += Real(j) * s[i * (lbx + 1) + (j - 1)];
            Kf[d][i * lb1 + j] = k;
          }
      }
      detail::scatter_pair3(G, basis, a, b, 0,
          [&](int, const int *a3, int, const int *b3) -> std::array<Real, 3> {
            auto S = [&](int d) { return Sf[d][a3[d] * lb1 + b3[d]]; };
            auto K = [&](int d) { return Kf[d][a3[d] * lb1 + b3[d]]; };
            return {K(0) * S(1) * S(2), S(0) * K(1) * S(2), S(0) * S(1) * K(2)};
          });
    }
  return G;
}

/// Position-weighted kinetic matrices KM_c = <mu| x_c (-1/2 nabla^2) |nu> about
/// the coordinate origin (c = x, y, z). Built by promoting the bra angular
/// momentum by one unit: x_c = (x_c - A_c) + A_c. The GIAO kinetic
/// field-derivative primitive (the analogue of nuclear_moment_matrices).
template <class Real>
std::array<std::vector<Real>, 3>
kinetic_moment_matrices(const ShellBasis<Real> &basis) {
  const int nao = basis.nao;
  std::array<std::vector<Real>, 3> KM;
  for (auto &m : KM) m.assign(static_cast<std::size_t>(nao) * nao, Real(0));
  const int ns = static_cast<int>(basis.shells.size());
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const int la = sa.l, lb = sb.l, la1 = la + 1, lb1 = lb + 1;
      std::vector<Real> S[3], T[3]; // bra up to la+1, ket base
      for (int d = 0; d < 3; ++d) {
        std::vector<Real> s;
        int lbx;
        detail::overlap_1d(sa.alpha, sa.center[d], sb.alpha, sb.center[d], la, lb, 1, 2,
                           s, lbx);
        S[d].assign(static_cast<std::size_t>(la1 + 1) * lb1, Real(0));
        for (int i = 0; i <= la1; ++i)
          for (int j = 0; j <= lb; ++j) S[d][i * lb1 + j] = s[i * (lbx + 1) + j];
        std::vector<Real> t;
        detail::kinetic_1d(s, lbx, la1, lb, sb.alpha, t); // bra up to la1
        T[d].assign(static_cast<std::size_t>(la1 + 1) * lb1, Real(0));
        for (int i = 0; i <= la1; ++i)
          for (int j = 0; j <= lb; ++j) T[d][i * lb1 + j] = t[i * lb1 + j];
      }
      auto Si = [&](int d, int i, int j) { return S[d][i * lb1 + j]; };
      auto Ti = [&](int d, int i, int j) { return T[d][i * lb1 + j]; };
      detail::scatter_pair3(KM, basis, a, b, 0,
          [&](int, const int *a3, int, const int *b3) -> std::array<Real, 3> {
            auto kin = [&](int ix, int iy, int iz) {
              return Ti(0, ix, b3[0]) * Si(1, iy, b3[1]) * Si(2, iz, b3[2]) +
                     Si(0, ix, b3[0]) * Ti(1, iy, b3[1]) * Si(2, iz, b3[2]) +
                     Si(0, ix, b3[0]) * Si(1, iy, b3[1]) * Ti(2, iz, b3[2]);
            };
            const Real Tb = kin(a3[0], a3[1], a3[2]);
            return {kin(a3[0] + 1, a3[1], a3[2]) + sa.center[0] * Tb,
                    kin(a3[0], a3[1] + 1, a3[2]) + sa.center[1] * Tb,
                    kin(a3[0], a3[1], a3[2] + 1) + sa.center[2] * Tb};
          });
    }
  return KM;
}

/// Trace(D^T M) -- contract a density (or any nao x nao matrix) with a
/// property matrix to get the property value.
template <class Real>
Real property_value(const std::vector<Real> &D, const std::vector<Real> &M) {
  Real s{};
  for (std::size_t i = 0; i < D.size(); ++i)
    s += D[i] * M[i];
  return s;
}

} // namespace intti
