// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Derivative integrals: the black-box enabler (analytic forces, Hessians,
// magnetic response). The McMurchie-Davidson centre-shift relation makes
// every derivative a linear combination of shifted-angular-momentum
// integrals from the existing engine. Differentiating an AO w.r.t. the
// electron coordinate raises/lowers its Cartesian power:
//   d/dx [ (x-A)^l e^{-a(x-A)^2} ] = l (x-A)^{l-1} e^{...} - 2a (x-A)^{l+1} e^{...}
// so <nabla_x mu|nu> = l_x S(mu[l_x-1], nu) - 2 a_mu S(mu[l_x+1], nu), etc.
// These are the libcint int1e_ip* integrals (gradient on the bra).
//
// Matrix-level API: whole nao x nao gradient-component matrices.

#include <array>
#include <cstddef>
#include <vector>

#include "fock.hpp"
#include "gto.hpp"
#include "hermite1d.hpp"
#include "math.hpp"
#include "nuclear.hpp" // PointCharge
#include "oneel.hpp"   // detail::overlap_1d, kinetic_1d
#include "tgrid.hpp"

namespace intti {

namespace detail {
/// Shift operator D[F](i,j) = i*F(i-1,j) - 2 alpha * F(i+1,j) on the bra
/// index i, i.e. d/dx acting on the bra 1D factor. F is a flat (la+2)x(lbx+1)
/// table (bra extended by 1). Returns the derivative 1D table (la+1)x(lb+1).
template <class Real>
void bra_shift(const std::vector<Real> &F, int lbx, int la, int lb, Real alpha,
               std::vector<Real> &dF) {
  dF.assign(static_cast<std::size_t>(la + 1) * (lb + 1), Real(0));
  auto at = [&](int i, int j) { return F[i * (lbx + 1) + j]; };
  for (int i = 0; i <= la; ++i)
    for (int j = 0; j <= lb; ++j) {
      Real v = -2 * alpha * at(i + 1, j);
      if (i >= 1) v += Real(i) * at(i - 1, j);
      dF[i * (lb + 1) + j] = v;
    }
}

/// Per-primitive-pair derivative blocks, component-major
/// out[(d*nca+ka)*ncb+kb] with d the Cartesian direction of the bra gradient.
/// These are the single source of truth for the shift algebra: the primitive
/// host builders below loop them over shell pairs, and the contraction-aware
/// builders (contracted.hpp) loop them over PRIMITIVE pairs and accumulate
/// coefficient-weighted into the contracted block, so a generally-contracted
/// derivative costs one primitive-pair block evaluation, not one per contracted
/// function pair. Each block OVERWRITES its output.

/// <nabla mu | nu>, PySCF int1e_ipovlp.
template <class Real>
void overlap_deriv_block(const PrimitiveShell<Real> &sa, const PrimitiveShell<Real> &sb,
                         Real *out) {
  const int la = sa.l, lb = sb.l, lb1 = lb + 1;
  const int nca = ncart(la), ncb = ncart(lb);
  const std::size_t pstride = static_cast<std::size_t>(nca) * ncb;
  std::vector<Real> s[3], ds[3];
  int lbx;
  for (int d = 0; d < 3; ++d) {
    overlap_1d(sa.alpha, sa.center[d], sb.alpha, sb.center[d], la, lb, 1, 0, s[d], lbx);
    bra_shift(s[d], lbx, la, lb, sa.alpha, ds[d]);
  }
  auto S = [&](int d, int i, int j) { return s[d][i * (lbx + 1) + j]; };
  auto DS = [&](int d, int i, int j) { return ds[d][i * lb1 + j]; };
  for (int ka = 0; ka < nca; ++ka) {
    int a3[3];
    cart_comp(la, ka, a3[0], a3[1], a3[2]);
    for (int kb = 0; kb < ncb; ++kb) {
      int b3[3];
      cart_comp(lb, kb, b3[0], b3[1], b3[2]);
      const std::size_t k = static_cast<std::size_t>(ka) * ncb + kb;
      out[0 * pstride + k] = DS(0, a3[0], b3[0]) * S(1, a3[1], b3[1]) * S(2, a3[2], b3[2]);
      out[1 * pstride + k] = S(0, a3[0], b3[0]) * DS(1, a3[1], b3[1]) * S(2, a3[2], b3[2]);
      out[2 * pstride + k] = S(0, a3[0], b3[0]) * S(1, a3[1], b3[1]) * DS(2, a3[2], b3[2]);
    }
  }
}

/// <nabla mu | T | nu>, PySCF int1e_ipkin.
template <class Real>
void kinetic_deriv_block(const PrimitiveShell<Real> &sa, const PrimitiveShell<Real> &sb,
                         Real *out) {
  const int la = sa.l, lb = sb.l, lb1 = lb + 1;
  const int nca = ncart(la), ncb = ncart(lb);
  const std::size_t pstride = static_cast<std::size_t>(nca) * ncb;
  std::vector<Real> s[3], t[3], ds[3], dt[3];
  int lbx;
  for (int d = 0; d < 3; ++d) {
    // overlap up to bra la+1, ket lb+2 (kinetic needs ket +2)
    overlap_1d(sa.alpha, sa.center[d], sb.alpha, sb.center[d], la, lb, 1, 2, s[d], lbx);
    kinetic_1d(s[d], lbx, la + 1, lb, sb.alpha, t[d]); // t: (la+2)x(lb+1)
    bra_shift(s[d], lbx, la, lb, sa.alpha, ds[d]);     // dS: (la+1)x(lb+1)
    // dT from t (bra index up to la+1): the same shift with stride lb1
    dt[d].assign(static_cast<std::size_t>(la + 1) * (lb + 1), Real(0));
    auto T = [&](int i, int j) { return t[d][i * lb1 + j]; };
    for (int i = 0; i <= la; ++i)
      for (int j = 0; j <= lb; ++j) {
        Real v = -2 * sa.alpha * T(i + 1, j);
        if (i >= 1) v += Real(i) * T(i - 1, j);
        dt[d][i * lb1 + j] = v;
      }
  }
  auto S = [&](int d, int i, int j) { return s[d][i * (lbx + 1) + j]; };
  auto Tk = [&](int d, int i, int j) { return t[d][i * lb1 + j]; };
  auto DS = [&](int d, int i, int j) { return ds[d][i * lb1 + j]; };
  auto DT = [&](int d, int i, int j) { return dt[d][i * lb1 + j]; };
  for (int ka = 0; ka < nca; ++ka) {
    int a3[3];
    cart_comp(la, ka, a3[0], a3[1], a3[2]);
    for (int kb = 0; kb < ncb; ++kb) {
      int b3[3];
      cart_comp(lb, kb, b3[0], b3[1], b3[2]);
      const int ax = a3[0], ay = a3[1], az = a3[2];
      const int bx = b3[0], by = b3[1], bz = b3[2];
      const std::size_t k = static_cast<std::size_t>(ka) * ncb + kb;
      // T = Tx Sy Sz + Sx Ty Sz + Sx Sy Tz; apply nabla to the bra per dir
      out[0 * pstride + k] = DT(0, ax, bx) * S(1, ay, by) * S(2, az, bz) +
                             DS(0, ax, bx) * Tk(1, ay, by) * S(2, az, bz) +
                             DS(0, ax, bx) * S(1, ay, by) * Tk(2, az, bz);
      out[1 * pstride + k] = Tk(0, ax, bx) * DS(1, ay, by) * S(2, az, bz) +
                             S(0, ax, bx) * DT(1, ay, by) * S(2, az, bz) +
                             S(0, ax, bx) * DS(1, ay, by) * Tk(2, az, bz);
      out[2 * pstride + k] = Tk(0, ax, bx) * S(1, ay, by) * DS(2, az, bz) +
                             S(0, ax, bx) * Tk(1, ay, by) * DS(2, az, bz) +
                             S(0, ax, bx) * S(1, ay, by) * DT(2, az, bz);
    }
  }
}

/// <nabla mu | sum_C w_C/|r-R_C| | nu>, PySCF int1e_ipnuc (w = -Z) or
/// int1e_iprinv (a single unit charge). The Hellmann-Feynman dR_C term is a
/// separate operator and is not included here.
template <class Real>
void nuclear_deriv_block(const PrimitiveShell<Real> &sa, const PrimitiveShell<Real> &sb,
                         const std::vector<PointCharge<Real>> &charges,
                         const TGrid<Real> &grid, Real *out) {
  const Real pi = pi_v<Real>();
  const int la = sa.l, lb = sb.l, lae = la + 1; // bra extended by one
  const int nca = ncart(la), ncb = ncart(lb);
  const std::size_t pstride = static_cast<std::size_t>(nca) * ncb;
  for (std::size_t i = 0; i < 3 * pstride; ++i) out[i] = Real(0);
  const Real p = sa.alpha + sb.alpha, mu = sa.alpha * sb.alpha / p;
  const int esz = (lae + 1) * (lb + 1) * (lae + lb + 1);
  Real Pd[3];
  std::vector<Real> E(static_cast<std::size_t>(3) * esz);
  for (int d = 0; d < 3; ++d) {
    Pd[d] = (sa.alpha * sa.center[d] + sb.alpha * sb.center[d]) / p;
    const Real ab = sa.center[d] - sb.center[d];
    e_coeffs(lae, lb, p, Pd[d] - sa.center[d], Pd[d] - sb.center[d], exp_(-mu * ab * ab),
             E.data() + d * esz);
  }
  // 1D nuclear factor g1[d][k*(lb+1)+j] with bra power k (0..lae); the
  // derivative mixes k, so the t and charge sums stay outside the shift.
  std::vector<Real> B(lae + lb + 1);
  const int stride = lb + 1;
  std::vector<Real> g1[3];
  for (int d = 0; d < 3; ++d)
    g1[d].assign(static_cast<std::size_t>(lae + 1) * stride, Real(0));
  const int nt = grid.n();
  for (const auto &c : charges)
    for (int it = 0; it < nt; ++it) {
      const Real t = grid.t[it], denom = p + t * t;
      const Real theta = p * t * t / denom, pref = sqrt_(pi / denom);
      const Real wt = grid.w[it] * c.weight;
      for (int d = 0; d < 3; ++d) {
        hermite_b(lae + lb, theta, Pd[d] - c.R[d], B.data());
        const Real *Ed = E.data() + d * esz;
        for (int k = 0; k <= lae; ++k)
          for (int j = 0; j <= lb; ++j) {
            Real acc = 0;
            for (int tau = 0; tau <= k + j; ++tau)
              acc += Ed[(k * (lb + 1) + j) * (lae + lb + 1) + tau] * B[tau];
            g1[d][k * stride + j] = pref * acc;
          }
      }
      auto g = [&](int d, int k, int j) { return g1[d][k * stride + j]; };
      for (int ka = 0; ka < nca; ++ka) {
        int a3[3];
        cart_comp(la, ka, a3[0], a3[1], a3[2]);
        for (int kb = 0; kb < ncb; ++kb) {
          int b3[3];
          cart_comp(lb, kb, b3[0], b3[1], b3[2]);
          // d/dx acting on bra: l*g(a-1) - 2alpha*g(a+1)
          Real Dx = -2 * sa.alpha * g(0, a3[0] + 1, b3[0]);
          if (a3[0] >= 1) Dx += Real(a3[0]) * g(0, a3[0] - 1, b3[0]);
          Real Dy = -2 * sa.alpha * g(1, a3[1] + 1, b3[1]);
          if (a3[1] >= 1) Dy += Real(a3[1]) * g(1, a3[1] - 1, b3[1]);
          Real Dz = -2 * sa.alpha * g(2, a3[2] + 1, b3[2]);
          if (a3[2] >= 1) Dz += Real(a3[2]) * g(2, a3[2] - 1, b3[2]);
          const Real gx = g(0, a3[0], b3[0]), gy = g(1, a3[1], b3[1]), gz = g(2, a3[2], b3[2]);
          const std::size_t k = static_cast<std::size_t>(ka) * ncb + kb;
          out[0 * pstride + k] += wt * Dx * gy * gz;
          out[1 * pstride + k] += wt * gx * Dy * gz;
          out[2 * pstride + k] += wt * gx * gy * Dz;
        }
      }
    }
}

// ---- device (GPU) 1e derivative paths ---------------------------------------
// Reuse make_1e_pairs (E on the host device PairTable, bra extended by 1 for the
// d/dR_bra shift) and assemble the three gradient-component matrices on device.
// Gradients are not symmetric, so all ordered pairs are used and no transpose is
// written. Dispatch is for float/double/long double; __float128 keeps host.

template <class Real>
std::array<std::vector<Real>, 3> overlap_deriv_dev(const ShellBasis<Real> &basis) {
  const int nao = basis.nao;
  auto op = make_1e_pairs(basis, 1, 0, Real(0), true);
  const int npair = op.npair;
  const std::size_t plane = static_cast<std::size_t>(nao) * nao;
  Kokkos::View<Real *> Gd("intti::odev::G", 3 * plane);
  auto pv = op.tab.p, Ev = op.tab.E, alphav = op.alpha;
  auto lav = op.tab.la, lbv = op.tab.lb, eoffv = op.tab.e_off, aoa = op.aoa, aob = op.aob;
  const Real pi = pi_v<Real>();
  Kokkos::parallel_for(
      "intti::odev", Kokkos::RangePolicy<>(0, npair), KOKKOS_LAMBDA(int p) {
        const int lax = lav(p), lb0 = lbv(p), la0 = lax - 1, n1 = lax + lb0 + 1;
        const int esz = (lax + 1) * (lb0 + 1) * n1;
        const Real pref = sqrt_(pi / pv(p)), al = alphav(p);
        const int eo = eoffv(p), oa = aoa(p), ob = aob(p);
        auto S1 = [&](int d, int i, int j) {
          return pref * Ev(eo + d * esz + (i * (lb0 + 1) + j) * n1);
        };
        auto DS = [&](int d, int i, int j) {
          Real v = -2 * al * S1(d, i + 1, j);
          if (i >= 1) v += Real(i) * S1(d, i - 1, j);
          return v;
        };
        for (int ka = 0; ka < ncart(la0); ++ka) {
          int a3[3];
          cart_comp(la0, ka, a3[0], a3[1], a3[2]);
          for (int kb = 0; kb < ncart(lb0); ++kb) {
            int b3[3];
            cart_comp(lb0, kb, b3[0], b3[1], b3[2]);
            const std::size_t idx = static_cast<std::size_t>(oa + ka) * nao + ob + kb;
            const Real sx = S1(0, a3[0], b3[0]), sy = S1(1, a3[1], b3[1]), sz = S1(2, a3[2], b3[2]);
            Gd(0 * plane + idx) = DS(0, a3[0], b3[0]) * sy * sz;
            Gd(1 * plane + idx) = sx * DS(1, a3[1], b3[1]) * sz;
            Gd(2 * plane + idx) = sx * sy * DS(2, a3[2], b3[2]);
          }
        }
      });
  auto flat = to_host(Gd);
  std::array<std::vector<Real>, 3> G;
  for (int c = 0; c < 3; ++c)
    G[c].assign(flat.begin() + c * plane, flat.begin() + (c + 1) * plane);
  return G;
}

template <class Real>
std::array<std::vector<Real>, 3> kinetic_deriv_dev(const ShellBasis<Real> &basis) {
  const int nao = basis.nao;
  auto op = make_1e_pairs(basis, 1, 2, Real(0), true);
  const int npair = op.npair;
  const std::size_t plane = static_cast<std::size_t>(nao) * nao;
  Kokkos::View<Real *> Gd("intti::kdev::G", 3 * plane);
  auto pv = op.tab.p, Ev = op.tab.E, alphav = op.alpha, betav = op.beta;
  auto lav = op.tab.la, lbv = op.tab.lb, eoffv = op.tab.e_off, aoa = op.aoa, aob = op.aob;
  const Real pi = pi_v<Real>();
  Kokkos::parallel_for(
      "intti::kdev", Kokkos::RangePolicy<>(0, npair), KOKKOS_LAMBDA(int p) {
        const int lax = lav(p), lb2 = lbv(p), lb0 = lb2 - 2, la0 = lax - 1, n1 = lax + lb2 + 1;
        const int esz = (lax + 1) * (lb2 + 1) * n1;
        const Real pref = sqrt_(pi / pv(p)), al = alphav(p), bta = betav(p);
        const int eo = eoffv(p), oa = aoa(p), ob = aob(p);
        auto S1 = [&](int d, int i, int j) {
          return pref * Ev(eo + d * esz + (i * (lb2 + 1) + j) * n1);
        };
        auto T1 = [&](int d, int i, int j) {
          Real v = -2 * bta * bta * S1(d, i, j + 2) + bta * (2 * j + 1) * S1(d, i, j);
          if (j >= 2) v -= Real(0.5) * j * (j - 1) * S1(d, i, j - 2);
          return v;
        };
        auto DS = [&](int d, int i, int j) {
          Real v = -2 * al * S1(d, i + 1, j);
          if (i >= 1) v += Real(i) * S1(d, i - 1, j);
          return v;
        };
        auto DT = [&](int d, int i, int j) {
          Real v = -2 * al * T1(d, i + 1, j);
          if (i >= 1) v += Real(i) * T1(d, i - 1, j);
          return v;
        };
        for (int ka = 0; ka < ncart(la0); ++ka) {
          int a3[3];
          cart_comp(la0, ka, a3[0], a3[1], a3[2]);
          for (int kb = 0; kb < ncart(lb0); ++kb) {
            int b3[3];
            cart_comp(lb0, kb, b3[0], b3[1], b3[2]);
            const int ax = a3[0], ay = a3[1], az = a3[2], bx = b3[0], by = b3[1], bz = b3[2];
            const Real sx = S1(0, ax, bx), sy = S1(1, ay, by), sz = S1(2, az, bz);
            const Real tx = T1(0, ax, bx), ty = T1(1, ay, by), tz = T1(2, az, bz);
            const Real dsx = DS(0, ax, bx), dsy = DS(1, ay, by), dsz = DS(2, az, bz);
            const Real dtx = DT(0, ax, bx), dty = DT(1, ay, by), dtz = DT(2, az, bz);
            const std::size_t idx = static_cast<std::size_t>(oa + ka) * nao + ob + kb;
            Gd(0 * plane + idx) = dtx * sy * sz + dsx * ty * sz + dsx * sy * tz;
            Gd(1 * plane + idx) = tx * dsy * sz + sx * dty * sz + sx * dsy * tz;
            Gd(2 * plane + idx) = tx * sy * dsz + sx * ty * dsz + sx * sy * dtz;
          }
        }
      });
  auto flat = to_host(Gd);
  std::array<std::vector<Real>, 3> G;
  for (int c = 0; c < 3; ++c)
    G[c].assign(flat.begin() + c * plane, flat.begin() + (c + 1) * plane);
  return G;
}

template <class Real>
std::array<std::vector<Real>, 3>
nuclear_deriv_dev(const ShellBasis<Real> &basis,
                  const std::vector<PointCharge<Real>> &charges, const TGrid<Real> &grid) {
  const int nao = basis.nao;
  auto op = make_1e_pairs(basis, 1, 0, Real(0), true);
  const int npair = op.npair, nt = grid.n(), npc = static_cast<int>(charges.size());
  const std::size_t plane = static_cast<std::size_t>(nao) * nao;
  auto tv = to_device(grid.t, "intti::ndev::t");
  auto wv = to_device(grid.w, "intti::ndev::w");
  std::vector<Real> hcw(npc);
  for (int c = 0; c < npc; ++c) hcw[c] = charges[c].weight;
  auto cw = to_device(hcw, "intti::ndev::cw");
  Kokkos::View<Real *[3], Kokkos::LayoutLeft> cR("intti::ndev::cR", npc);
  {
    auto h = Kokkos::create_mirror_view(cR);
    for (int c = 0; c < npc; ++c)
      for (int d = 0; d < 3; ++d) h(c, d) = charges[c].R[d];
    Kokkos::deep_copy(cR, h);
  }
  Kokkos::View<Real *> Gd("intti::ndev::G", 3 * plane);
  auto pv = op.tab.p, Ev = op.tab.E, alphav = op.alpha;
  auto Pv = op.tab.P;
  auto lav = op.tab.la, lbv = op.tab.lb, eoffv = op.tab.e_off, aoa = op.aoa, aob = op.aob;
  const Real pi = pi_v<Real>();
  Kokkos::parallel_for(
      "intti::ndev", Kokkos::RangePolicy<>(0, npair), KOKKOS_LAMBDA(int p) {
        const int lax = lav(p), lb0 = lbv(p), la0 = lax - 1, n1 = lax + lb0 + 1;
        const int esz = (lax + 1) * (lb0 + 1) * n1;
        const Real pp = pv(p), al = alphav(p);
        const int eo = eoffv(p), oa = aoa(p), ob = aob(p);
        const Real Px = Pv(p, 0), Py = Pv(p, 1), Pz = Pv(p, 2);
        Real B[2 * LMAX + 2];
        for (int c = 0; c < npc; ++c) {
          const Real Rx = cR(c, 0), Ry = cR(c, 1), Rz = cR(c, 2), wc = cw(c);
          for (int it = 0; it < nt; ++it) {
            const Real t = tv(it), denom = pp + t * t, theta = pp * t * t / denom;
            const Real pref = sqrt_(pi / denom), wt = wv(it) * wc;
            // g(d,k,j) = pref * sum_tau E(d;k,j;tau) B_d[tau]; recompute B per axis
            auto gfun = [&](int d, Real Pc, Real Rc, int k, int j) {
              hermite_b(lax + lb0, theta, Pc - Rc, B);
              const int base = eo + d * esz + (k * (lb0 + 1) + j) * n1;
              Real s = 0;
              for (int tt = 0; tt <= k + j; ++tt) s += Ev(base + tt) * B[tt];
              return pref * s;
            };
            for (int ka = 0; ka < ncart(la0); ++ka) {
              int a3[3];
              cart_comp(la0, ka, a3[0], a3[1], a3[2]);
              for (int kb = 0; kb < ncart(lb0); ++kb) {
                int b3[3];
                cart_comp(lb0, kb, b3[0], b3[1], b3[2]);
                const Real gx = gfun(0, Px, Rx, a3[0], b3[0]);
                const Real gy = gfun(1, Py, Ry, a3[1], b3[1]);
                const Real gz = gfun(2, Pz, Rz, a3[2], b3[2]);
                Real Dx = -2 * al * gfun(0, Px, Rx, a3[0] + 1, b3[0]);
                if (a3[0] >= 1) Dx += Real(a3[0]) * gfun(0, Px, Rx, a3[0] - 1, b3[0]);
                Real Dy = -2 * al * gfun(1, Py, Ry, a3[1] + 1, b3[1]);
                if (a3[1] >= 1) Dy += Real(a3[1]) * gfun(1, Py, Ry, a3[1] - 1, b3[1]);
                Real Dz = -2 * al * gfun(2, Pz, Rz, a3[2] + 1, b3[2]);
                if (a3[2] >= 1) Dz += Real(a3[2]) * gfun(2, Pz, Rz, a3[2] - 1, b3[2]);
                const std::size_t idx = static_cast<std::size_t>(oa + ka) * nao + ob + kb;
                Gd(0 * plane + idx) += wt * Dx * gy * gz;
                Gd(1 * plane + idx) += wt * gx * Dy * gz;
                Gd(2 * plane + idx) += wt * gx * gy * Dz;
              }
            }
          }
        }
      });
  auto flat = to_host(Gd);
  std::array<std::vector<Real>, 3> G;
  for (int c = 0; c < 3; ++c)
    G[c].assign(flat.begin() + c * plane, flat.begin() + (c + 1) * plane);
  return G;
}

} // namespace detail

/// Overlap gradient <nabla mu | nu>: three nao x nao matrices (x, y, z),
/// matching PySCF int1e_ipovlp.
template <class Real>
std::array<std::vector<Real>, 3> overlap_deriv(const ShellBasis<Real> &basis) {
  if constexpr (kokkos_scalar_v<Real>)
    return detail::overlap_deriv_dev(basis);
  const int nao = basis.nao;
  std::array<std::vector<Real>, 3> G;
  for (auto &g : G) g.assign(static_cast<std::size_t>(nao) * nao, Real(0));
  const int ns = static_cast<int>(basis.shells.size());
  std::vector<Real> blk;
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const int nca = ncart(sa.l), ncb = ncart(sb.l);
      const std::size_t pstride = static_cast<std::size_t>(nca) * ncb;
      blk.assign(3 * pstride, Real(0));
      detail::overlap_deriv_block(sa, sb, blk.data());
      for (int d = 0; d < 3; ++d)
        for (int ka = 0; ka < nca; ++ka)
          for (int kb = 0; kb < ncb; ++kb)
            G[d][(basis.ao_off[a] + ka) * static_cast<std::size_t>(nao) + basis.ao_off[b] +
                 kb] = blk[d * pstride + static_cast<std::size_t>(ka) * ncb + kb];
    }
  return G;
}

/// Kinetic-energy gradient <nabla mu | T | nu>, matching PySCF int1e_ipkin.
template <class Real>
std::array<std::vector<Real>, 3> kinetic_deriv(const ShellBasis<Real> &basis) {
  if constexpr (kokkos_scalar_v<Real>)
    return detail::kinetic_deriv_dev(basis);
  const int nao = basis.nao;
  std::array<std::vector<Real>, 3> G;
  for (auto &g : G) g.assign(static_cast<std::size_t>(nao) * nao, Real(0));
  const int ns = static_cast<int>(basis.shells.size());
  std::vector<Real> blk;
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const int nca = ncart(sa.l), ncb = ncart(sb.l);
      const std::size_t pstride = static_cast<std::size_t>(nca) * ncb;
      blk.assign(3 * pstride, Real(0));
      detail::kinetic_deriv_block(sa, sb, blk.data());
      for (int d = 0; d < 3; ++d)
        for (int ka = 0; ka < nca; ++ka)
          for (int kb = 0; kb < ncb; ++kb)
            G[d][(basis.ao_off[a] + ka) * static_cast<std::size_t>(nao) + basis.ao_off[b] +
                 kb] = blk[d * pstride + static_cast<std::size_t>(ka) * ncb + kb];
    }
  return G;
}

/// Nuclear-attraction gradient <nabla mu | sum_C w_C/|r-R_C| | nu>: three
/// nao x nao matrices. With charges carrying w = -Z this is PySCF int1e_ipnuc
/// (the bra-gradient piece of the nuclear-attraction force; the
/// Hellmann-Feynman dR_C term is separate).
template <class Real>
std::array<std::vector<Real>, 3>
nuclear_deriv(const ShellBasis<Real> &basis,
              const std::vector<PointCharge<Real>> &charges,
              const TGrid<Real> &grid) {
  if constexpr (kokkos_scalar_v<Real>) {
    bool ok = true;
    for (const auto &s : basis.shells)
      if (s.l > LMAX) ok = false;
    if (ok) return detail::nuclear_deriv_dev(basis, charges, grid);
  }
  const int nao = basis.nao;
  std::array<std::vector<Real>, 3> G;
  for (auto &g : G) g.assign(static_cast<std::size_t>(nao) * nao, Real(0));
  const int ns = static_cast<int>(basis.shells.size());
  std::vector<Real> blk;
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const int nca = ncart(sa.l), ncb = ncart(sb.l);
      const std::size_t pstride = static_cast<std::size_t>(nca) * ncb;
      blk.assign(3 * pstride, Real(0));
      detail::nuclear_deriv_block(sa, sb, charges, grid, blk.data());
      for (int d = 0; d < 3; ++d)
        for (int ka = 0; ka < nca; ++ka)
          for (int kb = 0; kb < ncb; ++kb)
            G[d][(basis.ao_off[a] + ka) * static_cast<std::size_t>(nao) + basis.ao_off[b] +
                 kb] = blk[d * pstride + static_cast<std::size_t>(ka) * ncb + kb];
    }
  return G;
}

} // namespace intti
