// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

#include <stdexcept>
#include <type_traits>
#include <vector>

#include "batch.hpp"
#include "gto.hpp"
#include "hermite1d.hpp"
#include "tgrid.hpp"

namespace intti {

/// Primitive Cartesian ERI quartet (ab|cd) by t quadrature:
///   (ab|cd) = sum_i w_i Ix(t_i) Iy(t_i) Iz(t_i) + tail_coeff * S_abcd,
/// where I_d are analytic 1D two-pair integrals and S_abcd is the
/// four-orbital overlap (the delta-function tail correction; only active
/// for truncated grids, i.e. TMapping::LinLog).
///
/// out is an array of ncart(la)*ncart(lb)*ncart(lc)*ncart(ld) values in
/// row-major (a, b, c, d) component order. Primitives are unnormalized.
///
/// Builtin floating-point Real executes through the batched Kokkos driver
/// (batch of one; see batch.hpp for the throughput API); class-type scalars
/// (e.g. MPFR wrappers) run on a serial host path.
template <class Scalar>
void eri_quartet(const ShellPair<Scalar> &bra, const ShellPair<Scalar> &ket,
                 const TGrid<real_t<Scalar>> &grid, Scalar *out) {
  using Real = real_t<Scalar>;   // exponents, grid, prefactors: always real
  const int la = bra.la, lb = bra.lb, lc = ket.la, ld = ket.lb;
  if (la > LMAX || lb > LMAX || lc > LMAX || ld > LMAX)
    throw std::invalid_argument("eri_quartet: shell angular momentum exceeds LMAX");

  if constexpr (kokkos_scalar_v<Scalar>) {
    auto tab = make_pair_table<Scalar>({bra, ket});
    auto batch = make_batch(tab, {{0, 1}});
    QuartetWorkspace<Scalar> ws;
    ws.chunk = 1;
    Kokkos::View<Scalar *> outv("intti::quartet::out", batch.nout_total);
    eri_quartets(tab, batch, grid, outv, ws);
    auto oh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, outv);
    for (std::size_t k = 0; k < batch.nout_total; ++k)
      out[k] = oh(k);
    return;
  } else {
    // serial host path for class-type scalars (arbitrary precision)
    const int Ltot = la + lb + lc + ld;
    const int nf = Ltot + 1; // Hermite orders 0..Ltot
    const int ncomb = (la + 1) * (lb + 1) * (lc + 1) * (ld + 1);
    const int nt = grid.n();
    const int nout = ncart(la) * ncart(lb) * ncart(lc) * ncart(ld);

    const Real p = bra.p, q = ket.p;
    const Real rho = p * q / (p + q);
    const Real pi = pi_v<Real>();
    Scalar X[3];
    for (int d = 0; d < 3; ++d)
      X[d] = bra.P[d] - ket.P[d];

    // f_n = sum_{t+tau=n} E_t^{bra} E_tau^{ket} (-1)^tau per direction and
    // per-direction component combination; index n + nf*(combo + ncomb*d)
    std::vector<Scalar> fh(static_cast<std::size_t>(nf) * ncomb * 3, Scalar(0));
    {
      std::vector<Scalar> Eb((la + 1) * (lb + 1) * (la + lb + 1));
      std::vector<Scalar> Ek((lc + 1) * (ld + 1) * (lc + ld + 1));
      for (int d = 0; d < 3; ++d) {
        e_coeffs(la, lb, p, bra.P[d] - bra.A[d], bra.P[d] - bra.B[d], bra.K[d], Eb.data());
        e_coeffs(lc, ld, q, ket.P[d] - ket.A[d], ket.P[d] - ket.B[d], ket.K[d], Ek.data());
        auto eb = [&](int i, int j, int t) { return Eb[(i * (lb + 1) + j) * (la + lb + 1) + t]; };
        auto ek = [&](int i, int j, int t) { return Ek[(i * (ld + 1) + j) * (lc + ld + 1) + t]; };
        for (int ia = 0; ia <= la; ++ia)
          for (int ib = 0; ib <= lb; ++ib)
            for (int ic = 0; ic <= lc; ++ic)
              for (int id = 0; id <= ld; ++id) {
                const int combo = ((ia * (lb + 1) + ib) * (lc + 1) + ic) * (ld + 1) + id;
                for (int t = 0; t <= ia + ib; ++t)
                  for (int tau = 0; tau <= ic + id; ++tau) {
                    const Scalar term = eb(ia, ib, t) * ek(ic, id, tau);
                    fh[(t + tau) + nf * (combo + ncomb * d)] += tau % 2 ? -term : term;
                  }
              }
      }
    }

    // map output component -> per-direction combination indices; k + nout*d
    std::vector<int> ch(static_cast<std::size_t>(nout) * 3);
    for (int ka = 0; ka < ncart(la); ++ka) {
      int a3[3];
      cart_comp(la, ka, a3[0], a3[1], a3[2]);
      for (int kb = 0; kb < ncart(lb); ++kb) {
        int b3[3];
        cart_comp(lb, kb, b3[0], b3[1], b3[2]);
        for (int kc = 0; kc < ncart(lc); ++kc) {
          int c3[3];
          cart_comp(lc, kc, c3[0], c3[1], c3[2]);
          for (int kd = 0; kd < ncart(ld); ++kd) {
            int d3[3];
            cart_comp(ld, kd, d3[0], d3[1], d3[2]);
            const int k = ((ka * ncart(lb) + kb) * ncart(lc) + kc) * ncart(ld) + kd;
            for (int d = 0; d < 3; ++d)
              ch[k + nout * d] =
                  ((a3[d] * (lb + 1) + b3[d]) * (lc + 1) + c3[d]) * (ld + 1) + d3[d];
          }
        }
      }
    }

    // delta-function tail: Losilla's leading term plus higher orders. Per axis
    // and per Laplacian order m, the 1D "derivative overlap" is the t->infinity
    // overlap with the ket differentiated 2m times, which only raises the
    // Hermite-integral index: D^{(2m)} = spref * sum_n fh_n B_{n+2m}. Layout
    // s1d[m*sstride + combo + ncomb*d]. (theta -> rho, prefactor sqrt(pi/(p+q)).)
    const int K = grid.tail_coeff != Real(0) ? grid.tail_order : 0;
    const std::size_t sstride = static_cast<std::size_t>(ncomb) * 3;
    std::vector<Scalar> s1d((K + 1) * sstride, Scalar(0));
    Real bcoef[TAIL_KMAX + 1] = {}, invf[TAIL_KMAX + 1] = {};
    if (grid.tail_coeff != Real(0)) {
      std::vector<Scalar> B(nf + 2 * K);
      const Real spref = sqrt_(pi / (p + q));
      for (int d = 0; d < 3; ++d) {
        hermite_b(Ltot + 2 * K, rho, X[d], B.data());
        for (int combo = 0; combo < ncomb; ++combo)
          for (int m = 0; m <= K; ++m) {
            Scalar s{};
            for (int n = 0; n < nf; ++n)
              s += fh[n + nf * (combo + ncomb * d)] * B[n + 2 * m];
            s1d[m * sstride + combo + ncomb * d] = spref * s;
          }
      }
      // combination coefficients: b_k = pi/(4^k (k+1) t_c^{2k+2}), 1/m!
      const Real tc2 = grid.t_c * grid.t_c;
      Real p4 = 1, tcp = tc2, fact = 1;
      for (int kk = 0; kk <= K; ++kk) {
        bcoef[kk] = pi / (p4 * (kk + 1) * tcp);
        if (kk > 0) fact *= kk;
        invf[kk] = Real(1) / fact;
        p4 *= 4;
        tcp *= tc2;
      }
    }

    std::vector<Scalar> g(static_cast<std::size_t>(nt) * ncomb * 3);
    std::vector<Scalar> B(nf);
    for (int i = 0; i < nt; ++i) {
      const Real t = grid.t[i];
      const Real D = p * q + t * t * (p + q);
      const Real theta = t * t * p * q / D;
      const Real pref = pi / sqrt_(D);
      for (int d = 0; d < 3; ++d) {
        hermite_b(Ltot, theta, X[d], B.data());
        for (int combo = 0; combo < ncomb; ++combo) {
          Scalar s{};
          for (int n = 0; n < nf; ++n)
            s += fh[n + nf * (combo + ncomb * d)] * B[n];
          g[i + nt * (combo + ncomb * d)] = pref * s;
        }
      }
    }
    for (int k = 0; k < nout; ++k) {
      const int cx = ch[k], cy = ch[k + nout], cz = ch[k + 2 * nout];
      Scalar val{};
      for (int i = 0; i < nt; ++i)
        val += grid.w[i] * g[i + nt * cx] * g[i + nt * (cy + ncomb)] *
               g[i + nt * (cz + 2 * ncomb)];
      if (grid.tail_coeff != Real(0)) {
        Scalar vt{};
        for (int a = 0; a <= K; ++a)
          for (int b = 0; b <= K - a; ++b)
            for (int c = 0; c <= K - a - b; ++c)
              vt += bcoef[a + b + c] * invf[a] * invf[b] * invf[c] *
                    s1d[a * sstride + cx] * s1d[b * sstride + cy + ncomb] *
                    s1d[c * sstride + cz + 2 * ncomb];
        val += vt;
      }
      out[k] = val;
    }
  }
}

} // namespace intti
