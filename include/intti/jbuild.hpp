// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

#include <stdexcept>
#include <type_traits>
#include <vector>

#include "batch.hpp"
#include "device.hpp"
#include "gto.hpp"
#include "hermite1d.hpp"
#include "multipole.hpp"
#include "tgrid.hpp"

namespace intti {

/// Maximum total pair angular momentum (l_a + l_b) supported by the J-build
/// stack buffers.
inline constexpr int JLMAX = 8;

namespace detail {

/// Far-field (multipole) J contribution -- the FMM far-field of coulomb_build
/// (M-MP). For a bra pair p and ket pair q the near-field t-node sweep couples
/// through the factorised Bx*By*Bz; when the pairs are well separated
/// (alpha_pq |P_p - P_q|^2 > far_cut, alpha_pq = p_p p_q/(p_p+p_q)) the same
/// coupling is the exponent-free multipole tensor T_{tuv}(X) contracted with
/// the density-weighted ket Hermite moments -- grid-free and, for Gaussian
/// pairs, exact up to the neglected exp(-alpha_pq R^2). ADDS the far part into
/// J; the device kernel skips exactly these pairs (same predicate), so the two
/// partition the ket sum with no gap and no double count. Host-side for now
/// (the tensor recursion's scratch is unfriendly to GPU registers; device port
/// is M18). Real centres only.
template <class Real>
void coulomb_farfield_add(const PairTable<Real> &pairs, const Real *D,
                          const std::vector<int> &h_prod,
                          const std::vector<int> &h_hoff, Real far_cut, Real *J,
                          const Real *Q, const Real *bound, Real tau,
                          int rank, int nranks) {
  const int npair = pairs.npair;
  const Real pi = pi_v<Real>();
  const bool screen = tau > Real(0) && Q != nullptr && bound != nullptr;
  // host mirrors of the pair data (keep native View indexing)
  auto p_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, pairs.p);
  auto P_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, pairs.P);
  auto E_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, pairs.E);
  auto eoff_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, pairs.e_off);
  const auto &la_h = pairs.h_la;
  const auto &lb_h = pairs.h_lb;
  const int nherm = h_hoff[npair];

  // per-pair density-weighted Hermite moments d^q (phase 1, host)
  std::vector<Real> dq(nherm, Real(0));
  for (int q = 0; q < npair; ++q) {
    const int la = la_h[q], lb = lb_h[q], n1 = la + lb + 1;
    const int esz = (la + 1) * (lb + 1) * n1;
    const int off = h_hoff[q];
    for (int ka = 0; ka < ncart(la); ++ka) {
      int a3[3];
      cart_comp(la, ka, a3[0], a3[1], a3[2]);
      for (int kb = 0; kb < ncart(lb); ++kb) {
        int b3[3];
        cart_comp(lb, kb, b3[0], b3[1], b3[2]);
        const Real c = D[h_prod[q] + ka * ncart(lb) + kb];
        if (c == Real(0)) continue;
        const Real *Ex = &E_h(eoff_h(q) + 0 * esz + (a3[0] * (lb + 1) + b3[0]) * n1);
        const Real *Ey = &E_h(eoff_h(q) + 1 * esz + (a3[1] * (lb + 1) + b3[1]) * n1);
        const Real *Ez = &E_h(eoff_h(q) + 2 * esz + (a3[2] * (lb + 1) + b3[2]) * n1);
        for (int t = 0; t <= a3[0] + b3[0]; ++t)
          for (int u = 0; u <= a3[1] + b3[1]; ++u)
            for (int v = 0; v <= a3[2] + b3[2]; ++v)
              dq[off + (t * n1 + u) * n1 + v] += c * Ex[t] * Ey[u] * Ez[v];
      }
    }
  }

  std::vector<Real> jp(nherm, Real(0)), T;
  for (int p = 0; p < npair; ++p) {
    if (nranks > 1 && p % nranks != rank) continue;
    const int lap = la_h[p], lbp = lb_h[p], Lp = lap + lbp, np1 = Lp + 1;
    const int joff = h_hoff[p];
    const Real pp = p_h(p);
    for (int q = 0; q < npair; ++q) {
      if (screen && Q[p] * bound[q] < tau) continue;
      const Real pq = p_h(q);
      Real X[3], R2 = 0;
      for (int d = 0; d < 3; ++d) {
        X[d] = P_h(p, d) - P_h(q, d);
        R2 += X[d] * X[d];
      }
      if (pp * pq / (pp + pq) * R2 <= far_cut) continue; // near: handled on device
      const int Lq = la_h[q] + lb_h[q], nq1 = Lq + 1;
      const int doff = h_hoff[q];
      const int L = Lp + Lq, Dt = L + 1;
      T.assign(static_cast<std::size_t>(Dt) * Dt * Dt, Real(0));
      multipole_tensor(L, X, T.data());
      const Real pref = (pi * pi * pi) / (pp * pq * sqrt_(pp * pq));
      for (int t = 0; t < np1; ++t)
        for (int u = 0; u < np1; ++u)
          for (int v = 0; v < np1; ++v) {
            Real s = 0;
            for (int a = 0; a < nq1; ++a)
              for (int b = 0; b < nq1; ++b)
                for (int c = 0; c < nq1; ++c) {
                  const Real m = dq[doff + (a * nq1 + b) * nq1 + c];
                  const int sgn = (a + b + c) & 1;
                  const Real term = m * T[((t + a) * Dt + (u + b)) * Dt + (v + c)];
                  s += sgn ? -term : term;
                }
            jp[joff + (t * np1 + u) * np1 + v] += pref * s;
          }
    }
  }

  // phase 3 (host): transform back through the bra E tables, ADD into J
  for (int p = 0; p < npair; ++p) {
    if (nranks > 1 && p % nranks != rank) continue;
    const int la = la_h[p], lb = lb_h[p], n1 = la + lb + 1;
    const int esz = (la + 1) * (lb + 1) * n1;
    const int joff = h_hoff[p];
    for (int ka = 0; ka < ncart(la); ++ka) {
      int a3[3];
      cart_comp(la, ka, a3[0], a3[1], a3[2]);
      for (int kb = 0; kb < ncart(lb); ++kb) {
        int b3[3];
        cart_comp(lb, kb, b3[0], b3[1], b3[2]);
        const Real *Ex = &E_h(eoff_h(p) + 0 * esz + (a3[0] * (lb + 1) + b3[0]) * n1);
        const Real *Ey = &E_h(eoff_h(p) + 1 * esz + (a3[1] * (lb + 1) + b3[1]) * n1);
        const Real *Ez = &E_h(eoff_h(p) + 2 * esz + (a3[2] * (lb + 1) + b3[2]) * n1);
        Real s = 0;
        for (int t = 0; t <= a3[0] + b3[0]; ++t)
          for (int u = 0; u <= a3[1] + b3[1]; ++u)
            for (int v = 0; v <= a3[2] + b3[2]; ++v)
              s += Ex[t] * Ey[u] * Ez[v] * jp[joff + (t * n1 + u) * n1 + v];
        J[h_prod[p] + ka * ncart(lb) + kb] += s;
      }
    }
  }
}

} // namespace detail

/// Coulomb matrix build in Hermite t-space:
///   J_q = sum_q' D_q' (q | q')
/// over the pair-product basis of the table, without ever forming quartets:
/// the density is contracted into per-pair Hermite tensors
///   d^q_{tau nu phi} = sum_comp D_{q,comp} E^x_tau E^y_nu E^z_phi   (once),
/// each (bra p, ket q, t node) couples through the Gaussian-derivative
/// arrays B_n (hermite1d.hpp) as three sequential 1D mode contractions, and
/// the result transforms back through the bra E tables. Cost
/// O(nt npair^2 L^4); the delta tail (LinLog grids) is the same contraction
/// at the t -> infinity limit.
///
/// D and J are indexed like CholeskyBasis: prod_offset per pair, Cartesian
/// components within (comp = ka*ncart(lb)+kb). Symmetry weighting of the
/// density (x2 for off-diagonal AO pairs in a triangular pair list) is the
/// caller's contract.
/// Optional screening: pass per-pair Schwarz factors Q and ket bounds
/// bound[q] = Q_q * max|D_q|; ket pairs with Q_p * bound[q] < tau are
/// skipped (tau = 0 or null pointers disable screening).
///
/// MPI distribution (M8): when nranks > 1, each rank computes only the
/// bra pairs p with p % nranks == rank and leaves the rest zero; the owned
/// sets are disjoint, so summing the per-rank J matrices (MPI_Allreduce)
/// reproduces the serial result exactly. Defaults reproduce the serial
/// (rank = 0, nranks = 1) behavior byte-for-byte.
///
/// Optional far-field (multipole) acceleration: far_tau > 0 turns on the FMM
/// near/far split. A ket pair q well separated from the bra pair p
/// (alpha_pq |P_p-P_q|^2 > -ln(far_tau), alpha_pq = p_p p_q/(p_p+p_q)) is
/// handled by the exponent-free multipole tensor instead of the t-node sweep,
/// with relative error ~ far_tau. far_tau = 0 (default) reproduces the pure
/// t-quadrature build exactly.
template <class Real>
void coulomb_build(const PairTable<Real> &pairs, const Real *D,
                   const TGrid<Real> &grid, Real *J, const Real *Q = nullptr,
                   const Real *bound = nullptr, Real tau = Real(0),
                   int rank = 0, int nranks = 1, Real far_tau = Real(0),
                   int p0 = 0, int p1 = -1, bool tiled = false) {
  static_assert(kokkos_scalar_v<Real>,
                "coulomb_build requires float, double or long double");
  // Output tiling: when tiled, phase 1 (the full ket Hermite density d^q) is
  // shared, and phases 2/3 (the O(npair) per-bra-pair tensors j^p and the output
  // J^p) are restricted to bra pairs [p0,p1) and staged at tile-local Hermite/
  // product offsets, so j^p/J^p occupy the tile footprint instead of the whole
  // O(npair) arrays. Per-element identical to the untiled build (bit-identical
  // regardless of tile size); d^q and D stay resident (output-half tiling).
  const bool screen = tau > Real(0) && Q != nullptr && bound != nullptr;
  const bool far = far_tau > Real(0);
  const Real far_cut = far ? -log_(far_tau) : Real(0);
  const int npair = pairs.npair;
  const int nt = grid.n();
  const Real pi = pi_v<Real>();
  const Real tail_coeff = grid.tail_coeff;
  const int Ktail = tail_coeff != Real(0) ? grid.tail_order : 0;
  const Real tc = grid.t_c;

  // per-pair offsets: products (D/J indexing) and Hermite tensors (L+1)^3
  std::vector<int> h_prod(npair + 1, 0), h_hoff(npair + 1, 0);
  for (int ip = 0; ip < npair; ++ip) {
    const int la = pairs.h_la[ip], lb = pairs.h_lb[ip];
    if (la + lb > JLMAX)
      throw std::invalid_argument("coulomb_build: pair angular momentum exceeds JLMAX");
    h_prod[ip + 1] = h_prod[ip] + ncart(la) * ncart(lb);
    const int n1 = la + lb + 1;
    h_hoff[ip + 1] = h_hoff[ip] + n1 * n1 * n1;
  }
  const int nprod = h_prod[npair], nherm = h_hoff[npair];
  const int P1 = (p1 < 0) ? npair : p1;
  const int hbase = tiled ? h_hoff[p0] : 0;   // tile-local Hermite offset
  const int pbase = tiled ? h_prod[p0] : 0;   // tile-local product offset
  const int jp_sz = tiled ? (h_hoff[P1] - hbase) : nherm;
  const int jv_sz = tiled ? (h_prod[P1] - pbase) : nprod;

  auto prodv = detail::to_device(h_prod, "intti::j::prod");
  auto hoffv = detail::to_device(h_hoff, "intti::j::hoff");
  auto Dv = detail::to_device(D, static_cast<std::size_t>(nprod), "intti::j::D");
  Kokkos::View<Real *> dq("intti::j::dq", nherm), jp("intti::j::jp", jp_sz);
  Kokkos::View<Real *> Jv("intti::j::J", jv_sz);
  auto Qv = screen ? detail::to_device(Q, static_cast<std::size_t>(npair), "intti::j::Q")
                   : Kokkos::View<Real *>("intti::j::Q", 1);
  auto bv = screen ? detail::to_device(bound, static_cast<std::size_t>(npair), "intti::j::bound")
                   : Kokkos::View<Real *>("intti::j::bound", 1);
  auto lav = pairs.la;
  auto lbv = pairs.lb;
  auto pv = pairs.p;
  auto Pv = pairs.P;
  auto eoffv = pairs.e_off;
  auto Ev = pairs.E;
  auto tv = grid.t_dev;
  auto wv = grid.w_dev;

  // phase 1: Hermite density tensors d^q (t-independent)
  Kokkos::parallel_for(
      "intti::j::density", Kokkos::RangePolicy<>(0, npair), KOKKOS_LAMBDA(int q) {
        const int la = lav(q), lb = lbv(q), L = la + lb, n1 = L + 1;
        const int esz = (la + 1) * (lb + 1) * n1;
        const int off = hoffv(q);
        for (int i = 0; i < n1 * n1 * n1; ++i)
          dq(off + i) = 0;
        for (int ka = 0; ka < ncart(la); ++ka) {
          int a3[3];
          cart_comp(la, ka, a3[0], a3[1], a3[2]);
          for (int kb = 0; kb < ncart(lb); ++kb) {
            int b3[3];
            cart_comp(lb, kb, b3[0], b3[1], b3[2]);
            const Real c = Dv(prodv(q) + ka * ncart(lb) + kb);
            if (c == Real(0)) continue;
            const Real *Ex = &Ev(eoffv(q) + 0 * esz + (a3[0] * (lb + 1) + b3[0]) * n1);
            const Real *Ey = &Ev(eoffv(q) + 1 * esz + (a3[1] * (lb + 1) + b3[1]) * n1);
            const Real *Ez = &Ev(eoffv(q) + 2 * esz + (a3[2] * (lb + 1) + b3[2]) * n1);
            for (int tau = 0; tau <= a3[0] + b3[0]; ++tau)
              for (int nu = 0; nu <= a3[1] + b3[1]; ++nu)
                for (int ph = 0; ph <= a3[2] + b3[2]; ++ph)
                  dq(off + (tau * n1 + nu) * n1 + ph) += c * Ex[tau] * Ey[nu] * Ez[ph];
          }
        }
      });

  // phase 2: accumulate j^p over ket pairs and t nodes
  Kokkos::parallel_for(
      "intti::j::couple", Kokkos::RangePolicy<>(tiled ? p0 : 0, P1), KOKKOS_LAMBDA(int p) {
        const int lap = lav(p), lbp = lbv(p), Lp = lap + lbp, np1 = Lp + 1;
        const int joff = hoffv(p) - hbase; // tile-local
        for (int i = 0; i < np1 * np1 * np1; ++i)
          jp(joff + i) = 0;
        if (nranks > 1 && p % nranks != rank) return;
        const Real pp = pv(p);
        // higher-order delta tail folded as pseudo-nodes: one per Laplacian-order
        // combination (a,b,c), a+b+c <= Ktail, with the pair-independent weight
        // b_{a+b+c}/(a!b!c!), b_k = pi/(4^k (k+1) t_c^{2k+2}), and per-axis
        // Hermite-index shifts 2a/2b/2c (a ket derivative raises the index).
        int tsa[TAIL_NCOMBO], tsb[TAIL_NCOMBO], tsc[TAIL_NCOMBO], ntail = 0;
        Real twf[TAIL_NCOMBO];
        if (tail_coeff != Real(0)) {
          Real bcoef[TAIL_KMAX + 1], invf[TAIL_KMAX + 1];
          const Real tc2 = tc * tc;
          Real p4 = 1, tcp = tc2, fact = 1;
          for (int kk = 0; kk <= Ktail; ++kk) {
            bcoef[kk] = pi / (p4 * (kk + 1) * tcp);
            if (kk > 0) fact *= kk;
            invf[kk] = Real(1) / fact;
            p4 *= 4;
            tcp *= tc2;
          }
          for (int a = 0; a <= Ktail; ++a)
            for (int b = 0; b <= Ktail - a; ++b)
              for (int c = 0; c <= Ktail - a - b; ++c) {
                tsa[ntail] = a;
                tsb[ntail] = b;
                tsc[ntail] = c;
                twf[ntail] = bcoef[a + b + c] * invf[a] * invf[b] * invf[c];
                ntail++;
              }
        }
        for (int q = 0; q < npair; ++q) {
          if (screen && Qv(p) * bv(q) < tau) continue;
          const int Lq = lav(q) + lbv(q), nq1 = Lq + 1;
          const int doff = hoffv(q);
          const Real pq = pv(q);
          Real X[3];
          Real R2 = 0;
          for (int d = 0; d < 3; ++d) {
            X[d] = Pv(p, d) - Pv(q, d);
            R2 += X[d] * X[d];
          }
          // far pairs go to the multipole post-pass (same predicate)
          if (far && pp * pq / (pp + pq) * R2 > far_cut) continue;
          const int nB = Lp + Lq + 1;
          Real Bx[2 * JLMAX + 2 * TAIL_KMAX + 1], By[2 * JLMAX + 2 * TAIL_KMAX + 1],
              Bz[2 * JLMAX + 2 * TAIL_KMAX + 1];
          Real t1[(JLMAX + 1) * (JLMAX + 1) * (JLMAX + 1)];
          Real t2[(JLMAX + 1) * (JLMAX + 1) * (JLMAX + 1)];
          // t nodes plus, for truncated grids, the delta-tail pseudo-nodes
          const int nsweep = nt + ntail;
          for (int it = 0; it < nsweep; ++it) {
            Real theta, wpref;
            int sa = 0, sb = 0, sc = 0; // per-axis Hermite-index shifts (2*order)
            if (it < nt) {
              const Real t = tv(it);
              const Real Dden = pp * pq + t * t * (pp + pq);
              theta = t * t * pp * pq / Dden;
              const Real pr = pi / sqrt_(Dden);
              wpref = wv(it) * pr * pr * pr;
            } else {
              const int j = it - nt;
              sa = 2 * tsa[j];
              sb = 2 * tsb[j];
              sc = 2 * tsc[j];
              theta = pp * pq / (pp + pq); // rho_pq
              const Real pr = sqrt_(pi / (pp + pq));
              wpref = twf[j] * pr * pr * pr;
            }
            hermite_b(nB - 1 + sa, theta, X[0], Bx);
            hermite_b(nB - 1 + sb, theta, X[1], By);
            hermite_b(nB - 1 + sc, theta, X[2], Bz);
            // mode contractions with the (-1)^tau ket signs
            for (int t = 0; t < np1; ++t)
              for (int nu = 0; nu < nq1; ++nu)
                for (int ph = 0; ph < nq1; ++ph) {
                  Real s = 0;
                  for (int tau = 0; tau < nq1; ++tau) {
                    const Real term = dq(doff + (tau * nq1 + nu) * nq1 + ph) * Bx[t + tau + sa];
                    s += tau % 2 ? -term : term;
                  }
                  t1[(t * nq1 + nu) * nq1 + ph] = s;
                }
            for (int t = 0; t < np1; ++t)
              for (int u = 0; u < np1; ++u)
                for (int ph = 0; ph < nq1; ++ph) {
                  Real s = 0;
                  for (int nu = 0; nu < nq1; ++nu) {
                    const Real term = t1[(t * nq1 + nu) * nq1 + ph] * By[u + nu + sb];
                    s += nu % 2 ? -term : term;
                  }
                  t2[(t * np1 + u) * nq1 + ph] = s;
                }
            for (int t = 0; t < np1; ++t)
              for (int u = 0; u < np1; ++u)
                for (int v = 0; v < np1; ++v) {
                  Real s = 0;
                  for (int ph = 0; ph < nq1; ++ph) {
                    const Real term = t2[(t * np1 + u) * nq1 + ph] * Bz[v + ph + sc];
                    s += ph % 2 ? -term : term;
                  }
                  jp(joff + (t * np1 + u) * np1 + v) += wpref * s;
                }
          }
        }
      });

  // phase 3: transform back through the bra E tables
  Kokkos::parallel_for(
      "intti::j::output", Kokkos::RangePolicy<>(tiled ? p0 : 0, P1), KOKKOS_LAMBDA(int p) {
        const int la = lav(p), lb = lbv(p), L = la + lb, n1 = L + 1;
        const int esz = (la + 1) * (lb + 1) * n1;
        const int joff = hoffv(p) - hbase; // tile-local
        for (int ka = 0; ka < ncart(la); ++ka) {
          int a3[3];
          cart_comp(la, ka, a3[0], a3[1], a3[2]);
          for (int kb = 0; kb < ncart(lb); ++kb) {
            int b3[3];
            cart_comp(lb, kb, b3[0], b3[1], b3[2]);
            const Real *Ex = &Ev(eoffv(p) + 0 * esz + (a3[0] * (lb + 1) + b3[0]) * n1);
            const Real *Ey = &Ev(eoffv(p) + 1 * esz + (a3[1] * (lb + 1) + b3[1]) * n1);
            const Real *Ez = &Ev(eoffv(p) + 2 * esz + (a3[2] * (lb + 1) + b3[2]) * n1);
            Real s = 0;
            for (int t = 0; t <= a3[0] + b3[0]; ++t)
              for (int u = 0; u <= a3[1] + b3[1]; ++u)
                for (int v = 0; v <= a3[2] + b3[2]; ++v)
                  s += Ex[t] * Ey[u] * Ez[v] * jp(joff + (t * n1 + u) * n1 + v);
            Jv(prodv(p) - pbase + ka * ncart(lb) + kb) = s;
          }
        }
      });

  auto hJ = Kokkos::create_mirror_view(Jv);
  Kokkos::deep_copy(hJ, Jv);
  for (int i = 0; i < jv_sz; ++i)
    J[pbase + i] = hJ(i);

  // FMM far-field: add the multipole contribution of the pairs the device
  // kernel skipped (same near/far predicate -> exact partition). Not supported
  // in the tiled path (the tiled driver uses far_tau = 0).
  if (far && !tiled)
    detail::coulomb_farfield_add(pairs, D, h_prod, h_hoff, far_cut, J, Q, bound,
                                 tau, rank, nranks);
}

} // namespace intti
