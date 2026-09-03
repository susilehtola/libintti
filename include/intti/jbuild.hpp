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
#include "tgrid.hpp"

namespace intti {

/// Maximum total pair angular momentum (l_a + l_b) supported by the J-build
/// stack buffers.
inline constexpr int JLMAX = 8;

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
template <class Real>
void coulomb_build(const PairTable<Real> &pairs, const Real *D,
                   const TGrid<Real> &grid, Real *J, const Real *Q = nullptr,
                   const Real *bound = nullptr, Real tau = Real(0),
                   int rank = 0, int nranks = 1) {
  static_assert(kokkos_scalar_v<Real>,
                "coulomb_build requires float, double or long double");
  const bool screen = tau > Real(0) && Q != nullptr && bound != nullptr;
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

  auto prodv = detail::to_device(h_prod, "intti::j::prod");
  auto hoffv = detail::to_device(h_hoff, "intti::j::hoff");
  auto Dv = detail::to_device(D, static_cast<std::size_t>(nprod), "intti::j::D");
  Kokkos::View<Real *> dq("intti::j::dq", nherm), jp("intti::j::jp", nherm);
  Kokkos::View<Real *> Jv("intti::j::J", nprod);
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
      "intti::j::couple", Kokkos::RangePolicy<>(0, npair), KOKKOS_LAMBDA(int p) {
        const int lap = lav(p), lbp = lbv(p), Lp = lap + lbp, np1 = Lp + 1;
        const int joff = hoffv(p);
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
          for (int d = 0; d < 3; ++d)
            X[d] = Pv(p, d) - Pv(q, d);
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
      "intti::j::output", Kokkos::RangePolicy<>(0, npair), KOKKOS_LAMBDA(int p) {
        const int la = lav(p), lb = lbv(p), L = la + lb, n1 = L + 1;
        const int esz = (la + 1) * (lb + 1) * n1;
        const int joff = hoffv(p);
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
            Jv(prodv(p) + ka * ncart(lb) + kb) = s;
          }
        }
      });

  auto hJ = Kokkos::create_mirror_view(Jv);
  Kokkos::deep_copy(hJ, Jv);
  for (int i = 0; i < nprod; ++i)
    J[i] = hJ(i);
}

} // namespace intti
