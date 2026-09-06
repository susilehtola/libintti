// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "gto.hpp"
#include "hermite1d.hpp"
#include "tgrid.hpp"

namespace intti {

/// Device-resident SoA table of shell pairs with precomputed MD E tables.
/// The E tables are t-independent: they are computed once at table build and
/// reused across every quartet and every t node (the pair-level reuse
/// economy). The batched drivers require builtin floating-point Real;
/// class-type scalars (arbitrary precision) use the single-quartet
/// eri_quartet() host path instead.
// NOTE: the template parameter is the SCALAR type. Only the product centres and
// the E coefficients become complex for London/GIAO pairs in a finite magnetic
// field; the exponent sum p = alpha+beta (and hence D, theta, rho and the
// prefactors) stays real, so it is typed real_t<Real>. Kokkos::complex is a
// permitted scalar here (device-runnable); std::complex is not.
template <class Real> struct PairTable {
  static_assert(kokkos_scalar_v<Real>,
                "PairTable requires float, double, long double or Kokkos::complex; std::complex and __float128 scalars use the serial eri_quartet() path");
  int npair{0};
  Kokkos::View<real_t<Real> *> p;                  ///< alpha + beta (always real)
  Kokkos::View<Real *[3], Kokkos::LayoutLeft> P;   ///< product centers
  Kokkos::View<int *> la, lb;
  Kokkos::View<int *> e_off; ///< offset of the pair's E block (3 directions)
  /// E tables, ragged: pair ip, direction d starts at e_off(ip) + d*esz where
  /// esz = (la+1)(lb+1)(la+lb+1)
  Kokkos::View<Real *> E;

  // host copies used for batch planning
  std::vector<int> h_la, h_lb, h_eoff;
};

/// Size of one direction's E table.
inline int e_size(int la, int lb) { return (la + 1) * (lb + 1) * (la + lb + 1); }

template <class Real>
PairTable<Real> make_pair_table(const std::vector<ShellPair<Real>> &pairs) {
  PairTable<Real> tab;
  tab.npair = static_cast<int>(pairs.size());
  tab.h_la.resize(tab.npair);
  tab.h_lb.resize(tab.npair);
  tab.h_eoff.resize(tab.npair);
  int etot = 0;
  for (int ip = 0; ip < tab.npair; ++ip) {
    tab.h_la[ip] = pairs[ip].la;
    tab.h_lb[ip] = pairs[ip].lb;
    tab.h_eoff[ip] = etot;
    etot += 3 * e_size(pairs[ip].la, pairs[ip].lb);
  }
  tab.p = Kokkos::View<real_t<Real> *>("intti::pairs::p", tab.npair);
  tab.P = Kokkos::View<Real *[3], Kokkos::LayoutLeft>("intti::pairs::P", tab.npair);
  tab.la = Kokkos::View<int *>("intti::pairs::la", tab.npair);
  tab.lb = Kokkos::View<int *>("intti::pairs::lb", tab.npair);
  tab.e_off = Kokkos::View<int *>("intti::pairs::e_off", tab.npair);
  tab.E = Kokkos::View<Real *>("intti::pairs::E", etot);
  auto hp = Kokkos::create_mirror_view(tab.p);
  auto hP = Kokkos::create_mirror_view(tab.P);
  auto hla = Kokkos::create_mirror_view(tab.la);
  auto hlb = Kokkos::create_mirror_view(tab.lb);
  auto hoff = Kokkos::create_mirror_view(tab.e_off);
  auto hE = Kokkos::create_mirror_view(tab.E);
  for (int ip = 0; ip < tab.npair; ++ip) {
    const auto &sp = pairs[ip];
    hp(ip) = sp.p;
    hla(ip) = sp.la;
    hlb(ip) = sp.lb;
    hoff(ip) = tab.h_eoff[ip];
    const int esz = e_size(sp.la, sp.lb);
    for (int d = 0; d < 3; ++d) {
      hP(ip, d) = sp.P[d];
      e_coeffs(sp.la, sp.lb, sp.p, sp.P[d] - sp.A[d], sp.P[d] - sp.B[d], sp.K[d],
               &hE(tab.h_eoff[ip] + d * esz));
    }
  }
  Kokkos::deep_copy(tab.p, hp);
  Kokkos::deep_copy(tab.P, hP);
  Kokkos::deep_copy(tab.la, hla);
  Kokkos::deep_copy(tab.lb, hlb);
  Kokkos::deep_copy(tab.e_off, hoff);
  Kokkos::deep_copy(tab.E, hE);
  return tab;
}

/// A batch of quartets, as (bra pair, ket pair) index lists into a PairTable.
template <class Real> struct QuartetBatch {
  Kokkos::View<int *[2], Kokkos::LayoutLeft> quartets;
  Kokkos::View<int *> out_offset; ///< exclusive scan of per-quartet ncart products
  int nq{0};
  int nout_total{0};
  // host planning data
  std::vector<std::pair<int, int>> h_quartets;
  std::vector<int> h_offset;
};

template <class Real>
QuartetBatch<Real> make_batch(const PairTable<Real> &pairs,
                              const std::vector<std::pair<int, int>> &quartets) {
  QuartetBatch<Real> b;
  b.nq = static_cast<int>(quartets.size());
  b.h_quartets = quartets;
  b.h_offset.resize(b.nq + 1);
  b.h_offset[0] = 0;
  for (int iq = 0; iq < b.nq; ++iq) {
    const auto [ib, ik] = quartets[iq];
    const int nout = ncart(pairs.h_la[ib]) * ncart(pairs.h_lb[ib]) *
                     ncart(pairs.h_la[ik]) * ncart(pairs.h_lb[ik]);
    b.h_offset[iq + 1] = b.h_offset[iq] + nout;
  }
  b.nout_total = b.h_offset[b.nq];
  b.quartets = Kokkos::View<int *[2], Kokkos::LayoutLeft>("intti::batch::q", b.nq);
  b.out_offset = Kokkos::View<int *>("intti::batch::off", b.nq + 1);
  auto hq = Kokkos::create_mirror_view(b.quartets);
  auto ho = Kokkos::create_mirror_view(b.out_offset);
  for (int iq = 0; iq < b.nq; ++iq) {
    hq(iq, 0) = quartets[iq].first;
    hq(iq, 1) = quartets[iq].second;
    ho(iq) = b.h_offset[iq];
  }
  ho(b.nq) = b.h_offset[b.nq];
  Kokkos::deep_copy(b.quartets, hq);
  Kokkos::deep_copy(b.out_offset, ho);
  return b;
}

/// Sort a quartet list by (la,lb,lc,ld) class so batches have uniform work.
template <class Real>
void sort_by_class(const PairTable<Real> &pairs,
                   std::vector<std::pair<int, int>> &quartets) {
  auto key = [&](const std::pair<int, int> &q) {
    return ((pairs.h_la[q.first] * (LMAX + 1) + pairs.h_lb[q.first]) * (LMAX + 1) +
            pairs.h_la[q.second]) *
               (LMAX + 1) +
           pairs.h_lb[q.second];
  };
  std::stable_sort(quartets.begin(), quartets.end(),
                   [&](const auto &a, const auto &b) { return key(a) < key(b); });
}

/// Preallocated scratch for the batched driver; grows monotonically and is
/// never shrunk, so repeated calls do not allocate (hot path).
template <class Real> struct QuartetWorkspace {
  static_assert(kokkos_scalar_v<Real>,
                "QuartetWorkspace requires float, double or long double");
  int chunk{256}; ///< quartets processed per kernel sweep
  Kokkos::View<Real ***, Kokkos::LayoutLeft> f; ///< (nf, ncomb, 3*chunk)
  Kokkos::View<Real ***, Kokkos::LayoutLeft> g; ///< (nt, ncomb, 3*chunk)
  /// tail 1D derivative-overlaps, (ncomb, 3*chunk, TAIL_KMAX+1): last index is
  /// the Laplacian order m (D^{(2m)}); only order 0 is used for the leading tail
  Kokkos::View<Real ***, Kokkos::LayoutLeft> s1;

  void ensure(int nf, int ncomb, int nt) {
    if (static_cast<int>(f.extent(0)) < nf || static_cast<int>(f.extent(1)) < ncomb ||
        static_cast<int>(f.extent(2)) < 3 * chunk)
      f = Kokkos::View<Real ***, Kokkos::LayoutLeft>("intti::ws::f", nf, ncomb, 3 * chunk);
    if (static_cast<int>(g.extent(0)) < nt || static_cast<int>(g.extent(1)) < ncomb ||
        static_cast<int>(g.extent(2)) < 3 * chunk)
      g = Kokkos::View<Real ***, Kokkos::LayoutLeft>("intti::ws::g", nt, ncomb, 3 * chunk);
    if (static_cast<int>(s1.extent(0)) < ncomb || static_cast<int>(s1.extent(1)) < 3 * chunk)
      s1 = Kokkos::View<Real ***, Kokkos::LayoutLeft>("intti::ws::s1", ncomb, 3 * chunk,
                                                      TAIL_KMAX + 1);
  }
};

namespace detail {

/// Shared body of the batched drivers. Accumulate=false: out(offset+k) = val;
/// Accumulate=true: out(segment(iq)+k) += coeff(iq)*val (atomic).
template <class Real, bool Accumulate>
void eri_quartets_impl(const PairTable<Real> &pairs, const QuartetBatch<Real> &batch,
                       const TGrid<real_t<Real>> &grid, Kokkos::View<Real *> out,
                       QuartetWorkspace<Real> &ws, Kokkos::View<const Real *> coeff,
                       Kokkos::View<const int *> segment) {
  static_assert(kokkos_scalar_v<Real>,
                "batched driver requires a builtin floating-point type or "
                "Kokkos::complex; use eri_quartet() for class-type scalars");
  using R = real_t<Real>; // exponents, t grid, prefactors: always real
  const int nt = grid.n();
  const R pi = pi_v<R>();
  const R tail_coeff = grid.tail_coeff;
  const int Ktail = tail_coeff != R(0) ? grid.tail_order : 0;
  const R tc = grid.t_c;
  auto tv = grid.t_dev;
  auto wv = grid.w_dev;
  auto qv = batch.quartets;
  auto offv = batch.out_offset;
  auto pv = pairs.p;
  auto Pv = pairs.P;
  auto lav = pairs.la;
  auto lbv = pairs.lb;
  auto eoffv = pairs.e_off;
  auto Ev = pairs.E;

  for (int q0 = 0; q0 < batch.nq; q0 += ws.chunk) {
    const int nq = std::min(ws.chunk, batch.nq - q0);
    // chunk-wide maxima for the workspace shape
    int nfmax = 1, ncombmax = 1, noutmax = 1;
    for (int iq = q0; iq < q0 + nq; ++iq) {
      const auto [ib, ik] = batch.h_quartets[iq];
      const int la = pairs.h_la[ib], lb = pairs.h_lb[ib];
      const int lc = pairs.h_la[ik], ld = pairs.h_lb[ik];
      nfmax = std::max(nfmax, la + lb + lc + ld + 1);
      ncombmax = std::max(ncombmax, (la + 1) * (lb + 1) * (lc + 1) * (ld + 1));
      noutmax = std::max(noutmax, ncart(la) * ncart(lb) * ncart(lc) * ncart(ld));
    }
    ws.ensure(nfmax, ncombmax, nt);
    auto f = ws.f;
    auto g = ws.g;
    auto s1 = ws.s1;
    const int nf_max = nfmax, ncomb_max = ncombmax;

    // phase F: assemble f coefficients, one thread per (quartet, direction)
    Kokkos::parallel_for(
        "intti::batch::f",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nq, 3}),
        KOKKOS_LAMBDA(int jq, int d) {
          const int ib = qv(q0 + jq, 0), ik = qv(q0 + jq, 1);
          const int la = lav(ib), lb = lbv(ib), lc = lav(ik), ld = lbv(ik);
          const int nf = la + lb + lc + ld + 1;
          const int ncomb = (la + 1) * (lb + 1) * (lc + 1) * (ld + 1);
          const Real *Eb = &Ev(eoffv(ib) + d * (la + 1) * (lb + 1) * (la + lb + 1));
          const Real *Ek = &Ev(eoffv(ik) + d * (lc + 1) * (ld + 1) * (lc + ld + 1));
          const int col = 3 * jq + d;
          for (int combo = 0; combo < ncomb; ++combo)
            for (int n = 0; n < nf; ++n)
              f(n, combo, col) = 0;
          for (int ia = 0; ia <= la; ++ia)
            for (int ibb = 0; ibb <= lb; ++ibb)
              for (int ic = 0; ic <= lc; ++ic)
                for (int id = 0; id <= ld; ++id) {
                  const int combo =
                      ((ia * (lb + 1) + ibb) * (lc + 1) + ic) * (ld + 1) + id;
                  for (int t = 0; t <= ia + ibb; ++t) {
                    const Real eb = Eb[(ia * (lb + 1) + ibb) * (la + lb + 1) + t];
                    for (int tau = 0; tau <= ic + id; ++tau) {
                      const Real term =
                          eb * Ek[(ic * (ld + 1) + id) * (lc + ld + 1) + tau];
                      f(t + tau, combo, col) += tau % 2 ? -term : term;
                    }
                  }
                }
          // tail 1D derivative-overlaps D^{(2m)} = spref sum_n f_n B_{n+2m}
          // (theta -> rho limit); a ket spatial derivative only raises the
          // Hermite index. Cheap, do here.
          if (tail_coeff != R(0)) {
            Real B[4 * LMAX + 2 * TAIL_KMAX + 1]; // Hermite array: scalar (complex centres)
            const R p = pv(ib), q = pv(ik);
            const R rho = p * q / (p + q);
            const Real X = Pv(ib, d) - Pv(ik, d); // product-centre difference: scalar
            hermite_b(nf - 1 + 2 * Ktail, rho, X, B);
            const R spref = sqrt_(pi / (p + q));
            for (int combo = 0; combo < ncomb; ++combo)
              for (int m = 0; m <= Ktail; ++m) {
                Real s{};
                for (int n = 0; n < nf; ++n)
                  s += f(n, combo, col) * B[n + 2 * m];
                s1(combo, col, m) = spref * s;
              }
          }
        });

    // phase G: per-node 1D integrals, one thread per (quartet, t node)
    Kokkos::parallel_for(
        "intti::batch::g",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nq, nt}),
        KOKKOS_LAMBDA(int jq, int i) {
          const int ib = qv(q0 + jq, 0), ik = qv(q0 + jq, 1);
          const int la = lav(ib), lb = lbv(ib), lc = lav(ik), ld = lbv(ik);
          const int nf = la + lb + lc + ld + 1;
          const int ncomb = (la + 1) * (lb + 1) * (lc + 1) * (ld + 1);
          const R p = pv(ib), q = pv(ik);
          Real B[4 * LMAX + 1]; // scalar: complex centres for London pairs
          const R t = tv(i);
          const R D = p * q + t * t * (p + q);
          const R theta = t * t * p * q / D;
          const R pref = pi / sqrt_(D);
          for (int d = 0; d < 3; ++d) {
            const int col = 3 * jq + d;
            hermite_b(nf - 1, theta, Pv(ib, d) - Pv(ik, d), B);
            for (int combo = 0; combo < ncomb; ++combo) {
              Real s{};
              for (int n = 0; n < nf; ++n)
                s += f(n, combo, col) * B[n];
              g(i, combo, col) = pref * s;
            }
          }
        });

    // phase A: assemble output components
    Kokkos::parallel_for(
        "intti::batch::assemble",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nq, noutmax}),
        KOKKOS_LAMBDA(int jq, int k) {
          const int ib = qv(q0 + jq, 0), ik = qv(q0 + jq, 1);
          const int la = lav(ib), lb = lbv(ib), lc = lav(ik), ld = lbv(ik);
          const int nca = ncart(la), ncb = ncart(lb), ncc = ncart(lc), ncd = ncart(ld);
          if (k >= nca * ncb * ncc * ncd) return;
          // decode component -> per-direction combination indices
          const int kd = k % ncd, kc = (k / ncd) % ncc;
          const int kb = (k / (ncd * ncc)) % ncb, ka = k / (ncd * ncc * ncb);
          int a3[3], b3[3], c3[3], d3[3];
          cart_comp(la, ka, a3[0], a3[1], a3[2]);
          cart_comp(lb, kb, b3[0], b3[1], b3[2]);
          cart_comp(lc, kc, c3[0], c3[1], c3[2]);
          cart_comp(ld, kd, d3[0], d3[1], d3[2]);
          Real val{};
          {
            int cmb[3];
            for (int d = 0; d < 3; ++d)
              cmb[d] = ((a3[d] * (lb + 1) + b3[d]) * (lc + 1) + c3[d]) * (ld + 1) + d3[d];
            for (int i = 0; i < nt; ++i)
              val += wv(i) * g(i, cmb[0], 3 * jq) * g(i, cmb[1], 3 * jq + 1) *
                     g(i, cmb[2], 3 * jq + 2);
            if (tail_coeff != R(0)) {
              // tail = sum_{a+b+c<=K} b_{a+b+c}/(a!b!c!) D^{2a}_x D^{2b}_y D^{2c}_z,
              // b_k = pi/(4^k (k+1) t_c^{2k+2}); b_0/D^0^3 is the leading delta term
              R bcoef[TAIL_KMAX + 1], invf[TAIL_KMAX + 1];
              const R tc2 = tc * tc;
              R p4 = 1, tcp = tc2, fact = 1;
              for (int kk = 0; kk <= Ktail; ++kk) {
                bcoef[kk] = pi / (p4 * (kk + 1) * tcp);
                if (kk > 0) fact *= kk;
                invf[kk] = R(1) / fact;
                p4 *= 4;
                tcp *= tc2;
              }
              for (int ta = 0; ta <= Ktail; ++ta)
                for (int tb = 0; tb <= Ktail - ta; ++tb)
                  for (int tcc = 0; tcc <= Ktail - ta - tb; ++tcc)
                    val += bcoef[ta + tb + tcc] * invf[ta] * invf[tb] * invf[tcc] *
                           s1(cmb[0], 3 * jq, ta) * s1(cmb[1], 3 * jq + 1, tb) *
                           s1(cmb[2], 3 * jq + 2, tcc);
            }
          }
          if constexpr (Accumulate)
            Kokkos::atomic_add(&out(segment(q0 + jq) + k), coeff(q0 + jq) * val);
          else
            out(offv(q0 + jq) + k) = val;
        });
  }
  Kokkos::fence();
}

} // namespace detail

/// Batched primitive Cartesian ERI quartets. out must have batch.nout_total
/// entries; quartet iq's components are at [out_offset(iq), out_offset(iq+1)).
template <class Real>
void eri_quartets(const PairTable<Real> &pairs, const QuartetBatch<Real> &batch,
                  const TGrid<real_t<Real>> &grid, Kokkos::View<Real *> out,
                  QuartetWorkspace<Real> &ws) {
  detail::eri_quartets_impl<Real, false>(pairs, batch, grid, out, ws, {}, {});
}

/// Accumulating variant for contractions: out(segment(iq) + k) +=
/// coeff(iq) * quartet(iq, k). Quartets sharing a segment must share their
/// (la, lb, lc, ld) class; a contracted ERI block is one batch over the
/// primitive product list with coefficient products as coeff.
template <class Real>
void eri_quartets_accumulate(const PairTable<Real> &pairs,
                             const QuartetBatch<Real> &batch,
                             Kokkos::View<const Real *> coeff,
                             Kokkos::View<const int *> segment,
                             const TGrid<real_t<Real>> &grid, Kokkos::View<Real *> out,
                             QuartetWorkspace<Real> &ws) {
  detail::eri_quartets_impl<Real, true>(pairs, batch, grid, out, ws, coeff, segment);
}

} // namespace intti
