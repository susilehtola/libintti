// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Exchange (K) matrix build in Hermite t-space. Unlike the Coulomb build,
// the density indices straddle the two pairs (K_ab = sum_cd D_cd (ac|bd)),
// so there is no pair-local pre-contraction: the cost is that of screened
// on-the-fly quartet evaluation, organized pairwise so that the E tables and
// B_n arrays are shared and nothing is materialized. Cauchy-Schwarz plus
// density screening is what makes this tractable.

#include <stdexcept>
#include <type_traits>
#include <vector>

#include "batch.hpp"
#include "device.hpp"
#include "gto.hpp"
#include "hermite1d.hpp"
#include "symmetry.hpp"
#include "tgrid.hpp"

namespace intti {

/// Maximum shell angular momentum supported by the exchange-build stack
/// buffers.
inline constexpr int KLMAX = 4;

namespace detail {
/// forward declaration; defined in fock.hpp
}

template <class Real> struct ShellBasis;

/// K_{a ka, b kb} = sum_{c kc, d kd} D_{c kc, d kd} (a ka, c kc | b kb, d kd).
/// D and K are nao x nao row-major Cartesian AO matrices. Pairs (a c) and
/// (b d) with Q_ac Q_bd max|D_cd| < tau are skipped.
///
/// MPI distribution (M8): when nranks > 1, each rank computes only the
/// (a, b) shell blocks with (a*ns+b) % nranks == rank and leaves the rest
/// zero; the owned sets are disjoint, so summing the per-rank K matrices
/// (MPI_Allreduce) reproduces the serial result exactly. Defaults reproduce
/// the serial (rank = 0, nranks = 1) behavior byte-for-byte.
///
/// sym selects the permutational-symmetry policy (see symmetry.hpp). The
/// default Symmetry::None preserves the byte-for-byte guarantees above;
/// Symmetry::Full exploits the 8-fold ERI symmetry with atomic scatter (~4x
/// fewer quartet evals, but nondeterministic / not bit-reproducible).
template <class Real>
void exchange_build(const ShellBasis<Real> &basis, const Real *D,
                    const TGrid<Real> &grid, Real *K, Real tau = Real(1e-12),
                    int rank = 0, int nranks = 1,
                    Symmetry sym = Symmetry::None);

namespace detail {

template <class Real>
void exchange_build_impl_sym8(const std::vector<PrimitiveShell<Real>> &shells,
                              const std::vector<int> &ao_off, int nao, const Real *D,
                              const TGrid<Real> &grid, Real *K, Real tau,
                              const std::vector<Real> &Qex, const PairTable<Real> &tab,
                              int rank, int nranks);

template <class Real>
void exchange_build_impl(const std::vector<PrimitiveShell<Real>> &shells,
                         const std::vector<int> &ao_off, int nao, const Real *D,
                         const TGrid<Real> &grid, Real *K, Real tau,
                         const std::vector<Real> &Qex,
                         const PairTable<Real> &tab, int rank = 0,
                         int nranks = 1, Symmetry sym = Symmetry::None) {
  static_assert(kokkos_scalar_v<Real>,
                "exchange_build requires a builtin floating-point type in M5");
  if (sym == Symmetry::Full) {
    exchange_build_impl_sym8(shells, ao_off, nao, D, grid, K, tau, Qex, tab, rank, nranks);
    return;
  }
  const int ns = static_cast<int>(shells.size());
  const int nt = grid.n();
  const Real pi = pi_v<Real>();
  const Real tail_coeff = grid.tail_coeff;
  const int Ktail = tail_coeff != Real(0) ? grid.tail_order : 0;
  const Real tc = grid.t_c;

  for (const auto &sh : shells)
    if (sh.l > KLMAX)
      throw std::invalid_argument("exchange_build: shell angular momentum exceeds KLMAX");

  // per-(c,d) shell-block density maxima for screening
  std::vector<Real> maxD(static_cast<std::size_t>(ns) * ns, Real(0));
  for (int c = 0; c < ns; ++c)
    for (int d = 0; d < ns; ++d) {
      Real m = 0;
      for (int kc = 0; kc < ncart(shells[c].l); ++kc)
        for (int kd = 0; kd < ncart(shells[d].l); ++kd) {
          const Real v = D[(ao_off[c] + kc) * nao + ao_off[d] + kd];
          const Real a = v < 0 ? -v : v;
          if (a > m) m = a;
        }
      maxD[c * ns + d] = m;
    }

  // stage everything on device
  auto Dv = detail::to_device(D, static_cast<std::size_t>(nao) * nao, "intti::k::D");
  auto Qv = detail::to_device(Qex, "intti::k::Q");
  auto mDv = detail::to_device(maxD, "intti::k::maxD");
  auto aov = detail::to_device(ao_off, "intti::k::ao");
  std::vector<int> ls(ns);
  for (int i = 0; i < ns; ++i) ls[i] = shells[i].l;
  auto lv = detail::to_device(ls, "intti::k::l");
  Kokkos::View<Real *> Kv("intti::k::K", static_cast<std::size_t>(nao) * nao);
  Kokkos::deep_copy(Kv, Real(0));
  auto pv = tab.p;
  auto Pv = tab.P;
  auto eoffv = tab.e_off;
  auto Ev = tab.E;
  auto tv = grid.t_dev;
  auto wv = grid.w_dev;

  constexpr int KC1 = KLMAX + 1;
  constexpr int KNC = (KLMAX + 1) * (KLMAX + 2) / 2;

  Kokkos::parallel_for(
      "intti::k::build", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ns, ns}),
      KOKKOS_LAMBDA(int a, int b) {
        // K is symmetric (K_ab = sum_cd D_cd (ac|bd) = K_ba): compute the upper
        // triangle a <= b only and mirror the block into (b,a) -- ~2x.
        if (b < a) return;
        if (nranks > 1 && (a * ns + b) % nranks != rank) return;
        const int la = lv(a), lb = lv(b);
        const int nca = ncart(la), ncb = ncart(lb);
        Real Kblk[KNC * KNC];
        for (int i = 0; i < nca * ncb; ++i)
          Kblk[i] = 0;
        int a3s[KNC][3], b3s[KNC][3];
        for (int k = 0; k < nca; ++k)
          cart_comp(la, k, a3s[k][0], a3s[k][1], a3s[k][2]);
        for (int k = 0; k < ncb; ++k)
          cart_comp(lb, k, b3s[k][0], b3s[k][1], b3s[k][2]);
        // higher-order delta-tail pseudo-nodes: one per Laplacian order (a,b,c),
        // a+b+c <= Ktail, weight b_{a+b+c}/(a!b!c!) and per-axis index shift 2*.
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
          for (int ta = 0; ta <= Ktail; ++ta)
            for (int tb = 0; tb <= Ktail - ta; ++tb)
              for (int tcc = 0; tcc <= Ktail - ta - tb; ++tcc) {
                tsa[ntail] = ta;
                tsb[ntail] = tb;
                tsc[ntail] = tcc;
                twf[ntail] = bcoef[ta + tb + tcc] * invf[ta] * invf[tb] * invf[tcc];
                ntail++;
              }
        }
        for (int c = 0; c < ns; ++c) {
          const int lc = lv(c), ncc = ncart(lc);
          const int p = a * ns + c;
          int c3s[KNC][3];
          for (int k = 0; k < ncc; ++k)
            cart_comp(lc, k, c3s[k][0], c3s[k][1], c3s[k][2]);
          for (int d = 0; d < ns; ++d) {
            const int q = b * ns + d;
            if (Qv(p) * Qv(q) * mDv(c * ns + d) < tau) continue;
            const int ld = lv(d), ncd = ncart(ld);
            int d3s[KNC][3];
            for (int k = 0; k < ncd; ++k)
              cart_comp(ld, k, d3s[k][0], d3s[k][1], d3s[k][2]);
            const Real pp = pv(p), pq = pv(q);
            Real X[3];
            for (int dd = 0; dd < 3; ++dd)
              X[dd] = Pv(p, dd) - Pv(q, dd);
            const int np_e = la + lc + 1, nq_e = lb + ld + 1;
            const int eszp = (la + 1) * (lc + 1) * np_e;
            const int eszq = (lb + 1) * (ld + 1) * nq_e;
            const int nB = np_e + nq_e - 1;
            const int nsweep = nt + ntail;
            for (int it = 0; it < nsweep; ++it) {
              Real theta, wt, prefd;
              int sh[3] = {0, 0, 0}; // per-axis Hermite-index shift (2*order)
              if (it < nt) {
                const Real t = tv(it);
                const Real Dden = pp * pq + t * t * (pp + pq);
                theta = t * t * pp * pq / Dden;
                prefd = pi / sqrt_(Dden);
                wt = wv(it);
              } else {
                const int j = it - nt;
                sh[0] = 2 * tsa[j];
                sh[1] = 2 * tsb[j];
                sh[2] = 2 * tsc[j];
                theta = pp * pq / (pp + pq);
                prefd = sqrt_(pi / (pp + pq));
                wt = twf[j];
              }
              // per-direction pair-pair 1D integrals over all component combos
              Real g[3][KC1 * KC1 * KC1 * KC1];
              Real B[4 * KLMAX + 2 * TAIL_KMAX + 1];
              for (int dd = 0; dd < 3; ++dd) {
                hermite_b(nB - 1 + sh[dd], theta, X[dd], B);
                const Real *Ep = &Ev(eoffv(p) + dd * eszp);
                const Real *Eq = &Ev(eoffv(q) + dd * eszq);
                for (int ia = 0; ia <= la; ++ia)
                  for (int ic = 0; ic <= lc; ++ic)
                    for (int ib = 0; ib <= lb; ++ib)
                      for (int id = 0; id <= ld; ++id) {
                        Real s = 0;
                        for (int t = 0; t <= ia + ic; ++t) {
                          const Real ep = Ep[(ia * (lc + 1) + ic) * np_e + t];
                          for (int u = 0; u <= ib + id; ++u) {
                            const Real term = ep * Eq[(ib * (ld + 1) + id) * nq_e + u] *
                                              B[t + u + sh[dd]];
                            s += u % 2 ? -term : term;
                          }
                        }
                        g[dd][((ia * (lc + 1) + ic) * (lb + 1) + ib) * (ld + 1) + id] =
                            prefd * s;
                      }
              }
              // contract with the density block
              for (int ka = 0; ka < nca; ++ka)
                for (int kb = 0; kb < ncb; ++kb) {
                  Real s = 0;
                  for (int kc = 0; kc < ncc; ++kc)
                    for (int kd = 0; kd < ncd; ++kd) {
                      Real v = wt;
                      for (int dd = 0; dd < 3; ++dd)
                        v *= g[dd][((a3s[ka][dd] * (lc + 1) + c3s[kc][dd]) * (lb + 1) +
                                    b3s[kb][dd]) *
                                       (ld + 1) +
                                   d3s[kd][dd]];
                      s += Dv((aov(c) + kc) * nao + aov(d) + kd) * v;
                    }
                  Kblk[ka * ncb + kb] += s;
                }
            }
          }
        }
        for (int ka = 0; ka < nca; ++ka)
          for (int kb = 0; kb < ncb; ++kb) {
            const Real v = Kblk[ka * ncb + kb];
            Kv((aov(a) + ka) * nao + aov(b) + kb) = v;
            if (a != b) Kv((aov(b) + kb) * nao + aov(a) + ka) = v; // mirror
          }
      });

  auto hK = Kokkos::create_mirror_view(Kv);
  Kokkos::deep_copy(hK, Kv);
  for (std::size_t i = 0; i < static_cast<std::size_t>(nao) * nao; ++i)
    K[i] = hK(i);
}

/// Full 8-fold-symmetry exchange build (Symmetry::Full). Each unique quartet
/// (bra pair (a,c) with a<=c, ket pair (b,d) with b<=d, bra pair index >= ket
/// pair index) is evaluated once; its four symmetry-distinct density
/// contractions are accumulated over the t-nodes and the deduplicated 8-element
/// orbit is atomic-scattered into K. ~4x fewer quartet evals than the
/// deterministic build; nondeterministic summation order (see ExchangeAlgo).
template <class Real>
void exchange_build_impl_sym8(const std::vector<PrimitiveShell<Real>> &shells,
                              const std::vector<int> &ao_off, int nao, const Real *D,
                              const TGrid<Real> &grid, Real *K, Real tau,
                              const std::vector<Real> &Qex, const PairTable<Real> &tab,
                              int rank, int nranks) {
  static_assert(kokkos_scalar_v<Real>,
                "exchange_build requires a builtin floating-point type in M5");
  const int ns = static_cast<int>(shells.size());
  const int nt = grid.n();
  const Real pi = pi_v<Real>();
  const Real tail_coeff = grid.tail_coeff;
  const int Ktail = tail_coeff != Real(0) ? grid.tail_order : 0;
  const Real tc = grid.t_c;

  for (const auto &sh : shells)
    if (sh.l > KLMAX)
      throw std::invalid_argument("exchange_build: shell angular momentum exceeds KLMAX");

  std::vector<Real> maxD(static_cast<std::size_t>(ns) * ns, Real(0));
  for (int c = 0; c < ns; ++c)
    for (int d = 0; d < ns; ++d) {
      Real m = 0;
      for (int kc = 0; kc < ncart(shells[c].l); ++kc)
        for (int kd = 0; kd < ncart(shells[d].l); ++kd) {
          const Real v = D[(ao_off[c] + kc) * nao + ao_off[d] + kd];
          const Real a = v < 0 ? -v : v;
          if (a > m) m = a;
        }
      maxD[c * ns + d] = m;
    }

  // triangular bra/ket pair list (a<=c), and its pair index a*ns+c into the
  // rectangular PairTable
  std::vector<int> trA, trC, trP;
  for (int a = 0; a < ns; ++a)
    for (int c = a; c < ns; ++c) {
      trA.push_back(a);
      trC.push_back(c);
      trP.push_back(a * ns + c);
    }
  const int nptri = static_cast<int>(trA.size());

  auto Dv = detail::to_device(D, static_cast<std::size_t>(nao) * nao, "intti::k8::D");
  auto Qv = detail::to_device(Qex, "intti::k8::Q");
  auto mDv = detail::to_device(maxD, "intti::k8::maxD");
  auto aov = detail::to_device(ao_off, "intti::k8::ao");
  std::vector<int> ls(ns);
  for (int i = 0; i < ns; ++i) ls[i] = shells[i].l;
  auto lv = detail::to_device(ls, "intti::k8::l");
  auto trAv = detail::to_device(trA, "intti::k8::trA");
  auto trCv = detail::to_device(trC, "intti::k8::trC");
  auto trPv = detail::to_device(trP, "intti::k8::trP");
  Kokkos::View<Real *> Kv("intti::k8::K", static_cast<std::size_t>(nao) * nao);
  Kokkos::deep_copy(Kv, Real(0));
  auto pv = tab.p;
  auto Pv = tab.P;
  auto eoffv = tab.e_off;
  auto Ev = tab.E;
  auto tv = grid.t_dev;
  auto wv = grid.w_dev;

  constexpr int KC1 = KLMAX + 1;
  constexpr int KNC = (KLMAX + 1) * (KLMAX + 2) / 2;

  Kokkos::parallel_for(
      "intti::k8::build", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nptri, nptri}),
      KOKKOS_LAMBDA(int iP, int iQ) {
        if (iQ > iP) return; // canonical: bra pair index >= ket pair index
        if (nranks > 1 && ((iP * nptri + iQ) % nranks) != rank) return;
        const int a = trAv(iP), c = trCv(iP), b = trAv(iQ), d = trCv(iQ);
        const int la = lv(a), lc = lv(c), lb = lv(b), ld = lv(d);
        const int nca = ncart(la), ncc = ncart(lc), ncb = ncart(lb), ncd = ncart(ld);
        // screen against the max |D| over the four contracted shell pairs
        const int p = trPv(iP), q = trPv(iQ);
        {
          Real md = mDv(c * ns + d);
          if (mDv(a * ns + d) > md) md = mDv(a * ns + d);
          if (mDv(c * ns + b) > md) md = mDv(c * ns + b);
          if (mDv(a * ns + b) > md) md = mDv(a * ns + b);
          if (Qv(p) * Qv(q) * md < tau) return;
        }
        int a3s[KNC][3], b3s[KNC][3], c3s[KNC][3], d3s[KNC][3];
        for (int k = 0; k < nca; ++k) cart_comp(la, k, a3s[k][0], a3s[k][1], a3s[k][2]);
        for (int k = 0; k < ncc; ++k) cart_comp(lc, k, c3s[k][0], c3s[k][1], c3s[k][2]);
        for (int k = 0; k < ncb; ++k) cart_comp(lb, k, b3s[k][0], b3s[k][1], b3s[k][2]);
        for (int k = 0; k < ncd; ++k) cart_comp(ld, k, d3s[k][0], d3s[k][1], d3s[k][2]);

        // four symmetry-distinct density contractions of the quartet (ac|bd):
        //   Acc1[ka,kb] += W D_cd,  Acc2[kc,kb] += W D_ad,
        //   Acc3[ka,kd] += W D_cb,  Acc4[kc,kd] += W D_ab
        Real Acc1[KNC * KNC], Acc2[KNC * KNC], Acc3[KNC * KNC], Acc4[KNC * KNC];
        for (int i = 0; i < nca * ncb; ++i) Acc1[i] = 0;
        for (int i = 0; i < ncc * ncb; ++i) Acc2[i] = 0;
        for (int i = 0; i < nca * ncd; ++i) Acc3[i] = 0;
        for (int i = 0; i < ncc * ncd; ++i) Acc4[i] = 0;

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
          for (int ta = 0; ta <= Ktail; ++ta)
            for (int tb = 0; tb <= Ktail - ta; ++tb)
              for (int tcc = 0; tcc <= Ktail - ta - tb; ++tcc) {
                tsa[ntail] = ta;
                tsb[ntail] = tb;
                tsc[ntail] = tcc;
                twf[ntail] = bcoef[ta + tb + tcc] * invf[ta] * invf[tb] * invf[tcc];
                ntail++;
              }
        }
        const Real pp = pv(p), pq = pv(q);
        Real X[3];
        for (int dd = 0; dd < 3; ++dd) X[dd] = Pv(p, dd) - Pv(q, dd);
        const int np_e = la + lc + 1, nq_e = lb + ld + 1;
        const int eszp = (la + 1) * (lc + 1) * np_e;
        const int eszq = (lb + 1) * (ld + 1) * nq_e;
        const int nB = np_e + nq_e - 1;
        const int nsweep = nt + ntail;
        for (int it = 0; it < nsweep; ++it) {
          Real theta, wt, prefd;
          int sh[3] = {0, 0, 0};
          if (it < nt) {
            const Real t = tv(it);
            const Real Dden = pp * pq + t * t * (pp + pq);
            theta = t * t * pp * pq / Dden;
            prefd = pi / sqrt_(Dden);
            wt = wv(it);
          } else {
            const int j = it - nt;
            sh[0] = 2 * tsa[j];
            sh[1] = 2 * tsb[j];
            sh[2] = 2 * tsc[j];
            theta = pp * pq / (pp + pq);
            prefd = sqrt_(pi / (pp + pq));
            wt = twf[j];
          }
          Real g[3][KC1 * KC1 * KC1 * KC1];
          Real B[4 * KLMAX + 2 * TAIL_KMAX + 1];
          for (int dd = 0; dd < 3; ++dd) {
            hermite_b(nB - 1 + sh[dd], theta, X[dd], B);
            const Real *Ep = &Ev(eoffv(p) + dd * eszp);
            const Real *Eq = &Ev(eoffv(q) + dd * eszq);
            for (int ia = 0; ia <= la; ++ia)
              for (int ic = 0; ic <= lc; ++ic)
                for (int ib = 0; ib <= lb; ++ib)
                  for (int id = 0; id <= ld; ++id) {
                    Real s = 0;
                    for (int t = 0; t <= ia + ic; ++t) {
                      const Real ep = Ep[(ia * (lc + 1) + ic) * np_e + t];
                      for (int u = 0; u <= ib + id; ++u) {
                        const Real term =
                            ep * Eq[(ib * (ld + 1) + id) * nq_e + u] * B[t + u + sh[dd]];
                        s += u % 2 ? -term : term;
                      }
                    }
                    g[dd][((ia * (lc + 1) + ic) * (lb + 1) + ib) * (ld + 1) + id] = prefd * s;
                  }
          }
          // accumulate the four contractions; W is the same per-node quartet
          for (int ka = 0; ka < nca; ++ka)
            for (int kc = 0; kc < ncc; ++kc)
              for (int kb = 0; kb < ncb; ++kb)
                for (int kd = 0; kd < ncd; ++kd) {
                  Real Wc = wt;
                  for (int dd = 0; dd < 3; ++dd)
                    Wc *= g[dd][((a3s[ka][dd] * (lc + 1) + c3s[kc][dd]) * (lb + 1) +
                                 b3s[kb][dd]) *
                                    (ld + 1) +
                                d3s[kd][dd]];
                  Acc1[ka * ncb + kb] += Wc * Dv((aov(c) + kc) * nao + aov(d) + kd);
                  Acc2[kc * ncb + kb] += Wc * Dv((aov(a) + ka) * nao + aov(d) + kd);
                  Acc3[ka * ncd + kd] += Wc * Dv((aov(c) + kc) * nao + aov(b) + kb);
                  Acc4[kc * ncd + kd] += Wc * Dv((aov(a) + ka) * nao + aov(b) + kb);
                }
        }

        // scatter the deduplicated 8-element orbit. Each op is
        //   K[canon[t0], canon[t2]] += (accumulator, possibly transposed).
        const int canon[4] = {a, c, b, d};
        const int tupleSlots[8][4] = {{0, 1, 2, 3}, {2, 3, 0, 1}, {1, 0, 2, 3}, {2, 3, 1, 0},
                                      {0, 1, 3, 2}, {3, 2, 0, 1}, {1, 0, 3, 2}, {3, 2, 1, 0}};
        const int accOf[8] = {1, 1, 2, 2, 3, 3, 4, 4};
        const bool trOf[8] = {false, true, false, true, false, true, false, true};
        for (int op = 0; op < 8; ++op) {
          const int s0 = canon[tupleSlots[op][0]], s1 = canon[tupleSlots[op][1]];
          const int s2 = canon[tupleSlots[op][2]], s3 = canon[tupleSlots[op][3]];
          bool dup = false; // drop ops whose shell 4-tuple already appeared
          for (int e = 0; e < op; ++e)
            if (canon[tupleSlots[e][0]] == s0 && canon[tupleSlots[e][1]] == s1 &&
                canon[tupleSlots[e][2]] == s2 && canon[tupleSlots[e][3]] == s3)
              dup = true;
          if (dup) continue;
          const int R = s0, C = s2;
          const int acc = accOf[op];
          const bool tr = trOf[op];
          const Real *A = acc == 1 ? Acc1 : acc == 2 ? Acc2 : acc == 3 ? Acc3 : Acc4;
          const int coldim = (acc == 1 || acc == 2) ? ncb : ncd;
          const int ncr = ncart(lv(R)), ncc2 = ncart(lv(C));
          for (int kr = 0; kr < ncr; ++kr)
            for (int kx = 0; kx < ncc2; ++kx) {
              const Real val = tr ? A[kx * coldim + kr] : A[kr * coldim + kx];
              Kokkos::atomic_add(&Kv((aov(R) + kr) * nao + aov(C) + kx), val);
            }
        }
      });

  auto hK = Kokkos::create_mirror_view(Kv);
  Kokkos::deep_copy(hK, Kv);
  for (std::size_t i = 0; i < static_cast<std::size_t>(nao) * nao; ++i)
    K[i] = hK(i);
}

} // namespace detail

} // namespace intti
