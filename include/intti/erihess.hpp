// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Two-electron energy Hessian -- the black-box frequencies piece. The MD
// centre-shift d/dR_{p,e} = 2 alpha_p [.+1_e] - m_{p,e} [.-1_e] that gives the
// gradient (erigrad.hpp) is applied twice for the second derivative:
//   * same shell (p == q): needs the quartet at l_p+2, l_p, l_p-2;
//   * different shells (p != q): needs it at l_p+/-1 and l_q+/-1 jointly.
// Contracted with the closed-shell two-particle density this gives
// d^2 E_2e / dR_{p,e} dR_{q,f}, returned as a (3 nshell) x (3 nshell) matrix
// (rows/cols ordered shell-major, Cartesian-minor). Matrix-level: density in,
// Hessian out; quartets stay internal.
//
// Permutational symmetry (see erigrad.hpp): loop canonical shell quartets only
// and replay the *unchanged* contraction on every distinct orbit member,
// sharing one canonical block cache across the orbit and fetching permuted
// views per member (~8x fewer quartet evals). The canonical cache also replaces
// the previous per-quartet std::map with a plain offset->block store.

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <vector>

#include "batch.hpp"   // PairTable, make_batch, eri_quartets
#include "contracted.hpp" // detail::ShellFanout
#include "device.hpp"  // detail::to_device / to_host
#include "erigrad.hpp" // detail::comp_index, detail::eri_block4, pair_schwarz_margin
#include "fock.hpp"
#include "gto.hpp"
#include "tgrid.hpp"

namespace intti {

namespace detail {

// device (GPU) 2e Hessian: batched-quartet consumer WITH the 8-fold symmetry,
// as in the gradient. Only CANONICAL quartets are enumerated, and each one's 33
// per-position l-offset patterns (single +-2, and pairs of +-1; those with all
// l+o >= 0) are batched once and looked up by a 625-entry key table. The
// contraction is then replayed on each distinct orbit member, reading the
// canonical blocks through the permutation instead of materialising them.
//
// One order up from the gradient, BOTH the offset pattern and the component
// indices permute: since member slot t is canonical slot pm[t],
//   oc[pm[t]] = o[t]   and   ic[pm[t]] = comp_index(..., m[t]).
// The permuted offset is always still in the pattern list, because permuting a
// "single +-2 at a slot" or "+-1 at two slots" pattern yields another of the
// same family. tau > 0 applies the host Schwarz(+2 margin)+density screen.
// l <= LMAX-2.
template <class Real>
std::vector<Real> two_electron_hessian_dev(const ShellBasis<Real> &basis,
                                           const ShellFanout<Real> &fan, const Real *D,
                                           const TGrid<Real> &grid, Real tau) {
  const int ns = static_cast<int>(basis.shells.size());
  const int nao = fan.nao, dim = 3 * fan.nsh;
  std::vector<int> hL(ns), hOff(ns);
  std::vector<Real> hAl(ns);
  for (int i = 0; i < ns; ++i) {
    hL[i] = basis.shells[i].l;
    hOff[i] = fan.base[i];
    hAl[i] = basis.shells[i].alpha;
  }
  auto Dm = [&](int i, int j) { return D[static_cast<std::size_t>(i) * nao + j]; };
  std::vector<Real> Q, Dmax;
  if (tau > Real(0)) {
    Q.assign(static_cast<std::size_t>(ns) * ns, Real(0));
    Dmax.assign(static_cast<std::size_t>(ns) * ns, Real(0));
    for (int i = 0; i < ns; ++i)
      for (int j = 0; j < ns; ++j) {
        PrimitiveShell<Real> ai = basis.shells[i], aj = basis.shells[j];
        ai.l += 2;
        aj.l += 2;
        Q[i * ns + j] = pair_schwarz_margin(ai, aj, grid);
        // EFFECTIVE density bound: the primitive pair (i,j) enters the energy
        // weighted by its contraction coefficients, so the bound must carry them.
        const int nci = ncart(basis.shells[i].l), ncj = ncart(basis.shells[j].l);
        Real m = 0;
        for (int ci = 0; ci < fan.nctr[i]; ++ci)
          for (int cj = 0; cj < fan.nctr[j]; ++cj) {
            const Real wij = std::abs(fan.w[fan.coff[i] + ci] * fan.w[fan.coff[j] + cj]);
            for (int ki = 0; ki < nci; ++ki)
              for (int kj = 0; kj < ncj; ++kj)
                m = std::max(m, wij * std::abs(Dm(fan.base[i] + ci * nci + ki,
                                                  fan.base[j] + cj * ncj + kj)));
          }
        Dmax[i * ns + j] = m;
      }
  }
  // 33 offset patterns: {0}, single +-2 at each p, and each pair {p<q} x 4 signs
  std::vector<std::array<int, 4>> pats;
  pats.push_back({0, 0, 0, 0});
  for (int p = 0; p < 4; ++p) {
    std::array<int, 4> a{}, b{};
    a[p] = 2; b[p] = -2;
    pats.push_back(a); pats.push_back(b);
  }
  const int sgn[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
  for (int p = 0; p < 4; ++p)
    for (int q = p + 1; q < 4; ++q)
      for (int s = 0; s < 4; ++s) {
        std::array<int, 4> o{};
        o[p] = sgn[s][0];
        o[q] = sgn[s][1];
        pats.push_back(o);
      }
  const int NP = static_cast<int>(pats.size()); // 33
  auto enc = [](const std::array<int, 4> &o) {
    return (((o[0] + 2) * 5 + (o[1] + 2)) * 5 + (o[2] + 2)) * 5 + (o[3] + 2);
  };
  std::vector<int> idxkey(625, -1);
  for (int pi = 0; pi < NP; ++pi) idxkey[enc(pats[pi])] = pi;

  std::vector<ShellPair<Real>> plist;
  auto add_pair = [&](int si, int di, int sj, int dj) {
    PrimitiveShell<Real> a = basis.shells[si], b = basis.shells[sj];
    a.l += di;
    b.l += dj;
    plist.push_back(make_pair(a, b));
    return static_cast<int>(plist.size()) - 1;
  };
  std::vector<std::pair<int, int>> quartets;
  std::vector<int> ca, cbv_, ccv_, cdv_;   // canonical shells per canonical quartet
  std::vector<int> cqEnt;                  // NP pattern entries per canonical quartet
  std::vector<int> jq, jp0, jp1, jp2, jp3; // per job: canonical quartet + permutation
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b <= a; ++b) {
      const int Pab = a * ns + b;
      for (int c = 0; c < ns; ++c)
        for (int d = 0; d <= c; ++d) {
          if (c * ns + d > Pab) continue;
          const int canon[4] = {a, b, c, d};
          if (tau > Real(0)) {
            Real dw = 0;
            for (int i = 0; i < 4; ++i)
              for (int jj = 0; jj < 4; ++jj)
                dw = std::max(dw, Dmax[canon[i] * ns + canon[jj]]);
            const Real almax = std::max({hAl[a], hAl[b], hAl[c], hAl[d]});
            if (4 * almax * almax * Q[a * ns + b] * Q[c * ns + d] * dw * dw < tau) continue;
          }
          const int L[4] = {hL[a], hL[b], hL[c], hL[d]};
          const int qidx = static_cast<int>(ca.size());
          ca.push_back(a); cbv_.push_back(b); ccv_.push_back(c); cdv_.push_back(d);
          for (int pi = 0; pi < NP; ++pi) {
            const auto &o = pats[pi];
            bool ok = true;
            for (int p = 0; p < 4; ++p)
              if (L[p] + o[p] < 0) ok = false;
            if (!ok) { cqEnt.push_back(-1); continue; }
            cqEnt.push_back(static_cast<int>(quartets.size()));
            quartets.push_back({add_pair(canon[0], o[0], canon[1], o[1]),
                                add_pair(canon[2], o[2], canon[3], o[3])});
          }
          int seen[8][4];
          int nseen = 0;
          for (int g = 0; g < 8; ++g) {
            const int *pm = detail::eri_perms[g];
            const int mem[4] = {canon[pm[0]], canon[pm[1]], canon[pm[2]], canon[pm[3]]};
            bool dup = false;
            for (int t = 0; t < nseen && !dup; ++t)
              dup = seen[t][0] == mem[0] && seen[t][1] == mem[1] && seen[t][2] == mem[2] &&
                    seen[t][3] == mem[3];
            if (dup) continue;
            for (int t = 0; t < 4; ++t) seen[nseen][t] = mem[t];
            ++nseen;
            jq.push_back(qidx);
            jp0.push_back(pm[0]); jp1.push_back(pm[1]);
            jp2.push_back(pm[2]); jp3.push_back(pm[3]);
          }
        }
    }
  const int njob = static_cast<int>(jq.size());
  auto tab = make_pair_table(plist);
  auto batch = make_batch(tab, quartets);
  QuartetWorkspace<Real> ws;
  Kokkos::View<Real *> out("intti::h2::out", batch.nout_total);
  eri_quartets(tab, batch, grid, out, ws);

  auto shL = to_device(hL, "intti::h2::L"), shOff = to_device(hOff, "intti::h2::off");
  auto shAlv = to_device(hAl, "intti::h2::al");
  auto Dd = to_device(D, static_cast<std::size_t>(nao) * nao, "intti::h2::D");
  auto cav = to_device(ca, "h2::ca"), cbv = to_device(cbv_, "h2::cb");
  auto ccv = to_device(ccv_, "h2::cc"), cdv = to_device(cdv_, "h2::cd");
  auto jqv = to_device(jq, "h2::jq");
  auto p0v = to_device(jp0, "h2::p0"), p1v = to_device(jp1, "h2::p1");
  auto p2v = to_device(jp2, "h2::p2"), p3v = to_device(jp3, "h2::p3");
  auto dEnt = to_device(cqEnt, "intti::h2::ent"), didx = to_device(idxkey, "intti::h2::idx");
  auto fnc = to_device(fan.nctr, "h2::nctr"), fco = to_device(fan.coff, "h2::coff");
  auto fpar = to_device(fan.parent, "h2::parent");
  auto fw = to_device(fan.w, "h2::w");
  auto offv = batch.out_offset;
  Kokkos::View<Real *> Hd("intti::h2::H", static_cast<std::size_t>(dim) * dim);
  Kokkos::parallel_for(
      "intti::h2::digest", Kokkos::RangePolicy<>(0, njob), KOKKOS_LAMBDA(int j) {
        const int cq = jqv(j);
        const int canon[4] = {cav(cq), cbv(cq), ccv(cq), cdv(cq)};
        const int pm[4] = {p0v(j), p1v(j), p2v(j), p3v(j)};
        int Lc[4];
        for (int t = 0; t < 4; ++t) Lc[t] = shL(canon[t]);
        int mem[4], L[4], off[4];
        Real al[4];
        for (int t = 0; t < 4; ++t) {
          mem[t] = canon[pm[t]];
          L[t] = Lc[pm[t]];
          off[t] = shOff(mem[t]);
          al[t] = shAlv(mem[t]);
        }
        auto Dmk = [&](int i, int k) { return Dd(static_cast<std::size_t>(i) * nao + k); };
        // rawval takes MEMBER-slot offsets and components; BOTH are mapped onto
        // the canonical block, since member slot t is canonical slot pm[t]:
        //   oc[pm[t]] = o[t]   and   ic[pm[t]] = comp_index(..., m[t]).
        auto rawval = [&](const int o[4], const int m[4][3]) -> Real {
          int oc[4];
          for (int t = 0; t < 4; ++t) oc[pm[t]] = o[t];
          const int key = (((oc[0] + 2) * 5 + (oc[1] + 2)) * 5 + (oc[2] + 2)) * 5 + (oc[3] + 2);
          const int pi = didx(key);
          if (pi < 0) return Real(0);
          const int ent = dEnt(cq * NP + pi);
          if (ent < 0) return Real(0);
          int nn[4], id[4];
          for (int t = 0; t < 4; ++t) {
            const int lt = L[t] + o[t];
            if (m[t][0] < 0 || m[t][1] < 0 || m[t][2] < 0 || m[t][0] + m[t][1] + m[t][2] != lt)
              return Real(0);
            nn[pm[t]] = ncart(lt);
            id[pm[t]] = comp_index(lt, m[t][0], m[t][1]);
          }
          return out(offv(ent) +
                     (((static_cast<std::size_t>(id[0]) * nn[1] + id[1]) * nn[2] + id[2]) * nn[3] +
                      id[3]));
        };
        const int nb = ncart(L[1]), ncc = ncart(L[2]), nd = ncart(L[3]);
        int bm[4][3];
        for (int ka = 0; ka < ncart(L[0]); ++ka) {
          cart_comp(L[0], ka, bm[0][0], bm[0][1], bm[0][2]);
          for (int kb = 0; kb < nb; ++kb) {
            cart_comp(L[1], kb, bm[1][0], bm[1][1], bm[1][2]);
            for (int kc = 0; kc < ncc; ++kc) {
              cart_comp(L[2], kc, bm[2][0], bm[2][1], bm[2][2]);
              for (int kd = 0; kd < nd; ++kd) {
                cart_comp(L[3], kd, bm[3][0], bm[3][1], bm[3][2]);
                // Contraction fan-out. The Hessian's shell indices and the
                // derivative value d2 below depend only on the primitive shells,
                // never on which contracted function of a shell is meant, so the
                // whole nctr^4 sum collapses into this ONE density prefactor --
                // computed per Cartesian quartet, not per (e,f) pair. All trip
                // counts are 1 for a primitive basis.
                Real coeff = 0;
                const int nca0 = ncart(L[0]);
                for (int cA = 0; cA < fnc(mem[0]); ++cA) {
                  const Real wA = fw(fco(mem[0]) + cA);
                  const int I = off[0] + cA * nca0 + ka;
                  for (int cB = 0; cB < fnc(mem[1]); ++cB) {
                    const Real wAB = wA * fw(fco(mem[1]) + cB);
                    const int Jj = off[1] + cB * nb + kb;
                    for (int cC = 0; cC < fnc(mem[2]); ++cC) {
                      const Real wABC = wAB * fw(fco(mem[2]) + cC);
                      const int Kk = off[2] + cC * ncc + kc;
                      for (int cD = 0; cD < fnc(mem[3]); ++cD) {
                        const Real w = wABC * fw(fco(mem[3]) + cD);
                        const int Ll = off[3] + cD * nd + kd;
                        coeff += w * (Real(0.5) * Dmk(I, Jj) * Dmk(Kk, Ll) -
                                      Real(0.25) * Dmk(I, Kk) * Dmk(Jj, Ll));
                      }
                    }
                  }
                }
                if (coeff == Real(0)) continue;
                auto mset = [&](int p, const int mp[3], int q, const int mq[3], int mm[4][3]) {
                  for (int r = 0; r < 4; ++r)
                    for (int t = 0; t < 3; ++t) mm[r][t] = bm[r][t];
                  for (int t = 0; t < 3; ++t) mm[p][t] = mp[t];
                  if (q != p)
                    for (int t = 0; t < 3; ++t) mm[q][t] = mq[t];
                };
                for (int p = 0; p < 4; ++p)
                  for (int q = 0; q < 4; ++q) {
                    const Real ap = al[p], aq = al[q];
                    for (int e = 0; e < 3; ++e)
                      for (int fdir = 0; fdir < 3; ++fdir) {
                        Real d2 = 0;
                        int mm[4][3];
                        if (p == q) {
                          const int *mp = bm[p];
                          const int de = (e == fdir) ? 1 : 0;
                          int T[4][3];
                          for (int t = 0; t < 3; ++t)
                            T[0][t] = T[1][t] = T[2][t] = T[3][t] = mp[t];
                          T[0][e] += 1; T[0][fdir] += 1;
                          T[1][fdir] += 1; T[1][e] -= 1;
                          T[2][e] += 1; T[2][fdir] -= 1;
                          T[3][e] -= 1; T[3][fdir] -= 1;
                          const int o2[4] = {p == 0 ? 2 : 0, p == 1 ? 2 : 0, p == 2 ? 2 : 0,
                                             p == 3 ? 2 : 0};
                          const int o0[4] = {0, 0, 0, 0};
                          const int om2[4] = {p == 0 ? -2 : 0, p == 1 ? -2 : 0, p == 2 ? -2 : 0,
                                              p == 3 ? -2 : 0};
                          mset(p, T[0], q, T[0], mm);
                          d2 += 2 * ap * (2 * ap * rawval(o2, mm));
                          mset(p, T[1], q, T[1], mm);
                          d2 += 2 * ap * (-(Real(mp[e]) + de) * rawval(o0, mm));
                          mset(p, T[2], q, T[2], mm);
                          d2 += -Real(mp[fdir]) * (2 * ap * rawval(o0, mm));
                          mset(p, T[3], q, T[3], mm);
                          d2 += -Real(mp[fdir]) * (-(Real(mp[e]) - de) * rawval(om2, mm));
                        } else {
                          const int *mp = bm[p], *mq = bm[q];
                          int Pe1[3], Pe0[3], Qf1[3], Qf0[3];
                          for (int t = 0; t < 3; ++t) {
                            Pe1[t] = mp[t]; Pe0[t] = mp[t];
                            Qf1[t] = mq[t]; Qf0[t] = mq[t];
                          }
                          Pe1[e] += 1; Pe0[e] -= 1;
                          Qf1[fdir] += 1; Qf0[fdir] -= 1;
                          auto oo = [&](int dp, int dq, int out4[4]) {
                            for (int t = 0; t < 4; ++t) out4[t] = 0;
                            out4[p] += dp; out4[q] += dq;
                          };
                          int oarr[4];
                          mset(p, Pe1, q, Qf1, mm); oo(1, 1, oarr);
                          d2 += 4 * ap * aq * rawval(oarr, mm);
                          mset(p, Pe1, q, Qf0, mm); oo(1, -1, oarr);
                          d2 += -2 * ap * Real(mq[fdir]) * rawval(oarr, mm);
                          mset(p, Pe0, q, Qf1, mm); oo(-1, 1, oarr);
                          d2 += -Real(mp[e]) * 2 * aq * rawval(oarr, mm);
                          mset(p, Pe0, q, Qf0, mm); oo(-1, -1, oarr);
                          d2 += Real(mp[e]) * Real(mq[fdir]) * rawval(oarr, mm);
                        }
                        if (d2 != Real(0))
                          Kokkos::atomic_add(
                              &Hd((3 * fpar(mem[p]) + e) * static_cast<std::size_t>(dim) +
                                  3 * fpar(mem[q]) + fdir),
                              coeff * d2);
                      }
                  }
              }
            }
          }
        }
      });
  return to_host(Hd);
}

} // namespace detail

/// d^2 E_2e / dR_{p,e} dR_{q,f} for
///   E_2e = 1/2 sum D_mn D_ls (mn|ls) - 1/4 sum D_ml D_ns (mn|ls),
/// as a (3 ns) x (3 ns) row-major matrix, index 3*shell + Cartesian. D is the
/// nao x nao symmetric AO density (row-major). tau > 0 enables Schwarz +
/// density screening (tau = 0, the default, is exact).
template <class Real>
std::vector<Real> two_electron_hessian_impl(const ShellBasis<Real> &basis,
                                            const detail::ShellFanout<Real> &fan,
                                            const Real *D, const TGrid<Real> &grid, Real tau) {
  if constexpr (kokkos_scalar_v<Real>) {
    bool ok = true;
    for (const auto &s : basis.shells)
      if (s.l > LMAX - 2) ok = false; // second derivative promotes by 2
    if (ok) return detail::two_electron_hessian_dev(basis, fan, D, grid, tau);
  }
  const int ns = static_cast<int>(basis.shells.size());
  const int nao = fan.nao, dim = 3 * fan.nsh;
  std::vector<Real> H(static_cast<std::size_t>(dim) * dim, Real(0));
  auto Dm = [&](int i, int j) { return D[static_cast<std::size_t>(i) * nao + j]; };
  auto Hadd = [&](int p, int e, int q, int f, Real v) {
    H[(3 * p + e) * static_cast<std::size_t>(dim) + 3 * q + f] += v;
  };

  // per-shell-pair Schwarz bounds (with a +2 promotion margin for the Hessian's
  // second derivative) and density-block maxima, used only when tau > 0.
  std::vector<Real> Q, Dmax;
  if (tau > Real(0)) {
    Q.assign(static_cast<std::size_t>(ns) * ns, Real(0));
    Dmax.assign(static_cast<std::size_t>(ns) * ns, Real(0));
    for (int i = 0; i < ns; ++i)
      for (int j = 0; j < ns; ++j) {
        // margin: promote each shell by up to 2 for the second-derivative reach
        PrimitiveShell<Real> ai = basis.shells[i], aj = basis.shells[j];
        ai.l += 2;
        aj.l += 2;
        Q[i * ns + j] = detail::pair_schwarz_margin(ai, aj, grid);
        // effective (contraction-weighted) density bound; see the device path
        const int nci = ncart(basis.shells[i].l), ncj = ncart(basis.shells[j].l);
        Real m = 0;
        for (int ci = 0; ci < fan.nctr[i]; ++ci)
          for (int cj = 0; cj < fan.nctr[j]; ++cj) {
            const Real wij = std::abs(fan.w[fan.coff[i] + ci] * fan.w[fan.coff[j] + cj]);
            for (int ki = 0; ki < nci; ++ki)
              for (int kj = 0; kj < ncj; ++kj)
                m = std::max(m, wij * std::abs(Dm(fan.base[i] + ci * nci + ki,
                                                  fan.base[j] + cj * ncj + kj)));
          }
        Dmax[i * ns + j] = m;
      }
  }

  // contract one (member) shell quartet: the original, unmodified second-
  // derivative body. getblock(o_m) returns the member-layout ERI block at
  // per-position l-offset o_m (or nullptr if any promoted l < 0).
  auto contribute = [&](const int mem[4], const int L[4], const Real alp[4], const int off[4],
                        auto &&getblock) {
    const int nb = ncart(L[1]), nc = ncart(L[2]), nd = ncart(L[3]);
    auto rawval = [&](const std::array<int, 4> &o, const int m[4][3]) -> Real {
      const auto *blk = getblock(o);
      if (!blk) return Real(0);
      int nn[4], id[4];
      for (int p = 0; p < 4; ++p) {
        const int lp = L[p] + o[p];
        if (m[p][0] < 0 || m[p][1] < 0 || m[p][2] < 0 || m[p][0] + m[p][1] + m[p][2] != lp)
          return Real(0);
        nn[p] = ncart(lp);
        id[p] = detail::comp_index(lp, m[p][0], m[p][1]);
      }
      return (*blk)[((static_cast<std::size_t>(id[0]) * nn[1] + id[1]) * nn[2] + id[2]) * nn[3] +
                    id[3]];
    };

    int bm[4][3];
    for (int ka = 0; ka < ncart(L[0]); ++ka) {
      cart_comp(L[0], ka, bm[0][0], bm[0][1], bm[0][2]);
      for (int kb = 0; kb < nb; ++kb) {
        cart_comp(L[1], kb, bm[1][0], bm[1][1], bm[1][2]);
        for (int kc = 0; kc < nc; ++kc) {
          cart_comp(L[2], kc, bm[2][0], bm[2][1], bm[2][2]);
          for (int kd = 0; kd < nd; ++kd) {
            cart_comp(L[3], kd, bm[3][0], bm[3][1], bm[3][2]);
            // Contraction fan-out; see the device path. The whole nctr^4 sum
            // collapses into this one density prefactor because neither d2 nor
            // the Hessian's shell indices depend on the contracted-function index.
            Real coeff = 0;
            const int nca0 = ncart(L[0]);
            for (int cA = 0; cA < fan.nctr[mem[0]]; ++cA) {
              const Real wA = fan.w[fan.coff[mem[0]] + cA];
              const int I = off[0] + cA * nca0 + ka;
              for (int cB = 0; cB < fan.nctr[mem[1]]; ++cB) {
                const Real wAB = wA * fan.w[fan.coff[mem[1]] + cB];
                const int Jj = off[1] + cB * nb + kb;
                for (int cC = 0; cC < fan.nctr[mem[2]]; ++cC) {
                  const Real wABC = wAB * fan.w[fan.coff[mem[2]] + cC];
                  const int Kk = off[2] + cC * nc + kc;
                  for (int cD = 0; cD < fan.nctr[mem[3]]; ++cD) {
                    const Real w = wABC * fan.w[fan.coff[mem[3]] + cD];
                    const int Ll = off[3] + cD * nd + kd;
                    coeff += w * (Real(0.5) * Dm(I, Jj) * Dm(Kk, Ll) -
                                  Real(0.25) * Dm(I, Kk) * Dm(Jj, Ll));
                  }
                }
              }
            }
            if (coeff == Real(0)) continue;

            auto mset = [&](int p, const int mp[3], int q, const int mq[3], int mm[4][3]) {
              for (int r = 0; r < 4; ++r)
                for (int t = 0; t < 3; ++t) mm[r][t] = bm[r][t];
              for (int t = 0; t < 3; ++t) mm[p][t] = mp[t];
              if (q != p)
                for (int t = 0; t < 3; ++t) mm[q][t] = mq[t];
            };

            for (int p = 0; p < 4; ++p)
              for (int q = 0; q < 4; ++q) {
                const Real ap = alp[p], aq = alp[q];
                for (int e = 0; e < 3; ++e)
                  for (int fdir = 0; fdir < 3; ++fdir) {
                    Real d2 = 0;
                    int mm[4][3];
                    if (p == q) {
                      const int *mp = bm[p];
                      const int de = (e == fdir) ? 1 : 0;
                      int T[4][3];
                      for (int t = 0; t < 3; ++t)
                        T[0][t] = T[1][t] = T[2][t] = T[3][t] = mp[t];
                      T[0][e] += 1; T[0][fdir] += 1;
                      T[1][fdir] += 1; T[1][e] -= 1;
                      T[2][e] += 1; T[2][fdir] -= 1;
                      T[3][e] -= 1; T[3][fdir] -= 1;
                      const std::array<int, 4> o2 = {p == 0 ? 2 : 0, p == 1 ? 2 : 0,
                                                     p == 2 ? 2 : 0, p == 3 ? 2 : 0};
                      const std::array<int, 4> o0 = {0, 0, 0, 0};
                      const std::array<int, 4> om2 = {p == 0 ? -2 : 0, p == 1 ? -2 : 0,
                                                      p == 2 ? -2 : 0, p == 3 ? -2 : 0};
                      mset(p, T[0], q, T[0], mm);
                      d2 += 2 * ap * (2 * ap * rawval(o2, mm));
                      mset(p, T[1], q, T[1], mm);
                      d2 += 2 * ap * (-(Real(mp[e]) + de) * rawval(o0, mm));
                      mset(p, T[2], q, T[2], mm);
                      d2 += -Real(mp[fdir]) * (2 * ap * rawval(o0, mm));
                      mset(p, T[3], q, T[3], mm);
                      d2 += -Real(mp[fdir]) * (-(Real(mp[e]) - de) * rawval(om2, mm));
                    } else {
                      const int *mp = bm[p], *mq = bm[q];
                      int Pe1[3], Pe0[3], Qf1[3], Qf0[3];
                      for (int t = 0; t < 3; ++t) {
                        Pe1[t] = mp[t]; Pe0[t] = mp[t];
                        Qf1[t] = mq[t]; Qf0[t] = mq[t];
                      }
                      Pe1[e] += 1; Pe0[e] -= 1;
                      Qf1[fdir] += 1; Qf0[fdir] -= 1;
                      auto oo = [&](int dp, int dq) {
                        std::array<int, 4> o = {0, 0, 0, 0};
                        o[p] += dp; o[q] += dq;
                        return o;
                      };
                      mset(p, Pe1, q, Qf1, mm);
                      d2 += 4 * ap * aq * rawval(oo(1, 1), mm);
                      mset(p, Pe1, q, Qf0, mm);
                      d2 += -2 * ap * Real(mq[fdir]) * rawval(oo(1, -1), mm);
                      mset(p, Pe0, q, Qf1, mm);
                      d2 += -Real(mp[e]) * 2 * aq * rawval(oo(-1, 1), mm);
                      mset(p, Pe0, q, Qf0, mm);
                      d2 += Real(mp[e]) * Real(mq[fdir]) * rawval(oo(-1, -1), mm);
                    }
                    if (d2 != Real(0))
                      Hadd(fan.parent[mem[p]], e, fan.parent[mem[q]], fdir, coeff * d2);
                  }
              }
          }
        }
      }
    }
  };

  // canonical shell quartets: a>=b, c>=d, pair(a,b) >= pair(c,d)
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b <= a; ++b) {
      const int Pab = a * ns + b;
      for (int cc = 0; cc < ns; ++cc)
        for (int dd = 0; dd <= cc; ++dd) {
          if (cc * ns + dd > Pab) continue;
          const int canon[4] = {a, b, cc, dd};
          const int Lc[4] = {basis.shells[a].l, basis.shells[b].l, basis.shells[cc].l,
                             basis.shells[dd].l};

          if (tau > Real(0)) {
            const Real qab = Q[a * ns + b], qcd = Q[cc * ns + dd];
            Real dw = 0;
            const int sh[4] = {a, b, cc, dd};
            for (int i = 0; i < 4; ++i)
              for (int j = 0; j < 4; ++j) dw = std::max(dw, Dmax[sh[i] * ns + sh[j]]);
            const Real almax = std::max({basis.shells[a].alpha, basis.shells[b].alpha,
                                         basis.shells[cc].alpha, basis.shells[dd].alpha});
            if (4 * almax * almax * qab * qcd * dw * dw < tau) continue;
          }

          // canonical block cache, shared across the orbit (one eval per offset)
          std::map<std::array<int, 4>, std::vector<Real>> ccache;
          auto canon_block = [&](const std::array<int, 4> &oc) -> const std::vector<Real> * {
            for (int p = 0; p < 4; ++p)
              if (Lc[p] + oc[p] < 0) return nullptr;
            auto it = ccache.find(oc);
            if (it == ccache.end()) {
              auto mk = [&](int p) {
                auto s = basis.shells[canon[p]];
                s.l += oc[p];
                return s;
              };
              it = ccache.emplace(oc, detail::eri_block4(mk(0), mk(1), mk(2), mk(3), grid)).first;
            }
            return &it->second;
          };

          int seen[8][4];
          int nseen = 0;
          for (int g = 0; g < 8; ++g) {
            const int *pm = detail::eri_perms[g];
            const int mem[4] = {canon[pm[0]], canon[pm[1]], canon[pm[2]], canon[pm[3]]};
            bool dup = false;
            for (int t = 0; t < nseen && !dup; ++t)
              dup = seen[t][0] == mem[0] && seen[t][1] == mem[1] && seen[t][2] == mem[2] &&
                    seen[t][3] == mem[3];
            if (dup) continue;
            for (int t = 0; t < 4; ++t) seen[nseen][t] = mem[t];
            ++nseen;

            const int Lm[4] = {Lc[pm[0]], Lc[pm[1]], Lc[pm[2]], Lc[pm[3]]};
            const Real alm[4] = {basis.shells[mem[0]].alpha, basis.shells[mem[1]].alpha,
                                 basis.shells[mem[2]].alpha, basis.shells[mem[3]].alpha};
            const int offm[4] = {fan.base[mem[0]], fan.base[mem[1]], fan.base[mem[2]],
                                 fan.base[mem[3]]};

            // member block accessor: permute the canonical block at the permuted
            // offset into member layout, cached per member offset.
            std::map<std::array<int, 4>, std::vector<Real>> mcache;
            auto getblock = [&, pm, Lm](const std::array<int, 4> &om) -> const std::vector<Real> * {
              std::array<int, 4> oc{};
              for (int j = 0; j < 4; ++j) oc[pm[j]] = om[j];
              const auto *cb = canon_block(oc);
              if (!cb) return nullptr;
              auto it = mcache.find(om);
              if (it == mcache.end()) {
                int dCc[4];
                for (int i = 0; i < 4; ++i) dCc[i] = ncart(Lc[i] + oc[i]);
                it = mcache.emplace(om, detail::permute_block(*cb, dCc, pm)).first;
              }
              return &it->second;
            };
            contribute(mem, Lm, alm, offm, getblock);
          }
        }
    }
  return H;
}

/// Two-electron contribution to the molecular Hessian, (3 nshell) x (3 nshell),
/// contracted with the closed-shell two-particle density built from D.
template <class Real>
std::vector<Real> two_electron_hessian(const ShellBasis<Real> &basis, const Real *D,
                                       const TGrid<Real> &grid, Real tau = Real(0)) {
  return two_electron_hessian_impl(basis, detail::identity_fanout(basis), D, grid, tau);
}

/// Same, over a generally-contracted basis. The kernel runs on the expanded
/// primitive shells; the contraction enters only through the density prefactor
/// and the primitive-to-contracted shell map, so every primitive quartet's
/// second derivatives are evaluated once and the result stays indexed by
/// CONTRACTED shell -- which is what a caller folds onto atoms.
template <class Real>
std::vector<Real> two_electron_hessian(const ContractedBasis<Real> &cb, const Real *D,
                                       const TGrid<Real> &grid, Real tau = Real(0)) {
  ShellBasis<Real> prims;
  auto fan = detail::expand_contracted(cb, prims);
  return two_electron_hessian_impl(prims, fan, D, grid, tau);
}

} // namespace intti
