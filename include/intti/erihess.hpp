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

#include "erigrad.hpp" // detail::comp_index, detail::eri_block4, permute_block, eri_perms
#include "fock.hpp"
#include "gto.hpp"
#include "tgrid.hpp"

namespace intti {

/// d^2 E_2e / dR_{p,e} dR_{q,f} for
///   E_2e = 1/2 sum D_mn D_ls (mn|ls) - 1/4 sum D_ml D_ns (mn|ls),
/// as a (3 ns) x (3 ns) row-major matrix, index 3*shell + Cartesian. D is the
/// nao x nao symmetric AO density (row-major). tau > 0 enables Schwarz +
/// density screening (tau = 0, the default, is exact).
template <class Real>
std::vector<Real> two_electron_hessian(const ShellBasis<Real> &basis, const Real *D,
                                       const TGrid<Real> &grid, Real tau = Real(0)) {
  const int ns = static_cast<int>(basis.shells.size());
  const int nao = basis.nao, dim = 3 * ns;
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
        Real m = 0;
        for (int ki = 0; ki < ncart(basis.shells[i].l); ++ki)
          for (int kj = 0; kj < ncart(basis.shells[j].l); ++kj)
            m = std::max(m, std::abs(Dm(basis.ao_off[i] + ki, basis.ao_off[j] + kj)));
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
            const Real coeff =
                Real(0.5) * Dm(off[0] + ka, off[1] + kb) * Dm(off[2] + kc, off[3] + kd) -
                Real(0.25) * Dm(off[0] + ka, off[2] + kc) * Dm(off[1] + kb, off[3] + kd);
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
                    if (d2 != Real(0)) Hadd(mem[p], e, mem[q], fdir, coeff * d2);
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
            const int offm[4] = {basis.ao_off[mem[0]], basis.ao_off[mem[1]],
                                 basis.ao_off[mem[2]], basis.ao_off[mem[3]]};

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

} // namespace intti
