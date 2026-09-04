// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Geometric gradient of the RI (density-fitting) Coulomb energy -- the force
// that matches the RI-J Fock build (ri.hpp), so an RI-SCF differentiates the
// surface it actually optimises on. With the fit coefficients gamma = M^{-1} d,
// d_P = sum_mn (mn|P) D_mn, M_PQ = (P|Q), the RI-J energy is
//   E_J = 1/2 d^T M^{-1} d = 1/2 sum_P d_P gamma_P,
// and (the metric inverse being stationary) its gradient is
//   dE_J/dx = sum_P gamma_P dd_P/dx  -  1/2 sum_PQ gamma_P gamma_Q dM_PQ/dx.
// Both terms need only derivatives of the 2-/3-centre Coulomb integrals, which
// are ghost-augmented quartets (ncenter.hpp), so the erigrad centre-shift
// (2 alpha [.+1] - m [.-1]) drives them; the zero-exponent ghost never moves.
// Matrix-level: density in, per-shell forces out; no quartet is exposed.

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <vector>

#include "blas.hpp"    // detail::gemm (row-major GEMM)
#include "erigrad.hpp" // detail::comp_index, detail::eri_block4
#include "fock.hpp"
#include "gto.hpp"
#include "ncenter.hpp" // detail::ghost_pair pattern (ghost shell)
#include "ri.hpp"      // detail::syevd
#include "tgrid.hpp"

namespace intti {

/// Per-shell forces of the RI energy: forb[shell] over the orbital basis,
/// faux[shell] over the auxiliary basis. The caller maps both to atoms.
template <class Real> struct RIGrad {
  std::vector<std::array<Real, 3>> forb, faux;
};

namespace detail {

/// Contribution of differentiating shell position `pos` (0..3) of the quartet
/// (s0 s1 | s2 s3) to a force vector out[3], contracted with a coefficient
/// coeff(k0,k1,k2,k3) over the four Cartesian component indices. Uses the MD
/// centre-shift d/dR = 2 alpha [.+1] - m [.-1] (erigrad). A zero-exponent ghost
/// must never be passed as `pos` (its derivative is zero and is simply skipped
/// by the caller).
template <class Real, class CoeffFn>
void quartet_pos_grad(const PrimitiveShell<Real> &s0, const PrimitiveShell<Real> &s1,
                      const PrimitiveShell<Real> &s2, const PrimitiveShell<Real> &s3,
                      int pos, const TGrid<Real> &grid, CoeffFn coeff, Real out[3]) {
  const PrimitiveShell<Real> sh[4] = {s0, s1, s2, s3};
  const int L[4] = {s0.l, s1.l, s2.l, s3.l};
  const int nc[4] = {ncart(L[0]), ncart(L[1]), ncart(L[2]), ncart(L[3])};
  const int lp = L[pos];
  const Real ap = sh[pos].alpha;
  auto promote = [&](int dl) {
    PrimitiveShell<Real> s[4] = {s0, s1, s2, s3};
    s[pos].l += dl;
    return detail::eri_block4(s[0], s[1], s[2], s[3], grid);
  };
  auto plus = promote(1);
  std::vector<Real> minus;
  if (lp >= 1) minus = promote(-1);
  int npl[4], nmi[4];
  for (int i = 0; i < 4; ++i) {
    npl[i] = nc[i];
    nmi[i] = nc[i];
  }
  npl[pos] = ncart(lp + 1);
  if (lp >= 1) nmi[pos] = ncart(lp - 1);
  auto idx = [](const int n[4], int a, int b, int c, int d) {
    return ((static_cast<std::size_t>(a) * n[1] + b) * n[2] + c) * n[3] + d;
  };
  int k[4];
  for (k[0] = 0; k[0] < nc[0]; ++k[0])
    for (k[1] = 0; k[1] < nc[1]; ++k[1])
      for (k[2] = 0; k[2] < nc[2]; ++k[2])
        for (k[3] = 0; k[3] < nc[3]; ++k[3]) {
          const Real cf = coeff(k[0], k[1], k[2], k[3]);
          if (cf == Real(0)) continue;
          int b3[3];
          cart_comp(lp, k[pos], b3[0], b3[1], b3[2]);
          for (int e = 0; e < 3; ++e) {
            int m3[3] = {b3[0], b3[1], b3[2]};
            m3[e] += 1;
            const int ip = detail::comp_index(lp + 1, m3[0], m3[1]);
            int ii[4] = {k[0], k[1], k[2], k[3]};
            ii[pos] = ip;
            Real term = 2 * ap * plus[idx(npl, ii[0], ii[1], ii[2], ii[3])];
            if (b3[e] >= 1) {
              int mm[3] = {b3[0], b3[1], b3[2]};
              mm[e] -= 1;
              const int im = detail::comp_index(lp - 1, mm[0], mm[1]);
              int jj[4] = {k[0], k[1], k[2], k[3]};
              jj[pos] = im;
              term -= Real(b3[e]) * minus[idx(nmi, jj[0], jj[1], jj[2], jj[3])];
            }
            out[e] += cf * term;
          }
        }
}

/// Zero-exponent unit-s ghost at a shell's centre (the RI ket partner).
template <class Real>
PrimitiveShell<Real> ghost_shell(const PrimitiveShell<Real> &s) {
  return {Real(0), {s.center[0], s.center[1], s.center[2]}, 0};
}

/// Full (uncontracted) derivative block d/dR_{pos,dir} of the quartet
/// (s0 s1 | s2 s3): out has the base component shape, out[idx] = the MD
/// centre-shift 2 alpha [.+1_dir] - m [.-1_dir] of position `pos`.
template <class Real>
std::vector<Real> quartet_pos_deriv_block(const PrimitiveShell<Real> &s0,
                                          const PrimitiveShell<Real> &s1,
                                          const PrimitiveShell<Real> &s2,
                                          const PrimitiveShell<Real> &s3, int pos, int dir,
                                          const TGrid<Real> &grid) {
  const PrimitiveShell<Real> sh[4] = {s0, s1, s2, s3};
  const int L[4] = {s0.l, s1.l, s2.l, s3.l};
  const int nc[4] = {ncart(L[0]), ncart(L[1]), ncart(L[2]), ncart(L[3])};
  const int lp = L[pos];
  const Real ap = sh[pos].alpha;
  auto promote = [&](int dl) {
    PrimitiveShell<Real> s[4] = {s0, s1, s2, s3};
    s[pos].l += dl;
    return detail::eri_block4(s[0], s[1], s[2], s[3], grid);
  };
  auto plus = promote(1);
  std::vector<Real> minus;
  if (lp >= 1) minus = promote(-1);
  int npl[4], nmi[4];
  for (int i = 0; i < 4; ++i) {
    npl[i] = nc[i];
    nmi[i] = nc[i];
  }
  npl[pos] = ncart(lp + 1);
  if (lp >= 1) nmi[pos] = ncart(lp - 1);
  auto idx = [](const int n[4], int a, int b, int c, int d) {
    return ((static_cast<std::size_t>(a) * n[1] + b) * n[2] + c) * n[3] + d;
  };
  std::vector<Real> out(static_cast<std::size_t>(nc[0]) * nc[1] * nc[2] * nc[3], Real(0));
  int k[4];
  for (k[0] = 0; k[0] < nc[0]; ++k[0])
    for (k[1] = 0; k[1] < nc[1]; ++k[1])
      for (k[2] = 0; k[2] < nc[2]; ++k[2])
        for (k[3] = 0; k[3] < nc[3]; ++k[3]) {
          int b3[3];
          cart_comp(lp, k[pos], b3[0], b3[1], b3[2]);
          int m3[3] = {b3[0], b3[1], b3[2]};
          m3[dir] += 1;
          const int ip = detail::comp_index(lp + 1, m3[0], m3[1]);
          int ii[4] = {k[0], k[1], k[2], k[3]};
          ii[pos] = ip;
          Real term = 2 * ap * plus[idx(npl, ii[0], ii[1], ii[2], ii[3])];
          if (b3[dir] >= 1) {
            int mm[3] = {b3[0], b3[1], b3[2]};
            mm[dir] -= 1;
            const int im = detail::comp_index(lp - 1, mm[0], mm[1]);
            int jj[4] = {k[0], k[1], k[2], k[3]};
            jj[pos] = im;
            term -= Real(b3[dir]) * minus[idx(nmi, jj[0], jj[1], jj[2], jj[3])];
          }
          out[idx(nc, k[0], k[1], k[2], k[3])] = term;
        }
  return out;
}

/// Second geometric derivative d^2/dR_{p,e} dR_{q,f} of the quartet
/// (s0 s1 | s2 s3), contracted with coeff(k0,k1,k2,k3) over the components,
/// returned as out[e][f]. The MD centre-shift applied twice (l+/-2 for p==q,
/// l+/-1 x l+/-1 for p!=q); p, q must be non-ghost positions.
template <class Real, class CoeffFn>
void quartet_pos_hess(const PrimitiveShell<Real> &s0, const PrimitiveShell<Real> &s1,
                      const PrimitiveShell<Real> &s2, const PrimitiveShell<Real> &s3,
                      int p, int q, const TGrid<Real> &grid, CoeffFn coeff,
                      Real out[3][3]) {
  const PrimitiveShell<Real> sh[4] = {s0, s1, s2, s3};
  const int L[4] = {s0.l, s1.l, s2.l, s3.l};
  const int nc[4] = {ncart(L[0]), ncart(L[1]), ncart(L[2]), ncart(L[3])};
  const Real ap = sh[p].alpha, aq = sh[q].alpha;
  std::map<std::array<int, 4>, std::vector<Real>> cache;
  auto block = [&](const std::array<int, 4> &o) -> const std::vector<Real> * {
    for (int i = 0; i < 4; ++i)
      if (L[i] + o[i] < 0) return nullptr;
    auto it = cache.find(o);
    if (it == cache.end()) {
      PrimitiveShell<Real> s[4] = {s0, s1, s2, s3};
      for (int i = 0; i < 4; ++i) s[i].l += o[i];
      it = cache.emplace(o, detail::eri_block4(s[0], s[1], s[2], s[3], grid)).first;
    }
    return &it->second;
  };
  auto rawval = [&](const std::array<int, 4> &o, const int m[4][3]) -> Real {
    const auto *blk = block(o);
    if (!blk) return Real(0);
    int nn[4], id[4];
    for (int i = 0; i < 4; ++i) {
      const int li = L[i] + o[i];
      if (m[i][0] < 0 || m[i][1] < 0 || m[i][2] < 0 || m[i][0] + m[i][1] + m[i][2] != li)
        return Real(0);
      nn[i] = ncart(li);
      id[i] = detail::comp_index(li, m[i][0], m[i][1]);
    }
    return (*blk)[((static_cast<std::size_t>(id[0]) * nn[1] + id[1]) * nn[2] + id[2]) * nn[3] +
                  id[3]];
  };
  for (int e = 0; e < 3; ++e)
    for (int f = 0; f < 3; ++f) out[e][f] = Real(0);
  int bm[4][3];
  int k[4];
  for (k[0] = 0; k[0] < nc[0]; ++k[0])
    for (k[1] = 0; k[1] < nc[1]; ++k[1])
      for (k[2] = 0; k[2] < nc[2]; ++k[2])
        for (k[3] = 0; k[3] < nc[3]; ++k[3]) {
          const Real cf = coeff(k[0], k[1], k[2], k[3]);
          if (cf == Real(0)) continue;
          for (int i = 0; i < 4; ++i) cart_comp(L[i], k[i], bm[i][0], bm[i][1], bm[i][2]);
          auto set = [&](int pp, const int mp[3], int qq, const int mq[3], int mm[4][3]) {
            for (int r = 0; r < 4; ++r)
              for (int t = 0; t < 3; ++t) mm[r][t] = bm[r][t];
            for (int t = 0; t < 3; ++t) mm[pp][t] = mp[t];
            if (qq != pp)
              for (int t = 0; t < 3; ++t) mm[qq][t] = mq[t];
          };
          for (int e = 0; e < 3; ++e)
            for (int f = 0; f < 3; ++f) {
              Real d2 = 0;
              int mm[4][3];
              if (p == q) {
                const int *mp = bm[p];
                const int de = (e == f) ? 1 : 0;
                int T[4][3];
                for (int t = 0; t < 3; ++t)
                  T[0][t] = T[1][t] = T[2][t] = T[3][t] = mp[t];
                T[0][e] += 1; T[0][f] += 1;
                T[1][f] += 1; T[1][e] -= 1;
                T[2][e] += 1; T[2][f] -= 1;
                T[3][e] -= 1; T[3][f] -= 1;
                const std::array<int, 4> o2 = {p == 0 ? 2 : 0, p == 1 ? 2 : 0, p == 2 ? 2 : 0,
                                               p == 3 ? 2 : 0};
                const std::array<int, 4> o0 = {0, 0, 0, 0};
                const std::array<int, 4> om2 = {p == 0 ? -2 : 0, p == 1 ? -2 : 0,
                                                p == 2 ? -2 : 0, p == 3 ? -2 : 0};
                set(p, T[0], q, T[0], mm);
                d2 += 2 * ap * (2 * ap * rawval(o2, mm));
                set(p, T[1], q, T[1], mm);
                d2 += 2 * ap * (-(Real(mp[e]) + de) * rawval(o0, mm));
                set(p, T[2], q, T[2], mm);
                d2 += -Real(mp[f]) * (2 * ap * rawval(o0, mm));
                set(p, T[3], q, T[3], mm);
                d2 += -Real(mp[f]) * (-(Real(mp[e]) - de) * rawval(om2, mm));
              } else {
                const int *mp = bm[p], *mq = bm[q];
                int Pe1[3], Pe0[3], Qf1[3], Qf0[3];
                for (int t = 0; t < 3; ++t) {
                  Pe1[t] = mp[t]; Pe0[t] = mp[t];
                  Qf1[t] = mq[t]; Qf0[t] = mq[t];
                }
                Pe1[e] += 1; Pe0[e] -= 1;
                Qf1[f] += 1; Qf0[f] -= 1;
                auto oo = [&](int dp, int dq) {
                  std::array<int, 4> o = {0, 0, 0, 0};
                  o[p] += dp; o[q] += dq;
                  return o;
                };
                set(p, Pe1, q, Qf1, mm);
                d2 += 4 * ap * aq * rawval(oo(1, 1), mm);
                set(p, Pe1, q, Qf0, mm);
                d2 += -2 * ap * Real(mq[f]) * rawval(oo(1, -1), mm);
                set(p, Pe0, q, Qf1, mm);
                d2 += -Real(mp[e]) * 2 * aq * rawval(oo(-1, 1), mm);
                set(p, Pe0, q, Qf0, mm);
                d2 += Real(mp[e]) * Real(mq[f]) * rawval(oo(-1, -1), mm);
              }
              out[e][f] += cf * d2;
            }
        }
}

} // namespace detail

/// Geometric gradient of the RI Coulomb energy E_J = 1/2 sum_mn D_mn J_mn
/// (J from ri_jk) w.r.t. the orbital- and auxiliary-shell centres. D is the
/// nao x nao symmetric AO density (row-major). tau_lin drops metric
/// eigenvalues below tau_lin*max, matching ri_fit.
template <class Real>
RIGrad<Real> ri_j_gradient(const ShellBasis<Real> &orb, const ShellBasis<Real> &aux,
                           const Real *D, const TGrid<Real> &grid,
                           Real tau_lin = Real(1e-10)) {
  const int nao = orb.nao, naux = aux.nao;
  auto M = coulomb_2c(aux, grid);             // naux x naux
  auto T = coulomb_3c(orb, aux, grid);        // nao*nao x naux
  // d_P = sum_mn (mn|P) D_mn = (T^T D)[P], T viewed as N x naux, D as N-vector
  std::vector<Real> d(naux, Real(0));
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  detail::gemm('T', 'N', naux, 1, static_cast<int>(N), Real(1), T.data(), naux, D, 1, Real(0),
               d.data(), 1);
  // gamma = M^{-1} d via the eigendecomposition (pseudo-inverse with cutoff)
  std::vector<Real> V = M, eval(naux);
  detail::syevd(naux, V.data(), eval.data()); // V: eigenvectors (columns, col-major)
  Real emax = 0;
  for (Real e : eval) emax = std::max(emax, e);
  std::vector<Real> gamma(naux, Real(0));
  for (int kk = 0; kk < naux; ++kk) {
    if (eval[kk] <= tau_lin * emax) continue;
    Real vd = 0;
    for (int P = 0; P < naux; ++P) vd += V[kk * naux + P] * d[P];
    const Real s = vd / eval[kk];
    for (int P = 0; P < naux; ++P) gamma[P] += s * V[kk * naux + P];
  }

  RIGrad<Real> g;
  g.forb.assign(orb.shells.size(), {Real(0), Real(0), Real(0)});
  g.faux.assign(aux.shells.size(), {Real(0), Real(0), Real(0)});
  auto Dm = [&](int i, int j) { return D[static_cast<std::size_t>(i) * nao + j]; };

  // Term A: sum_P gamma_P sum_mn D_mn d(mn|P)/dx, quartet (m n | P ghost)
  const int nso = static_cast<int>(orb.shells.size());
  const int nsa = static_cast<int>(aux.shells.size());
  for (int m = 0; m < nso; ++m)
    for (int n = 0; n < nso; ++n)
      for (int a = 0; a < nsa; ++a) {
        const auto &sm = orb.shells[m], &sn = orb.shells[n], &sP = aux.shells[a];
        const auto gh = detail::ghost_shell(sP);
        const int om = orb.ao_off[m], on = orb.ao_off[n], oP = aux.ao_off[a];
        const int nn = ncart(sn.l), nPc = ncart(sP.l);
        auto coeff = [&](int km, int kn, int kP, int) {
          return Dm(om + km, on + kn) * gamma[oP + kP];
        };
        (void)nn;
        (void)nPc;
        const int posshell[3] = {m, n, a};
        for (int pos = 0; pos < 3; ++pos) {
          Real out[3] = {0, 0, 0};
          detail::quartet_pos_grad(sm, sn, sP, gh, pos, grid, coeff, out);
          auto &F = (pos == 2) ? g.faux[a] : g.forb[posshell[pos]];
          for (int e = 0; e < 3; ++e) F[e] += out[e];
        }
      }

  // Term B: -1/2 sum_PQ gamma_P gamma_Q dM_PQ/dx, quartet (P ghost | Q ghost)
  for (int a = 0; a < nsa; ++a)
    for (int b = 0; b < nsa; ++b) {
      const auto &sP = aux.shells[a], &sQ = aux.shells[b];
      const auto ghP = detail::ghost_shell(sP), ghQ = detail::ghost_shell(sQ);
      const int oP = aux.ao_off[a], oQ = aux.ao_off[b];
      auto coeff = [&](int kP, int, int kQ, int) {
        return Real(-0.5) * gamma[oP + kP] * gamma[oQ + kQ];
      };
      for (int pos : {0, 2}) {
        Real out[3] = {0, 0, 0};
        detail::quartet_pos_grad(sP, ghP, sQ, ghQ, pos, grid, coeff, out);
        auto &F = (pos == 0) ? g.faux[a] : g.faux[b];
        for (int e = 0; e < 3; ++e) F[e] += out[e];
      }
    }
  return g;
}

/// Geometric Hessian of the RI Coulomb energy E_J, as a (3 ncen) x (3 ncen)
/// matrix with ncen = (#orbital shells) + (#auxiliary shells), ordered orbital
/// shells first then auxiliary shells (Cartesian-minor). Uses the envelope form
///   d^2E_J/dxdy = gamma^T d_xy - 1/2 gamma^T M_xy gamma + r_x^T M^{-1} r_y,
/// r_x = d_x - M_x gamma. The caller maps shells to atoms and sums.
template <class Real>
std::vector<Real> ri_j_hessian(const ShellBasis<Real> &orb, const ShellBasis<Real> &aux,
                               const Real *D, const TGrid<Real> &grid,
                               Real tau_lin = Real(1e-10)) {
  const int nao = orb.nao, naux = aux.nao;
  const int nso = static_cast<int>(orb.shells.size());
  const int nsa = static_cast<int>(aux.shells.size());
  const int ncen = nso + nsa, dim = 3 * ncen;
  auto M = coulomb_2c(aux, grid);
  auto T = coulomb_3c(orb, aux, grid);
  auto Dm = [&](int i, int j) { return D[static_cast<std::size_t>(i) * nao + j]; };
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  // d_P and M^{-1}: d_P = sum_mn (mn|P) D_mn = (T^T D)[P]
  std::vector<Real> d(naux, Real(0));
  detail::gemm('T', 'N', naux, 1, static_cast<int>(N), Real(1), T.data(), naux, D, 1, Real(0),
               d.data(), 1);
  std::vector<Real> Vv = M, eval(naux);
  detail::syevd(naux, Vv.data(), eval.data());
  Real emax = 0;
  for (Real e : eval) emax = std::max(emax, e);
  std::vector<Real> Minv(static_cast<std::size_t>(naux) * naux, Real(0)), gamma(naux, Real(0));
  for (int kk = 0; kk < naux; ++kk) {
    if (eval[kk] <= tau_lin * emax) continue;
    const Real inv = Real(1) / eval[kk];
    Real vd = 0;
    for (int P = 0; P < naux; ++P) vd += Vv[kk * naux + P] * d[P];
    for (int P = 0; P < naux; ++P) {
      gamma[P] += inv * vd * Vv[kk * naux + P];
      for (int Q = 0; Q < naux; ++Q)
        Minv[P * naux + Q] += inv * Vv[kk * naux + P] * Vv[kk * naux + Q];
    }
  }
  auto cshell = [&](bool isaux, int s) { return isaux ? nso + s : s; };

  // first-derivative residual tensor r[x][P] = d_x[P] - (M_x gamma)[P]
  std::vector<Real> r(static_cast<std::size_t>(dim) * naux, Real(0));
  auto radd = [&](int cs, int dir, int P, Real v) {
    r[(static_cast<std::size_t>(3 * cs + dir)) * naux + P] += v;
  };
  const auto ghost = [&](const PrimitiveShell<Real> &s) { return detail::ghost_shell(s); };
  // d_x: 3-centre (m n | a ghost)
  for (int m = 0; m < nso; ++m)
    for (int n = 0; n < nso; ++n)
      for (int a = 0; a < nsa; ++a) {
        const auto &sm = orb.shells[m], &sn = orb.shells[n], &sP = aux.shells[a];
        const auto gh = ghost(sP);
        const int om = orb.ao_off[m], on = orb.ao_off[n], oP = aux.ao_off[a];
        const int nm = ncart(sm.l), nn = ncart(sn.l), nP = ncart(sP.l);
        const int cs[3] = {cshell(false, m), cshell(false, n), cshell(true, a)};
        for (int pos = 0; pos < 3; ++pos)
          for (int dir = 0; dir < 3; ++dir) {
            auto blk = detail::quartet_pos_deriv_block(sm, sn, sP, gh, pos, dir, grid);
            for (int km = 0; km < nm; ++km)
              for (int kn = 0; kn < nn; ++kn) {
                const Real dmn = Dm(om + km, on + kn);
                if (dmn == Real(0)) continue;
                for (int kP = 0; kP < nP; ++kP)
                  radd(cs[pos], dir, oP + kP,
                       dmn * blk[((static_cast<std::size_t>(km) * nn + kn) * nP + kP)]);
              }
          }
      }
  // -(M_x gamma): 2-centre (a ghost | b ghost), free aux index = a
  for (int a = 0; a < nsa; ++a)
    for (int b = 0; b < nsa; ++b) {
      const auto &sA = aux.shells[a], &sB = aux.shells[b];
      const auto ghA = ghost(sA), ghB = ghost(sB);
      const int oA = aux.ao_off[a], oB = aux.ao_off[b];
      const int nA = ncart(sA.l), nB = ncart(sB.l);
      for (int pos : {0, 2})
        for (int dir = 0; dir < 3; ++dir) {
          auto blk = detail::quartet_pos_deriv_block(sA, ghA, sB, ghB, pos, dir, grid);
          const int cs = (pos == 0) ? cshell(true, a) : cshell(true, b);
          for (int ka = 0; ka < nA; ++ka) {
            Real acc = 0;
            for (int kb = 0; kb < nB; ++kb)
              acc += blk[static_cast<std::size_t>(ka) * nB + kb] * gamma[oB + kb];
            radd(cs, dir, oA + ka, -acc);
          }
        }
    }
  // s[x] = M^{-1} r[x]: s[x][P] = sum_Q r[x][Q] Minv[P][Q] = (r Minv^T)[x][P]
  std::vector<Real> s(static_cast<std::size_t>(dim) * naux, Real(0));
  detail::gemm('N', 'T', dim, naux, naux, Real(1), r.data(), naux, Minv.data(), naux,
               Real(0), s.data(), naux);

  std::vector<Real> H(static_cast<std::size_t>(dim) * dim, Real(0));
  auto Hadd = [&](int x, int y, Real v) { H[static_cast<std::size_t>(x) * dim + y] += v; };
  // response term H[x][y] = sum_P r[x][P] s[y][P] = (r s^T)[x][y]
  detail::gemm('N', 'T', dim, dim, naux, Real(1), r.data(), naux, s.data(), naux, Real(1),
               H.data(), dim);
  // direct term 1: gamma^T d_xy, 3-centre with coeff D_mn gamma_P
  for (int m = 0; m < nso; ++m)
    for (int n = 0; n < nso; ++n)
      for (int a = 0; a < nsa; ++a) {
        const auto &sm = orb.shells[m], &sn = orb.shells[n], &sP = aux.shells[a];
        const auto gh = ghost(sP);
        const int om = orb.ao_off[m], on = orb.ao_off[n], oP = aux.ao_off[a];
        auto coeff = [&](int km, int kn, int kP, int) {
          return Dm(om + km, on + kn) * gamma[oP + kP];
        };
        const int cs[3] = {cshell(false, m), cshell(false, n), cshell(true, a)};
        for (int p = 0; p < 3; ++p)
          for (int q = 0; q < 3; ++q) {
            Real o[3][3];
            detail::quartet_pos_hess(sm, sn, sP, gh, p, q, grid, coeff, o);
            for (int e = 0; e < 3; ++e)
              for (int f = 0; f < 3; ++f) Hadd(3 * cs[p] + e, 3 * cs[q] + f, o[e][f]);
          }
      }
  // direct term 2: -1/2 gamma^T M_xy gamma, 2-centre with coeff -1/2 gamma gamma
  for (int a = 0; a < nsa; ++a)
    for (int b = 0; b < nsa; ++b) {
      const auto &sA = aux.shells[a], &sB = aux.shells[b];
      const auto ghA = ghost(sA), ghB = ghost(sB);
      const int oA = aux.ao_off[a], oB = aux.ao_off[b];
      auto coeff = [&](int ka, int, int kb, int) {
        return Real(-0.5) * gamma[oA + ka] * gamma[oB + kb];
      };
      const int cs[2] = {cshell(true, a), cshell(true, b)};
      const int poss[2] = {0, 2};
      for (int pi = 0; pi < 2; ++pi)
        for (int qi = 0; qi < 2; ++qi) {
          Real o[3][3];
          detail::quartet_pos_hess(sA, ghA, sB, ghB, poss[pi], poss[qi], grid, coeff, o);
          for (int e = 0; e < 3; ++e)
            for (int f = 0; f < 3; ++f) Hadd(3 * cs[pi] + e, 3 * cs[qi] + f, o[e][f]);
        }
    }
  return H;
}

/// Geometric gradient of the RI exchange energy E_K = -1/4 sum_mn D_mn K_mn
/// (K from ri_jk) w.r.t. the orbital- and auxiliary-shell centres. With the
/// density-transformed 3-index H_{sn}^Q = sum_l D_sl (ln|Q), its fit
/// G_{ab}^R = sum_Q M^{-1}_RQ H_{ab}^Q, and E_K = -1/4 sum_Q H_{ns}^Q G_{sn}^Q,
///   dE_K/dx = -1/2 sum_{lnR} [sum_s D_ls G_{ns}^R] d(ln|R)/dx
///           + 1/4  sum_{TU}   [sum_sn G_{ns}^T G_{sn}^U] d(T|U)/dx.
template <class Real>
RIGrad<Real> ri_k_gradient(const ShellBasis<Real> &orb, const ShellBasis<Real> &aux,
                           const Real *D, const TGrid<Real> &grid,
                           Real tau_lin = Real(1e-10)) {
  const int nao = orb.nao, naux = aux.nao;
  auto M = coulomb_2c(aux, grid);
  auto T = coulomb_3c(orb, aux, grid); // T[(mu*nao+la)*naux+P] = (mu la|P)
  auto Dm = [&](int i, int j) { return D[static_cast<std::size_t>(i) * nao + j]; };
  const std::size_t no2 = static_cast<std::size_t>(nao) * nao;

  // M^{-1} (pseudo-inverse with cutoff)
  std::vector<Real> V = M, eval(naux);
  detail::syevd(naux, V.data(), eval.data());
  Real emax = 0;
  for (Real e : eval) emax = std::max(emax, e);
  std::vector<Real> Minv(static_cast<std::size_t>(naux) * naux, Real(0));
  for (int k = 0; k < naux; ++k) {
    if (eval[k] <= tau_lin * emax) continue;
    const Real s = Real(1) / eval[k];
    for (int R = 0; R < naux; ++R)
      for (int Q = 0; Q < naux; ++Q) Minv[R * naux + Q] += s * V[k * naux + R] * V[k * naux + Q];
  }

  // H[(s*nao+n)*naux+Q] = sum_l D_sl (l n|Q): H[s][(n,Q)] = sum_l D[s][l] T[l][(n,Q)]
  // = (D * T_reshaped)[s][(n,Q)] with T viewed as nao x (nao*naux).
  std::vector<Real> H(no2 * naux, Real(0));
  detail::gemm('N', 'N', nao, nao * naux, nao, Real(1), D, nao, T.data(), nao * naux, Real(0),
               H.data(), nao * naux);
  // G[(a*nao+b)*naux+R] = sum_Q M^{-1}_RQ H[ab][Q] = (H Minv^T)[ab][R]
  std::vector<Real> G(no2 * naux, Real(0));
  detail::gemm('N', 'T', static_cast<int>(no2), naux, naux, Real(1), H.data(), naux, Minv.data(),
               naux, Real(0), G.data(), naux);
  // c3[(l*nao+n)*naux+R] = -1/2 sum_s D_ls G[(n*nao+s)*naux+R]. Batched over n:
  // for each n, c3[l][n][R] = -1/2 (D * G[n])[l][R], G[n] the (sig,R) slice.
  std::vector<Real> c3(no2 * naux, Real(0));
  for (int n = 0; n < nao; ++n)
    detail::gemm('N', 'N', nao, naux, nao, Real(-0.5), D, nao,
                 G.data() + static_cast<std::size_t>(n) * nao * naux, naux, Real(0),
                 c3.data() + static_cast<std::size_t>(n) * naux, nao * naux);
  // c2[T*naux+U] = 1/4 sum_{sn} G[(n*nao+s)*naux+T] G[(s*nao+n)*naux+U]. With the
  // (n,s)->(s,n) inner transpose as a permuted copy Gp, c2 = 1/4 G^T Gp.
  std::vector<Real> Gp(no2 * naux, Real(0));
  for (int n = 0; n < nao; ++n)
    for (int sig = 0; sig < nao; ++sig)
      for (int R = 0; R < naux; ++R)
        Gp[(static_cast<std::size_t>(n) * nao + sig) * naux + R] =
            G[(static_cast<std::size_t>(sig) * nao + n) * naux + R];
  std::vector<Real> c2(static_cast<std::size_t>(naux) * naux, Real(0));
  detail::gemm('T', 'N', naux, naux, static_cast<int>(no2), Real(0.25), G.data(), naux, Gp.data(),
               naux, Real(0), c2.data(), naux);

  RIGrad<Real> g;
  g.forb.assign(orb.shells.size(), {Real(0), Real(0), Real(0)});
  g.faux.assign(aux.shells.size(), {Real(0), Real(0), Real(0)});
  const int nso = static_cast<int>(orb.shells.size());
  const int nsa = static_cast<int>(aux.shells.size());

  // Term 1: 3-centre d(l n|R) contracted with c3, quartet (l n | R ghost)
  for (int lsh = 0; lsh < nso; ++lsh)
    for (int nsh = 0; nsh < nso; ++nsh)
      for (int a = 0; a < nsa; ++a) {
        const auto &sl = orb.shells[lsh], &sn = orb.shells[nsh], &sR = aux.shells[a];
        const auto gh = detail::ghost_shell(sR);
        const int ol = orb.ao_off[lsh], on = orb.ao_off[nsh], oR = aux.ao_off[a];
        auto coeff = [&](int kl, int kn, int kR, int) {
          return c3[((static_cast<std::size_t>(ol + kl) * nao + on + kn) * naux) + oR + kR];
        };
        const int posshell[3] = {lsh, nsh, a};
        for (int pos = 0; pos < 3; ++pos) {
          Real out[3] = {0, 0, 0};
          detail::quartet_pos_grad(sl, sn, sR, gh, pos, grid, coeff, out);
          auto &F = (pos == 2) ? g.faux[a] : g.forb[posshell[pos]];
          for (int e = 0; e < 3; ++e) F[e] += out[e];
        }
      }
  // Term 2: 2-centre d(T|U) contracted with c2, quartet (T ghost | U ghost)
  for (int a = 0; a < nsa; ++a)
    for (int b = 0; b < nsa; ++b) {
      const auto &sT = aux.shells[a], &sU = aux.shells[b];
      const auto ghT = detail::ghost_shell(sT), ghU = detail::ghost_shell(sU);
      const int oT = aux.ao_off[a], oU = aux.ao_off[b];
      auto coeff = [&](int kT, int, int kU, int) {
        return c2[(static_cast<std::size_t>(oT + kT) * naux) + oU + kU];
      };
      for (int pos : {0, 2}) {
        Real out[3] = {0, 0, 0};
        detail::quartet_pos_grad(sT, ghT, sU, ghU, pos, grid, coeff, out);
        auto &F = (pos == 0) ? g.faux[a] : g.faux[b];
        for (int e = 0; e < 3; ++e) F[e] += out[e];
      }
    }
  return g;
}

/// Geometric Hessian of the RI exchange energy E_K, as a (3 ncen) x (3 ncen)
/// matrix (orbital shells then auxiliary shells). Envelope form: direct terms
/// (coeff3, coeff2 of ri_k_gradient contracted with the integral Hessians) plus
/// the response -1/2 sum_{ab} R_x[a,b]^T M^{-1} R_y[a,b],
///   R_x[a,b,Q] = dH[a,b,Q]/dx - sum_R dM_QR/dx G[a,b,R],
/// H[a,b,Q] = sum_l D_al (l b|Q), G = M^{-1} H.
template <class Real>
std::vector<Real> ri_k_hessian(const ShellBasis<Real> &orb, const ShellBasis<Real> &aux,
                               const Real *D, const TGrid<Real> &grid,
                               Real tau_lin = Real(1e-10)) {
  const int nao = orb.nao, naux = aux.nao;
  const int nso = static_cast<int>(orb.shells.size());
  const int nsa = static_cast<int>(aux.shells.size());
  const int ncen = nso + nsa, dim = 3 * ncen;
  auto M = coulomb_2c(aux, grid);
  auto T = coulomb_3c(orb, aux, grid); // (mu la|P)
  auto Dm = [&](int i, int j) { return D[static_cast<std::size_t>(i) * nao + j]; };
  const std::size_t no2 = static_cast<std::size_t>(nao) * nao;
  // M^{-1}
  std::vector<Real> Vv = M, eval(naux);
  detail::syevd(naux, Vv.data(), eval.data());
  Real emax = 0;
  for (Real e : eval) emax = std::max(emax, e);
  std::vector<Real> Minv(static_cast<std::size_t>(naux) * naux, Real(0));
  for (int k = 0; k < naux; ++k) {
    if (eval[k] <= tau_lin * emax) continue;
    const Real s = Real(1) / eval[k];
    for (int P = 0; P < naux; ++P)
      for (int Q = 0; Q < naux; ++Q) Minv[P * naux + Q] += s * Vv[k * naux + P] * Vv[k * naux + Q];
  }
  // H[(a*nao+b)*naux+Q] = sum_l D_al (l b|Q) = (D * T_reshaped); G = M^{-1} H = H Minv^T
  std::vector<Real> Hm(no2 * naux, Real(0)), G(no2 * naux, Real(0));
  detail::gemm('N', 'N', nao, nao * naux, nao, Real(1), D, nao, T.data(), nao * naux, Real(0),
               Hm.data(), nao * naux);
  detail::gemm('N', 'T', static_cast<int>(no2), naux, naux, Real(1), Hm.data(), naux, Minv.data(),
               naux, Real(0), G.data(), naux);
  // coeff3[(l*nao+n)*naux+R] = -1/2 sum_s D_ls G[(n*nao+s)*naux+R], batched over n
  std::vector<Real> coeff3(no2 * naux, Real(0));
  for (int n = 0; n < nao; ++n)
    detail::gemm('N', 'N', nao, naux, nao, Real(-0.5), D, nao,
                 G.data() + static_cast<std::size_t>(n) * nao * naux, naux, Real(0),
                 coeff3.data() + static_cast<std::size_t>(n) * naux, nao * naux);
  // coeff2[T*naux+U] = 1/4 sum_{sn} G[(n*nao+s)*T] G[(s*nao+n)*U] = 1/4 G^T Gp,
  // Gp the (n,s)->(s,n) inner-transposed copy of G.
  std::vector<Real> Gp(no2 * naux, Real(0));
  for (int n = 0; n < nao; ++n)
    for (int sig = 0; sig < nao; ++sig)
      for (int R = 0; R < naux; ++R)
        Gp[(static_cast<std::size_t>(n) * nao + sig) * naux + R] =
            G[(static_cast<std::size_t>(sig) * nao + n) * naux + R];
  std::vector<Real> coeff2(static_cast<std::size_t>(naux) * naux, Real(0));
  detail::gemm('T', 'N', naux, naux, static_cast<int>(no2), Real(0.25), G.data(), naux, Gp.data(),
               naux, Real(0), coeff2.data(), naux);

  auto cshell = [&](bool isaux, int sidx) { return isaux ? nso + sidx : sidx; };
  const auto ghost = [&](const PrimitiveShell<Real> &sx) { return detail::ghost_shell(sx); };
  std::vector<Real> Hess(static_cast<std::size_t>(dim) * dim, Real(0));
  auto Hadd = [&](int x, int y, Real v) { Hess[static_cast<std::size_t>(x) * dim + y] += v; };

  // direct terms
  for (int l = 0; l < nso; ++l)
    for (int n = 0; n < nso; ++n)
      for (int a = 0; a < nsa; ++a) {
        const auto &sl = orb.shells[l], &sn = orb.shells[n], &sR = aux.shells[a];
        const auto gh = ghost(sR);
        const int ol = orb.ao_off[l], on = orb.ao_off[n], oR = aux.ao_off[a];
        auto coeff = [&](int kl, int kn, int kR, int) {
          return coeff3[((static_cast<std::size_t>(ol + kl) * nao + on + kn) * naux) + oR + kR];
        };
        const int cs[3] = {cshell(false, l), cshell(false, n), cshell(true, a)};
        for (int p = 0; p < 3; ++p)
          for (int q = 0; q < 3; ++q) {
            Real o[3][3];
            detail::quartet_pos_hess(sl, sn, sR, gh, p, q, grid, coeff, o);
            for (int e = 0; e < 3; ++e)
              for (int f = 0; f < 3; ++f) Hadd(3 * cs[p] + e, 3 * cs[q] + f, o[e][f]);
          }
      }
  for (int a = 0; a < nsa; ++a)
    for (int b = 0; b < nsa; ++b) {
      const auto &sT = aux.shells[a], &sU = aux.shells[b];
      const auto ghT = ghost(sT), ghU = ghost(sU);
      const int oT = aux.ao_off[a], oU = aux.ao_off[b];
      auto coeff = [&](int kT, int, int kU, int) {
        return coeff2[(static_cast<std::size_t>(oT + kT) * naux) + oU + kU];
      };
      const int cs[2] = {cshell(true, a), cshell(true, b)};
      const int poss[2] = {0, 2};
      for (int pi = 0; pi < 2; ++pi)
        for (int qi = 0; qi < 2; ++qi) {
          Real o[3][3];
          detail::quartet_pos_hess(sT, ghT, sU, ghU, poss[pi], poss[qi], grid, coeff, o);
          for (int e = 0; e < 3; ++e)
            for (int f = 0; f < 3; ++f) Hadd(3 * cs[pi] + e, 3 * cs[qi] + f, o[e][f]);
        }
    }

  // response: R_x[(a*nao+b)*naux+Q] = dH[a,b,Q]/dx - sum_R dM_QR/dx G[a,b,R]
  std::vector<Real> R(static_cast<std::size_t>(dim) * no2 * naux, Real(0));
  auto Ridx = [&](int x, int a, int b, int Q) {
    return (static_cast<std::size_t>(x) * no2 + (static_cast<std::size_t>(a) * nao + b)) * naux + Q;
  };
  // part 1: dH/dx = sum_l D_al d(l b|Q)/dx, 3-centre (l n | c ghost)
  for (int l = 0; l < nso; ++l)
    for (int n = 0; n < nso; ++n)
      for (int c = 0; c < nsa; ++c) {
        const auto &sl = orb.shells[l], &sn = orb.shells[n], &sc = aux.shells[c];
        const auto gh = ghost(sc);
        const int ol = orb.ao_off[l], on = orb.ao_off[n], oc = aux.ao_off[c];
        const int nl = ncart(sl.l), nn = ncart(sn.l), nP = ncart(sc.l);
        const int cs[3] = {cshell(false, l), cshell(false, n), cshell(true, c)};
        for (int pos = 0; pos < 3; ++pos)
          for (int dir = 0; dir < 3; ++dir) {
            auto blk = detail::quartet_pos_deriv_block(sl, sn, sc, gh, pos, dir, grid);
            const int xi = 3 * cs[pos] + dir;
            for (int kl = 0; kl < nl; ++kl)
              for (int kn = 0; kn < nn; ++kn)
                for (int kQ = 0; kQ < nP; ++kQ) {
                  const Real dv = blk[((static_cast<std::size_t>(kl) * nn + kn) * nP + kQ)];
                  if (dv == Real(0)) continue;
                  const int lam = ol + kl, bb = on + kn, QQ = oc + kQ;
                  for (int aa = 0; aa < nao; ++aa) {
                    const Real dal = Dm(aa, lam);
                    if (dal != Real(0)) R[Ridx(xi, aa, bb, QQ)] += dal * dv;
                  }
                }
          }
      }
  // part 2: - sum_R dM_QR/dx G[a,b,R], 2-centre (c ghost | e ghost), Q from c
  for (int c = 0; c < nsa; ++c)
    for (int ee = 0; ee < nsa; ++ee) {
      const auto &sC = aux.shells[c], &sE = aux.shells[ee];
      const auto ghC = ghost(sC), ghE = ghost(sE);
      const int oC = aux.ao_off[c], oE = aux.ao_off[ee];
      const int nC = ncart(sC.l), nE = ncart(sE.l);
      for (int pos : {0, 2})
        for (int dir = 0; dir < 3; ++dir) {
          auto blk = detail::quartet_pos_deriv_block(sC, ghC, sE, ghE, pos, dir, grid);
          const int cs = (pos == 0) ? cshell(true, c) : cshell(true, ee);
          const int xi = 3 * cs + dir;
          for (int kQ = 0; kQ < nC; ++kQ)
            for (int kR = 0; kR < nE; ++kR) {
              const Real dv = blk[static_cast<std::size_t>(kQ) * nE + kR];
              if (dv == Real(0)) continue;
              const int QQ = oC + kQ, RR = oE + kR;
              for (int aa = 0; aa < nao; ++aa)
                for (int bb = 0; bb < nao; ++bb)
                  R[Ridx(xi, aa, bb, QQ)] -=
                      dv * G[(static_cast<std::size_t>(aa) * nao + bb) * naux + RR];
            }
        }
    }
  // S = M^{-1} R (over Q): S[(x,ab)][Q] = sum_Rr R[(x,ab)][Rr] Minv[Q][Rr]
  // = (R Minv^T) with (x,ab) flattened into one dim*no2 row index.
  std::vector<Real> S(static_cast<std::size_t>(dim) * no2 * naux, Real(0));
  detail::gemm('N', 'T', static_cast<int>(static_cast<std::size_t>(dim) * no2), naux, naux,
               Real(1), R.data(), naux, Minv.data(), naux, Real(0), S.data(), naux);
  // response Hess[x][y] += -1/2 sum_{a,b,Q} R_x[b,a,Q] S_y[a,b,Q]. L_GG couples
  // (a,b) with (b,a): form the (a,b)->(b,a) inner-permuted copy Rp of R, then the
  // sum over the flattened (a,b,Q) index is the GEMM Hess += -1/2 Rp S^T.
  std::vector<Real> Rp(static_cast<std::size_t>(dim) * no2 * naux, Real(0));
  for (int x = 0; x < dim; ++x)
    for (int a = 0; a < nao; ++a)
      for (int b = 0; b < nao; ++b)
        std::copy(R.data() + Ridx(x, b, a, 0), R.data() + Ridx(x, b, a, 0) + naux,
                  Rp.data() + Ridx(x, a, b, 0));
  const int Kab = static_cast<int>(no2 * naux);
  detail::gemm('N', 'T', dim, dim, Kab, Real(-0.5), Rp.data(), Kab, S.data(), Kab, Real(1),
               Hess.data(), dim);
  return Hess;
}

} // namespace intti
