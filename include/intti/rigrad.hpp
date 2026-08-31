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

#include <array>
#include <cstddef>
#include <vector>

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
  // d_P = sum_mn (mn|P) D_mn
  std::vector<Real> d(naux, Real(0));
  const std::size_t N = static_cast<std::size_t>(nao) * nao;
  for (std::size_t mn = 0; mn < N; ++mn)
    for (int P = 0; P < naux; ++P) d[P] += T[mn * naux + P] * D[mn];
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

} // namespace intti
