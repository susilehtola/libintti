// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Two- and three-center Coulomb integrals -- the RI (density-fitting) core.
// An auxiliary function P is treated as a shell paired with a zero-exponent
// unit-s "ghost" at the same centre (beta=0 => p=alpha, K=1: the libcint
// int2c2e/int3c2e trick), so both reduce to the ordinary quartet engine:
//   (P|Q)   = quartet( (P,ghost), (Q,ghost) )
//   (mu nu|P) = quartet( (mu,nu),  (P,ghost) ).
// Matrix/tensor-level API only: whole (P|Q) matrix and (mu nu|P) tensor.

#include <cstddef>
#include <vector>

#include "fock.hpp"
#include "gto.hpp"
#include "math.hpp"
#include "multipole.hpp"
#include "quartet.hpp"
#include "tgrid.hpp"

namespace intti {

namespace detail {
/// A shell paired with a zero-exponent unit-s ghost at the same centre.
template <class Real>
ShellPair<Real> ghost_pair(const PrimitiveShell<Real> &s) {
  PrimitiveShell<Real> ghost{Real(0), {s.center[0], s.center[1], s.center[2]}, 0};
  return make_pair(s, ghost);
}

/// True if bra and ket pairs are well separated for the multipole far-field:
/// alpha_pq |P_bra - P_ket|^2 > -ln(far_tau), alpha_pq = p_p p_q/(p_p+p_q).
template <class Real>
bool pair_far(const ShellPair<Real> &bra, const ShellPair<Real> &ket,
              Real far_cut) {
  Real R2 = 0;
  for (int d = 0; d < 3; ++d) {
    const Real x = bra.P[d] - ket.P[d];
    R2 += x * x;
  }
  return bra.p * ket.p / (bra.p + ket.p) * R2 > far_cut;
}
} // namespace detail

/// Two-center Coulomb metric (P|Q) over an auxiliary basis: naux x naux,
/// row-major, primitive Cartesian, unnormalized.
template <class Real>
std::vector<Real> coulomb_2c(const ShellBasis<Real> &aux, const TGrid<Real> &grid) {
  const int naux = aux.nao;
  std::vector<Real> M(static_cast<std::size_t>(naux) * naux, Real(0));
  const int ns = static_cast<int>(aux.shells.size());
  // (P|Q) = (Q|P): compute the a >= b triangle once and mirror the transpose.
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b <= a; ++b) {
      auto bra = detail::ghost_pair(aux.shells[a]);
      auto ket = detail::ghost_pair(aux.shells[b]);
      const int nP = ncart(aux.shells[a].l), nQ = ncart(aux.shells[b].l);
      std::vector<Real> blk(static_cast<std::size_t>(nP) * nQ);
      eri_quartet(bra, ket, grid, blk.data());
      for (int kP = 0; kP < nP; ++kP)
        for (int kQ = 0; kQ < nQ; ++kQ) {
          const Real v = blk[kP * nQ + kQ];
          M[(aux.ao_off[a] + kP) * naux + aux.ao_off[b] + kQ] = v;
          if (a != b)
            M[(aux.ao_off[b] + kQ) * naux + aux.ao_off[a] + kP] = v; // transpose mirror
        }
    }
  return M;
}

/// Three-center Coulomb (mu nu | P): row-major (mu, nu, P) tensor of size
/// nao*nao*naux, primitive Cartesian, unnormalized.
/// Optional far-field (multipole) acceleration: far_tau > 0 evaluates an
/// orbital pair well separated from an auxiliary function (alpha |P-R_aux|^2 >
/// -ln(far_tau)) through the exponent-free multipole tensor instead of the
/// t-quadrature, with relative error ~ far_tau. far_tau = 0 (default) is the
/// exact build.
template <class Real>
std::vector<Real> coulomb_3c(const ShellBasis<Real> &orb,
                             const ShellBasis<Real> &aux, const TGrid<Real> &grid,
                             Real far_tau = Real(0)) {
  const int nao = orb.nao, naux = aux.nao;
  std::vector<Real> T(static_cast<std::size_t>(nao) * nao * naux, Real(0));
  const int nso = static_cast<int>(orb.shells.size());
  const int nsa = static_cast<int>(aux.shells.size());
  const bool far = far_tau > Real(0);
  const Real far_cut = far ? -log_(far_tau) : Real(0);
  // (mu nu | P) = (nu mu | P): compute the m >= n triangle once and mirror.
  for (int m = 0; m < nso; ++m)
    for (int n = 0; n <= m; ++n) {
      auto bra = make_pair(orb.shells[m], orb.shells[n]);
      const int nm = ncart(orb.shells[m].l), nn = ncart(orb.shells[n].l);
      for (int a = 0; a < nsa; ++a) {
        auto ket = detail::ghost_pair(aux.shells[a]);
        const int nP = ncart(aux.shells[a].l);
        std::vector<Real> blk(static_cast<std::size_t>(nm) * nn * nP);
        if (far && detail::pair_far(bra, ket, far_cut))
          eri_quartet_farfield(bra, ket, blk.data());
        else
          eri_quartet(bra, ket, grid, blk.data());
        for (int km = 0; km < nm; ++km)
          for (int kn = 0; kn < nn; ++kn)
            for (int kP = 0; kP < nP; ++kP) {
              const Real v = blk[(km * nn + kn) * nP + kP];
              T[((static_cast<std::size_t>(orb.ao_off[m] + km) * nao +
                  orb.ao_off[n] + kn) *
                 naux) +
                aux.ao_off[a] + kP] = v;
              if (m != n)
                T[((static_cast<std::size_t>(orb.ao_off[n] + kn) * nao +
                    orb.ao_off[m] + km) *
                   naux) +
                  aux.ao_off[a] + kP] = v; // mu<->nu mirror
            }
      }
    }
  return T;
}

} // namespace intti
