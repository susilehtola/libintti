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

#include "contracted.hpp" // ContractedBasis, detail::effective_coeff/contracted_prim
#include "fock.hpp"
#include "gto.hpp"
#include "lkc.hpp"
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
      // LKC consumer: symmetric block scatter of the precomputed (P|Q) block.
      detail::scatter_pair(M, aux, a, b, +1,
          [&](int kP, const int *, int kQ, const int *) { return blk[kP * nQ + kQ]; });
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

/// Three-center Coulomb (mu nu | P) for the auxiliary SHELLS [sa0, sa1) only:
/// a row-major (mu, nu, Plocal) tensor of size nao*nao*nauxblk, where nauxblk is
/// the number of auxiliary AOs in the range. Lets the RI pipeline stream one
/// auxiliary block at a time instead of materialising the whole nao^2 x naux
/// tensor (memory-lean / out-of-core RI). far_tau as in coulomb_3c.
template <class Real>
std::vector<Real> coulomb_3c_auxblock(const ShellBasis<Real> &orb,
                                      const ShellBasis<Real> &aux, const TGrid<Real> &grid,
                                      int sa0, int sa1, Real far_tau = Real(0)) {
  const int nao = orb.nao;
  const int p0 = aux.ao_off[sa0], p1 = aux.ao_off[sa1];
  const int nauxblk = p1 - p0;
  std::vector<Real> T(static_cast<std::size_t>(nao) * nao * nauxblk, Real(0));
  const int nso = static_cast<int>(orb.shells.size());
  const bool far = far_tau > Real(0);
  const Real far_cut = far ? -log_(far_tau) : Real(0);
  for (int m = 0; m < nso; ++m)
    for (int n = 0; n <= m; ++n) {
      auto bra = make_pair(orb.shells[m], orb.shells[n]);
      const int nm = ncart(orb.shells[m].l), nn = ncart(orb.shells[n].l);
      for (int a = sa0; a < sa1; ++a) {
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
              const int Ploc = aux.ao_off[a] - p0 + kP;
              T[(static_cast<std::size_t>(orb.ao_off[m] + km) * nao + orb.ao_off[n] + kn) *
                    nauxblk +
                Ploc] = v;
              if (m != n)
                T[(static_cast<std::size_t>(orb.ao_off[n] + kn) * nao + orb.ao_off[m] + km) *
                      nauxblk +
                  Ploc] = v; // mu<->nu mirror
            }
      }
    }
  return T;
}

// ---- generally-contracted RI n-center tensors -------------------------------
// The contracted (P|Q) and (mu nu | P) reuse the primitive ghost-shell
// eri_quartet on every primitive combination, accumulating it coefficient-
// weighted (effective_coeff = basis-set coeff * cart_norm_pyscf) into the
// contracted block -- each primitive quartet evaluated once, shared across the
// contraction indices. AO order c*ncart(l)+k (matching contracted.hpp).

/// Two-center Coulomb metric (P|Q) over a generally-contracted auxiliary basis
/// (naux x naux, PySCF cart=True normalization).
template <class Real>
std::vector<Real> coulomb_2c(const ContractedBasis<Real> &aux, const TGrid<Real> &grid) {
  const int naux = aux.nao;
  std::vector<Real> M(static_cast<std::size_t>(naux) * naux, Real(0));
  const int ns = static_cast<int>(aux.shells.size());
  for (int A = 0; A < ns; ++A)
    for (int B = 0; B <= A; ++B) {
      const auto &SA = aux.shells[A], &SB = aux.shells[B];
      const int nP = ncart(SA.l), nQ = ncart(SB.l), nctA = SA.nctr(), nctB = SB.nctr();
      const int rowB = nctB * nQ;
      std::vector<Real> cblk(static_cast<std::size_t>(nctA) * nP * rowB, Real(0));
      std::vector<Real> blk(static_cast<std::size_t>(nP) * nQ);
      for (int pa = 0; pa < SA.nprim(); ++pa) {
        auto bra = detail::ghost_pair(detail::contracted_prim(SA, pa));
        for (int pb = 0; pb < SB.nprim(); ++pb) {
          auto ket = detail::ghost_pair(detail::contracted_prim(SB, pb));
          eri_quartet(bra, ket, grid, blk.data());
          for (int cA = 0; cA < nctA; ++cA) {
            const Real wa = detail::effective_coeff(SA, cA, pa);
            if (wa == Real(0)) continue;
            for (int cB = 0; cB < nctB; ++cB) {
              const Real w = wa * detail::effective_coeff(SB, cB, pb);
              for (int kP = 0; kP < nP; ++kP)
                for (int kQ = 0; kQ < nQ; ++kQ)
                  cblk[(static_cast<std::size_t>(cA) * nP + kP) * rowB + cB * nQ + kQ] +=
                      w * blk[kP * nQ + kQ];
            }
          }
        }
      }
      for (int cA = 0; cA < nctA; ++cA)
        for (int kP = 0; kP < nP; ++kP) {
          const std::size_t I = aux.ao_off[A] + static_cast<std::size_t>(cA) * nP + kP;
          for (int cB = 0; cB < nctB; ++cB)
            for (int kQ = 0; kQ < nQ; ++kQ) {
              const std::size_t J = aux.ao_off[B] + static_cast<std::size_t>(cB) * nQ + kQ;
              const Real v = cblk[(static_cast<std::size_t>(cA) * nP + kP) * rowB + cB * nQ + kQ];
              M[I * naux + J] = v;
              M[J * naux + I] = v;
            }
        }
    }
  return M;
}

/// Three-center Coulomb (mu nu | P) over generally-contracted orbital and
/// auxiliary bases: row-major (mu, nu, P) tensor of size nao*nao*naux (PySCF
/// cart=True normalization).
template <class Real>
std::vector<Real> coulomb_3c(const ContractedBasis<Real> &orb,
                             const ContractedBasis<Real> &aux, const TGrid<Real> &grid) {
  const int nao = orb.nao, naux = aux.nao;
  std::vector<Real> T(static_cast<std::size_t>(nao) * nao * naux, Real(0));
  const int nso = static_cast<int>(orb.shells.size());
  const int nsa = static_cast<int>(aux.shells.size());
  for (int Mi = 0; Mi < nso; ++Mi)
    for (int Ni = 0; Ni <= Mi; ++Ni) {
      const auto &SM = orb.shells[Mi], &SN = orb.shells[Ni];
      const int nm = ncart(SM.l), nn = ncart(SN.l), nctM = SM.nctr(), nctN = SN.nctr();
      for (int A = 0; A < nsa; ++A) {
        const auto &SA = aux.shells[A];
        const int nP = ncart(SA.l), nctA = SA.nctr();
        // contracted block cblk[(cM*nm+km)][(cN*nn+kn)][(cA*nP+kP)]
        const int dN = nctN * nn, dA = nctA * nP;
        std::vector<Real> cblk(static_cast<std::size_t>(nctM) * nm * dN * dA, Real(0));
        std::vector<Real> blk(static_cast<std::size_t>(nm) * nn * nP);
        for (int pm = 0; pm < SM.nprim(); ++pm)
          for (int pn = 0; pn < SN.nprim(); ++pn) {
            auto bra = make_pair(detail::contracted_prim(SM, pm), detail::contracted_prim(SN, pn));
            for (int pa = 0; pa < SA.nprim(); ++pa) {
              auto ket = detail::ghost_pair(detail::contracted_prim(SA, pa));
              eri_quartet(bra, ket, grid, blk.data());
              for (int cM = 0; cM < nctM; ++cM) {
                const Real wm = detail::effective_coeff(SM, cM, pm);
                if (wm == Real(0)) continue;
                for (int cN = 0; cN < nctN; ++cN) {
                  const Real wmn = wm * detail::effective_coeff(SN, cN, pn);
                  for (int cA = 0; cA < nctA; ++cA) {
                    const Real w = wmn * detail::effective_coeff(SA, cA, pa);
                    for (int km = 0; km < nm; ++km)
                      for (int kn = 0; kn < nn; ++kn)
                        for (int kP = 0; kP < nP; ++kP)
                          cblk[(((static_cast<std::size_t>(cM) * nm + km) * dN + cN * nn + kn) * dA) +
                               cA * nP + kP] += w * blk[(km * nn + kn) * nP + kP];
                  }
                }
              }
            }
          }
        const bool offdiag = (Mi != Ni);
        for (int cM = 0; cM < nctM; ++cM)
          for (int km = 0; km < nm; ++km) {
            const std::size_t I = orb.ao_off[Mi] + static_cast<std::size_t>(cM) * nm + km;
            for (int cN = 0; cN < nctN; ++cN)
              for (int kn = 0; kn < nn; ++kn) {
                const std::size_t Jn = orb.ao_off[Ni] + static_cast<std::size_t>(cN) * nn + kn;
                for (int cA = 0; cA < nctA; ++cA)
                  for (int kP = 0; kP < nP; ++kP) {
                    const std::size_t P = aux.ao_off[A] + static_cast<std::size_t>(cA) * nP + kP;
                    const Real v = cblk[(((static_cast<std::size_t>(cM) * nm + km) * dN + cN * nn + kn) * dA) +
                                        cA * nP + kP];
                    T[(I * nao + Jn) * naux + P] = v;
                    if (offdiag) T[(Jn * nao + I) * naux + P] = v;
                  }
              }
          }
      }
    }
  return T;
}

} // namespace intti
