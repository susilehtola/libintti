// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Natively generally-contracted shells and the contraction-aware matrix
// builders (roadmap M-SHARK #5). A ContractedShell carries nprim exponents and
// an nprim x nctr coefficient block, so one shell holds many contracted
// functions sharing the same primitive set -- the generally-contracted regime
// (ANO, cc-pVXZ) where the primitive intermediates dominate and must be shared.
//
// The engine core still works with UNNORMALIZED Cartesian primitives (gto.hpp,
// hermite1d.hpp); a contracted builder loops the primitive combinations of a
// shell pair, evaluates the primitive 1D tables ONCE per primitive pair, and
// accumulates them coefficient-weighted into the contracted block, reusing each
// primitive intermediate across every contracted-function / Cartesian-component
// pair. This is SHARK's "giant E-matrix" idea expressed on our t-quadrature /
// MD substrate: no primitive integral is recomputed per contraction index.
//
// Convention (pinned to PySCF cart=True, so the contracted matrices compare
// directly to int1e_ovlp_cart etc.): a basis-set contraction coefficient d_{cp}
// is defined against a unit-normalized primitive, so the weight on our
// unnormalized primitive is d_{cp} * cart_norm_pyscf(l, alpha_p) (a per-
// primitive, Cartesian-component-independent scalar; normalization.hpp). AO
// ordering within a shell is contracted-function-outer, Cartesian-inner:
// AO = ao_off[shell] + c*ncart(l) + k (matching the libcint facade,
// src/cint.cpp). The native API stays matrix-level: a whole nao x nao AO
// matrix per call, never individual shell blocks.

#include <cstddef>
#include <vector>

#include "gto.hpp"
#include "normalization.hpp"
#include "oneel.hpp" // detail::overlap_1d / kinetic_1d / multipole_1d, pair_gauss_prefactor

namespace intti {

/// A generally-contracted Cartesian Gaussian shell: nprim primitives (shared
/// exponents `alpha`) combined into nctr contracted functions by the
/// coefficient block `coeff`, row-major over (contracted function c, primitive
/// p): coeff[c*nprim + p]. Coefficients follow the standard basis-set
/// convention (defined against unit-normalized primitives).
template <class Real = double> struct ContractedShell {
  Real center[3]{};
  int l{0};
  std::vector<Real> alpha; ///< nprim exponents
  std::vector<Real> coeff; ///< nprim*nctr, coeff[c*nprim + p]
  int nprim() const { return static_cast<int>(alpha.size()); }
  int nctr() const {
    const int np = nprim();
    return np ? static_cast<int>(coeff.size()) / np : 0;
  }
};

/// A generally-contracted basis: contracted shells plus contracted-AO offsets.
/// Shell s contributes nctr(s)*ncart(l(s)) AOs.
template <class Real = double> struct ContractedBasis {
  std::vector<ContractedShell<Real>> shells;
  std::vector<int> ao_off; ///< prefix offsets (nshell+1)
  int nao{0};
};

/// Build a ContractedBasis from its shells, filling the contracted-AO offsets.
template <class Real>
ContractedBasis<Real> make_contracted_basis(std::vector<ContractedShell<Real>> shells) {
  ContractedBasis<Real> b;
  b.shells = std::move(shells);
  b.ao_off.resize(b.shells.size() + 1, 0);
  for (std::size_t i = 0; i < b.shells.size(); ++i)
    b.ao_off[i + 1] =
        b.ao_off[i] + b.shells[i].nctr() * ncart(b.shells[i].l);
  b.nao = b.ao_off.back();
  return b;
}

namespace detail {

/// A single primitive of a contracted shell, as the engine's PrimitiveShell.
template <class Real>
PrimitiveShell<Real> contracted_prim(const ContractedShell<Real> &s, int p) {
  return PrimitiveShell<Real>{s.alpha[p], {s.center[0], s.center[1], s.center[2]}, s.l};
}

/// LKC-style contracted scatter: write a contracted shell-pair block `cblk`
/// (row-major over (cA*nca+ka, cB*ncb+kb)) into the AO matrix `M` with the
/// contracted-AO offsets and a transpose mirror (mirror: +1 sym, -1 antisym,
/// 0 none). The (a==b) diagonal shell writes the symmetric block once per
/// element; off-diagonal shell pairs own disjoint AO ranges.
template <class Real>
void scatter_contracted(std::vector<Real> &M, const ContractedBasis<Real> &basis,
                        int a, int b, int mirror, const std::vector<Real> &cblk) {
  const int la = basis.shells[a].l, lb = basis.shells[b].l;
  const int nca = ncart(la), ncb = ncart(lb);
  const int nctA = basis.shells[a].nctr(), nctB = basis.shells[b].nctr();
  const std::size_t nao = static_cast<std::size_t>(basis.nao);
  const std::size_t oa = basis.ao_off[a], ob = basis.ao_off[b];
  const int rowB = nctB * ncb;
  for (int cA = 0; cA < nctA; ++cA)
    for (int ka = 0; ka < nca; ++ka) {
      const std::size_t i = oa + static_cast<std::size_t>(cA) * nca + ka;
      for (int cB = 0; cB < nctB; ++cB)
        for (int kb = 0; kb < ncb; ++kb) {
          const std::size_t j = ob + static_cast<std::size_t>(cB) * ncb + kb;
          const Real v = cblk[(static_cast<std::size_t>(cA) * nca + ka) * rowB +
                              cB * ncb + kb];
          M[i * nao + j] = v;
          if (mirror && i != j) M[j * nao + i] = mirror > 0 ? v : -v;
        }
    }
}

} // namespace detail

/// Overlap matrix S over a generally-contracted basis (nao x nao, row-major,
/// PySCF cart=True normalization). Primitive 1D tables are evaluated once per
/// primitive pair and shared across all contracted-function/Cartesian pairs.
template <class Real>
std::vector<Real> overlap_matrix(const ContractedBasis<Real> &basis,
                                 Real tau = Real(0)) {
  const int nao = basis.nao;
  std::vector<Real> S(static_cast<std::size_t>(nao) * nao, Real(0));
  const int ns = static_cast<int>(basis.shells.size());
  for (int a = 0; a < ns; ++a)
    for (int b = a; b < ns; ++b) {
      const auto &A = basis.shells[a], &B = basis.shells[b];
      const int la = A.l, lb = B.l, nca = ncart(la), ncb = ncart(lb);
      const int npa = A.nprim(), npb = B.nprim(), nctA = A.nctr(), nctB = B.nctr();
      const int rowB = nctB * ncb;
      std::vector<Real> cblk(static_cast<std::size_t>(nctA) * nca * rowB, Real(0));
      for (int pa = 0; pa < npa; ++pa) {
        const Real na = cart_norm_pyscf(la, A.alpha[pa]);
        for (int pb = 0; pb < npb; ++pb) {
          if (detail::pair_gauss_prefactor(detail::contracted_prim(A, pa),
                                           detail::contracted_prim(B, pb)) <= tau)
            continue;
          const Real nb = cart_norm_pyscf(lb, B.alpha[pb]);
          std::vector<Real> sx, sy, sz;
          int lbx;
          detail::overlap_1d(A.alpha[pa], A.center[0], B.alpha[pb], B.center[0], la, lb, 0, 0, sx, lbx);
          detail::overlap_1d(A.alpha[pa], A.center[1], B.alpha[pb], B.center[1], la, lb, 0, 0, sy, lbx);
          detail::overlap_1d(A.alpha[pa], A.center[2], B.alpha[pb], B.center[2], la, lb, 0, 0, sz, lbx);
          for (int ka = 0; ka < nca; ++ka) {
            int a3[3];
            cart_comp(la, ka, a3[0], a3[1], a3[2]);
            for (int kb = 0; kb < ncb; ++kb) {
              int b3[3];
              cart_comp(lb, kb, b3[0], b3[1], b3[2]);
              const Real prim = sx[a3[0] * (lbx + 1) + b3[0]] *
                                sy[a3[1] * (lbx + 1) + b3[1]] *
                                sz[a3[2] * (lbx + 1) + b3[2]];
              for (int cA = 0; cA < nctA; ++cA) {
                const Real wa = A.coeff[cA * npa + pa] * na;
                if (wa == Real(0)) continue;
                for (int cB = 0; cB < nctB; ++cB) {
                  const Real wb = B.coeff[cB * npb + pb] * nb;
                  cblk[(static_cast<std::size_t>(cA) * nca + ka) * rowB + cB * ncb + kb] +=
                      wa * wb * prim;
                }
              }
            }
          }
        }
      }
      detail::scatter_contracted(S, basis, a, b, +1, cblk);
    }
  return S;
}

/// Kinetic-energy matrix T = -1/2 <a|nabla^2|b> over a generally-contracted
/// basis (PySCF cart=True normalization).
template <class Real>
std::vector<Real> kinetic_matrix(const ContractedBasis<Real> &basis,
                                 Real tau = Real(0)) {
  const int nao = basis.nao;
  std::vector<Real> T(static_cast<std::size_t>(nao) * nao, Real(0));
  const int ns = static_cast<int>(basis.shells.size());
  for (int a = 0; a < ns; ++a)
    for (int b = a; b < ns; ++b) {
      const auto &A = basis.shells[a], &B = basis.shells[b];
      const int la = A.l, lb = B.l, nca = ncart(la), ncb = ncart(lb), lb1 = lb + 1;
      const int npa = A.nprim(), npb = B.nprim(), nctA = A.nctr(), nctB = B.nctr();
      const int rowB = nctB * ncb;
      std::vector<Real> cblk(static_cast<std::size_t>(nctA) * nca * rowB, Real(0));
      for (int pa = 0; pa < npa; ++pa) {
        const Real na = cart_norm_pyscf(la, A.alpha[pa]);
        for (int pb = 0; pb < npb; ++pb) {
          if (detail::pair_gauss_prefactor(detail::contracted_prim(A, pa),
                                           detail::contracted_prim(B, pb)) <= tau)
            continue;
          const Real nb = cart_norm_pyscf(lb, B.alpha[pb]);
          std::vector<Real> sx, sy, sz, tx, ty, tz;
          int lbx;
          detail::overlap_1d(A.alpha[pa], A.center[0], B.alpha[pb], B.center[0], la, lb, 0, 2, sx, lbx);
          detail::kinetic_1d(sx, lbx, la, lb, B.alpha[pb], tx);
          detail::overlap_1d(A.alpha[pa], A.center[1], B.alpha[pb], B.center[1], la, lb, 0, 2, sy, lbx);
          detail::kinetic_1d(sy, lbx, la, lb, B.alpha[pb], ty);
          detail::overlap_1d(A.alpha[pa], A.center[2], B.alpha[pb], B.center[2], la, lb, 0, 2, sz, lbx);
          detail::kinetic_1d(sz, lbx, la, lb, B.alpha[pb], tz);
          auto Sx = [&](int i, int j) { return sx[i * (lbx + 1) + j]; };
          auto Sy = [&](int i, int j) { return sy[i * (lbx + 1) + j]; };
          auto Sz = [&](int i, int j) { return sz[i * (lbx + 1) + j]; };
          for (int ka = 0; ka < nca; ++ka) {
            int a3[3];
            cart_comp(la, ka, a3[0], a3[1], a3[2]);
            for (int kb = 0; kb < ncb; ++kb) {
              int b3[3];
              cart_comp(lb, kb, b3[0], b3[1], b3[2]);
              const Real prim =
                  tx[a3[0] * lb1 + b3[0]] * Sy(a3[1], b3[1]) * Sz(a3[2], b3[2]) +
                  Sx(a3[0], b3[0]) * ty[a3[1] * lb1 + b3[1]] * Sz(a3[2], b3[2]) +
                  Sx(a3[0], b3[0]) * Sy(a3[1], b3[1]) * tz[a3[2] * lb1 + b3[2]];
              for (int cA = 0; cA < nctA; ++cA) {
                const Real wa = A.coeff[cA * npa + pa] * na;
                if (wa == Real(0)) continue;
                for (int cB = 0; cB < nctB; ++cB) {
                  const Real wb = B.coeff[cB * npb + pb] * nb;
                  cblk[(static_cast<std::size_t>(cA) * nca + ka) * rowB + cB * ncb + kb] +=
                      wa * wb * prim;
                }
              }
            }
          }
        }
      }
      detail::scatter_contracted(T, basis, a, b, +1, cblk);
    }
  return T;
}

} // namespace intti
