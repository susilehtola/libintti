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

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "gto.hpp"
#include "normalization.hpp"
#include "deriv.hpp"      // detail::overlap_deriv_block / kinetic_ / nuclear_
#include "nuclear.hpp"    // detail::attraction_pair_block, PointCharge
#include "oneel.hpp" // detail::overlap_1d / kinetic_1d / multipole_1d, pair_gauss_prefactor
#include "tgrid.hpp"

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

/// Effective coefficient of primitive p in contracted function c of a shell:
/// the basis-set coefficient times the primitive's PySCF cart=True
/// normalization -- the weight on the engine's UNNORMALIZED primitive.
template <class Real>
Real effective_coeff(const ContractedShell<Real> &s, int c, int p) {
  return s.coeff[c * s.nprim() + p] * cart_norm_pyscf(s.l, s.alpha[p]);
}

/// Expand a contracted basis into its primitive shells, recording for each
/// primitive shell its origin (contracted shell index, primitive index). This
/// is the primitive pair space the fused J/K engines run on; the contraction
/// enters only in the density-fold and output-gather, never as an nprim x nprim
/// matrix.
template <class Real>
void contracted_primitives(const ContractedBasis<Real> &basis,
                           std::vector<PrimitiveShell<Real>> &prims,
                           std::vector<int> &cshell, std::vector<int> &cprim) {
  prims.clear();
  cshell.clear();
  cprim.clear();
  for (int a = 0; a < static_cast<int>(basis.shells.size()); ++a)
    for (int p = 0; p < basis.shells[a].nprim(); ++p) {
      prims.push_back(contracted_prim(basis.shells[a], p));
      cshell.push_back(a);
      cprim.push_back(p);
    }
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

/// Generic contraction driver for symmetric 1e operators with `ncomp` matrix
/// components. `block(sa, sb, out)` fills the UNNORMALIZED primitive block of a
/// single primitive pair (sa, sb) as component-major out[(comp*nca+ka)*ncb+kb]
/// (nca=ncart(sa.l), ncb=ncart(sb.l)). The driver evaluates that block ONCE per
/// primitive pair and accumulates it, weighted by d_{cp} cart_norm_pyscf, into
/// every contracted-function pair -- the shared-intermediate general-contraction
/// path. Returns `ncomp` nao x nao matrices; `mirror` sets the transpose sign
/// (+1 symmetric, -1 antisymmetric, 0 = no symmetry). With mirror = 0 the
/// driver walks ALL ordered shell pairs and writes each block once, which is
/// what derivative matrices need: <nabla mu|nu> is neither symmetric nor
/// antisymmetric in general (translational invariance does make the S and T
/// gradients antisymmetric, but the nuclear-attraction gradient is not, so the
/// derivative builders take one uniform ordered-pair path). A pair whose
/// Gaussian prefactor is <= tau is skipped (tau=0 skips only exactly-zero
/// blocks).
template <class Real, class BlockFn>
std::vector<std::vector<Real>>
contracted_1e_multi(const ContractedBasis<Real> &basis, int ncomp, int mirror,
                    Real tau, BlockFn block) {
  const int nao = basis.nao;
  std::vector<std::vector<Real>> out(
      ncomp, std::vector<Real>(static_cast<std::size_t>(nao) * nao, Real(0)));
  const int ns = static_cast<int>(basis.shells.size());
  std::vector<Real> pblk, cblk;
  for (int a = 0; a < ns; ++a)
    for (int b = (mirror ? a : 0); b < ns; ++b) {
      const auto &A = basis.shells[a], &B = basis.shells[b];
      const int la = A.l, lb = B.l, nca = ncart(la), ncb = ncart(lb);
      const int npa = A.nprim(), npb = B.nprim(), nctA = A.nctr(), nctB = B.nctr();
      const int rowB = nctB * ncb;
      const std::size_t pstride = static_cast<std::size_t>(nca) * ncb;
      const std::size_t cstride = static_cast<std::size_t>(nctA) * nca * rowB;
      pblk.assign(static_cast<std::size_t>(ncomp) * pstride, Real(0));
      cblk.assign(static_cast<std::size_t>(ncomp) * cstride, Real(0));
      for (int pa = 0; pa < npa; ++pa) {
        const Real na = cart_norm_pyscf(la, A.alpha[pa]);
        const auto spa = detail::contracted_prim(A, pa);
        for (int pb = 0; pb < npb; ++pb) {
          const auto spb = detail::contracted_prim(B, pb);
          if (detail::pair_gauss_prefactor(spa, spb) <= tau) continue;
          const Real nb = cart_norm_pyscf(lb, B.alpha[pb]);
          block(spa, spb, pblk.data());
          for (int cA = 0; cA < nctA; ++cA) {
            const Real wa = A.coeff[cA * npa + pa] * na;
            if (wa == Real(0)) continue;
            for (int cB = 0; cB < nctB; ++cB) {
              const Real w = wa * B.coeff[cB * npb + pb] * nb;
              if (w == Real(0)) continue;
              for (int comp = 0; comp < ncomp; ++comp) {
                const Real *pb_c = pblk.data() + comp * pstride;
                Real *cb_c = cblk.data() + comp * cstride;
                for (int ka = 0; ka < nca; ++ka)
                  for (int kb = 0; kb < ncb; ++kb)
                    cb_c[(static_cast<std::size_t>(cA) * nca + ka) * rowB + cB * ncb + kb] +=
                        w * pb_c[ka * ncb + kb];
              }
            }
          }
        }
      }
      for (int comp = 0; comp < ncomp; ++comp) {
        std::vector<Real> cslice(cblk.begin() + static_cast<std::ptrdiff_t>(comp) * cstride,
                                 cblk.begin() + static_cast<std::ptrdiff_t>(comp + 1) * cstride);
        detail::scatter_contracted(out[comp], basis, a, b, mirror, cslice);
      }
    }
  return out;
}

} // namespace detail

/// Overlap matrix S over a generally-contracted basis (nao x nao, row-major,
/// PySCF cart=True normalization). Primitive 1D tables are evaluated once per
/// primitive pair and shared across all contracted-function/Cartesian pairs.
template <class Real>
std::vector<Real> overlap_matrix(const ContractedBasis<Real> &basis,
                                 Real tau = Real(0)) {
  auto out = detail::contracted_1e_multi(
      basis, 1, +1, tau,
      [](const PrimitiveShell<Real> &sa, const PrimitiveShell<Real> &sb, Real *o) {
        const int la = sa.l, lb = sb.l, ncb = ncart(lb);
        std::vector<Real> sx, sy, sz;
        int lbx;
        detail::overlap_1d(sa.alpha, sa.center[0], sb.alpha, sb.center[0], la, lb, 0, 0, sx, lbx);
        detail::overlap_1d(sa.alpha, sa.center[1], sb.alpha, sb.center[1], la, lb, 0, 0, sy, lbx);
        detail::overlap_1d(sa.alpha, sa.center[2], sb.alpha, sb.center[2], la, lb, 0, 0, sz, lbx);
        for (int ka = 0; ka < ncart(la); ++ka) {
          int a3[3];
          cart_comp(la, ka, a3[0], a3[1], a3[2]);
          for (int kb = 0; kb < ncb; ++kb) {
            int b3[3];
            cart_comp(lb, kb, b3[0], b3[1], b3[2]);
            o[ka * ncb + kb] = sx[a3[0] * (lbx + 1) + b3[0]] *
                               sy[a3[1] * (lbx + 1) + b3[1]] *
                               sz[a3[2] * (lbx + 1) + b3[2]];
          }
        }
      });
  return std::move(out[0]);
}

/// Kinetic-energy matrix T = -1/2 <a|nabla^2|b> over a generally-contracted
/// basis (PySCF cart=True normalization).
template <class Real>
std::vector<Real> kinetic_matrix(const ContractedBasis<Real> &basis,
                                 Real tau = Real(0)) {
  auto out = detail::contracted_1e_multi(
      basis, 1, +1, tau,
      [](const PrimitiveShell<Real> &sa, const PrimitiveShell<Real> &sb, Real *o) {
        const int la = sa.l, lb = sb.l, ncb = ncart(lb), lb1 = lb + 1;
        std::vector<Real> sx, sy, sz, tx, ty, tz;
        int lbx;
        detail::overlap_1d(sa.alpha, sa.center[0], sb.alpha, sb.center[0], la, lb, 0, 2, sx, lbx);
        detail::kinetic_1d(sx, lbx, la, lb, sb.alpha, tx);
        detail::overlap_1d(sa.alpha, sa.center[1], sb.alpha, sb.center[1], la, lb, 0, 2, sy, lbx);
        detail::kinetic_1d(sy, lbx, la, lb, sb.alpha, ty);
        detail::overlap_1d(sa.alpha, sa.center[2], sb.alpha, sb.center[2], la, lb, 0, 2, sz, lbx);
        detail::kinetic_1d(sz, lbx, la, lb, sb.alpha, tz);
        auto Sx = [&](int i, int j) { return sx[i * (lbx + 1) + j]; };
        auto Sy = [&](int i, int j) { return sy[i * (lbx + 1) + j]; };
        auto Sz = [&](int i, int j) { return sz[i * (lbx + 1) + j]; };
        for (int ka = 0; ka < ncart(la); ++ka) {
          int a3[3];
          cart_comp(la, ka, a3[0], a3[1], a3[2]);
          for (int kb = 0; kb < ncb; ++kb) {
            int b3[3];
            cart_comp(lb, kb, b3[0], b3[1], b3[2]);
            o[ka * ncb + kb] =
                tx[a3[0] * lb1 + b3[0]] * Sy(a3[1], b3[1]) * Sz(a3[2], b3[2]) +
                Sx(a3[0], b3[0]) * ty[a3[1] * lb1 + b3[1]] * Sz(a3[2], b3[2]) +
                Sx(a3[0], b3[0]) * Sy(a3[1], b3[1]) * tz[a3[2] * lb1 + b3[2]];
          }
        }
      });
  return std::move(out[0]);
}

/// Cartesian multipole matrices <a|(x-O)^ex (y-O)^ey (z-O)^ez|b> over a
/// generally-contracted basis, one per (ex,ey,ez) with total order <= max_order
/// in the same component order as multipole_labels() (order 0 = overlap). PySCF
/// cart=True normalization.
template <class Real>
std::vector<std::vector<Real>> multipole_matrices(const ContractedBasis<Real> &basis,
                                                  int max_order, const Real origin[3],
                                                  Real tau = Real(0)) {
  const auto comps = multipole_labels(max_order);
  const int ncomp = static_cast<int>(comps.size());
  return detail::contracted_1e_multi(
      basis, ncomp, +1, tau,
      [&](const PrimitiveShell<Real> &sa, const PrimitiveShell<Real> &sb, Real *o) {
        const int la = sa.l, lb = sb.l, nca = ncart(la), ncb = ncart(lb);
        std::vector<Real> s[3], m[3];
        int lbx;
        for (int d = 0; d < 3; ++d) {
          detail::overlap_1d(sa.alpha, sa.center[d], sb.alpha, sb.center[d], la, lb,
                             max_order, 0, s[d], lbx);
          detail::multipole_1d(s[d], lbx, la, lb, sa.center[d], origin[d], max_order, m[d]);
        }
        auto M = [&](int d, int e, int i, int j) {
          return m[d][(static_cast<std::size_t>(e) * (la + 1) + i) * (lb + 1) + j];
        };
        for (int ci = 0; ci < ncomp; ++ci) {
          const auto &e = comps[ci];
          Real *oc = o + static_cast<std::size_t>(ci) * nca * ncb;
          for (int ka = 0; ka < nca; ++ka) {
            int a3[3];
            cart_comp(la, ka, a3[0], a3[1], a3[2]);
            for (int kb = 0; kb < ncb; ++kb) {
              int b3[3];
              cart_comp(lb, kb, b3[0], b3[1], b3[2]);
              oc[ka * ncb + kb] = M(0, e[0], a3[0], b3[0]) * M(1, e[1], a3[1], b3[1]) *
                                  M(2, e[2], a3[2], b3[2]);
            }
          }
        }
      });
}

/// Nuclear-attraction matrix V_ab = sum_c weight_c <a|1/|r-R_c||b> over a
/// generally-contracted basis (PySCF cart=True normalization; pass charges with
/// weight = -Z, e.g. via nuclei_as_charges, to match int1e_nuc). far_tau > 0
/// enables the FMM far branch; tau screens contracted primitive pairs.
template <class Real>
std::vector<Real> nuclear_matrix(const ContractedBasis<Real> &basis,
                                 const std::vector<PointCharge<Real>> &charges,
                                 const TGrid<Real> &grid, Real tau = Real(0),
                                 Real far_tau = Real(0)) {
  auto out = detail::contracted_1e_multi(
      basis, 1, +1, tau,
      [&](const PrimitiveShell<Real> &sa, const PrimitiveShell<Real> &sb, Real *o) {
        detail::attraction_pair_block(sa, sb, charges, grid, Real(0), far_tau, o);
      });
  return std::move(out[0]);
}

/// Coulomb matrix J = sum_{KL} D_{KL} (IJ|KL) over a generally-contracted basis
/// (D, J are nao x nao row-major contracted-AO matrices; PySCF cart=True
/// normalization). Reuses the fused primitive J-engine (jbuild.hpp) on the
/// PRIMITIVE pairs of the contracted basis: the contraction enters only in the
/// density fold (contracted D -> per-primitive-pair effective density,
/// D_eff = C^T D C applied block-by-block) and the output gather (per-primitive-
/// pair J -> contracted J = C J_eff C^T), so no nao_prim x nao_prim matrix is
/// ever materialized. tau screens primitive pairs (Schwarz x density bound).
template <class Real>
void coulomb_build(const ContractedBasis<Real> &basis, const Real *D,
                   const TGrid<Real> &grid, Real *J, Real tau = Real(0),
                   int rank = 0, int nranks = 1) {
  const std::size_t nao = static_cast<std::size_t>(basis.nao);
  std::vector<PrimitiveShell<Real>> prims;
  std::vector<int> cs, cp;
  detail::contracted_primitives(basis, prims, cs, cp);
  const int nps = static_cast<int>(prims.size());
  std::vector<ShellPair<Real>> plist;
  std::vector<std::pair<int, int>> pshell;
  for (int i = 0; i < nps; ++i)
    for (int j = i; j < nps; ++j) {
      plist.push_back(make_pair(prims[i], prims[j]));
      pshell.push_back({i, j});
    }
  auto tab = make_pair_table(plist);
  const int npair = tab.npair;
  std::vector<int> off(npair + 1, 0);
  for (int p = 0; p < npair; ++p)
    off[p + 1] = off[p] + ncart(plist[p].la) * ncart(plist[p].lb);
  std::vector<Real> Dp(off[npair], Real(0)), Jp(off[npair], Real(0));
  // density fold: Dp[p] = C^T D C on this primitive pair, with the D_ij+D_ji
  // symmetry fold for distinct primitive shells (matching the engine).
  for (int p = 0; p < npair; ++p) {
    const int i = pshell[p].first, j = pshell[p].second;
    const int A = cs[i], B = cs[j], pa = cp[i], pb = cp[j];
    const int nca = ncart(plist[p].la), ncb = ncart(plist[p].lb);
    const int nctA = basis.shells[A].nctr(), nctB = basis.shells[B].nctr();
    const bool diag = (i == j);
    for (int ka = 0; ka < nca; ++ka)
      for (int kb = 0; kb < ncb; ++kb) {
        Real s = 0;
        for (int cA = 0; cA < nctA; ++cA) {
          const Real wA = detail::effective_coeff(basis.shells[A], cA, pa);
          if (wA == Real(0)) continue;
          const std::size_t I = basis.ao_off[A] + static_cast<std::size_t>(cA) * nca + ka;
          for (int cB = 0; cB < nctB; ++cB) {
            const std::size_t Jc = basis.ao_off[B] + static_cast<std::size_t>(cB) * ncb + kb;
            const Real dval = diag ? D[I * nao + Jc] : D[I * nao + Jc] + D[Jc * nao + I];
            s += wA * detail::effective_coeff(basis.shells[B], cB, pb) * dval;
          }
        }
        Dp[off[p] + ka * ncb + kb] = s;
      }
  }
  std::vector<Real> Q, bound;
  const Real *Qp = nullptr, *bp = nullptr;
  if (tau > Real(0)) {
    Q = schwarz(tab, plist, grid);
    bound.resize(npair);
    for (int p = 0; p < npair; ++p) {
      Real dmax = 0;
      for (int c = off[p]; c < off[p + 1]; ++c) {
        const Real a = Dp[c] < 0 ? -Dp[c] : Dp[c];
        if (a > dmax) dmax = a;
      }
      bound[p] = Q[p] * dmax;
    }
    Qp = Q.data();
    bp = bound.data();
  }
  coulomb_build(tab, Dp.data(), grid, Jp.data(), Qp, bp, tau, rank, nranks);
  // output gather: J = C J_eff C^T, accumulated per primitive pair (many
  // primitive pairs contribute to each contracted block).
  for (std::size_t i = 0; i < nao * nao; ++i) J[i] = Real(0);
  for (int p = 0; p < npair; ++p) {
    const int i = pshell[p].first, j = pshell[p].second;
    const int A = cs[i], B = cs[j], pa = cp[i], pb = cp[j];
    const int nca = ncart(plist[p].la), ncb = ncart(plist[p].lb);
    const int nctA = basis.shells[A].nctr(), nctB = basis.shells[B].nctr();
    const bool offdiag = (i != j);
    for (int ka = 0; ka < nca; ++ka)
      for (int kb = 0; kb < ncb; ++kb) {
        const Real jval = Jp[off[p] + ka * ncb + kb];
        if (jval == Real(0)) continue;
        for (int cA = 0; cA < nctA; ++cA) {
          const Real wA = detail::effective_coeff(basis.shells[A], cA, pa);
          if (wA == Real(0)) continue;
          const std::size_t I = basis.ao_off[A] + static_cast<std::size_t>(cA) * nca + ka;
          for (int cB = 0; cB < nctB; ++cB) {
            const std::size_t Jc = basis.ao_off[B] + static_cast<std::size_t>(cB) * ncb + kb;
            const Real add = wA * detail::effective_coeff(basis.shells[B], cB, pb) * jval;
            J[I * nao + Jc] += add;
            if (offdiag) J[Jc * nao + I] += add;
          }
        }
      }
  }
}

/// Exchange matrix K_{IJ} = sum_{KL} D_{KL} (IK|JL) over a generally-contracted
/// basis (D, K are nao x nao row-major contracted-AO matrices; PySCF cart=True
/// normalization). Memory-lean: reuses the primitive exchange t-space core on
/// the primitive pairs of the contracted basis, keeping D and K contracted (no
/// nao_prim x nao_prim matrix); see detail::exchange_build_contracted_impl.
/// tau > 0 enables Schwarz x effective-density screening of primitive ket pairs
/// (Q(ac) Q(bd) max|D_eff(c,d)| < tau skipped); tau = 0 (default) is exact.
template <class Real>
void exchange_build(const ContractedBasis<Real> &basis, const Real *D,
                    const TGrid<Real> &grid, Real *K, Real tau = Real(0)) {
  std::vector<PrimitiveShell<Real>> prims;
  std::vector<int> cshell, cprim;
  detail::contracted_primitives(basis, prims, cshell, cprim);
  const int nps = static_cast<int>(prims.size());
  const int ncs = static_cast<int>(basis.shells.size());
  const int naoc = basis.nao;
  // flat effective-coefficient pool: per shell A, [cA-major, prim-minor]
  std::vector<int> ecoff(ncs, 0), nprim_c(ncs), nctr_c(ncs);
  int etot = 0;
  for (int A = 0; A < ncs; ++A) {
    ecoff[A] = etot;
    nprim_c[A] = basis.shells[A].nprim();
    nctr_c[A] = basis.shells[A].nctr();
    etot += nprim_c[A] * nctr_c[A];
  }
  std::vector<Real> ecoef(etot);
  for (int A = 0; A < ncs; ++A)
    for (int cA = 0; cA < nctr_c[A]; ++cA)
      for (int p = 0; p < nprim_c[A]; ++p)
        ecoef[ecoff[A] + cA * nprim_c[A] + p] = detail::effective_coeff(basis.shells[A], cA, p);
  // rectangular primitive (i, c) pair table: pair index p = i*nps + c
  std::vector<ShellPair<Real>> plist;
  plist.reserve(static_cast<std::size_t>(nps) * nps);
  for (int i = 0; i < nps; ++i)
    for (int c = 0; c < nps; ++c) plist.push_back(make_pair(prims[i], prims[c]));
  auto tab = make_pair_table(plist);
  // screening data: Schwarz per primitive pair and the per-(c,d)-primitive-pair
  // maximum of the effective (contracted) ket density.
  std::vector<Real> Q, maxDeff;
  if (tau > Real(0)) {
    Q = schwarz(tab, plist, grid);
    maxDeff.assign(static_cast<std::size_t>(nps) * nps, Real(0));
    for (int c = 0; c < nps; ++c)
      for (int d = 0; d < nps; ++d) {
        const int C = cshell[c], pc = cprim[c], Dsh = cshell[d], pd = cprim[d];
        const int ncc = ncart(prims[c].l), ncd = ncart(prims[d].l);
        Real m = 0;
        for (int kc = 0; kc < ncc; ++kc)
          for (int kd = 0; kd < ncd; ++kd) {
            Real s = 0;
            for (int cC = 0; cC < nctr_c[C]; ++cC) {
              const Real wc = ecoef[ecoff[C] + cC * nprim_c[C] + pc];
              for (int cD = 0; cD < nctr_c[Dsh]; ++cD)
                s += wc * ecoef[ecoff[Dsh] + cD * nprim_c[Dsh] + pd] *
                     D[(basis.ao_off[C] + static_cast<std::size_t>(cC) * ncc + kc) * naoc +
                       basis.ao_off[Dsh] + static_cast<std::size_t>(cD) * ncd + kd];
            }
            const Real a = s < 0 ? -s : s;
            if (a > m) m = a;
          }
        maxDeff[static_cast<std::size_t>(c) * nps + d] = m;
      }
  }
  detail::exchange_build_contracted_impl(prims, cshell, cprim, ecoef, ecoff, nprim_c,
                                         nctr_c, basis.ao_off, naoc, D, grid, tab, tau, Q,
                                         maxDeff, K);
}

namespace detail {

/// Fan-out from a PRIMITIVE shell to the contracted AOs it feeds.
///
/// The two-electron DERIVATIVE kernels are driven over primitive shells, since
/// the MD shift acts on a primitive (each primitive carries its own alpha,
/// while the contraction coefficient is a position-independent constant). A
/// generally-contracted basis is therefore handled by expanding it to its
/// primitives and letting the DIGEST -- not the integral evaluation -- carry the
/// contraction: every primitive quartet is evaluated exactly once and scattered,
/// coefficient-weighted, into all nctr^4 contracted index combinations it feeds.
/// That keeps the shared-intermediate property of the contracted energy
/// builders; a decontract/recontract at the matrix level would instead pay
/// O(nprim^4) integrals and defeat the point of general contraction.
///
/// For an already-primitive basis every nctr is 1 and every weight is 1, so the
/// SAME kernel serves both bases with no second code path and no cost beyond a
/// unit-trip loop -- which is why this is a fan-out map rather than a separate
/// contracted kernel.
///
/// Contracted AO of primitive shell s, contracted function c, Cartesian k:
///     base[s] + c*ncart(l) + k,  weight w[coff[s] + c]
/// Shell-indexed outputs (the geometric Hessian) additionally need
/// parent[s]: the CONTRACTED shell that primitive shell s belongs to, so a
/// (3 nshell) x (3 nshell) result stays indexed by contracted shell -- moving a
/// contracted shell's centre moves all of its primitives together.
template <class Real> struct ShellFanout {
  int nao{0}, nsh{0};
  std::vector<int> nctr, coff, base, parent;
  std::vector<Real> w;
};

/// Trivial fan-out for a basis that is already primitive.
template <class Real> ShellFanout<Real> identity_fanout(const ShellBasis<Real> &b) {
  ShellFanout<Real> f;
  const int ns = static_cast<int>(b.shells.size());
  f.nao = b.nao;
  f.nsh = ns;
  f.nctr.assign(ns, 1);
  f.coff.resize(ns);
  f.base = b.ao_off;
  f.parent.resize(ns);
  f.w.assign(ns, Real(1));
  for (int i = 0; i < ns; ++i) f.coff[i] = f.parent[i] = i;
  return f;
}

/// Expand a contracted basis to primitive shells and build the fan-out that
/// maps each primitive back onto the contracted AOs, carrying the PySCF
/// cart=True effective coefficients (basis-set coefficient times the
/// primitive's cart_norm_pyscf; see contracted.hpp).
template <class Real>
ShellFanout<Real> expand_contracted(const ContractedBasis<Real> &cb,
                                    ShellBasis<Real> &prims) {
  std::vector<PrimitiveShell<Real>> ps;
  std::vector<int> cshell, cprim;
  contracted_primitives(cb, ps, cshell, cprim);
  prims = make_basis(ps);
  ShellFanout<Real> f;
  f.nao = cb.nao;
  f.nsh = static_cast<int>(cb.shells.size());
  const int nps = static_cast<int>(ps.size());
  f.nctr.resize(nps);
  f.coff.resize(nps);
  f.base.resize(nps);
  f.parent = cshell;
  int tot = 0;
  for (int i = 0; i < nps; ++i) {
    const auto &sh = cb.shells[cshell[i]];
    f.nctr[i] = sh.nctr();
    f.coff[i] = tot;
    f.base[i] = cb.ao_off[cshell[i]];
    tot += f.nctr[i];
  }
  f.w.resize(tot);
  for (int i = 0; i < nps; ++i) {
    const auto &sh = cb.shells[cshell[i]];
    for (int c = 0; c < f.nctr[i]; ++c)
      f.w[f.coff[i] + c] = effective_coeff(sh, c, cprim[i]);
  }
  return f;
}

} // namespace detail

// ---- derivative integrals over a generally-contracted basis -----------------
// The shift algebra lives once, in deriv.hpp's per-primitive-pair blocks; here
// it is driven by contracted_1e_multi, so a primitive-pair block is evaluated
// ONCE and shared across every contracted-function pair -- the same
// shared-intermediate property the energy builders have. Without this the
// facade could only differentiate an uncontracted basis, i.e. no standard basis
// set, so gradients and Hessians were a demonstrator rather than something
// runnable. mirror = 0: gradient matrices carry no transpose symmetry.

/// Overlap gradient <nabla mu | nu> over a contracted basis: three nao x nao
/// matrices, PySCF int1e_ipovlp (cart=True) with the bra-gradient convention.
template <class Real>
std::array<std::vector<Real>, 3> overlap_deriv(const ContractedBasis<Real> &basis,
                                               Real tau = Real(0)) {
  auto out = detail::contracted_1e_multi(
      basis, 3, 0, tau,
      [](const PrimitiveShell<Real> &sa, const PrimitiveShell<Real> &sb, Real *o) {
        detail::overlap_deriv_block(sa, sb, o);
      });
  return {std::move(out[0]), std::move(out[1]), std::move(out[2])};
}

/// Kinetic-energy gradient <nabla mu | T | nu> over a contracted basis, PySCF
/// int1e_ipkin.
template <class Real>
std::array<std::vector<Real>, 3> kinetic_deriv(const ContractedBasis<Real> &basis,
                                               Real tau = Real(0)) {
  auto out = detail::contracted_1e_multi(
      basis, 3, 0, tau,
      [](const PrimitiveShell<Real> &sa, const PrimitiveShell<Real> &sb, Real *o) {
        detail::kinetic_deriv_block(sa, sb, o);
      });
  return {std::move(out[0]), std::move(out[1]), std::move(out[2])};
}

/// Nuclear-attraction gradient <nabla mu | sum_C w_C/|r-R_C| | nu> over a
/// contracted basis. Charges carrying w = -Z give PySCF int1e_ipnuc; a single
/// unit charge gives int1e_iprinv. The Hellmann-Feynman dR_C term is a separate
/// operator and is not included.
template <class Real>
std::array<std::vector<Real>, 3>
nuclear_deriv(const ContractedBasis<Real> &basis,
              const std::vector<PointCharge<Real>> &charges, const TGrid<Real> &grid,
              Real tau = Real(0)) {
  auto out = detail::contracted_1e_multi(
      basis, 3, 0, tau,
      [&](const PrimitiveShell<Real> &sa, const PrimitiveShell<Real> &sb, Real *o) {
        detail::nuclear_deriv_block(sa, sb, charges, grid, o);
      });
  return {std::move(out[0]), std::move(out[1]), std::move(out[2])};
}

} // namespace intti
