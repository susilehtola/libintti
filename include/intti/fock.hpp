// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Shell-level AO-matrix Fock builders: the user-facing entry point wrapping
// the pair-space Hermite t-space kernels.

#include <cmath>
#include <utility>
#include <vector>

#include "batch.hpp"
#include "jbuild.hpp"
#include "kbuild.hpp"
#include "tgrid.hpp"

namespace intti {

/// Shell list with Cartesian AO offsets.
template <class Real> struct ShellBasis {
  std::vector<PrimitiveShell<Real>> shells;
  std::vector<int> ao_off;
  int nao{0};
};

template <class Real>
ShellBasis<Real> make_basis(std::vector<PrimitiveShell<Real>> shells) {
  ShellBasis<Real> b;
  b.shells = std::move(shells);
  b.ao_off.resize(b.shells.size() + 1, 0);
  for (std::size_t i = 0; i < b.shells.size(); ++i)
    b.ao_off[i + 1] = b.ao_off[i] + ncart(b.shells[i].l);
  b.nao = b.ao_off.back();
  return b;
}

/// Triangular shell-pair list of a basis, plus the (i, j) shell indices of
/// each pair (the row bookkeeping shared by the AO-level builders and the
/// Cholesky unpacking).
template <class Real>
void make_shell_pairs(const ShellBasis<Real> &basis,
                      std::vector<ShellPair<Real>> &plist,
                      std::vector<std::pair<int, int>> &pshell) {
  const int ns = static_cast<int>(basis.shells.size());
  plist.clear();
  pshell.clear();
  for (int i = 0; i < ns; ++i)
    for (int j = i; j < ns; ++j) {
      plist.push_back(make_pair(basis.shells[i], basis.shells[j]));
      pshell.push_back({i, j});
    }
}

/// Cauchy-Schwarz factors Q_p = sqrt(max_comp (p_comp | p_comp)) per pair,
/// from one batched diagonal call.
template <class Real>
std::vector<Real> schwarz(const PairTable<Real> &pairs,
                          const std::vector<ShellPair<Real>> &pair_list,
                          const TGrid<Real> &grid) {
  using std::sqrt;
  const int npair = pairs.npair;
  std::vector<std::pair<int, int>> diag(npair);
  for (int ip = 0; ip < npair; ++ip)
    diag[ip] = {ip, ip};
  auto batch = make_batch(pairs, diag);
  Kokkos::View<Real *> out("intti::schwarz", batch.nout_total);
  QuartetWorkspace<Real> ws;
  eri_quartets(pairs, batch, grid, out, ws);
  auto oh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  std::vector<Real> Q(npair);
  int off = 0;
  for (int ip = 0; ip < npair; ++ip) {
    const int nc = ncart(pair_list[ip].la) * ncart(pair_list[ip].lb);
    Real qmax = 0;
    for (int c = 0; c < nc; ++c) {
      const Real v = oh(off + c * nc + c);
      if (v > qmax) qmax = v;
    }
    Q[ip] = sqrt(qmax);
    off += nc * nc;
  }
  return Q;
}

/// Coulomb matrix at the AO level: J = sum_cd D_cd (ab|cd), D and J are
/// nao x nao row-major Cartesian AO matrices. Screening drops ket pairs
/// whose Schwarz-bounded contribution falls below tau (0 = off).
///
/// MPI distribution (M8): rank/nranks partition the bra shell-pair index
/// space (see jbuild.hpp::coulomb_build); defaults reproduce the serial
/// result byte-for-byte. See mpi.hpp for the MPI_Allreduce-wrapped entry
/// point.
template <class Real>
void coulomb_build(const ShellBasis<Real> &basis, const Real *D,
                   const TGrid<Real> &grid, Real *J, Real tau = Real(0),
                   int rank = 0, int nranks = 1) {
  std::vector<ShellPair<Real>> plist;
  std::vector<std::pair<int, int>> pshell;
  make_shell_pairs(basis, plist, pshell);
  auto tab = make_pair_table(plist);
  const int npair = tab.npair;
  // pair-product density with the symmetry fold (D_ij + D_ji off-diagonal)
  std::vector<int> off(npair + 1, 0);
  for (int p = 0; p < npair; ++p)
    off[p + 1] = off[p] + ncart(plist[p].la) * ncart(plist[p].lb);
  std::vector<Real> Dp(off[npair]), Jp(off[npair]);
  const int nao = basis.nao;
  for (int p = 0; p < npair; ++p) {
    const auto [i, j] = pshell[p];
    const int ncb = ncart(plist[p].lb);
    for (int ka = 0; ka < ncart(plist[p].la); ++ka)
      for (int kb = 0; kb < ncb; ++kb) {
        const int r = basis.ao_off[i] + ka, c = basis.ao_off[j] + kb;
        Dp[off[p] + ka * ncb + kb] =
            i == j ? D[r * nao + c] : D[r * nao + c] + D[c * nao + r];
      }
  }
  // screening bounds
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
  // scatter to the AO matrix (J is symmetric)
  for (int p = 0; p < npair; ++p) {
    const auto [i, j] = pshell[p];
    const int ncb = ncart(plist[p].lb);
    for (int ka = 0; ka < ncart(plist[p].la); ++ka)
      for (int kb = 0; kb < ncb; ++kb) {
        const int r = basis.ao_off[i] + ka, c = basis.ao_off[j] + kb;
        J[r * nao + c] = Jp[off[p] + ka * ncb + kb];
        J[c * nao + r] = J[r * nao + c];
      }
  }
}

template <class Real>
void exchange_build(const ShellBasis<Real> &basis, const Real *D,
                    const TGrid<Real> &grid, Real *K, Real tau, int rank,
                    int nranks) {
  const int ns = static_cast<int>(basis.shells.size());
  // full rectangular (a, c) pair set: the density index straddles the pairs
  std::vector<ShellPair<Real>> plist;
  plist.reserve(static_cast<std::size_t>(ns) * ns);
  for (int a = 0; a < ns; ++a)
    for (int c = 0; c < ns; ++c)
      plist.push_back(make_pair(basis.shells[a], basis.shells[c]));
  auto tab = make_pair_table(plist);
  auto Q = schwarz(tab, plist, grid);
  detail::exchange_build_impl(basis.shells, basis.ao_off, basis.nao, D, grid, K,
                              tau, Q, tab, rank, nranks);
}

} // namespace intti
