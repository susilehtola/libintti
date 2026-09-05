// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Grid-RI Fock builders: Coulomb (J) and exchange (K) on the hp-adaptive
// finite-element grid (roadmap M-FE grid-RI). The FE grid is a resolution of
// the identity with the grid as the auxiliary basis -- represent the density
// (J) or the co-densities (K) on the grid, apply the t-adapted DAGE to get the
// Coulomb potential DIRECTLY (no (P|Q)^{-1} fit, no auxiliary set), and contract
// against the AO products. Compared to the analytic GTO route this is fitting-
// free and metric-free; the AO enters ONLY through a pointwise evaluator, so
// higher l (the x^lx y^ly z^lz factor here; real solid harmonics later) and
// contraction are just evaluation, not a transform.
//
// This is a host reference implementation (matrix-level API: density/orbitals
// in, J/K matrix out; never per-quartet). It precomputes the AO values on the
// N^3 grid (memory nao*N^3), so it is intended for correctness/experimentation
// and small-to-moderate grids; a Kokkos/streaming port is future work. The AO
// convention is the library's unnormalized Cartesian primitive, matching
// coulomb_build/exchange_build, so results compare directly to them.

#include <cmath>
#include <cstddef>
#include <vector>

#include "contracted.hpp" // ContractedBasis, detail::effective_coeff
#include "fegrid.hpp"
#include "fock.hpp" // ShellBasis
#include "gto.hpp"  // ncart, cart_comp
#include "tgrid.hpp"

namespace intti {

/// Build an hp FE grid whose per-axis resolution covers every AO-product
/// envelope of `basis`: for shells i, j the product density is a Gaussian of
/// exponent alpha_i + alpha_j centred at the weighted midpoint; collect those
/// (exponent, axis-projected centre) as the 1D Gaussians to resolve to `eps`.
template <class Real>
FEGrid1D<Real> grid_for_basis(const ShellBasis<Real> &basis, Real eps = Real(1e-5),
                              int pmin = 4, int pmax = 16) {
  std::vector<FEGaussian1D<Real>> ax;
  const auto &sh = basis.shells;
  for (std::size_t i = 0; i < sh.size(); ++i)
    for (std::size_t j = 0; j < sh.size(); ++j) {
      const Real p = sh[i].alpha + sh[j].alpha;
      for (int d = 0; d < 3; ++d)
        ax.push_back({p, (sh[i].alpha * sh[i].center[d] + sh[j].alpha * sh[j].center[d]) / p});
    }
  return make_fegrid1d_hp(ax, eps, pmin, pmax);
}

namespace detail {
template <class Real> Real gr_ipow(Real x, int n) {
  Real r = 1;
  for (int i = 0; i < n; ++i) r *= x;
  return r;
}
} // namespace detail

/// Evaluate every (unnormalized Cartesian) AO of `basis` on the 3D grid, in
/// make_basis AO order: the return value ao[a] is the N^3 tensor (row-major
/// ix,iy,iz) of AO a = ao_off[shell] + cart. General in l via cart_comp.
template <class Real>
std::vector<std::vector<Real>> ao_values_on_grid(const ShellBasis<Real> &basis,
                                                 const FEGrid1D<Real> &grid) {
  const int N = grid.N;
  const std::size_t N3 = static_cast<std::size_t>(N) * N * N;
  std::vector<std::vector<Real>> ao(basis.nao, std::vector<Real>(N3));
  const int ns = static_cast<int>(basis.shells.size());
  // per shell / axis: Gaussian and coordinate offset along each axis
  for (int s = 0; s < ns; ++s) {
    const auto &sh = basis.shells[s];
    std::vector<Real> gx(N), gy(N), gz(N), dx(N), dy(N), dz(N);
    for (int i = 0; i < N; ++i) {
      dx[i] = grid.xnode[i] - sh.center[0];
      dy[i] = grid.xnode[i] - sh.center[1];
      dz[i] = grid.xnode[i] - sh.center[2];
      gx[i] = std::exp(-sh.alpha * dx[i] * dx[i]);
      gy[i] = std::exp(-sh.alpha * dy[i] * dy[i]);
      gz[i] = std::exp(-sh.alpha * dz[i] * dz[i]);
    }
    for (int k = 0; k < ncart(sh.l); ++k) {
      int lx, ly, lz;
      cart_comp(sh.l, k, lx, ly, lz);
      Real *out = ao[basis.ao_off[s] + k].data();
      for (int ix = 0; ix < N; ++ix) {
        const Real fx = detail::gr_ipow(dx[ix], lx) * gx[ix];
        for (int iy = 0; iy < N; ++iy) {
          const Real fxy = fx * detail::gr_ipow(dy[iy], ly) * gy[iy];
          const std::size_t base = (static_cast<std::size_t>(ix) * N + iy) * N;
          for (int iz = 0; iz < N; ++iz)
            out[base + iz] = fxy * detail::gr_ipow(dz[iz], lz) * gz[iz];
        }
      }
    }
  }
  return ao;
}

namespace detail {
/// Grid-RI Coulomb from precomputed AO values (n AOs, each an N^3 tensor).
template <class Real>
std::vector<Real> grid_coulomb_from_ao(const std::vector<std::vector<Real>> &ao, int n,
                                       const Real *D, const FEGrid1D<Real> &grid,
                                       const TGrid<Real> &tgrid, int nv) {
  const int N = grid.N;
  const std::size_t N3 = static_cast<std::size_t>(N) * N * N;
  std::vector<Real> rho(N3, Real(0));
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      const Real d = D[i * n + j];
      if (d == Real(0)) continue;
      const Real *ci = ao[i].data();
      const Real *cj = ao[j].data();
      for (std::size_t g = 0; g < N3; ++g) rho[g] += d * ci[g] * cj[g];
    }
  auto V = fe_dage3d(grid, tgrid, rho, nv);
  std::vector<Real> J(static_cast<std::size_t>(n) * n, Real(0));
  std::vector<Real> prod(N3);
  for (int i = 0; i < n; ++i)
    for (int j = i; j < n; ++j) {
      const Real *ci = ao[i].data();
      const Real *cj = ao[j].data();
      for (std::size_t g = 0; g < N3; ++g) prod[g] = ci[g] * cj[g];
      const Real v = fe_inner(grid, prod, V);
      J[i * n + j] = J[j * n + i] = v;
    }
  return J;
}

/// Grid-RI exchange from precomputed AO values.
template <class Real>
std::vector<Real> grid_exchange_from_ao(const std::vector<std::vector<Real>> &ao, int n,
                                        const Real *Cocc, int nocc, const FEGrid1D<Real> &grid,
                                        const TGrid<Real> &tgrid, int nv) {
  const int N = grid.N;
  const std::size_t N3 = static_cast<std::size_t>(N) * N * N;
  std::vector<Real> K(static_cast<std::size_t>(n) * n, Real(0));
  std::vector<Real> phi(N3);
  for (int c = 0; c < nocc; ++c) {
    for (std::size_t g = 0; g < N3; ++g) phi[g] = Real(0);
    for (int u = 0; u < n; ++u) {
      const Real m = Cocc[u * nocc + c];
      if (m == Real(0)) continue;
      const Real *cu = ao[u].data();
      for (std::size_t g = 0; g < N3; ++g) phi[g] += m * cu[g];
    }
    std::vector<std::vector<Real>> gco(n, std::vector<Real>(N3)), Vco(n);
    for (int v = 0; v < n; ++v) {
      const Real *cv = ao[v].data();
      for (std::size_t g = 0; g < N3; ++g) gco[v][g] = cv[g] * phi[g];
      Vco[v] = fe_dage3d(grid, tgrid, gco[v], nv);
    }
    for (int u = 0; u < n; ++u)
      for (int v = u; v < n; ++v) {
        const Real kv = fe_inner(grid, gco[u], Vco[v]);
        K[u * n + v] += kv;
        if (u != v) K[v * n + u] += kv;
      }
  }
  return K;
}
} // namespace detail

/// Grid-RI Coulomb matrix J over a primitive Cartesian basis (density-on-grid +
/// one DAGE). `tgrid` selects the kernel (coulomb for 1/r).
template <class Real>
std::vector<Real> grid_coulomb_build(const ShellBasis<Real> &basis, const Real *D,
                                     const FEGrid1D<Real> &grid, const TGrid<Real> &tgrid,
                                     int nv = 24) {
  return detail::grid_coulomb_from_ao(ao_values_on_grid(basis, grid), basis.nao, D, grid, tgrid, nv);
}

/// Grid-RI exchange matrix K over a primitive Cartesian basis, via co-densities
/// g_ui = chi_u phi_i with phi_i = sum_u Cocc[u*nocc+i] chi_u (density
/// D = Cocc Cocc^T). nocc DAGEs per orbital.
template <class Real>
std::vector<Real> grid_exchange_build(const ShellBasis<Real> &basis, const Real *Cocc,
                                      int nocc, const FEGrid1D<Real> &grid,
                                      const TGrid<Real> &tgrid, int nv = 24) {
  return detail::grid_exchange_from_ao(ao_values_on_grid(basis, grid), basis.nao, Cocc, nocc,
                                       grid, tgrid, nv);
}

// ---- generally-contracted basis: contraction is FREE on the grid ------------
// The AO enters only through its pointwise value, so a contracted AO is one sum
// chi_a = sum_p ec(A,cA,p) x^lx y^ly z^lz e^{-alpha_p r^2} evaluated per point;
// the grid cost scales with nao (contracted), not nprim. The result matches the
// analytic contracted coulomb_build / exchange_build (contracted.hpp).

/// hp FE grid covering every AO-product envelope of a contracted basis: for
/// primitives p in shell A and q in shell B the product envelope has exponent
/// alpha_p + alpha_q; collect those (exponent, axis-projected centre).
template <class Real>
FEGrid1D<Real> grid_for_basis(const ContractedBasis<Real> &basis, Real eps = Real(1e-5),
                              int pmin = 4, int pmax = 16) {
  std::vector<FEGaussian1D<Real>> ax;
  const auto &sh = basis.shells;
  for (std::size_t A = 0; A < sh.size(); ++A)
    for (std::size_t B = 0; B < sh.size(); ++B)
      for (int p = 0; p < sh[A].nprim(); ++p)
        for (int q = 0; q < sh[B].nprim(); ++q) {
          const Real pe = sh[A].alpha[p] + sh[B].alpha[q];
          for (int d = 0; d < 3; ++d)
            ax.push_back({pe, (sh[A].alpha[p] * sh[A].center[d] +
                               sh[B].alpha[q] * sh[B].center[d]) / pe});
        }
  return make_fegrid1d_hp(ax, eps, pmin, pmax);
}

/// Evaluate every contracted Cartesian AO on the grid (AO order = ao_off[A] +
/// cA*ncart(l) + cart, matching contracted.hpp / the cint facade). The
/// primitive value is formed once per (shell, prim, cart) and accumulated into
/// each contracted function with its effective coefficient (basis-set coeff *
/// cart_norm_pyscf).
template <class Real>
std::vector<std::vector<Real>> ao_values_on_grid(const ContractedBasis<Real> &basis,
                                                 const FEGrid1D<Real> &grid) {
  const int N = grid.N;
  const std::size_t N3 = static_cast<std::size_t>(N) * N * N;
  std::vector<std::vector<Real>> ao(basis.nao, std::vector<Real>(N3, Real(0)));
  const int ns = static_cast<int>(basis.shells.size());
  for (int A = 0; A < ns; ++A) {
    const auto &sh = basis.shells[A];
    const int nc = ncart(sh.l), nct = sh.nctr(), npr = sh.nprim();
    std::vector<Real> gx(N), gy(N), gz(N), dx(N), dy(N), dz(N);
    for (int p = 0; p < npr; ++p) {
      const Real a = sh.alpha[p];
      for (int i = 0; i < N; ++i) {
        dx[i] = grid.xnode[i] - sh.center[0];
        dy[i] = grid.xnode[i] - sh.center[1];
        dz[i] = grid.xnode[i] - sh.center[2];
        gx[i] = std::exp(-a * dx[i] * dx[i]);
        gy[i] = std::exp(-a * dy[i] * dy[i]);
        gz[i] = std::exp(-a * dz[i] * dz[i]);
      }
      for (int k = 0; k < nc; ++k) {
        int lx, ly, lz;
        cart_comp(sh.l, k, lx, ly, lz);
        for (int cA = 0; cA < nct; ++cA) {
          const Real ec = detail::effective_coeff(sh, cA, p);
          if (ec == Real(0)) continue;
          Real *out = ao[basis.ao_off[A] + cA * nc + k].data();
          for (int ix = 0; ix < N; ++ix) {
            const Real fx = ec * detail::gr_ipow(dx[ix], lx) * gx[ix];
            for (int iy = 0; iy < N; ++iy) {
              const Real fxy = fx * detail::gr_ipow(dy[iy], ly) * gy[iy];
              const std::size_t base = (static_cast<std::size_t>(ix) * N + iy) * N;
              for (int iz = 0; iz < N; ++iz)
                out[base + iz] += fxy * detail::gr_ipow(dz[iz], lz) * gz[iz];
            }
          }
        }
      }
    }
  }
  return ao;
}

/// Grid-RI Coulomb J over a generally-contracted basis (matches
/// coulomb_build(ContractedBasis)). Contraction is absorbed into the pointwise
/// AO evaluation -- grid cost scales with nao, not nprim.
template <class Real>
std::vector<Real> grid_coulomb_build(const ContractedBasis<Real> &basis, const Real *D,
                                     const FEGrid1D<Real> &grid, const TGrid<Real> &tgrid,
                                     int nv = 24) {
  return detail::grid_coulomb_from_ao(ao_values_on_grid(basis, grid), basis.nao, D, grid, tgrid, nv);
}

/// Grid-RI exchange K over a generally-contracted basis (matches
/// exchange_build(ContractedBasis) on D = Cocc Cocc^T).
template <class Real>
std::vector<Real> grid_exchange_build(const ContractedBasis<Real> &basis, const Real *Cocc,
                                      int nocc, const FEGrid1D<Real> &grid,
                                      const TGrid<Real> &tgrid, int nv = 24) {
  return detail::grid_exchange_from_ao(ao_values_on_grid(basis, grid), basis.nao, Cocc, nocc,
                                       grid, tgrid, nv);
}

} // namespace intti
