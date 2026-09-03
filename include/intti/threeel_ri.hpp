// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// RI (density-fitting) folding of the mean-field three-electron Coulomb energy.
//
// The direct term is E3 = sum_{abcdef} G_{abcdef} D_ad D_be D_cf, an O(nao^6)
// sextet loop. It is exactly the classical functional of the density,
//   E3 = int rho_D(r) V_D(r)^2 dr,
// rho_D = sum_{mu nu} D_{mu nu} chi_mu chi_nu the electron density and V_D its
// Coulomb potential. The three-electron INTEGRALS are still Mehine-direct
// (threeel.hpp); only the CONTRACTION is refolded. Fitting rho_D to an auxiliary
// basis, rho_D ~ sum_P d_P chi_P with d = M^{-1} g, M_{PQ}=(P|Q), g_P=(P|rho_D),
// gives
//   E3 ~ sum_{PQR} d_P d_Q d_R T_{RPQ},  T_{RPQ} = int chi_R V_P V_Q,
// where T is the three-electron integral of three single auxiliary functions
// (each its own density; the ghost-partner trick of ncenter.hpp). Cost is
// O(naux^3) three-electron aux integrals + O(nao^2 naux) two-electron, versus
// O(nao^6); and it is EXACT when rho_D lies in span(aux). All quantities are the
// unnormalised primitive Cartesian convention shared by coulomb_2c/3c and the
// detail::three_electron_raw family, so D is the unnormalised primitive density.

#include <cstddef>
#include <vector>

#include "cholesky.hpp" // detail::syevd
#include "gto.hpp"      // ShellBasis, ncart, cart_comp
#include "ncenter.hpp"  // coulomb_2c, coulomb_3c
#include "threeel.hpp"  // CartGauss, detail::three_electron_raw

namespace intti {
namespace detail {

/// Expand a ShellBasis into one CartGauss per AO, in the AO order used by the
/// tensor builders (ao_off + cart_comp), so a density in that order lines up.
template <class Real>
std::vector<CartGauss<Real>> shellbasis_to_cartgauss(const ShellBasis<Real> &b) {
  std::vector<CartGauss<Real>> out(b.nao);
  for (std::size_t s = 0; s < b.shells.size(); ++s) {
    const auto &sh = b.shells[s];
    const int nc = ncart(sh.l);
    for (int k = 0; k < nc; ++k) {
      CartGauss<Real> g;
      g.alpha = sh.alpha;
      for (int d = 0; d < 3; ++d) g.center[d] = sh.center[d];
      cart_comp(sh.l, k, g.l[0], g.l[1], g.l[2]);
      out[b.ao_off[s] + k] = g;
    }
  }
  return out;
}

/// d = M^{-1} g via symmetric eigendecomposition, dropping eigenvalues below
/// tau * (largest eigenvalue) to tame auxiliary linear dependence.
template <class Real>
std::vector<Real> te_solve_metric(std::vector<Real> M, const std::vector<Real> &g, int n,
                                  Real tau) {
  std::vector<Real> w(n);
  syevd(n, M.data(), w.data()); // M -> eigenvectors (columns, col-major), w ascending
  const Real cutoff = tau * w[n - 1];
  std::vector<Real> d(n, Real(0));
  for (int k = 0; k < n; ++k) {
    if (w[k] <= cutoff) continue;
    Real proj = 0;
    for (int i = 0; i < n; ++i) proj += M[static_cast<std::size_t>(k) * n + i] * g[i];
    const Real c = proj / w[k];
    for (int i = 0; i < n; ++i) d[i] += c * M[static_cast<std::size_t>(k) * n + i];
  }
  return d;
}

} // namespace detail

/// RI-folded three-body Coulomb energy E3 = int rho_D V_D^2. `D` is the
/// unnormalised primitive density (nao x nao, row-major) in `orb`'s AO order;
/// `aux` is the fitting basis; `grid` a Coulomb t-grid. `tau` drops
/// near-dependent auxiliary directions. Exact when rho_D lies in span(aux).
template <class Real>
Real three_electron_energy_ri(const ShellBasis<Real> &orb, const std::vector<Real> &D,
                              const ShellBasis<Real> &aux, const TGrid<Real> &grid,
                              Real tau = 1e-10) {
  const int nao = orb.nao, naux = aux.nao;
  const auto M = coulomb_2c(aux, grid);           // (P|Q)
  const auto T3c = coulomb_3c(orb, aux, grid);    // (mu nu|P), row-major (mu,nu,P)
  // g_P = sum_{mu nu} D_{mu nu} (mu nu|P)
  std::vector<Real> g(naux, Real(0));
  for (int mu = 0; mu < nao; ++mu)
    for (int nu = 0; nu < nao; ++nu) {
      const Real Dmn = D[static_cast<std::size_t>(mu) * nao + nu];
      if (Dmn == Real(0)) continue;
      const Real *row = &T3c[(static_cast<std::size_t>(mu) * nao + nu) * naux];
      for (int P = 0; P < naux; ++P) g[P] += Dmn * row[P];
    }
  const auto d = detail::te_solve_metric(M, g, naux, tau);
  // E3 = sum_{PQR} d_P d_Q d_R T_{RPQ},  T_{RPQ} = three_electron_raw with each
  // auxiliary function as its own electron density (ghost partner, K=1).
  const auto auxg = detail::shellbasis_to_cartgauss(aux);
  CartGauss<Real> ghost{Real(0), {Real(0), Real(0), Real(0)}, {0, 0, 0}};
  Real E = 0;
  for (int R = 0; R < naux; ++R) {
    if (d[R] == Real(0)) continue;
    for (int P = 0; P < naux; ++P) {
      if (d[P] == Real(0)) continue;
      const Real dRP = d[R] * d[P];
      for (int Q = 0; Q < naux; ++Q) {
        if (d[Q] == Real(0)) continue;
        E += dRP * d[Q] *
             detail::three_electron_raw(auxg[R], auxg[P], auxg[Q], ghost, ghost, ghost, grid);
      }
    }
  }
  return E;
}

/// Effective one-body (Fock) contribution of the RI-folded three-body energy:
/// F_{mu nu} = dE3^RI/dD_{mu nu}, an nao x nao matrix (row-major, unnormalised
/// primitive convention). Because E3^RI = sum_{ijk} d_i d_j d_k T_{ijk} is cubic
/// in the fitted density d, and d = M^+ g is linear in D (g_P = sum D_{mu nu}
/// (mu nu|P)), the chain rule gives an auxiliary-space gradient G (scattered from
/// each T like the direct Fock), then F_{mu nu} = sum_Q (mu nu|Q) [M^+ G]_Q.
/// O(naux^3) + O(nao^2 naux), matching the energy; sum_{mu nu} F D = 3 E3^RI.
template <class Real>
std::vector<Real> three_electron_fock_ri(const ShellBasis<Real> &orb, const std::vector<Real> &D,
                                         const ShellBasis<Real> &aux, const TGrid<Real> &grid,
                                         Real tau = 1e-10) {
  const int nao = orb.nao, naux = aux.nao;
  const auto M = coulomb_2c(aux, grid);
  const auto T3c = coulomb_3c(orb, aux, grid); // (mu nu|P)
  std::vector<Real> g(naux, Real(0));
  for (int mu = 0; mu < nao; ++mu)
    for (int nu = 0; nu < nao; ++nu) {
      const Real Dmn = D[static_cast<std::size_t>(mu) * nao + nu];
      if (Dmn == Real(0)) continue;
      const Real *row = &T3c[(static_cast<std::size_t>(mu) * nao + nu) * naux];
      for (int P = 0; P < naux; ++P) g[P] += Dmn * row[P];
    }
  const auto d = detail::te_solve_metric(M, g, naux, tau);
  // auxiliary-space gradient: each aux 3-electron integral T_{ijk} scatters into
  // its three slots (the derivative of d_i d_j d_k), independent of any symmetry.
  const auto auxg = detail::shellbasis_to_cartgauss(aux);
  CartGauss<Real> ghost{Real(0), {Real(0), Real(0), Real(0)}, {0, 0, 0}};
  std::vector<Real> Gaux(naux, Real(0));
  for (int i = 0; i < naux; ++i)
    for (int j = 0; j < naux; ++j)
      for (int k = 0; k < naux; ++k) {
        // contributes only if at least two of d_i,d_j,d_k are nonzero
        const int nz = (d[i] != Real(0)) + (d[j] != Real(0)) + (d[k] != Real(0));
        if (nz < 2) continue;
        const Real T =
            detail::three_electron_raw(auxg[i], auxg[j], auxg[k], ghost, ghost, ghost, grid);
        Gaux[i] += d[j] * d[k] * T;
        Gaux[j] += d[i] * d[k] * T;
        Gaux[k] += d[i] * d[j] * T;
      }
  const auto z = detail::te_solve_metric(M, Gaux, naux, tau);
  // F_{mu nu} = sum_Q (mu nu|Q) z_Q
  std::vector<Real> F(static_cast<std::size_t>(nao) * nao, Real(0));
  for (int mu = 0; mu < nao; ++mu)
    for (int nu = 0; nu < nao; ++nu) {
      const Real *row = &T3c[(static_cast<std::size_t>(mu) * nao + nu) * naux];
      Real s = 0;
      for (int Q = 0; Q < naux; ++Q) s += row[Q] * z[Q];
      F[static_cast<std::size_t>(mu) * nao + nu] = s;
    }
  return F;
}

/// Effective two-body Coulomb reduction of the three-body term, via RI.
/// Contracting electron 3 of G with a density Dc gives an effective 2-electron
/// operator Omega_{mu nu, la si} = sum_cf G_{mu la c, nu si f} Dc_cf; folding it
/// into a Coulomb (J) build with a density Db (contract electron 2) collapses to
/// the one-body matrix
///   J_{mu nu} = <mu nu | V_Dc V_Db> = int rho_{mu nu}(r) V_Dc(r) V_Db(r) dr,
/// V_D the Coulomb potential of density D. With Dc = Db = D this is the
/// electron-1-slot piece <mu|V_D^2|nu> of the three-body Fock; a separate build
/// density Db is the transcorrelated / response use. Fitting both densities to
/// `aux`, J_{mu nu} = sum_{PQ} dc_P db_Q T_{mu nu, P Q}, where
/// T_{mu nu, P Q} = int rho_{mu nu} V_P V_Q is the three-electron integral of the
/// orbital pair (mu,nu) with two single auxiliaries. O(nao^2 naux^2) vs O(nao^6),
/// exact when both densities lie in span(aux). (The exchange, K, part needs the
/// full nao^4 x naux W-tensor and is a heavier follow-up.) Unnormalised
/// primitive convention; returns the nao x nao matrix, row-major.
template <class Real>
std::vector<Real> three_electron_effective_coulomb_ri(const ShellBasis<Real> &orb,
                                                      const std::vector<Real> &Dc,
                                                      const std::vector<Real> &Db,
                                                      const ShellBasis<Real> &aux,
                                                      const TGrid<Real> &grid, Real tau = 1e-10) {
  const int nao = orb.nao, naux = aux.nao;
  const auto M = coulomb_2c(aux, grid);
  const auto T3c = coulomb_3c(orb, aux, grid); // (mu nu|P)
  auto fit = [&](const std::vector<Real> &D) {
    std::vector<Real> g(naux, Real(0));
    for (int mu = 0; mu < nao; ++mu)
      for (int nu = 0; nu < nao; ++nu) {
        const Real Dmn = D[static_cast<std::size_t>(mu) * nao + nu];
        if (Dmn == Real(0)) continue;
        const Real *row = &T3c[(static_cast<std::size_t>(mu) * nao + nu) * naux];
        for (int P = 0; P < naux; ++P) g[P] += Dmn * row[P];
      }
    return detail::te_solve_metric(M, g, naux, tau);
  };
  const auto dc = fit(Dc), db = fit(Db);
  const auto orbg = detail::shellbasis_to_cartgauss(orb);
  const auto auxg = detail::shellbasis_to_cartgauss(aux);
  CartGauss<Real> ghost{Real(0), {Real(0), Real(0), Real(0)}, {0, 0, 0}};
  std::vector<Real> J(static_cast<std::size_t>(nao) * nao, Real(0));
  for (int mu = 0; mu < nao; ++mu)
    for (int nu = mu; nu < nao; ++nu) { // J symmetric in mu,nu
      Real jval = 0;
      for (int P = 0; P < naux; ++P) {
        if (dc[P] == Real(0)) continue;
        for (int Q = 0; Q < naux; ++Q) {
          const Real w = dc[P] * db[Q];
          if (w == Real(0)) continue;
          jval += w * detail::three_electron_raw(orbg[mu], auxg[P], auxg[Q], orbg[nu], ghost,
                                                 ghost, grid);
        }
      }
      J[static_cast<std::size_t>(mu) * nao + nu] = jval;
      J[static_cast<std::size_t>(nu) * nao + mu] = jval;
    }
  return J;
}

/// Effective two-body EXCHANGE (K) channel of the three-body term, via RI.
/// The same effective operator Omega (electron 3 contracted with Dc) folded into
/// an exchange build with Db does NOT collapse to one body, because Db couples
/// the two kets across electrons:
///   K_{mu la} = sum_{nu si} Omega_{mu nu, la si} Db_{nu si}
///             = int V_Dc(1) chi_mu(1) chi_la(2) Gamma_Db(1,2) / r12,
/// Gamma_Db(1,2) = sum_{nu si} chi_nu(1) Db_{nu si} chi_si(2) the density-matrix
/// kernel. Fitting only V_Dc ~ sum_P dc_P V_P,
///   K_{mu la} = sum_P dc_P sum_{nu si} Db_{nu si} W^P_{mu nu, la si},
///   W^P_{mu nu, la si} = int rho_{mu nu}(1) rho_{la si}(2) chi_P(3) / r12 / r13
///                      = three_electron_raw(mu, la, P, nu, si, ghost),
/// so cost is O(nao^4 naux) (no nao^6). Exact when Dc lies in span(aux); Db is
/// used directly. Note K is NOT symmetric in (mu,la): only electron 1 carries
/// the V_Dc dressing. Unnormalised primitive convention; nao x nao, row-major.
template <class Real>
std::vector<Real> three_electron_effective_exchange_ri(const ShellBasis<Real> &orb,
                                                       const std::vector<Real> &Dc,
                                                       const std::vector<Real> &Db,
                                                       const ShellBasis<Real> &aux,
                                                       const TGrid<Real> &grid, Real tau = 1e-10) {
  const int nao = orb.nao, naux = aux.nao;
  const auto M = coulomb_2c(aux, grid);
  const auto T3c = coulomb_3c(orb, aux, grid);
  std::vector<Real> g(naux, Real(0));
  for (int mu = 0; mu < nao; ++mu)
    for (int nu = 0; nu < nao; ++nu) {
      const Real Dmn = Dc[static_cast<std::size_t>(mu) * nao + nu];
      if (Dmn == Real(0)) continue;
      const Real *row = &T3c[(static_cast<std::size_t>(mu) * nao + nu) * naux];
      for (int P = 0; P < naux; ++P) g[P] += Dmn * row[P];
    }
  const auto dc = detail::te_solve_metric(M, g, naux, tau);
  const auto orbg = detail::shellbasis_to_cartgauss(orb);
  const auto auxg = detail::shellbasis_to_cartgauss(aux);
  CartGauss<Real> ghost{Real(0), {Real(0), Real(0), Real(0)}, {0, 0, 0}};
  std::vector<Real> K(static_cast<std::size_t>(nao) * nao, Real(0));
  for (int P = 0; P < naux; ++P) {
    if (dc[P] == Real(0)) continue;
    const Real dcP = dc[P];
    for (int mu = 0; mu < nao; ++mu)
      for (int la = 0; la < nao; ++la) {
        Real acc = 0;
        for (int nu = 0; nu < nao; ++nu)
          for (int si = 0; si < nao; ++si) {
            const Real Dbns = Db[static_cast<std::size_t>(nu) * nao + si];
            if (Dbns == Real(0)) continue;
            acc += Dbns * detail::three_electron_raw(orbg[mu], orbg[la], auxg[P], orbg[nu],
                                                     orbg[si], ghost, grid);
          }
        K[static_cast<std::size_t>(mu) * nao + la] += dcP * acc;
      }
  }
  return K;
}

} // namespace intti
