// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// One-electron geometric Hessians and the nuclear-repulsion Hessian -- the
// pieces of the molecular Hessian that need only centre (bra/ket) derivatives,
// assembled at the matrix level (weight matrix in, Hessian out).
//
// For a 1e matrix M whose element M_mn depends on the bra centre A (shell a)
// and ket centre B (shell b), the second geometric derivative routes the
// order-2 shift blocks of geoderiv.hpp: d^2_{A_e A_f} into H[a,e][a,f] and the
// mixed d_{A_e} d_{B_f} into H[a,e][b,f]. Summed over all ordered shell pairs
// (with a symmetric weight/density) these two families give the complete,
// symmetric (3 nshell) x (3 nshell) Hessian. Overlap (weighted by the
// energy-weighted density) and kinetic (weighted by the density) use this;
// nuclear attraction additionally needs operator-centre derivatives and is
// handled elsewhere.

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "fock.hpp"
#include "geoderiv.hpp"
#include "gto.hpp"
#include "nuclear.hpp"

namespace intti {

namespace detail {
/// Assemble the (3 ns) x (3 ns) Hessian of sum_mn W_mn M_mn from a geometric-
/// derivative callable gd(na, nb) -> whole nao x nao matrix d^na_A d^nb_B M.
template <class Real, class GD>
std::vector<Real> oneel_hessian_assemble(const ShellBasis<Real> &basis, const Real *W,
                                         GD &&gd) {
  const int ns = static_cast<int>(basis.shells.size());
  const int nao = basis.nao, dim = 3 * ns;
  std::vector<Real> H(static_cast<std::size_t>(dim) * dim, Real(0));
  std::vector<int> ao2sh(nao);
  for (int s = 0; s < ns; ++s)
    for (int k = basis.ao_off[s]; k < basis.ao_off[s + 1]; ++k) ao2sh[k] = s;
  auto Wm = [&](int i, int j) { return W[static_cast<std::size_t>(i) * nao + j]; };
  auto Hadd = [&](int p, int e, int q, int f, Real v) {
    H[(3 * p + e) * static_cast<std::size_t>(dim) + 3 * q + f] += v;
  };
  const std::array<int, 3> zero = {0, 0, 0};
  auto at = [&](const std::vector<Real> &M, int i, int j) {
    return M[static_cast<std::size_t>(i) * nao + j];
  };
  // Each ordered AO pair (mu in shell a, nu in shell b) contributes its second
  // derivative to four families, depending on which centre each of the two
  // derivatives hits (bra a / ket b):
  //   bra-bra -> H[a,e][a,f], bra-ket -> H[a,e][b,f],
  //   ket-bra -> H[b,e][a,f], ket-ket -> H[b,e][b,f].
  // g^{kk}_ef[mu,nu] = g^{bb}_ef[nu,mu] and g^{kb}_ef[mu,nu] = g^{bk}_fe[nu,mu],
  // so only the bra-differentiated matrices Mbb (na=e+f) and Mbk (na=e,nb=f)
  // are needed, read at [mu,nu] and [nu,mu].
  for (int e = 0; e < 3; ++e)
    for (int f = 0; f < 3; ++f) {
      std::array<int, 3> nbb = {0, 0, 0}, na = {0, 0, 0}, nb = {0, 0, 0};
      nbb[e] += 1;
      nbb[f] += 1;      // bb: d^2 on bra
      na[e] += 1;
      nb[f] += 1;       // bk: d_e on bra, d_f on ket
      auto Mbb = gd(nbb, zero);
      auto Mbk = gd(na, nb);
      for (int mu = 0; mu < nao; ++mu) {
        const int a = ao2sh[mu];
        for (int nu = 0; nu < nao; ++nu) {
          const int b = ao2sh[nu];
          const Real w = Wm(mu, nu);
          Hadd(a, e, a, f, w * at(Mbb, mu, nu)); // bra-bra
          Hadd(b, e, b, f, w * at(Mbb, nu, mu)); // ket-ket
          Hadd(a, e, b, f, w * at(Mbk, mu, nu)); // bra-ket
          Hadd(b, e, a, f, w * at(Mbk, nu, mu)); // ket-bra
        }
      }
    }
  return H;
}
} // namespace detail

/// Hessian of sum_mn W_mn S_mn (W typically the energy-weighted density): the
/// overlap/Pulay contribution to the molecular Hessian, (3 ns) x (3 ns).
template <class Real>
std::vector<Real> overlap_hessian(const ShellBasis<Real> &basis, const Real *W) {
  return detail::oneel_hessian_assemble(
      basis, W, [&](const std::array<int, 3> &na, const std::array<int, 3> &nb) {
        return overlap_geoderiv(basis, na, nb);
      });
}

/// Hessian of sum_mn D_mn T_mn: the kinetic contribution to the molecular
/// Hessian, (3 ns) x (3 ns).
template <class Real>
std::vector<Real> kinetic_hessian(const ShellBasis<Real> &basis, const Real *D) {
  return detail::oneel_hessian_assemble(
      basis, D, [&](const std::array<int, 3> &na, const std::array<int, 3> &nb) {
        return kinetic_geoderiv(basis, na, nb);
      });
}

/// Hessian of the nuclear-attraction energy sum_mn D_mn V_mn,
/// V_mn = sum_C w_C <m|1/|r-R_C||n>, w.r.t. the shell centres, as a
/// (3 ns) x (3 ns) matrix. Each single-charge term depends on the bra centre,
/// the ket centre, and the nucleus R_C; by translational invariance the
/// operator-centre derivative is d/dR_C = -(d_A + d_B), so the whole Hessian
/// is assembled from bra/ket geometric derivatives (nuclear_geoderiv) with a
/// per-shell coefficient of +1 on the bra/ket centre and -1 on the nucleus.
/// charge_shell[c] gives the shell index whose centre coincides with charge c
/// (the atom carrying the nucleus); differentiating that shell moves the
/// nucleus too.
template <class Real>
std::vector<Real>
nuclear_attraction_hessian(const ShellBasis<Real> &basis,
                           const std::vector<PointCharge<Real>> &charges,
                           const TGrid<Real> &grid, const std::vector<int> &charge_shell,
                           const Real *D) {
  const int ns = static_cast<int>(basis.shells.size());
  const int nao = basis.nao, dim = 3 * ns;
  std::vector<Real> H(static_cast<std::size_t>(dim) * dim, Real(0));
  std::vector<int> ao2sh(nao);
  for (int s = 0; s < ns; ++s)
    for (int k = basis.ao_off[s]; k < basis.ao_off[s + 1]; ++k) ao2sh[k] = s;
  auto Dm = [&](int i, int j) { return D[static_cast<std::size_t>(i) * nao + j]; };
  auto Hadd = [&](int p, int e, int q, int f, Real v) {
    H[(3 * p + e) * static_cast<std::size_t>(dim) + 3 * q + f] += v;
  };
  const std::array<int, 3> zero = {0, 0, 0};
  for (std::size_t c = 0; c < charges.size(); ++c) {
    const int nc = charge_shell[c];
    std::vector<PointCharge<Real>> one{charges[c]};
    for (int e = 0; e < 3; ++e)
      for (int f = 0; f < 3; ++f) {
        std::array<int, 3> nbb = {0, 0, 0}, na = {0, 0, 0}, nb = {0, 0, 0};
        nbb[e] += 1;
        nbb[f] += 1;
        na[e] += 1;
        nb[f] += 1;
        auto Mbb = nuclear_geoderiv(basis, one, grid, nbb, zero); // d^bra_e d^bra_f
        auto Mbk = nuclear_geoderiv(basis, one, grid, na, nb);    // d^bra_e d^ket_f
        auto at = [&](const std::vector<Real> &M, int i, int j) {
          return M[static_cast<std::size_t>(i) * nao + j];
        };
        for (int mu = 0; mu < nao; ++mu) {
          const int a = ao2sh[mu];
          for (int nu = 0; nu < nao; ++nu) {
            const int b = ao2sh[nu];
            const Real w = Dm(mu, nu);
            if (w == Real(0)) continue;
            // d^x_e d^y_f V at (mu,nu): x,y in {bra,ket}
            const Real Dbb = at(Mbb, mu, nu), Dkk = at(Mbb, nu, mu);
            const Real Dbk = at(Mbk, mu, nu), Dkb = at(Mbk, nu, mu);
            // per-shell coefficients of the bra/ket derivative in d/dR_s
            auto cbra = [&](int s) { return Real((s == a) - (s == nc)); };
            auto cket = [&](int s) { return Real((s == b) - (s == nc)); };
            // unique shells among {a, b, nc} (coincidences handled by cbra/cket)
            int U[3], nu_s = 0;
            for (int cand : {a, b, nc}) {
              bool seen = false;
              for (int t = 0; t < nu_s; ++t) seen |= (U[t] == cand);
              if (!seen) U[nu_s++] = cand;
            }
            for (int i1 = 0; i1 < nu_s; ++i1)
              for (int i2 = 0; i2 < nu_s; ++i2) {
                const int s1 = U[i1], s2 = U[i2];
                const Real coeff =
                    cbra(s1) * (cbra(s2) * Dbb + cket(s2) * Dbk) +
                    cket(s1) * (cbra(s2) * Dkb + cket(s2) * Dkk);
                if (coeff != Real(0)) Hadd(s1, e, s2, f, w * coeff);
              }
          }
        }
      }
  }
  return H;
}

/// Nuclear-repulsion Hessian d^2 E_nn / dR_{I,e} dR_{J,f} for
/// E_nn = sum_{I<J} Z_I Z_J / |R_I - R_J|, as a (3 N) x (3 N) matrix
/// (N = number of point charges; here weight = Z). Analytic.
template <class Real>
std::vector<Real>
nuclear_repulsion_hessian(const std::vector<PointCharge<Real>> &charges) {
  const int N = static_cast<int>(charges.size()), dim = 3 * N;
  std::vector<Real> H(static_cast<std::size_t>(dim) * dim, Real(0));
  auto Z = [&](int i) { return charges[i].weight; };
  for (int I = 0; I < N; ++I)
    for (int J = 0; J < N; ++J) {
      if (I == J) continue;
      Real r[3], R2 = 0;
      for (int d = 0; d < 3; ++d) {
        r[d] = charges[I].R[d] - charges[J].R[d];
        R2 += r[d] * r[d];
      }
      const Real R = std::sqrt(R2), R3 = R * R2, R5 = R3 * R2;
      const Real ZZ = Z(I) * Z(J);
      for (int e = 0; e < 3; ++e)
        for (int f = 0; f < 3; ++f) {
          const Real block = ZZ * ((e == f ? Real(1) / R3 : Real(0)) - 3 * r[e] * r[f] / R5);
          // off-diagonal I,J
          H[(3 * I + e) * static_cast<std::size_t>(dim) + 3 * J + f] += block;
          // diagonal I,I gets -sum over J (translational invariance)
          H[(3 * I + e) * static_cast<std::size_t>(dim) + 3 * I + f] -= block;
        }
    }
  return H;
}

} // namespace intti
