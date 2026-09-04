// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Yukawa / bound-state Helmholtz primitives -- M-HK. The bound-state Helmholtz
// Green's function G_kappa(r) = e^{-kappa r}/(4 pi r) (kappa = sqrt(-2 eps)) is a
// Yukawa (screened Coulomb) kernel, and the orbital-update integral-equation SCF
// (Park 2017, JCTC 13, 654) applies it in place of diagonalising the Fock
// matrix. The key point for libintti: the Yukawa kernel's t-representation
//   e^{-kappa r}/r = (2/sqrt(pi)) int_0^inf e^{-t^2 r^2} e^{-kappa^2/4t^2} dt
// differs from Coulomb ONLY by the node weight factor e^{-kappa^2/4t^2}, folded
// into the ExpSum grid. So EVERY existing builder computes the Yukawa analogue
// unchanged when fed a Yukawa grid:
//   - attraction_accumulate / nuclear_matrix -> Yukawa attraction (this file)
//   - eri_quartet / coulomb_build / exchange_build -> Yukawa ERI / J / K
// and, crucially, a single t-resolved integral tensor is common to every
// orbital energy kappa_i (only the scalar weight e^{-kappa_i^2/4t^2} changes),
// so the per-orbital Helmholtz loop is a cheap re-weighting, not N_occ kernel
// applies (see the M-HK roadmap entry).

#include <algorithm>
#include <vector>

#include "fock.hpp"
#include "kernel.hpp"
#include "nuclear.hpp"
#include "quartet.hpp"
#include "tgrid.hpp"

namespace intti {

/// ExpSum t-grid for the Yukawa / bound-state Helmholtz kernel e^{-kappa r}/r.
/// The sinc rule is exponentially convergent here because e^{-kappa^2/4t^2}
/// makes the integrand decay at BOTH ends; finer h tightens it. The exponent
/// range should span the pair exponents the grid will act on.
template <class Real>
TGrid<Real> yukawa_grid(Real kappa, Real alpha_min = Real(1e-2),
                        Real alpha_max = Real(1e3), Real h = Real(0.2)) {
  return make_tgrid(yukawa(kappa), exp_sum_spec_for_range(alpha_min, alpha_max, h));
}

/// Yukawa attraction matrix V_{mu nu} = sum_c w_c <mu| e^{-kappa|r-R_c|}/|r-R_c|
/// |nu> -- the bound-state Helmholtz Green's function applied to point sources
/// (the nuclear-potential piece of the Helmholtz orbital update). Same machinery
/// as nuclear_matrix; only the grid weights carry kappa. The grid exponent range
/// is taken from the basis pair exponents (2*[min,max] shell alpha), padded.
template <class Real>
std::vector<Real> yukawa_attraction_matrix(
    const ShellBasis<Real> &basis, const std::vector<PointCharge<Real>> &charges,
    Real kappa, Real tau = Real(0), Real h = Real(0.2)) {
  Real amax = 0;
  for (const auto &s : basis.shells) amax = std::max(amax, s.alpha);
  // ExpSum es_tmax = 30 sqrt(alpha_max): the attraction with a nearby charge
  // (small r) needs large t, so the range must be generous at the high end (a
  // narrow range truncates the ~t^{-2} tail regardless of node density). Floor
  // the span at [1e-2, 1e3] and let the basis widen it for tight cores.
  const Real alo = std::min(Real(1e-2), Real(0.2) * amax);
  const Real ahi = std::max(Real(1e3), Real(20) * amax);
  auto grid = yukawa_grid(kappa, alo, ahi, h);
  std::vector<Real> V(static_cast<std::size_t>(basis.nao) * basis.nao, Real(0));
  detail::attraction_accumulate(basis, charges, grid, tau, V.data());
  return V;
}

/// Two-kernel matrix M(kappa)_{mu nu} = <chi_mu | G_kappa V | chi_nu>, the
/// operator the Helmholtz orbital update applies (M-HK step 2): the bound-state
/// Green's function G_kappa convolved with the point-charge potential
/// V = sum_C w_C / |r - R_C|. Two kernels: Yukawa between r,r' (G_kappa =
/// (1/4pi) e^{-kappa|r-r'|}/|r-r'|, a Yukawa grid) and Coulomb between r' and
/// each charge (unfolded on a Coulomb t-grid as a Gaussian at R_C of exponent
/// t^2). So
///   M_{mu nu} = sum_C w_C (1/4pi) sum_t w_t
///               <chi_mu . ghost_A | e^{-kappa r12}/r12 | chi_nu . G_{R_C}^t>,
/// each factor an ordinary geminal-style quartet (ghost = zero-exponent unit-s
/// on the bra so electron 1 carries only chi_mu). Validated by the fixed-point
/// consistency M(kappa0) c0 = -1/2 S c0 at the H_core ground state
/// (references/hk_poc.cpp). Only the Yukawa node weight carries kappa, so a
/// t-resolved tensor amortises the per-orbital loop; this direct form builds one
/// kappa. charges carry w_C (use -Z for attraction); host, real pairs.
template <class Real>
std::vector<Real> helmholtz_nuclear_matrix(
    const ShellBasis<Real> &basis, const std::vector<PointCharge<Real>> &charges,
    Real kappa, Real amin = Real(1e-2), Real amax = Real(1e3),
    Real h = Real(0.15)) {
  const int nao = basis.nao;
  std::vector<Real> M(static_cast<std::size_t>(nao) * nao, Real(0));
  const int ns = static_cast<int>(basis.shells.size());
  const Real pi = pi_v<Real>();
  auto ygrid = yukawa_grid(kappa, amin, amax, h); // e^{-kappa r}/r = 4 pi G_kappa
  auto cgrid = make_tgrid(coulomb<Real>());       // 1/r_C unfold
  for (int mu = 0; mu < ns; ++mu)
    for (int nu = 0; nu < ns; ++nu) {
      const auto &sm = basis.shells[mu], &sn = basis.shells[nu];
      const int ncm = ncart(sm.l), ncn = ncart(sn.l);
      PrimitiveShell<Real> ghost{Real(0), {sm.center[0], sm.center[1], sm.center[2]}, 0};
      std::vector<Real> acc(static_cast<std::size_t>(ncm) * ncn, Real(0));
      std::vector<Real> blk(static_cast<std::size_t>(ncm) * ncn);
      for (const auto &C : charges) {
        const auto bra = make_pair(sm, ghost);
        for (int j = 0; j < cgrid.n(); ++j) {
          PrimitiveShell<Real> gC{cgrid.t[j] * cgrid.t[j], {C.R[0], C.R[1], C.R[2]}, 0};
          eri_quartet(bra, make_pair(sn, gC), ygrid, blk.data());
          const Real w = C.weight * cgrid.w[j] / (4 * pi);
          for (int i = 0; i < ncm * ncn; ++i) acc[i] += w * blk[i];
        }
      }
      for (int ka = 0; ka < ncm; ++ka)
        for (int kb = 0; kb < ncn; ++kb)
          M[static_cast<std::size_t>(basis.ao_off[mu] + ka) * nao + basis.ao_off[nu] + kb] =
              acc[ka * ncn + kb];
    }
  return M;
}

} // namespace intti
