// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// General multi-density Coulomb/exchange builder.
//
// The response machinery of every property -- CPHF for nuclear displacements,
// NMR shieldings, magnetisabilities, polarisabilities -- needs J and K for a SET
// of trial densities that are not, in general, symmetric. This is the interface
// the established codes converged on: a vector of densities in, a vector of
// J/K matrices out, each density tagged with its symmetry and with which terms
// it wants. Cf. Dalton's SIRFCK(FMAT, DMAT, NDMAT, ISYMDM, IFCTYP) -- where
// IFCTYP encodes exactly (no symmetry / symmetric / antisymmetric) x (Coulomb /
// exchange / both) -- and VeloxChem's FockDriver::compute(screener, densities,
// labels, ...) with mat_t {symmetric, antisymmetric, general}.
//
// WHY THE SYMMETRY TAG MATTERS, precisely:
//
//   J_mn(D) = sum_ls (mn|ls) D_ls.  The ERI is symmetric under l <-> s, so the
//   antisymmetric part of D cancels pairwise and J sees ONLY the symmetric part.
//   J is therefore always symmetric, J(antisymmetric D) is exactly zero, and
//   coulomb_build's D + D^T fold is exact for any density.
//
//   K_mn(D) = sum_ls (ml|ns) D_ls.  Here K(D)^T = K(D^T): K is symmetric only
//   when D is, and for antisymmetric D the correct K is ANTISYMMETRIC. The fused
//   exchange_build computes the upper triangle and mirrors it, which cannot
//   represent that -- so a general density needs the path below.
//
// The builder is otherwise operator-agnostic along the axes that are already
// free: the kernel enters through the t-grid (Coulomb, erf/erfc range
// separation, Yukawa) and the scalar type through the template (real, or
// Kokkos::complex for a finite magnetic field). What is NOT yet folded in is
// the perturbation axis -- d/dR, d/dB, spin-orbit -- which is a linear
// combination of angular-momentum-shifted quartets with operator-specific
// coefficients; those remain their own builders for now.

#include <cstddef>
#include <vector>

#include "batch.hpp"
#include "device.hpp"
#include "fock.hpp"
#include "gto.hpp"
#include "tgrid.hpp"

namespace intti {

/// Symmetry of a density matrix about its diagonal. This is NOT the same thing
/// as Symmetry (symmetry.hpp), which selects the permutational policy for the
/// integrals themselves.
enum class DensitySymmetry { General, Symmetric, Antisymmetric };

/// Which two-electron terms a density wants built.
enum class FockTerms { Coulomb, Exchange, CoulombExchange };

/// One density and what to do with it. D is nao x nao, row-major.
template <class Real> struct JKRequest {
  const Real *D{nullptr};
  DensitySymmetry sym{DensitySymmetry::Symmetric};
  FockTerms terms{FockTerms::CoulombExchange};
};

/// J and K per request, in request order. An entry is empty when that term was
/// not asked for.
template <class Real> struct JKResult {
  std::vector<std::vector<Real>> J, K;
};

namespace detail {

/// General path: every ordered shell quartet, no assumption about the density.
/// One pass over the integrals digests all requests, which is the point of the
/// vector interface -- the ERIs are the expensive part and they are shared.
/// Only the bra-ket fold (ab|cd) = (cd|ab) is used; the intra-pair folds are
/// available for real integrals but interact with the general-D exchange
/// accumulation, so they are left to the fused symmetric path.
template <class Real>
JKResult<Real> jk_build_general(const ShellBasis<Real> &basis,
                                const std::vector<JKRequest<Real>> &reqs,
                                const TGrid<Real> &grid, Real tau) {
  const int ns = static_cast<int>(basis.shells.size());
  const int nao = basis.nao;
  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
  const int nreq = static_cast<int>(reqs.size());

  std::vector<ShellPair<Real>> plist;
  std::vector<int> hoa, hob;
  plist.reserve(static_cast<std::size_t>(ns) * ns);
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      plist.push_back(make_pair(basis.shells[a], basis.shells[b]));
      hoa.push_back(basis.ao_off[a]);
      hob.push_back(basis.ao_off[b]);
    }
  const int npair = static_cast<int>(plist.size());
  auto tab = make_pair_table(plist);
  std::vector<Real> Q;
  if (tau > Real(0)) Q = schwarz(tab, plist, grid);
  std::vector<std::pair<int, int>> quartets;
  for (int ib = 0; ib < npair; ++ib)
    for (int ik = ib; ik < npair; ++ik) {
      if (tau > Real(0) && Q[ib] * Q[ik] < tau) continue;
      quartets.push_back({ib, ik});
    }
  auto batch = make_batch(tab, quartets);
  QuartetWorkspace<Real> ws;
  Kokkos::View<Real *> out("intti::jk::out", batch.nout_total);
  eri_quartets(tab, batch, grid, out, ws);

  // densities and per-request flags, flattened onto the device
  std::vector<Real> hD(static_cast<std::size_t>(nreq) * n2);
  std::vector<int> hwantJ(nreq), hwantK(nreq);
  for (int r = 0; r < nreq; ++r) {
    const bool wj = reqs[r].terms != FockTerms::Exchange;
    const bool wk = reqs[r].terms != FockTerms::Coulomb;
    hwantJ[r] = wj ? 1 : 0;
    hwantK[r] = wk ? 1 : 0;
    for (std::size_t i = 0; i < n2; ++i) hD[r * n2 + i] = reqs[r].D[i];
  }
  auto Dd = to_device(hD, "intti::jk::D");
  auto wJ = to_device(hwantJ, "intti::jk::wj"), wK = to_device(hwantK, "intti::jk::wk");
  auto oav = to_device(hoa, "intti::jk::oa"), obv = to_device(hob, "intti::jk::ob");
  auto lav = tab.la, lbv = tab.lb;
  auto qv = batch.quartets, offv = batch.out_offset;
  Kokkos::View<Real *> Jd("intti::jk::J", static_cast<std::size_t>(nreq) * n2);
  Kokkos::View<Real *> Kd("intti::jk::K", static_cast<std::size_t>(nreq) * n2);
  const std::size_t plane = n2;
  Kokkos::parallel_for(
      "intti::jk::digest", Kokkos::RangePolicy<>(0, batch.nq), KOKKOS_LAMBDA(int iq) {
        const int ib = qv(iq, 0), ik = qv(iq, 1);
        const int na = ncart(lav(ib)), nb = ncart(lbv(ib));
        const int nc = ncart(lav(ik)), nd = ncart(lbv(ik));
        const int oa = oav(ib), ob = obv(ib), oc = oav(ik), od = obv(ik);
        const std::int64_t base = offv(iq);
        for (int ka = 0; ka < na; ++ka)
          for (int kb = 0; kb < nb; ++kb)
            for (int kc = 0; kc < nc; ++kc)
              for (int kd = 0; kd < nd; ++kd) {
                const Real v =
                    out(base + (((static_cast<std::size_t>(ka) * nb + kb) * nc + kc) * nd + kd));
                const std::size_t iab = static_cast<std::size_t>(oa + ka) * nao + ob + kb;
                const std::size_t icd = static_cast<std::size_t>(oc + kc) * nao + od + kd;
                const std::size_t iac = static_cast<std::size_t>(oa + ka) * nao + oc + kc;
                const std::size_t ibd = static_cast<std::size_t>(ob + kb) * nao + od + kd;
                const std::size_t ica = static_cast<std::size_t>(oc + kc) * nao + oa + ka;
                const std::size_t idb = static_cast<std::size_t>(od + kd) * nao + ob + kb;
                for (int r = 0; r < nreq; ++r) {
                  const std::size_t o = static_cast<std::size_t>(r) * plane;
                  // (ab|cd): J_ab += v D_cd ; K_ac += v D_bd
                  if (wJ(r)) Kokkos::atomic_add(&Jd(o + iab), v * Dd(o + icd));
                  if (wK(r)) Kokkos::atomic_add(&Kd(o + iac), v * Dd(o + ibd));
                  if (ib == ik) continue;
                  // the same block read as (cd|ab): bra-ket fold. NOTE the
                  // exchange convention here is K_mn = sum_ls (ml|ns) D_ls, so
                  // the swapped orientation gives K[c,a] += v D[d,b] -- giao2e
                  // uses (ml|sn) instead and its swap maps differently.
                  if (wJ(r)) Kokkos::atomic_add(&Jd(o + icd), v * Dd(o + iab));
                  if (wK(r)) Kokkos::atomic_add(&Kd(o + ica), v * Dd(o + idb));
                }
              }
      });
  Kokkos::fence();
  auto hJ = to_host(Jd), hK = to_host(Kd);
  JKResult<Real> res;
  res.J.resize(nreq);
  res.K.resize(nreq);
  for (int r = 0; r < nreq; ++r) {
    if (hwantJ[r]) res.J[r].assign(hJ.begin() + r * n2, hJ.begin() + (r + 1) * n2);
    if (hwantK[r]) res.K[r].assign(hK.begin() + r * n2, hK.begin() + (r + 1) * n2);
  }
  return res;
}

} // namespace detail

/// Coulomb and exchange matrices for a set of densities, one pass over the
/// integrals. Densities tagged Symmetric are served by the tuned fused engines
/// (coulomb_build / exchange_build); General and Antisymmetric densities go
/// through the general path, which makes no assumption about the density and is
/// the only correct route for the antisymmetric perturbed densities that
/// magnetic response produces.
template <class Real>
JKResult<Real> jk_build(const ShellBasis<Real> &basis,
                        const std::vector<JKRequest<Real>> &reqs, const TGrid<Real> &grid,
                        Real tau = Real(0)) {
  const int nreq = static_cast<int>(reqs.size());
  const std::size_t n2 = static_cast<std::size_t>(basis.nao) * basis.nao;
  bool all_symmetric = true;
  for (const auto &r : reqs)
    if (r.sym != DensitySymmetry::Symmetric) all_symmetric = false;
  if (!all_symmetric) return detail::jk_build_general(basis, reqs, grid, tau);

  JKResult<Real> res;
  res.J.resize(nreq);
  res.K.resize(nreq);
  for (int r = 0; r < nreq; ++r) {
    if (reqs[r].terms != FockTerms::Exchange) {
      res.J[r].assign(n2, Real(0));
      coulomb_build(basis, reqs[r].D, grid, res.J[r].data(), tau);
    }
    if (reqs[r].terms != FockTerms::Coulomb) {
      res.K[r].assign(n2, Real(0));
      exchange_build(basis, reqs[r].D, grid, res.K[r].data(), tau);
    }
  }
  return res;
}

} // namespace intti
