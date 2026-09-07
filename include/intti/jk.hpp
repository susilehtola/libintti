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
#include "erigrad.hpp" // detail::md_grad_terms, comp_index
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

/// Derivative Coulomb/exchange matrices: for each shell centre s and Cartesian
/// direction e,
///   dJ_mn/dR_se = sum_ls d(mn|ls)/dR_se D_ls,
///   dK_mn/dR_se = sum_ls d(ml|ns)/dR_se D_ls.
///
/// This is what a CPHF right-hand side is built from, for ANY perturbation that
/// moves the nuclei -- and it is the piece the library was missing: the existing
/// two_electron_gradient contracts these against a second density straight into
/// a force and throws the matrices away, which is all an energy gradient needs
/// and nothing that response needs.
///
/// Per shell centre, not per atom; the caller maps shells to atoms and sums, as
/// it already does for two_electron_gradient. Matrix (s, e) of request r starts
/// at (3*s + e) * nao * nao.
template <class Real> struct JKDerivResult {
  int nshell{0}, nao{0};
  std::vector<std::vector<Real>> J, K;
};

namespace detail {

/// The derivative of a quartet with respect to the FOURTH slot is obtained from
/// translational invariance, d/dA + d/dB + d/dC + d/dD = 0, rather than by
/// evaluating two more shifted quartets: the integral depends only on centre
/// differences. That is exact, and saves a quarter of the shifted evaluations.
template <class Real>
JKDerivResult<Real> jk_deriv_general(const ShellBasis<Real> &basis,
                                     const std::vector<JKRequest<Real>> &reqs,
                                     const TGrid<Real> &grid, Real tau) {
  const int ns = static_cast<int>(basis.shells.size());
  const int nao = basis.nao;
  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
  const int nreq = static_cast<int>(reqs.size());
  const int npert = 3 * ns;

  std::vector<int> hL(ns), hOff(ns);
  std::vector<Real> hAl(ns);
  for (int i = 0; i < ns; ++i) {
    hL[i] = basis.shells[i].l;
    hOff[i] = basis.ao_off[i];
    hAl[i] = basis.shells[i].alpha;
  }
  // memoised (shell, shift) x (shell, shift) pairs: base plus the four singly
  // shifted forms each ordered pair needs
  std::vector<ShellPair<Real>> plist;
  std::vector<int> pid(static_cast<std::size_t>(ns) * 3 * ns * 3, -1);
  auto pair_id = [&](int si, int di, int sj, int dj) {
    if (hL[si] + di < 0 || hL[sj] + dj < 0) return -1;
    const std::size_t key =
        ((static_cast<std::size_t>(si) * 3 + (di + 1)) * ns + sj) * 3 + (dj + 1);
    if (pid[key] < 0) {
      PrimitiveShell<Real> a = basis.shells[si], b = basis.shells[sj];
      a.l += di;
      b.l += dj;
      plist.push_back(make_pair(a, b));
      pid[key] = static_cast<int>(plist.size()) - 1;
    }
    return pid[key];
  };
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      pair_id(a, 0, b, 0);
      pair_id(a, 1, b, 0);
      pair_id(a, -1, b, 0);
      pair_id(a, 0, b, 1);
      pair_id(a, 0, b, -1);
    }
  auto tab = make_pair_table(plist);
  std::vector<Real> Q;
  if (tau > Real(0)) Q = schwarz(tab, plist, grid);
  auto scale = [&](int s) {
    const Real two_al = 2 * hAl[s];
    return two_al > Real(hL[s]) ? two_al : Real(hL[s]);
  };
  // jobs: one per ordered shell quartet, carrying the six shifted entries it
  // needs (slots a, b on the bra; c on the ket; slot d from invariance)
  std::vector<std::pair<int, int>> quartets;
  std::vector<int> ja, jb, jc, jd, eap, eam, ebp, ebm, ecp, ecm;
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b)
      for (int c = 0; c < ns; ++c)
        for (int d = 0; d < ns; ++d) {
          if (tau > Real(0)) {
            const int pb = pair_id(a, 0, b, 0), pk = pair_id(c, 0, d, 0);
            if (scale(a) * scale(b) * scale(c) * scale(d) * Q[pb] * Q[pk] < tau) continue;
          }
          const int kd0 = pair_id(c, 0, d, 0);
          auto emit = [&](int bra, int ket) {
            if (bra < 0 || ket < 0) return -1;
            const int e = static_cast<int>(quartets.size());
            quartets.push_back({bra, ket});
            return e;
          };
          ja.push_back(a); jb.push_back(b); jc.push_back(c); jd.push_back(d);
          eap.push_back(emit(pair_id(a, 1, b, 0), kd0));
          eam.push_back(emit(pair_id(a, -1, b, 0), kd0));
          ebp.push_back(emit(pair_id(a, 0, b, 1), kd0));
          ebm.push_back(emit(pair_id(a, 0, b, -1), kd0));
          ecp.push_back(emit(pair_id(a, 0, b, 0), pair_id(c, 1, d, 0)));
          ecm.push_back(emit(pair_id(a, 0, b, 0), pair_id(c, -1, d, 0)));
        }
  const int njob = static_cast<int>(ja.size());
  auto batch = make_batch(tab, quartets);
  QuartetWorkspace<Real> ws;
  Kokkos::View<Real *> out("intti::jkd::out", batch.nout_total);
  eri_quartets(tab, batch, grid, out, ws);

  std::vector<Real> hD(static_cast<std::size_t>(nreq) * n2);
  std::vector<int> hwantJ(nreq), hwantK(nreq);
  for (int r = 0; r < nreq; ++r) {
    hwantJ[r] = reqs[r].terms != FockTerms::Exchange;
    hwantK[r] = reqs[r].terms != FockTerms::Coulomb;
    for (std::size_t i = 0; i < n2; ++i) hD[r * n2 + i] = reqs[r].D[i];
  }
  auto Dd = to_device(hD, "intti::jkd::D");
  auto wJ = to_device(hwantJ, "intti::jkd::wj"), wK = to_device(hwantK, "intti::jkd::wk");
  auto shL = to_device(hL, "intti::jkd::L"), shOff = to_device(hOff, "intti::jkd::off");
  auto shAl = to_device(hAl, "intti::jkd::al");
  auto dja = to_device(ja, "jkd::a"), djb = to_device(jb, "jkd::b");
  auto djc = to_device(jc, "jkd::c"), djd = to_device(jd, "jkd::d");
  auto dap = to_device(eap, "jkd::ap"), dam = to_device(eam, "jkd::am");
  auto dbp = to_device(ebp, "jkd::bp"), dbm = to_device(ebm, "jkd::bm");
  auto dcp = to_device(ecp, "jkd::cp"), dcm = to_device(ecm, "jkd::cm");
  auto offv = batch.out_offset;
  const std::size_t stride = static_cast<std::size_t>(npert) * n2;
  Kokkos::View<Real *> Jd("intti::jkd::J", static_cast<std::size_t>(nreq) * stride);
  Kokkos::View<Real *> Kd("intti::jkd::K", static_cast<std::size_t>(nreq) * stride);
  Kokkos::parallel_for(
      "intti::jkd::digest", Kokkos::RangePolicy<>(0, njob), KOKKOS_LAMBDA(int j) {
        const int a = dja(j), b = djb(j), c = djc(j), d = djd(j);
        const int la = shL(a), lb = shL(b), lc = shL(c), ld = shL(d);
        const int na = ncart(la), nb = ncart(lb), nc = ncart(lc), nd = ncart(ld);
        const int oa = shOff(a), ob = shOff(b), oc = shOff(c), od = shOff(d);
        const int ent[3][2] = {{dap(j), dam(j)}, {dbp(j), dbm(j)}, {dcp(j), dcm(j)}};
        for (int ka = 0; ka < na; ++ka) {
          int a3[3];
          cart_comp(la, ka, a3[0], a3[1], a3[2]);
          for (int kb = 0; kb < nb; ++kb) {
            int b3[3];
            cart_comp(lb, kb, b3[0], b3[1], b3[2]);
            for (int kc = 0; kc < nc; ++kc) {
              int c3[3];
              cart_comp(lc, kc, c3[0], c3[1], c3[2]);
              for (int kd = 0; kd < nd; ++kd) {
                const std::size_t iab = static_cast<std::size_t>(oa + ka) * nao + ob + kb;
                const std::size_t icd = static_cast<std::size_t>(oc + kc) * nao + od + kd;
                const std::size_t iac = static_cast<std::size_t>(oa + ka) * nao + oc + kc;
                const std::size_t ibd = static_cast<std::size_t>(ob + kb) * nao + od + kd;
                for (int e = 0; e < 3; ++e) {
                  // d/dR of each of the first three slots; the fourth follows
                  // from translational invariance
                  Real dv[4];
                  for (int slot = 0; slot < 3; ++slot) {
                    int sg[2], ci[2];
                    Real co[2];
                    int nt;
                    if (slot == 0)
                      nt = md_grad_terms(la, a3, shAl(a), e, sg, ci, co);
                    else if (slot == 1)
                      nt = md_grad_terms(lb, b3, shAl(b), e, sg, ci, co);
                    else
                      nt = md_grad_terms(lc, c3, shAl(c), e, sg, ci, co);
                    Real v = 0;
                    for (int t = 0; t < nt; ++t) {
                      const int ee = ent[slot][sg[t]];
                      if (ee < 0) continue;
                      const int lp = (sg[t] == 0) ? 1 : -1;
                      const int ma = (slot == 0) ? ncart(la + lp) : na;
                      const int mb = (slot == 1) ? ncart(lb + lp) : nb;
                      const int mc = (slot == 2) ? ncart(lc + lp) : nc;
                      const int ia = (slot == 0) ? ci[t] : ka;
                      const int ib2 = (slot == 1) ? ci[t] : kb;
                      const int ic = (slot == 2) ? ci[t] : kc;
                      (void)ma;
                      const std::size_t idx =
                          ((static_cast<std::size_t>(ia) * mb + ib2) * mc + ic) * nd + kd;
                      v += co[t] * out(offv(ee) + idx);
                    }
                    // md_grad_terms is the ELECTRONIC gradient; the centre
                    // derivative is minus it
                    dv[slot] = -v;
                  }
                  dv[3] = -(dv[0] + dv[1] + dv[2]);
                  const int tgt[4] = {a, b, c, d};
                  for (int slot = 0; slot < 4; ++slot) {
                    const std::size_t po = (static_cast<std::size_t>(3 * tgt[slot] + e)) * n2;
                    for (int r = 0; r < nreq; ++r) {
                      const std::size_t o = static_cast<std::size_t>(r) * stride + po;
                      if (wJ(r)) Kokkos::atomic_add(&Jd(o + iab), dv[slot] * Dd(r * n2 + icd));
                      if (wK(r)) Kokkos::atomic_add(&Kd(o + iac), dv[slot] * Dd(r * n2 + ibd));
                    }
                  }
                }
              }
            }
          }
        }
      });
  Kokkos::fence();
  auto hJ = to_host(Jd), hK = to_host(Kd);
  JKDerivResult<Real> res;
  res.nshell = ns;
  res.nao = nao;
  res.J.resize(nreq);
  res.K.resize(nreq);
  for (int r = 0; r < nreq; ++r) {
    if (hwantJ[r]) res.J[r].assign(hJ.begin() + r * stride, hJ.begin() + (r + 1) * stride);
    if (hwantK[r]) res.K[r].assign(hK.begin() + r * stride, hK.begin() + (r + 1) * stride);
  }
  return res;
}

} // namespace detail

/// Derivative J/K matrices for a set of densities. See JKDerivResult.
template <class Real>
JKDerivResult<Real> jk_deriv_build(const ShellBasis<Real> &basis,
                                   const std::vector<JKRequest<Real>> &reqs,
                                   const TGrid<Real> &grid, Real tau = Real(0)) {
  return detail::jk_deriv_general(basis, reqs, grid, tau);
}

} // namespace intti
