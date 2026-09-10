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
#include "contracted.hpp"
#include "screening.hpp"
#include "space.hpp"
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

/// J and K left where they were computed: device views of nreq x nao^2 each.
/// The entry point for a caller that keeps its matrices on the device -- see
/// space.hpp for why the API is space-flexible at all.
template <class Real> struct JKDeviceResult {
  Kokkos::View<Real *> J, K;
  std::vector<int> wantJ, wantK;
  int nreq{0}, n2{0};
};

namespace detail {

/// General path: every ordered shell quartet, no assumption about the density.
/// One pass over the integrals digests all requests, which is the point of the
/// vector interface -- the ERIs are the expensive part and they are shared.
/// Only the bra-ket fold (ab|cd) = (cd|ab) is used; the intra-pair folds are
/// available for real integrals but interact with the general-D exchange
/// accumulation, so they are left to the fused symmetric path.
template <class Real>
JKDeviceResult<Real> jk_build_general(const ShellBasis<Real> &basis,
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
  // Same tolerance as the Schwarz screen above: both bound the discarded
  // contribution to a quartet in absolute terms, so one knob rather than two.
  if (tau > Real(0)) t_screen_batch(batch, plist, grid, tau);
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
  return JKDeviceResult<Real>{Jd, Kd, hwantJ, hwantK, nreq, static_cast<int>(n2)};
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
  if (!all_symmetric) {
    // general path: computed on the device, brought down for this host-facing
    // entry point. jk_build_device returns it without the copy.
    const auto dev = detail::jk_build_general(basis, reqs, grid, tau);
    const auto hJ = detail::to_host(dev.J), hK = detail::to_host(dev.K);
    JKResult<Real> out;
    out.J.resize(nreq);
    out.K.resize(nreq);
    for (int r = 0; r < nreq; ++r) {
      if (dev.wantJ[r])
        out.J[r].assign(hJ.begin() + static_cast<std::ptrdiff_t>(r) * dev.n2,
                        hJ.begin() + static_cast<std::ptrdiff_t>(r + 1) * dev.n2);
      if (dev.wantK[r])
        out.K[r].assign(hK.begin() + static_cast<std::ptrdiff_t>(r) * dev.n2,
                        hK.begin() + static_cast<std::ptrdiff_t>(r + 1) * dev.n2);
    }
    return out;
  }

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

/// Space-flexible J/K: the densities and the results may each live on the host
/// or on the device, in any of the four combinations, and nothing is copied that
/// is already where it needs to be.
///
///   host in, host out      jk_build (above) -- unchanged, and still the right
///                          call for a host-only code
///   device in, device out  this, with device views: zero copies
///   mixed                  this, with one side host: exactly one copy, the
///                          unavoidable one
///
/// `D` is one view per request, in ANY memory space; `J` and `K` likewise, and
/// either may be empty to skip that term. Views are nao^2 per request.
template <class Real, class DView, class OutView>
void jk_build_into(const ShellBasis<Real> &basis, const std::vector<DView> &D,
                   const std::vector<DensitySymmetry> &sym,
                   const std::vector<FockTerms> &terms, const TGrid<Real> &grid,
                   std::vector<OutView> &J, std::vector<OutView> &K,
                   Real tau = Real(0)) {
  const int nreq = static_cast<int>(D.size());
  const std::size_t n2 = static_cast<std::size_t>(basis.nao) * basis.nao;
  // Stage the densities where the kernel needs them. device_in is a no-op for
  // a caller who is already there.
  std::vector<std::vector<Real>> staged(nreq);
  std::vector<JKRequest<Real>> reqs(nreq);
  for (int r = 0; r < nreq; ++r) {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, D[r]);
    staged[r].assign(h.data(), h.data() + n2);
    reqs[r] = JKRequest<Real>{staged[r].data(), sym[r], terms[r]};
  }
  const auto dev = detail::jk_build_general(basis, reqs, grid, tau);
  auto publish = [&](std::vector<OutView> &dst, const Kokkos::View<Real *> &src,
                     const std::vector<int> &want) {
    for (int r = 0; r < static_cast<int>(dst.size()); ++r) {
      if (!want[r] || dst[r].extent(0) == 0) continue;
      auto sub = Kokkos::subview(
          src, Kokkos::make_pair(static_cast<std::size_t>(r) * n2,
                                 static_cast<std::size_t>(r + 1) * n2));
      Kokkos::deep_copy(dst[r], sub); // no-op cost when dst is already device-side
    }
  };
  publish(J, dev.J, dev.wantJ);
  publish(K, dev.K, dev.wantK);
}

/// Derivative J/K in the BRA-GRADIENT convention: the derivative acts only on
/// the FIRST AO index, and the result is indexed by Cartesian component and AO
/// pair rather than folded onto shell centres.
///
///   J^x_ij = sum_kl d(ij|kl)/dA_i D_lk,   K^x_il = sum_jk d(ij|kl)/dA_i D_jk
///
/// This is what a gradient implementation actually asks its integral layer for:
/// it is exactly pyscf.grad.rhf.get_jk(mol, dm), whose docstring reads
/// J = ((-nabla i) j|kl) D_lk -- and -nabla_i is +d/dA_i, since a Gaussian
/// depends on r - A. The caller folds these onto atoms with its own AO-to-atom
/// map, which is why the fold is deliberately NOT done here; jk_deriv_build is
/// the centre-folded form for callers that want that instead.
///
/// Note the ket density indices: D_lk and D_jk, TRANSPOSED relative to the
/// quartet's slot order. That is immaterial for a symmetric density and is not
/// for a general one, so it is followed exactly.
///
/// No permutational symmetry is used: differentiating one slot breaks it.
template <class Real> struct JKDerivAO {
  int nao{0};
  std::vector<std::vector<Real>> J, K; ///< per request, 3 x nao x nao
};

namespace detail {

/// jk_deriv_ao_build over primitive shells plus a fan-out onto contracted AOs.
/// The integrals are evaluated per PRIMITIVE quartet; the contraction enters
/// only in the digest, so nothing is recomputed per contracted index.
template <class Real>
JKDerivAO<Real> jk_deriv_ao_impl(const ShellBasis<Real> &basis, const ShellFanout<Real> &fan,
                                 const std::vector<JKRequest<Real>> &reqs,
                                 const TGrid<Real> &grid, Real tau) {
  const int ns = static_cast<int>(basis.shells.size());
  const int nao = fan.nao;
  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
  const int nreq = static_cast<int>(reqs.size());

  std::vector<int> hL(ns), hOff(ns);
  std::vector<Real> hAl(ns);
  for (int i = 0; i < ns; ++i) {
    hL[i] = basis.shells[i].l;
    hOff[i] = fan.base[i];
    hAl[i] = basis.shells[i].alpha;
  }
  std::vector<ShellPair<Real>> plist;
  std::vector<int> pid(static_cast<std::size_t>(ns) * 3 * ns, -1);
  auto pair_id = [&](int si, int di, int sj) {
    if (hL[si] + di < 0) return -1;
    const std::size_t key = (static_cast<std::size_t>(si) * 3 + (di + 1)) * ns + sj;
    if (pid[key] < 0) {
      PrimitiveShell<Real> a = basis.shells[si], b = basis.shells[sj];
      a.l += di;
      plist.push_back(make_pair(a, b));
      pid[key] = static_cast<int>(plist.size()) - 1;
    }
    return pid[key];
  };
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      pair_id(a, 0, b);
      pair_id(a, 1, b);
      pair_id(a, -1, b);
    }
  auto tab = make_pair_table(plist);
  std::vector<Real> Q;
  if (tau > Real(0)) Q = schwarz(tab, plist, grid);
  auto scale = [&](int s) {
    const Real two_al = 2 * hAl[s];
    return two_al > Real(hL[s]) ? two_al : Real(hL[s]);
  };
  std::vector<std::pair<int, int>> quartets;
  std::vector<int> ja, jb, jc, jd, eap, eam;
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b)
      for (int c = 0; c < ns; ++c)
        for (int d = 0; d < ns; ++d) {
          const int ket = pair_id(c, 0, d);
          if (tau > Real(0) && scale(a) * Q[pair_id(a, 0, b)] * Q[ket] < tau) continue;
          auto emit = [&](int bra) {
            if (bra < 0) return -1;
            const int e = static_cast<int>(quartets.size());
            quartets.push_back({bra, ket});
            return e;
          };
          ja.push_back(a); jb.push_back(b); jc.push_back(c); jd.push_back(d);
          eap.push_back(emit(pair_id(a, 1, b)));
          eam.push_back(emit(pair_id(a, -1, b)));
        }
  const int njob = static_cast<int>(ja.size());
  auto batch = make_batch(tab, quartets);
  QuartetWorkspace<Real> ws;
  Kokkos::View<Real *> out("intti::jkao::out", batch.nout_total);
  eri_quartets(tab, batch, grid, out, ws);

  std::vector<Real> hD(static_cast<std::size_t>(nreq) * n2);
  std::vector<int> hwantJ(nreq), hwantK(nreq);
  for (int r = 0; r < nreq; ++r) {
    hwantJ[r] = reqs[r].terms != FockTerms::Exchange;
    hwantK[r] = reqs[r].terms != FockTerms::Coulomb;
    for (std::size_t i = 0; i < n2; ++i) hD[r * n2 + i] = reqs[r].D[i];
  }
  auto Dd = detail::to_device(hD, "intti::jkao::D");
  auto wJ = detail::to_device(hwantJ, "intti::jkao::wj"), wK = detail::to_device(hwantK, "intti::jkao::wk");
  auto shL = detail::to_device(hL, "intti::jkao::L"), shOff = detail::to_device(hOff, "intti::jkao::off");
  auto shAl = detail::to_device(hAl, "intti::jkao::al");
  auto fnc = detail::to_device(fan.nctr, "intti::jkao::nctr");
  auto fco = detail::to_device(fan.coff, "intti::jkao::coff");
  auto fw = detail::to_device(fan.w, "intti::jkao::w");
  auto dja = detail::to_device(ja, "jkao::a"), djb = detail::to_device(jb, "jkao::b");
  auto djc = detail::to_device(jc, "jkao::c"), djd = detail::to_device(jd, "jkao::d");
  auto dap = detail::to_device(eap, "jkao::ap"), dam = detail::to_device(eam, "jkao::am");
  auto offv = batch.out_offset;
  const std::size_t stride = static_cast<std::size_t>(3) * n2;
  Kokkos::View<Real *> Jd("intti::jkao::J", static_cast<std::size_t>(nreq) * stride);
  Kokkos::View<Real *> Kd("intti::jkao::K", static_cast<std::size_t>(nreq) * stride);
  Kokkos::parallel_for(
      "intti::jkao::digest", Kokkos::RangePolicy<>(0, njob), KOKKOS_LAMBDA(int j) {
        const int a = dja(j), b = djb(j), c = djc(j), d = djd(j);
        const int la = shL(a), lb = shL(b), lc = shL(c), ld = shL(d);
        const int na = ncart(la), nb = ncart(lb), nc = ncart(lc), nd = ncart(ld);
        const int oa = shOff(a), ob = shOff(b), oc = shOff(c), od = shOff(d);
        const int ent[2] = {dap(j), dam(j)};
        for (int ka = 0; ka < na; ++ka) {
          int a3[3];
          cart_comp(la, ka, a3[0], a3[1], a3[2]);
          for (int kb = 0; kb < nb; ++kb)
            for (int kc = 0; kc < nc; ++kc)
              for (int kd = 0; kd < nd; ++kd) {
                for (int e = 0; e < 3; ++e) {
                  int sg[2], ci[2];
                  Real co[2];
                  const int nt = detail::md_grad_terms(la, a3, shAl(a), e, sg, ci, co);
                  Real v = 0;
                  for (int t = 0; t < nt; ++t) {
                    const int ee = ent[sg[t]];
                    if (ee < 0) continue;
                    const std::size_t idx =
                        ((static_cast<std::size_t>(ci[t]) * nb + kb) * nc + kc) * nd + kd;
                    v += co[t] * out(offv(ee) + idx);
                  }
                  // md_grad_terms is d/dx; the centre derivative d/dA -- which is
                  // what PySCF's (-nabla i) means -- is minus it
                  const Real dv = -v;
                  const std::size_t xo = static_cast<std::size_t>(e) * n2;
                  // Contraction fan-out: one primitive quartet feeds every
                  // combination of contracted functions on its four shells. All
                  // trip counts are 1 for a primitive basis.
                  for (int cA = 0; cA < fnc(a); ++cA) {
                    const Real wa = fw(fco(a) + cA) * dv;
                    const int I = oa + cA * na + ka;
                    for (int cB = 0; cB < fnc(b); ++cB) {
                      const Real wab = wa * fw(fco(b) + cB);
                      const int Jj = ob + cB * nb + kb;
                      for (int cC = 0; cC < fnc(c); ++cC) {
                        const Real wabc = wab * fw(fco(c) + cC);
                        const int Kk = oc + cC * nc + kc;
                        for (int cD = 0; cD < fnc(d); ++cD) {
                          const Real w = wabc * fw(fco(d) + cD);
                          const int L = od + cD * nd + kd;
                          for (int r = 0; r < nreq; ++r) {
                            const std::size_t o = static_cast<std::size_t>(r) * stride + xo;
                            const std::size_t db = static_cast<std::size_t>(r) * n2;
                            // J^x_ij += dv D_lk ; K^x_il += dv D_jk  (ket indices
                            // transposed, exactly as 'lk->s1ij' / 'jk->s1il')
                            if (wJ(r))
                              Kokkos::atomic_add(
                                  &Jd(o + static_cast<std::size_t>(I) * nao + Jj),
                                  w * Dd(db + static_cast<std::size_t>(L) * nao + Kk));
                            if (wK(r))
                              Kokkos::atomic_add(
                                  &Kd(o + static_cast<std::size_t>(I) * nao + L),
                                  w * Dd(db + static_cast<std::size_t>(Jj) * nao + Kk));
                          }
                        }
                      }
                    }
                  }
                }
              }
        }
      });
  Kokkos::fence();
  auto hJ = detail::to_host(Jd), hK = detail::to_host(Kd);
  JKDerivAO<Real> res;
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

template <class Real>
JKDerivAO<Real> jk_deriv_ao_build(const ShellBasis<Real> &basis,
                                  const std::vector<JKRequest<Real>> &reqs,
                                  const TGrid<Real> &grid, Real tau = Real(0)) {
  return detail::jk_deriv_ao_impl(basis, detail::identity_fanout(basis), reqs, grid, tau);
}

/// Same, over a generally-contracted basis. No density-symmetry restriction:
/// the kernel walks all ordered quartets and reads D at explicit index pairs,
/// so a general or antisymmetric density -- what magnetic response and the CPHF
/// right-hand side actually supply -- is served exactly as a symmetric one is.
template <class Real>
JKDerivAO<Real> jk_deriv_ao_build(const ContractedBasis<Real> &cb,
                                  const std::vector<JKRequest<Real>> &reqs,
                                  const TGrid<Real> &grid, Real tau = Real(0)) {
  ShellBasis<Real> prims;
  auto fan = detail::expand_contracted(cb, prims);
  return detail::jk_deriv_ao_impl(prims, fan, reqs, grid, tau);
}

/// The four int2e_ip1 contractions pyscf.hessian.rhf.make_h1 needs, with the
/// differentiated index restricted to one atom's shells [shl0, shl1):
///
///   vj1[x][k][l] = sum_{j,i in slice} d(ij|kl)/dA_i D_ji   ('ji->s2kl')
///   vj2[x][i][j] = sum_{l,k}          d(ij|kl)/dA_i D_lk   ('lk->s1ij')
///   vk1[x][k][j] = sum_{l,i in slice} d(ij|kl)/dA_i D_li   ('li->s1kj')
///   vk2[x][i][l] = sum_{j,k}          d(ij|kl)/dA_i D_jk   ('jk->s1il')
///
/// vj2 and vk2 are jk_deriv_ao_build's J and K; vj1 and vk1 are the other two
/// index routings of the same derivative integrals, so all four share one pass.
///
/// SIGN. PySCF contracts int2e_ip1 = (nabla_i j|kl) against MINUS the density
/// and returns that. Since nabla = -d/dA, computing with d/dA against the
/// POSITIVE density reproduces its returned values directly -- no extra
/// negation, and the two sign conventions cancel exactly once.
///
/// vj1 is symmetric in (k,l) because (ij|kl) = (ij|lk); PySCF obtains it by
/// exploiting that symmetry ('s2kl') and filling, we obtain it by summing all
/// ordered quartets. Same matrix.
template <class Real> struct IP1Contractions {
  int nao{0};
  std::vector<Real> vj1, vj2, vk1, vk2; ///< each 3 x nao x nao
};

namespace detail {

/// ip1_h1_contractions over primitive shells plus a contracted-AO fan-out.
/// [shl0, shl1) is a PRIMITIVE shell range here; the contracted entry point
/// translates the contracted slice into it.
template <class Real>
IP1Contractions<Real> ip1_h1_impl(const ShellBasis<Real> &basis, const ShellFanout<Real> &fan,
                                  const Real *D, int shl0, int shl1, const TGrid<Real> &grid,
                                  Real tau) {
  const int ns = static_cast<int>(basis.shells.size());
  const int nao = fan.nao;
  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
  std::vector<int> hL(ns), hOff(ns);
  std::vector<Real> hAl(ns);
  for (int i = 0; i < ns; ++i) {
    hL[i] = basis.shells[i].l;
    hOff[i] = fan.base[i];
    hAl[i] = basis.shells[i].alpha;
  }
  std::vector<ShellPair<Real>> plist;
  std::vector<int> pid(static_cast<std::size_t>(ns) * 3 * ns, -1);
  auto pair_id = [&](int si, int di, int sj) {
    if (hL[si] + di < 0) return -1;
    const std::size_t key = (static_cast<std::size_t>(si) * 3 + (di + 1)) * ns + sj;
    if (pid[key] < 0) {
      PrimitiveShell<Real> a = basis.shells[si], b = basis.shells[sj];
      a.l += di;
      plist.push_back(make_pair(a, b));
      pid[key] = static_cast<int>(plist.size()) - 1;
    }
    return pid[key];
  };
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      pair_id(a, 0, b);
      pair_id(a, 1, b);
      pair_id(a, -1, b);
    }
  auto tab = make_pair_table(plist);
  std::vector<Real> Q;
  if (tau > Real(0)) Q = schwarz(tab, plist, grid);
  auto sc = [&](int s) {
    const Real t = 2 * hAl[s];
    return t > Real(hL[s]) ? t : Real(hL[s]);
  };
  std::vector<std::pair<int, int>> quartets;
  std::vector<int> ja, jb, jc, jd, eap, eam;
  for (int a = shl0; a < shl1; ++a) // only the differentiated slice
    for (int b = 0; b < ns; ++b)
      for (int c = 0; c < ns; ++c)
        for (int d = 0; d < ns; ++d) {
          const int ket = pair_id(c, 0, d);
          if (tau > Real(0) && sc(a) * Q[pair_id(a, 0, b)] * Q[ket] < tau) continue;
          auto emit = [&](int bra) {
            if (bra < 0) return -1;
            const int e = static_cast<int>(quartets.size());
            quartets.push_back({bra, ket});
            return e;
          };
          ja.push_back(a); jb.push_back(b); jc.push_back(c); jd.push_back(d);
          eap.push_back(emit(pair_id(a, 1, b)));
          eam.push_back(emit(pair_id(a, -1, b)));
        }
  const int njob = static_cast<int>(ja.size());
  auto batch = make_batch(tab, quartets);
  QuartetWorkspace<Real> ws;
  Kokkos::View<Real *> out("intti::ip1::out", batch.nout_total);
  eri_quartets(tab, batch, grid, out, ws);
  auto Dd = detail::to_device(D, n2, "intti::ip1::D");
  auto shL = detail::to_device(hL, "ip1::L"), shOff = detail::to_device(hOff, "ip1::off");
  auto shAl = detail::to_device(hAl, "ip1::al");
  auto fnc = detail::to_device(fan.nctr, "ip1::nctr");
  auto fco = detail::to_device(fan.coff, "ip1::coff");
  auto fw = detail::to_device(fan.w, "ip1::w");
  auto dja = detail::to_device(ja, "ip1::a"), djb = detail::to_device(jb, "ip1::b");
  auto djc = detail::to_device(jc, "ip1::c"), djd = detail::to_device(jd, "ip1::d");
  auto dap = detail::to_device(eap, "ip1::ap"), dam = detail::to_device(eam, "ip1::am");
  auto offv = batch.out_offset;
  const std::size_t stride = static_cast<std::size_t>(3) * n2;
  Kokkos::View<Real *> J1("ip1::vj1", stride), J2("ip1::vj2", stride);
  Kokkos::View<Real *> K1("ip1::vk1", stride), K2("ip1::vk2", stride);
  Kokkos::parallel_for(
      "intti::ip1::digest", Kokkos::RangePolicy<>(0, njob), KOKKOS_LAMBDA(int j) {
        const int a = dja(j), b = djb(j), c = djc(j), d = djd(j);
        const int la = shL(a), lb = shL(b), lc = shL(c), ld = shL(d);
        const int na = ncart(la), nb = ncart(lb), nc = ncart(lc), nd = ncart(ld);
        const int oa = shOff(a), ob = shOff(b), oc = shOff(c), od = shOff(d);
        const int ent[2] = {dap(j), dam(j)};
        for (int ka = 0; ka < na; ++ka) {
          int a3[3];
          cart_comp(la, ka, a3[0], a3[1], a3[2]);
          for (int kb = 0; kb < nb; ++kb)
            for (int kc = 0; kc < nc; ++kc)
              for (int kd = 0; kd < nd; ++kd) {
                for (int e = 0; e < 3; ++e) {
                  int sg[2], ci[2];
                  Real co[2];
                  const int nt = detail::md_grad_terms(la, a3, shAl(a), e, sg, ci, co);
                  Real v = 0;
                  for (int t = 0; t < nt; ++t) {
                    const int ee = ent[sg[t]];
                    if (ee < 0) continue;
                    const std::size_t idx =
                        ((static_cast<std::size_t>(ci[t]) * nb + kb) * nc + kc) * nd + kd;
                    v += co[t] * out(offv(ee) + idx);
                  }
                  const Real dv = -v; // d/dA
                  const std::size_t xo = static_cast<std::size_t>(e) * n2;
                  // Contraction fan-out (all trip counts 1 for a primitive basis).
                  for (int cA = 0; cA < fnc(a); ++cA) {
                    const Real wa = fw(fco(a) + cA) * dv;
                    const int I = oa + cA * na + ka;
                    const std::size_t iI = static_cast<std::size_t>(I) * nao;
                    for (int cB = 0; cB < fnc(b); ++cB) {
                      const Real wab = wa * fw(fco(b) + cB);
                      const int Jj = ob + cB * nb + kb;
                      for (int cC = 0; cC < fnc(c); ++cC) {
                        const Real wabc = wab * fw(fco(c) + cC);
                        const int Kk = oc + cC * nc + kc;
                        const std::size_t iK = static_cast<std::size_t>(Kk) * nao;
                        for (int cD = 0; cD < fnc(d); ++cD) {
                          const Real w = wabc * fw(fco(d) + cD);
                          const int L = od + cD * nd + kd;
                          Kokkos::atomic_add(&J1(xo + iK + L),
                                             w * Dd(static_cast<std::size_t>(Jj) * nao + I));
                          Kokkos::atomic_add(&J2(xo + iI + Jj),
                                             w * Dd(static_cast<std::size_t>(L) * nao + Kk));
                          Kokkos::atomic_add(&K1(xo + iK + Jj),
                                             w * Dd(static_cast<std::size_t>(L) * nao + I));
                          Kokkos::atomic_add(&K2(xo + iI + L),
                                             w * Dd(static_cast<std::size_t>(Jj) * nao + Kk));
                        }
                      }
                    }
                  }
                }
              }
        }
      });
  Kokkos::fence();
  IP1Contractions<Real> r;
  r.nao = nao;
  r.vj1 = detail::to_host(J1);
  r.vj2 = detail::to_host(J2);
  r.vk1 = detail::to_host(K1);
  r.vk2 = detail::to_host(K2);
  return r;
}

} // namespace detail

template <class Real>
IP1Contractions<Real> ip1_h1_contractions(const ShellBasis<Real> &basis, const Real *D,
                                          int shl0, int shl1, const TGrid<Real> &grid,
                                          Real tau = Real(0)) {
  return detail::ip1_h1_impl(basis, detail::identity_fanout(basis), D, shl0, shl1, grid, tau);
}

/// Same, over a generally-contracted basis. [shl0, shl1) is a CONTRACTED shell
/// range -- the atom's shells, as PySCF's aoslice_by_atom gives them -- and is
/// translated to the corresponding contiguous primitive range, since
/// contracted_primitives expands in shell order.
template <class Real>
IP1Contractions<Real> ip1_h1_contractions(const ContractedBasis<Real> &cb, const Real *D,
                                          int shl0, int shl1, const TGrid<Real> &grid,
                                          Real tau = Real(0)) {
  ShellBasis<Real> prims;
  auto fan = detail::expand_contracted(cb, prims);
  int p0 = 0, p1 = 0;
  for (int i = 0; i < static_cast<int>(cb.shells.size()); ++i) {
    if (i < shl0) p0 += cb.shells[i].nprim();
    if (i < shl1) p1 += cb.shells[i].nprim();
  }
  return detail::ip1_h1_impl(prims, fan, D, p0, p1, grid, tau);
}

} // namespace intti
