// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Cross-basis two-electron matrices: the bra and ket sides live in DIFFERENT
// basis sets. Two distinct use cases, and they differ in an important way.
//
//   * Coulomb (e.g. electron-proton / nuclear-electronic orbital methods):
//       V_mn = sum_ls (mn|ls) D_ls,   m,n in basis A,  l,s in basis B.
//     Each shell PAIR is homogeneous -- the bra pair is entirely in A, the ket
//     pair entirely in B. The interaction is charge-weighted by the caller
//     (electron-proton is attractive), so no sign is imposed here.
//
//   * Exchange (e.g. projection-free initial guesses, where the guess density
//     lives in a different basis than the target):
//       K_mn = sum_ls (m l|s n) D_ls,  m,n in basis A,  l,s in basis B.
//     Here the pairs are HETEROGENEOUS: the bra pair is (m in A, l in B) and
//     the ket pair is (s in B, n in A). make_pair takes two arbitrary shells,
//     so this needs no new machinery either -- only the right index routing.
//
// Both reduce EXACTLY to the ordinary coulomb_build / exchange_build when the
// two bases are the same, which is how they are validated.
//
// Implementation: the same batched-quartet-consumer pattern as the rest of the
// device path -- enumerate the quartets, evaluate them in one eri_quartets
// call, and digest on device with atomic accumulation. Correctness-first: all
// ordered quartets (no permutational symmetry, which is only partly available
// across distinct bases anyway).

#include <cstddef>
#include <utility>
#include <vector>

#include "batch.hpp"
#include "device.hpp"
#include "fock.hpp"
#include "gto.hpp"
#include "tgrid.hpp"

namespace intti {

namespace detail {

/// Shared quartet-batch construction for the cross-basis builders. `braPairs`
/// and `ketPairs` are (shell,shell) index lists into the two bases; the pair
/// table holds the bra pairs first, then the ket pairs.
template <class Real> struct CrossBatch {
  Kokkos::View<Real *> out;
  Kokkos::View<int *> o0, o1, o2, o3; ///< AO offsets of the four positions
  Kokkos::View<int *> l0, l1, l2, l3; ///< angular momenta of the four positions
  Kokkos::View<int *> boff;
  Kokkos::View<int *> qb, qk; ///< per-quartet bra/ket entry into the lists
  int nq{0};
};

} // namespace detail

/// Cross-basis Coulomb: V (nA x nA, row-major) = sum_ls (mn|ls) D_ls with m,n
/// in `bra` and l,s in `ket`; D is nB x nB row-major over `ket`.
/// With bra == ket this is exactly coulomb_build.
template <class Real>
std::vector<Real> coulomb_cross(const ShellBasis<Real> &bra, const ShellBasis<Real> &ket,
                                const Real *D, const TGrid<Real> &grid) {
  const int nA = bra.nao, nB = ket.nao;
  const int nsA = static_cast<int>(bra.shells.size());
  const int nsB = static_cast<int>(ket.shells.size());
  std::vector<ShellPair<Real>> plist;
  std::vector<int> hoa, hob, hla, hlb; // bra pairs
  for (int a = 0; a < nsA; ++a)
    for (int b = 0; b < nsA; ++b) {
      plist.push_back(make_pair(bra.shells[a], bra.shells[b]));
      hoa.push_back(bra.ao_off[a]);
      hob.push_back(bra.ao_off[b]);
      hla.push_back(bra.shells[a].l);
      hlb.push_back(bra.shells[b].l);
    }
  const int nbra = static_cast<int>(plist.size());
  std::vector<int> hoc, hod, hlc, hld; // ket pairs
  for (int c = 0; c < nsB; ++c)
    for (int d = 0; d < nsB; ++d) {
      plist.push_back(make_pair(ket.shells[c], ket.shells[d]));
      hoc.push_back(ket.ao_off[c]);
      hod.push_back(ket.ao_off[d]);
      hlc.push_back(ket.shells[c].l);
      hld.push_back(ket.shells[d].l);
    }
  const int nket = static_cast<int>(plist.size()) - nbra;
  auto tab = make_pair_table(plist);
  std::vector<std::pair<int, int>> quartets;
  std::vector<int> hqb, hqk;
  quartets.reserve(static_cast<std::size_t>(nbra) * nket);
  for (int ib = 0; ib < nbra; ++ib)
    for (int ik = 0; ik < nket; ++ik) {
      quartets.push_back({ib, nbra + ik});
      hqb.push_back(ib);
      hqk.push_back(ik);
    }
  auto batch = make_batch(tab, quartets);
  QuartetWorkspace<Real> ws;
  Kokkos::View<Real *> out("intti::xj::out", batch.nout_total);
  eri_quartets(tab, batch, grid, out, ws);

  auto Dd = detail::to_device(D, static_cast<std::size_t>(nB) * nB, "intti::xj::D");
  auto oa = detail::to_device(hoa, "xj::oa"), ob = detail::to_device(hob, "xj::ob");
  auto la = detail::to_device(hla, "xj::la"), lb = detail::to_device(hlb, "xj::lb");
  auto oc = detail::to_device(hoc, "xj::oc"), od = detail::to_device(hod, "xj::od");
  auto lc = detail::to_device(hlc, "xj::lc"), ld = detail::to_device(hld, "xj::ld");
  auto qb = detail::to_device(hqb, "xj::qb"), qk = detail::to_device(hqk, "xj::qk");
  auto offv = batch.out_offset;
  Kokkos::View<Real *> Vd("intti::xj::V", static_cast<std::size_t>(nA) * nA);
  Kokkos::parallel_for(
      "intti::xj::digest", Kokkos::RangePolicy<>(0, batch.nq), KOKKOS_LAMBDA(int iq) {
        const int ib = qb(iq), ik = qk(iq);
        const int na = ncart(la(ib)), nb = ncart(lb(ib));
        const int nc = ncart(lc(ik)), nd = ncart(ld(ik));
        const int base = offv(iq);
        for (int ka = 0; ka < na; ++ka)
          for (int kb = 0; kb < nb; ++kb) {
            Real acc = 0;
            for (int kc = 0; kc < nc; ++kc)
              for (int kd = 0; kd < nd; ++kd)
                acc += out(base + (((static_cast<std::size_t>(ka) * nb + kb) * nc + kc) * nd + kd)) *
                       Dd(static_cast<std::size_t>(oc(ik) + kc) * nB + od(ik) + kd);
            Kokkos::atomic_add(
                &Vd(static_cast<std::size_t>(oa(ib) + ka) * nA + ob(ib) + kb), acc);
          }
      });
  return detail::to_host(Vd);
}

/// Cross-basis exchange: K (nA x nA) = sum_ls (m l|s n) D_ls with m,n in `bra`
/// and l,s in `ket`; D is nB x nB over `ket`. The shell pairs are heterogeneous
/// (bra pair = (A,B), ket pair = (B,A)). With bra == ket this is exactly
/// exchange_build.
template <class Real>
std::vector<Real> exchange_cross(const ShellBasis<Real> &bra, const ShellBasis<Real> &ket,
                                 const Real *D, const TGrid<Real> &grid) {
  const int nA = bra.nao, nB = ket.nao;
  const int nsA = static_cast<int>(bra.shells.size());
  const int nsB = static_cast<int>(ket.shells.size());
  std::vector<ShellPair<Real>> plist;
  // bra pairs (m in A, l in B)
  std::vector<int> hom, hol, hlm, hll;
  for (int m = 0; m < nsA; ++m)
    for (int l = 0; l < nsB; ++l) {
      plist.push_back(make_pair(bra.shells[m], ket.shells[l]));
      hom.push_back(bra.ao_off[m]);
      hol.push_back(ket.ao_off[l]);
      hlm.push_back(bra.shells[m].l);
      hll.push_back(ket.shells[l].l);
    }
  const int nbra = static_cast<int>(plist.size());
  // ket pairs (s in B, n in A)
  std::vector<int> hos, hon, hls, hln;
  for (int s = 0; s < nsB; ++s)
    for (int n = 0; n < nsA; ++n) {
      plist.push_back(make_pair(ket.shells[s], bra.shells[n]));
      hos.push_back(ket.ao_off[s]);
      hon.push_back(bra.ao_off[n]);
      hls.push_back(ket.shells[s].l);
      hln.push_back(bra.shells[n].l);
    }
  const int nket = static_cast<int>(plist.size()) - nbra;
  auto tab = make_pair_table(plist);
  std::vector<std::pair<int, int>> quartets;
  std::vector<int> hqb, hqk;
  quartets.reserve(static_cast<std::size_t>(nbra) * nket);
  for (int ib = 0; ib < nbra; ++ib)
    for (int ik = 0; ik < nket; ++ik) {
      quartets.push_back({ib, nbra + ik});
      hqb.push_back(ib);
      hqk.push_back(ik);
    }
  auto batch = make_batch(tab, quartets);
  QuartetWorkspace<Real> ws;
  Kokkos::View<Real *> out("intti::xk::out", batch.nout_total);
  eri_quartets(tab, batch, grid, out, ws);

  auto Dd = detail::to_device(D, static_cast<std::size_t>(nB) * nB, "intti::xk::D");
  auto om = detail::to_device(hom, "xk::om"), ol = detail::to_device(hol, "xk::ol");
  auto lm = detail::to_device(hlm, "xk::lm"), ll = detail::to_device(hll, "xk::ll");
  auto os_ = detail::to_device(hos, "xk::os"), on = detail::to_device(hon, "xk::on");
  auto ls = detail::to_device(hls, "xk::ls"), ln = detail::to_device(hln, "xk::ln");
  auto qb = detail::to_device(hqb, "xk::qb"), qk = detail::to_device(hqk, "xk::qk");
  auto offv = batch.out_offset;
  Kokkos::View<Real *> Kd("intti::xk::K", static_cast<std::size_t>(nA) * nA);
  Kokkos::parallel_for(
      "intti::xk::digest", Kokkos::RangePolicy<>(0, batch.nq), KOKKOS_LAMBDA(int iq) {
        const int ib = qb(iq), ik = qk(iq);
        const int nm = ncart(lm(ib)), nl = ncart(ll(ib));
        const int nss = ncart(ls(ik)), nn = ncart(ln(ik));
        const int base = offv(iq);
        for (int km = 0; km < nm; ++km)
          for (int kn = 0; kn < nn; ++kn) {
            Real acc = 0;
            for (int kl = 0; kl < nl; ++kl)
              for (int ks = 0; ks < nss; ++ks)
                acc += out(base + (((static_cast<std::size_t>(km) * nl + kl) * nss + ks) * nn + kn)) *
                       Dd(static_cast<std::size_t>(ol(ib) + kl) * nB + os_(ik) + ks);
            Kokkos::atomic_add(
                &Kd(static_cast<std::size_t>(om(ib) + km) * nA + on(ik) + kn), acc);
          }
      });
  return detail::to_host(Kd);
}

} // namespace intti
