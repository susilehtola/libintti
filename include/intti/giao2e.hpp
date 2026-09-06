// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Two-electron GIAO field derivatives dJ/dB, dK/dB -- the two-electron half of
// the first-order Fock matrix for magnetic response (NMR shieldings,
// magnetizabilities). The Coulomb operator 1/r12 is multiplicative, so as with
// the overlap and nuclear cases the magnetic field differentiates only the
// London plane-wave phases of the two shell-pair products:
//
//   d/dB_k (ab|cd) = (i/2)[e_k x (R_a - R_b)] . (a r1 b|cd)
//                  + (i/2)[e_k x (R_c - R_d)] . (ab|c r2 d),
//
// with (a r1_e b|cd) = (a^{+e} b|cd) + R_a[e] (ab|cd) the position-weighted
// ERI (bra promoted by one unit; the gauge origin cancels because the phase
// vectors are pair-centre *differences*). At B = 0 the ERIs are real, so the
// derivative is i times a real combination of real ERI blocks. Contracting
// with the AO density gives the purely imaginary matrices
//   dJ_munu/dB_k = sum_ls d/dB_k (mn|ls) D_ls,
//   dK_munu/dB_k = sum_ls d/dB_k (ml|ns) D_ls.
// Matrix-level: density in, dJ/dK matrices out; quartets stay internal.
//
// Permutational symmetry (see erigrad.hpp): loop canonical shell quartets only
// and replay the *unchanged* per-quartet contraction on every distinct orbit
// member. The body recomputes the phase-vector cross products from each
// member's own centres, so the sign flips under a<->b, c<->d and (ab)<->(cd)
// are handled automatically; only the (permutation-invariant) base and
// singly-promoted ERI blocks are shared (~8x fewer quartet evals).

#include <array>
#include <complex>
#include <cstdint>
#include <cstddef>
#include <vector>

#include "batch.hpp"   // PairTable, make_batch, eri_quartets
#include "device.hpp"  // detail::to_device / to_host
#include "erigrad.hpp" // detail::comp_index, detail::eri_block4, permute_block, eri_perms
#include "fock.hpp"
#include "giao.hpp" // make_giao_pair (finite-field London pairs)
#include "gto.hpp"
#include "quartet.hpp"
#include "tgrid.hpp"

namespace intti {

/// Complex Coulomb and exchange matrices in a FINITE magnetic field B -- the
/// two-electron half of the London/GIAO Fock matrix, the piece a finite-field
/// SCF needs (as opposed to the dJ/dB, dK/dB response derivatives below).
/// A GIAO ERI is just eri_quartet() on complex scalars (make_giao_pair shifts
/// the product centre into the complex plane; the t quadrature never forms a
/// Boys function, so nothing else changes). With the FIRST index of each London
/// pair conjugated (make_giao_pair),
///   J_mn = sum_ls (mn|ls)_B D_ls,   K_mn = sum_ls (ml|sn)_B D_ls.
/// NOTE the exchange index order (ml|sn), NOT the (ml|ns) of giao_jk_dB: for a
/// real ERI the two coincide (which is why the B=0 derivative may use either),
/// but at finite B only (ml|sn) gives a Hermitian K. Using (ab|cd)* = (ba|dc)
/// and (ab|cd) = (cd|ab), both J and K are Hermitian for Hermitian D.
/// D is the complex (Hermitian) AO density, row-major.
///
/// The London phases break the INTRA-PAIR folds of the 8-fold ERI symmetry
/// ((mn|ls) != (nm|ls); it conjugates instead), but the bra-ket swap
/// (mn|ls) = (ls|mn) survives -- it needs only that 1/r12 is symmetric in the
/// two electrons. So the builder enumerates ib <= ik over ordered pairs and
/// replays each off-diagonal block in both orientations: 2-fold, not 8-fold.
template <class Real> struct GiaoJK {
  std::vector<std::complex<Real>> J, K;
};

/// Field-free J/K for a COMPLEX (Hermitian) density -- the complex-MO path,
/// e.g. complex/generalised HF or spin-orbit-coupled densities. This is exactly
/// giao_jk at zero field: real integrals contracted with a complex density.
///
/// Use this rather than coulomb_build/exchange_build when the MO coefficients
/// are complex. The real builders cannot serve that case:
///   * exchange_build assumes K_ab = K_ba, which holds only for a SYMMETRIC
///     density -- so even splitting K(D) = K(Re D) + i K(Im D) is unsafe, since
///     Im D of a Hermitian D is antisymmetric;
///   * coulomb_build applies a symmetry fold (D_ij + D_ji), i.e. it sees only
///     the symmetric part by construction.
/// Here no permutational symmetry is assumed (all ordered quartets), and the
/// exchange uses the Hermitian-correct K_mn = sum_ls (ml|sn) D_ls.
///
/// Physically J and K differ: the real ERI is symmetric under the ket-pair
/// swap, so the antisymmetric Im D cancels and J depends only on Re D (and is
/// real), while K genuinely depends on Im D. Both are Hermitian for Hermitian D.
template <class Real> using ComplexJK = GiaoJK<Real>;

template <class Real>
ComplexJK<Real> complex_jk(const ShellBasis<Real> &basis, const std::complex<Real> *D,
                           const TGrid<Real> &grid, Real tau = Real(0)) {
  const Real zero_field[3] = {Real(0), Real(0), Real(0)};
  return giao_jk(basis, D, zero_field, grid, tau);
}

namespace detail {

/// Schwarz bound for London (complex) pair densities. The ERI is a BILINEAR
/// form in the pair densities, not a sesquilinear one: with rho_P = omega_a^*
/// omega_b and the convention (ab|cd) = int int rho_(ab)(1) rho_(cd)(2)/r12,
///
///   (ab|cd) = <rho_ba | rho_cd>,   <f|g> = int int f^*(1) g(2) / r12,
///
/// because rho_P^* = rho_(ba). Cauchy-Schwarz then gives
///
///   |(ab|cd)| <= sqrt((ab|ba)) sqrt((dc|cd)),
///
/// and bra-ket symmetry turns the second factor into sqrt((cd|dc)) -- the same
/// functional form as the first. So ONE vector over ordered pairs suffices,
///
///   Q_P = sqrt(max_components |(P | P_reversed)|),
///
/// and it bounds both slots. This is exactly the construction fock.hpp's
/// schwarz() static_asserts about: (P|P) stops being the pair-density norm as
/// soon as the pairs are complex, so the real routine must not be reused here.
/// Pair ip = a*ns + b, so the reversed pair is (ip % ns)*ns + ip / ns.
template <class Real>
std::vector<Real> giao_schwarz(const PairTable<Kokkos::complex<Real>> &tab,
                               const std::vector<ShellPair<Kokkos::complex<Real>>> &plist,
                               int ns, const TGrid<Real> &grid) {
  using KC = Kokkos::complex<Real>;
  const int npair = static_cast<int>(plist.size());
  std::vector<std::pair<int, int>> qs(npair);
  for (int ip = 0; ip < npair; ++ip) qs[ip] = {ip, (ip % ns) * ns + ip / ns};
  auto batch = make_batch(tab, qs);
  QuartetWorkspace<KC> ws;
  Kokkos::View<KC *> out("intti::gjk::schwarz", batch.nout_total);
  eri_quartets(tab, batch, grid, out, ws);
  auto oh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  std::vector<Real> Q(npair);
  for (int ip = 0; ip < npair; ++ip) {
    const int na = ncart(plist[ip].la), nb = ncart(plist[ip].lb);
    const std::size_t base = batch.h_offset[ip];
    Real qmax = 0;
    for (int ka = 0; ka < na; ++ka)
      for (int kb = 0; kb < nb; ++kb) {
        // the ket is the reversed pair, so this component's partner is (kb, ka)
        const std::size_t idx =
            ((static_cast<std::size_t>(ka) * nb + kb) * nb + kb) * na + ka;
        const Real v = abs_(oh(base + idx));
        if (v > qmax) qmax = v;
      }
    Q[ip] = sqrt_(qmax);
  }
  return Q;
}

/// Device finite-field J/K: the London pairs are built on the host (complex
/// centres), pushed through the batched ERI driver instantiated on
/// Kokkos::complex, and digested on device. Real and imaginary parts are
/// accumulated into separate real Views so only real atomics are needed.
/// `tau` (0 = off) screens the ordered pair x pair quartet loop with the
/// complex Schwarz bound above.
template <class Real>
GiaoJK<Real> giao_jk_dev(const ShellBasis<Real> &basis, const std::complex<Real> *D,
                         const Real B[3], const TGrid<Real> &grid, Real tau) {
  using KC = Kokkos::complex<Real>;
  using C = std::complex<Real>;
  const int ns = static_cast<int>(basis.shells.size());
  const int nao = basis.nao;
  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
  // all ordered shell pairs (the London phases break the 8-fold symmetry)
  std::vector<ShellPair<KC>> plist;
  std::vector<int> hpa, hpb, hoa, hob;
  plist.reserve(static_cast<std::size_t>(ns) * ns);
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      plist.push_back(to_kokkos_pair(make_giao_pair(basis.shells[a], basis.shells[b], B)));
      hpa.push_back(a);
      hpb.push_back(b);
      hoa.push_back(basis.ao_off[a]);
      hob.push_back(basis.ao_off[b]);
    }
  const int npair = static_cast<int>(plist.size());
  auto tab = make_pair_table(plist);
  std::vector<Real> Q;
  if (tau > Real(0)) Q = giao_schwarz(tab, plist, ns, grid);
  // Only the BRA-KET swap survives the London phases, so enumerate ib <= ik and
  // let the digest replay each off-diagonal block in both orientations: half the
  // ERI evaluations, every ordered quartet still counted exactly once.
  std::vector<std::pair<int, int>> quartets;
  quartets.reserve(static_cast<std::size_t>(npair) * (npair + 1) / 2);
  for (int ib = 0; ib < npair; ++ib)
    for (int ik = ib; ik < npair; ++ik) {
      if (tau > Real(0) && Q[ib] * Q[ik] < tau) continue;
      quartets.push_back({ib, ik});
    }
  auto batch = make_batch(tab, quartets);
  QuartetWorkspace<KC> ws;
  Kokkos::View<KC *> out("intti::gjk::out", batch.nout_total);
  eri_quartets(tab, batch, grid, out, ws);

  Kokkos::View<KC *> Dd("intti::gjk::D", n2);
  {
    auto h = Kokkos::create_mirror_view(Dd);
    for (std::size_t i = 0; i < n2; ++i) h(i) = KC(D[i].real(), D[i].imag());
    Kokkos::deep_copy(Dd, h);
  }
  auto lav = tab.la, lbv = tab.lb;
  auto oav = to_device(hoa, "intti::gjk::oa"), obv = to_device(hob, "intti::gjk::ob");
  auto qv = batch.quartets, offv = batch.out_offset;
  Kokkos::View<Real *> Jr("Jr", n2), Ji("Ji", n2), Kr("Kr", n2), Ki("Ki", n2);
  Kokkos::parallel_for(
      "intti::gjk::digest", Kokkos::RangePolicy<>(0, batch.nq), KOKKOS_LAMBDA(int iq) {
        const int ib = qv(iq, 0), ik = qv(iq, 1);
        const int na = ncart(lav(ib)), nb = ncart(lbv(ib));
        const int nc = ncart(lav(ik)), nd = ncart(lbv(ik));
        const int oa = oav(ib), ob = obv(ib), oc = oav(ik), od = obv(ik);
        const std::int64_t base = offv(iq);
        for (int ka = 0; ka < na; ++ka)
          for (int kb = 0; kb < nb; ++kb)
            for (int kc = 0; kc < nc; ++kc)
              for (int kd = 0; kd < nd; ++kd) {
                const KC v = out(base + (((static_cast<std::size_t>(ka) * nb + kb) * nc + kc) * nd + kd));
                // J_mn = sum (mn|ls) D_ls : (m,n)=(a,b), (l,s)=(c,d)
                const KC dJ = v * Dd(static_cast<std::size_t>(oc + kc) * nao + od + kd);
                const std::size_t ij = static_cast<std::size_t>(oa + ka) * nao + ob + kb;
                Kokkos::atomic_add(&Jr(ij), dJ.real());
                Kokkos::atomic_add(&Ji(ij), dJ.imag());
                // K_mn = sum (ml|sn) D_ls : (m,l)=(a,b), (s,n)=(c,d)
                const KC dK = v * Dd(static_cast<std::size_t>(ob + kb) * nao + oc + kc);
                const std::size_t ik2 = static_cast<std::size_t>(oa + ka) * nao + od + kd;
                Kokkos::atomic_add(&Kr(ik2), dK.real());
                Kokkos::atomic_add(&Ki(ik2), dK.imag());
                if (ib == ik) continue;
                // the same block read as (cd|ab): bra-ket swap, value unchanged
                const KC dJ2 = v * Dd(static_cast<std::size_t>(oa + ka) * nao + ob + kb);
                const std::size_t ij2 = static_cast<std::size_t>(oc + kc) * nao + od + kd;
                Kokkos::atomic_add(&Jr(ij2), dJ2.real());
                Kokkos::atomic_add(&Ji(ij2), dJ2.imag());
                const KC dK2 = v * Dd(static_cast<std::size_t>(od + kd) * nao + oa + ka);
                const std::size_t ik3 = static_cast<std::size_t>(oc + kc) * nao + ob + kb;
                Kokkos::atomic_add(&Kr(ik3), dK2.real());
                Kokkos::atomic_add(&Ki(ik3), dK2.imag());
              }
      });
  auto hJr = to_host(Jr), hJi = to_host(Ji), hKr = to_host(Kr), hKi = to_host(Ki);
  GiaoJK<Real> outjk;
  outjk.J.resize(n2);
  outjk.K.resize(n2);
  for (std::size_t i = 0; i < n2; ++i) {
    outjk.J[i] = C(hJr[i], hJi[i]);
    outjk.K[i] = C(hKr[i], hKi[i]);
  }
  return outjk;
}

} // namespace detail

/// `tau` (0 = off) screens the quartet loop with the complex Schwarz bound
/// giao_schwarz(); it applies to the device path, which is the one taken for
/// every scalar Kokkos supports. The generic fallback below is unscreened.
template <class Real>
GiaoJK<Real> giao_jk(const ShellBasis<Real> &basis, const std::complex<Real> *D,
                     const Real B[3], const TGrid<Real> &grid, Real tau = Real(0)) {
  if constexpr (kokkos_scalar_v<Kokkos::complex<Real>>)
    return detail::giao_jk_dev(basis, D, B, grid, tau);
  using C = std::complex<Real>;
  const int ns = static_cast<int>(basis.shells.size());
  const int nao = basis.nao;
  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
  GiaoJK<Real> out;
  out.J.assign(n2, C(0));
  out.K.assign(n2, C(0));
  std::vector<C> blk;
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto bra = make_giao_pair(basis.shells[a], basis.shells[b], B);
      const int na = ncart(basis.shells[a].l), nb = ncart(basis.shells[b].l);
      const int oa = basis.ao_off[a], ob = basis.ao_off[b];
      for (int c = 0; c < ns; ++c)
        for (int d = 0; d < ns; ++d) {
          const auto ket = make_giao_pair(basis.shells[c], basis.shells[d], B);
          const int nc = ncart(basis.shells[c].l), nd = ncart(basis.shells[d].l);
          const int oc = basis.ao_off[c], od = basis.ao_off[d];
          blk.assign(static_cast<std::size_t>(na) * nb * nc * nd, C(0));
          eri_quartet(bra, ket, grid, blk.data());
          for (int ka = 0; ka < na; ++ka)
            for (int kb = 0; kb < nb; ++kb)
              for (int kc = 0; kc < nc; ++kc)
                for (int kd = 0; kd < nd; ++kd) {
                  const C v = blk[((static_cast<std::size_t>(ka) * nb + kb) * nc + kc) * nd + kd];
                  // J_mn = sum (mn|ls) D_ls with (m,n)=(a,b), (l,s)=(c,d)
                  out.J[static_cast<std::size_t>(oa + ka) * nao + ob + kb] +=
                      v * D[static_cast<std::size_t>(oc + kc) * nao + od + kd];
                  // K_mn = sum (ml|sn) D_ls with (m,l)=(a,b), (s,n)=(c,d)
                  out.K[static_cast<std::size_t>(oa + ka) * nao + od + kd] +=
                      v * D[static_cast<std::size_t>(ob + kb) * nao + oc + kc];
                }
        }
    }
  return out;
}

namespace detail {

// device (GPU) GIAO 2e field derivative. At B=0 the ERIs are real, so dJ/dB and
// dK/dB are i times a real combination of real ERI blocks -- computed on the
// real device path here, wrapped in i on the host. Drop-symmetry: each ordered
// quartet is a member directly (no permute); 3 blocks per quartet (base, bra +1,
// ket +1) are batched and the phase-vector digestion runs on device with atomic
// accumulation into the real imaginary-part matrices. l <= LMAX-1.
/// `tau` (0 = off) screens the ordered quartet loop. The digest forms
///   dI = 0.5 (w1 x M1 + w2 x M2),  M1 = (a^{+e} b|cd) + R_a[e] (ab|cd),
/// (M2 the same on the ket), so the bound carries the phase vectors w = R_a-R_b,
/// R_c-R_d and the Schwarz factors of BOTH the base and the promoted pairs:
///   |dI| <= |w1|(Q_{a+1,b} Q_{cd} + |R_a| Q_{ab} Q_{cd})
///         + |w2|(Q_{ab} Q_{c+1,d} + |R_c| Q_{ab} Q_{cd}).
/// Note this bound is gauge-origin dependent through |R_a|, |R_c| (the RESULT is
/// not -- the phase vectors are centre differences), so a distant gauge origin
/// loosens it and screening simply prunes less.
template <class Real>
void giao_jk_dB_dev(const ShellBasis<Real> &basis, const Real *D, const TGrid<Real> &grid,
                    std::array<std::vector<Real>, 3> &Jim,
                    std::array<std::vector<Real>, 3> &Kim, Real tau) {
  const int ns = static_cast<int>(basis.shells.size());
  const int nao = basis.nao;
  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
  std::vector<int> hL(ns), hOff(ns);
  for (int i = 0; i < ns; ++i) {
    hL[i] = basis.shells[i].l;
    hOff[i] = basis.ao_off[i];
  }
  Kokkos::View<Real *[3], Kokkos::LayoutLeft> shC("intti::g2b::C", ns);
  {
    auto h = Kokkos::create_mirror_view(shC);
    for (int i = 0; i < ns; ++i)
      for (int d = 0; d < 3; ++d) h(i, d) = basis.shells[i].center[d];
    Kokkos::deep_copy(shC, h);
  }
  // Distinct pairs only -- the base (i,j) and the bra-promoted (i+1,j) -- memoised
  // on (shell, promotion) x shell. Building them inside the quartet loop instead
  // would put ~5 ns^4 pairs and their E tables in the table where 2 ns^2 are
  // distinct (it even built (a,b) twice per job).
  std::vector<ShellPair<Real>> plist;
  std::vector<int> pid(static_cast<std::size_t>(ns) * 2 * ns, -1);
  auto pair_id = [&](int si, int di, int sj) {
    const std::size_t key = (static_cast<std::size_t>(si) * 2 + di) * ns + sj;
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
    }
  auto tab = make_pair_table(plist);
  std::vector<Real> Q;
  if (tau > Real(0)) Q = schwarz(tab, plist, grid);
  auto maxabs = [](const Real *u, const Real *v) {
    Real m = 0;
    for (int e = 0; e < 3; ++e) {
      const Real x = abs_(v ? u[e] - v[e] : u[e]);
      if (x > m) m = x;
    }
    return m;
  };
  std::vector<std::pair<int, int>> quartets;
  std::vector<int> qa, qb, qc, qd, eBase, eBra, eKet;
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b)
      for (int c = 0; c < ns; ++c)
        for (int d = 0; d < ns; ++d) {
          const int pab = pair_id(a, 0, b), pcd = pair_id(c, 0, d);
          const int pAb = pair_id(a, 1, b), pCd = pair_id(c, 1, d);
          if (tau > Real(0)) {
            const Real qbra = Q[pab], qket = Q[pcd];
            const Real w1 = maxabs(basis.shells[a].center, basis.shells[b].center);
            const Real w2 = maxabs(basis.shells[c].center, basis.shells[d].center);
            const Real ra = maxabs(basis.shells[a].center, nullptr);
            const Real rc = maxabs(basis.shells[c].center, nullptr);
            const Real m1 = Q[pAb] * qket + ra * qbra * qket;
            const Real m2 = qbra * Q[pCd] + rc * qbra * qket;
            if (w1 * m1 + w2 * m2 < tau) continue;
          }
          qa.push_back(a); qb.push_back(b); qc.push_back(c); qd.push_back(d);
          eBase.push_back(static_cast<int>(quartets.size()));
          quartets.push_back({pab, pcd});
          eBra.push_back(static_cast<int>(quartets.size()));
          quartets.push_back({pAb, pcd});
          eKet.push_back(static_cast<int>(quartets.size()));
          quartets.push_back({pab, pCd});
        }
  const int njob = static_cast<int>(qa.size());
  auto batch = make_batch(tab, quartets);
  QuartetWorkspace<Real> ws;
  Kokkos::View<Real *> out("intti::g2b::out", batch.nout_total);
  eri_quartets(tab, batch, grid, out, ws);
  auto shL = to_device(hL, "intti::g2b::L"), shOff = to_device(hOff, "intti::g2b::off");
  auto Dd = to_device(D, n2, "intti::g2b::D");
  auto dqa = to_device(qa, "g2b::qa"), dqb = to_device(qb, "g2b::qb");
  auto dqc = to_device(qc, "g2b::qc"), dqd = to_device(qd, "g2b::qd");
  auto dB = to_device(eBase, "g2b::eB"), dBra = to_device(eBra, "g2b::eBr"), dKet = to_device(eKet, "g2b::eK");
  auto offv = batch.out_offset;
  Kokkos::View<Real *> Jd("intti::g2b::J", 3 * n2), Kd("intti::g2b::K", 3 * n2);
  Kokkos::parallel_for(
      "intti::g2b::digest", Kokkos::RangePolicy<>(0, njob), KOKKOS_LAMBDA(int j) {
        const int a = dqa(j), b = dqb(j), c = dqc(j), d = dqd(j);
        const int la = shL(a), lb = shL(b), lc = shL(c), ld = shL(d);
        const int na = ncart(la), nb = ncart(lb), nc = ncart(lc), nd = ncart(ld);
        const int oa = shOff(a), ob = shOff(b), oc = shOff(c), od = shOff(d);
        const int nbase[4] = {na, nb, nc, nd};
        const int npA[4] = {ncart(la + 1), nb, nc, nd};
        const int npC[4] = {na, nb, ncart(lc + 1), nd};
        const std::int64_t bB = offv(dB(j)), bBra = offv(dBra(j)), bKet = offv(dKet(j));
        const Real w1[3] = {shC(a, 0) - shC(b, 0), shC(a, 1) - shC(b, 1), shC(a, 2) - shC(b, 2)};
        const Real w2[3] = {shC(c, 0) - shC(d, 0), shC(c, 1) - shC(d, 1), shC(c, 2) - shC(d, 2)};
        const Real Ra[3] = {shC(a, 0), shC(a, 1), shC(a, 2)};
        const Real Rc[3] = {shC(c, 0), shC(c, 1), shC(c, 2)};
        auto idx4 = [](const int n[4], int i0, int i1, int i2, int i3) {
          return ((static_cast<std::size_t>(i0) * n[1] + i1) * n[2] + i2) * n[3] + i3;
        };
        for (int ka = 0; ka < na; ++ka) {
          int a3[3];
          cart_comp(la, ka, a3[0], a3[1], a3[2]);
          for (int kb = 0; kb < nb; ++kb)
            for (int kc = 0; kc < nc; ++kc) {
              int c3[3];
              cart_comp(lc, kc, c3[0], c3[1], c3[2]);
              for (int kd = 0; kd < nd; ++kd) {
                const Real b0 = out(bB + idx4(nbase, ka, kb, kc, kd));
                Real M1[3], M2[3];
                for (int e = 0; e < 3; ++e) {
                  int m[3] = {a3[0], a3[1], a3[2]};
                  m[e] += 1;
                  const int ip = comp_index(la + 1, m[0], m[1]);
                  M1[e] = out(bBra + idx4(npA, ip, kb, kc, kd)) + Ra[e] * b0;
                  int mc[3] = {c3[0], c3[1], c3[2]};
                  mc[e] += 1;
                  const int icp = comp_index(lc + 1, mc[0], mc[1]);
                  M2[e] = out(bKet + idx4(npC, ka, kb, icp, kd)) + Rc[e] * b0;
                }
                const Real dI[3] = {
                    Real(0.5) * ((-w1[2] * M1[1] + w1[1] * M1[2]) + (-w2[2] * M2[1] + w2[1] * M2[2])),
                    Real(0.5) * ((w1[2] * M1[0] - w1[0] * M1[2]) + (w2[2] * M2[0] - w2[0] * M2[2])),
                    Real(0.5) * ((-w1[1] * M1[0] + w1[0] * M1[1]) + (-w2[1] * M2[0] + w2[0] * M2[1]))};
                const Real Dcd = Dd(static_cast<std::size_t>(oc + kc) * nao + od + kd);
                const Real Dbd = Dd(static_cast<std::size_t>(ob + kb) * nao + od + kd);
                const std::size_t jjk = static_cast<std::size_t>(oa + ka) * nao + ob + kb;
                const std::size_t kkk = static_cast<std::size_t>(oa + ka) * nao + oc + kc;
                for (int k = 0; k < 3; ++k) {
                  Kokkos::atomic_add(&Jd(k * n2 + jjk), dI[k] * Dcd);
                  Kokkos::atomic_add(&Kd(k * n2 + kkk), dI[k] * Dbd);
                }
              }
            }
        }
      });
  auto hJ = to_host(Jd), hK = to_host(Kd);
  for (int k = 0; k < 3; ++k) {
    Jim[k].assign(hJ.begin() + k * n2, hJ.begin() + (k + 1) * n2);
    Kim[k].assign(hK.begin() + k * n2, hK.begin() + (k + 1) * n2);
  }
}

} // namespace detail

/// dJ/dB and dK/dB at B = 0: three purely imaginary nao x nao matrices each,
/// in the order {x, y, z}. D is the nao x nao AO density (row-major); the J/K
/// definitions are J_mn = sum (mn|ls) D_ls, K_mn = sum (ml|ns) D_ls.
template <class Real> struct GiaoJKderiv {
  std::array<std::vector<std::complex<Real>>, 3> dJ, dK;
};

/// `tau` (0 = off) screens the device path's quartet loop; see
/// giao_jk_dB_dev for the bound. The generic host fallback is unscreened.
template <class Real>
GiaoJKderiv<Real> giao_jk_dB(const ShellBasis<Real> &basis, const Real *D,
                             const TGrid<Real> &grid, Real tau = Real(0)) {
  using C = std::complex<Real>;
  const int ns = static_cast<int>(basis.shells.size());
  const int nao = basis.nao;
  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
  // accumulate the imaginary parts as real matrices, wrap in i at the end
  std::array<std::vector<Real>, 3> Jim, Kim;
  for (int k = 0; k < 3; ++k) {
    Jim[k].assign(n2, Real(0));
    Kim[k].assign(n2, Real(0));
  }
  if constexpr (kokkos_scalar_v<Real>) {
    bool ok = true;
    for (const auto &s : basis.shells)
      if (s.l >= LMAX) ok = false; // bra/ket promoted by one
    if (ok) {
      detail::giao_jk_dB_dev(basis, D, grid, Jim, Kim, tau);
      GiaoJKderiv<Real> out;
      for (int k = 0; k < 3; ++k) {
        out.dJ[k].assign(n2, C(0));
        out.dK[k].assign(n2, C(0));
        for (std::size_t i = 0; i < n2; ++i) {
          out.dJ[k][i] = C(Real(0), Jim[k][i]);
          out.dK[k][i] = C(Real(0), Kim[k][i]);
        }
      }
      return out;
    }
  }
  auto Dm = [&](int i, int j) { return D[static_cast<std::size_t>(i) * nao + j]; };
  auto idx4 = [](const int n[4], int i0, int i1, int i2, int i3) {
    return ((static_cast<std::size_t>(i0) * n[1] + i1) * n[2] + i2) * n[3] + i3;
  };

  // contract one (member) shell quartet: the original, unmodified GIAO body.
  // base/plusBra/plusKet are the member-layout base, bra-promoted (slot 0) and
  // ket-promoted (slot 2) ERI blocks.
  auto contribute = [&](const int mem[4], const int L[4], const int off[4],
                        const std::vector<Real> &base, const std::vector<Real> &plusBra,
                        const std::vector<Real> &plusKet) {
    const int la = L[0], lb = L[1], lc = L[2], ld = L[3];
    const int na = ncart(la), nb = ncart(lb), nc = ncart(lc), nd = ncart(ld);
    const int oa = off[0], ob = off[1], oc = off[2], od = off[3];
    const int nbase[4] = {na, nb, nc, nd};
    const int npA[4] = {ncart(la + 1), nb, nc, nd};
    const int npC[4] = {na, nb, ncart(lc + 1), nd};
    const Real *Ra = basis.shells[mem[0]].center, *Rb = basis.shells[mem[1]].center;
    const Real *Rc = basis.shells[mem[2]].center, *Rd = basis.shells[mem[3]].center;
    const Real w1[3] = {Ra[0] - Rb[0], Ra[1] - Rb[1], Ra[2] - Rb[2]};
    const Real w2[3] = {Rc[0] - Rd[0], Rc[1] - Rd[1], Rc[2] - Rd[2]};
    for (int ka = 0; ka < na; ++ka) {
      int a3[3];
      cart_comp(la, ka, a3[0], a3[1], a3[2]);
      for (int kb = 0; kb < nb; ++kb)
        for (int kc = 0; kc < nc; ++kc) {
          int c3[3];
          cart_comp(lc, kc, c3[0], c3[1], c3[2]);
          for (int kd = 0; kd < nd; ++kd) {
            const Real b0 = base[idx4(nbase, ka, kb, kc, kd)];
            Real M1[3], M2[3];
            for (int e = 0; e < 3; ++e) {
              int m[3] = {a3[0], a3[1], a3[2]};
              m[e] += 1;
              const int ip = detail::comp_index(la + 1, m[0], m[1]);
              M1[e] = plusBra[idx4(npA, ip, kb, kc, kd)] + Ra[e] * b0;
              int mc[3] = {c3[0], c3[1], c3[2]};
              mc[e] += 1;
              const int icp = detail::comp_index(lc + 1, mc[0], mc[1]);
              M2[e] = plusKet[idx4(npC, ka, kb, icp, kd)] + Rc[e] * b0;
            }
            const Real dI[3] = {
                Real(0.5) * ((-w1[2] * M1[1] + w1[1] * M1[2]) +
                             (-w2[2] * M2[1] + w2[1] * M2[2])),
                Real(0.5) * ((w1[2] * M1[0] - w1[0] * M1[2]) +
                             (w2[2] * M2[0] - w2[0] * M2[2])),
                Real(0.5) * ((-w1[1] * M1[0] + w1[0] * M1[1]) +
                             (-w2[1] * M2[0] + w2[0] * M2[1]))};
            const Real Dcd = Dm(oc + kc, od + kd);
            const Real Dbd = Dm(ob + kb, od + kd);
            const std::size_t jjk = (oa + ka) * static_cast<std::size_t>(nao) + ob + kb;
            const std::size_t kkk = (oa + ka) * static_cast<std::size_t>(nao) + oc + kc;
            for (int k = 0; k < 3; ++k) {
              Jim[k][jjk] += dI[k] * Dcd;
              Kim[k][kkk] += dI[k] * Dbd;
            }
          }
        }
    }
  };

  // canonical shell quartets: a>=b, c>=d, pair(a,b) >= pair(c,d)
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b <= a; ++b) {
      const int Pab = a * ns + b;
      for (int cc = 0; cc < ns; ++cc)
        for (int dd = 0; dd <= cc; ++dd) {
          if (cc * ns + dd > Pab) continue;
          const int canon[4] = {a, b, cc, dd};
          const int Lc[4] = {basis.shells[a].l, basis.shells[b].l, basis.shells[cc].l,
                             basis.shells[dd].l};
          const int dC[4] = {ncart(Lc[0]), ncart(Lc[1]), ncart(Lc[2]), ncart(Lc[3])};

          // canonical base and single (l+1) promotions of each slot
          auto cbase = detail::eri_block4(basis.shells[a], basis.shells[b], basis.shells[cc],
                                          basis.shells[dd], grid);
          std::vector<Real> cplus[4];
          for (int p = 0; p < 4; ++p) {
            auto mk = [&](int slot) {
              auto s = basis.shells[canon[slot]];
              if (slot == p) s.l += 1;
              return s;
            };
            cplus[p] = detail::eri_block4(mk(0), mk(1), mk(2), mk(3), grid);
          }

          int seen[8][4];
          int nseen = 0;
          for (int g = 0; g < 8; ++g) {
            const int *pm = detail::eri_perms[g];
            const int mem[4] = {canon[pm[0]], canon[pm[1]], canon[pm[2]], canon[pm[3]]};
            bool dup = false;
            for (int t = 0; t < nseen && !dup; ++t)
              dup = seen[t][0] == mem[0] && seen[t][1] == mem[1] && seen[t][2] == mem[2] &&
                    seen[t][3] == mem[3];
            if (dup) continue;
            for (int t = 0; t < 4; ++t) seen[nseen][t] = mem[t];
            ++nseen;

            const int Lm[4] = {Lc[pm[0]], Lc[pm[1]], Lc[pm[2]], Lc[pm[3]]};
            const int offm[4] = {basis.ao_off[mem[0]], basis.ao_off[mem[1]],
                                 basis.ao_off[mem[2]], basis.ao_off[mem[3]]};
            auto mbase = detail::permute_block(cbase, dC, pm);
            // member bra-promotion = canonical slot pm[0] promoted; ket = pm[2]
            int dBra[4], dKet[4];
            for (int i = 0; i < 4; ++i) dBra[i] = dKet[i] = dC[i];
            dBra[pm[0]] = ncart(Lc[pm[0]] + 1);
            dKet[pm[2]] = ncart(Lc[pm[2]] + 1);
            auto mBra = detail::permute_block(cplus[pm[0]], dBra, pm);
            auto mKet = detail::permute_block(cplus[pm[2]], dKet, pm);
            contribute(mem, Lm, offm, mbase, mBra, mKet);
          }
        }
    }
  GiaoJKderiv<Real> out;
  for (int k = 0; k < 3; ++k) {
    out.dJ[k].assign(n2, C(0));
    out.dK[k].assign(n2, C(0));
    for (std::size_t i = 0; i < n2; ++i) {
      out.dJ[k][i] = C(Real(0), Jim[k][i]);
      out.dK[k][i] = C(Real(0), Kim[k][i]);
    }
  }
  return out;
}

} // namespace intti
