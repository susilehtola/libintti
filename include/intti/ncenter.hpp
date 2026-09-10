// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Two- and three-center Coulomb integrals -- the RI (density-fitting) core.
// An auxiliary function P is treated as a shell paired with a zero-exponent
// unit-s "ghost" at the same centre (beta=0 => p=alpha, K=1: the libcint
// int2c2e/int3c2e trick), so both reduce to the ordinary quartet engine:
//   (P|Q)   = quartet( (P,ghost), (Q,ghost) )
//   (mu nu|P) = quartet( (mu,nu),  (P,ghost) ).
// Matrix/tensor-level API only: whole (P|Q) matrix and (mu nu|P) tensor.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "batch.hpp"      // PairTable, make_batch, eri_quartets (device driver)
#include "contracted.hpp" // ContractedBasis, detail::effective_coeff/contracted_prim
#include "device.hpp"     // detail::to_device / to_host
#include "fock.hpp"
#include "gto.hpp"
#include "lkc.hpp"
#include "math.hpp"
#include "multipole.hpp"
#include "quartet.hpp"
#include "tgrid.hpp"

namespace intti {

namespace detail {
/// A shell paired with a zero-exponent unit-s ghost at the same centre.
template <class Real>
ShellPair<Real> ghost_pair(const PrimitiveShell<Real> &s) {
  PrimitiveShell<Real> ghost{Real(0), {s.center[0], s.center[1], s.center[2]}, 0};
  return make_pair(s, ghost);
}

/// True if bra and ket pairs are well separated for the multipole far-field:
/// alpha_pq |P_bra - P_ket|^2 > -ln(far_tau), alpha_pq = p_p p_q/(p_p+p_q).
template <class Real>
bool pair_far(const ShellPair<Real> &bra, const ShellPair<Real> &ket,
              Real far_cut) {
  Real R2 = 0;
  for (int d = 0; d < 3; ++d) {
    const Real x = bra.P[d] - ket.P[d];
    R2 += x * x;
  }
  return bra.p * ket.p / (bra.p + ket.p) * R2 > far_cut;
}

// ---- device (GPU) n-center paths --------------------------------------------
// Instead of looping the single-quartet debug routine eri_quartet, enumerate all
// ghost-shell quartets and evaluate them in ONE batched device call
// (eri_quartets), then scatter the concatenated blocks into the tensor with a
// Kokkos parallel_for. Same ghost trick: (P|Q) = quartet((P,ghost),(Q,ghost)),
// (mu nu|P) = quartet((mu,nu),(P,ghost)).

/// Device (P|Q) metric over a primitive auxiliary basis.
template <class Real>
std::vector<Real> coulomb_2c_dev(const ShellBasis<Real> &aux, const TGrid<Real> &grid) {
  const int naux = aux.nao;
  const int ns = static_cast<int>(aux.shells.size());
  std::vector<ShellPair<Real>> plist(ns);
  std::vector<int> hoff(ns);
  for (int a = 0; a < ns; ++a) {
    plist[a] = ghost_pair(aux.shells[a]); // pair index == aux shell index
    hoff[a] = aux.ao_off[a];
  }
  auto tab = make_pair_table(plist);
  std::vector<std::pair<int, int>> quartets;
  quartets.reserve(static_cast<std::size_t>(ns) * (ns + 1) / 2);
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b <= a; ++b) quartets.push_back({a, b});
  sort_by_class(tab, quartets);
  auto batch = make_batch(tab, quartets);
  QuartetWorkspace<Real> ws;
  Kokkos::View<Real *> out("intti::c2c::out", batch.nout_total);
  eri_quartets(tab, batch, grid, out, ws);
  auto auxoff = to_device(hoff, "intti::c2c::auxoff");
  Kokkos::View<Real *> Md("intti::c2c::M", static_cast<std::size_t>(naux) * naux);
  auto qv = batch.quartets, offv = batch.out_offset;
  auto lav = tab.la;
  Kokkos::parallel_for(
      "intti::c2c::scatter", Kokkos::RangePolicy<>(0, batch.nq), KOKKOS_LAMBDA(int iq) {
        const int a = qv(iq, 0), b = qv(iq, 1);
        const int nP = ncart(lav(a)), nQ = ncart(lav(b));
        const std::int64_t off = offv(iq);
        const int oa = auxoff(a), ob = auxoff(b);
        for (int kP = 0; kP < nP; ++kP)
          for (int kQ = 0; kQ < nQ; ++kQ) {
            const Real v = out(off + kP * nQ + kQ);
            Md(static_cast<std::size_t>(oa + kP) * naux + ob + kQ) = v;
            Md(static_cast<std::size_t>(ob + kQ) * naux + oa + kP) = v;
          }
      });
  return to_host(Md);
}

/// Device (mu nu | P) tensor over primitive orbital + auxiliary bases (exact).
template <class Real>
std::vector<Real> coulomb_3c_dev(const ShellBasis<Real> &orb,
                                 const ShellBasis<Real> &aux, const TGrid<Real> &grid) {
  const int nao = orb.nao, naux = aux.nao;
  const int nso = static_cast<int>(orb.shells.size());
  const int nsa = static_cast<int>(aux.shells.size());
  // pair table: orbital shell pairs (m>=n) then aux ghost pairs, concatenated.
  std::vector<ShellPair<Real>> plist;
  std::vector<int> braM, braN;                     // orbital pair -> (m,n) shells
  for (int m = 0; m < nso; ++m)
    for (int n = 0; n <= m; ++n) {
      plist.push_back(make_pair(orb.shells[m], orb.shells[n]));
      braM.push_back(m);
      braN.push_back(n);
    }
  const int nbra = static_cast<int>(plist.size());
  for (int a = 0; a < nsa; ++a) plist.push_back(ghost_pair(aux.shells[a]));
  auto tab = make_pair_table(plist);
  std::vector<std::pair<int, int>> quartets;
  quartets.reserve(static_cast<std::size_t>(nbra) * nsa);
  for (int ib = 0; ib < nbra; ++ib)
    for (int a = 0; a < nsa; ++a) quartets.push_back({ib, nbra + a});
  sort_by_class(tab, quartets);
  auto batch = make_batch(tab, quartets);
  QuartetWorkspace<Real> ws;
  Kokkos::View<Real *> out("intti::c3c::out", batch.nout_total);
  eri_quartets(tab, batch, grid, out, ws);
  // side arrays indexed by pair id
  std::vector<int> hmoff(nbra), hnoff(nbra);
  for (int ib = 0; ib < nbra; ++ib) {
    hmoff[ib] = orb.ao_off[braM[ib]];
    hnoff[ib] = orb.ao_off[braN[ib]];
  }
  std::vector<int> haoff(nsa);
  for (int a = 0; a < nsa; ++a) haoff[a] = aux.ao_off[a];
  auto moff = to_device(hmoff, "intti::c3c::moff");
  auto noff = to_device(hnoff, "intti::c3c::noff");
  auto auxoff = to_device(haoff, "intti::c3c::auxoff");
  Kokkos::View<Real *> Td("intti::c3c::T", static_cast<std::size_t>(nao) * nao * naux);
  auto qv = batch.quartets, offv = batch.out_offset;
  auto lav = tab.la, lbv = tab.lb;
  Kokkos::parallel_for(
      "intti::c3c::scatter", Kokkos::RangePolicy<>(0, batch.nq), KOKKOS_LAMBDA(int iq) {
        const int ib = qv(iq, 0), ik = qv(iq, 1);      // bra pair id, ket (ghost) pair id
        const int aidx = ik - nbra;                    // aux shell = ket pair id - nbra
        const int nm = ncart(lav(ib)), nn = ncart(lbv(ib)), nP = ncart(lav(ik));
        const std::int64_t off = offv(iq);
        const int om = moff(ib), on = noff(ib), oP = auxoff(aidx);
        const bool offdiag = (om != on); // distinct orbital shells -> mu<->nu mirror
        for (int km = 0; km < nm; ++km)
          for (int kn = 0; kn < nn; ++kn)
            for (int kP = 0; kP < nP; ++kP) {
              const Real v = out(off + (km * nn + kn) * nP + kP);
              const std::size_t P = static_cast<std::size_t>(oP) + kP;
              Td(((static_cast<std::size_t>(om + km) * nao + on + kn) * naux) + P) = v;
              if (offdiag)
                Td(((static_cast<std::size_t>(on + kn) * nao + om + km) * naux) + P) = v;
            }
      });
  return to_host(Td);
}

// contracted-basis device n-center: batch every primitive ghost-quartet once and
// accumulate it coefficient-weighted (effective_coeff) into the contracted
// blocks on device with atomics -- the shared-primitive contraction.

/// Flat effective-coefficient table for a contracted basis: eff(shell,c,p) at
/// eoff[shell] + c*nprim(shell) + p, plus per-shell l/nprim/nctr/ao_off arrays.
template <class Real> struct ContractedDev {
  Kokkos::View<Real *> eff;
  Kokkos::View<int *> eoff, L, nprim, nctr, aoff;
};
template <class Real>
ContractedDev<Real> to_contracted_dev(const ContractedBasis<Real> &b, const char *tag) {
  const int ns = static_cast<int>(b.shells.size());
  std::vector<int> L(ns), np(ns), nc(ns), ao(ns), eo(ns);
  std::vector<Real> eff;
  int tot = 0;
  for (int s = 0; s < ns; ++s) {
    L[s] = b.shells[s].l;
    np[s] = b.shells[s].nprim();
    nc[s] = b.shells[s].nctr();
    ao[s] = b.ao_off[s];
    eo[s] = tot;
    tot += np[s] * nc[s];
  }
  eff.resize(tot);
  for (int s = 0; s < ns; ++s)
    for (int c = 0; c < nc[s]; ++c)
      for (int p = 0; p < np[s]; ++p)
        eff[eo[s] + c * np[s] + p] = effective_coeff(b.shells[s], c, p);
  ContractedDev<Real> dv;
  dv.eff = to_device(eff, tag);
  dv.eoff = to_device(eo, "intti::cdev::eoff");
  dv.L = to_device(L, "intti::cdev::L");
  dv.nprim = to_device(np, "intti::cdev::np");
  dv.nctr = to_device(nc, "intti::cdev::nc");
  dv.aoff = to_device(ao, "intti::cdev::ao");
  return dv;
}

template <class Real>
std::vector<Real> coulomb_2c_dev(const ContractedBasis<Real> &aux, const TGrid<Real> &grid) {
  const int naux = aux.nao, ns = static_cast<int>(aux.shells.size());
  auto dv = to_contracted_dev(aux, "intti::c2cc::eff");
  // primitive ghost-quartets for canonical shell pairs A>=B, all (pa,pb)
  std::vector<ShellPair<Real>> plist;
  std::vector<std::pair<int, int>> quartets;
  std::vector<int> qA, qB, qpa, qpb;
  for (int A = 0; A < ns; ++A)
    for (int B = 0; B <= A; ++B)
      for (int pa = 0; pa < aux.shells[A].nprim(); ++pa)
        for (int pb = 0; pb < aux.shells[B].nprim(); ++pb) {
          const int ib = static_cast<int>(plist.size());
          plist.push_back(ghost_pair(contracted_prim(aux.shells[A], pa)));
          const int ik = static_cast<int>(plist.size());
          plist.push_back(ghost_pair(contracted_prim(aux.shells[B], pb)));
          quartets.push_back({ib, ik});
          qA.push_back(A); qB.push_back(B); qpa.push_back(pa); qpb.push_back(pb);
        }
  auto tab = make_pair_table(plist);
  auto batch = make_batch(tab, quartets);
  QuartetWorkspace<Real> ws;
  Kokkos::View<Real *> out("intti::c2cc::out", batch.nout_total);
  eri_quartets(tab, batch, grid, out, ws);
  auto dqA = to_device(qA, "intti::c2cc::A"), dqB = to_device(qB, "intti::c2cc::B");
  auto dqpa = to_device(qpa, "intti::c2cc::pa"), dqpb = to_device(qpb, "intti::c2cc::pb");
  auto offv = batch.out_offset;
  auto eff = dv.eff, Lv = dv.L, npv = dv.nprim, ncv = dv.nctr, aov = dv.aoff, eov = dv.eoff;
  Kokkos::View<Real *> Md("intti::c2cc::M", static_cast<std::size_t>(naux) * naux);
  Kokkos::parallel_for(
      "intti::c2cc::scatter", Kokkos::RangePolicy<>(0, batch.nq), KOKKOS_LAMBDA(int iq) {
        const int A = dqA(iq), B = dqB(iq), pa = dqpa(iq), pb = dqpb(iq);
        const int lA = Lv(A), lB = Lv(B), nP = ncart(lA), nQ = ncart(lB);
        const std::int64_t base = offv(iq);
        for (int cA = 0; cA < ncv(A); ++cA) {
          const Real wa = eff(eov(A) + cA * npv(A) + pa);
          if (wa == Real(0)) continue;
          for (int cB = 0; cB < ncv(B); ++cB) {
            const Real w = wa * eff(eov(B) + cB * npv(B) + pb);
            if (w == Real(0)) continue;
            // The mirror is ONLY for the strictly-lower shell blocks. Within a
            // DIAGONAL shell block (A == B) the cA/kP and cB/kQ loops already
            // run over the full block, producing both (I,J) and (J,I)
            // directly, so mirroring there adds every off-diagonal element
            // twice. The primitive builder above gets away with mirroring
            // unconditionally because it ASSIGNS, which is idempotent; this one
            // must accumulate, to sum over the primitives of a contraction.
            //
            // s and p hide the bug: a same-centre two-centre integral vanishes
            // unless both components have even parity in every direction, so
            // the only nonzero off-diagonals in a diagonal block first appear
            // at l = 2, as the (xx|yy)-type trace pairs.
            for (int kP = 0; kP < nP; ++kP)
              for (int kQ = 0; kQ < nQ; ++kQ) {
                const std::size_t I = aov(A) + static_cast<std::size_t>(cA) * nP + kP;
                const std::size_t J = aov(B) + static_cast<std::size_t>(cB) * nQ + kQ;
                const Real v = w * out(base + kP * nQ + kQ);
                Kokkos::atomic_add(&Md(I * naux + J), v);
                if (A != B) Kokkos::atomic_add(&Md(J * naux + I), v);
              }
          }
        }
      });
  return to_host(Md);
}

/// [sa0, sa1) restricts the AUXILIARY shells, and the output auxiliary index is
/// LOCAL to that range, so the RI pipeline can stream one auxiliary block at a
/// time rather than materialising nao^2 x naux. Same role as the primitive
/// coulomb_3c_auxblock; the tiling has to reach the device path because that is
/// what actually runs for float and double.
template <class Real>
std::vector<Real> coulomb_3c_dev(const ContractedBasis<Real> &orb,
                                 const ContractedBasis<Real> &aux, const TGrid<Real> &grid,
                                 int sa0, int sa1) {
  const int nao = orb.nao;
  const int nso = static_cast<int>(orb.shells.size());
  const int p0 = aux.ao_off[sa0], naux = aux.ao_off[sa1] - p0;
  auto ov = to_contracted_dev(orb, "intti::c3cc::oeff");
  auto av = to_contracted_dev(aux, "intti::c3cc::aeff");
  std::vector<ShellPair<Real>> plist;
  std::vector<std::pair<int, int>> quartets;
  std::vector<int> qM, qN, qA, qpm, qpn, qpa;
  for (int M = 0; M < nso; ++M)
    for (int N = 0; N <= M; ++N)
      for (int A = sa0; A < sa1; ++A)
        for (int pm = 0; pm < orb.shells[M].nprim(); ++pm)
          for (int pn = 0; pn < orb.shells[N].nprim(); ++pn)
            for (int pa = 0; pa < aux.shells[A].nprim(); ++pa) {
              const int ib = static_cast<int>(plist.size());
              plist.push_back(make_pair(contracted_prim(orb.shells[M], pm),
                                        contracted_prim(orb.shells[N], pn)));
              const int ik = static_cast<int>(plist.size());
              plist.push_back(ghost_pair(contracted_prim(aux.shells[A], pa)));
              quartets.push_back({ib, ik});
              qM.push_back(M); qN.push_back(N); qA.push_back(A);
              qpm.push_back(pm); qpn.push_back(pn); qpa.push_back(pa);
            }
  auto tab = make_pair_table(plist);
  auto batch = make_batch(tab, quartets);
  QuartetWorkspace<Real> ws;
  Kokkos::View<Real *> out("intti::c3cc::out", batch.nout_total);
  eri_quartets(tab, batch, grid, out, ws);
  auto dM = to_device(qM, "intti::c3cc::M"), dN = to_device(qN, "intti::c3cc::N");
  auto dA = to_device(qA, "intti::c3cc::A"), dpm = to_device(qpm, "intti::c3cc::pm");
  auto dpn = to_device(qpn, "intti::c3cc::pn"), dpa = to_device(qpa, "intti::c3cc::pa");
  auto offv = batch.out_offset;
  auto oeff = ov.eff, oL = ov.L, onp = ov.nprim, onc = ov.nctr, oao = ov.aoff, oeo = ov.eoff;
  auto aeff = av.eff, aL = av.L, anp = av.nprim, anc = av.nctr, aao = av.aoff, aeo = av.eoff;
  Kokkos::View<Real *> Td("intti::c3cc::T", static_cast<std::size_t>(nao) * nao * naux);
  const std::size_t pbase = static_cast<std::size_t>(p0);
  Kokkos::parallel_for(
      "intti::c3cc::scatter", Kokkos::RangePolicy<>(0, batch.nq), KOKKOS_LAMBDA(int iq) {
        const int M = dM(iq), N = dN(iq), A = dA(iq), pm = dpm(iq), pn = dpn(iq), pa = dpa(iq);
        const int nm = ncart(oL(M)), nn = ncart(oL(N)), nP = ncart(aL(A));
        const std::int64_t base = offv(iq);
        const bool offdiag = (M != N);
        for (int cM = 0; cM < onc(M); ++cM) {
          const Real wm = oeff(oeo(M) + cM * onp(M) + pm);
          if (wm == Real(0)) continue;
          for (int cN = 0; cN < onc(N); ++cN) {
            const Real wmn = wm * oeff(oeo(N) + cN * onp(N) + pn);
            if (wmn == Real(0)) continue;
            for (int cA = 0; cA < anc(A); ++cA) {
              const Real w = wmn * aeff(aeo(A) + cA * anp(A) + pa);
              if (w == Real(0)) continue;
              for (int km = 0; km < nm; ++km)
                for (int kn = 0; kn < nn; ++kn)
                  for (int kP = 0; kP < nP; ++kP) {
                    const std::size_t I = oao(M) + static_cast<std::size_t>(cM) * nm + km;
                    const std::size_t Jn = oao(N) + static_cast<std::size_t>(cN) * nn + kn;
                    // auxiliary index is LOCAL to the tile
                    const std::size_t P =
                        aao(A) - pbase + static_cast<std::size_t>(cA) * nP + kP;
                    const Real v = w * out(base + (km * nn + kn) * nP + kP);
                    Kokkos::atomic_add(&Td((I * nao + Jn) * naux + P), v);
                    if (offdiag) Kokkos::atomic_add(&Td((Jn * nao + I) * naux + P), v);
                  }
            }
          }
        }
      });
  return to_host(Td);
}

} // namespace detail

/// Two-center Coulomb metric (P|Q) over an auxiliary basis: naux x naux,
/// row-major, primitive Cartesian, unnormalized.
template <class Real>
std::vector<Real> coulomb_2c(const ShellBasis<Real> &aux, const TGrid<Real> &grid) {
  if constexpr (kokkos_scalar_v<Real>)
    return detail::coulomb_2c_dev(aux, grid);
  const int naux = aux.nao;
  std::vector<Real> M(static_cast<std::size_t>(naux) * naux, Real(0));
  const int ns = static_cast<int>(aux.shells.size());
  // (P|Q) = (Q|P): compute the a >= b triangle once and mirror the transpose.
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b <= a; ++b) {
      auto bra = detail::ghost_pair(aux.shells[a]);
      auto ket = detail::ghost_pair(aux.shells[b]);
      const int nP = ncart(aux.shells[a].l), nQ = ncart(aux.shells[b].l);
      std::vector<Real> blk(static_cast<std::size_t>(nP) * nQ);
      eri_quartet(bra, ket, grid, blk.data());
      // LKC consumer: symmetric block scatter of the precomputed (P|Q) block.
      detail::scatter_pair(M, aux, a, b, +1,
          [&](int kP, const int *, int kQ, const int *) { return blk[kP * nQ + kQ]; });
    }
  return M;
}

/// Three-center Coulomb (mu nu | P): row-major (mu, nu, P) tensor of size
/// nao*nao*naux, primitive Cartesian, unnormalized.
/// Optional far-field (multipole) acceleration: far_tau > 0 evaluates an
/// orbital pair well separated from an auxiliary function (alpha |P-R_aux|^2 >
/// -ln(far_tau)) through the exponent-free multipole tensor instead of the
/// t-quadrature, with relative error ~ far_tau. far_tau = 0 (default) is the
/// exact build.
template <class Real>
std::vector<Real> coulomb_3c(const ShellBasis<Real> &orb,
                             const ShellBasis<Real> &aux, const TGrid<Real> &grid,
                             Real far_tau = Real(0)) {
  if constexpr (kokkos_scalar_v<Real>)
    if (far_tau == Real(0)) return detail::coulomb_3c_dev(orb, aux, grid);
  const int nao = orb.nao, naux = aux.nao;
  std::vector<Real> T(static_cast<std::size_t>(nao) * nao * naux, Real(0));
  const int nso = static_cast<int>(orb.shells.size());
  const int nsa = static_cast<int>(aux.shells.size());
  const bool far = far_tau > Real(0);
  const Real far_cut = far ? -log_(far_tau) : Real(0);
  // (mu nu | P) = (nu mu | P): compute the m >= n triangle once and mirror.
  for (int m = 0; m < nso; ++m)
    for (int n = 0; n <= m; ++n) {
      auto bra = make_pair(orb.shells[m], orb.shells[n]);
      const int nm = ncart(orb.shells[m].l), nn = ncart(orb.shells[n].l);
      for (int a = 0; a < nsa; ++a) {
        auto ket = detail::ghost_pair(aux.shells[a]);
        const int nP = ncart(aux.shells[a].l);
        std::vector<Real> blk(static_cast<std::size_t>(nm) * nn * nP);
        if (far && detail::pair_far(bra, ket, far_cut))
          eri_quartet_farfield(bra, ket, blk.data());
        else
          eri_quartet(bra, ket, grid, blk.data());
        for (int km = 0; km < nm; ++km)
          for (int kn = 0; kn < nn; ++kn)
            for (int kP = 0; kP < nP; ++kP) {
              const Real v = blk[(km * nn + kn) * nP + kP];
              T[((static_cast<std::size_t>(orb.ao_off[m] + km) * nao +
                  orb.ao_off[n] + kn) *
                 naux) +
                aux.ao_off[a] + kP] = v;
              if (m != n)
                T[((static_cast<std::size_t>(orb.ao_off[n] + kn) * nao +
                    orb.ao_off[m] + km) *
                   naux) +
                  aux.ao_off[a] + kP] = v; // mu<->nu mirror
            }
      }
    }
  return T;
}

/// Three-center Coulomb (mu nu | P) for the auxiliary SHELLS [sa0, sa1) only:
/// a row-major (mu, nu, Plocal) tensor of size nao*nao*nauxblk, where nauxblk is
/// the number of auxiliary AOs in the range. Lets the RI pipeline stream one
/// auxiliary block at a time instead of materialising the whole nao^2 x naux
/// tensor (memory-lean / out-of-core RI). far_tau as in coulomb_3c.
template <class Real>
std::vector<Real> coulomb_3c_auxblock(const ShellBasis<Real> &orb,
                                      const ShellBasis<Real> &aux, const TGrid<Real> &grid,
                                      int sa0, int sa1, Real far_tau = Real(0)) {
  const int nao = orb.nao;
  const int p0 = aux.ao_off[sa0], p1 = aux.ao_off[sa1];
  const int nauxblk = p1 - p0;
  std::vector<Real> T(static_cast<std::size_t>(nao) * nao * nauxblk, Real(0));
  const int nso = static_cast<int>(orb.shells.size());
  const bool far = far_tau > Real(0);
  const Real far_cut = far ? -log_(far_tau) : Real(0);
  for (int m = 0; m < nso; ++m)
    for (int n = 0; n <= m; ++n) {
      auto bra = make_pair(orb.shells[m], orb.shells[n]);
      const int nm = ncart(orb.shells[m].l), nn = ncart(orb.shells[n].l);
      for (int a = sa0; a < sa1; ++a) {
        auto ket = detail::ghost_pair(aux.shells[a]);
        const int nP = ncart(aux.shells[a].l);
        std::vector<Real> blk(static_cast<std::size_t>(nm) * nn * nP);
        if (far && detail::pair_far(bra, ket, far_cut))
          eri_quartet_farfield(bra, ket, blk.data());
        else
          eri_quartet(bra, ket, grid, blk.data());
        for (int km = 0; km < nm; ++km)
          for (int kn = 0; kn < nn; ++kn)
            for (int kP = 0; kP < nP; ++kP) {
              const Real v = blk[(km * nn + kn) * nP + kP];
              const int Ploc = aux.ao_off[a] - p0 + kP;
              T[(static_cast<std::size_t>(orb.ao_off[m] + km) * nao + orb.ao_off[n] + kn) *
                    nauxblk +
                Ploc] = v;
              if (m != n)
                T[(static_cast<std::size_t>(orb.ao_off[n] + kn) * nao + orb.ao_off[m] + km) *
                      nauxblk +
                  Ploc] = v; // mu<->nu mirror
            }
      }
    }
  return T;
}

// ---- generally-contracted RI n-center tensors -------------------------------
// The contracted (P|Q) and (mu nu | P) reuse the primitive ghost-shell
// eri_quartet on every primitive combination, accumulating it coefficient-
// weighted (effective_coeff = basis-set coeff * cart_norm_pyscf) into the
// contracted block -- each primitive quartet evaluated once, shared across the
// contraction indices. AO order c*ncart(l)+k (matching contracted.hpp).

/// Two-center Coulomb metric (P|Q) over a generally-contracted auxiliary basis
/// (naux x naux, PySCF cart=True normalization).
template <class Real>
std::vector<Real> coulomb_2c(const ContractedBasis<Real> &aux, const TGrid<Real> &grid) {
  if constexpr (kokkos_scalar_v<Real>)
    return detail::coulomb_2c_dev(aux, grid);
  const int naux = aux.nao;
  std::vector<Real> M(static_cast<std::size_t>(naux) * naux, Real(0));
  const int ns = static_cast<int>(aux.shells.size());
  for (int A = 0; A < ns; ++A)
    for (int B = 0; B <= A; ++B) {
      const auto &SA = aux.shells[A], &SB = aux.shells[B];
      const int nP = ncart(SA.l), nQ = ncart(SB.l), nctA = SA.nctr(), nctB = SB.nctr();
      const int rowB = nctB * nQ;
      std::vector<Real> cblk(static_cast<std::size_t>(nctA) * nP * rowB, Real(0));
      std::vector<Real> blk(static_cast<std::size_t>(nP) * nQ);
      for (int pa = 0; pa < SA.nprim(); ++pa) {
        auto bra = detail::ghost_pair(detail::contracted_prim(SA, pa));
        for (int pb = 0; pb < SB.nprim(); ++pb) {
          auto ket = detail::ghost_pair(detail::contracted_prim(SB, pb));
          eri_quartet(bra, ket, grid, blk.data());
          for (int cA = 0; cA < nctA; ++cA) {
            const Real wa = detail::effective_coeff(SA, cA, pa);
            if (wa == Real(0)) continue;
            for (int cB = 0; cB < nctB; ++cB) {
              const Real w = wa * detail::effective_coeff(SB, cB, pb);
              for (int kP = 0; kP < nP; ++kP)
                for (int kQ = 0; kQ < nQ; ++kQ)
                  cblk[(static_cast<std::size_t>(cA) * nP + kP) * rowB + cB * nQ + kQ] +=
                      w * blk[kP * nQ + kQ];
            }
          }
        }
      }
      for (int cA = 0; cA < nctA; ++cA)
        for (int kP = 0; kP < nP; ++kP) {
          const std::size_t I = aux.ao_off[A] + static_cast<std::size_t>(cA) * nP + kP;
          for (int cB = 0; cB < nctB; ++cB)
            for (int kQ = 0; kQ < nQ; ++kQ) {
              const std::size_t J = aux.ao_off[B] + static_cast<std::size_t>(cB) * nQ + kQ;
              const Real v = cblk[(static_cast<std::size_t>(cA) * nP + kP) * rowB + cB * nQ + kQ];
              M[I * naux + J] = v;
              M[J * naux + I] = v;
            }
        }
    }
  return M;
}

/// Three-center Coulomb (mu nu | P) over generally-contracted orbital and
/// auxiliary bases: row-major (mu, nu, P) tensor of size nao*nao*naux (PySCF
/// cart=True normalization).
/// [sa0, sa1) restricts the AUXILIARY shells; the returned tensor's auxiliary
/// index is local to that range. Streams one auxiliary block at a time instead
/// of materialising nao^2 x naux, which at nao = 1000, naux = 4000 is 32 GB.
template <class Real>
std::vector<Real> coulomb_3c_auxblock(const ContractedBasis<Real> &orb,
                                      const ContractedBasis<Real> &aux,
                                      const TGrid<Real> &grid, int sa0, int sa1) {
  if constexpr (kokkos_scalar_v<Real>)
    return detail::coulomb_3c_dev(orb, aux, grid, sa0, sa1);
  const int nao = orb.nao;
  const int pbase = aux.ao_off[sa0], naux = aux.ao_off[sa1] - pbase;
  std::vector<Real> T(static_cast<std::size_t>(nao) * nao * naux, Real(0));
  const int nso = static_cast<int>(orb.shells.size());
  for (int Mi = 0; Mi < nso; ++Mi)
    for (int Ni = 0; Ni <= Mi; ++Ni) {
      const auto &SM = orb.shells[Mi], &SN = orb.shells[Ni];
      const int nm = ncart(SM.l), nn = ncart(SN.l), nctM = SM.nctr(), nctN = SN.nctr();
      for (int A = sa0; A < sa1; ++A) {
        const auto &SA = aux.shells[A];
        const int nP = ncart(SA.l), nctA = SA.nctr();
        // contracted block cblk[(cM*nm+km)][(cN*nn+kn)][(cA*nP+kP)]
        const int dN = nctN * nn, dA = nctA * nP;
        std::vector<Real> cblk(static_cast<std::size_t>(nctM) * nm * dN * dA, Real(0));
        std::vector<Real> blk(static_cast<std::size_t>(nm) * nn * nP);
        for (int pm = 0; pm < SM.nprim(); ++pm)
          for (int pn = 0; pn < SN.nprim(); ++pn) {
            auto bra = make_pair(detail::contracted_prim(SM, pm), detail::contracted_prim(SN, pn));
            for (int pa = 0; pa < SA.nprim(); ++pa) {
              auto ket = detail::ghost_pair(detail::contracted_prim(SA, pa));
              eri_quartet(bra, ket, grid, blk.data());
              for (int cM = 0; cM < nctM; ++cM) {
                const Real wm = detail::effective_coeff(SM, cM, pm);
                if (wm == Real(0)) continue;
                for (int cN = 0; cN < nctN; ++cN) {
                  const Real wmn = wm * detail::effective_coeff(SN, cN, pn);
                  for (int cA = 0; cA < nctA; ++cA) {
                    const Real w = wmn * detail::effective_coeff(SA, cA, pa);
                    for (int km = 0; km < nm; ++km)
                      for (int kn = 0; kn < nn; ++kn)
                        for (int kP = 0; kP < nP; ++kP)
                          cblk[(((static_cast<std::size_t>(cM) * nm + km) * dN + cN * nn + kn) * dA) +
                               cA * nP + kP] += w * blk[(km * nn + kn) * nP + kP];
                  }
                }
              }
            }
          }
        const bool offdiag = (Mi != Ni);
        for (int cM = 0; cM < nctM; ++cM)
          for (int km = 0; km < nm; ++km) {
            const std::size_t I = orb.ao_off[Mi] + static_cast<std::size_t>(cM) * nm + km;
            for (int cN = 0; cN < nctN; ++cN)
              for (int kn = 0; kn < nn; ++kn) {
                const std::size_t Jn = orb.ao_off[Ni] + static_cast<std::size_t>(cN) * nn + kn;
                for (int cA = 0; cA < nctA; ++cA)
                  for (int kP = 0; kP < nP; ++kP) {
                    const std::size_t P =
                        aux.ao_off[A] - pbase + static_cast<std::size_t>(cA) * nP + kP;
                    const Real v = cblk[(((static_cast<std::size_t>(cM) * nm + km) * dN + cN * nn + kn) * dA) +
                                        cA * nP + kP];
                    T[(I * nao + Jn) * naux + P] = v;
                    if (offdiag) T[(Jn * nao + I) * naux + P] = v;
                  }
              }
          }
      }
    }
  return T;
}

/// Whole (mu nu | P) over contracted bases: the auxiliary block spanning every
/// shell. Kept as the one-liner it is, so there is a single implementation.
template <class Real>
std::vector<Real> coulomb_3c(const ContractedBasis<Real> &orb,
                             const ContractedBasis<Real> &aux, const TGrid<Real> &grid) {
  return coulomb_3c_auxblock(orb, aux, grid, 0, static_cast<int>(aux.shells.size()));
}

} // namespace intti
