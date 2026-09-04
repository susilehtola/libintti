// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Transcorrelated (TC) two-body integrals -- M-TC. The Jastrow-similarity
// transform H_TC = e^{-u} H e^{u} adds, besides the Hermitian pieces
// 1/2 nabla^2 u and 1/2 (nabla u)^2 (plain Gaussian-geminal J/K builds, see
// gaussian_geminal in tgrid.hpp) and the 3-body L term (threeel.hpp), ONE
// genuinely new integral: the NON-HERMITIAN operator nabla_1 u_12 . nabla_1
// (Haupt thesis 2024, Eq 2.29-2.31). Integrating by parts moves the geminal
// gradient (x1-x2) off the operator onto the electron-1 orbitals, collapsing
// the moment-weighted geminal to PLAIN Gaussian-geminal integrals with
// derivative-shifted orbitals:
//
//   <pq| nabla_1 u_12 . nabla_1 |rs>
//     = - sum_k c_k sum_d [ G_k( (d_d chi_p)(d_d chi_r), chi_q chi_s )
//                         + G_k(  chi_p (d_d^2 chi_r), chi_q chi_s ) ],
//
// where G_k(rho1, rho2) = <rho1| e^{-g_k r12^2} |rho2> is the plain geminal
// two-electron integral (the geminal TGrid sums over k automatically) and
// d_d chi(l) = l_d chi[l_d-1] - 2 alpha chi[l_d+1] is the MD center-shift
// derivative (deriv.hpp / erigrad.hpp). Validated to machine precision against
// two independent symbolic oracles (references/sympy_tc.py): full 6D Gaussian
// integration and the factorised 2D-per-axis form.
//
// electron 1 = (p, r), electron 2 = (q, s), i.e. <pq|O|rs>. Like eri_quartet,
// this per-quartet routine is INTERNAL / test-only; the matrix-level TC-Fock
// assembler (density in, non-Hermitian effective matrices out) is the public
// surface.

#include <algorithm>
#include <cstddef>
#include <map>
#include <utility>
#include <vector>

#include "fock.hpp"
#include "gto.hpp"
#include "hermite1d.hpp"
#include "math.hpp"
#include "quartet.hpp"
#include "tgrid.hpp"

namespace intti {

namespace detail {
/// index in shell l of the Cartesian component with powers (lx,ly,*) in
/// cart_comp order.
inline int tc_comp_index(int l, int lx, int ly) {
  return (l - lx) * (l - lx + 1) / 2 + (l - lx - ly);
}
} // namespace detail

/// Non-Hermitian TC two-body integral <pq| nabla_1 u . nabla_1 |rs> for the
/// Gaussian-geminal u carried by `geminal` (a gaussian_geminal TGrid). out has
/// ncart(lp)*ncart(lq)*ncart(lr)*ncart(ls) entries, laid out as
/// ((kp*ncq+kq)*ncr+kr)*ncs+ks. INTERNAL / test-only.
template <class Real>
void tc_gradu_grad_quartet(const PrimitiveShell<Real> &p,
                           const PrimitiveShell<Real> &q,
                           const PrimitiveShell<Real> &r,
                           const PrimitiveShell<Real> &s,
                           const TGrid<Real> &geminal, Real *out) {
  const int Lp = p.l, Lq = q.l, Lr = r.l, Ls = s.l;
  const int ncp = ncart(Lp), ncq = ncart(Lq), ncr = ncart(Lr), ncs = ncart(Ls);
  std::fill(out, out + static_cast<std::size_t>(ncp) * ncq * ncr * ncs, Real(0));

  // Geminal quartets (p+dlp, r+dlr | q, s) cached by the (dlp, dlr) shifts;
  // electron 1 = (p, r) so eri_quartet's bra pair is (p_shifted, r_shifted).
  std::map<std::pair<int, int>, std::vector<Real>> cache;
  auto block = [&](int dlp, int dlr) -> const std::vector<Real> & {
    const auto key = std::make_pair(dlp, dlr);
    auto it = cache.find(key);
    if (it == cache.end()) {
      std::vector<Real> blk;
      if (Lp + dlp >= 0 && Lr + dlr >= 0) {
        auto ps = p;
        ps.l += dlp;
        auto rs = r;
        rs.l += dlr;
        blk.assign(static_cast<std::size_t>(ncart(Lp + dlp)) * ncart(Lr + dlr) *
                       ncq * ncs,
                   Real(0));
        eri_quartet(make_pair(ps, rs), make_pair(q, s), geminal, blk.data());
      }
      it = cache.emplace(key, std::move(blk)).first;
    }
    return it->second;
  };

  // one shifted-shell contribution: powers (px,py,*) of p in shell Lp+dlp and
  // (rx,ry,*) of r in shell Lr+dlr, weighted by coeff, into out[kp,kq,kr,ks].
  auto accumulate = [&](int dlp, int rp[3], int dlr, int rr[3], Real coeff,
                        int kp, int kq, int kr, int ks) {
    const auto &blk = block(dlp, dlr);
    if (blk.empty()) return;
    const int Lrp = Lr + dlr, Lpp = Lp + dlp;
    const int ip = detail::tc_comp_index(Lpp, rp[0], rp[1]);
    const int ir = detail::tc_comp_index(Lrp, rr[0], rr[1]);
    const int ncrp = ncart(Lrp);
    const Real v = blk[((static_cast<std::size_t>(ip) * ncrp + ir) * ncq + kq) *
                           ncs +
                       ks];
    out[((static_cast<std::size_t>(kp) * ncq + kq) * ncr + kr) * ncs + ks] +=
        -coeff * v;
  };

  for (int kp = 0; kp < ncp; ++kp) {
    int p3[3];
    cart_comp(Lp, kp, p3[0], p3[1], p3[2]);
    for (int kr = 0; kr < ncr; ++kr) {
      int r3[3];
      cart_comp(Lr, kr, r3[0], r3[1], r3[2]);
      for (int kq = 0; kq < ncq; ++kq)
        for (int ks = 0; ks < ncs; ++ks)
          for (int d = 0; d < 3; ++d) {
            // d_d chi_p : (+1, -2 alpha_p) and (-1, p_d) if p_d>0
            // d_d chi_r : (+1, -2 alpha_r) and (-1, r_d) if r_d>0
            struct Sh {
              int dl;
              Real c;
            };
            Sh dp[2];
            int ndp = 0;
            dp[ndp++] = {+1, Real(-2) * p.alpha};
            if (p3[d] > 0) dp[ndp++] = {-1, Real(p3[d])};
            Sh dr[2];
            int ndr = 0;
            dr[ndr++] = {+1, Real(-2) * r.alpha};
            if (r3[d] > 0) dr[ndr++] = {-1, Real(r3[d])};
            // term 1: (d_d p)(d_d r)
            for (int a = 0; a < ndp; ++a)
              for (int b = 0; b < ndr; ++b) {
                int rp[3] = {p3[0], p3[1], p3[2]};
                rp[d] += dp[a].dl;
                int rr[3] = {r3[0], r3[1], r3[2]};
                rr[d] += dr[b].dl;
                accumulate(dp[a].dl, rp, dr[b].dl, rr, dp[a].c * dr[b].c, kp, kq,
                           kr, ks);
              }
            // term 2: p, d_d^2 r
            //   d_d^2 chi_r = r_d(r_d-1) chi[r_d-2] - 2a(2 r_d+1) chi[r_d]
            //                 + 4 a^2 chi[r_d+2]
            {
              int rr[3] = {r3[0], r3[1], r3[2]};
              accumulate(0, p3, +2, (rr[d] += 2, rr),
                         Real(4) * r.alpha * r.alpha, kp, kq, kr, ks);
            }
            {
              int rr[3] = {r3[0], r3[1], r3[2]};
              accumulate(0, p3, 0, rr,
                         Real(-2) * r.alpha * Real(2 * r3[d] + 1), kp, kq, kr,
                         ks);
            }
            if (r3[d] >= 2) {
              int rr[3] = {r3[0], r3[1], r3[2]};
              rr[d] -= 2;
              accumulate(0, p3, -2, rr, Real(r3[d] * (r3[d] - 1)), kp, kq, kr,
                         ks);
            }
          }
    }
  }
}

/// Matrix-level non-Hermitian TC two-body Fock contribution
///   F_{p r} = sum_{q s} D_{q s} <p q| nabla_1 u . nabla_1 |r s>
/// for the Gaussian-geminal u carried by `geminal`. D and the returned F are
/// nao x nao row-major Cartesian AO matrices; F is NOT symmetric (the operator
/// is non-Hermitian -- that is the point of TC). This is the public,
/// matrix-level consumer of the reduction: density in, effective matrix out,
/// quartets never materialised. (The Hermitian TC pieces 1/2 nabla^2 u and
/// 1/2 (nabla u)^2 are plain geminal J/K builds via gaussian_geminal; the 3-body
/// L term is threeel.hpp.)
///
/// Structure -- a geminal J-build (cf. coulomb_build), NOT an N^4 quartet loop:
///   F_{pr} = sum_{qs} D_{qs} ( rho~_{pr} | g | chi_q chi_s ),
/// where the bra "charge" rho~_{pr} = -div(chi_p grad chi_r) is a FIXED modified
/// pair density with a finite Hermite expansion. So: (phase 1) fold D into per
/// ket-pair Hermite tensors d^{qs}; (phase 2) couple each bra pair (p,r) to all
/// ket pairs through the geminal grid ONCE, giving the field j^{pr} in the bra
/// Hermite basis; (phase 3) contract j^{pr} with the modified bra moments
/// q~_{pr} = -sum_d M_d (x) S (x) S (per-axis: M_d combines the 1-D E-coeffs of
/// (d_d p)(d_d r) and p(d_d^2 r); S is the plain 1-D E-coeff). Cost is that of a
/// geminal J-build, O(npair^2 ng L^4), not O(N^4) materialised quartets. The bra
/// pairs are independent (each writes a disjoint F block) so the outer loop runs
/// in parallel on the host execution space. tau > 0 enables a Schwarz-style
/// magnitude x geminal-decay screen on the (bra, ket) coupling (0 = exact).
template <class Real>
std::vector<Real> tc_gradu_grad_build(const ShellBasis<Real> &basis,
                                      const Real *D, const TGrid<Real> &geminal,
                                      Real tau = Real(0)) {
  const int nao = basis.nao;
  std::vector<Real> F(static_cast<std::size_t>(nao) * nao, Real(0));
  const int ns = static_cast<int>(basis.shells.size());
  const int ng = geminal.n();
  const Real pi = pi_v<Real>();

  // ---- phase 1: per ordered ket shell-pair (sq,ss), density-weighted Hermite
  //      tensor d^{qs}_{tuv} = sum_{kq,ks} D_{qs} E^{qs} (tuv up to lq+ls). ----
  struct Ket {
    Real p, P[3];
    Real mk; // ||d||_1, screening magnitude
    int Lq;
    std::vector<Real> d; // (Lq+1)^3
  };
  std::vector<Ket> kets;
  kets.reserve(static_cast<std::size_t>(ns) * ns);
  for (int sq = 0; sq < ns; ++sq)
    for (int ss = 0; ss < ns; ++ss) {
      const auto &Q = basis.shells[sq], &Sh = basis.shells[ss];
      const int lq = Q.l, ls = Sh.l, n1 = lq + ls + 1;
      const int esz = (lq + 1) * (ls + 1) * n1;
      const Real p = Q.alpha + Sh.alpha, mu = Q.alpha * Sh.alpha / p;
      Ket k;
      k.p = p;
      k.Lq = lq + ls;
      std::vector<Real> E(static_cast<std::size_t>(3) * esz);
      for (int d = 0; d < 3; ++d) {
        k.P[d] = (Q.alpha * Q.center[d] + Sh.alpha * Sh.center[d]) / p;
        const Real ab = Q.center[d] - Sh.center[d];
        e_coeffs(lq, ls, p, k.P[d] - Q.center[d], k.P[d] - Sh.center[d],
                 exp_(-mu * ab * ab), E.data() + d * esz);
      }
      k.d.assign(static_cast<std::size_t>(n1) * n1 * n1, Real(0));
      const int oq = basis.ao_off[sq], os = basis.ao_off[ss];
      for (int kq = 0; kq < ncart(lq); ++kq) {
        int a3[3];
        cart_comp(lq, kq, a3[0], a3[1], a3[2]);
        for (int ksh = 0; ksh < ncart(ls); ++ksh) {
          int b3[3];
          cart_comp(ls, ksh, b3[0], b3[1], b3[2]);
          const Real Dv = D[static_cast<std::size_t>(oq + kq) * nao + os + ksh];
          if (Dv == Real(0)) continue;
          const Real *Ex = E.data() + 0 * esz + (a3[0] * (ls + 1) + b3[0]) * n1;
          const Real *Ey = E.data() + 1 * esz + (a3[1] * (ls + 1) + b3[1]) * n1;
          const Real *Ez = E.data() + 2 * esz + (a3[2] * (ls + 1) + b3[2]) * n1;
          for (int tx = 0; tx <= a3[0] + b3[0]; ++tx)
            for (int ty = 0; ty <= a3[1] + b3[1]; ++ty)
              for (int tz = 0; tz <= a3[2] + b3[2]; ++tz)
                k.d[(tx * n1 + ty) * n1 + tz] += Dv * Ex[tx] * Ey[ty] * Ez[tz];
        }
      }
      k.mk = Real(0);
      for (Real v : k.d) k.mk += v < 0 ? -v : v;
      kets.push_back(std::move(k));
    }

  // ---- bra loop over ordered shell-pairs (sp,sr): independent, parallel ----
  Kokkos::parallel_for(
      "intti::tc::gradu_grad",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, ns * ns),
      [&](const int bpr) {
      const int sp = bpr / ns, sr = bpr % ns;
      const auto &P = basis.shells[sp], &R = basis.shells[sr];
      const int lp = P.l, lr = R.l;
      const int lpm = lp + 1, lrm = lr + 2;       // raised for the derivatives
      const int Lb = lp + lr + 2, nb1 = Lb + 1;   // bra Hermite order
      const int nbm = lpm + lrm + 1;
      const int ebsz = (lpm + 1) * (lrm + 1) * nbm;
      const Real ppr = P.alpha + R.alpha, mu = P.alpha * R.alpha / ppr;
      Real Ppr[3];
      std::vector<Real> Eb(static_cast<std::size_t>(3) * ebsz);
      for (int d = 0; d < 3; ++d) {
        Ppr[d] = (P.alpha * P.center[d] + R.alpha * R.center[d]) / ppr;
        const Real ab = P.center[d] - R.center[d];
        e_coeffs(lpm, lrm, ppr, Ppr[d] - P.center[d], Ppr[d] - R.center[d],
                 exp_(-mu * ab * ab), Eb.data() + d * ebsz);
      }
      auto E1 = [&](int d, int i, int j) {
        return Eb.data() + d * ebsz + (i * (lrm + 1) + j) * nbm;
      };
      // bra magnitude bound for screening: ||q~||_1 <= ||Eb||_1 * (2 amax+Lb+1)^2
      Real mb = Real(0);
      if (tau > Real(0)) {
        Real en = 0;
        for (Real v : Eb) en += v < 0 ? -v : v;
        const Real amax = P.alpha > R.alpha ? P.alpha : R.alpha;
        const Real cf = 2 * amax + Real(Lb + 1);
        mb = en * cf * cf;
      }

      // phase 2: field j^{pr}_{tuv} from coupling to every ket pair
      std::vector<Real> jp(static_cast<std::size_t>(nb1) * nb1 * nb1, Real(0));
      std::vector<Real> Bx, By, Bz, t1, t2;
      for (const auto &k : kets) {
        Real X[3], R2 = 0;
        for (int d = 0; d < 3; ++d) {
          X[d] = Ppr[d] - k.P[d];
          R2 += X[d] * X[d];
        }
        // Schwarz-style screen: sum_g |w_g| (pi/sqrt D_g)^3 e^{-theta_g R^2}
        // bounds the geminal coupling geometry; * mb * ||d||_1 the contribution.
        if (tau > Real(0)) {
          Real gsum = 0;
          for (int it = 0; it < ng; ++it) {
            const Real g = geminal.t[it] * geminal.t[it];
            const Real Dden = ppr * k.p + g * (ppr + k.p);
            const Real th = g * ppr * k.p / Dden;
            const Real pr = pi / sqrt_(Dden);
            const Real w = geminal.w[it] < 0 ? -geminal.w[it] : geminal.w[it];
            gsum += w * pr * pr * pr * exp_(-th * R2);
          }
          if (mb * k.mk * gsum < tau) continue;
        }
        const int Lq = k.Lq, nq1 = Lq + 1, nB = Lb + Lq + 1;
        Bx.resize(nB);
        By.resize(nB);
        Bz.resize(nB);
        t1.assign(static_cast<std::size_t>(nb1) * nq1 * nq1, Real(0));
        t2.assign(static_cast<std::size_t>(nb1) * nb1 * nq1, Real(0));
        for (int it = 0; it < ng; ++it) {
          const Real t = geminal.t[it], g = t * t;
          const Real Dden = ppr * k.p + g * (ppr + k.p);
          const Real theta = g * ppr * k.p / Dden;
          const Real pr = pi / sqrt_(Dden);
          const Real wpref = geminal.w[it] * pr * pr * pr;
          hermite_b(nB - 1, theta, X[0], Bx.data());
          hermite_b(nB - 1, theta, X[1], By.data());
          hermite_b(nB - 1, theta, X[2], Bz.data());
          for (int a = 0; a < nb1; ++a)
            for (int nu = 0; nu < nq1; ++nu)
              for (int ph = 0; ph < nq1; ++ph) {
                Real s = 0;
                for (int tau = 0; tau < nq1; ++tau) {
                  const Real term = k.d[(tau * nq1 + nu) * nq1 + ph] * Bx[a + tau];
                  s += tau % 2 ? -term : term;
                }
                t1[(a * nq1 + nu) * nq1 + ph] = s;
              }
          for (int a = 0; a < nb1; ++a)
            for (int b = 0; b < nb1; ++b)
              for (int ph = 0; ph < nq1; ++ph) {
                Real s = 0;
                for (int nu = 0; nu < nq1; ++nu) {
                  const Real term = t1[(a * nq1 + nu) * nq1 + ph] * By[b + nu];
                  s += nu % 2 ? -term : term;
                }
                t2[(a * nb1 + b) * nq1 + ph] = s;
              }
          for (int a = 0; a < nb1; ++a)
            for (int b = 0; b < nb1; ++b)
              for (int c = 0; c < nb1; ++c) {
                Real s = 0;
                for (int ph = 0; ph < nq1; ++ph) {
                  const Real term = t2[(a * nb1 + b) * nq1 + ph] * Bz[c + ph];
                  s += ph % 2 ? -term : term;
                }
                jp[(a * nb1 + b) * nb1 + c] += wpref * s;
              }
        }
      }

      // phase 3: F_{pr} = sum_tuv q~_{pr}[tuv] j^{pr}[tuv], with
      //   q~ = -sum_d M_d (x) S (x) S  (M_d = 1-D coeffs of (d_d p)(d_d r) + p d_d^2 r)
      const int op = basis.ao_off[sp], orr = basis.ao_off[sr];
      std::vector<Real> S[3], M[3];
      for (int c = 0; c < 3; ++c) {
        S[c].resize(nb1);
        M[c].resize(nb1);
      }
      for (int kp = 0; kp < ncart(lp); ++kp) {
        int ap[3];
        cart_comp(lp, kp, ap[0], ap[1], ap[2]);
        for (int kr = 0; kr < ncart(lr); ++kr) {
          int cr[3];
          cart_comp(lr, kr, cr[0], cr[1], cr[2]);
          for (int d = 0; d < 3; ++d) {
            const int a = ap[d], c = cr[d];
            std::fill(S[d].begin(), S[d].end(), Real(0));
            std::fill(M[d].begin(), M[d].end(), Real(0));
            const Real *Sc = E1(d, a, c);
            for (int t = 0; t <= a + c; ++t) S[d][t] = Sc[t];
            // (d_d p)(d_d r): a c E[a-1,c-1] - 2ar a E[a-1,c+1]
            //                 - 2ap c E[a+1,c-1] + 4 ap ar E[a+1,c+1]
            if (a > 0 && c > 0) {
              const Real *e = E1(d, a - 1, c - 1);
              for (int t = 0; t <= a + c - 2; ++t) M[d][t] += Real(a * c) * e[t];
            }
            if (a > 0) {
              const Real *e = E1(d, a - 1, c + 1);
              for (int t = 0; t <= a + c; ++t) M[d][t] += Real(-2) * R.alpha * Real(a) * e[t];
            }
            if (c > 0) {
              const Real *e = E1(d, a + 1, c - 1);
              for (int t = 0; t <= a + c; ++t) M[d][t] += Real(-2) * P.alpha * Real(c) * e[t];
            }
            {
              const Real *e = E1(d, a + 1, c + 1);
              for (int t = 0; t <= a + c + 2; ++t)
                M[d][t] += Real(4) * P.alpha * R.alpha * e[t];
            }
            // p d_d^2 r: c(c-1) E[a,c-2] - 2ar(2c+1) E[a,c] + 4 ar^2 E[a,c+2]
            if (c >= 2) {
              const Real *e = E1(d, a, c - 2);
              for (int t = 0; t <= a + c - 2; ++t) M[d][t] += Real(c * (c - 1)) * e[t];
            }
            {
              const Real *e = E1(d, a, c);
              for (int t = 0; t <= a + c; ++t)
                M[d][t] += Real(-2) * R.alpha * Real(2 * c + 1) * e[t];
            }
            {
              const Real *e = E1(d, a, c + 2);
              for (int t = 0; t <= a + c + 2; ++t)
                M[d][t] += Real(4) * R.alpha * R.alpha * e[t];
            }
          }
          // F = - sum_d sum_tuv (M_d in dir d)(S in others) jp[tuv]
          Real acc = 0;
          for (int tx = 0; tx < nb1; ++tx)
            for (int ty = 0; ty < nb1; ++ty)
              for (int tz = 0; tz < nb1; ++tz) {
                const Real j = jp[(tx * nb1 + ty) * nb1 + tz];
                if (j == Real(0)) continue;
                acc += j * (M[0][tx] * S[1][ty] * S[2][tz] +
                            S[0][tx] * M[1][ty] * S[2][tz] +
                            S[0][tx] * S[1][ty] * M[2][tz]);
              }
          F[static_cast<std::size_t>(op + kp) * nao + orr + kr] = -acc;
        }
      }
      });
  Kokkos::fence();
  return F;
}

} // namespace intti
