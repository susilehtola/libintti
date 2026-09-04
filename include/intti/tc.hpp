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

#include "gto.hpp"
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

} // namespace intti
