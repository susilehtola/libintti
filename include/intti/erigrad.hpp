// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Two-electron energy gradient -- the black-box forces piece. The MD shift
// relation extends to the quartet: differentiating shell p of (ab|cd) w.r.t.
// its centre raises/lowers that shell's angular momentum
//   nabla_e phi_p[m] = m_e phi_p[m_e-1] - 2 alpha_p phi_p[m_e+1],
// so d/dR_p (ab|cd) = -(nabla_e) is a linear combination of quartets with
// shell p promoted and demoted. Contracted with the closed-shell two-particle
// density it gives dE_2e/dR per shell centre. Matrix-level: density in, forces
// out; individual quartets stay internal.

#include <array>
#include <cstddef>
#include <vector>

#include "fock.hpp"
#include "gto.hpp"
#include "quartet.hpp"
#include "tgrid.hpp"

namespace intti {

namespace detail {
/// index in shell l of the Cartesian component (lx,ly,lz) (cart_comp order).
inline int comp_index(int l, int lx, int ly) {
  return (l - lx) * (l - lx + 1) / 2 + (l - lx - ly);
}

/// (ab|cd) block for four explicit shells, layout ((ka*ncb+kb)*ncc+kc)*ncd+kd.
template <class Real>
std::vector<Real> eri_block4(const PrimitiveShell<Real> &a,
                             const PrimitiveShell<Real> &b,
                             const PrimitiveShell<Real> &c,
                             const PrimitiveShell<Real> &d, const TGrid<Real> &grid) {
  std::vector<Real> blk(static_cast<std::size_t>(ncart(a.l)) * ncart(b.l) *
                        ncart(c.l) * ncart(d.l));
  eri_quartet(make_pair(a, b), make_pair(c, d), grid, blk.data());
  return blk;
}
} // namespace detail

/// Gradient of the closed-shell two-electron energy
///   E_2e = 1/2 sum D_mn D_ls (mn|ls) - 1/4 sum D_ml D_ns (mn|ls)
/// with respect to each shell centre. Returns forces[nshell][3] = dE_2e/dR_s.
/// D is nao x nao symmetric (row-major).
template <class Real>
std::vector<std::array<Real, 3>>
two_electron_gradient(const ShellBasis<Real> &basis, const Real *D,
                      const TGrid<Real> &grid) {
  const int ns = static_cast<int>(basis.shells.size());
  const int nao = basis.nao;
  std::vector<std::array<Real, 3>> grad(ns, {Real(0), Real(0), Real(0)});
  auto Dm = [&](int i, int j) { return D[static_cast<std::size_t>(i) * nao + j]; };

  // shells with one Cartesian power shifted (for promote/demote blocks)
  auto shell = [&](int s, int dl) {
    auto sh = basis.shells[s];
    sh.l += dl;
    return sh;
  };

  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b)
      for (int cc = 0; cc < ns; ++cc)
        for (int dd = 0; dd < ns; ++dd) {
          const int la = basis.shells[a].l, lb = basis.shells[b].l;
          const int lc = basis.shells[cc].l, ld = basis.shells[dd].l;
          const int nb = ncart(lb), nc = ncart(lc), nd = ncart(ld);
          const int oa = basis.ao_off[a], ob = basis.ao_off[b];
          const int oc = basis.ao_off[cc], od = basis.ao_off[dd];
          const int pos_shell[4] = {a, b, cc, dd};
          const int pos_l[4] = {la, lb, lc, ld};
          const Real pos_alpha[4] = {basis.shells[a].alpha, basis.shells[b].alpha,
                                     basis.shells[cc].alpha, basis.shells[dd].alpha};
          // derivative w.r.t. each of the 4 shell positions
          for (int pos = 0; pos < 4; ++pos) {
            const int lp = pos_l[pos];
            // promoted (l+1) and demoted (l-1) blocks
            auto plus = detail::eri_block4(
                pos == 0 ? shell(a, 1) : basis.shells[a],
                pos == 1 ? shell(b, 1) : basis.shells[b],
                pos == 2 ? shell(cc, 1) : basis.shells[cc],
                pos == 3 ? shell(dd, 1) : basis.shells[dd], grid);
            std::vector<Real> minus;
            if (lp >= 1)
              minus = detail::eri_block4(
                  pos == 0 ? shell(a, -1) : basis.shells[a],
                  pos == 1 ? shell(b, -1) : basis.shells[b],
                  pos == 2 ? shell(cc, -1) : basis.shells[cc],
                  pos == 3 ? shell(dd, -1) : basis.shells[dd], grid);
            const int npl[4] = {ncart(pos == 0 ? la + 1 : la), ncart(pos == 1 ? lb + 1 : lb),
                                ncart(pos == 2 ? lc + 1 : lc), ncart(pos == 3 ? ld + 1 : ld)};
            const int nmi[4] = {ncart(pos == 0 ? la - 1 : la), ncart(pos == 1 ? lb - 1 : lb),
                                ncart(pos == 2 ? lc - 1 : lc), ncart(pos == 3 ? ld - 1 : ld)};
            auto idx = [](const int n[4], int i0, int i1, int i2, int i3) {
              return ((static_cast<std::size_t>(i0) * n[1] + i1) * n[2] + i2) * n[3] + i3;
            };
            for (int ka = 0; ka < ncart(la); ++ka) {
              int a3[3];
              cart_comp(la, ka, a3[0], a3[1], a3[2]);
              for (int kb = 0; kb < nb; ++kb) {
                int b3[3];
                cart_comp(lb, kb, b3[0], b3[1], b3[2]);
                for (int kc = 0; kc < nc; ++kc) {
                  int c3[3];
                  cart_comp(lc, kc, c3[0], c3[1], c3[2]);
                  for (int kd = 0; kd < nd; ++kd) {
                    int d3[3];
                    cart_comp(ld, kd, d3[0], d3[1], d3[2]);
                    const Real coeff =
                        Real(0.5) * Dm(oa + ka, ob + kb) * Dm(oc + kc, od + kd) -
                        Real(0.25) * Dm(oa + ka, oc + kc) * Dm(ob + kb, od + kd);
                    if (coeff == Real(0)) continue;
                    // shift the pos-shell component by +/-1 in each direction
                    int m[3];
                    const int *base3 = pos == 0 ? a3 : pos == 1 ? b3 : pos == 2 ? c3 : d3;
                    for (int e = 0; e < 3; ++e) {
                      m[0] = base3[0]; m[1] = base3[1]; m[2] = base3[2];
                      // d/dR = -nabla ; nabla_e = m_e[.-1] - 2a[.+1]
                      // grad += coeff * (2a * plus[m_e+1] - m_e * minus[m_e-1])
                      int kp = m[e] + 1;
                      m[e] = kp;
                      const int ip = detail::comp_index(lp + 1, m[0], m[1]);
                      Real term = 2 * pos_alpha[pos] *
                                  plus[pos == 0 ? idx(npl, ip, kb, kc, kd)
                                       : pos == 1 ? idx(npl, ka, ip, kc, kd)
                                       : pos == 2 ? idx(npl, ka, kb, ip, kd)
                                                  : idx(npl, ka, kb, kc, ip)];
                      m[e] = base3[e];
                      if (base3[e] >= 1) {
                        m[e] = base3[e] - 1;
                        const int im = detail::comp_index(lp - 1, m[0], m[1]);
                        term -= Real(base3[e]) *
                                minus[pos == 0 ? idx(nmi, im, kb, kc, kd)
                                      : pos == 1 ? idx(nmi, ka, im, kc, kd)
                                      : pos == 2 ? idx(nmi, ka, kb, im, kd)
                                                 : idx(nmi, ka, kb, kc, im)];
                        m[e] = base3[e];
                      }
                      grad[pos_shell[pos]][e] += coeff * term;
                    }
                  }
                }
              }
            }
          }
        }
  return grad;
}

} // namespace intti
