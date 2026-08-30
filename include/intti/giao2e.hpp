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

#include <array>
#include <complex>
#include <cstddef>
#include <vector>

#include "erigrad.hpp" // detail::comp_index, detail::eri_block4
#include "fock.hpp"
#include "gto.hpp"
#include "quartet.hpp"
#include "tgrid.hpp"

namespace intti {

/// dJ/dB and dK/dB at B = 0: three purely imaginary nao x nao matrices each,
/// in the order {x, y, z}. D is the nao x nao AO density (row-major); the J/K
/// definitions are J_mn = sum (mn|ls) D_ls, K_mn = sum (ml|ns) D_ls.
template <class Real> struct GiaoJKderiv {
  std::array<std::vector<std::complex<Real>>, 3> dJ, dK;
};

template <class Real>
GiaoJKderiv<Real> giao_jk_dB(const ShellBasis<Real> &basis, const Real *D,
                             const TGrid<Real> &grid) {
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
  auto Dm = [&](int i, int j) { return D[static_cast<std::size_t>(i) * nao + j]; };
  auto shell = [&](int s, int dl) {
    auto sh = basis.shells[s];
    sh.l += dl;
    return sh;
  };
  auto idx4 = [](const int n[4], int i0, int i1, int i2, int i3) {
    return ((static_cast<std::size_t>(i0) * n[1] + i1) * n[2] + i2) * n[3] + i3;
  };

  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b)
      for (int cc = 0; cc < ns; ++cc)
        for (int dd = 0; dd < ns; ++dd) {
          const int la = basis.shells[a].l, lb = basis.shells[b].l;
          const int lc = basis.shells[cc].l, ld = basis.shells[dd].l;
          const int na = ncart(la), nb = ncart(lb), nc = ncart(lc), nd = ncart(ld);
          const int oa = basis.ao_off[a], ob = basis.ao_off[b];
          const int oc = basis.ao_off[cc], od = basis.ao_off[dd];
          // base and bra-promoted quartet blocks (real ERIs at B = 0)
          auto base = detail::eri_block4(basis.shells[a], basis.shells[b],
                                         basis.shells[cc], basis.shells[dd], grid);
          auto plusA = detail::eri_block4(shell(a, 1), basis.shells[b],
                                          basis.shells[cc], basis.shells[dd], grid);
          auto plusC = detail::eri_block4(basis.shells[a], basis.shells[b],
                                          shell(cc, 1), basis.shells[dd], grid);
          const int nbase[4] = {na, nb, nc, nd};
          const int npA[4] = {ncart(la + 1), nb, nc, nd};
          const int npC[4] = {na, nb, ncart(lc + 1), nd};
          const Real w1[3] = {basis.shells[a].center[0] - basis.shells[b].center[0],
                              basis.shells[a].center[1] - basis.shells[b].center[1],
                              basis.shells[a].center[2] - basis.shells[b].center[2]};
          const Real w2[3] = {basis.shells[cc].center[0] - basis.shells[dd].center[0],
                              basis.shells[cc].center[1] - basis.shells[dd].center[1],
                              basis.shells[cc].center[2] - basis.shells[dd].center[2]};
          const Real *Ra = basis.shells[a].center, *Rc = basis.shells[cc].center;
          for (int ka = 0; ka < na; ++ka) {
            int a3[3];
            cart_comp(la, ka, a3[0], a3[1], a3[2]);
            for (int kb = 0; kb < nb; ++kb)
              for (int kc = 0; kc < nc; ++kc) {
                int c3[3];
                cart_comp(lc, kc, c3[0], c3[1], c3[2]);
                for (int kd = 0; kd < nd; ++kd) {
                  const Real b0 = base[idx4(nbase, ka, kb, kc, kd)];
                  // position-weighted blocks M1_e = (a^{+e} b|cd) + Ra_e base,
                  //                          M2_e = (ab|c^{+e} d) + Rc_e base
                  Real M1[3], M2[3];
                  for (int e = 0; e < 3; ++e) {
                    int m[3] = {a3[0], a3[1], a3[2]};
                    m[e] += 1;
                    const int ip = detail::comp_index(la + 1, m[0], m[1]);
                    M1[e] = plusA[idx4(npA, ip, kb, kc, kd)] + Ra[e] * b0;
                    int mc[3] = {c3[0], c3[1], c3[2]};
                    mc[e] += 1;
                    const int icp = detail::comp_index(lc + 1, mc[0], mc[1]);
                    M2[e] = plusC[idx4(npC, ka, kb, icp, kd)] + Rc[e] * b0;
                  }
                  // dI_k = (1/2)[ (e_k x w1).M1 + (e_k x w2).M2 ]  (real part;
                  // the derivative is i*dI_k)
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
