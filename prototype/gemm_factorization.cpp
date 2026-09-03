// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
// Prototype: SHARK-style BLAS-3 (GEMM) factorization of the per-quartet Hermite
// contraction on libintti's t-grid, validated against eri_quartet.
//
// Phase G currently is g[i,combo,d] = pref_i * sum_n fh[n,combo,d] B_n(theta_i,X_d).
// Factorized: build B_d (nf x nt), fold pref into it, then G_d = fhT_d(ncomb x nf)
// * B_d(nf x nt) via one GEMM per direction. Assembly: sum_i w_i Gx Gy Gz.
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include "intti/quartet.hpp"
#include "intti/cholesky.hpp" // detail::gemm_nn

using Real = double;
using intti::ncart;

static void eri_gemm(const intti::ShellPair<Real> &bra, const intti::ShellPair<Real> &ket,
                     const intti::TGrid<Real> &grid, std::vector<Real> &out) {
  const int la = bra.la, lb = bra.lb, lc = ket.la, ld = ket.lb;
  const int Ltot = la + lb + lc + ld, nf = Ltot + 1;
  const int ncomb = (la + 1) * (lb + 1) * (lc + 1) * (ld + 1);
  const int nt = grid.n(), nout = ncart(la) * ncart(lb) * ncart(lc) * ncart(ld);
  const Real p = bra.p, q = ket.p, pi = intti::pi_v<Real>();
  Real X[3];
  for (int d = 0; d < 3; ++d) X[d] = bra.P[d] - ket.P[d];

  // fh[n + nf*(combo + ncomb*d)] (same as eri_quartet), and its transpose fhT_d
  // as (ncomb x nf) column-major per direction for the GEMM.
  std::vector<Real> fh(static_cast<std::size_t>(nf) * ncomb * 3, 0);
  {
    std::vector<Real> Eb((la + 1) * (lb + 1) * (la + lb + 1)), Ek((lc + 1) * (ld + 1) * (lc + ld + 1));
    for (int d = 0; d < 3; ++d) {
      intti::e_coeffs(la, lb, p, bra.P[d] - bra.A[d], bra.P[d] - bra.B[d], bra.K[d], Eb.data());
      intti::e_coeffs(lc, ld, q, ket.P[d] - ket.A[d], ket.P[d] - ket.B[d], ket.K[d], Ek.data());
      auto eb = [&](int i, int j, int t) { return Eb[(i * (lb + 1) + j) * (la + lb + 1) + t]; };
      auto ek = [&](int i, int j, int t) { return Ek[(i * (ld + 1) + j) * (lc + ld + 1) + t]; };
      for (int ia = 0; ia <= la; ++ia) for (int ib = 0; ib <= lb; ++ib)
        for (int ic = 0; ic <= lc; ++ic) for (int id = 0; id <= ld; ++id) {
          const int combo = ((ia * (lb + 1) + ib) * (lc + 1) + ic) * (ld + 1) + id;
          for (int t = 0; t <= ia + ib; ++t) for (int tau = 0; tau <= ic + id; ++tau) {
            const Real term = eb(ia, ib, t) * ek(ic, id, tau);
            fh[(t + tau) + nf * (combo + ncomb * d)] += tau % 2 ? -term : term;
          }
        }
    }
  }
  // fhT_d: (ncomb x nf) column-major (ld=ncomb), one per direction
  std::vector<Real> fhT(static_cast<std::size_t>(ncomb) * nf * 3, 0);
  for (int d = 0; d < 3; ++d)
    for (int combo = 0; combo < ncomb; ++combo)
      for (int n = 0; n < nf; ++n)
        fhT[combo + ncomb * (n + nf * d)] = fh[n + nf * (combo + ncomb * d)];

  // B_d: (nf x nt) column-major, B_d[n,i] = pref_i * B_n(theta_i, X_d)
  std::vector<Real> Bd(static_cast<std::size_t>(nf) * nt * 3), G(static_cast<std::size_t>(ncomb) * nt * 3);
  std::vector<Real> B(nf);
  for (int i = 0; i < nt; ++i) {
    const Real t = grid.t[i], D = p * q + t * t * (p + q);
    const Real theta = t * t * p * q / D, pref = pi / std::sqrt(D);
    for (int d = 0; d < 3; ++d) {
      intti::hermite_b(Ltot, theta, X[d], B.data());
      for (int n = 0; n < nf; ++n) Bd[n + nf * (i + nt * d)] = pref * B[n];
    }
  }
  // GEMM per direction: G_d(ncomb x nt) = fhT_d(ncomb x nf) * B_d(nf x nt)
  for (int d = 0; d < 3; ++d)
    intti::detail::gemm_nn(ncomb, nt, nf, &fhT[nf * ncomb * d], ncomb, &Bd[nf * nt * d], nf,
                           &G[ncomb * nt * d], ncomb);

  // assembly: out[k] = sum_i w_i Gx[cx,i] Gy[cy,i] Gz[cz,i]
  out.assign(nout, 0);
  for (int ka = 0; ka < ncart(la); ++ka) { int a3[3]; intti::cart_comp(la, ka, a3[0], a3[1], a3[2]);
   for (int kb = 0; kb < ncart(lb); ++kb) { int b3[3]; intti::cart_comp(lb, kb, b3[0], b3[1], b3[2]);
    for (int kc = 0; kc < ncart(lc); ++kc) { int c3[3]; intti::cart_comp(lc, kc, c3[0], c3[1], c3[2]);
     for (int kd = 0; kd < ncart(ld); ++kd) { int d3[3]; intti::cart_comp(ld, kd, d3[0], d3[1], d3[2]);
      const int k = ((ka * ncart(lb) + kb) * ncart(lc) + kc) * ncart(ld) + kd;
      int cmb[3]; for (int d = 0; d < 3; ++d)
        cmb[d] = ((a3[d] * (lb + 1) + b3[d]) * (lc + 1) + c3[d]) * (ld + 1) + d3[d];
      Real v = 0;
      for (int i = 0; i < nt; ++i)
        v += grid.w[i] * G[cmb[0] + ncomb * i] * G[cmb[1] + ncomb * (i + nt)] *
             G[cmb[2] + ncomb * (i + 2 * nt)];
      out[k] = v;
     }}}}
}

int main(int argc, char **argv) {
  Kokkos::ScopeGuard g(argc, argv);
  auto sh = [](Real a, Real x, Real y, Real z, int l) { return intti::PrimitiveShell<Real>{a, {x, y, z}, l}; };
  auto grid = intti::make_tgrid(intti::coulomb());
  double worst = 0;
  int cases[][4] = {{0,0,0,0},{1,0,0,0},{1,1,0,0},{2,1,1,0},{1,1,1,1},{2,2,0,0}};
  for (auto &c : cases) {
    auto bra = intti::make_pair(sh(0.8,0,0,0,c[0]), sh(1.3,0.5,-0.2,0.4,c[1]));
    auto ket = intti::make_pair(sh(2.1,1.0,0.8,0.0,c[2]), sh(0.35,-0.4,0.3,1.1,c[3]));
    const int nout = ncart(c[0])*ncart(c[1])*ncart(c[2])*ncart(c[3]);
    std::vector<Real> ref(nout), gem;
    intti::eri_quartet(bra, ket, grid, ref.data());
    eri_gemm(bra, ket, grid, gem);
    double e = 0, sc = 0;
    for (int k = 0; k < nout; ++k) { e = std::max(e, std::fabs(gem[k]-ref[k])); sc = std::max(sc, std::fabs(ref[k])); }
    std::printf("l=(%d%d%d%d) nout=%2d  max_abs_diff=%.2e (scale %.2e)\n", c[0],c[1],c[2],c[3], nout, e, sc);
    worst = std::max(worst, e/sc);
  }
  std::printf("worst relative diff GEMM vs eri_quartet: %.2e\n", worst);
  return 0;
}
