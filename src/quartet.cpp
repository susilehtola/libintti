// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola

#include "intti/quartet.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "intti/hermite1d.hpp"

namespace intti {

void eri_quartet(const ShellPair &bra, const ShellPair &ket, const TGrid &grid,
                 double *out) {
  const int la = bra.la, lb = bra.lb, lc = ket.la, ld = ket.lb;
  if (la > LMAX || lb > LMAX || lc > LMAX || ld > LMAX)
    throw std::invalid_argument("eri_quartet: shell angular momentum exceeds LMAX");
  const int Ltot = la + lb + lc + ld;
  const int nf = Ltot + 1; // Hermite orders 0..Ltot
  // per-direction (ia, ib, ic, id) combinations
  const int ncomb = (la + 1) * (lb + 1) * (lc + 1) * (ld + 1);
  const int nt = grid.n();
  const int nout = ncart(la) * ncart(lb) * ncart(lc) * ncart(ld);

  const double p = bra.p, q = ket.p;
  const double rho = p * q / (p + q);
  double X[3];
  for (int d = 0; d < 3; ++d)
    X[d] = bra.P[d] - ket.P[d];

  // f_n = sum_{t+tau=n} E_t^{bra} E_tau^{ket} (-1)^tau per direction and
  // per-direction component combination; t is fastest in memory (LayoutLeft)
  Kokkos::View<double ***, Kokkos::LayoutLeft> f("intti::quartet::f", nf, ncomb, 3);
  auto fh = Kokkos::create_mirror_view(f);
  {
    std::vector<double> Eb((la + 1) * (lb + 1) * (la + lb + 1));
    std::vector<double> Ek((lc + 1) * (ld + 1) * (lc + ld + 1));
    for (int d = 0; d < 3; ++d) {
      e_coeffs(la, lb, p, bra.P[d] - bra.A[d], bra.P[d] - bra.B[d], bra.K[d], Eb.data());
      e_coeffs(lc, ld, q, ket.P[d] - ket.A[d], ket.P[d] - ket.B[d], ket.K[d], Ek.data());
      auto eb = [&](int i, int j, int t) { return Eb[(i * (lb + 1) + j) * (la + lb + 1) + t]; };
      auto ek = [&](int i, int j, int t) { return Ek[(i * (ld + 1) + j) * (lc + ld + 1) + t]; };
      for (int ia = 0; ia <= la; ++ia)
        for (int ib = 0; ib <= lb; ++ib)
          for (int ic = 0; ic <= lc; ++ic)
            for (int id = 0; id <= ld; ++id) {
              const int combo = ((ia * (lb + 1) + ib) * (lc + 1) + ic) * (ld + 1) + id;
              for (int t = 0; t <= ia + ib; ++t)
                for (int tau = 0; tau <= ic + id; ++tau)
                  fh(t + tau, combo, d) +=
                      eb(ia, ib, t) * ek(ic, id, tau) * (tau % 2 ? -1.0 : 1.0);
            }
    }
  }
  Kokkos::deep_copy(f, fh);

  // map output component -> per-direction combination indices
  Kokkos::View<int **, Kokkos::LayoutLeft> comps("intti::quartet::comps", nout, 3);
  auto ch = Kokkos::create_mirror_view(comps);
  for (int ka = 0; ka < ncart(la); ++ka) {
    int a3[3];
    cart_comp(la, ka, a3[0], a3[1], a3[2]);
    for (int kb = 0; kb < ncart(lb); ++kb) {
      int b3[3];
      cart_comp(lb, kb, b3[0], b3[1], b3[2]);
      for (int kc = 0; kc < ncart(lc); ++kc) {
        int c3[3];
        cart_comp(lc, kc, c3[0], c3[1], c3[2]);
        for (int kd = 0; kd < ncart(ld); ++kd) {
          int d3[3];
          cart_comp(ld, kd, d3[0], d3[1], d3[2]);
          const int k = ((ka * ncart(lb) + kb) * ncart(lc) + kc) * ncart(ld) + kd;
          for (int d = 0; d < 3; ++d)
            ch(k, d) = ((a3[d] * (lb + 1) + b3[d]) * (lc + 1) + c3[d]) * (ld + 1) + d3[d];
        }
      }
    }
  }
  Kokkos::deep_copy(comps, ch);

  // per-node, per-direction 1D integrals: g(i, combo, d), node index fastest
  Kokkos::View<double ***, Kokkos::LayoutLeft> g("intti::quartet::g", nt, ncomb, 3);
  auto tv = grid.t;
  auto wv = grid.w;
  const double X0 = X[0], X1 = X[1], X2 = X[2];
  Kokkos::parallel_for(
      "intti::quartet::nodes", Kokkos::RangePolicy<>(0, nt), KOKKOS_LAMBDA(int i) {
        double B[4 * LMAX + 1];
        const double t = tv(i);
        const double D = p * q + t * t * (p + q);
        const double theta = t * t * p * q / D;
        const double pref = M_PI / Kokkos::sqrt(D);
        const double Xd[3] = {X0, X1, X2};
        for (int d = 0; d < 3; ++d) {
          hermite_b(Ltot, theta, Xd[d], B);
          for (int combo = 0; combo < ncomb; ++combo) {
            double s = 0.0;
            for (int n = 0; n < nf; ++n)
              s += f(n, combo, d) * B[n];
            g(i, combo, d) = pref * s;
          }
        }
      });

  // delta-function tail: per-direction 1D overlaps at the t -> infinity limit
  // (theta -> rho, prefactor -> sqrt(pi/(p+q)))
  Kokkos::View<double **, Kokkos::LayoutLeft> s1d("intti::quartet::s1d", ncomb, 3);
  Kokkos::parallel_for(
      "intti::quartet::tail", Kokkos::RangePolicy<>(0, 3), KOKKOS_LAMBDA(int d) {
        double B[4 * LMAX + 1];
        const double Xd[3] = {X0, X1, X2};
        hermite_b(Ltot, rho, Xd[d], B);
        const double pref = Kokkos::sqrt(M_PI / (p + q));
        for (int combo = 0; combo < ncomb; ++combo) {
          double s = 0.0;
          for (int n = 0; n < nf; ++n)
            s += f(n, combo, d) * B[n];
          s1d(combo, d) = pref * s;
        }
      });

  // assemble components: quadrature sum + tail correction
  Kokkos::View<double *> outv("intti::quartet::out", nout);
  const double tail_coeff = grid.tail_coeff;
  Kokkos::parallel_for(
      "intti::quartet::assemble", Kokkos::RangePolicy<>(0, nout), KOKKOS_LAMBDA(int k) {
        const int cx = comps(k, 0), cy = comps(k, 1), cz = comps(k, 2);
        double val = 0.0;
        for (int i = 0; i < nt; ++i)
          val += wv(i) * g(i, cx, 0) * g(i, cy, 1) * g(i, cz, 2);
        val += tail_coeff * s1d(cx, 0) * s1d(cy, 1) * s1d(cz, 2);
        outv(k) = val;
      });

  auto oh = Kokkos::create_mirror_view(outv);
  Kokkos::deep_copy(oh, outv);
  for (int k = 0; k < nout; ++k)
    out[k] = oh(k);
}

} // namespace intti
