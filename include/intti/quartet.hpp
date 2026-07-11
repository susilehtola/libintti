// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola
#pragma once

#include <stdexcept>
#include <type_traits>
#include <vector>

#include "gto.hpp"
#include "hermite1d.hpp"
#include "tgrid.hpp"

namespace intti {

/// Primitive Cartesian ERI quartet (ab|cd) by t quadrature:
///   (ab|cd) = sum_i w_i Ix(t_i) Iy(t_i) Iz(t_i) + tail_coeff * S_abcd,
/// where I_d are analytic 1D two-pair integrals and S_abcd is the
/// four-orbital overlap (the delta-function tail correction; only active
/// for truncated grids, i.e. TMapping::LinLog).
///
/// out is an array of ncart(la)*ncart(lb)*ncart(lc)*ncart(ld) values in
/// row-major (a, b, c, d) component order. Primitives are unnormalized.
///
/// Executes with Kokkos parallel patterns for builtin floating-point Real
/// (float, double, long double); class-type scalars (e.g. MPFR wrappers)
/// run on a serial host path.
template <class Real>
void eri_quartet(const ShellPair<Real> &bra, const ShellPair<Real> &ket,
                 const TGrid<Real> &grid, Real *out) {
  const int la = bra.la, lb = bra.lb, lc = ket.la, ld = ket.lb;
  if (la > LMAX || lb > LMAX || lc > LMAX || ld > LMAX)
    throw std::invalid_argument("eri_quartet: shell angular momentum exceeds LMAX");
  const int Ltot = la + lb + lc + ld;
  const int nf = Ltot + 1; // Hermite orders 0..Ltot
  // per-direction (ia, ib, ic, id) combinations
  const int ncomb = (la + 1) * (lb + 1) * (lc + 1) * (ld + 1);
  const int nt = grid.n();
  const int nout = ncart(la) * ncart(lb) * ncart(lc) * ncart(ld);

  const Real p = bra.p, q = ket.p;
  const Real rho = p * q / (p + q);
  const Real pi = pi_v<Real>();
  Real X[3];
  for (int d = 0; d < 3; ++d)
    X[d] = bra.P[d] - ket.P[d];

  // f_n = sum_{t+tau=n} E_t^{bra} E_tau^{ket} (-1)^tau per direction and
  // per-direction component combination; index n + nf*(combo + ncomb*d)
  std::vector<Real> fh(static_cast<std::size_t>(nf) * ncomb * 3, Real(0));
  {
    std::vector<Real> Eb((la + 1) * (lb + 1) * (la + lb + 1));
    std::vector<Real> Ek((lc + 1) * (ld + 1) * (lc + ld + 1));
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
                for (int tau = 0; tau <= ic + id; ++tau) {
                  const Real term = eb(ia, ib, t) * ek(ic, id, tau);
                  fh[(t + tau) + nf * (combo + ncomb * d)] += tau % 2 ? -term : term;
                }
            }
    }
  }

  // map output component -> per-direction combination indices; k + nout*d
  std::vector<int> ch(static_cast<std::size_t>(nout) * 3);
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
            ch[k + nout * d] = ((a3[d] * (lb + 1) + b3[d]) * (lc + 1) + c3[d]) * (ld + 1) + d3[d];
        }
      }
    }
  }

  // delta-function tail: per-direction 1D overlaps at the t -> infinity
  // limit (theta -> rho, prefactor -> sqrt(pi/(p+q))); combo + ncomb*d
  std::vector<Real> s1d(static_cast<std::size_t>(ncomb) * 3, Real(0));
  if (grid.tail_coeff != Real(0)) {
    std::vector<Real> B(nf);
    const Real spref = sqrt_(pi / (p + q));
    for (int d = 0; d < 3; ++d) {
      hermite_b(Ltot, rho, X[d], B.data());
      for (int combo = 0; combo < ncomb; ++combo) {
        Real s{};
        for (int n = 0; n < nf; ++n)
          s += fh[n + nf * (combo + ncomb * d)] * B[n];
        s1d[combo + ncomb * d] = spref * s;
      }
    }
  }

  if constexpr (std::is_floating_point_v<Real>) {
    // Kokkos execution
    Kokkos::View<Real ***, Kokkos::LayoutLeft> f("intti::quartet::f", nf, ncomb, 3);
    Kokkos::View<int **, Kokkos::LayoutLeft> comps("intti::quartet::comps", nout, 3);
    Kokkos::View<Real **, Kokkos::LayoutLeft> s1("intti::quartet::s1d", ncomb, 3);
    {
      auto fm = Kokkos::create_mirror_view(f);
      auto cm = Kokkos::create_mirror_view(comps);
      auto sm = Kokkos::create_mirror_view(s1);
      for (int d = 0; d < 3; ++d) {
        for (int combo = 0; combo < ncomb; ++combo) {
          for (int n = 0; n < nf; ++n)
            fm(n, combo, d) = fh[n + nf * (combo + ncomb * d)];
          sm(combo, d) = s1d[combo + ncomb * d];
        }
        for (int k = 0; k < nout; ++k)
          cm(k, d) = ch[k + nout * d];
      }
      Kokkos::deep_copy(f, fm);
      Kokkos::deep_copy(comps, cm);
      Kokkos::deep_copy(s1, sm);
    }

    // per-node, per-direction 1D integrals: g(i, combo, d), node index fastest
    Kokkos::View<Real ***, Kokkos::LayoutLeft> g("intti::quartet::g", nt, ncomb, 3);
    auto tv = grid.t_dev;
    auto wv = grid.w_dev;
    const Real X0 = X[0], X1 = X[1], X2 = X[2];
    Kokkos::parallel_for(
        "intti::quartet::nodes", Kokkos::RangePolicy<>(0, nt), KOKKOS_LAMBDA(int i) {
          Real B[4 * LMAX + 1];
          const Real t = tv(i);
          const Real D = p * q + t * t * (p + q);
          const Real theta = t * t * p * q / D;
          const Real pref = pi / sqrt_(D);
          const Real Xd[3] = {X0, X1, X2};
          for (int d = 0; d < 3; ++d) {
            hermite_b(Ltot, theta, Xd[d], B);
            for (int combo = 0; combo < ncomb; ++combo) {
              Real s{};
              for (int n = 0; n < nf; ++n)
                s += f(n, combo, d) * B[n];
              g(i, combo, d) = pref * s;
            }
          }
        });

    // assemble components: quadrature sum + tail correction
    Kokkos::View<Real *> outv("intti::quartet::out", nout);
    const Real tail_coeff = grid.tail_coeff;
    Kokkos::parallel_for(
        "intti::quartet::assemble", Kokkos::RangePolicy<>(0, nout), KOKKOS_LAMBDA(int k) {
          const int cx = comps(k, 0), cy = comps(k, 1), cz = comps(k, 2);
          Real val{};
          for (int i = 0; i < nt; ++i)
            val += wv(i) * g(i, cx, 0) * g(i, cy, 1) * g(i, cz, 2);
          val += tail_coeff * s1(cx, 0) * s1(cy, 1) * s1(cz, 2);
          outv(k) = val;
        });

    auto oh = Kokkos::create_mirror_view(outv);
    Kokkos::deep_copy(oh, outv);
    for (int k = 0; k < nout; ++k)
      out[k] = oh(k);
  } else {
    // serial host execution for class-type scalars (arbitrary precision)
    std::vector<Real> g(static_cast<std::size_t>(nt) * ncomb * 3);
    std::vector<Real> B(nf);
    for (int i = 0; i < nt; ++i) {
      const Real t = grid.t[i];
      const Real D = p * q + t * t * (p + q);
      const Real theta = t * t * p * q / D;
      const Real pref = pi / sqrt_(D);
      for (int d = 0; d < 3; ++d) {
        hermite_b(Ltot, theta, X[d], B.data());
        for (int combo = 0; combo < ncomb; ++combo) {
          Real s{};
          for (int n = 0; n < nf; ++n)
            s += fh[n + nf * (combo + ncomb * d)] * B[n];
          g[i + nt * (combo + ncomb * d)] = pref * s;
        }
      }
    }
    for (int k = 0; k < nout; ++k) {
      const int cx = ch[k], cy = ch[k + nout], cz = ch[k + 2 * nout];
      Real val{};
      for (int i = 0; i < nt; ++i)
        val += grid.w[i] * g[i + nt * cx] * g[i + nt * (cy + ncomb)] *
               g[i + nt * (cz + 2 * ncomb)];
      val += grid.tail_coeff * s1d[cx] * s1d[cy + ncomb] * s1d[cz + 2 * ncomb];
      out[k] = val;
    }
  }
}

} // namespace intti
