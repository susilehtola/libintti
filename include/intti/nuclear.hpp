// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Nuclear attraction and Coulomb-potential collocation matrices via the
// t-quadrature. At fixed t the interaction of an AO pair with a point at C is
// the overlap of the pair's Hermite expansion with a Gaussian exp(-t^2(r-C)^2):
//
//   <a|1/|r-C||b> = (2/sqrt(pi)) int_0^inf dt  prod_d G_d(t),
//   G_d(t) = sqrt(pi/(p+t^2)) sum_tau E^d_tau B_tau(theta_t, P_d - C_d),
//   theta_t = p t^2/(p+t^2),
//
// with E the McMurchie-Davidson coefficients (e_coeffs) and B the
// Gaussian-derivative array (hermite_b). This is the critical primitive for
// local hybrids (the local exchange energy density is dominated by these), so
// it carries distance/Schwarz screening. Matrix-level API: whole nao x nao
// matrices, never per-shell blocks.

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "fock.hpp"
#include "gto.hpp"
#include "hermite1d.hpp"
#include "math.hpp"
#include "multipole.hpp"
#include "oneel.hpp" // detail::make_1e_pairs (device pair table + side arrays)
#include "tgrid.hpp"

namespace intti {

/// A weighted point (a nuclear charge Z at R, or a unit collocation point).
template <class Real> struct PointCharge {
  Real weight; ///< contribution weight (e.g. -Z for nuclear attraction)
  Real R[3];
};

namespace detail {

/// Nuclear-attraction block of ONE primitive pair (sa, sb):
///   acc[ka*ncb+kb] = sum_c weight_c <a_ka|1/|r-R_c||b_kb>,
/// OVERWRITING acc (length ncart(sa.l)*ncart(sb.l)). tau > 0 enables the
/// per-(pair,centre) decay screen; far_tau > 0 the FMM multipole far branch.
/// The shared per-primitive-pair kernel of the primitive nuclear builder
/// (attraction_accumulate) and the contracted one (contracted.hpp).
template <class Real, class Sink>
void attraction_pair_visit(const PrimitiveShell<Real> &sa,
                           const PrimitiveShell<Real> &sb,
                           const std::vector<PointCharge<Real>> &centers,
                           const TGrid<Real> &grid, Real tau, Real far_tau,
                           Sink &&sink) {
  const int nt = grid.n();
  const Real pi = pi_v<Real>();
  const bool screen = tau > Real(0);
  // FMM far-field: a centre well separated from the pair (p |P-c|^2 > far_cut)
  // is a monopole seen through the exponent-free multipole tensor T_{tuv}(P-c);
  // <a|1/r_c|b> = (pi/p)^{3/2} sum_tuv E^{ab}_tuv T_tuv, exact up to exp(-p R^2).
  const bool far = far_tau > Real(0);
  const Real far_cut = far ? -log_(far_tau) : Real(0);
  const int la = sa.l, lb = sb.l, esz = (la + 1) * (lb + 1) * (la + lb + 1);
  const Real p = sa.alpha + sb.alpha;
  const Real mu = sa.alpha * sb.alpha / p;
  Real Pd[3], expmu = 1;
  std::vector<Real> E(static_cast<std::size_t>(3) * esz);
  for (int d = 0; d < 3; ++d) {
    Pd[d] = (sa.alpha * sa.center[d] + sb.alpha * sb.center[d]) / p;
    const Real ab = sa.center[d] - sb.center[d];
    const Real Kd = exp_(-mu * ab * ab);
    expmu *= Kd;
    e_coeffs(la, lb, p, Pd[d] - sa.center[d], Pd[d] - sb.center[d], Kd,
             E.data() + d * esz);
  }
  const Real pair_bound = (2 * pi / p) * expmu; // (2 pi/p) exp(-mu R_AB^2)
  const int nca = ncart(la), ncb = ncart(lb);
  std::vector<Real> Bx(la + lb + 1), By(la + lb + 1), Bz(la + lb + 1), Tbuf, Tscr;
  std::vector<Real> blk(static_cast<std::size_t>(nca) * ncb);
  const int n1 = la + lb + 1;
  for (std::size_t ic = 0; ic < centers.size(); ++ic) {
    const auto &c = centers[ic];
    for (int i = 0; i < nca * ncb; ++i) blk[i] = Real(0);
    Real d2 = 0;
    if (screen || far)
      for (int d = 0; d < 3; ++d)
        d2 += (Pd[d] - c.R[d]) * (Pd[d] - c.R[d]);
    if (screen) {
      using std::sqrt;
      const Real fb = d2 * p > Real(1) ? Real(0.5) * sqrt(pi / (p * d2)) : Real(1);
      if (std::abs(c.weight) * pair_bound * fb < tau) continue;
    }
    if (far && p * d2 > far_cut) {
      // multipole far branch: T_{tuv}(P - c), monopole charge (no ket sign)
      const Real X[3] = {Pd[0] - c.R[0], Pd[1] - c.R[1], Pd[2] - c.R[2]};
      const int Dt = la + lb + 1;
      // scratch hoisted out of the centre loop: multipole_tensor's own buffer is
      // O((la+lb+1)^4), and allocating it per centre costs more than the tensor
      // at high angular momentum
      if (Tbuf.empty()) {
        Tbuf.assign(static_cast<std::size_t>(Dt) * Dt * Dt, Real(0));
        Tscr.assign(static_cast<std::size_t>(Dt) * Dt * Dt * Dt, Real(0));
      }
      multipole_tensor(la + lb, X, Tbuf.data(), Tscr.data());
      const Real pop = pi / p;
      const Real wpref = c.weight * pop * sqrt_(pop); // (pi/p)^{3/2}
      for (int ka = 0; ka < nca; ++ka) {
        int a3[3];
        cart_comp(la, ka, a3[0], a3[1], a3[2]);
        for (int kb = 0; kb < ncb; ++kb) {
          int b3[3];
          cart_comp(lb, kb, b3[0], b3[1], b3[2]);
          const Real *Ex = E.data() + 0 * esz + (a3[0] * (lb + 1) + b3[0]) * n1;
          const Real *Ey = E.data() + 1 * esz + (a3[1] * (lb + 1) + b3[1]) * n1;
          const Real *Ez = E.data() + 2 * esz + (a3[2] * (lb + 1) + b3[2]) * n1;
          Real s = 0;
          for (int tx = 0; tx <= a3[0] + b3[0]; ++tx)
            for (int ty = 0; ty <= a3[1] + b3[1]; ++ty)
              for (int tz = 0; tz <= a3[2] + b3[2]; ++tz)
                s += Ex[tx] * Ey[ty] * Ez[tz] * Tbuf[(tx * Dt + ty) * Dt + tz];
          blk[ka * ncb + kb] = wpref * s;
        }
      }
      sink(ic, blk.data());
      continue;
    }
    for (int it = 0; it < nt; ++it) {
      const Real t = grid.t[it];
      const Real denom = p + t * t;
      const Real theta = p * t * t / denom;
      const Real pref = sqrt_(pi / denom);
      hermite_b(la + lb, theta, Pd[0] - c.R[0], Bx.data());
      hermite_b(la + lb, theta, Pd[1] - c.R[1], By.data());
      hermite_b(la + lb, theta, Pd[2] - c.R[2], Bz.data());
      const Real wt = grid.w[it] * c.weight;
      for (int ka = 0; ka < nca; ++ka) {
        int a3[3];
        cart_comp(la, ka, a3[0], a3[1], a3[2]);
        for (int kb = 0; kb < ncb; ++kb) {
          int b3[3];
          cart_comp(lb, kb, b3[0], b3[1], b3[2]);
          auto gd = [&](int d, const Real *B) {
            const Real *Ed = E.data() + d * esz +
                             (a3[d] * (lb + 1) + b3[d]) * (la + lb + 1);
            Real s = 0;
            for (int tau_ = 0; tau_ <= a3[d] + b3[d]; ++tau_)
              s += Ed[tau_] * B[tau_];
            return pref * s;
          };
          blk[ka * ncb + kb] += wt * gd(0, Bx.data()) * gd(1, By.data()) *
                                gd(2, Bz.data());
        }
      }
    }
    sink(ic, blk.data());
  }
}

/// Accumulate sum_c weight_c <a|1/|r-R_c||b> into acc (ncart(la) x ncart(lb)),
/// OVERWRITING it: the original block form, now a sink over the visitor.
template <class Real>
void attraction_pair_block(const PrimitiveShell<Real> &sa,
                           const PrimitiveShell<Real> &sb,
                           const std::vector<PointCharge<Real>> &centers,
                           const TGrid<Real> &grid, Real tau, Real far_tau,
                           Real *acc) {
  const int n = ncart(sa.l) * ncart(sb.l);
  for (int i = 0; i < n; ++i) acc[i] = Real(0);
  attraction_pair_visit(sa, sb, centers, grid, tau, far_tau,
                        [&](std::size_t, const Real *blk) {
                          for (int i = 0; i < n; ++i) acc[i] += blk[i];
                        });
}

/// Accumulate sum_c weight_c <a|1/|r-R_c||b> into V (nao x nao, row-major).
/// tau > 0 enables a per-(pair,centre) Schwarz/decay screen.
template <class Real>
void attraction_accumulate(const ShellBasis<Real> &basis,
                           const std::vector<PointCharge<Real>> &centers,
                           const TGrid<Real> &grid, Real tau, Real *V,
                           Real far_tau = Real(0)) {
  const int nao = basis.nao;
  const int ns = static_cast<int>(basis.shells.size());
  std::vector<Real> acc;
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const int nca = ncart(sa.l), ncb = ncart(sb.l);
      acc.assign(static_cast<std::size_t>(nca) * ncb, Real(0));
      attraction_pair_block(sa, sb, centers, grid, tau, far_tau, acc.data());
      for (int ka = 0; ka < nca; ++ka)
        for (int kb = 0; kb < ncb; ++kb)
          V[(basis.ao_off[a] + ka) * nao + basis.ao_off[b] + kb] +=
              acc[ka * ncb + kb];
    }
}

/// Position-weighted nuclear attraction <a| x_c V |b> for c = x, y, z, with V
/// the point-charge Coulomb potential of `centers` and x_c the ABSOLUTE
/// coordinate (moment about the origin). Built by promoting the bra angular
/// momentum by one unit: x_c = (x_c - A_c) + A_c, so the first piece raises the
/// bra index and the second is A_c times the base attraction. Output M[c] is
/// nao x nao row-major. This is the GIAO nuclear field-derivative primitive.
template <class Real>
void nuclear_moment_accumulate(const ShellBasis<Real> &basis,
                               const std::vector<PointCharge<Real>> &centers,
                               const TGrid<Real> &grid, Real tau, Real *M0,
                               Real *M1, Real *M2) {
  const int nao = basis.nao;
  const int nt = grid.n();
  const Real pi = pi_v<Real>();
  const int ns = static_cast<int>(basis.shells.size());
  const bool screen = tau > Real(0);
  Real *Mc[3] = {M0, M1, M2};
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const int la = sa.l, lb = sb.l, lax = la + 1;
      const int esz = (lax + 1) * (lb + 1) * (lax + lb + 1);
      const Real p = sa.alpha + sb.alpha;
      const Real mu = sa.alpha * sb.alpha / p;
      Real Pd[3], expmu = 1;
      std::vector<Real> E(static_cast<std::size_t>(3) * esz);
      for (int d = 0; d < 3; ++d) {
        Pd[d] = (sa.alpha * sa.center[d] + sb.alpha * sb.center[d]) / p;
        const Real ab = sa.center[d] - sb.center[d];
        const Real Kd = exp_(-mu * ab * ab);
        expmu *= Kd;
        e_coeffs(lax, lb, p, Pd[d] - sa.center[d], Pd[d] - sb.center[d], Kd,
                 E.data() + d * esz);
      }
      const Real pair_bound = (2 * pi / p) * expmu;
      const int nca = ncart(la), ncb = ncart(lb);
      std::vector<Real> acc0(static_cast<std::size_t>(nca) * ncb, Real(0));
      std::vector<Real> acc[3] = {acc0, acc0, acc0};
      std::vector<Real> Bx(lax + lb + 1), By(lax + lb + 1), Bz(lax + lb + 1);
      const int ntab = lax + lb + 1;
      for (const auto &c : centers) {
        if (screen) {
          Real d2 = 0;
          for (int d = 0; d < 3; ++d) d2 += (Pd[d] - c.R[d]) * (Pd[d] - c.R[d]);
          using std::sqrt;
          const Real fb = d2 * p > Real(1) ? Real(0.5) * sqrt(pi / (p * d2)) : Real(1);
          if (std::abs(c.weight) * pair_bound * fb < tau) continue;
        }
        for (int it = 0; it < nt; ++it) {
          const Real t = grid.t[it];
          const Real denom = p + t * t;
          const Real theta = p * t * t / denom;
          const Real pref = sqrt_(pi / denom);
          hermite_b(lax + lb, theta, Pd[0] - c.R[0], Bx.data());
          hermite_b(lax + lb, theta, Pd[1] - c.R[1], By.data());
          hermite_b(lax + lb, theta, Pd[2] - c.R[2], Bz.data());
          const Real wt = grid.w[it] * c.weight;
          const Real *Bd[3] = {Bx.data(), By.data(), Bz.data()};
          for (int ka = 0; ka < nca; ++ka) {
            int a3[3];
            cart_comp(la, ka, a3[0], a3[1], a3[2]);
            for (int kb = 0; kb < ncb; ++kb) {
              int b3[3];
              cart_comp(lb, kb, b3[0], b3[1], b3[2]);
              // gd(d, ish): 1D factor with the bra index raised by ish in dir d
              auto gd = [&](int d, int ish) {
                const int i = a3[d] + ish;
                const Real *Ed = E.data() + d * esz + (i * (lb + 1) + b3[d]) * ntab;
                Real s = 0;
                for (int tt = 0; tt <= i + b3[d]; ++tt) s += Ed[tt] * Bd[d][tt];
                return pref * s;
              };
              const Real g0 = gd(0, 0), g1 = gd(1, 0), g2 = gd(2, 0);
              const Real base = g0 * g1 * g2;
              // <mu| x_c V |nu> = <mu^{+c}|V|nu> + A_c <mu|V|nu>
              acc[0][ka * ncb + kb] += wt * (gd(0, 1) * g1 * g2 + sa.center[0] * base);
              acc[1][ka * ncb + kb] += wt * (g0 * gd(1, 1) * g2 + sa.center[1] * base);
              acc[2][ka * ncb + kb] += wt * (g0 * g1 * gd(2, 1) + sa.center[2] * base);
            }
          }
        }
      }
      for (int c = 0; c < 3; ++c)
        for (int ka = 0; ka < nca; ++ka)
          for (int kb = 0; kb < ncb; ++kb)
            Mc[c][(basis.ao_off[a] + ka) * nao + basis.ao_off[b] + kb] +=
                acc[c][ka * ncb + kb];
    }
}

// Device (GPU) nuclear attraction, exact (no screening / FMM). Same t-quadrature
// as attraction_pair_block, on a Kokkos parallel_for over canonical shell pairs:
// E is precomputed on the host (make_1e_pairs); per pair the charge x t-node
// loop computes hermite_b on device once per (charge,t) and accumulates the
// axis-factorised block straight into V (and its transpose for a != b), so no
// large per-thread buffer is needed. la+lb <= 2*LMAX bounds the B stack arrays.
template <class Real>
std::vector<Real> nuclear_matrix_dev(const ShellBasis<Real> &basis,
                                     const std::vector<PointCharge<Real>> &charges,
                                     const TGrid<Real> &grid) {
  const int nao = basis.nao;
  auto op = detail::make_1e_pairs(basis, 0, 0, Real(0)); // exact: all pairs
  const int npair = op.npair, nt = grid.n(), npc = static_cast<int>(charges.size());
  auto tv = detail::to_device(grid.t, "intti::nuc::t");
  auto wv = detail::to_device(grid.w, "intti::nuc::w");
  std::vector<Real> hcw(npc);
  for (int c = 0; c < npc; ++c) hcw[c] = charges[c].weight;
  auto cw = detail::to_device(hcw, "intti::nuc::cw");
  Kokkos::View<Real *[3], Kokkos::LayoutLeft> cR("intti::nuc::cR", npc);
  {
    auto h = Kokkos::create_mirror_view(cR);
    for (int c = 0; c < npc; ++c)
      for (int d = 0; d < 3; ++d) h(c, d) = charges[c].R[d];
    Kokkos::deep_copy(cR, h);
  }
  Kokkos::View<Real *> Vd("intti::nuc::V", static_cast<std::size_t>(nao) * nao);
  auto pv = op.tab.p, Ev = op.tab.E;
  auto Pv = op.tab.P;
  auto lav = op.tab.la, lbv = op.tab.lb, eoffv = op.tab.e_off, aoa = op.aoa, aob = op.aob;
  const Real pi = pi_v<Real>();
  Kokkos::parallel_for(
      "intti::nuc::asm", Kokkos::RangePolicy<>(0, npair), KOKKOS_LAMBDA(int p) {
        const int la = lav(p), lb = lbv(p), n1 = la + lb + 1;
        const int esz = (la + 1) * (lb + 1) * n1;
        const Real pp = pv(p);
        const int eo = eoffv(p), oa = aoa(p), ob = aob(p);
        const bool mirror = (oa != ob);
        const Real Px = Pv(p, 0), Py = Pv(p, 1), Pz = Pv(p, 2);
        Real Bx[2 * LMAX + 1], By[2 * LMAX + 1], Bz[2 * LMAX + 1];
        for (int c = 0; c < npc; ++c) {
          const Real Rx = cR(c, 0), Ry = cR(c, 1), Rz = cR(c, 2), wc = cw(c);
          for (int it = 0; it < nt; ++it) {
            const Real t = tv(it), denom = pp + t * t, theta = pp * t * t / denom;
            const Real pref = sqrt_(pi / denom), wt = wv(it) * wc;
            hermite_b(la + lb, theta, Px - Rx, Bx);
            hermite_b(la + lb, theta, Py - Ry, By);
            hermite_b(la + lb, theta, Pz - Rz, Bz);
            for (int ka = 0; ka < ncart(la); ++ka) {
              int a3[3];
              cart_comp(la, ka, a3[0], a3[1], a3[2]);
              for (int kb = 0; kb < ncart(lb); ++kb) {
                int b3[3];
                cart_comp(lb, kb, b3[0], b3[1], b3[2]);
                Real g[3];
                for (int d = 0; d < 3; ++d) {
                  const Real *Ed = &Ev(eo + d * esz + (a3[d] * (lb + 1) + b3[d]) * n1);
                  const Real *B = (d == 0 ? Bx : (d == 1 ? By : Bz));
                  Real s = 0;
                  for (int tt = 0; tt <= a3[d] + b3[d]; ++tt) s += Ed[tt] * B[tt];
                  g[d] = pref * s;
                }
                const Real val = wt * g[0] * g[1] * g[2];
                const int r = oa + ka, cc = ob + kb;
                Vd(static_cast<std::size_t>(r) * nao + cc) += val;
                if (mirror) Vd(static_cast<std::size_t>(cc) * nao + r) += val;
              }
            }
          }
        }
      });
  return detail::to_host(Vd);
}

} // namespace detail

/// Nuclear attraction matrix V_ab = -sum_C Z_C <a|1/|r-R_C||b> (matches
/// PySCF int1e_nuc). charges carry Z (positive); tau enables screening.
template <class Real>
std::vector<Real> nuclear_matrix(const ShellBasis<Real> &basis,
                                 const std::vector<PointCharge<Real>> &charges,
                                 const TGrid<Real> &grid, Real tau = Real(0),
                                 Real far_tau = Real(0)) {
  if constexpr (kokkos_scalar_v<Real>) {
    // exact case -> GPU; screening / FMM / l > LMAX keep the host path
    if (tau == Real(0) && far_tau == Real(0)) {
      bool ok = true;
      for (const auto &s : basis.shells)
        if (s.l > LMAX) ok = false;
      if (ok) return detail::nuclear_matrix_dev(basis, charges, grid);
    }
  }
  std::vector<Real> V(static_cast<std::size_t>(basis.nao) * basis.nao, Real(0));
  // charges here already carry weight = -Z (caller sets it); provide a helper
  detail::attraction_accumulate(basis, charges, grid, tau, V.data(), far_tau);
  return V;
}

/// Position-weighted nuclear attraction {<a|x V|b>, <a|y V|b>, <a|z V|b>}
/// about the coordinate origin (the GIAO nuclear field-derivative primitive).
/// `charges` carry the operator weight (use nuclei_as_charges for -Z).
template <class Real>
std::array<std::vector<Real>, 3>
nuclear_moment_matrices(const ShellBasis<Real> &basis,
                        const std::vector<PointCharge<Real>> &charges,
                        const TGrid<Real> &grid, Real tau = Real(0)) {
  const std::size_t n2 = static_cast<std::size_t>(basis.nao) * basis.nao;
  std::array<std::vector<Real>, 3> M;
  for (auto &m : M) m.assign(n2, Real(0));
  detail::nuclear_moment_accumulate(basis, charges, grid, tau, M[0].data(),
                                    M[1].data(), M[2].data());
  return M;
}

/// Convenience: build charges with weight = -Z from (Z, R) nuclei.
template <class Real>
std::vector<PointCharge<Real>>
nuclei_as_charges(const std::vector<Real> &Z,
                  const std::vector<std::array<Real, 3>> &R) {
  std::vector<PointCharge<Real>> c(Z.size());
  for (std::size_t i = 0; i < Z.size(); ++i)
    c[i] = {-Z[i], {R[i][0], R[i][1], R[i][2]}};
  return c;
}

/// Coulomb-potential collocation: for each point r_g, the matrix
/// <mu|1/|r-r_g||nu>. This is the primitive local hybrids consume. Returned
/// as one nao x nao matrix per point.
template <class Real>
std::vector<std::vector<Real>>
potential_matrices(const ShellBasis<Real> &basis,
                   const std::vector<std::array<Real, 3>> &points,
                   const TGrid<Real> &grid, Real tau = Real(0),
                   Real far_tau = Real(0)) {
  std::vector<std::vector<Real>> out;
  out.reserve(points.size());
  const std::size_t n2 = static_cast<std::size_t>(basis.nao) * basis.nao;
  for (const auto &pt : points) {
    std::vector<Real> V(n2, Real(0));
    std::vector<PointCharge<Real>> one{{Real(1), {pt[0], pt[1], pt[2]}}};
    detail::attraction_accumulate(basis, one, grid, tau, V.data(), far_tau);
    out.push_back(std::move(V));
  }
  return out;
}

/// Coulomb potential of an AO DENSITY on a set of points:
///   V(g) = sum_ab D_ab <a| 1/|r - r_g| |b>,
/// i.e. the potential the density generates, sampled at r_g. Returns one value
/// per point.
///
/// This is the shape a grid consumer needs, and the reason potential_matrices
/// cannot serve it: that returns an nao x nao matrix PER POINT, which on an FE
/// grid is nao^2 N^3 -- 3.7e6 points at nao = 24 is 2e9 doubles, so it fails the
/// tractability rule outright. Contracting with D inside the point loop keeps
/// the footprint at O(npoints) and never forms the per-point matrix at all.
///
/// far_tau routes a point well separated from a pair (p |P - r_g|^2 >
/// -ln(far_tau)) through the exponent-free multipole tensor instead of the
/// t-quadrature -- exact up to exp(-p R^2), and the whole point of doing this
/// analytically rather than on a grid.
template <class Real>
std::vector<Real> potential_on_points(const ShellBasis<Real> &basis, const Real *D,
                                      const std::vector<std::array<Real, 3>> &points,
                                      const TGrid<Real> &grid, Real tau = Real(0),
                                      Real far_tau = Real(0)) {
  const int ns = static_cast<int>(basis.shells.size());
  const int nao = basis.nao;
  const std::size_t npt = points.size();
  std::vector<Real> V(npt, Real(0));
  if (npt == 0) return V;

  // density sub-block per canonical pair, with the off-diagonal counted once
  std::vector<std::pair<int, int>> pr;
  std::vector<std::vector<Real>> Dblk;
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b <= a; ++b) {
      const int nca = ncart(basis.shells[a].l), ncb = ncart(basis.shells[b].l);
      const int oa = basis.ao_off[a], ob = basis.ao_off[b];
      std::vector<Real> blk(static_cast<std::size_t>(nca) * ncb);
      for (int ka = 0; ka < nca; ++ka)
        for (int kb = 0; kb < ncb; ++kb) {
          const Real d = D[static_cast<std::size_t>(oa + ka) * nao + ob + kb];
          blk[ka * ncb + kb] =
              (a == b) ? d : d + D[static_cast<std::size_t>(ob + kb) * nao + oa + ka];
        }
      pr.push_back({a, b});
      Dblk.push_back(std::move(blk));
    }

  // Parallel over CHUNKS of points, not over pairs: each thread owns a disjoint
  // slice of V, so no atomics and no false sharing, and the visitor's scratch
  // stays thread-local. Each chunk repeats the per-pair E setup, which is
  // O((la+1)(lb+1)(la+lb+1)) against a chunk's worth of point work.
  const int nthread = Kokkos::DefaultHostExecutionSpace().concurrency();
  const std::size_t target = 512;
  std::size_t nchunk = (npt + target - 1) / target;
  if (nchunk < static_cast<std::size_t>(nthread)) nchunk = static_cast<std::size_t>(nthread);
  if (nchunk > npt) nchunk = npt;
  const std::size_t per = (npt + nchunk - 1) / nchunk;
  Kokkos::parallel_for(
      "intti::potential_on_points",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, static_cast<int>(nchunk)),
      [&](int ch) {
        const std::size_t g0 = static_cast<std::size_t>(ch) * per;
        if (g0 >= npt) return;
        const std::size_t g1 = g0 + per < npt ? g0 + per : npt;
        std::vector<PointCharge<Real>> pc;
        pc.reserve(g1 - g0);
        for (std::size_t g = g0; g < g1; ++g)
          pc.push_back({Real(1), {points[g][0], points[g][1], points[g][2]}});
        for (std::size_t ip = 0; ip < pr.size(); ++ip) {
          const auto &sa = basis.shells[pr[ip].first], &sb = basis.shells[pr[ip].second];
          const int nb = ncart(sa.l) * ncart(sb.l);
          const Real *db = Dblk[ip].data();
          detail::attraction_pair_visit(sa, sb, pc, grid, tau, far_tau,
                                        [&](std::size_t ic, const Real *blk) {
                                          Real s = 0;
                                          for (int i = 0; i < nb; ++i) s += db[i] * blk[i];
                                          V[g0 + ic] += s;
                                        });
        }
      });
  Kokkos::fence();
  return V;
}

/// Points grouped into boxes for the boxed far field. `start` holds nbox+1
/// offsets into a point list that is ORDERED BY BOX, so box b owns
/// [start[b], start[b+1]); `center` and `radius` bound each box, with radius the
/// largest |r_g - center| over its points.
template <class Real> struct PointBoxes {
  std::vector<int> start;
  std::vector<std::array<Real, 3>> center;
  std::vector<Real> radius;
  int nbox() const { return static_cast<int>(center.size()); }
};

/// V(g) = sum_ab D_ab <a|1/|r-r_g||b>, with the far field taken ONCE PER BOX
/// instead of once per point.
///
/// MEASURED: THIS DOES NOT BEAT THE PER-POINT FAR FIELD, and the shipped grid
/// path does not use it. Against potential_on_points with the same far_tau, on a
/// 4-centre chain with 27000 points in 216 boxes, it runs at 0.94x (s/p), 1.04x
/// (s/d) and 1.12x (s/f). It is kept because it is correct, tested, and is the
/// M2L kernel a real fast multipole method would need -- not because it is
/// currently worth calling.
///
/// The reason it does not pay is structural, and worth recording so the mistake
/// is not repeated. The box criterion is STRICTER than the point criterion, so
/// boxing can only reclassify points that the per-point path was already
/// handling cheaply; it can never convert an expensive near point into a far
/// one. Meanwhile each box needs the multipole tensor to order
/// (l_a + l_b) + lloc rather than l_a + l_b -- 13 against 3 for an s/p pair at
/// lloc = 10, and the tensor recursion is O(D^4). Fewer sites, each much more
/// expensive, and the two cancel.
///
/// What would pay is grouping the SOURCES as well: one multipole per source box
/// instead of one per pair, so the interaction count becomes
/// nbox x nbox rather than nbox x npair. That is a real hierarchical method and
/// a much larger piece of work; this routine is the half of it that exists.
///
/// Writing X_B = P - c_B and d = r_g - c_B, the multipole tensor expands as
///   T_tuv(X_B - d) = sum_abc (-1)^{a+b+c} (d^abc / a!b!c!) T_{t+a,u+b,v+c}(X_B),
/// so the far potential of a pair is the polynomial sum_abc L_abc d^abc with
///   L_abc = (-1)^{a+b+c}/(a!b!c!) (pi/p)^{3/2} sum_tuv Ehat_tuv T_{t+a,u+b,v+c}(X_B),
/// where Ehat are the pair's Hermite moments already contracted with the density
/// block. The L of every far pair accumulate into ONE polynomial per box, which
/// is then evaluated at the box's points -- so the far field costs
/// nbox x npair for the expansions plus npoints for the evaluation, rather than
/// npoints x npair.
///
/// A pair is far from a box when both hold: the Gaussian asymptotics are valid
/// over the whole box, p (|X_B| - a_B)^2 > -ln(far_tau), and the Taylor series
/// converges, a_B <= sep |X_B|. Everything else stays on the per-point path,
/// which keeps its own point-level near/far split.
///
/// The default sep is far_tau^{1/(lloc+1)}, from the nominal (a_B/|X_B|)^{lloc+1}
/// remainder. That is conservative: measured convergence is closer to sep^6 than
/// sep^11 at lloc = 10, so an explicit, looser sep is usually safe. Measured
/// error at far_tau = 1e-10, 4-centre chain: lloc = 10 gives 8e-11 at sep = 0.12,
/// 8e-9 at 0.25 and 1e-6 at 0.40.
template <class Real>
std::vector<Real>
potential_on_points_boxed(const ShellBasis<Real> &basis, const Real *D,
                          const std::vector<std::array<Real, 3>> &points,
                          const PointBoxes<Real> &boxes, const TGrid<Real> &grid,
                          Real tau = Real(0), Real far_tau = Real(1e-10),
                          int lloc = 10, Real sep = Real(-1)) {
  const int ns = static_cast<int>(basis.shells.size());
  const int nao = basis.nao;
  std::vector<Real> V(points.size(), Real(0));
  if (points.empty() || boxes.nbox() == 0) return V;
  // Truncating the Taylor series at lloc leaves (a_B/|X_B|)^{lloc+1}, so tie the
  // separation to the tolerance the caller already gave rather than inventing a
  // second knob.
  if (!(sep > Real(0)))
    sep = far_tau > Real(0) ? std::pow(static_cast<double>(far_tau),
                                       1.0 / (lloc + 1))
                            : Real(0.3);

  // per-pair invariants: density-contracted Hermite moments, centre, exponent
  struct PairData {
    int a, b, nt;
    Real p, P[3], wpref;
    std::vector<Real> Ehat, Dblk;
  };
  std::vector<PairData> pd;
  const Real pi = pi_v<Real>();
  int ntmax = 0;
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b <= a; ++b) {
      const auto &sa = basis.shells[a], &sb = basis.shells[b];
      const int nca = ncart(sa.l), ncb = ncart(sb.l);
      const int oa = basis.ao_off[a], ob = basis.ao_off[b];
      PairData e;
      e.a = a;
      e.b = b;
      e.Dblk.assign(static_cast<std::size_t>(nca) * ncb, Real(0));
      for (int ka = 0; ka < nca; ++ka)
        for (int kb = 0; kb < ncb; ++kb) {
          const Real d = D[static_cast<std::size_t>(oa + ka) * nao + ob + kb];
          e.Dblk[ka * ncb + kb] =
              (a == b) ? d : d + D[static_cast<std::size_t>(ob + kb) * nao + oa + ka];
        }
      const auto sp = make_pair(sa, sb);
      std::vector<Real> E;
      int nt = 0;
      detail::pair_hermite_moments(sp, E, nt);
      e.nt = nt;
      ntmax = std::max(ntmax, nt);
      e.Ehat.assign(static_cast<std::size_t>(nt) * nt * nt, Real(0));
      for (int ka = 0; ka < nca; ++ka)
        for (int kb = 0; kb < ncb; ++kb) {
          const Real w = e.Dblk[ka * ncb + kb];
          if (w == Real(0)) continue;
          const std::size_t off = static_cast<std::size_t>(ka * ncb + kb) * nt * nt * nt;
          for (int i = 0; i < nt * nt * nt; ++i) e.Ehat[i] += w * E[off + i];
        }
      e.p = sa.alpha + sb.alpha;
      for (int d3 = 0; d3 < 3; ++d3) e.P[d3] = Real(sp.P[d3]);
      const Real pop = pi / e.p;
      e.wpref = pop * sqrt_(pop); // (pi/p)^{3/2}
      pd.push_back(std::move(e));
    }

  const Real far_cut = far_tau > Real(0) ? -log_(far_tau) : Real(0);
  const int nc1 = lloc + 1;
  const int Dt = ntmax + lloc; // max multipole order + 1 over all pairs
  // 1/(a! b! c!) with the (-1)^{a+b+c} folded in
  std::vector<Real> inv_fact(nc1);
  {
    Real f = 1;
    for (int k = 0; k < nc1; ++k) {
      if (k > 0) f *= Real(k);
      inv_fact[k] = Real(1) / f;
    }
  }

  Kokkos::parallel_for(
      "intti::potential_boxed",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, boxes.nbox()),
      [&](int ib) {
        const int g0 = boxes.start[ib], g1 = boxes.start[ib + 1];
        if (g1 <= g0) return;
        const auto &cB = boxes.center[ib];
        const Real aB = boxes.radius[ib];
        std::vector<Real> Lc(static_cast<std::size_t>(nc1) * nc1 * nc1, Real(0));
        std::vector<Real> T(static_cast<std::size_t>(Dt) * Dt * Dt),
            W(static_cast<std::size_t>(Dt) * Dt * Dt * Dt);
        std::vector<PointCharge<Real>> nearpts;
        std::vector<int> nearidx;
        bool haveNear = false;

        for (const auto &e : pd) {
          Real X[3], r2 = 0;
          for (int d3 = 0; d3 < 3; ++d3) {
            X[d3] = e.P[d3] - cB[d3];
            r2 += X[d3] * X[d3];
          }
          const Real r = sqrt_(r2), dmin = r - aB;
          const bool far = far_tau > Real(0) && dmin > Real(0) &&
                           e.p * dmin * dmin > far_cut && aB <= sep * r;
          if (!far) {
            if (!haveNear) {
              nearpts.reserve(g1 - g0);
              for (int g = g0; g < g1; ++g)
                nearpts.push_back({Real(1), {points[g][0], points[g][1], points[g][2]}});
              haveNear = true;
            }
            const int nb = ncart(basis.shells[e.a].l) * ncart(basis.shells[e.b].l);
            const Real *db = e.Dblk.data();
            detail::attraction_pair_visit(basis.shells[e.a], basis.shells[e.b], nearpts,
                                          grid, tau, far_tau,
                                          [&](std::size_t ic, const Real *blk) {
                                            Real s = 0;
                                            for (int i = 0; i < nb; ++i) s += db[i] * blk[i];
                                            V[g0 + ic] += s;
                                          });
            continue;
          }
          // far: one local expansion, accumulated with every other far pair
          const int nt = e.nt, Lo = (nt - 1) + lloc, Dl = Lo + 1;
          multipole_tensor(Lo, X, T.data(), W.data());
          for (int ca = 0; ca < nc1; ++ca)
            for (int cb = 0; cb + ca < nc1; ++cb)
              for (int cc = 0; cc + cb + ca < nc1; ++cc) {
                Real s = 0;
                for (int t = 0; t < nt; ++t)
                  for (int u = 0; u < nt; ++u)
                    for (int v = 0; v < nt; ++v) {
                      const Real eh = e.Ehat[(static_cast<std::size_t>(t) * nt + u) * nt + v];
                      if (eh == Real(0)) continue;
                      s += eh * T[(static_cast<std::size_t>(t + ca) * Dl + u + cb) * Dl +
                                  v + cc];
                    }
                const Real sgn = ((ca + cb + cc) & 1) ? Real(-1) : Real(1);
                Lc[(static_cast<std::size_t>(ca) * nc1 + cb) * nc1 + cc] +=
                    sgn * inv_fact[ca] * inv_fact[cb] * inv_fact[cc] * e.wpref * s;
              }
        }

        // evaluate the accumulated polynomial at this box's points
        for (int g = g0; g < g1; ++g) {
          const Real dx = points[g][0] - cB[0], dy = points[g][1] - cB[1],
                     dz = points[g][2] - cB[2];
          Real s = 0, px = 1;
          for (int ca = 0; ca < nc1; ++ca) {
            Real py = px;
            for (int cb = 0; cb + ca < nc1; ++cb) {
              Real pz = py;
              for (int cc = 0; cc + cb + ca < nc1; ++cc) {
                s += Lc[(static_cast<std::size_t>(ca) * nc1 + cb) * nc1 + cc] * pz;
                pz *= dz;
              }
              py *= dy;
            }
            px *= dx;
          }
          V[g] += s;
        }
      });
  Kokkos::fence();
  return V;
}

} // namespace intti
