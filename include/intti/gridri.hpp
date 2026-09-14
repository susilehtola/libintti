// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Grid-RI Fock builders: Coulomb (J) and exchange (K) on the hp-adaptive
// finite-element grid (roadmap M-FE grid-RI). The FE grid is a resolution of
// the identity with the grid as the auxiliary basis -- represent the density
// (J) or the co-densities (K) on the grid, apply the t-adapted DAGE to get the
// Coulomb potential DIRECTLY (no (P|Q)^{-1} fit, no auxiliary set), and contract
// against the AO products. Compared to the analytic GTO route this is fitting-
// free and metric-free; the AO enters ONLY through a pointwise evaluator, so
// higher l (the x^lx y^ly z^lz factor here; real solid harmonics later) and
// contraction are just evaluation, not a transform.
//
// This is a host reference implementation (matrix-level API: density/orbitals
// in, J/K matrix out; never per-quartet). It precomputes the AO values on the
// N^3 grid (memory nao*N^3), so it is intended for correctness/experimentation
// and small-to-moderate grids; a Kokkos/streaming port is future work. The AO
// convention is the library's unnormalized Cartesian primitive, matching
// coulomb_build/exchange_build, so results compare directly to them.

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "c2s.hpp"        // c2s_matrix (Cartesian -> real solid harmonic)
#include "contracted.hpp" // ContractedBasis, detail::effective_coeff
#include "fegrid.hpp"
#include "fock.hpp" // ShellBasis
#include "gto.hpp"  // ncart, cart_comp
#include "nuclear.hpp" // potential_on_points (analytic V on the grid)
#include "tgrid.hpp"

namespace intti {

/// Build an hp FE grid whose per-axis resolution covers every AO-product
/// envelope of `basis`: for shells i, j the product density is a Gaussian of
/// exponent alpha_i + alpha_j centred at the weighted midpoint; collect those
/// (exponent, axis-projected centre) as the 1D Gaussians to resolve to `eps`.
template <class Real>
FEGrid1D<Real> grid_for_basis(const ShellBasis<Real> &basis, Real eps = Real(1e-5),
                              int pmin = 4, int pmax = 16) {
  std::vector<FEGaussian1D<Real>> ax;
  const auto &sh = basis.shells;
  int lmax = 0;
  for (const auto &s : sh) lmax = std::max(lmax, s.l);
  for (std::size_t i = 0; i < sh.size(); ++i)
    for (std::size_t j = 0; j < sh.size(); ++j) {
      const Real p = sh[i].alpha + sh[j].alpha;
      for (int d = 0; d < 3; ++d)
        ax.push_back({p, (sh[i].alpha * sh[i].center[d] + sh[j].alpha * sh[j].center[d]) / p});
    }
  // AO products reach per-axis polynomial degree up to 2*lmax; resolve that
  // polynomial x Gaussian factor (not just the envelope) to eps.
  return make_fegrid1d_hp(ax, eps, pmin, pmax, 2 * lmax);
}

namespace detail {
template <class Real> Real gr_ipow(Real x, int n) {
  Real r = 1;
  for (int i = 0; i < n; ++i) r *= x;
  return r;
}
} // namespace detail

/// Evaluate every (unnormalized Cartesian) AO of `basis` on the 3D grid, in
/// make_basis AO order: the return value ao[a] is the N^3 tensor (row-major
/// ix,iy,iz) of AO a = ao_off[shell] + cart. General in l via cart_comp.
template <class Real>
std::vector<std::vector<Real>> ao_values_on_grid(const ShellBasis<Real> &basis,
                                                 const FEGrid1D<Real> &grid) {
  const int N = grid.N;
  const std::size_t N3 = static_cast<std::size_t>(N) * N * N;
  std::vector<std::vector<Real>> ao(basis.nao, std::vector<Real>(N3));
  const int ns = static_cast<int>(basis.shells.size());
  // per shell / axis: Gaussian and coordinate offset along each axis
  for (int s = 0; s < ns; ++s) {
    const auto &sh = basis.shells[s];
    std::vector<Real> gx(N), gy(N), gz(N), dx(N), dy(N), dz(N);
    for (int i = 0; i < N; ++i) {
      dx[i] = grid.xnode[i] - sh.center[0];
      dy[i] = grid.xnode[i] - sh.center[1];
      dz[i] = grid.xnode[i] - sh.center[2];
      gx[i] = std::exp(-sh.alpha * dx[i] * dx[i]);
      gy[i] = std::exp(-sh.alpha * dy[i] * dy[i]);
      gz[i] = std::exp(-sh.alpha * dz[i] * dz[i]);
    }
    for (int k = 0; k < ncart(sh.l); ++k) {
      int lx, ly, lz;
      cart_comp(sh.l, k, lx, ly, lz);
      Real *out = ao[basis.ao_off[s] + k].data();
      for (int ix = 0; ix < N; ++ix) {
        const Real fx = detail::gr_ipow(dx[ix], lx) * gx[ix];
        for (int iy = 0; iy < N; ++iy) {
          const Real fxy = fx * detail::gr_ipow(dy[iy], ly) * gy[iy];
          const std::size_t base = (static_cast<std::size_t>(ix) * N + iy) * N;
          for (int iz = 0; iz < N; ++iz)
            out[base + iz] = fxy * detail::gr_ipow(dz[iz], lz) * gz[iz];
        }
      }
    }
  }
  return ao;
}

namespace detail {
/// AO values on the DEVICE: ao(a, g) for AO a and row-major grid point g
/// (g = (ix*N+iy)*N+iz). Cartesian primitive shells evaluated directly on the
/// device (per-AO exponent/centre/Cartesian powers staged as Views).
template <class Real>
Kokkos::View<Real **> ao_on_grid_dev(const ShellBasis<Real> &basis, const FEGrid1D<Real> &grid) {
  const int nao = basis.nao, N = grid.N;
  const std::size_t N3 = static_cast<std::size_t>(N) * N * N;
  std::vector<Real> al(nao), cx(nao), cy(nao), cz(nao);
  std::vector<int> lx(nao), ly(nao), lz(nao);
  for (int s = 0; s < static_cast<int>(basis.shells.size()); ++s) {
    const auto &sh = basis.shells[s];
    for (int k = 0; k < ncart(sh.l); ++k) {
      int a3[3];
      cart_comp(sh.l, k, a3[0], a3[1], a3[2]);
      const int a = basis.ao_off[s] + k;
      al[a] = sh.alpha; cx[a] = sh.center[0]; cy[a] = sh.center[1]; cz[a] = sh.center[2];
      lx[a] = a3[0]; ly[a] = a3[1]; lz[a] = a3[2];
    }
  }
  auto Av = to_device(al, "gr::al"), Cx = to_device(cx, "gr::cx"),
       Cy = to_device(cy, "gr::cy"), Cz = to_device(cz, "gr::cz");
  auto Lx = to_device(lx, "gr::lx"), Ly = to_device(ly, "gr::ly"), Lz = to_device(lz, "gr::lz");
  auto xn = to_device(grid.xnode, "gr::xn");
  Kokkos::View<Real **> ao("gr::ao", nao, N3);
  Kokkos::parallel_for(
      "gr::aoeval", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nao, static_cast<int>(N3)}),
      KOKKOS_LAMBDA(int a, int g) {
        const int ix = g / (N * N), iy = (g / N) % N, iz = g % N;
        const Real dx = xn(ix) - Cx(a), dy = xn(iy) - Cy(a), dz = xn(iz) - Cz(a);
        Real v = Kokkos::exp(-Av(a) * (dx * dx + dy * dy + dz * dz));
        for (int i = 0; i < Lx(a); ++i) v *= dx;
        for (int i = 0; i < Ly(a); ++i) v *= dy;
        for (int i = 0; i < Lz(a); ++i) v *= dz;
        ao(a, g) = v;
      });
  return ao;
}

/// AO values on the DEVICE for a generally-contracted basis: each AO sums its
/// shell's primitives with the effective coefficient (basis-set coeff *
/// cart_norm_pyscf) times the Cartesian monomial. Per-AO centre/powers and a
/// flattened (alpha, ec) primitive pool are staged as Views.
template <class Real>
Kokkos::View<Real **> ao_on_grid_dev(const ContractedBasis<Real> &basis,
                                     const FEGrid1D<Real> &grid) {
  const int nao = basis.nao, N = grid.N;
  const std::size_t N3 = static_cast<std::size_t>(N) * N * N;
  std::vector<Real> cx(nao), cy(nao), cz(nao), apool, ecpool;
  std::vector<int> lx(nao), ly(nao), lz(nao), poff(nao), pn(nao);
  for (int A = 0; A < static_cast<int>(basis.shells.size()); ++A) {
    const auto &sh = basis.shells[A];
    const int nc = ncart(sh.l), np = sh.nprim();
    for (int cA = 0; cA < sh.nctr(); ++cA)
      for (int k = 0; k < nc; ++k) {
        int a3[3];
        cart_comp(sh.l, k, a3[0], a3[1], a3[2]);
        const int a = basis.ao_off[A] + cA * nc + k;
        cx[a] = sh.center[0]; cy[a] = sh.center[1]; cz[a] = sh.center[2];
        lx[a] = a3[0]; ly[a] = a3[1]; lz[a] = a3[2];
        poff[a] = static_cast<int>(apool.size());
        pn[a] = np;
        for (int p = 0; p < np; ++p) {
          apool.push_back(sh.alpha[p]);
          ecpool.push_back(effective_coeff(sh, cA, p));
        }
      }
  }
  auto Cx = to_device(cx, "gr::cx"), Cy = to_device(cy, "gr::cy"), Cz = to_device(cz, "gr::cz");
  auto Ap = to_device(apool, "gr::ap"), Ecp = to_device(ecpool, "gr::ecp");
  auto Lx = to_device(lx, "gr::lx"), Ly = to_device(ly, "gr::ly"), Lz = to_device(lz, "gr::lz");
  auto Poff = to_device(poff, "gr::poff"), Pn = to_device(pn, "gr::pn");
  auto xn = to_device(grid.xnode, "gr::xn");
  Kokkos::View<Real **> ao("gr::caop", nao, N3);
  Kokkos::parallel_for(
      "gr::aoeval_c", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nao, static_cast<int>(N3)}),
      KOKKOS_LAMBDA(int a, int g) {
        const int ix = g / (N * N), iy = (g / N) % N, iz = g % N;
        const Real dx = xn(ix) - Cx(a), dy = xn(iy) - Cy(a), dz = xn(iz) - Cz(a);
        const Real r2 = dx * dx + dy * dy + dz * dz;
        Real rad = 0;
        const int o = Poff(a), np = Pn(a);
        for (int p = 0; p < np; ++p) rad += Ecp(o + p) * Kokkos::exp(-Ap(o + p) * r2);
        for (int i = 0; i < Lx(a); ++i) rad *= dx;
        for (int i = 0; i < Ly(a); ++i) rad *= dy;
        for (int i = 0; i < Lz(a); ++i) rad *= dz;
        ao(a, g) = rad;
      });
  return ao;
}

// (the host->device AO bridge is no longer needed: contracted and spherical AOs
// are now evaluated directly on the device below.)

/// AO values on the DEVICE for the spherical (real solid harmonic) AOs of a
/// Cartesian basis: evaluate the Cartesian AOs on the device, then combine per
/// shell via the c2s matrix (a flattened (coeff, cart-AO-index) pool). Result is
/// (nao_spherical) x N^3.
template <class Real>
Kokkos::View<Real **> ao_on_grid_spherical_dev(const ShellBasis<Real> &basis,
                                               const FEGrid1D<Real> &grid) {
  const int N = grid.N;
  const std::size_t N3 = static_cast<std::size_t>(N) * N * N;
  auto cart = ao_on_grid_dev(basis, grid); // device Cartesian AOs
  int nsph = 0;
  for (const auto &s : basis.shells) nsph += 2 * s.l + 1;
  std::vector<Real> cpool;
  std::vector<int> ipool, soff(nsph), sn(nsph);
  int so = 0;
  for (int s = 0; s < static_cast<int>(basis.shells.size()); ++s) {
    const int l = basis.shells[s].l, nc = ncart(l), nm = 2 * l + 1;
    const auto C = c2s_matrix<Real>(l);
    const int co = basis.ao_off[s];
    for (int m = 0; m < nm; ++m) {
      const int a = so + m;
      soff[a] = static_cast<int>(cpool.size());
      int cnt = 0;
      for (int k = 0; k < nc; ++k) {
        const Real c = C[static_cast<std::size_t>(m) * nc + k];
        if (c != Real(0)) { cpool.push_back(c); ipool.push_back(co + k); ++cnt; }
      }
      sn[a] = cnt;
    }
    so += nm;
  }
  auto Cp = to_device(cpool, "gr::sc");
  auto Ip = to_device(ipool, "gr::si"), Soff = to_device(soff, "gr::so"), Sn = to_device(sn, "gr::sn");
  Kokkos::View<Real **> sph("gr::sph", nsph, N3);
  Kokkos::parallel_for(
      "gr::aoeval_s", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nsph, static_cast<int>(N3)}),
      KOKKOS_LAMBDA(int m, int g) {
        Real v = 0;
        const int o = Soff(m), n = Sn(m);
        for (int t = 0; t < n; ++t) v += Cp(o + t) * cart(Ip(o + t), g);
        sph(m, g) = v;
      });
  return sph;
}

/// Grid-RI Coulomb entirely on the device from a device AO View: rho(g) =
/// sum_ab D_ab ao(a,g) ao(b,g); V = DAGE(rho); J_ab = sum_g w(g) ao(a,g)
/// ao(b,g) V(g). Returns the host J matrix.
template <class Real>
std::vector<Real> grid_coulomb_dev(const Kokkos::View<Real **> &ao, int nao,
                                   const FEGrid1D<Real> &grid, const TGrid<Real> &tgrid,
                                   const Real *D, int nv) {
  const int N = grid.N;
  const std::size_t N3 = static_cast<std::size_t>(N) * N * N;
  auto Dd = to_device(D, static_cast<std::size_t>(nao) * nao, "gr::D");
  Kokkos::View<Real *> rho("gr::rho", N3);
  Kokkos::parallel_for(
      "gr::rho", N3, KOKKOS_LAMBDA(std::size_t g) {
        Real r = 0;
        for (int a = 0; a < nao; ++a) {
          Real sa = 0;
          for (int b = 0; b < nao; ++b) sa += Dd(a * nao + b) * ao(b, g);
          r += ao(a, g) * sa;
        }
        rho(g) = r;
      });
  auto V = fe_dage3d_dev(grid, tgrid, rho, nv, Real(8));
  auto xw = to_device(grid.xw, "gr::xw");
  Kokkos::View<Real **> Jd("gr::Jd", nao, nao);
  Kokkos::parallel_for(
      "gr::contractJ", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nao, nao}),
      KOKKOS_LAMBDA(int a, int b) {
        if (b < a) return;
        Real s = 0;
        for (std::size_t g = 0; g < N3; ++g) {
          const int ix = g / (N * N), iy = (g / N) % N, iz = g % N;
          s += xw(ix) * xw(iy) * xw(iz) * ao(a, g) * ao(b, g) * V(g);
        }
        Jd(a, b) = s;
        Jd(b, a) = s;
      });
  auto hJ = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Jd);
  std::vector<Real> J(static_cast<std::size_t>(nao) * nao);
  for (int a = 0; a < nao; ++a)
    for (int b = 0; b < nao; ++b) J[a * nao + b] = hJ(a, b);
  return J;
}

/// Grid-RI exchange entirely on the device from a device AO View: for each
/// occupied orbital c, phi = sum_u Cocc_uc ao(u); for each v, V_v = DAGE(ao(v)
/// phi); K_uv += sum_g w(g) ao(u,g) phi(g) V_v(g). Returns the host K matrix.
template <class Real>
std::vector<Real> grid_exchange_dev(const Kokkos::View<Real **> &ao, int nao,
                                    const Real *Cocc, int nocc, const FEGrid1D<Real> &grid,
                                    const TGrid<Real> &tgrid, int nv) {
  const int N = grid.N;
  const std::size_t N3 = static_cast<std::size_t>(N) * N * N;
  auto Cd = to_device(Cocc, static_cast<std::size_t>(nao) * nocc, "gr::C");
  auto xw = to_device(grid.xw, "gr::xw");
  Kokkos::View<Real **> Kd("gr::Kd", nao, nao);
  Kokkos::deep_copy(Kd, Real(0));
  Kokkos::View<Real *> phi("gr::phi", N3), gco("gr::gco", N3);
  for (int c = 0; c < nocc; ++c) {
    Kokkos::parallel_for(
        "gr::phi", N3, KOKKOS_LAMBDA(std::size_t g) {
          Real p = 0;
          for (int u = 0; u < nao; ++u) p += Cd(u * nocc + c) * ao(u, g);
          phi(g) = p;
        });
    for (int v = 0; v < nao; ++v) {
      Kokkos::parallel_for(
          "gr::gco", N3, KOKKOS_LAMBDA(std::size_t g) { gco(g) = ao(v, g) * phi(g); });
      auto Vv = fe_dage3d_dev(grid, tgrid, gco, nv, Real(8));
      Kokkos::parallel_for(
          "gr::contractK", nao, KOKKOS_LAMBDA(int u) {
            Real s = 0;
            for (std::size_t g = 0; g < N3; ++g) {
              const int ix = g / (N * N), iy = (g / N) % N, iz = g % N;
              s += xw(ix) * xw(iy) * xw(iz) * ao(u, g) * phi(g) * Vv(g);
            }
            Kd(u, v) += s;
          });
    }
  }
  auto hK = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Kd);
  std::vector<Real> K(static_cast<std::size_t>(nao) * nao);
  for (int a = 0; a < nao; ++a)
    for (int b = 0; b < nao; ++b) K[a * nao + b] = hK(a, b);
  return K;
}
} // namespace detail

/// The 3D grid points of an FE grid, in the row-major (ix,iy,iz) order the
/// AO and density tensors use.
template <class Real>
std::vector<std::array<Real, 3>> fegrid_points(const FEGrid1D<Real> &grid) {
  const int N = grid.N;
  std::vector<std::array<Real, 3>> pts;
  pts.reserve(static_cast<std::size_t>(N) * N * N);
  for (int ix = 0; ix < N; ++ix)
    for (int iy = 0; iy < N; ++iy)
      for (int iz = 0; iz < N; ++iz)
        pts.push_back({grid.xnode[ix], grid.xnode[iy], grid.xnode[iz]});
  return pts;
}

/// Grid-RI Coulomb matrix J over a primitive Cartesian basis (density-on-grid +
/// one DAGE). `tgrid` selects the kernel (coulomb for 1/r).
template <class Real>
std::vector<Real> grid_coulomb_build(const ShellBasis<Real> &basis, const Real *D,
                                     const FEGrid1D<Real> &grid, const TGrid<Real> &tgrid,
                                     int nv = 24) {
  return detail::grid_coulomb_dev(detail::ao_on_grid_dev(basis, grid), basis.nao, grid, tgrid, D, nv);
}

/// Grid-RI Coulomb with the POTENTIAL taken analytically instead of by DAGE.
///
/// The grid route represents the density on the grid and convolves it
/// numerically, which throws away the fact that -- for a GTO basis -- the source
/// is a sum of Gaussian products whose Coulomb potential is known in closed
/// form. potential_on_points evaluates that potential directly, and routes a
/// point well separated from a pair through the exponent-free multipole tensor
/// (far_tau), which is the cheap far field the DAGE has no equivalent of: its
/// only screen is the v-clamp, and that fires solely when |t (u1-u2)| exceeds
/// vmax across a whole element, so below t ~ 1.7 on a representative grid NO
/// element is ever skipped.
///
/// Only the final quadrature J_ab = sum_g w_g chi_a chi_b V remains a grid
/// approximation, so this is also strictly more accurate than the DAGE route at
/// the same grid -- the density no longer has to be resolved, only the AO
/// products that were going to be integrated anyway.
///
/// DAGE remains the general path: it is operator- and basis-agnostic, and a
/// density that is not a sum of Gaussian products (an NAO product, a numerical
/// orbital) has no closed-form potential to use here.
template <class Real>
std::vector<Real> grid_coulomb_build_analytic(const ShellBasis<Real> &basis,
                                              const Real *D, const FEGrid1D<Real> &grid,
                                              const TGrid<Real> &tgrid,
                                              Real tau = Real(0),
                                              Real far_tau = Real(1e-14)) {
  const int N = grid.N, nao = basis.nao;
  const std::size_t N3 = static_cast<std::size_t>(N) * N * N;
  const auto Vh = potential_on_points(basis, D, fegrid_points(grid), tgrid, tau, far_tau);
  auto V = detail::to_device(Vh, "gr::Vanalytic");
  auto ao = detail::ao_on_grid_dev(basis, grid);
  auto xw = detail::to_device(grid.xw, "gr::xw");
  Kokkos::View<Real **> Jd("gr::Jd", nao, nao);
  Kokkos::parallel_for(
      "gr::contractJ", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nao, nao}),
      KOKKOS_LAMBDA(int a, int b) {
        if (b < a) return;
        Real s = 0;
        for (std::size_t g = 0; g < N3; ++g) {
          const int ix = g / (N * N), iy = (g / N) % N, iz = g % N;
          s += xw(ix) * xw(iy) * xw(iz) * ao(a, g) * ao(b, g) * V(g);
        }
        Jd(a, b) = s;
        Jd(b, a) = s;
      });
  auto hJ = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Jd);
  std::vector<Real> J(static_cast<std::size_t>(nao) * nao);
  for (int a = 0; a < nao; ++a)
    for (int b = 0; b < nao; ++b) J[a * nao + b] = hJ(a, b);
  return J;
}

/// Grid-RI exchange matrix K over a primitive Cartesian basis, via co-densities
/// g_ui = chi_u phi_i with phi_i = sum_u Cocc[u*nocc+i] chi_u (density
/// D = Cocc Cocc^T). nocc DAGEs per orbital.
template <class Real>
std::vector<Real> grid_exchange_build(const ShellBasis<Real> &basis, const Real *Cocc,
                                      int nocc, const FEGrid1D<Real> &grid,
                                      const TGrid<Real> &tgrid, int nv = 24) {
  return detail::grid_exchange_dev(detail::ao_on_grid_dev(basis, grid), basis.nao, Cocc, nocc,
                                   grid, tgrid, nv);
}

// ---- spherical AOs: the c2s transform is FREE on the grid -------------------
// A real solid harmonic is a fixed linear combination of the Cartesian
// monomials of its degree (c2s_matrix, c2s.hpp), and cart_index matches
// cart_comp, so a spherical AO is evaluated pointwise as chi_sph_m = sum_k
// c2s(l)[m][k] chi_cart_k -- no c2s contraction on the integrals, the compact
// 2l+1 space, and no Cartesian contaminant. The grid-RI J/K then reuse the same
// detail cores on the spherical AO values.

/// Number of spherical AOs of a Cartesian ShellBasis (sum over shells of 2l+1).
template <class Real> int nao_spherical(const ShellBasis<Real> &basis) {
  int n = 0;
  for (const auto &s : basis.shells) n += 2 * s.l + 1;
  return n;
}

/// Evaluate every real-solid-harmonic (spherical) AO on the grid, shell-major
/// with m = -l..+l within each shell: chi_sph = c2s(l) . chi_cart per shell.
template <class Real>
std::vector<std::vector<Real>> ao_values_on_grid_spherical(const ShellBasis<Real> &basis,
                                                           const FEGrid1D<Real> &grid) {
  const int N = grid.N;
  const std::size_t N3 = static_cast<std::size_t>(N) * N * N;
  auto cart = ao_values_on_grid(basis, grid);
  std::vector<std::vector<Real>> sph(nao_spherical(basis), std::vector<Real>(N3, Real(0)));
  const int ns = static_cast<int>(basis.shells.size());
  int so = 0;
  for (int s = 0; s < ns; ++s) {
    const int l = basis.shells[s].l, nc = ncart(l), nm = 2 * l + 1;
    const auto C = c2s_matrix<Real>(l); // (2l+1) x nc, row-major
    const int co = basis.ao_off[s];
    for (int m = 0; m < nm; ++m) {
      Real *out = sph[so + m].data();
      for (int k = 0; k < nc; ++k) {
        const Real c = C[static_cast<std::size_t>(m) * nc + k];
        if (c == Real(0)) continue;
        const Real *cc = cart[co + k].data();
        for (std::size_t g = 0; g < N3; ++g) out[g] += c * cc[g];
      }
    }
    so += nm;
  }
  return sph;
}

/// Grid-RI Coulomb J over the SPHERICAL AOs of a Cartesian basis (D_sph and the
/// returned J are nao_spherical x nao_spherical). Matches the c2s-transform of
/// the exact Cartesian J.
template <class Real>
std::vector<Real> grid_coulomb_build_spherical(const ShellBasis<Real> &basis, const Real *D,
                                               const FEGrid1D<Real> &grid,
                                               const TGrid<Real> &tgrid, int nv = 24) {
  return detail::grid_coulomb_dev(detail::ao_on_grid_spherical_dev(basis, grid),
                                  nao_spherical(basis), grid, tgrid, D, nv);
}

/// Grid-RI exchange K over the SPHERICAL AOs (Cocc is nao_spherical x nocc).
template <class Real>
std::vector<Real> grid_exchange_build_spherical(const ShellBasis<Real> &basis, const Real *Cocc,
                                                int nocc, const FEGrid1D<Real> &grid,
                                                const TGrid<Real> &tgrid, int nv = 24) {
  return detail::grid_exchange_dev(detail::ao_on_grid_spherical_dev(basis, grid),
                                   nao_spherical(basis), Cocc, nocc, grid, tgrid, nv);
}

// ---- generally-contracted basis: contraction is FREE on the grid ------------
// The AO enters only through its pointwise value, so a contracted AO is one sum
// chi_a = sum_p ec(A,cA,p) x^lx y^ly z^lz e^{-alpha_p r^2} evaluated per point;
// the grid cost scales with nao (contracted), not nprim. The result matches the
// analytic contracted coulomb_build / exchange_build (contracted.hpp).

/// hp FE grid covering every AO-product envelope of a contracted basis: for
/// primitives p in shell A and q in shell B the product envelope has exponent
/// alpha_p + alpha_q; collect those (exponent, axis-projected centre).
template <class Real>
FEGrid1D<Real> grid_for_basis(const ContractedBasis<Real> &basis, Real eps = Real(1e-5),
                              int pmin = 4, int pmax = 16) {
  std::vector<FEGaussian1D<Real>> ax;
  const auto &sh = basis.shells;
  int lmax = 0;
  for (const auto &s : sh) lmax = std::max(lmax, s.l);
  for (std::size_t A = 0; A < sh.size(); ++A)
    for (std::size_t B = 0; B < sh.size(); ++B)
      for (int p = 0; p < sh[A].nprim(); ++p)
        for (int q = 0; q < sh[B].nprim(); ++q) {
          const Real pe = sh[A].alpha[p] + sh[B].alpha[q];
          for (int d = 0; d < 3; ++d)
            ax.push_back({pe, (sh[A].alpha[p] * sh[A].center[d] +
                               sh[B].alpha[q] * sh[B].center[d]) / pe});
        }
  // resolve the degree-up-to-2*lmax AO-product polynomial x Gaussian to eps.
  return make_fegrid1d_hp(ax, eps, pmin, pmax, 2 * lmax);
}

/// Evaluate every contracted Cartesian AO on the grid (AO order = ao_off[A] +
/// cA*ncart(l) + cart, matching contracted.hpp / the cint facade). The
/// primitive value is formed once per (shell, prim, cart) and accumulated into
/// each contracted function with its effective coefficient (basis-set coeff *
/// cart_norm_pyscf).
template <class Real>
std::vector<std::vector<Real>> ao_values_on_grid(const ContractedBasis<Real> &basis,
                                                 const FEGrid1D<Real> &grid) {
  const int N = grid.N;
  const std::size_t N3 = static_cast<std::size_t>(N) * N * N;
  std::vector<std::vector<Real>> ao(basis.nao, std::vector<Real>(N3, Real(0)));
  const int ns = static_cast<int>(basis.shells.size());
  for (int A = 0; A < ns; ++A) {
    const auto &sh = basis.shells[A];
    const int nc = ncart(sh.l), nct = sh.nctr(), npr = sh.nprim();
    std::vector<Real> gx(N), gy(N), gz(N), dx(N), dy(N), dz(N);
    for (int p = 0; p < npr; ++p) {
      const Real a = sh.alpha[p];
      for (int i = 0; i < N; ++i) {
        dx[i] = grid.xnode[i] - sh.center[0];
        dy[i] = grid.xnode[i] - sh.center[1];
        dz[i] = grid.xnode[i] - sh.center[2];
        gx[i] = std::exp(-a * dx[i] * dx[i]);
        gy[i] = std::exp(-a * dy[i] * dy[i]);
        gz[i] = std::exp(-a * dz[i] * dz[i]);
      }
      for (int k = 0; k < nc; ++k) {
        int lx, ly, lz;
        cart_comp(sh.l, k, lx, ly, lz);
        for (int cA = 0; cA < nct; ++cA) {
          const Real ec = detail::effective_coeff(sh, cA, p);
          if (ec == Real(0)) continue;
          Real *out = ao[basis.ao_off[A] + cA * nc + k].data();
          for (int ix = 0; ix < N; ++ix) {
            const Real fx = ec * detail::gr_ipow(dx[ix], lx) * gx[ix];
            for (int iy = 0; iy < N; ++iy) {
              const Real fxy = fx * detail::gr_ipow(dy[iy], ly) * gy[iy];
              const std::size_t base = (static_cast<std::size_t>(ix) * N + iy) * N;
              for (int iz = 0; iz < N; ++iz)
                out[base + iz] += fxy * detail::gr_ipow(dz[iz], lz) * gz[iz];
            }
          }
        }
      }
    }
  }
  return ao;
}

/// Grid-RI Coulomb J over a generally-contracted basis (matches
/// coulomb_build(ContractedBasis)). Contraction is absorbed into the pointwise
/// AO evaluation -- grid cost scales with nao, not nprim.
template <class Real>
std::vector<Real> grid_coulomb_build(const ContractedBasis<Real> &basis, const Real *D,
                                     const FEGrid1D<Real> &grid, const TGrid<Real> &tgrid,
                                     int nv = 24) {
  return detail::grid_coulomb_dev(detail::ao_on_grid_dev(basis, grid), basis.nao, grid, tgrid, D, nv);
}

/// Grid-RI exchange K over a generally-contracted basis (matches
/// exchange_build(ContractedBasis) on D = Cocc Cocc^T).
template <class Real>
std::vector<Real> grid_exchange_build(const ContractedBasis<Real> &basis, const Real *Cocc,
                                      int nocc, const FEGrid1D<Real> &grid,
                                      const TGrid<Real> &tgrid, int nv = 24) {
  return detail::grid_exchange_dev(detail::ao_on_grid_dev(basis, grid), basis.nao, Cocc, nocc,
                                   grid, tgrid, nv);
}

} // namespace intti
