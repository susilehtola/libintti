// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Orbital-product ("fit") functions across basis-set families and their
// kernel-mediated interactions. Mathematical basis validated in
// prototype/psc_validation.py and documented in docs/psc.md.

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vector>

#include "bessel.hpp"
#include "gto.hpp"
#include "hermite1d.hpp"
#include "quartet.hpp"
#include "tgrid.hpp"

namespace intti {

/// A single Cartesian component of an analytic GTO pair product.
template <class Real> struct GTOProduct {
  ShellPair<Real> pair;
  int comp_a{0}, comp_b{0};
};

/// 2D prolate spheroidal grid for two-center products with foci at the two
/// centers: r_A = (R/2)(xi+eta), r_B = (R/2)(xi-eta),
/// dV = (R/2)^3 (xi^2-eta^2) dxi deta dphi. Gauss-Legendre nodes in
/// eta on [-1,1] and xi on [1, xi_max].
template <class Real> struct PSCGrid {
  Real focus_a[3], focus_b[3];
  Real origin[3]; ///< midpoint of the foci
  Real axis[3];   ///< unit vector a -> b
  Real e1[3], e2[3]; ///< transverse frame (deterministic from axis)
  Real R{0};
  int n_xi{0}, n_eta{0};
  std::vector<Real> zloc; ///< local z of each node (n_xi*n_eta, eta fastest)
  std::vector<Real> rho;  ///< cylindrical radius of each node
  std::vector<Real> w2d;  ///< weights incl. the Jacobian
  int nnode() const { return n_xi * n_eta; }
};

/// xi_max heuristic from the slowest exponent of the products the grid must
/// hold (validated in prototype/psc_validation.py).
template <class Real> Real psc_xi_max(Real zmin, Real R) {
  using std::sqrt;
  return 1 + 12 / sqrt(zmin * (R / 2) * (R / 2)) + 2;
}

template <class Real>
PSCGrid<Real> make_psc_grid(const Real a[3], const Real b[3], Real xi_max,
                            int n_xi = 48, int n_eta = 48) {
  using std::sqrt;
  PSCGrid<Real> g;
  Real R2 = 0;
  for (int d = 0; d < 3; ++d) {
    g.focus_a[d] = a[d];
    g.focus_b[d] = b[d];
    g.origin[d] = (a[d] + b[d]) / 2;
    R2 += (b[d] - a[d]) * (b[d] - a[d]);
  }
  g.R = sqrt(R2);
  if (g.R <= Real(0)) throw std::invalid_argument("make_psc_grid: coincident foci");
  for (int d = 0; d < 3; ++d)
    g.axis[d] = (b[d] - a[d]) / g.R;
  // deterministic transverse frame: e1 from the global axis least aligned
  // with the interfocal axis, so collinear grids share their phi origin
  {
    int k = 0;
    Real amin = std::abs(g.axis[0]);
    for (int d = 1; d < 3; ++d)
      if (std::abs(g.axis[d]) < amin) {
        amin = std::abs(g.axis[d]);
        k = d;
      }
    Real v[3] = {0, 0, 0};
    v[k] = 1;
    // e1 = normalize(v - (v.axis) axis); e2 = axis x e1
    Real dot = v[0] * g.axis[0] + v[1] * g.axis[1] + v[2] * g.axis[2];
    Real n = 0;
    for (int d = 0; d < 3; ++d) {
      g.e1[d] = v[d] - dot * g.axis[d];
      n += g.e1[d] * g.e1[d];
    }
    n = sqrt(n);
    for (int d = 0; d < 3; ++d)
      g.e1[d] /= n;
    g.e2[0] = g.axis[1] * g.e1[2] - g.axis[2] * g.e1[1];
    g.e2[1] = g.axis[2] * g.e1[0] - g.axis[0] * g.e1[2];
    g.e2[2] = g.axis[0] * g.e1[1] - g.axis[1] * g.e1[0];
  }
  g.n_xi = n_xi;
  g.n_eta = n_eta;
  std::vector<Real> xi(n_xi), wxi(n_xi), eta(n_eta), weta(n_eta);
  gauss_legendre(n_xi, Real(1), xi_max, xi.data(), wxi.data());
  gauss_legendre(n_eta, Real(-1), Real(1), eta.data(), weta.data());
  g.zloc.resize(g.nnode());
  g.rho.resize(g.nnode());
  g.w2d.resize(g.nnode());
  const Real h = g.R / 2;
  for (int i = 0; i < n_xi; ++i)
    for (int j = 0; j < n_eta; ++j) {
      const int k = i * n_eta + j;
      g.zloc[k] = h * xi[i] * eta[j];
      Real arg = (xi[i] * xi[i] - 1) * (1 - eta[j] * eta[j]);
      g.rho[k] = h * sqrt(arg > Real(0) ? arg : Real(0));
      g.w2d[k] = h * h * h * (xi[i] * xi[i] - eta[j] * eta[j]) * wxi[i] * weta[j];
    }
  return g;
}

/// Truncation point t_c that a PSC grid can support without spurious
/// diagonal capture of the delta contribution (docs/psc.md): the kernel
/// width 1/t must stay above the node spacing in the region carrying the
/// product mass. h is taken as the largest nearest-neighbor spacing in the
/// (z, rho) half-plane over the inner part of the xi range (the far tail
/// carries no mass); the prefactor is calibrated by
/// Interaction.TailCorrectionMatters.
template <class Real> Real resolution_tc(const PSCGrid<Real> &g) {
  using std::sqrt;
  Real hmax = 0;
  const int inner = g.n_xi / 4 > 2 ? g.n_xi / 4 : 2; // mass-carrying xi rows
  for (int i = 0; i + 1 < inner; ++i)
    for (int j = 0; j < g.n_eta; ++j) {
      const int k = i * g.n_eta + j, k2 = (i + 1) * g.n_eta + j;
      const Real dz = g.zloc[k] - g.zloc[k2];
      const Real dr = g.rho[k] - g.rho[k2];
      const Real h = sqrt(dz * dz + dr * dr);
      if (h > hmax) hmax = h;
    }
  for (int i = 0; i < inner; ++i)
    for (int j = 0; j + 1 < g.n_eta; ++j) {
      const int k = i * g.n_eta + j, k2 = i * g.n_eta + j + 1;
      const Real dz = g.zloc[k] - g.zloc[k2];
      const Real dr = g.rho[k] - g.rho[k2];
      const Real h = sqrt(dz * dz + dr * dr);
      if (h > hmax) hmax = h;
    }
  return Real(3) / hmax;
}

/// Two-center product on a PSC grid: complex Fourier components
/// f_m(node) = (1/(2 pi)) int chi e^{-i m phi} dphi, m = -mmax..mmax (exact
/// truncation, mmax = l_a + l_b). Stored m-major at index (m+mmax)*nnode+node.
template <class Real> struct PSCProduct {
  PSCGrid<Real> grid;
  int mmax{0};
  std::vector<Real> f_re, f_im;
};

/// Tabulate one Cartesian component of a GTO shell-pair product on a PSC
/// grid whose foci coincide with the pair's centers.
template <class Real>
PSCProduct<Real> psc_product(const ShellPair<Real> &sp, int ka, int kb,
                             const PSCGrid<Real> &grid) {
  using std::cos;
  using std::sin;
  for (int d = 0; d < 3; ++d)
    if (std::abs(grid.focus_a[d] - sp.A[d]) > Real(1e-12) ||
        std::abs(grid.focus_b[d] - sp.B[d]) > Real(1e-12))
      throw std::invalid_argument("psc_product: grid foci must be the pair centers");
  PSCProduct<Real> P;
  P.grid = grid;
  P.mmax = sp.la + sp.lb;
  const int nphi = 2 * P.mmax + 1;
  const int nn = grid.nnode();
  P.f_re.assign(static_cast<std::size_t>(2 * P.mmax + 1) * nn, Real(0));
  P.f_im.assign(static_cast<std::size_t>(2 * P.mmax + 1) * nn, Real(0));
  const Real twopi = 2 * pi_v<Real>();
  for (int k = 0; k < nphi; ++k) {
    const Real phi = twopi * k / nphi;
    const Real c = cos(phi), s = sin(phi);
    for (int i = 0; i < nn; ++i) {
      Real r[3];
      for (int d = 0; d < 3; ++d)
        r[d] = grid.origin[d] + grid.axis[d] * grid.zloc[i] +
               grid.rho[i] * (c * grid.e1[d] + s * grid.e2[d]);
      const Real chi = pair_component_value(sp, ka, kb, r);
      for (int m = -P.mmax; m <= P.mmax; ++m) {
        const Real ang = -m * phi; // f_m = (1/N) sum chi e^{-i m phi_k}
        P.f_re[(m + P.mmax) * static_cast<std::size_t>(nn) + i] += chi * cos(ang) / nphi;
        P.f_im[(m + P.mmax) * static_cast<std::size_t>(nn) + i] += chi * sin(ang) / nphi;
      }
    }
  }
  return P;
}

/// A product resolved on a Cartesian tensor grid (FEM / bubbles-and-cube).
template <class Real> struct TensorGridProduct {
  std::vector<Real> x, y, z;    ///< per-axis nodes
  std::vector<Real> wx, wy, wz; ///< per-axis weights
  std::vector<Real> values;     ///< (nx, ny, nz), z fastest
};

template <class Real>
using ProductFunction =
    std::variant<GTOProduct<Real>, PSCProduct<Real>, TensorGridProduct<Real>>;

namespace detail {

/// point cloud: positions and weight*value products
template <class Real> struct Cloud {
  std::vector<Real> pts; ///< 3*n, xyz per point
  std::vector<Real> gw;  ///< weight * value
};

template <class Real>
Cloud<Real> cloud_of(const PSCProduct<Real> &f, int nphi = 0) {
  using std::cos;
  using std::sin;
  if (nphi <= 0) nphi = std::max(6, 2 * f.mmax + 1);
  Cloud<Real> c;
  const int nn = f.grid.nnode();
  c.pts.resize(static_cast<std::size_t>(3) * nn * nphi);
  c.gw.resize(static_cast<std::size_t>(nn) * nphi);
  const Real twopi = 2 * pi_v<Real>();
  for (int k = 0; k < nphi; ++k) {
    const Real phi = twopi * k / nphi;
    const Real co = cos(phi), si = sin(phi);
    for (int i = 0; i < nn; ++i) {
      const std::size_t idx = static_cast<std::size_t>(k) * nn + i;
      for (int d = 0; d < 3; ++d)
        c.pts[3 * idx + d] = f.grid.origin[d] + f.grid.axis[d] * f.grid.zloc[i] +
                             f.grid.rho[i] * (co * f.grid.e1[d] + si * f.grid.e2[d]);
      // chi(phi) = f_0 + 2 sum_{m>0} (Re f_m cos m phi - Im f_m sin m phi)
      Real chi = f.f_re[f.mmax * static_cast<std::size_t>(nn) + i];
      for (int m = 1; m <= f.mmax; ++m) {
        const Real re = f.f_re[(m + f.mmax) * static_cast<std::size_t>(nn) + i];
        const Real im = f.f_im[(m + f.mmax) * static_cast<std::size_t>(nn) + i];
        chi += 2 * (re * cos(m * phi) - im * sin(m * phi));
      }
      c.gw[idx] = f.grid.w2d[i] * (twopi / nphi) * chi;
    }
  }
  return c;
}

template <class Real>
Cloud<Real> cloud_of(const TensorGridProduct<Real> &f, int /*nphi*/ = 0) {
  Cloud<Real> c;
  const int nx = f.x.size(), ny = f.y.size(), nz = f.z.size();
  c.pts.resize(static_cast<std::size_t>(3) * nx * ny * nz);
  c.gw.resize(static_cast<std::size_t>(nx) * ny * nz);
  std::size_t idx = 0;
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k, ++idx) {
        c.pts[3 * idx + 0] = f.x[i];
        c.pts[3 * idx + 1] = f.y[j];
        c.pts[3 * idx + 2] = f.z[k];
        c.gw[idx] = f.wx[i] * f.wy[j] * f.wz[k] * f.values[idx];
      }
  return c;
}

/// sum_i w_t(i) exp(-t_i^2 d2) accumulated over cloud x cloud point pairs;
/// also returns the plain kernel value at t = t_ref (for the tail estimate)
template <class Real>
void cloud_cloud_sums(const Cloud<Real> &a, const Cloud<Real> &b,
                      const TGrid<Real> &grid, Real t_ref, Real &quad, Real &kref) {
  const std::size_t na = a.gw.size(), nb = b.gw.size();
  const int nt = grid.n();
  quad = 0;
  kref = 0;
  if constexpr (kokkos_scalar_v<Real>) {
    Kokkos::View<Real *> pa("pa", 3 * na), pb("pb", 3 * nb), ga("ga", na), gb("gb", nb);
    Kokkos::View<Real *> tv("tv", nt), wv("wv", nt);
    auto ha = Kokkos::create_mirror_view(pa);
    auto hb = Kokkos::create_mirror_view(pb);
    auto hga = Kokkos::create_mirror_view(ga);
    auto hgb = Kokkos::create_mirror_view(gb);
    auto ht = Kokkos::create_mirror_view(tv);
    auto hw = Kokkos::create_mirror_view(wv);
    for (std::size_t i = 0; i < 3 * na; ++i) ha(i) = a.pts[i];
    for (std::size_t i = 0; i < 3 * nb; ++i) hb(i) = b.pts[i];
    for (std::size_t i = 0; i < na; ++i) hga(i) = a.gw[i];
    for (std::size_t i = 0; i < nb; ++i) hgb(i) = b.gw[i];
    for (int i = 0; i < nt; ++i) {
      ht(i) = grid.t[i];
      hw(i) = grid.w[i];
    }
    Kokkos::deep_copy(pa, ha);
    Kokkos::deep_copy(pb, hb);
    Kokkos::deep_copy(ga, hga);
    Kokkos::deep_copy(gb, hgb);
    Kokkos::deep_copy(tv, ht);
    Kokkos::deep_copy(wv, hw);
    Real q = 0, kr = 0;
    Kokkos::parallel_reduce(
        "intti::interaction::cloud2",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0},
                                               {static_cast<long>(na), static_cast<long>(nb)}),
        KOKKOS_LAMBDA(long i, long j, Real &acc, Real &acck) {
          Real d2 = 0;
          for (int d = 0; d < 3; ++d) {
            const Real dd = pa(3 * i + d) - pb(3 * j + d);
            d2 += dd * dd;
          }
          const Real gg = ga(i) * gb(j);
          Real s = 0;
          for (int it = 0; it < nt; ++it)
            s += wv(it) * exp_(-tv(it) * tv(it) * d2);
          acc += gg * s;
          acck += gg * exp_(-t_ref * t_ref * d2);
        },
        q, kr);
    quad = q;
    kref = kr;
  } else {
    for (std::size_t i = 0; i < na; ++i)
      for (std::size_t j = 0; j < nb; ++j) {
        Real d2 = 0;
        for (int d = 0; d < 3; ++d) {
          const Real dd = a.pts[3 * i + d] - b.pts[3 * j + d];
          d2 += dd * dd;
        }
        const Real gg = a.gw[i] * b.gw[j];
        Real s = 0;
        for (int it = 0; it < nt; ++it)
          s += grid.w[it] * exp_(-grid.t[it] * grid.t[it] * d2);
        quad += gg * s;
        kref += gg * exp_(-t_ref * t_ref * d2);
      }
  }
}

/// analytic GTO product x cloud: per t and cloud point the bra integral is
/// prod_d sqrt(pi/(p+t^2)) sum_n E_n^d B_n(theta, P_d - r_d),
/// theta = p t^2/(p + t^2); the quadrature sum and the exact overlap
/// (pair value on the cloud) come out together.
template <class Real>
void gto_cloud_sums(const GTOProduct<Real> &f, const Cloud<Real> &b,
                    const TGrid<Real> &grid, Real &quad, Real &overlap) {
  const auto &sp = f.pair;
  int a3[3], b3[3];
  cart_comp(sp.la, f.comp_a, a3[0], a3[1], a3[2]);
  cart_comp(sp.lb, f.comp_b, b3[0], b3[1], b3[2]);
  // E tables per direction
  const int esz = (sp.la + 1) * (sp.lb + 1) * (sp.la + sp.lb + 1);
  std::vector<Real> E(3 * esz);
  for (int d = 0; d < 3; ++d)
    e_coeffs(sp.la, sp.lb, sp.p, sp.P[d] - sp.A[d], sp.P[d] - sp.B[d], sp.K[d],
             E.data() + d * esz);
  const int nE = sp.la + sp.lb + 1;
  std::vector<Real> Erow(3 * nE);
  int nrow[3];
  for (int d = 0; d < 3; ++d) {
    nrow[d] = a3[d] + b3[d] + 1;
    for (int n = 0; n < nrow[d]; ++n)
      Erow[d * nE + n] = E[d * esz + (a3[d] * (sp.lb + 1) + b3[d]) * (sp.la + sp.lb + 1) + n];
  }
  const std::size_t nb = b.gw.size();
  const int nt = grid.n();
  const Real p = sp.p;
  const Real pi = pi_v<Real>();
  quad = 0;
  overlap = 0;
  // serial host loop; FP parallelization of this path lands with M4 batching
  std::vector<Real> B(nrow[0] + nrow[1] + nrow[2]);
  for (std::size_t j = 0; j < nb; ++j) {
    const Real r[3] = {b.pts[3 * j], b.pts[3 * j + 1], b.pts[3 * j + 2]};
    Real s = 0;
    for (int it = 0; it < nt; ++it) {
      const Real t = grid.t[it];
      const Real denom = p + t * t;
      const Real theta = p * t * t / denom;
      const Real pref = sqrt_(pi / denom);
      Real val = 1;
      for (int d = 0; d < 3; ++d) {
        hermite_b(nrow[d] - 1, theta, sp.P[d] - r[d], B.data());
        Real acc = 0;
        for (int n = 0; n < nrow[d]; ++n)
          acc += Erow[d * nE + n] * B[n];
        val *= pref * acc;
      }
      s += grid.w[it] * val;
    }
    quad += b.gw[j] * s;
    overlap += b.gw[j] * pair_component_value(sp, f.comp_a, f.comp_b, r);
  }
}

/// coaxiality: parallel axes (same orientation) and collinear origins
template <class Real>
bool coaxial(const PSCGrid<Real> &g1, const PSCGrid<Real> &g2) {
  Real dot = 0;
  for (int d = 0; d < 3; ++d)
    dot += g1.axis[d] * g2.axis[d];
  if (dot < Real(1) - Real(1e-12)) return false;
  Real off[3], proj = 0;
  for (int d = 0; d < 3; ++d) {
    off[d] = g2.origin[d] - g1.origin[d];
    proj += off[d] * g1.axis[d];
  }
  Real lat2 = 0;
  for (int d = 0; d < 3; ++d) {
    const Real l = off[d] - proj * g1.axis[d];
    lat2 += l * l;
  }
  return lat2 < Real(1e-24);
}

/// coaxial PSC x PSC: m-diagonal Bessel coupling, summed over grid-point
/// pairs and t nodes; also returns the kernel value at t_ref for the tail
template <class Real>
void psc_coaxial_sums(const PSCProduct<Real> &f, const PSCProduct<Real> &g,
                      const TGrid<Real> &grid, Real t_ref, Real &quad, Real &kref) {
  const int nn1 = f.grid.nnode(), nn2 = g.grid.nnode();
  const int mm = std::min(f.mmax, g.mmax);
  const int nt = grid.n();
  // global axial coordinate of each node
  Real zoff1 = 0, zoff2 = 0;
  for (int d = 0; d < 3; ++d) {
    zoff1 += f.grid.origin[d] * f.grid.axis[d];
    zoff2 += g.grid.origin[d] * f.grid.axis[d];
  }
  const Real fourpi2 = 4 * pi_v<Real>() * pi_v<Real>();
  // per-m pair coefficient: c_m = f_0 g_0 (m=0), 2(Re f_m Re g_m + Im f_m Im g_m)
  // (m>0), using f_{-m} = conj(f_m) for real products
  auto cm_of = [&](int i, int j, int m) {
    const std::size_t k1 = (m + f.mmax) * static_cast<std::size_t>(nn1) + i;
    const std::size_t k2 = (m + g.mmax) * static_cast<std::size_t>(nn2) + j;
    const Real re = f.f_re[k1] * g.f_re[k2] + f.f_im[k1] * g.f_im[k2];
    return m == 0 ? re : 2 * re;
  };
  quad = 0;
  kref = 0;
  if constexpr (kokkos_scalar_v<Real>) {
    const int mmax1 = f.mmax, mmax2 = g.mmax;
    Kokkos::View<Real *> z1v("z1", nn1), r1v("r1", nn1), w1v("w1", nn1);
    Kokkos::View<Real *> z2v("z2", nn2), r2v("r2", nn2), w2v("w2", nn2);
    Kokkos::View<Real **, Kokkos::LayoutLeft> F1re("F1re", nn1, 2 * mmax1 + 1),
        F1im("F1im", nn1, 2 * mmax1 + 1), F2re("F2re", nn2, 2 * mmax2 + 1),
        F2im("F2im", nn2, 2 * mmax2 + 1);
    Kokkos::View<Real *> tv("tv", nt), wv("wv", nt);
    {
      auto hz1 = Kokkos::create_mirror_view(z1v);
      auto hr1 = Kokkos::create_mirror_view(r1v);
      auto hw1 = Kokkos::create_mirror_view(w1v);
      auto hz2 = Kokkos::create_mirror_view(z2v);
      auto hr2 = Kokkos::create_mirror_view(r2v);
      auto hw2 = Kokkos::create_mirror_view(w2v);
      auto h1re = Kokkos::create_mirror_view(F1re);
      auto h1im = Kokkos::create_mirror_view(F1im);
      auto h2re = Kokkos::create_mirror_view(F2re);
      auto h2im = Kokkos::create_mirror_view(F2im);
      auto ht = Kokkos::create_mirror_view(tv);
      auto hw = Kokkos::create_mirror_view(wv);
      for (int i = 0; i < nn1; ++i) {
        hz1(i) = zoff1 + f.grid.zloc[i];
        hr1(i) = f.grid.rho[i];
        hw1(i) = f.grid.w2d[i];
        for (int k = 0; k < 2 * mmax1 + 1; ++k) {
          h1re(i, k) = f.f_re[k * static_cast<std::size_t>(nn1) + i];
          h1im(i, k) = f.f_im[k * static_cast<std::size_t>(nn1) + i];
        }
      }
      for (int j = 0; j < nn2; ++j) {
        hz2(j) = zoff2 + g.grid.zloc[j];
        hr2(j) = g.grid.rho[j];
        hw2(j) = g.grid.w2d[j];
        for (int k = 0; k < 2 * mmax2 + 1; ++k) {
          h2re(j, k) = g.f_re[k * static_cast<std::size_t>(nn2) + j];
          h2im(j, k) = g.f_im[k * static_cast<std::size_t>(nn2) + j];
        }
      }
      for (int it = 0; it < nt; ++it) {
        ht(it) = grid.t[it];
        hw(it) = grid.w[it];
      }
      Kokkos::deep_copy(z1v, hz1);
      Kokkos::deep_copy(r1v, hr1);
      Kokkos::deep_copy(w1v, hw1);
      Kokkos::deep_copy(z2v, hz2);
      Kokkos::deep_copy(r2v, hr2);
      Kokkos::deep_copy(w2v, hw2);
      Kokkos::deep_copy(F1re, h1re);
      Kokkos::deep_copy(F1im, h1im);
      Kokkos::deep_copy(F2re, h2re);
      Kokkos::deep_copy(F2im, h2im);
      Kokkos::deep_copy(tv, ht);
      Kokkos::deep_copy(wv, hw);
    }
    Real q = 0, kr = 0;
    Kokkos::parallel_reduce(
        "intti::interaction::coaxial",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nn1, nn2}),
        KOKKOS_LAMBDA(int i, int j, Real &acc, Real &acck) {
          const Real dz = z1v(i) - z2v(j);
          const Real dr = r1v(i) - r2v(j);
          const Real sep2 = dz * dz + dr * dr;
          const Real rr2 = 2 * r1v(i) * r2v(j);
          const Real wij = w1v(i) * w2v(j);
          Real cm[2 * LMAX + 1];
          for (int m = 0; m <= mm; ++m) {
            const Real re = F1re(i, m + mmax1) * F2re(j, m + mmax2) +
                            F1im(i, m + mmax1) * F2im(j, m + mmax2);
            cm[m] = m == 0 ? re : 2 * re;
          }
          Real iv[2 * LMAX + 1];
          Real sq = 0;
          for (int it = 0; it < nt; ++it) {
            const Real t2 = tv(it) * tv(it);
            ive_ladder(mm, t2 * rr2, iv);
            Real chan = 0;
            for (int m = 0; m <= mm; ++m)
              chan += cm[m] * iv[m];
            sq += wv(it) * exp_(-t2 * sep2) * chan;
          }
          acc += wij * sq;
          {
            const Real t2 = t_ref * t_ref;
            ive_ladder(mm, t2 * rr2, iv);
            Real chan = 0;
            for (int m = 0; m <= mm; ++m)
              chan += cm[m] * iv[m];
            acck += wij * exp_(-t2 * sep2) * chan;
          }
        },
        q, kr);
    quad = fourpi2 * q;
    kref = fourpi2 * kr;
  } else {
    std::vector<Real> iv(mm + 1);
    for (int i = 0; i < nn1; ++i) {
      const Real z1 = zoff1 + f.grid.zloc[i];
      const Real r1 = f.grid.rho[i];
      const Real w1 = f.grid.w2d[i];
      for (int j = 0; j < nn2; ++j) {
        const Real dz = z1 - (zoff2 + g.grid.zloc[j]);
        const Real dr = r1 - g.grid.rho[j];
        const Real sep2 = dz * dz + dr * dr;
        const Real rr2 = 2 * r1 * g.grid.rho[j];
        const Real wij = w1 * g.grid.w2d[j];
        Real cm[2 * LMAX + 1];
        for (int m = 0; m <= mm; ++m)
          cm[m] = cm_of(i, j, m);
        Real sq = 0;
        for (int it = 0; it < nt; ++it) {
          const Real t2 = grid.t[it] * grid.t[it];
          ive_ladder(mm, t2 * rr2, iv.data());
          Real chan = 0;
          for (int m = 0; m <= mm; ++m)
            chan += cm[m] * iv[m];
          sq += grid.w[it] * exp_(-t2 * sep2) * chan;
        }
        quad += wij * sq;
        {
          const Real t2 = t_ref * t_ref;
          ive_ladder(mm, t2 * rr2, iv.data());
          Real chan = 0;
          for (int m = 0; m <= mm; ++m)
            chan += cm[m] * iv[m];
          kref += wij * exp_(-t2 * sep2) * chan;
        }
      }
    }
    quad *= fourpi2;
    kref *= fourpi2;
  }
}

template <class Real> bool is_grid_rep(const ProductFunction<Real> &f) {
  return !std::holds_alternative<GTOProduct<Real>>(f);
}

} // namespace detail

/// Kernel-mediated interaction (f | kernel | g) on the shared t grid.
/// Analytic GTO x GTO uses the quartet machinery; coaxial PSC pairs use the
/// m-diagonal Bessel coupling; every other pairing goes through uniform-phi
/// point clouds. Grid-represented products require a truncated (LinLog)
/// TGrid; the truncated tail is restored with the delta-function correction,
/// with the overlap taken exactly (GTO x grid) or estimated self-consistently
/// from the kernel value at t_c (see docs/psc.md).
///
/// Tail order: GTO x GTO goes through eri_quartet and so inherits the FULL
/// higher-order tail series (grid.tail_order, tgrid.hpp). The grid-represented
/// paths use only the leading (order-0) delta term pi*S/t_c^2 -- higher orders
/// need the Laplacian-overlap moments M_k = int rho_f (nabla^2)^k rho_g, which
/// are feasible for GTO x cloud (the Laplacian lands on the analytic GTO,
/// evaluated at the cloud points) but not for cloud x cloud / coaxial PSC, where
/// even the overlap S is only estimated from the kernel value at t_c. So
/// tail_order > 0 is honoured for GTO x GTO and ignored by the grid branches.
template <class Real>
Real interaction(const ProductFunction<Real> &f, const ProductFunction<Real> &g,
                 const TGrid<Real> &grid) {
  using detail::Cloud;
  // GTO x GTO: quartet machinery
  if (std::holds_alternative<GTOProduct<Real>>(f) &&
      std::holds_alternative<GTOProduct<Real>>(g)) {
    const auto &a = std::get<GTOProduct<Real>>(f);
    const auto &b = std::get<GTOProduct<Real>>(g);
    const int ncb = ncart(a.pair.lb), ncc = ncart(b.pair.la), ncd = ncart(b.pair.lb);
    std::vector<Real> buf(ncart(a.pair.la) * ncb * ncc * ncd);
    eri_quartet(a.pair, b.pair, grid, buf.data());
    return buf[((a.comp_a * ncb + a.comp_b) * ncc + b.comp_a) * ncd + b.comp_b];
  }
  // grid-represented products need the truncated grid + tail (docs/psc.md)
  if ((detail::is_grid_rep(f) || detail::is_grid_rep(g)) &&
      grid.tail_coeff == Real(0) && grid.t_c == Real(0))
    throw std::invalid_argument(
        "interaction: grid-represented products require a truncated (LinLog) "
        "t grid with the delta-function tail; see docs/psc.md");

  const Real pi = pi_v<Real>();
  const Real tc = grid.t_c;
  const Real s_from_kref = tc > Real(0) ? tc * tc * tc / (pi * sqrt_(pi)) : Real(0);

  // coaxial PSC x PSC: Bessel path
  if (std::holds_alternative<PSCProduct<Real>>(f) &&
      std::holds_alternative<PSCProduct<Real>>(g)) {
    const auto &a = std::get<PSCProduct<Real>>(f);
    const auto &b = std::get<PSCProduct<Real>>(g);
    if (detail::coaxial(a.grid, b.grid)) {
      Real quad, kref;
      detail::psc_coaxial_sums(a, b, grid, tc, quad, kref);
      return quad + grid.tail_coeff * s_from_kref * kref;
    }
  }
  // GTO x cloud (either order)
  if (std::holds_alternative<GTOProduct<Real>>(f) ||
      std::holds_alternative<GTOProduct<Real>>(g)) {
    const bool f_is_gto = std::holds_alternative<GTOProduct<Real>>(f);
    const auto &gp = std::get<GTOProduct<Real>>(f_is_gto ? f : g);
    const auto &other = f_is_gto ? g : f;
    Cloud<Real> c = std::holds_alternative<PSCProduct<Real>>(other)
                        ? detail::cloud_of(std::get<PSCProduct<Real>>(other))
                        : detail::cloud_of(std::get<TensorGridProduct<Real>>(other));
    Real quad, overlap;
    detail::gto_cloud_sums(gp, c, grid, quad, overlap);
    return quad + grid.tail_coeff * overlap;
  }
  // cloud x cloud
  auto cloud = [](const ProductFunction<Real> &pf) {
    if (std::holds_alternative<PSCProduct<Real>>(pf))
      return detail::cloud_of(std::get<PSCProduct<Real>>(pf));
    return detail::cloud_of(std::get<TensorGridProduct<Real>>(pf));
  };
  Cloud<Real> ca = cloud(f), cb = cloud(g);
  Real quad, kref;
  detail::cloud_cloud_sums(ca, cb, grid, tc, quad, kref);
  return quad + grid.tail_coeff * s_from_kref * kref;
}

} // namespace intti
