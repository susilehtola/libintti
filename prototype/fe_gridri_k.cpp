// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
//
// Prototype: grid-RI exchange (K) via co-densities (roadmap M-FE grid-RI). The
// exchange matrix is K_uv = sum_i (ui|vi), where i runs over occupied orbitals
// and (ui|vi) is the Coulomb interaction of the CO-DENSITIES g_ui = chi_u phi_i
// and g_vi = chi_v phi_i. On the FE grid: for each occupied orbital i form the
// co-densities g_vi(r) = chi_v(r) phi_i(r), DAGE each to V_vi(r), and contract
// K_uv = sum_i <g_ui | V_vi>. Same machinery as grid-RI J (density-on-grid +
// DAGE), no aux fit / no (P|Q)^{-1}. The kernel is the same t-quadrature.
//
// Density D = M M^T (rank = ncols(M)), so the occupied orbitals are simply the
// columns of M (phi_c = sum_u M_uc chi_u) and K_uv = sum_c (uc|vc) -- matching
// the library's exact GTO exchange_build(D). s-shell basis (clean first demo).
//
// Build:
//   g++ -std=c++20 -O3 -fopenmp -I include -I include/intti \
//       -isystem /usr/include/kokkos prototype/fe_gridri_k.cpp -o /tmp/fe_gridri_k \
//       -lkokkoscore && OMP_PROC_BIND=false /tmp/fe_gridri_k

#include <cmath>
#include <cstdio>
#include <vector>

#include "intti/fegrid.hpp"
#include "intti/fock.hpp"   // ShellBasis, make_basis, exchange_build
#include "intti/kernel.hpp"
#include "intti/tgrid.hpp"

struct SPrim { double alpha, A[3]; };

int main(int argc, char **argv) {
  Kokkos::ScopeGuard guard(argc, argv);

  std::vector<SPrim> bas = {
      {3.0, {0.0, 0.0, 0.0}}, {1.0, {0.0, 0.0, 0.0}},
      {2.5, {0.0, 0.0, 1.5}}, {0.9, {0.0, 0.0, 1.5}}};
  const int n = (int)bas.size();
  const int nocc = 1; // rank-1 density (D = v v^T): 4 DAGE, keeps it fast

  // occupied orbitals M (n x nocc); density D = M M^T
  std::vector<double> M((std::size_t)n * nocc);
  for (int u = 0; u < n; ++u)
    for (int c = 0; c < nocc; ++c) M[u * nocc + c] = 0.3 + 0.25 * std::sin(1.3 * u + 0.7 * c);
  std::vector<double> D((std::size_t)n * n, 0.0);
  for (int u = 0; u < n; ++u)
    for (int v = 0; v < n; ++v)
      for (int c = 0; c < nocc; ++c) D[u * n + v] += M[u * nocc + c] * M[v * nocc + c];

  // grid axis Gaussians: co-densities chi_v phi_c are sums of product Gaussians
  // (exponent a_u+a_v). Cover all AO-product pairs' projections (isotropic grid).
  std::vector<intti::FEGaussian1D<double>> ax;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double p = bas[i].alpha + bas[j].alpha;
      for (int d = 0; d < 3; ++d)
        ax.push_back({p, (bas[i].alpha * bas[i].A[d] + bas[j].alpha * bas[j].A[d]) / p});
    }
  auto grid = intti::make_fegrid1d_hp(ax, 3e-2);
  const int N = grid.N;
  const std::size_t N3 = (std::size_t)N * N * N;

  // AO values on the grid
  std::vector<std::vector<double>> chi(n, std::vector<double>(N3));
  for (int i = 0; i < n; ++i)
    for (int ix = 0; ix < N; ++ix)
      for (int iy = 0; iy < N; ++iy)
        for (int iz = 0; iz < N; ++iz) {
          double x = grid.xnode[ix] - bas[i].A[0], y = grid.xnode[iy] - bas[i].A[1],
                 z = grid.xnode[iz] - bas[i].A[2];
          chi[i][((std::size_t)ix * N + iy) * N + iz] = std::exp(-bas[i].alpha * (x * x + y * y + z * z));
        }

  intti::TGridSpec<double> tspec; tspec.n = 16;
  auto tg = intti::make_tgrid(intti::coulomb<double>(), tspec);

  std::vector<double> K((std::size_t)n * n, 0.0);
  for (int c = 0; c < nocc; ++c) {
    // phi_c on the grid
    std::vector<double> phi(N3, 0.0);
    for (int u = 0; u < n; ++u) {
      const double m = M[u * nocc + c];
      for (std::size_t g = 0; g < N3; ++g) phi[g] += m * chi[u][g];
    }
    // co-densities g_v = chi_v phi_c, and their potentials V_v = DAGE(g_v)
    std::vector<std::vector<double>> gco(n, std::vector<double>(N3)), Vco(n);
    for (int v = 0; v < n; ++v) {
      for (std::size_t g = 0; g < N3; ++g) gco[v][g] = chi[v][g] * phi[g];
      Vco[v] = intti::fe_dage3d(grid, tg, gco[v], 16);
    }
    // K_uv += <g_u | V_v>
    for (int u = 0; u < n; ++u)
      for (int v = u; v < n; ++v) {
        double kv = intti::fe_inner(grid, gco[u], Vco[v]);
        K[u * n + v] += kv;
        if (u != v) K[v * n + u] += kv;
      }
  }

  // reference: exact GTO exchange on the same unnormalized s-primitives
  std::vector<intti::PrimitiveShell<double>> shells;
  for (auto &b : bas) shells.push_back({b.alpha, {b.A[0], b.A[1], b.A[2]}, 0});
  auto sb = intti::make_basis(shells);
  auto gj = intti::make_tgrid(intti::coulomb<double>());
  std::vector<double> Kref((std::size_t)n * n, 0.0);
  intti::exchange_build(sb, D.data(), gj, Kref.data(), 0.0);

  double worst = 0, scale = 0;
  for (std::size_t k = 0; k < Kref.size(); ++k) {
    worst = std::max(worst, std::fabs(K[k] - Kref[k]));
    scale = std::max(scale, std::fabs(Kref[k]));
  }
  std::printf("grid-RI K vs exact GTO K: %d AOs, %d occ, hp grid %d nodes/axis\n", n, nocc, N);
  std::printf("  worst |dK| = %.3e, rel = %.3e\n", worst, worst / scale);
  for (int u = 0; u < n; ++u) {
    std::printf("  K_grid/K_ref row %d:", u);
    for (int v = 0; v < n; ++v) std::printf(" %.4f/%.4f", K[u * n + v], Kref[u * n + v]);
    std::printf("\n");
  }
  const bool ok = worst / scale < 1.5e-1;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
