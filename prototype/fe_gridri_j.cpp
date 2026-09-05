// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
//
// Prototype: grid-RI Coulomb (J) on a molecular GTO basis (roadmap M-FE grid-RI
// payoff). The FE route IS resolution-of-the-identity with the FE grid as the
// auxiliary basis: represent the total density rho(r) = sum_uv D_uv chi_u chi_v
// on the hp-adaptive tensorial grid, apply the t-adapted DAGE to get its Coulomb
// potential V(r) DIRECTLY (no (P|Q)^{-1} fit), and contract J_uv = <chi_u chi_v
// | V>. Validated against the library's exact GTO coulomb_build on the same
// (unnormalized-primitive) basis + density: grid-RI J -> exact GTO J as the grid
// resolves. s-shell basis (clean first demo; p+ needs the x^l factors on grid).
//
// Build:
//   g++ -std=c++20 -O3 -fopenmp -I include -I include/intti \
//       -isystem /usr/include/kokkos prototype/fe_gridri_j.cpp -o /tmp/fe_gridri_j \
//       -lkokkoscore && OMP_PROC_BIND=false /tmp/fe_gridri_j

#include <cmath>
#include <cstdio>
#include <vector>

#include "intti/fegrid.hpp"
#include "intti/fock.hpp"   // ShellBasis, make_basis, coulomb_build
#include "intti/kernel.hpp"
#include "intti/tgrid.hpp"

struct SPrim { double alpha, A[3]; }; // an s primitive exp(-alpha |r-A|^2)

int main(int argc, char **argv) {
  Kokkos::ScopeGuard guard(argc, argv);

  // small molecular s-basis: two centres, tight + diffuse on each
  std::vector<SPrim> bas = {
      {3.0, {0.0, 0.0, 0.0}}, {0.8, {0.0, 0.0, 0.0}},
      {2.5, {0.0, 0.0, 1.6}}, {0.7, {0.0, 0.0, 1.6}}};
  const int n = (int)bas.size();

  // a symmetric test density
  std::vector<double> D((std::size_t)n * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) D[i * n + j] = 0.2 + 0.3 * std::cos(0.9 * i + 1.1 * j);
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) { double a = 0.5 * (D[i * n + j] + D[j * n + i]); D[i * n + j] = D[j * n + i] = a; }

  // grid axis Gaussians: the product densities chi_i chi_j are Gaussians with
  // exponent a_i+a_j centred at (a_i A_i + a_j A_j)/(a_i+a_j) -- the density's
  // constituents. Collect their projections on every axis (isotropic grid).
  std::vector<intti::FEGaussian1D<double>> ax;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double p = bas[i].alpha + bas[j].alpha;
      for (int d = 0; d < 3; ++d) {
        double P = (bas[i].alpha * bas[i].A[d] + bas[j].alpha * bas[j].A[d]) / p;
        ax.push_back({p, P});
      }
    }
  auto grid = intti::make_fegrid1d_hp(ax, 1e-3);
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

  // total density rho = sum_ij D_ij chi_i chi_j
  std::vector<double> rho(N3, 0.0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      const double d = D[i * n + j];
      for (std::size_t g = 0; g < N3; ++g) rho[g] += d * chi[i][g] * chi[j][g];
    }

  // DAGE: V = Coulomb potential of rho on the grid
  intti::TGridSpec<double> tspec; tspec.n = 24;
  auto tg = intti::make_tgrid(intti::coulomb<double>(), tspec);
  auto V = intti::fe_dage3d(grid, tg, rho, 16);

  // grid-RI J_ij = <chi_i chi_j | V>
  std::vector<double> Jgrid((std::size_t)n * n, 0.0);
  for (int i = 0; i < n; ++i)
    for (int j = i; j < n; ++j) {
      std::vector<double> prod(N3);
      for (std::size_t g = 0; g < N3; ++g) prod[g] = chi[i][g] * chi[j][g];
      double v = intti::fe_inner(grid, prod, V);
      Jgrid[i * n + j] = Jgrid[j * n + i] = v;
    }

  // reference: exact GTO Coulomb build on the same unnormalized s-primitives
  std::vector<intti::PrimitiveShell<double>> shells;
  for (auto &b : bas) shells.push_back({b.alpha, {b.A[0], b.A[1], b.A[2]}, 0});
  auto sb = intti::make_basis(shells);
  auto gj = intti::make_tgrid(intti::coulomb<double>());
  std::vector<double> Jref((std::size_t)n * n, 0.0);
  intti::coulomb_build(sb, D.data(), gj, Jref.data());

  double worst = 0, scale = 0;
  for (std::size_t k = 0; k < Jref.size(); ++k) {
    worst = std::max(worst, std::fabs(Jgrid[k] - Jref[k]));
    scale = std::max(scale, std::fabs(Jref[k]));
  }
  std::printf("grid-RI J vs exact GTO J: %d AOs, hp grid %d nodes/axis (%lld pts)\n",
              n, N, (long long)N3);
  std::printf("  worst |dJ| = %.3e, rel = %.3e\n", worst, worst / scale);
  for (int i = 0; i < n; ++i) {
    std::printf("  J_grid/J_ref row %d:", i);
    for (int j = 0; j < n; ++j) std::printf(" %.5f/%.5f", Jgrid[i * n + j], Jref[i * n + j]);
    std::printf("\n");
  }
  const bool ok = worst / scale < 5e-2;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
