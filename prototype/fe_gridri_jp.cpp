// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
//
// Prototype: grid-RI Coulomb (J) with p (and higher-l) shells (roadmap M-FE
// grid-RI). Extends fe_gridri_j from s-only to general Cartesian shells: the AO
// evaluator gains the x^lx y^ly z^lz polynomial factor. This is the whole cost
// of higher angular momentum on the grid -- a pointwise AO evaluation (the same
// point made for spherical/contracted AOs). Grid-RI J needs only ONE DAGE (for
// the total density), so it stays cheap. Validated against the library's exact
// GTO coulomb_build on the same unnormalized-Cartesian basis + density.
//
// Build:
//   g++ -std=c++20 -O3 -fopenmp -I include -I include/intti \
//       -isystem /usr/include/kokkos prototype/fe_gridri_jp.cpp -o /tmp/fe_gridri_jp \
//       -lkokkoscore && OMP_PROC_BIND=false /tmp/fe_gridri_jp

#include <cmath>
#include <cstdio>
#include <vector>

#include "intti/fegrid.hpp"
#include "intti/fock.hpp"   // ShellBasis, make_basis, coulomb_build
#include "intti/gto.hpp"    // ncart, cart_comp, PrimitiveShell
#include "intti/kernel.hpp"
#include "intti/tgrid.hpp"

static double ipow(double x, int n) { double r = 1; for (int i = 0; i < n; ++i) r *= x; return r; }

int main(int argc, char **argv) {
  Kokkos::ScopeGuard guard(argc, argv);

  // basis with p shells: s+p on centre A, s on centre B (unnormalized Cartesian)
  std::vector<intti::PrimitiveShell<double>> shells = {
      {2.0, {0.0, 0.0, 0.0}, 0},
      {1.3, {0.0, 0.0, 0.0}, 1},   // p shell: px, py, pz
      {1.0, {0.0, 0.0, 1.4}, 0}};
  auto sb = intti::make_basis(shells);
  const int n = sb.nao; // 1 + 3 + 1 = 5
  const int ns = (int)shells.size();

  // symmetric test density
  std::vector<double> D((std::size_t)n * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) D[i * n + j] = 0.15 + 0.25 * std::cos(0.8 * i + 1.2 * j);
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) { double a = 0.5 * (D[i * n + j] + D[j * n + i]); D[i * n + j] = D[j * n + i] = a; }

  // grid axis Gaussians from shell-pair product envelopes (exponent a_i+a_j)
  std::vector<intti::FEGaussian1D<double>> ax;
  for (int i = 0; i < ns; ++i)
    for (int j = 0; j < ns; ++j) {
      double p = shells[i].alpha + shells[j].alpha;
      for (int d = 0; d < 3; ++d)
        ax.push_back({p, (shells[i].alpha * shells[i].center[d] + shells[j].alpha * shells[j].center[d]) / p});
    }
  auto grid = intti::make_fegrid1d_hp(ax, 1e-4);
  const int N = grid.N;
  const std::size_t N3 = (std::size_t)N * N * N;

  // evaluate every Cartesian AO on the grid, in make_basis AO order
  std::vector<std::vector<double>> chi(n, std::vector<double>(N3));
  for (int s = 0; s < ns; ++s) {
    const auto &sh = shells[s];
    for (int k = 0; k < intti::ncart(sh.l); ++k) {
      int lx, ly, lz;
      intti::cart_comp(sh.l, k, lx, ly, lz);
      const int a = sb.ao_off[s] + k;
      for (int ix = 0; ix < N; ++ix)
        for (int iy = 0; iy < N; ++iy)
          for (int iz = 0; iz < N; ++iz) {
            double x = grid.xnode[ix] - sh.center[0], y = grid.xnode[iy] - sh.center[1],
                   z = grid.xnode[iz] - sh.center[2];
            chi[a][((std::size_t)ix * N + iy) * N + iz] =
                ipow(x, lx) * ipow(y, ly) * ipow(z, lz) * std::exp(-sh.alpha * (x * x + y * y + z * z));
          }
    }
  }

  // total density rho = sum_ij D_ij chi_i chi_j; one DAGE -> V
  std::vector<double> rho(N3, 0.0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      const double d = D[i * n + j];
      for (std::size_t g = 0; g < N3; ++g) rho[g] += d * chi[i][g] * chi[j][g];
    }
  intti::TGridSpec<double> tspec; tspec.n = 32;
  auto V = intti::fe_dage3d(grid, intti::make_tgrid(intti::coulomb<double>(), tspec), rho, 20);

  // grid-RI J_ij = <chi_i chi_j | V>
  std::vector<double> Jgrid((std::size_t)n * n, 0.0);
  for (int i = 0; i < n; ++i)
    for (int j = i; j < n; ++j) {
      std::vector<double> prod(N3);
      for (std::size_t g = 0; g < N3; ++g) prod[g] = chi[i][g] * chi[j][g];
      double v = intti::fe_inner(grid, prod, V);
      Jgrid[i * n + j] = Jgrid[j * n + i] = v;
    }

  // reference: exact GTO Coulomb build
  auto gj = intti::make_tgrid(intti::coulomb<double>());
  std::vector<double> Jref((std::size_t)n * n, 0.0);
  intti::coulomb_build(sb, D.data(), gj, Jref.data());

  double worst = 0, scale = 0;
  for (std::size_t k = 0; k < Jref.size(); ++k) {
    worst = std::max(worst, std::fabs(Jgrid[k] - Jref[k]));
    scale = std::max(scale, std::fabs(Jref[k]));
  }
  std::printf("grid-RI J with p shells vs exact GTO J: %d AOs (s+p+s), hp grid %d nodes/axis\n", n, N);
  std::printf("  worst |dJ| = %.3e, rel = %.3e\n", worst, worst / scale);
  const bool ok = worst / scale < 1e-2;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
