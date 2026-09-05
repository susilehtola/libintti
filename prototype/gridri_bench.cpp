// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
//
// Benchmark: grid-RI (finite-element) J vs the analytic t-quadrature engine
// (coulomb_build), on the RIGHT axis for RI -- FIXED geometry, GROWING basis.
// A rich, high-angular-momentum basis on a fixed molecule makes the AO product
// space {chi_u chi_v} massively linearly dependent, which is exactly where RI
// wins: the analytic engine pays growing nao and expensive high-l quartets,
// while grid-RI is insensitive to the product-space redundancy -- the density
// rho(r) is one smooth function, the grid that resolves it is set by the
// molecule's spatial extent + exponent range (~independent of nao), and J is a
// single DAGE (fixed) plus a contraction. (An earlier version grew the molecule,
// so the grid grew with it -- the wrong axis, which hid the RI advantage.)
//
// Two fixed atoms; level maxl puts shells l = 0..maxl (two exponents each,
// tight+diffuse) on each atom. J only (one DAGE regardless of nao -- the clean
// scaling demonstration; K would add nocc*nao DAGEs).
//
// Build:
//   g++ -std=c++20 -O3 -fopenmp -I include -I include/intti \
//       -isystem /usr/include/kokkos prototype/gridri_bench.cpp -o /tmp/gridri_bench \
//       -lkokkoscore && OMP_PROC_BIND=false /tmp/gridri_bench

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "intti/fock.hpp"
#include "intti/gridri.hpp"
#include "intti/kernel.hpp"
#include "intti/tgrid.hpp"

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char **argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  const double eps = 1.5e-2;
  const double zs[2] = {0.0, 1.4};
  const double expo[2] = {3.0, 0.8}; // tight + diffuse per l
  std::printf("fixed 2-atom geometry, growing angular momentum (J only)\n");
  std::printf("%-5s %-5s %-6s  %10s %10s %8s   %8s\n", "maxl", "nao", "gridN",
              "J_ana(s)", "J_grid(s)", "grid/ana", "Jrel");
  for (int maxl = 0; maxl <= 3; ++maxl) {
    std::vector<intti::PrimitiveShell<double>> shells;
    for (double z : zs)
      for (int l = 0; l <= maxl; ++l)
        for (double a : expo) shells.push_back({a, {0.0, 0.0, z}, l});
    auto sb = intti::make_basis(shells);
    const int n = sb.nao;
    std::vector<double> D((std::size_t)n * n);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) D[i * n + j] = 0.1 + 0.2 * std::cos(0.7 * i + 1.1 * j);
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j) { double m = 0.5*(D[i*n+j]+D[j*n+i]); D[i*n+j]=D[j*n+i]=m; }

    auto gj = intti::make_tgrid(intti::coulomb<double>());
    intti::TGridSpec<double> spec; spec.n = 20;
    auto gtg = intti::make_tgrid(intti::coulomb<double>(), spec);
    auto grid = intti::grid_for_basis(sb, eps);

    std::vector<double> Ja((std::size_t)n * n, 0.0);
    auto t0 = clk::now();
    intti::coulomb_build(sb, D.data(), gj, Ja.data());
    auto t1 = clk::now();
    auto Jg = intti::grid_coulomb_build(sb, D.data(), grid, gtg, 20);
    auto t2 = clk::now();

    double w = 0, s = 0;
    for (std::size_t k = 0; k < Ja.size(); ++k) { w = std::max(w, std::fabs(Jg[k]-Ja[k])); s = std::max(s, std::fabs(Ja[k])); }
    const double tA = secs(t0, t1), tG = secs(t1, t2);
    std::printf("%-5d %-5d %-6d  %10.4f %10.4f %8.1f   %8.1e\n", maxl, n, grid.N, tA, tG, tG / tA, w / s);
    std::fflush(stdout);
  }
  return 0;
}
