// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
//
// Benchmark: grid-RI (finite-element) J/K vs the analytic t-quadrature engine
// (coulomb_build / exchange_build), across a growing line of atoms (roadmap
// M-FE "benchmark vs GTO-RI"). Reports, per size: nao, the grid N/axis, the
// build times, the ratio, and the grid-RI accuracy vs the analytic result.
//
// Honest expectation on THIS box (OpenMP backend, no GPU): the analytic engine
// wins at these accessible sizes -- its per-quartet work is cheap and screened,
// while the grid pays a large N^3 up front. The point of the benchmark is the
// SCALING TREND and the accuracy, not the absolute crossover, which is a GPU /
// large-system / high-l / heavily-contracted story (where the grid's nao-linear
// AO cost and lack of the analytic's formal nao^4 pair growth pay off).
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
  const double eps = 1.5e-2; // grid interpolation tolerance
  std::printf("%-4s %-5s %-6s  %10s %10s %7s  %10s %10s %7s   %8s %8s\n", "nat", "nao", "gridN",
              "J_ana(s)", "J_grid(s)", "J x", "K_ana(s)", "K_grid(s)", "K x", "Jrel", "Krel");
  for (int nat = 1; nat <= 3; ++nat) {
    // a line of atoms along z, each with an s and a p shell (unnormalized prims)
    std::vector<intti::PrimitiveShell<double>> shells;
    for (int a = 0; a < nat; ++a) {
      const double z = 1.6 * a;
      shells.push_back({1.8, {0.0, 0.0, z}, 0});
      shells.push_back({0.9, {0.0, 0.0, z}, 1});
    }
    auto sb = intti::make_basis(shells);
    const int n = sb.nao;

    // symmetric density and a rank-2 occupied set for K
    std::vector<double> D((std::size_t)n * n);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) D[i * n + j] = 0.1 + 0.2 * std::cos(0.7 * i + 1.1 * j);
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j) { double m = 0.5 * (D[i*n+j]+D[j*n+i]); D[i*n+j]=D[j*n+i]=m; }
    const int nocc = 2;
    std::vector<double> C((std::size_t)n * nocc), Dk((std::size_t)n * n, 0.0);
    for (int u = 0; u < n; ++u)
      for (int c = 0; c < nocc; ++c) C[u * nocc + c] = 0.2 + 0.15 * std::sin(0.9 * u + c);
    for (int u = 0; u < n; ++u)
      for (int v = 0; v < n; ++v)
        for (int c = 0; c < nocc; ++c) Dk[u * n + v] += C[u * nocc + c] * C[v * nocc + c];

    auto gj = intti::make_tgrid(intti::coulomb<double>());
    intti::TGridSpec<double> spec; spec.n = 20;
    auto gtg = intti::make_tgrid(intti::coulomb<double>(), spec);
    auto grid = intti::grid_for_basis(sb, eps);

    // analytic J / grid-RI J
    std::vector<double> Ja((std::size_t)n * n, 0.0);
    auto t0 = clk::now();
    intti::coulomb_build(sb, D.data(), gj, Ja.data());
    auto t1 = clk::now();
    auto Jg = intti::grid_coulomb_build(sb, D.data(), grid, gtg, 20);
    auto t2 = clk::now();

    // analytic K / grid-RI K
    std::vector<double> Ka((std::size_t)n * n, 0.0);
    auto t3 = clk::now();
    intti::exchange_build(sb, Dk.data(), gj, Ka.data(), 0.0);
    auto t4 = clk::now();
    auto Kg = intti::grid_exchange_build(sb, C.data(), nocc, grid, gtg, 16);
    auto t5 = clk::now();

    auto relerr = [&](const std::vector<double> &g, const std::vector<double> &a) {
      double w = 0, s = 0;
      for (std::size_t k = 0; k < a.size(); ++k) { w = std::max(w, std::fabs(g[k]-a[k])); s = std::max(s, std::fabs(a[k])); }
      return w / s;
    };
    const double tJa = secs(t0, t1), tJg = secs(t1, t2), tKa = secs(t3, t4), tKg = secs(t4, t5);
    std::printf("%-4d %-5d %-6d  %10.4f %10.4f %7.1f  %10.4f %10.4f %7.1f   %8.1e %8.1e\n",
                nat, n, grid.N, tJa, tJg, tJg / tJa, tKa, tKg, tKg / tKa,
                relerr(Jg, Ja), relerr(Kg, Ka));
    std::fflush(stdout);
  }
  return 0;
}
