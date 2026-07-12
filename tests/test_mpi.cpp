// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola

// Standalone MPI correctness check for M8 (not a gtest binary: MPI must be
// initialized before Kokkos, and the pass/fail verdict must be agreed by all
// ranks before anyone calls MPI_Finalize). Run under mpirun -np {1,2,4}; see
// tests/CMakeLists.txt for the registered ctests.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <mpi.h>

#include "intti/fock.hpp"
#include "intti/mpi.hpp"

namespace {

// same mixed s/p/d two-center system as tests/test_fock.cpp
intti::ShellBasis<double> test_basis() {
  return intti::make_basis<double>({
      {1.2, {0.0, 0.0, 0.0}, 0},
      {0.3, {0.0, 0.0, 0.0}, 0},
      {0.8, {0.0, 0.0, 0.0}, 1},
      {1.5, {0.0, 0.0, 1.4}, 0},
      {0.5, {0.0, 0.0, 1.4}, 1},
      {0.9, {0.0, 0.0, 1.4}, 2},
  });
}

std::vector<double> random_symmetric(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> D(static_cast<std::size_t>(n) * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j <= i; ++j)
      D[i * n + j] = D[j * n + i] = u(rng);
  return D;
}

double max_abs_diff(const std::vector<double> &a, const std::vector<double> &b) {
  double m = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

double max_abs(const std::vector<double> &a) {
  double m = 0;
  for (double v : a)
    m = std::max(m, std::abs(v));
  return m;
}

} // namespace

int main(int argc, char **argv) {
  // MPI must be initialized before Kokkos (roadmap M8 requirement).
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  int allok = 0;
  {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto b = test_basis();
    auto D = random_symmetric(b.nao, 5);
    auto grid = intti::make_tgrid(intti::coulomb());
    const std::size_t n2 = static_cast<std::size_t>(b.nao) * b.nao;

    // serial reference, computed identically (and deterministically) on
    // every rank
    std::vector<double> Jref(n2), Kref(n2);
    intti::coulomb_build(b, D.data(), grid, Jref.data());
    intti::exchange_build(b, D.data(), grid, Kref.data(), 0.0);

    // MPI-distributed builds (all ranks must call these together)
    std::vector<double> J(n2), K(n2);
    intti::coulomb_build_mpi(b, D.data(), grid, J.data(), MPI_COMM_WORLD);
    intti::exchange_build_mpi(b, D.data(), grid, K.data(), MPI_COMM_WORLD, 0.0);

    const double relJ = max_abs_diff(J, Jref) / max_abs(Jref);
    const double relK = max_abs_diff(K, Kref) / max_abs(Kref);
    const int ok = (relJ < 1e-15 && relK < 1e-15) ? 1 : 0;

    MPI_Allreduce(&ok, &allok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (rank == 0) {
      std::printf("test_mpi: nranks=%d relJ=%.3e relK=%.3e -> %s\n", size, relJ,
                  relK, allok ? "PASS" : "FAIL");
    }
  }

  Kokkos::finalize();
  MPI_Finalize();
  return allok ? 0 : 1;
}
