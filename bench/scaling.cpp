// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

// J/K scaling over water clusters in real basis sets. Not part of ctest: a
// single def2-QZVPPD point runs for minutes to hours.
//
// Usage:  scaling <dump-directory> [nwater...]
// with dumps written by prototype/dump_basis.py as <dir>/<basis>_<n>.txt.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "basis_io.hpp"
#include "intti/fock.hpp"
#include "intti/kernel.hpp"
#include "intti/tgrid.hpp"

namespace {

std::vector<double> model_density(int nao) {
  std::vector<double> D(static_cast<std::size_t>(nao) * nao);
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j)
      D[static_cast<std::size_t>(i) * nao + j] = 0.05 * std::cos(0.7 * i + 1.3 * j);
  for (int i = 0; i < nao; ++i)
    for (int j = i + 1; j < nao; ++j) {
      const double a = 0.5 * (D[static_cast<std::size_t>(i) * nao + j] +
                              D[static_cast<std::size_t>(j) * nao + i]);
      D[static_cast<std::size_t>(i) * nao + j] = a;
      D[static_cast<std::size_t>(j) * nao + i] = a;
    }
  return D;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
      .count();
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <dump-directory> [nwater...]\n", argv[0]);
    return 1;
  }
  const std::string dir = argv[1];
  std::vector<int> sizes;
  for (int i = 2; i < argc; ++i) sizes.push_back(std::atoi(argv[i]));
  if (sizes.empty()) sizes = {1, 2, 4};

  Kokkos::initialize(argc, argv);
  {
    auto tg = intti::make_tgrid(intti::coulomb());
    std::printf("%-13s %5s %6s %8s %12s %12s\n", "basis", "nwat", "nao", "npair",
                "J ms", "K ms");
    for (const char *b : {"def2-SVP", "def2-QZVPPD"})
      for (int n : sizes) {
        const std::string path = dir + "/" + b + "_" + std::to_string(n) + ".txt";
        intti::ContractedBasis<double> cb;
        try {
          cb = intti_bench::load_basis(path);
        } catch (const std::exception &e) {
          std::fprintf(stderr, "skip %s: %s\n", path.c_str(), e.what());
          continue;
        }
        const int nao = cb.nao;
        const auto D = model_density(nao);
        std::vector<double> J(static_cast<std::size_t>(nao) * nao, 0.0);
        std::vector<double> K(static_cast<std::size_t>(nao) * nao, 0.0);
        auto t0 = std::chrono::steady_clock::now();
        intti::coulomb_build(cb, D.data(), tg, J.data(), 1e-10);
        const double tj = ms_since(t0);
        t0 = std::chrono::steady_clock::now();
        intti::exchange_build(cb, D.data(), tg, K.data(), 1e-10);
        const double tk = ms_since(t0);
        std::printf("%-13s %5d %6d %8d %12.0f %12.0f\n", b, n, nao,
                    static_cast<int>(cb.shells.size() * (cb.shells.size() + 1) / 2), tj,
                    tk);
        std::fflush(stdout);
      }
  }
  Kokkos::finalize();
  return 0;
}
