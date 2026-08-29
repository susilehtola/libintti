// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

// Dump the full Cartesian ERI tensor of an uncontracted shell list, for
// cross-validation against PySCF (prototype/pyscf_validation.py).
// usage: intti_dump_eri <spec> <out.bin> <none|comp|cca>
// spec lines: x y z l alpha   (Bohr; one primitive shell per line)

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "intti/fock.hpp"
#include "intti/normalization.hpp"
#include "intti/quartet.hpp"

int main(int argc, char **argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <spec> <out.bin> <none|comp|cca>\n", argv[0]);
    return 1;
  }
  Kokkos::ScopeGuard guard(argc, argv);
  std::vector<intti::PrimitiveShell<double>> shells;
  {
    std::ifstream in(argv[1]);
    if (!in) {
      std::fprintf(stderr, "cannot open %s\n", argv[1]);
      return 1;
    }
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream ss(line);
      double x, y, z, alpha;
      int l;
      ss >> x >> y >> z >> l >> alpha;
      shells.push_back({alpha, {x, y, z}, l});
    }
  }
  auto basis = intti::make_basis(shells);
  const int nao = basis.nao;
  // per-AO normalization factors
  std::vector<double> norm(nao, 1.0);
  const std::string mode = argv[3];
  for (std::size_t s = 0; s < shells.size(); ++s)
    for (int k = 0; k < intti::ncart(shells[s].l); ++k) {
      int lx, ly, lz;
      intti::cart_comp(shells[s].l, k, lx, ly, lz);
      if (mode == "comp")
        norm[basis.ao_off[s] + k] =
            intti::cart_norm_component(lx, ly, lz, shells[s].alpha);
      else if (mode == "cca")
        norm[basis.ao_off[s] + k] = intti::cart_norm_cca(shells[s].l, shells[s].alpha);
      else if (mode == "pyscf")
        norm[basis.ao_off[s] + k] =
            intti::cart_norm_pyscf(shells[s].l, shells[s].alpha);
    }
  std::vector<double> eri(static_cast<std::size_t>(nao) * nao * nao * nao);
  auto grid = intti::make_tgrid(intti::coulomb());
  const int ns = static_cast<int>(shells.size());
  for (int i = 0; i < ns; ++i)
    for (int j = 0; j < ns; ++j)
      for (int k = 0; k < ns; ++k)
        for (int l = 0; l < ns; ++l) {
          const auto bra = intti::make_pair(shells[i], shells[j]);
          const auto ket = intti::make_pair(shells[k], shells[l]);
          const int na = intti::ncart(shells[i].l), nb = intti::ncart(shells[j].l);
          const int nc = intti::ncart(shells[k].l), nd = intti::ncart(shells[l].l);
          std::vector<double> block(na * nb * nc * nd);
          intti::eri_quartet(bra, ket, grid, block.data());
          for (int ka = 0; ka < na; ++ka)
            for (int kb = 0; kb < nb; ++kb)
              for (int kc = 0; kc < nc; ++kc)
                for (int kd = 0; kd < nd; ++kd) {
                  const int A = basis.ao_off[i] + ka, B = basis.ao_off[j] + kb;
                  const int C = basis.ao_off[k] + kc, E = basis.ao_off[l] + kd;
                  eri[((static_cast<std::size_t>(A) * nao + B) * nao + C) * nao + E] =
                      norm[A] * norm[B] * norm[C] * norm[E] *
                      block[((ka * nb + kb) * nc + kc) * nd + kd];
                }
        }
  std::ofstream out(argv[2], std::ios::binary);
  out.write(reinterpret_cast<const char *>(eri.data()),
            static_cast<std::streamsize>(eri.size() * sizeof(double)));
  std::printf("nao %d\n", nao);
  return 0;
}
