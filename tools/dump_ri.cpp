// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

// Dump 2- and 3-center Coulomb integrals for validation against PySCF
// (prototype/pyscf_ri_validation.py).
// usage: intti_dump_ri <orb_spec> <aux_spec> <out.bin>
// spec lines: x y z l alpha (Bohr). Output prints "nao <n> naux <m>", then
// binary doubles: (P|Q) [naux*naux], then (mu nu|P) [nao*nao*naux], both
// PySCF-normalized, row-major.

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "intti/ncenter.hpp"
#include "intti/normalization.hpp"

static std::vector<intti::PrimitiveShell<double>> read_spec(const char *path) {
  std::vector<intti::PrimitiveShell<double>> shells;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    double x, y, z, alpha;
    int l;
    ss >> x >> y >> z >> l >> alpha;
    shells.push_back({alpha, {x, y, z}, l});
  }
  return shells;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <orb_spec> <aux_spec> <out.bin>\n", argv[0]);
    return 1;
  }
  Kokkos::ScopeGuard guard(argc, argv);
  auto orb = intti::make_basis(read_spec(argv[1]));
  auto aux = intti::make_basis(read_spec(argv[2]));
  const int nao = orb.nao, naux = aux.nao;
  std::vector<double> no(nao, 1.0), na(naux, 1.0);
  for (std::size_t s = 0; s < orb.shells.size(); ++s)
    for (int k = 0; k < intti::ncart(orb.shells[s].l); ++k)
      no[orb.ao_off[s] + k] = intti::cart_norm_pyscf(orb.shells[s].l, orb.shells[s].alpha);
  for (std::size_t s = 0; s < aux.shells.size(); ++s)
    for (int k = 0; k < intti::ncart(aux.shells[s].l); ++k)
      na[aux.ao_off[s] + k] = intti::cart_norm_pyscf(aux.shells[s].l, aux.shells[s].alpha);

  auto grid = intti::make_tgrid(intti::coulomb());
  auto M = intti::coulomb_2c(aux, grid);
  auto T = intti::coulomb_3c(orb, aux, grid);
  std::ofstream out(argv[3], std::ios::binary);
  for (int P = 0; P < naux; ++P)
    for (int Q = 0; Q < naux; ++Q) {
      double v = na[P] * na[Q] * M[P * naux + Q];
      out.write(reinterpret_cast<const char *>(&v), sizeof v);
    }
  for (int mu = 0; mu < nao; ++mu)
    for (int nu = 0; nu < nao; ++nu)
      for (int P = 0; P < naux; ++P) {
        double v = no[mu] * no[nu] * na[P] *
                   T[(static_cast<std::size_t>(mu) * nao + nu) * naux + P];
        out.write(reinterpret_cast<const char *>(&v), sizeof v);
      }
  std::printf("nao %d naux %d\n", nao, naux);
  return 0;
}
