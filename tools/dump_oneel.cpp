// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola

// Dump one-electron property matrices (overlap, kinetic, dipole, quadrupole)
// of an uncontracted shell list for cross-validation against PySCF
// (prototype/pyscf_oneel_validation.py).
// usage: intti_dump_oneel <spec> <out.bin>
// spec lines: x y z l alpha   (Bohr; one primitive shell per line)
// output: nao (printed), then binary doubles: S, T, then dipole x,y,z, then
// quadrupole xx,xy,xz,yy,yz,zz  (each nao*nao, row-major, PySCF-normalized).

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "intti/deriv.hpp"
#include "intti/nuclear.hpp"
#include "intti/normalization.hpp"
#include "intti/nuclear.hpp"
#include "intti/oneel.hpp"

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <spec> <out.bin>\n", argv[0]);
    return 1;
  }
  Kokkos::ScopeGuard guard(argc, argv);
  std::vector<intti::PrimitiveShell<double>> shells;
  {
    std::ifstream in(argv[1]);
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
  // per-AO PySCF cartesian normalization
  std::vector<double> nrm(nao, 1.0);
  for (std::size_t s = 0; s < shells.size(); ++s)
    for (int k = 0; k < intti::ncart(shells[s].l); ++k)
      nrm[basis.ao_off[s] + k] = intti::cart_norm_pyscf(shells[s].l, shells[s].alpha);

  auto S = intti::overlap_matrix(basis);
  auto T = intti::kinetic_matrix(basis);
  const double origin[3] = {0, 0, 0};
  auto M = intti::multipole_matrices(basis, 2, origin); // 0:1, 1:x y z, 4:xx..zz

  std::ofstream out(argv[2], std::ios::binary);
  auto write_norm = [&](const std::vector<double> &Mx) {
    std::vector<double> tmp(Mx.size());
    for (int i = 0; i < nao; ++i)
      for (int j = 0; j < nao; ++j)
        tmp[i * nao + j] = nrm[i] * nrm[j] * Mx[i * nao + j];
    out.write(reinterpret_cast<const char *>(tmp.data()),
              static_cast<std::streamsize>(tmp.size() * sizeof(double)));
  };
  write_norm(S);
  write_norm(T);
  for (int c = 1; c <= 3; ++c) write_norm(M[c]);   // dipole x,y,z
  for (int c = 4; c <= 9; ++c) write_norm(M[c]);   // quadrupole xx,xy,xz,yy,yz,zz
  // Coulomb-potential collocation at two points (validated vs int1e_rinv)
  auto grid = intti::make_tgrid(intti::coulomb());
  std::vector<std::array<double, 3>> pts = {{0.0, 0.0, 0.4}, {0.2, -0.1, 0.9}};
  auto Vp = intti::potential_matrices(basis, pts, grid);
  for (auto &V : Vp) write_norm(V);
  // gradient matrices <nabla mu|nu> (int1e_ipovlp) and <nabla mu|T|nu> (ipkin)
  auto dS = intti::overlap_deriv(basis);
  auto dT = intti::kinetic_deriv(basis);
  for (int d = 0; d < 3; ++d) write_norm(dS[d]);
  for (int d = 0; d < 3; ++d) write_norm(dT[d]);
  // nuclear-attraction gradient <nabla mu|1/|r-p0||nu> (int1e_iprinv @ p0)
  std::vector<intti::PointCharge<double>> one{{1.0, {pts[0][0], pts[0][1], pts[0][2]}}};
  auto dV = intti::nuclear_deriv(basis, one, grid);
  for (int d = 0; d < 3; ++d) write_norm(dV[d]);
  // angular momentum <mu|(r-O) x nabla|nu> about origin
  auto Lm = intti::angular_momentum(basis, origin);
  for (int d = 0; d < 3; ++d) write_norm(Lm[d]);
  std::printf("nao %d\n", nao);
  return 0;
}
