// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Reader for the shell dumps written by prototype/dump_basis.py, so the
// scaling benchmarks can run real basis sets (def2-SVP, def2-QZVPPD, ...)
// without a basis-set parser in the library.
//
// The libcint -> intti conversion below is the SAME formula the validated
// facade uses (src/cint.cpp::contracted_basis_from); nothing about
// normalization is re-derived here.

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "intti/contracted.hpp"
#include "intti/math.hpp"
#include "intti/normalization.hpp"

namespace intti_bench {

inline double coeff_rescale(int l) {
  if (l <= 1) return std::sqrt((2 * l + 1) / (4 * intti::pi_v<double>()));
  return 1.0;
}

inline intti::ContractedBasis<double> load_basis(const std::string &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open basis dump: " + path);
  int nsh = 0;
  in >> nsh;
  std::vector<intti::ContractedShell<double>> shells;
  shells.reserve(nsh);
  for (int i = 0; i < nsh; ++i) {
    double cx, cy, cz;
    int l, npr, nct;
    in >> cx >> cy >> cz >> l >> npr >> nct;
    intti::ContractedShell<double> sh;
    sh.center[0] = cx;
    sh.center[1] = cy;
    sh.center[2] = cz;
    sh.l = l;
    sh.alpha.resize(npr);
    for (int p = 0; p < npr; ++p) in >> sh.alpha[p];
    sh.coeff.resize(static_cast<std::size_t>(nct) * npr);
    const double rs = coeff_rescale(l);
    for (int c = 0; c < nct; ++c)
      for (int p = 0; p < npr; ++p) {
        double v;
        in >> v;
        sh.coeff[static_cast<std::size_t>(c) * npr + p] =
            v * rs / intti::cart_norm_pyscf(l, sh.alpha[p]);
      }
    shells.push_back(std::move(sh));
  }
  if (!in) throw std::runtime_error("truncated basis dump: " + path);
  return intti::make_contracted_basis(std::move(shells));
}

} // namespace intti_bench
