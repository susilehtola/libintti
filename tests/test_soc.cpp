// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/fock.hpp"
#include "intti/kernel.hpp"
#include "intti/nuclear.hpp"
#include "intti/soc.hpp"
#include "intti/tgrid.hpp"

// One-electron spin-orbit integrals (M17). Validated against a mixed finite
// difference of the nuclear-attraction integral (independent oracle):
//   W_k,uv = -eps_kij sum_A Z_A d^2/dR_A,i dR_v,j  <chi_u| 1/r_A |chi_v>,
// using (r-R_A)_i/r_A^3 = d/dR_A,i (1/r_A) and d_j chi_v = -d/dR_v,j chi_v.
// Also checks the required antisymmetry W_k = -W_k^T.

namespace {

// <mu|Z/r_A|nu> element (mu at Amu fixed, nu at Anu+dnu; charge Z at RA+dRA).
double Velem(const std::array<double, 3> &Amu, const std::array<double, 3> &Anu,
             const std::array<double, 3> &RA, double Z, const intti::TGrid<double> &grid,
             const std::array<double, 3> &dRA, const std::array<double, 3> &dnu) {
  std::vector<intti::PrimitiveShell<double>> shells = {
      {1.2, {Amu[0], Amu[1], Amu[2]}, 0},
      {0.8, {Anu[0] + dnu[0], Anu[1] + dnu[1], Anu[2] + dnu[2]}, 0}};
  auto basis = intti::make_basis(shells);
  std::vector<intti::PointCharge<double>> ch = {{Z, {RA[0] + dRA[0], RA[1] + dRA[1], RA[2] + dRA[2]}}};
  auto V = intti::nuclear_matrix(basis, ch, grid);
  return V[0 * 2 + 1]; // <mu|.|nu>
}

TEST(SpinOrbit, OneElectronVsFiniteDifference) {
  const std::array<double, 3> Amu{0.0, 0.0, 0.0}, Anu{0.3, 0.5, 1.2}, RA{-0.4, 0.2, 0.7};
  const double Z = 1.0, h = 2e-3;
  auto grid = intti::make_tgrid(intti::coulomb<double>());

  // reference basis + analytic SO
  std::vector<intti::PrimitiveShell<double>> shells = {
      {1.2, {Amu[0], Amu[1], Amu[2]}, 0}, {0.8, {Anu[0], Anu[1], Anu[2]}, 0}};
  auto basis = intti::make_basis(shells);
  std::vector<intti::PointCharge<double>> ch = {{Z, {RA[0], RA[1], RA[2]}}};
  auto W = intti::spin_orbit_1e(basis, ch, grid);

  // mixed FD: d^2/dR_A,i dR_nu,j <mu|Z/r_A|nu>
  auto e = [](int ax) { std::array<double, 3> v{0, 0, 0}; v[ax] = 1.0; return v; };
  auto mixed = [&](int i, int j) {
    auto pi = e(i), pj = e(j);
    auto sc = [&](double si, double sj) {
      std::array<double, 3> dRA{si * h * pi[0], si * h * pi[1], si * h * pi[2]};
      std::array<double, 3> dnu{sj * h * pj[0], sj * h * pj[1], sj * h * pj[2]};
      return Velem(Amu, Anu, RA, Z, grid, dRA, dnu);
    };
    return (sc(1, 1) - sc(1, -1) - sc(-1, 1) + sc(-1, -1)) / (4 * h * h);
  };
  // W_x = -(mixed(y,z)-mixed(z,y)), cyclic
  const double wx = -(mixed(1, 2) - mixed(2, 1));
  const double wy = -(mixed(2, 0) - mixed(0, 2));
  const double wz = -(mixed(0, 1) - mixed(1, 0));
  const int n = basis.nao; // 2
  EXPECT_NEAR(W[0][0 * n + 1], wx, 1e-5) << "W_x";
  EXPECT_NEAR(W[1][0 * n + 1], wy, 1e-5) << "W_y";
  EXPECT_NEAR(W[2][0 * n + 1], wz, 1e-5) << "W_z";
  // must be genuinely nonzero (not a trivial pass)
  EXPECT_GT(std::abs(wx) + std::abs(wy) + std::abs(wz), 1e-4);
}

TEST(SpinOrbit, Antisymmetric) {
  // s + p on two centres, two nuclei -> a nontrivial antisymmetric W_k
  std::vector<intti::PrimitiveShell<double>> shells = {
      {1.0, {0.0, 0.0, 0.0}, 0}, {0.7, {0.0, 0.0, 0.0}, 1},
      {0.9, {0.4, -0.3, 1.1}, 0}};
  auto basis = intti::make_basis(shells);
  auto grid = intti::make_tgrid(intti::coulomb<double>());
  std::vector<intti::PointCharge<double>> ch = {{3.0, {0, 0, 0}}, {1.0, {0.4, -0.3, 1.1}}};
  auto W = intti::spin_orbit_1e(basis, ch, grid);
  const int n = basis.nao;
  double asym = 0, mag = 0;
  for (int d = 0; d < 3; ++d)
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        asym = std::max(asym, std::abs(W[d][i * n + j] + W[d][j * n + i]));
        mag = std::max(mag, std::abs(W[d][i * n + j]));
      }
  EXPECT_GT(mag, 1e-3) << "W vanished";
  EXPECT_LT(asym, 1e-12 * (mag + 1)) << "W not antisymmetric";
}

} // namespace
