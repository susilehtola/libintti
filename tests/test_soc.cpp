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

// Two-electron spin-orbit (Coulomb-type), M17 second half. Independent oracle:
// the mixed finite difference of the ordinary Coulomb build in the electron-1
// centres. With mu at A_mu+delta and lambda at A_lambda+delta+eta (delta shifts
// both, eta only lambda), F(delta,eta) = [coulomb_build]_{mu,lambda} satisfies
//   d^2/ddelta_i deta_j F|0 = <mu (d_{1,j} lambda) (r12)_i/r12^3 nu sigma> D_ns,
// so Y_k = eps_kij d^2/ddelta_i deta_j F -- the operator differentiated directly
// (no by-parts reduction assumed), a genuine check of the reduced builder.

TEST(SpinOrbit, TwoElectronCoulombVsFiniteDifference) {
  // four s primitives: mu, lambda on electron 1; the whole basis is the density
  const std::array<std::array<double, 3>, 4> A{
      {{0.0, 0.0, 0.0}, {0.4, 0.1, -0.3}, {-0.2, 0.5, 0.7}, {0.6, -0.4, 0.2}}};
  const std::array<double, 4> al{1.3, 0.9, 1.1, 0.7};
  auto mk = [&](const std::array<std::array<double, 3>, 4> &cen) {
    std::vector<intti::PrimitiveShell<double>> sh;
    for (int i = 0; i < 4; ++i) sh.push_back({al[i], {cen[i][0], cen[i][1], cen[i][2]}, 0});
    return intti::make_basis(sh);
  };
  auto grid = intti::make_tgrid(intti::coulomb<double>());
  auto base = mk(A);
  const int n = base.nao; // 4
  // a nontrivial symmetric density
  std::vector<double> D(static_cast<std::size_t>(n) * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) D[i * n + j] = 0.3 + 0.1 * i - 0.05 * j + 0.02 * i * j;
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) D[j * n + i] = D[i * n + j];

  auto Y = intti::spin_orbit_2e_coulomb(base, D.data(), grid);

  // F(delta,eta) = sum_{c,d} D_cd (mu lambda | c d) with ONLY the electron-1 bra
  // pair shifted (mu at A0+delta, lambda at A1+delta+eta) and the electron-2 ket
  // shells c,d fixed -- differentiating exactly what the builder differentiates
  // (coulomb_build would also move shells 0,1 where they appear in the ket sum).
  const double h = 3e-3;
  auto Fval = [&](const std::array<double, 3> &delta, const std::array<double, 3> &eta) {
    intti::PrimitiveShell<double> mu{al[0], {A[0][0] + delta[0], A[0][1] + delta[1], A[0][2] + delta[2]}, 0};
    intti::PrimitiveShell<double> lam{
        al[1], {A[1][0] + delta[0] + eta[0], A[1][1] + delta[1] + eta[1], A[1][2] + delta[2] + eta[2]}, 0};
    double f = 0;
    for (int c = 0; c < 4; ++c)
      for (int d = 0; d < 4; ++d) {
        intti::PrimitiveShell<double> sc{al[c], {A[c][0], A[c][1], A[c][2]}, 0};
        intti::PrimitiveShell<double> sd{al[d], {A[d][0], A[d][1], A[d][2]}, 0};
        f += D[c * n + d] * intti::detail::eri_block4(mu, lam, sc, sd, grid)[0];
      }
    return f;
  };
  auto e = [](int ax) { std::array<double, 3> v{0, 0, 0}; v[ax] = 1.0; return v; };
  auto mixed = [&](int i, int j) {
    auto pi = e(i), pj = e(j);
    auto sc = [&](double si, double sj) {
      std::array<double, 3> dl{si * h * pi[0], si * h * pi[1], si * h * pi[2]};
      std::array<double, 3> et{sj * h * pj[0], sj * h * pj[1], sj * h * pj[2]};
      return Fval(dl, et);
    };
    return (sc(1, 1) - sc(1, -1) - sc(-1, 1) + sc(-1, -1)) / (4 * h * h);
  };
  const double yx = mixed(1, 2) - mixed(2, 1);
  const double yy = mixed(2, 0) - mixed(0, 2);
  const double yz = mixed(0, 1) - mixed(1, 0);
  // second-order centre FD: accuracy floor ~ h^2 ~ 1e-5 relative
  EXPECT_NEAR(Y[0][0 * n + 1], yx, 1e-4 * (std::abs(yx) + 1)) << "Y_x";
  EXPECT_NEAR(Y[1][0 * n + 1], yy, 1e-4 * (std::abs(yy) + 1)) << "Y_y";
  EXPECT_NEAR(Y[2][0 * n + 1], yz, 1e-4 * (std::abs(yz) + 1)) << "Y_z";
  EXPECT_GT(std::abs(yx) + std::abs(yy) + std::abs(yz), 1e-5) << "trivially zero";
}

// Exchange-type 2e SO: Ke_k,us = sum_{l,n} D_ln (u l|SO_k|n s), the density
// bridging one electron-1 index (lambda) and one electron-2 index (nu). Oracle:
// the mixed centre FD with mu (output row) shifted by delta, lambda (summed) by
// delta+eta, nu,sigma fixed -- differentiating the raw operator directly.
TEST(SpinOrbit, TwoElectronExchangeVsFiniteDifference) {
  const std::array<std::array<double, 3>, 4> A{
      {{0.0, 0.0, 0.0}, {0.4, 0.1, -0.3}, {-0.2, 0.5, 0.7}, {0.6, -0.4, 0.2}}};
  const std::array<double, 4> al{1.3, 0.9, 1.1, 0.7};
  auto grid = intti::make_tgrid(intti::coulomb<double>());
  std::vector<intti::PrimitiveShell<double>> sh;
  for (int i = 0; i < 4; ++i) sh.push_back({al[i], {A[i][0], A[i][1], A[i][2]}, 0});
  auto base = intti::make_basis(sh);
  const int n = base.nao; // 4
  std::vector<double> D(static_cast<std::size_t>(n) * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) D[i * n + j] = 0.3 + 0.1 * i - 0.05 * j + 0.02 * i * j;
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) D[j * n + i] = D[i * n + j];

  auto Ke = intti::spin_orbit_2e_exchange(base, D.data(), grid);

  // Fe(delta,eta) = sum_{lambda,nu} D_{lambda,nu} (mu lambda | nu sigma), mu=0
  // (row) shifted delta, lambda shifted delta+eta, nu summed, sigma=1 fixed.
  const int muS = 0, sigS = 1;
  const double h = 3e-3;
  auto Fe = [&](const std::array<double, 3> &dl, const std::array<double, 3> &et) {
    intti::PrimitiveShell<double> mu{al[muS], {A[muS][0] + dl[0], A[muS][1] + dl[1], A[muS][2] + dl[2]}, 0};
    intti::PrimitiveShell<double> sig{al[sigS], {A[sigS][0], A[sigS][1], A[sigS][2]}, 0};
    double f = 0;
    for (int lb = 0; lb < 4; ++lb)
      for (int nu = 0; nu < 4; ++nu) {
        intti::PrimitiveShell<double> lam{
            al[lb], {A[lb][0] + dl[0] + et[0], A[lb][1] + dl[1] + et[1], A[lb][2] + dl[2] + et[2]}, 0};
        intti::PrimitiveShell<double> sn{al[nu], {A[nu][0], A[nu][1], A[nu][2]}, 0};
        f += D[lb * n + nu] * intti::detail::eri_block4(mu, lam, sn, sig, grid)[0];
      }
    return f;
  };
  auto e = [](int ax) { std::array<double, 3> v{0, 0, 0}; v[ax] = 1.0; return v; };
  auto mixed = [&](int i, int j) {
    auto pi = e(i), pj = e(j);
    auto sc = [&](double si, double sj) {
      std::array<double, 3> dl{si * h * pi[0], si * h * pi[1], si * h * pi[2]};
      std::array<double, 3> et{sj * h * pj[0], sj * h * pj[1], sj * h * pj[2]};
      return Fe(dl, et);
    };
    return (sc(1, 1) - sc(1, -1) - sc(-1, 1) + sc(-1, -1)) / (4 * h * h);
  };
  const double kx = mixed(1, 2) - mixed(2, 1);
  const double ky = mixed(2, 0) - mixed(0, 2);
  const double kz = mixed(0, 1) - mixed(1, 0);
  const std::size_t idx = static_cast<std::size_t>(muS) * n + sigS;
  EXPECT_NEAR(Ke[0][idx], kx, 1e-4 * (std::abs(kx) + 1)) << "Ke_x";
  EXPECT_NEAR(Ke[1][idx], ky, 1e-4 * (std::abs(ky) + 1)) << "Ke_y";
  EXPECT_NEAR(Ke[2][idx], kz, 1e-4 * (std::abs(kz) + 1)) << "Ke_z";
  EXPECT_GT(std::abs(kx) + std::abs(ky) + std::abs(kz), 1e-5) << "trivially zero";
}

TEST(SpinOrbit, TwoElectronAntisymmetric) {
  // s + p on two centres -> nontrivial antisymmetric Y_k
  std::vector<intti::PrimitiveShell<double>> shells = {
      {1.0, {0.0, 0.0, 0.0}, 0}, {0.7, {0.2, -0.1, 0.4}, 1},
      {0.9, {-0.3, 0.5, 0.1}, 0}};
  auto bas = intti::make_basis(shells);
  auto grid = intti::make_tgrid(intti::coulomb<double>());
  const int n = bas.nao;
  std::vector<double> D(static_cast<std::size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) D[i * n + j] = 0.2 + 0.03 * (i + 1) * (j + 1);
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) D[j * n + i] = D[i * n + j];
  auto Y = intti::spin_orbit_2e_coulomb(bas, D.data(), grid);
  double asym = 0, mag = 0;
  for (int d = 0; d < 3; ++d)
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        asym = std::max(asym, std::abs(Y[d][i * n + j] + Y[d][j * n + i]));
        mag = std::max(mag, std::abs(Y[d][i * n + j]));
      }
  EXPECT_GT(mag, 1e-3) << "Y vanished";
  EXPECT_LT(asym, 1e-11 * (mag + 1)) << "Y not antisymmetric";
}


TEST(SpinOrbit, TwoElectronScreeningMatchesExact) {
  // The ordered quartet loop is pruned with a Schwarz bound on the
  // derivative-expanded quartets, carrying max(2 alpha, l) per bra shell for the
  // MD centre-shift coefficients. On an extended chain -- where screening is
  // supposed to bite -- a tight tau must reproduce the unscreened matrices, and
  // a loose one must degrade smoothly rather than drop whole blocks.
  std::vector<intti::PrimitiveShell<double>> shells;
  for (int i = 0; i < 8; ++i)
    shells.push_back({0.6 + 0.3 * i, {4.0 * i, 0.2 * (i % 3), -0.1 * (i % 2)}, i % 2});
  auto basis = intti::make_basis(shells);
  auto grid = intti::make_tgrid(intti::coulomb());
  const int nao = basis.nao;
  std::vector<double> D(static_cast<std::size_t>(nao) * nao, 0.0);
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j)
      D[i * nao + j] = (i == j) ? 1.0 : 0.15 / (1 + std::abs(i - j));
  const auto Yex = intti::spin_orbit_2e_coulomb(basis, D.data(), grid);
  const auto Kex = intti::spin_orbit_2e_exchange(basis, D.data(), grid);
  double scale = 0;
  for (int k = 0; k < 3; ++k)
    for (double v : Yex[k]) scale = std::max(scale, std::abs(v));
  ASSERT_GT(scale, 0.0);
  for (double tau : {1e-12, 1e-10}) {
    const auto Y = intti::spin_orbit_2e_coulomb(basis, D.data(), grid, tau);
    const auto K = intti::spin_orbit_2e_exchange(basis, D.data(), grid, tau);
    for (int k = 0; k < 3; ++k)
      for (std::size_t i = 0; i < Yex[k].size(); ++i) {
        EXPECT_NEAR(Y[k][i], Yex[k][i], 1e-9 * scale) << "coulomb tau=" << tau << " k=" << k;
        EXPECT_NEAR(K[k][i], Kex[k][i], 1e-9 * scale) << "exchange tau=" << tau << " k=" << k;
      }
  }
  // tau = 0 enumerates exactly the same quartets, so it agrees to round-off --
  // not bit-for-bit: the digest scatters with atomics, whose summation order
  // varies between runs of identical code.
  const auto Y0 = intti::spin_orbit_2e_coulomb(basis, D.data(), grid, 0.0);
  for (int k = 0; k < 3; ++k)
    for (std::size_t i = 0; i < Yex[k].size(); ++i)
      EXPECT_NEAR(Y0[k][i], Yex[k][i], 1e-14 * scale) << "tau=0 k=" << k;
}

} // namespace
