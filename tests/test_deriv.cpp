// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/deriv.hpp"
#include "intti/nuclear.hpp"
#include "intti/oneel.hpp"
#include "intti/tgrid.hpp"

namespace {

using Shell = intti::PrimitiveShell<double>;

// shell 0 (p) on centre A, shell 1 (s), shell 2 (d) on centre B != A
std::vector<Shell> shells(const double A[3], const double B[3]) {
  return {{0.9, {A[0], A[1], A[2]}, 1},
          {1.3, {B[0], B[1], B[2]}, 0},
          {0.7, {B[0], B[1], B[2]}, 2}};
}

// finite-difference d/dA_x of a 1e matrix (shell 0's centre) for the block
// (mu in shell 0, nu in shell 1|2), compared to the analytic bra gradient
// <nabla mu|nu> (note: d/dA = -nabla on the electron coordinate).
template <class F>
void fd_check(F matrix_of, const std::array<std::vector<double>, 3> &G) {
  const double A[3] = {0.1, -0.2, 0.3}, B[3] = {0.5, 0.4, -0.6};
  auto bas = intti::make_basis(shells(A, B));
  const int nao = bas.nao;
  const double h = 1e-5;
  for (int d = 0; d < 3; ++d) {
    double Ap[3] = {A[0], A[1], A[2]}, Am[3] = {A[0], A[1], A[2]};
    Ap[d] += h;
    Am[d] -= h;
    auto Mp = matrix_of(intti::make_basis(shells(Ap, B)));
    auto Mm = matrix_of(intti::make_basis(shells(Am, B)));
    // rows of shell 0 are AOs 0..2 (p); columns are shell 1 (AO 3) and shell 2
    for (int mu = 0; mu < 3; ++mu)
      for (int nu = 3; nu < nao; ++nu) {
        const double fd = (Mp[mu * nao + nu] - Mm[mu * nao + nu]) / (2 * h);
        // d/dA = -nabla
        EXPECT_NEAR(-G[d][mu * nao + nu], fd, 1e-6 * (std::abs(fd) + 1.0))
            << "d=" << d << " mu=" << mu << " nu=" << nu;
      }
  }
}

TEST(Deriv, OverlapGradientVsFiniteDifference) {
  const double A[3] = {0.1, -0.2, 0.3}, B[3] = {0.5, 0.4, -0.6};
  auto bas = intti::make_basis(shells(A, B));
  auto G = intti::overlap_deriv(bas);
  fd_check([](const intti::ShellBasis<double> &b) { return intti::overlap_matrix(b); }, G);
}

TEST(Deriv, KineticGradientVsFiniteDifference) {
  const double A[3] = {0.1, -0.2, 0.3}, B[3] = {0.5, 0.4, -0.6};
  auto bas = intti::make_basis(shells(A, B));
  auto G = intti::kinetic_deriv(bas);
  fd_check([](const intti::ShellBasis<double> &b) { return intti::kinetic_matrix(b); }, G);
}

TEST(Deriv, NuclearGradientVsFiniteDifference) {
  const double A[3] = {0.1, -0.2, 0.3}, B[3] = {0.5, 0.4, -0.6};
  auto grid = intti::make_tgrid(intti::coulomb());
  std::vector<intti::PointCharge<double>> ch{{-1.0, {0.0, 0.0, 0.9}}};
  auto bas = intti::make_basis(shells(A, B));
  auto G = intti::nuclear_deriv(bas, ch, grid);
  auto Vof = [&](const intti::ShellBasis<double> &b) {
    return intti::nuclear_matrix(b, ch, grid);
  };
  const int nao = bas.nao;
  const double h = 1e-5;
  for (int d = 0; d < 3; ++d) {
    double Ap[3] = {A[0], A[1], A[2]}, Am[3] = {A[0], A[1], A[2]};
    Ap[d] += h;
    Am[d] -= h;
    auto Vp = Vof(intti::make_basis(shells(Ap, B)));
    auto Vm = Vof(intti::make_basis(shells(Am, B)));
    for (int mu = 0; mu < 3; ++mu)
      for (int nu = 3; nu < nao; ++nu) {
        const double fd = (Vp[mu * nao + nu] - Vm[mu * nao + nu]) / (2 * h);
        EXPECT_NEAR(-G[d][mu * nao + nu], fd, 1e-6 * (std::abs(fd) + 1.0))
            << "d=" << d << " mu=" << mu << " nu=" << nu;
      }
  }
}

} // namespace
