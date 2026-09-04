// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
//
// M-TC: the non-Hermitian transcorrelated two-body integral
//   <pq| nabla_1 u_12 . nabla_1 |rs>,  u = sum_k c_k e^{-g_k r12^2}.
// tc_gradu_grad_quartet reduces it (integration by parts) to plain
// Gaussian-geminal integrals with derivative-shifted orbitals. Reference values
// come from the independent symbolic oracle references/sympy_tc.py (direct 6D
// Gaussian integration of the operator -- no by-parts, no libintti).

#include <cmath>

#include <gtest/gtest.h>

#include "intti/tc.hpp"

namespace {

using intti::PrimitiveShell;

// Fixed geometry shared with references/sympy_tc.py:
//   p: a=1.2 @ (0,0,0)   q: b=0.9 @ (0,1,0)
//   r: c=0.8 @ (1/2,0,0) s: d=1.1 @ (1/5,9/10,3/10)
static PrimitiveShell<double> P(int l) { return {1.2, {0, 0, 0}, l}; }
static PrimitiveShell<double> Q(int l) { return {0.9, {0, 1, 0}, l}; }
static PrimitiveShell<double> R(int l) { return {0.8, {0.5, 0, 0}, l}; }
static PrimitiveShell<double> S(int l) { return {1.1, {0.2, 0.9, 0.3}, l}; }

TEST(TC, NonHermitianSSSS) {
  auto gem = intti::gaussian_geminal<double>({0.7}, {1.0});
  double out[1];
  intti::tc_gradu_grad_quartet(P(0), Q(0), R(0), S(0), gem, out);
  // oracle: full 6D + factorised 2D symbolic (references/sympy_tc.py)
  EXPECT_NEAR(out[0], 0.6919329300223959, 1e-12);
}

TEST(TC, NonHermitianGeminalSum) {
  auto gem = intti::gaussian_geminal<double>({0.2, 0.8, 3.0}, {0.5, 0.3, 0.2});
  double out[1];
  intti::tc_gradu_grad_quartet(P(0), Q(0), R(0), S(0), gem, out);
  EXPECT_NEAR(out[0], 0.5896308883697698, 1e-12);
}

// Ket orbital r carries angular momentum (the operator's derivative acts on it).
TEST(TC, NonHermitianKetPtype) {
  auto gem = intti::gaussian_geminal<double>({0.7}, {1.0});
  double out[3]; // r is l=1: components px, py, pz
  intti::tc_gradu_grad_quartet(P(0), Q(0), R(1), S(0), gem, out);
  EXPECT_NEAR(out[0], -0.373931732962435, 1e-12); // r = p_x
}

// Both electron-1 orbitals carry angular momentum.
TEST(TC, NonHermitianBraKetPtype) {
  auto gem = intti::gaussian_geminal<double>({0.7}, {1.0});
  double out[9]; // p is l=1 (3), r is l=1 (3); layout (kp*1+kq)*3+kr, kq=ks=0
  intti::tc_gradu_grad_quartet(P(1), Q(0), R(1), S(0), gem, out);
  // p = p_y (kp=1), r = p_x (kr=0)
  EXPECT_NEAR(out[(1 * 1 + 0) * 3 + 0], -0.015163762479314398, 1e-12);
}

} // namespace
