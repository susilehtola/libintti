// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/contracted.hpp"
#include "intti/fock.hpp"
#include "intti/normalization.hpp"
#include "intti/oneel.hpp"

// Contraction-aware 1e builders (M-SHARK #5): validated with (a) an independent
// closed-form contracted-Gaussian overlap and (b) a decontract -> primitive ->
// recontract reference against the (analytically validated) primitive
// overlap/kinetic builders, which isolates the new contraction+normalization
// layer.

namespace {

using intti::ContractedBasis;
using intti::ContractedShell;

// C S_prim C^T reference: decontract the basis into one primitive shell per
// Gaussian, build the primitive matrix with the primitive-ShellBasis builder,
// and contract it with C[contracted AO][primitive AO] = d_{cp} *
// cart_norm_pyscf(l, alpha_p) (same Cartesian component). This reuses the
// independently validated primitive builders and tests only the contraction.
template <class Builder>
std::vector<double> contract_ref(const ContractedBasis<double> &cb, Builder prim_builder) {
  std::vector<intti::PrimitiveShell<double>> prims;
  // primbase[a][p] = primitive-AO offset of shell a's primitive p
  std::vector<std::vector<int>> primbase(cb.shells.size());
  int po = 0;
  for (std::size_t a = 0; a < cb.shells.size(); ++a) {
    const auto &A = cb.shells[a];
    primbase[a].resize(A.nprim());
    for (int p = 0; p < A.nprim(); ++p) {
      primbase[a][p] = po;
      prims.push_back(intti::PrimitiveShell<double>{
          A.alpha[p], {A.center[0], A.center[1], A.center[2]}, A.l});
      po += intti::ncart(A.l);
    }
  }
  auto pbasis = intti::make_basis(prims);
  const int npao = pbasis.nao, ncao = cb.nao;
  const std::vector<double> Sp = prim_builder(pbasis);
  // dense C (ncao x npao)
  std::vector<double> C(static_cast<std::size_t>(ncao) * npao, 0.0);
  for (std::size_t a = 0; a < cb.shells.size(); ++a) {
    const auto &A = cb.shells[a];
    const int nc = intti::ncart(A.l);
    for (int cA = 0; cA < A.nctr(); ++cA)
      for (int p = 0; p < A.nprim(); ++p) {
        const double w = A.coeff[cA * A.nprim() + p] * intti::cart_norm_pyscf(A.l, A.alpha[p]);
        for (int k = 0; k < nc; ++k) {
          const int I = cb.ao_off[a] + cA * nc + k;
          const int P = primbase[a][p] + k;
          C[static_cast<std::size_t>(I) * npao + P] = w;
        }
      }
  }
  // S_c = C Sp C^T
  std::vector<double> CSp(static_cast<std::size_t>(ncao) * npao, 0.0);
  for (int I = 0; I < ncao; ++I)
    for (int P = 0; P < npao; ++P) {
      double s = 0;
      for (int Q = 0; Q < npao; ++Q)
        s += C[static_cast<std::size_t>(I) * npao + Q] * Sp[static_cast<std::size_t>(Q) * npao + P];
      CSp[static_cast<std::size_t>(I) * npao + P] = s;
    }
  std::vector<double> Sc(static_cast<std::size_t>(ncao) * ncao, 0.0);
  for (int I = 0; I < ncao; ++I)
    for (int J = 0; J < ncao; ++J) {
      double s = 0;
      for (int P = 0; P < npao; ++P)
        s += CSp[static_cast<std::size_t>(I) * npao + P] * C[static_cast<std::size_t>(J) * npao + P];
      Sc[static_cast<std::size_t>(I) * ncao + J] = s;
    }
  return Sc;
}

// A small generally-contracted test basis: an s shell with 3 primitives and
// TWO contracted functions (general contraction), plus a p shell with 2
// primitives and 1 contracted function on a second center.
ContractedBasis<double> test_basis() {
  ContractedShell<double> s;
  s.center[0] = 0; s.center[1] = 0; s.center[2] = 0;
  s.l = 0;
  s.alpha = {5.0, 1.2, 0.4};
  s.coeff = {0.15, 0.55, 0.45,   // contracted function 0
             -0.10, 0.20, 0.90}; // contracted function 1
  ContractedShell<double> p;
  p.center[0] = 0; p.center[1] = 0; p.center[2] = 1.3;
  p.l = 1;
  p.alpha = {1.0, 0.35};
  p.coeff = {0.6, 0.5};
  return intti::make_contracted_basis<double>({s, p});
}

// Closed-form self-overlap of a single contracted s function:
//   <chi|chi> = sum_pq d_p d_q N_p N_q (pi/(a_p+a_q))^{3/2},  N_p=(2a_p/pi)^{3/4}.
TEST(Contracted, SShellSelfOverlapAnalytic) {
  ContractedShell<double> s;
  s.l = 0;
  s.alpha = {3.0, 0.8, 0.25};
  s.coeff = {0.3, 0.6, 0.5};
  auto cb = intti::make_contracted_basis<double>({s});
  auto S = intti::overlap_matrix(cb);
  ASSERT_EQ(cb.nao, 1);
  double ref = 0;
  for (int p = 0; p < 3; ++p)
    for (int q = 0; q < 3; ++q) {
      const double Np = std::pow(2 * s.alpha[p] / M_PI, 0.75);
      const double Nq = std::pow(2 * s.alpha[q] / M_PI, 0.75);
      ref += s.coeff[p] * s.coeff[q] * Np * Nq *
             std::pow(M_PI / (s.alpha[p] + s.alpha[q]), 1.5);
    }
  EXPECT_NEAR(S[0], ref, 1e-13 * std::abs(ref));
}

TEST(Contracted, OverlapVsDecontractRecontract) {
  auto cb = test_basis();
  auto S = intti::overlap_matrix(cb);
  auto ref = contract_ref(cb, [](const intti::ShellBasis<double> &b) {
    return intti::overlap_matrix(b);
  });
  ASSERT_EQ(S.size(), ref.size());
  double worst = 0, scale = 0;
  for (std::size_t i = 0; i < S.size(); ++i) {
    worst = std::max(worst, std::abs(S[i] - ref[i]));
    scale = std::max(scale, std::abs(ref[i]));
  }
  EXPECT_LT(worst, 1e-13 * scale) << "contracted overlap != decontract/recontract";
}

TEST(Contracted, KineticVsDecontractRecontract) {
  auto cb = test_basis();
  auto T = intti::kinetic_matrix(cb);
  auto ref = contract_ref(cb, [](const intti::ShellBasis<double> &b) {
    return intti::kinetic_matrix(b);
  });
  ASSERT_EQ(T.size(), ref.size());
  double worst = 0, scale = 0;
  for (std::size_t i = 0; i < T.size(); ++i) {
    worst = std::max(worst, std::abs(T[i] - ref[i]));
    scale = std::max(scale, std::abs(ref[i]));
  }
  EXPECT_LT(worst, 1e-13 * scale) << "contracted kinetic != decontract/recontract";
}

TEST(Contracted, MultipoleVsDecontractRecontract) {
  auto cb = test_basis();
  const double origin[3] = {0.1, -0.2, 0.3};
  const int mo = 2;
  auto Mc = intti::multipole_matrices(cb, mo, origin);
  const int ncomp = static_cast<int>(Mc.size());
  for (int ci = 0; ci < ncomp; ++ci) {
    auto ref = contract_ref(cb, [ci, &origin, mo](const intti::ShellBasis<double> &b) {
      return intti::multipole_matrices(b, mo, origin)[ci];
    });
    double worst = 0, scale = 0;
    for (std::size_t i = 0; i < Mc[ci].size(); ++i) {
      worst = std::max(worst, std::abs(Mc[ci][i] - ref[i]));
      scale = std::max(scale, std::abs(ref[i]));
    }
    EXPECT_LT(worst, 1e-12 * scale + 1e-14) << "multipole component " << ci;
  }
}

// Symmetry and offset bookkeeping: S is symmetric and its dimension is the sum
// of nctr*ncart(l) over shells.
TEST(Contracted, SymmetricAndSized) {
  auto cb = test_basis();
  auto S = intti::overlap_matrix(cb);
  EXPECT_EQ(cb.nao, 2 * 1 + 1 * 3); // 2 s-contractions + 1 p-contraction (3 cart)
  const int n = cb.nao;
  double asym = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      asym = std::max(asym, std::abs(S[i * n + j] - S[j * n + i]));
  EXPECT_LT(asym, 1e-14);
}

} // namespace
