// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/contracted.hpp"
#include "intti/fock.hpp"
#include "intti/kernel.hpp"
#include "intti/ncenter.hpp"
#include "intti/normalization.hpp"
#include "intti/nuclear.hpp"
#include "intti/oneel.hpp"
#include "intti/tgrid.hpp"

// Contraction-aware 1e builders (M-SHARK #5): validated with (a) an independent
// closed-form contracted-Gaussian overlap and (b) a decontract -> primitive ->
// recontract reference against the (analytically validated) primitive
// overlap/kinetic builders, which isolates the new contraction+normalization
// layer.

namespace {

using intti::ContractedBasis;
using intti::ContractedShell;

// Decontract a contracted basis into one primitive shell per Gaussian, and
// build the contraction map C[contracted AO][primitive AO] = d_{cp} *
// cart_norm_pyscf(l, alpha_p) (same Cartesian component). Returns npao; fills
// pbasis and the dense (ncao x npao) C.
int decontract(const ContractedBasis<double> &cb, intti::ShellBasis<double> &pbasis,
               std::vector<double> &C) {
  std::vector<intti::PrimitiveShell<double>> prims;
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
  pbasis = intti::make_basis(prims);
  const int npao = pbasis.nao, ncao = cb.nao;
  C.assign(static_cast<std::size_t>(ncao) * npao, 0.0);
  for (std::size_t a = 0; a < cb.shells.size(); ++a) {
    const auto &A = cb.shells[a];
    const int nc = intti::ncart(A.l);
    for (int cA = 0; cA < A.nctr(); ++cA)
      for (int p = 0; p < A.nprim(); ++p) {
        const double w = A.coeff[cA * A.nprim() + p] * intti::cart_norm_pyscf(A.l, A.alpha[p]);
        for (int k = 0; k < nc; ++k)
          C[static_cast<std::size_t>(cb.ao_off[a] + cA * nc + k) * npao + primbase[a][p] + k] = w;
      }
  }
  return npao;
}

// C M_prim C^T (ncao x ncao) from a primitive matrix M_prim (npao x npao).
std::vector<double> conjugate(const std::vector<double> &C, int ncao, int npao,
                             const std::vector<double> &M) {
  std::vector<double> CM(static_cast<std::size_t>(ncao) * npao, 0.0);
  for (int I = 0; I < ncao; ++I)
    for (int P = 0; P < npao; ++P) {
      double s = 0;
      for (int Q = 0; Q < npao; ++Q)
        s += C[static_cast<std::size_t>(I) * npao + Q] * M[static_cast<std::size_t>(Q) * npao + P];
      CM[static_cast<std::size_t>(I) * npao + P] = s;
    }
  std::vector<double> out(static_cast<std::size_t>(ncao) * ncao, 0.0);
  for (int I = 0; I < ncao; ++I)
    for (int J = 0; J < ncao; ++J) {
      double s = 0;
      for (int P = 0; P < npao; ++P)
        s += CM[static_cast<std::size_t>(I) * npao + P] * C[static_cast<std::size_t>(J) * npao + P];
      out[static_cast<std::size_t>(I) * ncao + J] = s;
    }
  return out;
}

// C^T M_contr C (npao x npao) from a contracted matrix M_contr (ncao x ncao).
std::vector<double> pushdown(const std::vector<double> &C, int ncao, int npao,
                            const std::vector<double> &M) {
  std::vector<double> MC(static_cast<std::size_t>(ncao) * npao, 0.0); // M C  (ncao x npao)
  for (int I = 0; I < ncao; ++I)
    for (int P = 0; P < npao; ++P) {
      double s = 0;
      for (int J = 0; J < ncao; ++J)
        s += M[static_cast<std::size_t>(I) * ncao + J] * C[static_cast<std::size_t>(J) * npao + P];
      MC[static_cast<std::size_t>(I) * npao + P] = s;
    }
  std::vector<double> out(static_cast<std::size_t>(npao) * npao, 0.0);
  for (int P = 0; P < npao; ++P)
    for (int Q = 0; Q < npao; ++Q) {
      double s = 0;
      for (int I = 0; I < ncao; ++I)
        s += C[static_cast<std::size_t>(I) * npao + P] * MC[static_cast<std::size_t>(I) * npao + Q];
      out[static_cast<std::size_t>(P) * npao + Q] = s;
    }
  return out;
}

// C M_prim C^T reference for a symmetric 1e operator built by prim_builder.
template <class Builder>
std::vector<double> contract_ref(const ContractedBasis<double> &cb, Builder prim_builder) {
  intti::ShellBasis<double> pbasis;
  std::vector<double> C;
  const int npao = decontract(cb, pbasis, C);
  return conjugate(C, cb.nao, npao, prim_builder(pbasis));
}

// A small generally-contracted test basis exercising l = 0, 1, 2 and general
// contraction (nctr > 1): an s shell (3 primitives -> 2 contracted functions),
// a p shell (2 primitives -> 1) on a second center, and a d shell (2 primitives
// -> 2 contracted functions) -- the d shell tests the l >= 2 contraction
// bookkeeping (ncart = 6 and the cart_norm_pyscf sqrt(4 pi/(2l+1)) factor).
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
  ContractedShell<double> d;
  d.center[0] = 0; d.center[1] = 0.2; d.center[2] = 0.5;
  d.l = 2;
  d.alpha = {1.5, 0.5};
  d.coeff = {0.7, 0.4,   // contracted function 0
             0.2, 0.9};  // contracted function 1
  return intti::make_contracted_basis<double>({s, p, d});
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

TEST(Contracted, NuclearVsDecontractRecontract) {
  auto cb = test_basis();
  auto grid = intti::make_tgrid(intti::coulomb<double>());
  std::vector<intti::PointCharge<double>> charges = {
      {-3.0, {0.0, 0.0, 0.0}}, {-1.0, {0.0, 0.0, 1.3}}};
  auto V = intti::nuclear_matrix(cb, charges, grid);
  auto ref = contract_ref(cb, [&](const intti::ShellBasis<double> &b) {
    return intti::nuclear_matrix(b, charges, grid);
  });
  ASSERT_EQ(V.size(), ref.size());
  double worst = 0, scale = 0;
  for (std::size_t i = 0; i < V.size(); ++i) {
    worst = std::max(worst, std::abs(V[i] - ref[i]));
    scale = std::max(scale, std::abs(ref[i]));
  }
  EXPECT_LT(worst, 1e-11 * scale) << "contracted nuclear != decontract/recontract";
}

TEST(Contracted, CoulombVsDecontractRecontract) {
  auto cb = test_basis();
  const int n = cb.nao;
  auto grid = intti::make_tgrid(intti::coulomb<double>());
  // a symmetric test density over the contracted AOs
  std::vector<double> D(static_cast<std::size_t>(n) * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) D[i * n + j] = 0.1 + 0.3 * std::sin(0.7 * i + 1.3 * j);
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) {
      const double a = 0.5 * (D[i * n + j] + D[j * n + i]);
      D[i * n + j] = D[j * n + i] = a;
    }
  std::vector<double> J(static_cast<std::size_t>(n) * n, 0.0);
  intti::coulomb_build(cb, D.data(), grid, J.data());
  // reference: D_eff = C^T D C; J_eff = primitive coulomb_build; J = C J_eff C^T
  intti::ShellBasis<double> pbasis;
  std::vector<double> C;
  const int npao = decontract(cb, pbasis, C);
  auto Deff = pushdown(C, n, npao, D);
  std::vector<double> Jeff(static_cast<std::size_t>(npao) * npao, 0.0);
  intti::coulomb_build(pbasis, Deff.data(), grid, Jeff.data());
  auto ref = conjugate(C, n, npao, Jeff);
  double worst = 0, scale = 0;
  for (std::size_t i = 0; i < J.size(); ++i) {
    worst = std::max(worst, std::abs(J[i] - ref[i]));
    scale = std::max(scale, std::abs(ref[i]));
  }
  EXPECT_LT(worst, 1e-11 * scale) << "contracted J != decontract/recontract";
}

// A symmetric test density over n contracted AOs.
std::vector<double> sym_density(int n) {
  std::vector<double> D(static_cast<std::size_t>(n) * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) D[i * n + j] = 0.1 + 0.3 * std::sin(0.7 * i + 1.3 * j);
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) {
      const double a = 0.5 * (D[i * n + j] + D[j * n + i]);
      D[i * n + j] = D[j * n + i] = a;
    }
  return D;
}

// Screened contracted K matches the exact build: the dropped ket pairs are each
// below tau, so the total deviation is far under a loose bound.
TEST(Contracted, ExchangeScreeningMatchesExact) {
  auto cb = test_basis();
  const int n = cb.nao;
  auto grid = intti::make_tgrid(intti::coulomb<double>());
  auto D = sym_density(n);
  std::vector<double> K0(static_cast<std::size_t>(n) * n, 0.0);
  std::vector<double> Ks(static_cast<std::size_t>(n) * n, 0.0);
  intti::exchange_build(cb, D.data(), grid, K0.data(), 0.0);
  intti::exchange_build(cb, D.data(), grid, Ks.data(), 1e-10);
  double worst = 0;
  for (std::size_t i = 0; i < K0.size(); ++i)
    worst = std::max(worst, std::abs(K0[i] - Ks[i]));
  EXPECT_LT(worst, 1e-8) << "screening corrupts K beyond tau";
}

TEST(Contracted, ExchangeVsDecontractRecontract) {
  auto cb = test_basis();
  const int n = cb.nao;
  auto grid = intti::make_tgrid(intti::coulomb<double>());
  std::vector<double> D(static_cast<std::size_t>(n) * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) D[i * n + j] = 0.1 + 0.3 * std::sin(0.7 * i + 1.3 * j);
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) {
      const double a = 0.5 * (D[i * n + j] + D[j * n + i]);
      D[i * n + j] = D[j * n + i] = a;
    }
  std::vector<double> K(static_cast<std::size_t>(n) * n, 0.0);
  intti::exchange_build(cb, D.data(), grid, K.data());
  // reference: D_eff = C^T D C; K_eff = primitive exchange (tau=0); K = C K_eff C^T
  intti::ShellBasis<double> pbasis;
  std::vector<double> C;
  const int npao = decontract(cb, pbasis, C);
  auto Deff = pushdown(C, n, npao, D);
  std::vector<double> Keff(static_cast<std::size_t>(npao) * npao, 0.0);
  intti::exchange_build(pbasis, Deff.data(), grid, Keff.data(), 0.0);
  auto ref = conjugate(C, n, npao, Keff);
  double worst = 0, scale = 0;
  for (std::size_t i = 0; i < K.size(); ++i) {
    worst = std::max(worst, std::abs(K[i] - ref[i]));
    scale = std::max(scale, std::abs(ref[i]));
  }
  EXPECT_LT(worst, 1e-11 * scale) << "contracted K != decontract/recontract";
}

// Symmetry and offset bookkeeping: S is symmetric and its dimension is the sum
// of nctr*ncart(l) over shells.
TEST(Contracted, SymmetricAndSized) {
  auto cb = test_basis();
  auto S = intti::overlap_matrix(cb);
  EXPECT_EQ(cb.nao, 2 * 1 + 1 * 3 + 2 * 6); // s(2x1) + p(1x3) + d(2x6) = 17
  const int n = cb.nao;
  double asym = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      asym = std::max(asym, std::abs(S[i * n + j] - S[j * n + i]));
  EXPECT_LT(asym, 1e-14);
}

// A small contracted auxiliary basis (s + p, general contraction).
ContractedBasis<double> aux_sp() {
  ContractedShell<double> s;
  s.center[0] = 0; s.center[1] = 0; s.center[2] = 0;
  s.l = 0; s.alpha = {2.5, 1.0}; s.coeff = {0.6, 0.5};
  ContractedShell<double> p;
  p.center[0] = 0.3; p.center[1] = 0; p.center[2] = 0;
  p.l = 1; p.alpha = {1.8, 0.9}; p.coeff = {0.55, 0.5};
  return intti::make_contracted_basis<double>({s, p});
}

// Contracted 2-center Coulomb (P|Q) == C_aux (primitive (P|Q)) C_aux^T.
TEST(Contracted, Coulomb2cVsDecontract) {
  auto aux = aux_sp();
  auto grid = intti::make_tgrid(intti::coulomb<double>());
  auto M = intti::coulomb_2c(aux, grid);
  intti::ShellBasis<double> pb;
  std::vector<double> C;
  const int npao = decontract(aux, pb, C);
  auto ref = conjugate(C, aux.nao, npao, intti::coulomb_2c(pb, grid));
  ASSERT_EQ(M.size(), ref.size());
  double worst = 0, scale = 0;
  for (std::size_t i = 0; i < M.size(); ++i) {
    worst = std::max(worst, std::abs(M[i] - ref[i]));
    scale = std::max(scale, std::abs(ref[i]));
  }
  EXPECT_LT(worst, 1e-11 * scale) << "contracted (P|Q) != decontract/recontract";
}

// Contracted 3-center Coulomb (mu nu | P) == the C_orb x C_orb x C_aux transform
// of the primitive (mu nu | P).
TEST(Contracted, Coulomb3cVsDecontract) {
  auto orb = aux_sp(); // s + p (orbital basis)
  auto aux = aux_sp(); // s + p (auxiliary basis)
  auto grid = intti::make_tgrid(intti::coulomb<double>());
  auto T = intti::coulomb_3c(orb, aux, grid);
  const int no = orb.nao, na = aux.nao;
  intti::ShellBasis<double> pbo, pba;
  std::vector<double> Co, Ca;
  const int npo = decontract(orb, pbo, Co); // Co: no x npo
  const int npa = decontract(aux, pba, Ca); // Ca: na x npa
  auto Tp = intti::coulomb_3c(pbo, pba, grid); // (npo, npo, npa)
  // Tc[I][J][P] = sum_ijp Co[I][i] Co[J][j] Ca[P][p] Tp[i][j][p]
  std::vector<double> ref((std::size_t)no * no * na, 0.0);
  for (int I = 0; I < no; ++I)
    for (int J = 0; J < no; ++J)
      for (int P = 0; P < na; ++P) {
        double s = 0;
        for (int i = 0; i < npo; ++i) {
          const double ci = Co[(std::size_t)I * npo + i];
          if (ci == 0) continue;
          for (int j = 0; j < npo; ++j) {
            const double cij = ci * Co[(std::size_t)J * npo + j];
            if (cij == 0) continue;
            for (int p = 0; p < npa; ++p)
              s += cij * Ca[(std::size_t)P * npa + p] *
                   Tp[((std::size_t)i * npo + j) * npa + p];
          }
        }
        ref[((std::size_t)I * no + J) * na + P] = s;
      }
  ASSERT_EQ(T.size(), ref.size());
  double worst = 0, scale = 0;
  for (std::size_t k = 0; k < T.size(); ++k) {
    worst = std::max(worst, std::abs(T[k] - ref[k]));
    scale = std::max(scale, std::abs(ref[k]));
  }
  EXPECT_LT(worst, 1e-11 * scale) << "contracted (mu nu|P) != decontract/recontract";
}

} // namespace
