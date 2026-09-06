// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

// RI-RHF driven entirely by libintti matrix builders, against PySCF df.RHF.
//
// This is the milestone-13 oracle and the first stage of the program capstone:
// every earlier PySCF comparison validates ONE integral class in isolation, so
// nothing so far checks that the pieces COMPOSE into a correct self-consistent
// field. Here S, T, V and the RI J/K all enter one energy.
//
// Why the comparison is convention-free. The RHF energy is invariant under any
// nonsingular transformation of the AO basis (the SCF solves a generalised
// eigenproblem with S), and the RI J/K is a projection onto the auxiliary span
// in the Coulomb metric, so it is invariant under nonsingular transformations of
// the auxiliary basis too. Normalisation and shell ordering therefore cannot
// affect the number; only the SPANS have to match. The orbital basis below is
// s/p only, which is the same span Cartesian or spherical; the auxiliary basis
// carries d functions, so the reference is generated with cart=True.
//
// The driver lives here rather than in include/intti: libintti is an integrals
// library, and the point of the capstone is that a complete SCF can be written
// against the public MATRIX builders without ever touching a quartet.

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "intti/cholesky.hpp" // detail::syevd
#include "intti/fock.hpp"
#include "intti/nuclear.hpp"
#include "intti/oneel.hpp"
#include "intti/ri.hpp"
#include "intti/tgrid.hpp"

namespace {

using Shell = intti::PrimitiveShell<double>;

struct Atom {
  double Z;
  std::array<double, 3> R;
};

// H2O at the experimental geometry, in bohr (matches the reference script).
const std::vector<Atom> kAtoms = {{8.0, {0.0, 0.0, -0.1230376}},
                                  {1.0, {0.0, 1.4300472, 0.9762012}},
                                  {1.0, {0.0, -1.4300472, 0.9762012}}};

// Uncontracted s/p orbital basis: no contraction question, and s/p span the
// same space in Cartesian and spherical form.
std::vector<Shell> orbital_shells() {
  const std::vector<double> os = {130.70932, 23.808861, 6.4436083, 1.1695961, 0.3803890};
  const std::vector<double> op = {5.0331513, 1.1695961, 0.3803890};
  const std::vector<double> hs = {3.4252509, 0.6239137, 0.1688554};
  std::vector<Shell> sh;
  const auto &O = kAtoms[0].R;
  for (double a : os) sh.push_back({a, {O[0], O[1], O[2]}, 0});
  for (double a : op) sh.push_back({a, {O[0], O[1], O[2]}, 1});
  for (int h = 1; h <= 2; ++h) {
    const auto &H = kAtoms[h].R;
    for (double a : hs) sh.push_back({a, {H[0], H[1], H[2]}, 0});
  }
  return sh;
}

// Auxiliary basis spanning the orbital products (s x s, s x p, p x p -> up to d).
std::vector<Shell> auxiliary_shells() {
  const std::vector<double> as = {52.0, 13.0, 3.2, 0.8, 0.25};
  const std::vector<double> ap = {6.4, 1.6, 0.4};
  const std::vector<double> ad = {2.4, 0.6};
  const std::vector<double> bs = {6.8, 1.7, 0.42};
  const std::vector<double> bp = {1.4};
  std::vector<Shell> sh;
  const auto &O = kAtoms[0].R;
  for (double a : as) sh.push_back({a, {O[0], O[1], O[2]}, 0});
  for (double a : ap) sh.push_back({a, {O[0], O[1], O[2]}, 1});
  for (double a : ad) sh.push_back({a, {O[0], O[1], O[2]}, 2});
  for (int h = 1; h <= 2; ++h) {
    const auto &H = kAtoms[h].R;
    for (double a : bs) sh.push_back({a, {H[0], H[1], H[2]}, 0});
    for (double a : bp) sh.push_back({a, {H[0], H[1], H[2]}, 1});
  }
  return sh;
}

double nuclear_repulsion(const std::vector<Atom> &at) {
  double e = 0;
  for (std::size_t i = 0; i < at.size(); ++i)
    for (std::size_t j = i + 1; j < at.size(); ++j) {
      double r2 = 0;
      for (int d = 0; d < 3; ++d) {
        const double x = at[i].R[d] - at[j].R[d];
        r2 += x * x;
      }
      e += at[i].Z * at[j].Z / std::sqrt(r2);
    }
  return e;
}

// S^{-1/2} with a linear-dependence cutoff. detail::syevd leaves the
// eigenvectors as ROWS of the row-major matrix (see the note in ri.hpp).
std::vector<double> inverse_sqrt(std::vector<double> S, int n, double cut) {
  std::vector<double> ev(n);
  intti::detail::syevd(n, S.data(), ev.data());
  double emax = 0;
  for (double e : ev) emax = std::max(emax, e);
  std::vector<double> X(static_cast<std::size_t>(n) * n, 0.0);
  for (int k = 0; k < n; ++k) {
    if (ev[k] <= cut * emax) continue;
    const double s = 1.0 / std::sqrt(ev[k]);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) X[i * n + j] += s * S[k * n + i] * S[k * n + j];
  }
  return X;
}

struct Scf {
  double E{0};
  int iters{0};
  bool converged{false};
  std::vector<double> C, D, eps;
};

/// RHF: S/T/V from the one-electron builders, J and K from `jk`, everything at
/// the matrix level. Plain damped iteration -- the point is the energy, not the
/// convergence accelerator. `jk` is whichever two-electron engine is under test
/// (the RI fit or the fused direct builders), so the same driver validates both.
template <class JK>
Scf rhf(const std::vector<Shell> &orb_sh, const std::vector<Atom> &atoms, int nocc,
        const intti::TGrid<double> &grid, JK &&jk) {
  auto orb = intti::make_basis(orb_sh);
  const int n = orb.nao;
  const std::size_t n2 = static_cast<std::size_t>(n) * n;

  std::vector<double> Z;
  std::vector<std::array<double, 3>> R;
  for (const auto &a : atoms) {
    Z.push_back(a.Z);
    R.push_back(a.R);
  }
  const auto S = intti::overlap_matrix(orb);
  const auto T = intti::kinetic_matrix(orb);
  const auto V = intti::nuclear_matrix(orb, intti::nuclei_as_charges(Z, R), grid);
  std::vector<double> H(n2);
  for (std::size_t i = 0; i < n2; ++i) H[i] = T[i] + V[i];

  const auto X = inverse_sqrt(S, n, 1e-10);
  const double Enuc = nuclear_repulsion(atoms);

  Scf out;
  out.D.assign(n2, 0.0);
  std::vector<double> F(n2), J(n2), K(n2), Fp(n2), Dnew(n2);
  double Eprev = 0;
  for (int it = 0; it < 200; ++it) {
    jk(orb, out.D.data(), J.data(), K.data());
    for (std::size_t i = 0; i < n2; ++i) F[i] = H[i] + J[i] - 0.5 * K[i];
    double E = Enuc;
    for (std::size_t i = 0; i < n2; ++i) E += 0.5 * out.D[i] * (H[i] + F[i]);
    // F' = X F X (X symmetric), diagonalise, back-transform
    std::vector<double> tmp(n2, 0.0);
    for (int i = 0; i < n; ++i)
      for (int k = 0; k < n; ++k) {
        double s = 0;
        for (int j = 0; j < n; ++j) s += F[i * n + j] * X[j * n + k];
        tmp[i * n + k] = s;
      }
    for (int i = 0; i < n; ++i)
      for (int k = 0; k < n; ++k) {
        double s = 0;
        for (int j = 0; j < n; ++j) s += X[i * n + j] * tmp[j * n + k];
        Fp[i * n + k] = s;
      }
    out.eps.assign(n, 0.0);
    intti::detail::syevd(n, Fp.data(), out.eps.data()); // eigenvectors as rows
    out.C.assign(n2, 0.0);
    for (int i = 0; i < n; ++i)
      for (int k = 0; k < n; ++k) {
        double s = 0;
        for (int j = 0; j < n; ++j) s += X[i * n + j] * Fp[k * n + j];
        out.C[i * n + k] = s;
      }
    for (std::size_t i = 0; i < n2; ++i) Dnew[i] = 0.0;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        double s = 0;
        for (int k = 0; k < nocc; ++k) s += out.C[i * n + k] * out.C[j * n + k];
        Dnew[i * n + j] = 2.0 * s;
      }
    double dmax = 0;
    for (std::size_t i = 0; i < n2; ++i) dmax = std::max(dmax, std::abs(Dnew[i] - out.D[i]));
    const double mix = 0.6;
    for (std::size_t i = 0; i < n2; ++i) out.D[i] = mix * Dnew[i] + (1 - mix) * out.D[i];
    out.iters = it + 1;
    if (it > 0 && std::abs(E - Eprev) < 1e-11 && dmax < 1e-7) {
      out.E = E;
      out.converged = true;
      return out;
    }
    Eprev = E;
    out.E = E;
  }
  return out;
}

// PySCF references: prototype/pyscf_scf_validation.py, same molecule, the same
// uncontracted s/p orbital basis and s/p/d auxiliary basis, cart=True (20 AOs,
// 38 auxiliary functions, E_nuc = 9.220256432808192). The milestone asks for
// 1e-8 Ha; the two codes actually agree to ~5e-12, so the tests are held an
// order of magnitude tighter than that to catch regressions rather than pass.
constexpr double kPyscfRIRHF = -75.154815462790509;
constexpr double kPyscfRHF = -75.160335735903971;

TEST(Scf, RiRhfEnergyMatchesPyscfDensityFitting) {
  auto grid = intti::make_tgrid(intti::coulomb());
  const auto fit = intti::ri_fit(intti::make_basis(orbital_shells()),
                                 intti::make_basis(auxiliary_shells()), grid);
  const auto scf =
      rhf(orbital_shells(), kAtoms, 5, grid,
          [&](const intti::ShellBasis<double> &, const double *D, double *J, double *K) {
            intti::ri_jk(fit, D, J, K);
          });
  ASSERT_TRUE(scf.converged) << "SCF did not converge in " << scf.iters << " iterations";
  EXPECT_NEAR(scf.E, kPyscfRIRHF, 1e-9) << "RI-RHF total energy vs PySCF df.RHF";
}

TEST(Scf, ExactRhfEnergyMatchesPyscf) {
  // The same composition check without RI: the fused direct J and K engines
  // (which share no code with the quartet path) driven to self-consistency.
  auto grid = intti::make_tgrid(intti::coulomb());
  const auto scf =
      rhf(orbital_shells(), kAtoms, 5, grid,
          [&](const intti::ShellBasis<double> &orb, const double *D, double *J, double *K) {
            // tight Schwarz threshold: well below the 1e-9 the energy is held to
            intti::coulomb_build(orb, D, grid, J, 1e-14);
            intti::exchange_build(orb, D, grid, K, 1e-14);
          });
  ASSERT_TRUE(scf.converged) << "SCF did not converge in " << scf.iters << " iterations";
  EXPECT_NEAR(scf.E, kPyscfRHF, 1e-9) << "RHF total energy vs PySCF scf.RHF";
  // and the RI error must be small but NONZERO -- otherwise the RI test above
  // would pass just as well with an auxiliary basis that does nothing
  const double dRI = std::abs(kPyscfRIRHF - kPyscfRHF); // 5.5e-3 for this aux set
  EXPECT_GT(dRI, 1e-6) << "auxiliary basis is not actually approximating anything";
  EXPECT_LT(dRI, 5e-2) << "auxiliary basis is too poor to be a meaningful fit";
}

TEST(Scf, NuclearRepulsionMatchesPyscf) {
  // the one piece of the total energy that is not an integral -- pinned so a
  // geometry typo shows up here rather than as an integral discrepancy
  EXPECT_NEAR(nuclear_repulsion(kAtoms), 9.220256432808192, 1e-12);
}

} // namespace
