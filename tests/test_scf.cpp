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
#include "intti/deriv.hpp"
#include "intti/erigrad.hpp"
#include "intti/fock.hpp"
#include "intti/nuclear.hpp"
#include "intti/oneel.hpp"
#include "intti/ri.hpp"
#include "intti/rigrad.hpp"
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
  for (int it = 0; it < 500; ++it) {
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
    // tight: an RHF gradient at a non-stationary density carries an error
    // LINEAR in the density error, so a 1e-7 density is only good for ~1e-8
    // forces -- far too loose for the 1e-9 the gradient test asks for.
    if (it > 0 && std::abs(E - Eprev) < 1e-13 && dmax < 1e-11) {
      out.E = E;
      out.converged = true;
      return out;
    }
    Eprev = E;
    out.E = E;
  }
  return out;
}

/// Atom index of each shell returned by orbital_shells().
std::vector<int> orbital_shell_atoms() {
  std::vector<int> a;
  for (int i = 0; i < 8; ++i) a.push_back(0); // O: 5 s + 3 p
  for (int i = 0; i < 3; ++i) a.push_back(1); // H1: 3 s
  for (int i = 0; i < 3; ++i) a.push_back(2); // H2: 3 s
  return a;
}

/// Atom index of each shell returned by auxiliary_shells().
std::vector<int> auxiliary_shell_atoms() {
  std::vector<int> a;
  for (int i = 0; i < 10; ++i) a.push_back(0); // O: 5 s + 3 p + 2 d
  for (int i = 0; i < 4; ++i) a.push_back(1);  // H1: 3 s + 1 p
  for (int i = 0; i < 4; ++i) a.push_back(2);  // H2: 3 s + 1 p
  return a;
}

/// Analytic RHF gradient, assembled from the library's derivative builders.
///
///   dE/dR_A = sum_mn D_mn dH_mn/dR_A + dE_2e/dR_A - sum_mn W_mn dS_mn/dR_A
///             + dE_nuc/dR_A,   W_mn = 2 sum_i^occ eps_i C_mi C_ni.
///
/// The 1e builders return the BRA gradient <grad mu|..|nu> (PySCF's int1e_ip*),
/// and d/dR_A of a function of (r - R_A) is minus the electronic gradient, so
/// dX_mn/dR_A = -Xip[m][n] when mu sits on A, -Xip[n][m] when nu does. The
/// Hellmann-Feynman term -- the nuclear attraction moving with its own charge --
/// comes from translational invariance of the one-charge integral,
/// d/dR_C = -(d/dR_bra + d/dR_ket), so it is one nuclear_deriv per atom with
/// that atom as the only charge.
template <class Grad2e>
std::vector<std::array<double, 3>> rhf_gradient(const std::vector<Shell> &orb_sh,
                                                const std::vector<int> &shell_atom,
                                                const std::vector<Atom> &atoms,
                                                int nocc, const Scf &scf,
                                                const intti::TGrid<double> &grid,
                                                Grad2e &&grad2e) {
  auto orb = intti::make_basis(orb_sh);
  const int n = orb.nao, na = static_cast<int>(atoms.size());
  const std::size_t n2 = static_cast<std::size_t>(n) * n;
  // AO -> atom
  std::vector<int> ao_atom(n);
  for (std::size_t s = 0; s < orb_sh.size(); ++s) {
    const int off = orb.ao_off[s], nc = intti::ncart(orb_sh[s].l);
    for (int k = 0; k < nc; ++k) ao_atom[off + k] = shell_atom[s];
  }
  std::vector<double> Z;
  std::vector<std::array<double, 3>> R;
  for (const auto &a : atoms) {
    Z.push_back(a.Z);
    R.push_back(a.R);
  }
  // energy-weighted density
  std::vector<double> W(n2, 0.0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double s = 0;
      for (int k = 0; k < nocc; ++k) s += scf.eps[k] * scf.C[i * n + k] * scf.C[j * n + k];
      W[i * n + j] = 2.0 * s;
    }
  const auto Sip = intti::overlap_deriv(orb);
  const auto Tip = intti::kinetic_deriv(orb);
  const auto Vip = intti::nuclear_deriv(orb, intti::nuclei_as_charges(Z, R), grid);

  std::vector<std::array<double, 3>> g(na, {0.0, 0.0, 0.0});
  for (int c = 0; c < 3; ++c)
    for (int m = 0; m < n; ++m)
      for (int p = 0; p < n; ++p) {
        const std::size_t mp = static_cast<std::size_t>(m) * n + p;
        const std::size_t pm = static_cast<std::size_t>(p) * n + m;
        const double hb = -scf.D[mp] * (Tip[c][mp] + Vip[c][mp]) + W[mp] * Sip[c][mp];
        const double hk = -scf.D[mp] * (Tip[c][pm] + Vip[c][pm]) + W[mp] * Sip[c][pm];
        g[ao_atom[m]][c] += hb;
        g[ao_atom[p]][c] += hk;
      }
  // Hellmann-Feynman: one charge at a time
  for (int A = 0; A < na; ++A) {
    const std::vector<double> ZA = {Z[A]};
    const std::vector<std::array<double, 3>> RA = {R[A]};
    const auto VA = intti::nuclear_deriv(orb, intti::nuclei_as_charges(ZA, RA), grid);
    for (int c = 0; c < 3; ++c) {
      double s = 0;
      for (std::size_t i = 0; i < n2; ++i) s += scf.D[i] * VA[c][i];
      g[A][c] += 2.0 * s; // D symmetric: sum D (V + V^T) = 2 sum D V
    }
  }
  // two-electron part -- whichever engine is under test, accumulated per atom
  grad2e(orb, scf.D.data(), g);
  // nuclear repulsion
  for (int A = 0; A < na; ++A)
    for (int B = 0; B < na; ++B) {
      if (A == B) continue;
      double r2 = 0, d[3];
      for (int c = 0; c < 3; ++c) {
        d[c] = atoms[A].R[c] - atoms[B].R[c];
        r2 += d[c] * d[c];
      }
      const double r3 = r2 * std::sqrt(r2);
      for (int c = 0; c < 3; ++c) g[A][c] -= Z[A] * Z[B] * d[c] / r3;
    }
  return g;
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

// PySCF reference gradient, dE/dR in Ha/bohr, one row per atom (same script).
constexpr double kPyscfGrad[3][3] = {
    {-0.000000000000000, 0.000000000000001, 0.012969965273731},
    {0.000000000000000, -0.014363854679873, -0.006484982636865},
    {-0.000000000000000, 0.014363854679869, -0.006484982636865}};

TEST(Scf, RhfGradientMatchesPyscf) {
  // Capstone stage B: the analytic force, assembled from the derivative
  // builders on top of a converged SCF. This is the first EXTERNAL reference
  // for the two-electron derivatives -- until now they were pinned only against
  // our own finite differences, which cannot catch a shared convention error.
  auto grid = intti::make_tgrid(intti::coulomb());
  const auto scf =
      rhf(orbital_shells(), kAtoms, 5, grid,
          [&](const intti::ShellBasis<double> &orb, const double *D, double *J, double *K) {
            intti::coulomb_build(orb, D, grid, J, 1e-14);
            intti::exchange_build(orb, D, grid, K, 1e-14);
          });
  ASSERT_TRUE(scf.converged);
  const auto sa = orbital_shell_atoms();
  const auto g = rhf_gradient(
      orbital_shells(), sa, kAtoms, 5, scf, grid,
      [&](const intti::ShellBasis<double> &orb, const double *D,
          std::vector<std::array<double, 3>> &acc) {
        const auto g2 = intti::two_electron_gradient(orb, D, grid);
        for (std::size_t s = 0; s < g2.size(); ++s)
          for (int c = 0; c < 3; ++c) acc[sa[s]][c] += g2[s][c];
      });
  ASSERT_EQ(g.size(), 3u);
  double mag = 0;
  for (int a = 0; a < 3; ++a)
    for (int c = 0; c < 3; ++c) mag = std::max(mag, std::abs(kPyscfGrad[a][c]));
  ASSERT_GT(mag, 1e-3) << "reference gradient is trivially zero";
  for (int a = 0; a < 3; ++a)
    for (int c = 0; c < 3; ++c)
      EXPECT_NEAR(g[a][c], kPyscfGrad[a][c], 1e-9) << "atom " << a << " comp " << c;
  // translational invariance: the total force on the molecule must vanish
  for (int c = 0; c < 3; ++c) {
    double t = 0;
    for (int a = 0; a < 3; ++a) t += g[a][c];
    EXPECT_NEAR(t, 0.0, 1e-9) << "net force, component " << c;
  }
}

// PySCF reference gradient for the DENSITY-FITTED SCF (same script).
constexpr double kPyscfDfGrad[3][3] = {
    {0.000000000000002, -0.000000000000000, 0.015577724938646},
    {-0.000000000000001, -0.015219419166869, -0.007788862469326},
    {0.000000000000001, 0.015219419166862, -0.007788862469325}};

TEST(Scf, RiRhfGradientMatchesPyscfDensityFitting) {
  // The RI force is what a density-fitted geometry optimisation actually runs
  // on. Unlike the exact gradient it also carries forces on the AUXILIARY
  // shells: those are atom-centred, so they move with their atom and must be
  // accumulated into the same per-atom total. Dropping them silently would look
  // almost right -- hence an external reference rather than our own FD.
  auto grid = intti::make_tgrid(intti::coulomb());
  auto orbb = intti::make_basis(orbital_shells());
  auto auxb = intti::make_basis(auxiliary_shells());
  const auto fit = intti::ri_fit(orbb, auxb, grid);
  const auto scf =
      rhf(orbital_shells(), kAtoms, 5, grid,
          [&](const intti::ShellBasis<double> &, const double *D, double *J, double *K) {
            intti::ri_jk(fit, D, J, K);
          });
  ASSERT_TRUE(scf.converged);
  const auto sa = orbital_shell_atoms();
  const auto xa = auxiliary_shell_atoms();
  const auto g = rhf_gradient(
      orbital_shells(), sa, kAtoms, 5, scf, grid,
      [&](const intti::ShellBasis<double> &orb, const double *D,
          std::vector<std::array<double, 3>> &acc) {
        const auto gj = intti::ri_j_gradient(orb, auxb, D, grid);
        const auto gk = intti::ri_k_gradient(orb, auxb, D, grid);
        for (std::size_t s = 0; s < gj.forb.size(); ++s)
          for (int c = 0; c < 3; ++c) acc[sa[s]][c] += gj.forb[s][c] + gk.forb[s][c];
        for (std::size_t s = 0; s < gj.faux.size(); ++s)
          for (int c = 0; c < 3; ++c) acc[xa[s]][c] += gj.faux[s][c] + gk.faux[s][c];
      });
  double mag = 0;
  for (int a = 0; a < 3; ++a)
    for (int c = 0; c < 3; ++c) mag = std::max(mag, std::abs(kPyscfDfGrad[a][c]));
  ASSERT_GT(mag, 1e-3) << "reference gradient is trivially zero";
  for (int a = 0; a < 3; ++a)
    for (int c = 0; c < 3; ++c)
      EXPECT_NEAR(g[a][c], kPyscfDfGrad[a][c], 1e-9) << "atom " << a << " comp " << c;
  for (int c = 0; c < 3; ++c) {
    double t = 0;
    for (int a = 0; a < 3; ++a) t += g[a][c];
    EXPECT_NEAR(t, 0.0, 1e-9) << "net force, component " << c;
  }
}

TEST(Scf, NuclearRepulsionMatchesPyscf) {
  // the one piece of the total energy that is not an integral -- pinned so a
  // geometry typo shows up here rather than as an integral discrepancy
  EXPECT_NEAR(nuclear_repulsion(kAtoms), 9.220256432808192, 1e-12);
}

} // namespace
