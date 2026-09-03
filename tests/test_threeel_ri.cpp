// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/threeel_ri.hpp"

#ifdef INTTI_HAVE_QUADMATH
#include <quadmath.h>
#endif

namespace {

using intti::CartGauss;

// Exact O(nao^6) reference: E3 = sum_{abcdef} G_abcdef D_ad D_be D_cf over the
// unnormalised primitive orbitals (the same convention the RI path uses).
template <class Real>
Real exact_e3(const std::vector<intti::CartGauss<Real>> &g, const std::vector<Real> &D, int n,
              const intti::TGrid<Real> &grid) {
  Real E = 0;
  for (int a = 0; a < n; ++a)
    for (int d = 0; d < n; ++d)
      for (int b = 0; b < n; ++b)
        for (int e = 0; e < n; ++e)
          for (int c = 0; c < n; ++c)
            for (int f = 0; f < n; ++f)
              E += D[a * n + d] * D[b * n + e] * D[c * n + f] *
                   intti::detail::three_electron_raw(g[a], g[b], g[c], g[d], g[e], g[f], grid);
  return E;
}

// Direct O(nao^6) raw Fock: F_pq = dE3/dD_pq, each sextet scattered into its
// three density slots (the unnormalised-primitive analogue of three_electron_fock).
std::vector<double> direct_fock_raw(const std::vector<CartGauss<double>> &g,
                                    const std::vector<double> &D, int n,
                                    const intti::TGrid<double> &grid) {
  std::vector<double> F(static_cast<std::size_t>(n) * n, 0.0);
  for (int a = 0; a < n; ++a)
    for (int d = 0; d < n; ++d)
      for (int b = 0; b < n; ++b)
        for (int e = 0; e < n; ++e)
          for (int c = 0; c < n; ++c)
            for (int f = 0; f < n; ++f) {
              const double G =
                  intti::detail::three_electron_raw(g[a], g[b], g[c], g[d], g[e], g[f], grid);
              F[a * n + d] += G * D[b * n + e] * D[c * n + f];
              F[b * n + e] += G * D[a * n + d] * D[c * n + f];
              F[c * n + f] += G * D[a * n + d] * D[b * n + e];
            }
  return F;
}

TEST(ThreeElRI, ExactWhenAuxSpansProducts) {
  // Two s orbitals -> three distinct products (s-Gaussians at the product
  // centres). With the auxiliary basis chosen to be exactly those products, the
  // density lies in span(aux), the fit is exact, and the RI-folded energy must
  // reproduce the direct sextet sum to numerical precision.
  auto grid = intti::make_tgrid(intti::coulomb());
  auto orb = intti::make_basis<double>({{1.0, {0.0, 0.0, 0.0}, 0}, {0.8, {0.5, 0.0, 0.0}, 0}});
  const std::vector<double> D = {1.0, 0.3, 0.3, 0.7};
  const double p01 = (1.0 * 0.0 + 0.8 * 0.5) / 1.8; // product centre of the 0-1 pair
  auto aux = intti::make_basis<double>({{2.0, {0.0, 0.0, 0.0}, 0},
                                        {1.6, {0.5, 0.0, 0.0}, 0},
                                        {1.8, {p01, 0.0, 0.0}, 0}});
  const double eri = intti::three_electron_energy_ri(orb, D, aux, grid);
  const auto og = intti::detail::shellbasis_to_cartgauss(orb);
  const double eex = exact_e3(og, D, 2, grid);
  EXPECT_NEAR(eri, eex, 1e-9 * std::abs(eex)) << "RI must be exact when aux spans the products";
}

TEST(ThreeElRI, ApproximatesWithIncompleteAux) {
  // A smaller auxiliary basis that does NOT span the products: RI now
  // approximates. It should be close but measurably different from exact.
  auto grid = intti::make_tgrid(intti::coulomb());
  auto orb = intti::make_basis<double>({{1.0, {0.0, 0.0, 0.0}, 0}, {0.8, {0.5, 0.0, 0.0}, 0}});
  const std::vector<double> D = {1.0, 0.3, 0.3, 0.7};
  // two diffuse-ish auxiliaries only
  auto aux = intti::make_basis<double>({{1.9, {0.0, 0.0, 0.0}, 0}, {1.7, {0.5, 0.0, 0.0}, 0}});
  const double eri = intti::three_electron_energy_ri(orb, D, aux, grid);
  const auto og = intti::detail::shellbasis_to_cartgauss(orb);
  const double eex = exact_e3(og, D, 2, grid);
  EXPECT_GT(std::abs(eri - eex), 1e-9 * std::abs(eex)) << "incomplete aux should not be exact";
  EXPECT_NEAR(eri, eex, 0.2 * std::abs(eex)) << "but the RI fit should still be reasonable";
}

TEST(ThreeElRI, ExactForHigherLProducts) {
  // A single p-shell's pairwise products chi_pi chi_pj = (r-A)_i (r-A)_j
  // e^{-2a(r-A)^2} span exactly one d-shell at the same centre with exponent 2a.
  // So RI is again exact -- and this exercises the cart_comp AO ordering for
  // both the orbital p-shell and the auxiliary d-shell.
  auto grid = intti::make_tgrid(intti::coulomb());
  const double a = 0.9;
  auto orb = intti::make_basis<double>({{a, {0.1, 0.2, 0.3}, 1}});          // p-shell, nao=3
  auto aux = intti::make_basis<double>({{2 * a, {0.1, 0.2, 0.3}, 2}});      // d-shell, naux=6
  // symmetric 3x3 density over the p components
  const std::vector<double> D = {0.7, 0.2, -0.1, 0.2, 0.5, 0.15, -0.1, 0.15, 0.9};
  const double eri = intti::three_electron_energy_ri(orb, D, aux, grid);
  const auto og = intti::detail::shellbasis_to_cartgauss(orb);
  const double eex = exact_e3(og, D, 3, grid);
  EXPECT_NEAR(eri, eex, 1e-9 * std::abs(eex)) << "RI must be exact for the p x p -> d products";
}

TEST(ThreeElRI, FockMatchesDirectWhenAuxSpansProducts) {
  // When aux spans the products the RI fold is exact, so the RI-folded Fock must
  // equal the direct raw Fock (an independent O(nao^6) evaluation).
  auto grid = intti::make_tgrid(intti::coulomb());
  // s case
  {
    auto orb = intti::make_basis<double>({{1.0, {0.0, 0.0, 0.0}, 0}, {0.8, {0.5, 0.0, 0.0}, 0}});
    const std::vector<double> D = {1.0, 0.3, 0.3, 0.7};
    const double p01 = (1.0 * 0.0 + 0.8 * 0.5) / 1.8;
    auto aux = intti::make_basis<double>({{2.0, {0.0, 0.0, 0.0}, 0},
                                          {1.6, {0.5, 0.0, 0.0}, 0},
                                          {1.8, {p01, 0.0, 0.0}, 0}});
    const auto Fri = intti::three_electron_fock_ri(orb, D, aux, grid);
    const auto Fex = direct_fock_raw(intti::detail::shellbasis_to_cartgauss(orb), D, 2, grid);
    double mx = 0, md = 0;
    for (std::size_t i = 0; i < Fri.size(); ++i) {
      mx = std::max(mx, std::abs(Fex[i]));
      md = std::max(md, std::abs(Fri[i] - Fex[i]));
    }
    EXPECT_LT(md, 1e-8 * mx) << "s: RI Fock != direct Fock";
  }
  // p x p -> d case (exercises cart_comp ordering in the Fock path)
  {
    const double a = 0.9;
    auto orb = intti::make_basis<double>({{a, {0.1, 0.2, 0.3}, 1}});
    auto aux = intti::make_basis<double>({{2 * a, {0.1, 0.2, 0.3}, 2}});
    const std::vector<double> D = {0.7, 0.2, -0.1, 0.2, 0.5, 0.15, -0.1, 0.15, 0.9};
    const auto Fri = intti::three_electron_fock_ri(orb, D, aux, grid);
    const auto Fex = direct_fock_raw(intti::detail::shellbasis_to_cartgauss(orb), D, 3, grid);
    double mx = 0, md = 0;
    for (std::size_t i = 0; i < Fri.size(); ++i) {
      mx = std::max(mx, std::abs(Fex[i]));
      md = std::max(md, std::abs(Fri[i] - Fex[i]));
    }
    EXPECT_LT(md, 1e-8 * mx) << "p: RI Fock != direct Fock";
  }
}

TEST(ThreeElRI, CholeskyAuxEnergyMatchesExact) {
  // The in-library Cholesky auxiliary (product shells over the CD-selected
  // pairs) spans rho_D at a tight threshold, so the folded energy matches the
  // direct sextet sum -- no external auxiliary basis supplied.
  auto grid = intti::make_tgrid(intti::coulomb());
  intti::CholeskyOptions<double> opt;
  opt.tau = 1e-12;
  {
    auto orb = intti::make_basis<double>({{1.0, {0.0, 0.0, 0.0}, 0},
                                          {0.8, {0.5, 0.0, 0.0}, 0},
                                          {1.3, {0.0, 0.4, 0.0}, 0}});
    const std::vector<double> D = {1.0, 0.3, 0.1, 0.3, 0.7, 0.2, 0.1, 0.2, 0.9};
    const double Ecd = intti::three_electron_energy_cd(orb, D, grid, opt);
    const double Eex = exact_e3(intti::detail::shellbasis_to_cartgauss(orb), D, 3, grid);
    EXPECT_NEAR(Ecd, Eex, 1e-6 * std::abs(Eex)) << "s CD-aux energy != exact";
  }
  {
    const double a = 0.9;
    auto orb = intti::make_basis<double>({{a, {0.1, 0.2, 0.3}, 1}});
    const std::vector<double> D = {0.7, 0.2, -0.1, 0.2, 0.5, 0.15, -0.1, 0.15, 0.9};
    const double Ecd = intti::three_electron_energy_cd(orb, D, grid, opt);
    const double Eex = exact_e3(intti::detail::shellbasis_to_cartgauss(orb), D, 3, grid);
    EXPECT_NEAR(Ecd, Eex, 1e-6 * std::abs(Eex)) << "p CD-aux energy != exact";
  }
}

TEST(ThreeElRI, CdEnergyLongDouble) {
  // The CD-aux 3-body folding runs at long double end to end: precision-generic
  // pivoted Cholesky (one-step, no LAPACK) + Jacobi metric solve.
  using LD = long double;
  auto grid = intti::make_tgrid(intti::coulomb<LD>());
  auto orb = intti::make_basis<LD>(
      {{1.0L, {0.0L, 0.0L, 0.0L}, 0}, {0.8L, {0.5L, 0.0L, 0.0L}, 0}, {1.3L, {0.0L, 0.4L, 0.0L}, 0}});
  const std::vector<LD> D = {1.0L, 0.3L, 0.1L, 0.3L, 0.7L, 0.2L, 0.1L, 0.2L, 0.9L};
  intti::CholeskyOptions<LD> opt;
  opt.tau = 1e-14L;
  const LD Ecd = intti::three_electron_energy_cd(orb, D, grid, opt, static_cast<LD>(1e-14L));
  const LD Eex = exact_e3(intti::detail::shellbasis_to_cartgauss(orb), D, 3, grid);
  EXPECT_LT(static_cast<double>(std::abs(Ecd - Eex)), 1e-9 * static_cast<double>(std::abs(Eex)));
}

#ifdef INTTI_HAVE_QUADMATH
TEST(ThreeElRI, CdEnergyQuad) {
  using Q = __float128;
  auto q = [](const char *s) { return strtoflt128(s, nullptr); };
  auto grid = intti::make_tgrid(intti::coulomb<Q>());
  auto orb = intti::make_basis<Q>(
      {{q("1.0"), {q("0"), q("0"), q("0")}, 0}, {q("0.8"), {q("0.5"), q("0"), q("0")}, 0}});
  const std::vector<Q> D = {q("1.0"), q("0.3"), q("0.3"), q("0.7")};
  intti::CholeskyOptions<Q> opt;
  opt.tau = q("1e-28");
  const Q Ecd = intti::three_electron_energy_cd(orb, D, grid, opt, q("1e-28"));
  const Q Eex = exact_e3(intti::detail::shellbasis_to_cartgauss(orb), D, 2, grid);
  EXPECT_LT(static_cast<double>(fabsq(Ecd - Eex)), 1e-12 * static_cast<double>(fabsq(Eex)));
}
#endif

// Direct effective two-body Coulomb: J_mu nu = sum_{la si, c f} Db_la si Dc_c f
// G_{mu la c, nu si f}, electron 2 contracted by Db and electron 3 by Dc.
std::vector<double> direct_eff_coulomb(const std::vector<CartGauss<double>> &g,
                                       const std::vector<double> &Dc, const std::vector<double> &Db,
                                       int n, const intti::TGrid<double> &grid) {
  std::vector<double> J(static_cast<std::size_t>(n) * n, 0.0);
  for (int mu = 0; mu < n; ++mu)
    for (int nu = 0; nu < n; ++nu) {
      double j = 0;
      for (int la = 0; la < n; ++la)
        for (int si = 0; si < n; ++si) {
          const double Dl = Db[la * n + si];
          if (Dl == 0) continue;
          for (int c = 0; c < n; ++c)
            for (int f = 0; f < n; ++f)
              j += Dl * Dc[c * n + f] *
                   intti::detail::three_electron_raw(g[mu], g[la], g[c], g[nu], g[si], g[f], grid);
        }
      J[mu * n + nu] = j;
    }
  return J;
}

TEST(ThreeElRI, EffectiveCoulombMatchesDirect) {
  auto grid = intti::make_tgrid(intti::coulomb());
  auto cmp = [](const std::vector<double> &A, const std::vector<double> &B, double tol,
                const char *msg) {
    double mx = 0, md = 0;
    for (std::size_t i = 0; i < A.size(); ++i) {
      mx = std::max(mx, std::abs(B[i]));
      md = std::max(md, std::abs(A[i] - B[i]));
    }
    EXPECT_LT(md, tol * mx) << msg;
  };
  // s orbitals, aux = the three products; check Dc=Db and Dc!=Db.
  {
    auto orb = intti::make_basis<double>({{1.0, {0.0, 0.0, 0.0}, 0}, {0.8, {0.5, 0.0, 0.0}, 0}});
    const double p01 = (1.0 * 0.0 + 0.8 * 0.5) / 1.8;
    auto aux = intti::make_basis<double>({{2.0, {0.0, 0.0, 0.0}, 0},
                                          {1.6, {0.5, 0.0, 0.0}, 0},
                                          {1.8, {p01, 0.0, 0.0}, 0}});
    const auto og = intti::detail::shellbasis_to_cartgauss(orb);
    const std::vector<double> D = {1.0, 0.3, 0.3, 0.7};
    cmp(intti::three_electron_effective_coulomb_ri(orb, D, D, aux, grid),
        direct_eff_coulomb(og, D, D, 2, grid), 1e-8, "s Dc=Db");
    const std::vector<double> Dc = {0.9, -0.2, -0.2, 0.4}, Db = {0.5, 0.1, 0.1, 1.1};
    cmp(intti::three_electron_effective_coulomb_ri(orb, Dc, Db, aux, grid),
        direct_eff_coulomb(og, Dc, Db, 2, grid), 1e-8, "s Dc!=Db");
  }
  // p x p -> d (cart_comp ordering in the effective-Coulomb path)
  {
    const double a = 0.9;
    auto orb = intti::make_basis<double>({{a, {0.1, 0.2, 0.3}, 1}});
    auto aux = intti::make_basis<double>({{2 * a, {0.1, 0.2, 0.3}, 2}});
    const auto og = intti::detail::shellbasis_to_cartgauss(orb);
    const std::vector<double> D = {0.7, 0.2, -0.1, 0.2, 0.5, 0.15, -0.1, 0.15, 0.9};
    cmp(intti::three_electron_effective_coulomb_ri(orb, D, D, aux, grid),
        direct_eff_coulomb(og, D, D, 3, grid), 1e-8, "p Dc=Db");
  }
}

// Direct effective two-body exchange: K_mu la = sum_{nu si} Db_nu si sum_{c f}
// Dc_c f G_{mu la c, nu si f} (electron 3 contracted by Dc, Db couples the kets).
std::vector<double> direct_eff_exchange(const std::vector<CartGauss<double>> &g,
                                        const std::vector<double> &Dc,
                                        const std::vector<double> &Db, int n,
                                        const intti::TGrid<double> &grid) {
  std::vector<double> K(static_cast<std::size_t>(n) * n, 0.0);
  for (int mu = 0; mu < n; ++mu)
    for (int la = 0; la < n; ++la) {
      double k = 0;
      for (int nu = 0; nu < n; ++nu)
        for (int si = 0; si < n; ++si) {
          const double Dbns = Db[nu * n + si];
          if (Dbns == 0) continue;
          for (int c = 0; c < n; ++c)
            for (int f = 0; f < n; ++f)
              k += Dbns * Dc[c * n + f] *
                   intti::detail::three_electron_raw(g[mu], g[la], g[c], g[nu], g[si], g[f], grid);
        }
      K[mu * n + la] = k;
    }
  return K;
}

TEST(ThreeElRI, EffectiveExchangeMatchesDirect) {
  auto grid = intti::make_tgrid(intti::coulomb());
  auto cmp = [](const std::vector<double> &A, const std::vector<double> &B, double tol,
                const char *msg) {
    double mx = 0, md = 0;
    for (std::size_t i = 0; i < A.size(); ++i) {
      mx = std::max(mx, std::abs(B[i]));
      md = std::max(md, std::abs(A[i] - B[i]));
    }
    EXPECT_LT(md, tol * mx) << msg;
  };
  {
    auto orb = intti::make_basis<double>({{1.0, {0.0, 0.0, 0.0}, 0}, {0.8, {0.5, 0.0, 0.0}, 0}});
    const double p01 = (1.0 * 0.0 + 0.8 * 0.5) / 1.8;
    auto aux = intti::make_basis<double>({{2.0, {0.0, 0.0, 0.0}, 0},
                                          {1.6, {0.5, 0.0, 0.0}, 0},
                                          {1.8, {p01, 0.0, 0.0}, 0}});
    const auto og = intti::detail::shellbasis_to_cartgauss(orb);
    const std::vector<double> Dc = {0.9, -0.2, -0.2, 0.4}, Db = {0.5, 0.1, 0.1, 1.1};
    cmp(intti::three_electron_effective_exchange_ri(orb, Dc, Db, aux, grid),
        direct_eff_exchange(og, Dc, Db, 2, grid), 1e-8, "s exchange Dc!=Db");
  }
  {
    const double a = 0.9;
    auto orb = intti::make_basis<double>({{a, {0.1, 0.2, 0.3}, 1}});
    auto aux = intti::make_basis<double>({{2 * a, {0.1, 0.2, 0.3}, 2}});
    const auto og = intti::detail::shellbasis_to_cartgauss(orb);
    const std::vector<double> D = {0.7, 0.2, -0.1, 0.2, 0.5, 0.15, -0.1, 0.15, 0.9};
    cmp(intti::three_electron_effective_exchange_ri(orb, D, D, aux, grid),
        direct_eff_exchange(og, D, D, 3, grid), 1e-8, "p exchange");
  }
}

TEST(ThreeElRI, FockIsEnergyGradientAndObeysEuler) {
  // The RI Fock is the gradient of the RI energy (finite difference) and, by
  // cubic homogeneity of E3^RI in D, satisfies sum_{mu nu} F_{mu nu} D_{mu nu} = 3E.
  auto grid = intti::make_tgrid(intti::coulomb());
  auto orb = intti::make_basis<double>({{1.0, {0.0, 0.0, 0.0}, 0}, {0.8, {0.5, 0.0, 0.0}, 0}});
  auto aux = intti::make_basis<double>(
      {{1.9, {0.0, 0.0, 0.0}, 0}, {1.7, {0.5, 0.0, 0.0}, 0}, {2.3, {0.25, 0.0, 0.0}, 0}});
  const std::vector<double> D = {1.0, 0.3, 0.2, 0.7};
  const int n = 2;
  const auto F = intti::three_electron_fock_ri(orb, D, aux, grid);
  const double E = intti::three_electron_energy_ri(orb, D, aux, grid);
  double trace = 0;
  for (int i = 0; i < n * n; ++i) trace += F[i] * D[i];
  EXPECT_NEAR(trace, 3 * E, 1e-9 * std::abs(E)) << "Euler sum F D = 3E";
  const double h = 1e-4;
  for (int p = 0; p < n; ++p)
    for (int q = 0; q < n; ++q) {
      auto shift = [&](double s) {
        std::vector<double> Dm = D;
        Dm[p * n + q] += s;
        return intti::three_electron_energy_ri(orb, Dm, aux, grid);
      };
      const double fd = (shift(-2 * h) - 8 * shift(-h) + 8 * shift(h) - shift(2 * h)) / (12 * h);
      EXPECT_NEAR(F[p * n + q], fd, 1e-6 * (std::abs(fd) + 1)) << "F[" << p << "," << q << "]";
    }
}

} // namespace
