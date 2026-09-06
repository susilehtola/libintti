// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <complex>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "intti/giao.hpp"
#include "intti/giao2e.hpp"
#include "intti/quartet.hpp"

namespace {

using C = std::complex<double>;
using Shell = intti::PrimitiveShell<double>;

// geometry shared with prototype/giao_validation.py
const double kA[3] = {0.0, 0.1, -0.3};
const double kB[3] = {0.5, -0.2, 0.4};
const double kC[3] = {1.0, 0.8, 0.0};
const double kD[3] = {-0.4, 0.3, 1.1};

Shell shell(double alpha, const double *c, int l) {
  return {alpha, {c[0], c[1], c[2]}, l};
}

// component index of (lx, ly, lz) within its shell, in cart_comp order
int comp_of(int l, int lx, int ly, int lz) {
  for (int k = 0; k < intti::ncart(l); ++k) {
    int x, y, z;
    intti::cart_comp(l, k, x, y, z);
    if (x == lx && y == ly && z == lz) return k;
  }
  return -1;
}

struct Case {
  const char *name;
  int a[3], b[3], c[3], d[3];
};
const Case kCases[] = {
    {"ssss", {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
    {"psss", {1, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
    {"ppss", {1, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 0, 0}},
    {"pppp", {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}},
    {"ddpp", {2, 0, 0}, {0, 1, 1}, {1, 0, 0}, {0, 1, 0}},
};

// analytic complex-Boys references from prototype/giao_validation.py
struct Ref {
  double B;
  const char *name;
  C value;
};
const Ref kRef[] = {
    {0.0, "ssss", {6.74103108806459517e-01, 0.0}},
    {0.0, "psss", {2.16749279067938866e-01, 0.0}},
    {0.0, "ppss", {4.69038326555242108e-02, 0.0}},
    {0.0, "pppp", {1.51187211578021029e-02, 0.0}},
    {0.0, "ddpp", {1.52204166868650137e-03, 0.0}},
    {0.1, "ssss", {6.73469852404976144e-01, 1.27370823807923562e-02}},
    {0.1, "psss", {2.16597919615563267e-01, 1.30758359091596890e-03}},
    {0.1, "ppss", {4.68854928625059217e-02, -3.82301090568826674e-04}},
    {0.1, "pppp", {1.51114561185702791e-02, -2.02897420003488130e-04}},
    {0.1, "ddpp", {1.52088992163401443e-03, 8.61842538933540135e-05}},
    {1.0, "ssss", {6.13825046875959845e-01, 1.16426804025966030e-01}},
    {1.0, "psss", {2.02155574240581032e-01, 1.19477361987506622e-02}},
    {1.0, "ppss", {4.50928058513658317e-02, -3.48658274145548014e-03}},
    {1.0, "pppp", {1.44095319786463703e-02, -1.88551547907515708e-03}},
    {1.0, "ddpp", {1.40420909115434494e-03, 8.34889399041915405e-04}},
};

int lof(const int c[3]) { return c[0] + c[1] + c[2]; }

// one GIAO ERI component, through the ordinary quartet driver
C giao_eri(const Case &cs, const double Bf[3]) {
  const int la = lof(cs.a), lb = lof(cs.b), lc = lof(cs.c), ld = lof(cs.d);
  auto bra = intti::make_giao_pair(shell(0.9, kA, la), shell(1.3, kB, lb), Bf);
  auto ket = intti::make_giao_pair(shell(1.1, kC, lc), shell(0.6, kD, ld), Bf);
  auto grid = intti::make_tgrid(intti::coulomb());
  std::vector<C> out(intti::ncart(la) * intti::ncart(lb) * intti::ncart(lc) *
                     intti::ncart(ld));
  intti::eri_quartet(bra, ket, grid, out.data());
  const int ka = comp_of(la, cs.a[0], cs.a[1], cs.a[2]);
  const int kb = comp_of(lb, cs.b[0], cs.b[1], cs.b[2]);
  const int kc = comp_of(lc, cs.c[0], cs.c[1], cs.c[2]);
  const int kd = comp_of(ld, cs.d[0], cs.d[1], cs.d[2]);
  const int ncb = intti::ncart(lb), ncc = intti::ncart(lc), ncd = intti::ncart(ld);
  return out[((ka * ncb + kb) * ncc + kc) * ncd + kd];
}

TEST(GIAO, MatchesComplexBoysReference) {
  for (const auto &r : kRef) {
    const double Bf[3] = {0.0, 0.0, r.B};
    for (const auto &cs : kCases) {
      if (std::string(cs.name) != r.name) continue;
      const C v = giao_eri(cs, Bf);
      const double scale = std::abs(r.value);
      EXPECT_LT(std::abs(v - r.value), 1e-13 * scale)
          << "B=" << r.B << " case " << r.name << " got " << v << " want " << r.value;
    }
  }
}

TEST(GIAO, ZeroFieldReducesToRealERI) {
  const double Bf[3] = {0.0, 0.0, 0.0};
  auto grid = intti::make_tgrid(intti::coulomb());
  for (const auto &cs : kCases) {
    const int la = lof(cs.a), lb = lof(cs.b), lc = lof(cs.c), ld = lof(cs.d);
    const int n = intti::ncart(la) * intti::ncart(lb) * intti::ncart(lc) *
                  intti::ncart(ld);
    std::vector<C> cval(n);
    std::vector<double> rval(n);
    intti::eri_quartet(intti::make_giao_pair(shell(0.9, kA, la), shell(1.3, kB, lb), Bf),
                       intti::make_giao_pair(shell(1.1, kC, lc), shell(0.6, kD, ld), Bf),
                       grid, cval.data());
    intti::eri_quartet(intti::make_pair(shell(0.9, kA, la), shell(1.3, kB, lb)),
                       intti::make_pair(shell(1.1, kC, lc), shell(0.6, kD, ld)),
                       grid, rval.data());
    double maxr = 0;
    for (int k = 0; k < n; ++k)
      maxr = std::max(maxr, std::abs(rval[k]));
    for (int k = 0; k < n; ++k) {
      EXPECT_LT(std::abs(cval[k].real() - rval[k]), 1e-14 * maxr) << cs.name;
      EXPECT_LT(std::abs(cval[k].imag()), 1e-15 * maxr) << cs.name;
    }
  }
}

TEST(GIAO, FieldReversalConjugates) {
  // K = (1/2) B x (R_b - R_a) is odd in B, and the orbitals are otherwise
  // real, so the whole integral must conjugate when the field flips.
  for (const auto &cs : kCases) {
    const double Bp[3] = {0.2, -0.3, 0.7};
    const double Bm[3] = {-0.2, 0.3, -0.7};
    const C vp = giao_eri(cs, Bp);
    const C vm = giao_eri(cs, Bm);
    EXPECT_LT(std::abs(vm - std::conj(vp)), 1e-14 * std::abs(vp)) << cs.name;
    EXPECT_GT(std::abs(vp.imag()), 1e-6 * std::abs(vp)) << "field must do something";
  }
}

TEST(GIAO, BraSwapConjugates) {
  // swapping the two bra shells reverses K_ab, conjugating the pair density;
  // this is the sharpest check of the conjugation convention in giao.hpp
  const double Bf[3] = {0.0, 0.15, 0.4};
  auto grid = intti::make_tgrid(intti::coulomb());
  // (ab|cd) with s shells everywhere, so components are trivial
  auto ab = intti::make_giao_pair(shell(0.9, kA, 0), shell(1.3, kB, 0), Bf);
  auto ba = intti::make_giao_pair(shell(1.3, kB, 0), shell(0.9, kA, 0), Bf);
  auto cd = intti::make_giao_pair(shell(1.1, kC, 0), shell(0.6, kD, 0), Bf);
  auto dc = intti::make_giao_pair(shell(0.6, kD, 0), shell(1.1, kC, 0), Bf);
  C v_abcd, v_badc;
  intti::eri_quartet(ab, cd, grid, &v_abcd);
  intti::eri_quartet(ba, dc, grid, &v_badc);
  EXPECT_LT(std::abs(v_badc - std::conj(v_abcd)), 1e-14 * std::abs(v_abcd));
  EXPECT_GT(std::abs(v_abcd.imag()), 1e-6 * std::abs(v_abcd));
}

intti::ShellBasis<double> giao_basis() {
  return intti::make_basis<double>({
      {0.9, {kA[0], kA[1], kA[2]}, 0},
      {1.3, {kB[0], kB[1], kB[2]}, 1},
      {0.6, {kC[0], kC[1], kC[2]}, 2},
  });
}

TEST(GIAO, OverlapZeroFieldRealMatch) {
  auto bas = giao_basis();
  const double Bf[3] = {0.0, 0.0, 0.0};
  auto Sc = intti::giao_overlap(bas, Bf);
  auto Sr = intti::overlap_matrix(bas);
  double mx = 0, dev = 0;
  for (std::size_t i = 0; i < Sr.size(); ++i) {
    mx = std::max(mx, std::abs(Sr[i]));
    dev = std::max(dev, std::abs(Sc[i].real() - Sr[i]));
    dev = std::max(dev, std::abs(Sc[i].imag()));
  }
  EXPECT_LT(dev, 1e-14 * mx);
}

TEST(GIAO, OverlapFieldDerivativeVsFiniteDiff) {
  // analytic dS/dB_k against a central finite difference of the complex GIAO
  // overlap in each field direction (independent second check)
  auto bas = giao_basis();
  auto dS = intti::giao_overlap_dB(bas);
  const double h = 1e-4;
  double worst = 0, scale = 0;
  for (int k = 0; k < 3; ++k) {
    double Bp[3] = {0, 0, 0}, Bm[3] = {0, 0, 0};
    Bp[k] = h;
    Bm[k] = -h;
    auto Sp = intti::giao_overlap(bas, Bp);
    auto Sm = intti::giao_overlap(bas, Bm);
    for (std::size_t i = 0; i < Sp.size(); ++i) {
      const C fd = (Sp[i] - Sm[i]) / (2.0 * h);
      worst = std::max(worst, std::abs(fd - dS[k][i]));
      scale = std::max(scale, std::abs(dS[k][i]));
    }
  }
  EXPECT_GT(scale, 1e-3) << "derivative must be nonzero";
  EXPECT_LT(worst, 1e-8 * (scale + 1)) << "analytic dS/dB != finite difference";
}

TEST(GIAO, OverlapFieldDerivativeHermitian) {
  // S(B) is Hermitian for all B, so dS/dB_k is Hermitian; being purely
  // imaginary, its imaginary part is therefore antisymmetric (a i = -a^T i)
  auto bas = giao_basis();
  auto dS = intti::giao_overlap_dB(bas);
  const int n = bas.nao;
  for (int k = 0; k < 3; ++k) {
    double remax = 0, asym = 0, mx = 0;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        remax = std::max(remax, std::abs(dS[k][i * n + j].real()));
        asym = std::max(asym,
                        std::abs(dS[k][i * n + j].imag() + dS[k][j * n + i].imag()));
        mx = std::max(mx, std::abs(dS[k][i * n + j].imag()));
      }
    EXPECT_LT(remax, 1e-14) << "component " << k << " not purely imaginary";
    EXPECT_LT(asym, 1e-13 * (mx + 1)) << "component " << k << " not Hermitian";
  }
}

// brute-force <omega_mu | -1/2 nabla^2 | omega_nu> by real-space quadrature
// with a finite-difference Laplacian -- fully independent of the MD machinery.
// London orbital (gauge origin O = 0): (r-C)^l e^{-a(r-C)^2} e^{i a_phase . r},
// a_phase = -1/2 B x C.
C grid_giao_kinetic(double a, const double *Ca, const int *la, double b,
                    const double *Cb, const int *lb, const double *Bf, int n,
                    double L) {
  auto Bxv = [&](const double *Cv, double *out) {
    out[0] = -0.5 * (Bf[1] * Cv[2] - Bf[2] * Cv[1]);
    out[1] = -0.5 * (Bf[2] * Cv[0] - Bf[0] * Cv[2]);
    out[2] = -0.5 * (Bf[0] * Cv[1] - Bf[1] * Cv[0]);
  };
  double aa[3], ab[3];
  Bxv(Ca, aa);
  Bxv(Cb, ab);
  const double h = 2 * L / (n - 1);
  auto orb = [&](double x, double y, double z, double e, const double *Cc,
                 const int *l, const double *ph) {
    const double dx = x - Cc[0], dy = y - Cc[1], dz = z - Cc[2];
    double g = std::exp(-e * (dx * dx + dy * dy + dz * dz));
    for (int i = 0; i < l[0]; ++i) g *= dx;
    for (int i = 0; i < l[1]; ++i) g *= dy;
    for (int i = 0; i < l[2]; ++i) g *= dz;
    const double phase = ph[0] * x + ph[1] * y + ph[2] * z;
    return C(g * std::cos(phase), g * std::sin(phase));
  };
  C acc(0);
  for (int i = 1; i < n - 1; ++i)
    for (int j = 1; j < n - 1; ++j)
      for (int k = 1; k < n - 1; ++k) {
        const double x = -L + i * h, y = -L + j * h, z = -L + k * h;
        const C ket = orb(x, y, z, b, Cb, lb, ab);
        const C lap =
            (orb(x + h, y, z, b, Cb, lb, ab) + orb(x - h, y, z, b, Cb, lb, ab) +
             orb(x, y + h, z, b, Cb, lb, ab) + orb(x, y - h, z, b, Cb, lb, ab) +
             orb(x, y, z + h, b, Cb, lb, ab) + orb(x, y, z - h, b, Cb, lb, ab) -
             6.0 * ket) /
            (h * h);
        acc += std::conj(orb(x, y, z, a, Ca, la, aa)) * (-0.5 * lap);
      }
  return acc * (h * h * h);
}

TEST(GIAO, KineticMatchesRealSpaceGrid) {
  // independent oracle: giao_kinetic vs a real-space grid integral, for a
  // p_x/s pair in a field along x (x,y components of the phase both active)
  const double Bf[3] = {0.6, 0.0, 0.0};
  auto bas = intti::make_basis<double>(
      {{0.9, {kA[0], kA[1], kA[2]}, 1}, {1.3, {kB[0], kB[1], kB[2]}, 0}});
  const double O[3] = {0.0, 0.0, 0.0};
  auto T = intti::giao_kinetic(bas, Bf, O);
  const int nao = bas.nao; // p(3) + s(1)
  const int lax[3] = {1, 0, 0}, ls[3] = {0, 0, 0};
  const C ref = grid_giao_kinetic(0.9, kA, lax, 1.3, kB, ls, Bf, 181, 8.0);
  const C got = T[0 * nao + 3]; // <p_x(a)| T | s(b)>
  // grid is coarse (finite-difference Laplacian); ~1e-3 agreement expected
  EXPECT_LT(std::abs(got - ref), 3e-3 * std::abs(ref))
      << "giao_kinetic " << got << " vs grid " << ref;
}

TEST(GIAO, KineticZeroFieldRealMatch) {
  auto bas = giao_basis();
  const double Bf[3] = {0.0, 0.0, 0.0}, O[3] = {0.1, -0.2, 0.3};
  auto Tc = intti::giao_kinetic(bas, Bf, O);
  auto Tr = intti::kinetic_matrix(bas);
  double mx = 0, dev = 0;
  for (std::size_t i = 0; i < Tr.size(); ++i) {
    mx = std::max(mx, std::abs(Tr[i]));
    dev = std::max(dev, std::abs(Tc[i].real() - Tr[i]));
    dev = std::max(dev, std::abs(Tc[i].imag()));
  }
  EXPECT_LT(dev, 1e-13 * mx);
}

TEST(GIAO, KineticFieldDerivativeVsFiniteDiff) {
  // -1/2 nabla^2 also differentiates the London phase, so dT/dB carries a
  // gradient term beyond the phase-weighted kinetic; check the full analytic
  // dT/dB against a central finite difference of the exact T(B).
  auto bas = giao_basis();
  const double O[3] = {0.1, -0.2, 0.3};
  auto dT = intti::giao_kinetic_dB(bas, O);
  const double h = 1e-4;
  double worst = 0, scale = 0;
  for (int k = 0; k < 3; ++k) {
    double Bp[3] = {0, 0, 0}, Bm[3] = {0, 0, 0};
    Bp[k] = h;
    Bm[k] = -h;
    auto Tp = intti::giao_kinetic(bas, Bp, O);
    auto Tm = intti::giao_kinetic(bas, Bm, O);
    for (std::size_t i = 0; i < Tp.size(); ++i) {
      const C fd = (Tp[i] - Tm[i]) / (2.0 * h);
      worst = std::max(worst, std::abs(fd - dT[k][i]));
      scale = std::max(scale, std::abs(dT[k][i]));
    }
  }
  EXPECT_GT(scale, 1e-3) << "derivative must be nonzero";
  EXPECT_LT(worst, 1e-7 * (scale + 1)) << "analytic dT/dB != finite difference";
}

// finite-field GIAO J and K reference by direct complex quartet summation:
//   J(B)_mn = sum_ls (mn|ls)_giao D_ls,  K(B)_mn = sum_ls (ml|ns)_giao D_ls.
void giao_jk_ref(const intti::ShellBasis<double> &bas, const double *D,
                 const double *Bf, const intti::TGrid<double> &grid,
                 std::vector<C> &J, std::vector<C> &K) {
  const int ns = static_cast<int>(bas.shells.size()), nao = bas.nao;
  J.assign(static_cast<std::size_t>(nao) * nao, C(0));
  K.assign(static_cast<std::size_t>(nao) * nao, C(0));
  auto Dm = [&](int i, int j) { return D[static_cast<std::size_t>(i) * nao + j]; };
  for (int a = 0; a < ns; ++a)
    for (int b = 0; b < ns; ++b)
      for (int cc = 0; cc < ns; ++cc)
        for (int dd = 0; dd < ns; ++dd) {
          const int la = bas.shells[a].l, lb = bas.shells[b].l, lc = bas.shells[cc].l,
                    ld = bas.shells[dd].l;
          const int na = intti::ncart(la), nb = intti::ncart(lb), ncc = intti::ncart(lc),
                    nd = intti::ncart(ld);
          std::vector<C> blk(static_cast<std::size_t>(na) * nb * ncc * nd);
          intti::eri_quartet(intti::make_giao_pair(bas.shells[a], bas.shells[b], Bf),
                             intti::make_giao_pair(bas.shells[cc], bas.shells[dd], Bf),
                             grid, blk.data());
          const int oa = bas.ao_off[a], ob = bas.ao_off[b], oc = bas.ao_off[cc],
                    odd = bas.ao_off[dd];
          for (int ka = 0; ka < na; ++ka)
            for (int kb = 0; kb < nb; ++kb)
              for (int kc = 0; kc < ncc; ++kc)
                for (int kd = 0; kd < nd; ++kd) {
                  const C v = blk[((static_cast<std::size_t>(ka) * nb + kb) * ncc + kc) * nd + kd];
                  J[(oa + ka) * static_cast<std::size_t>(nao) + ob + kb] +=
                      v * Dm(oc + kc, odd + kd);
                  K[(oa + ka) * static_cast<std::size_t>(nao) + oc + kc] +=
                      v * Dm(ob + kb, odd + kd);
                }
        }
}

TEST(GIAO, JKFieldDerivativeVsFiniteDiff) {
  // analytic dJ/dB, dK/dB against a central finite difference of the exact
  // finite-field GIAO J(B)/K(B) built by direct complex quartet summation.
  auto bas = giao_basis(); // s, p, d on three centres
  const int nao = bas.nao;
  auto grid = intti::make_tgrid(intti::coulomb());
  // symmetric positive test density
  std::vector<double> D(static_cast<std::size_t>(nao) * nao, 0.0);
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j) D[i * nao + j] = 0.3 / (1 + std::abs(i - j)) + (i == j ? 0.5 : 0);
  auto d = intti::giao_jk_dB(bas, D.data(), grid);
  const double h = 1e-4;
  double worstJ = 0, worstK = 0, scaleJ = 0, scaleK = 0;
  for (int k = 0; k < 3; ++k) {
    double Bp[3] = {0, 0, 0}, Bm[3] = {0, 0, 0};
    Bp[k] = h;
    Bm[k] = -h;
    std::vector<C> Jp, Kp, Jm, Km;
    giao_jk_ref(bas, D.data(), Bp, grid, Jp, Kp);
    giao_jk_ref(bas, D.data(), Bm, grid, Jm, Km);
    for (std::size_t i = 0; i < Jp.size(); ++i) {
      const C fdJ = (Jp[i] - Jm[i]) / (2.0 * h), fdK = (Kp[i] - Km[i]) / (2.0 * h);
      worstJ = std::max(worstJ, std::abs(fdJ - d.dJ[k][i]));
      worstK = std::max(worstK, std::abs(fdK - d.dK[k][i]));
      scaleJ = std::max(scaleJ, std::abs(d.dJ[k][i]));
      scaleK = std::max(scaleK, std::abs(d.dK[k][i]));
    }
  }
  EXPECT_GT(scaleJ, 1e-3);
  EXPECT_GT(scaleK, 1e-3);
  EXPECT_LT(worstJ, 1e-7 * (scaleJ + 1)) << "analytic dJ/dB != finite difference";
  EXPECT_LT(worstK, 1e-7 * (scaleK + 1)) << "analytic dK/dB != finite difference";
}

TEST(GIAO, JKFieldDerivativeImaginaryAndReal) {
  // dJ/dB and dK/dB are purely imaginary at B=0 (real ERIs, i from the phase);
  // dJ is Hermitian (imag part antisymmetric), so its diagonal is zero
  auto bas = giao_basis();
  const int nao = bas.nao;
  auto grid = intti::make_tgrid(intti::coulomb());
  std::vector<double> D(static_cast<std::size_t>(nao) * nao, 0.0);
  for (int i = 0; i < nao; ++i) D[i * nao + i] = 1.0;
  auto d = intti::giao_jk_dB(bas, D.data(), grid);
  for (int k = 0; k < 3; ++k)
    for (std::size_t i = 0; i < d.dJ[k].size(); ++i) {
      EXPECT_EQ(d.dJ[k][i].real(), 0.0);
      EXPECT_EQ(d.dK[k][i].real(), 0.0);
    }
}

TEST(GIAO, NuclearZeroFieldRealMatch) {
  auto bas = giao_basis();
  auto grid = intti::make_tgrid(intti::coulomb());
  std::vector<intti::PointCharge<double>> chg{{-1.0, {0.0, 0.1, -0.3}},
                                              {-9.0, {0.5, -0.2, 0.4}}};
  const double Bf[3] = {0.0, 0.0, 0.0};
  auto Vc = intti::giao_nuclear(bas, chg, grid, Bf);
  auto Vr = intti::nuclear_matrix(bas, chg, grid);
  double mx = 0, dev = 0;
  for (std::size_t i = 0; i < Vr.size(); ++i) {
    mx = std::max(mx, std::abs(Vr[i]));
    dev = std::max(dev, std::abs(Vc[i].real() - Vr[i]));
    dev = std::max(dev, std::abs(Vc[i].imag()));
  }
  EXPECT_LT(dev, 1e-13 * mx);
}

TEST(GIAO, NuclearFieldDerivativeVsFiniteDiff) {
  auto bas = giao_basis();
  auto grid = intti::make_tgrid(intti::coulomb());
  std::vector<intti::PointCharge<double>> chg{{-1.0, {0.0, 0.1, -0.3}},
                                              {-9.0, {0.5, -0.2, 0.4}}};
  auto dV = intti::giao_nuclear_dB(bas, chg, grid);
  const double h = 1e-4;
  double worst = 0, scale = 0;
  for (int k = 0; k < 3; ++k) {
    double Bp[3] = {0, 0, 0}, Bm[3] = {0, 0, 0};
    Bp[k] = h;
    Bm[k] = -h;
    auto Vp = intti::giao_nuclear(bas, chg, grid, Bp);
    auto Vm = intti::giao_nuclear(bas, chg, grid, Bm);
    for (std::size_t i = 0; i < Vp.size(); ++i) {
      const C fd = (Vp[i] - Vm[i]) / (2.0 * h);
      worst = std::max(worst, std::abs(fd - dV[k][i]));
      scale = std::max(scale, std::abs(dV[k][i]));
    }
  }
  EXPECT_GT(scale, 1e-3) << "derivative must be nonzero";
  EXPECT_LT(worst, 1e-7 * (scale + 1)) << "analytic dV/dB != finite difference";
}

// ---- finite-field two-electron J/K (giao_jk) --------------------------------
// A small s/p basis and a Hermitian complex density.
intti::ShellBasis<double> jk_basis() {
  return intti::make_basis<double>({{0.9, {kA[0], kA[1], kA[2]}, 0},
                                    {1.3, {kB[0], kB[1], kB[2]}, 1},
                                    {0.6, {kC[0], kC[1], kC[2]}, 0}});
}

std::vector<C> hermitian_density(int nao) {
  std::vector<C> D(static_cast<std::size_t>(nao) * nao);
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j)
      D[i * nao + j] = C(0.2 + 0.05 * (i + j), 0.03 * (i - j)); // Hermitian
  return D;
}

TEST(GIAO, FiniteFieldJKZeroFieldMatchesRealJK) {
  auto bas = jk_basis();
  const int nao = bas.nao;
  auto grid = intti::make_tgrid(intti::coulomb());
  // real symmetric density (imaginary part zero) so the B=0 limit is the real J/K
  std::vector<double> Dr(static_cast<std::size_t>(nao) * nao);
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j) Dr[i * nao + j] = 0.2 + 0.05 * (i + j);
  std::vector<C> Dc(Dr.size());
  for (std::size_t i = 0; i < Dr.size(); ++i) Dc[i] = C(Dr[i], 0.0);

  const double B0[3] = {0.0, 0.0, 0.0};
  auto jk = intti::giao_jk(bas, Dc.data(), B0, grid);

  std::vector<double> Jr(static_cast<std::size_t>(nao) * nao),
      Kr(static_cast<std::size_t>(nao) * nao);
  intti::coulomb_build(bas, Dr.data(), grid, Jr.data());
  intti::exchange_build(bas, Dr.data(), grid, Kr.data(), 0.0);
  double mx = 0;
  for (std::size_t i = 0; i < Jr.size(); ++i) mx = std::max(mx, std::abs(Jr[i]));
  ASSERT_GT(mx, 1e-6);
  for (std::size_t i = 0; i < Jr.size(); ++i) {
    EXPECT_LT(std::abs(jk.J[i].real() - Jr[i]), 1e-12 * mx) << "J real at B=0, i=" << i;
    EXPECT_LT(std::abs(jk.J[i].imag()), 1e-13 * mx) << "J imag at B=0";
    EXPECT_LT(std::abs(jk.K[i].real() - Kr[i]), 1e-12 * mx) << "K real at B=0, i=" << i;
    EXPECT_LT(std::abs(jk.K[i].imag()), 1e-13 * mx) << "K imag at B=0";
  }
}

TEST(GIAO, FiniteFieldJKHermitian) {
  auto bas = jk_basis();
  const int nao = bas.nao;
  auto grid = intti::make_tgrid(intti::coulomb());
  auto D = hermitian_density(nao); // complex Hermitian
  const double Bp[3] = {0.2, -0.3, 0.7};
  auto jk = intti::giao_jk(bas, D.data(), Bp, grid);
  double mx = 0, herm = 0, imag = 0;
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j) {
      const std::size_t ij = static_cast<std::size_t>(i) * nao + j;
      const std::size_t ji = static_cast<std::size_t>(j) * nao + i;
      mx = std::max(mx, std::abs(jk.J[ij]));
      herm = std::max(herm, std::abs(jk.J[ij] - std::conj(jk.J[ji])));
      herm = std::max(herm, std::abs(jk.K[ij] - std::conj(jk.K[ji])));
      imag = std::max(imag, std::abs(jk.J[ij].imag()));
    }
  ASSERT_GT(mx, 1e-6);
  EXPECT_GT(imag, 1e-6 * mx) << "the field must make J genuinely complex";
  EXPECT_LT(herm, 1e-12 * mx) << "J,K must be Hermitian for a Hermitian density";
}

// B -> -B conjugates the London phases, so with a REAL density (which is its
// own conjugate) the Fock pieces must conjugate. (With a complex density the
// physical density conjugates too, so this is the clean statement.)
TEST(GIAO, FiniteFieldJKFieldReversalConjugates) {
  auto bas = jk_basis();
  const int nao = bas.nao;
  auto grid = intti::make_tgrid(intti::coulomb());
  std::vector<C> D(static_cast<std::size_t>(nao) * nao);
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j) D[i * nao + j] = C(0.2 + 0.05 * (i + j), 0.0);
  const double Bp[3] = {0.2, -0.3, 0.7}, Bm[3] = {-0.2, 0.3, -0.7};
  auto jp = intti::giao_jk(bas, D.data(), Bp, grid);
  auto jm = intti::giao_jk(bas, D.data(), Bm, grid);
  double mx = 0, rev = 0, imag = 0;
  for (std::size_t i = 0; i < D.size(); ++i) {
    mx = std::max(mx, std::abs(jp.J[i]));
    rev = std::max(rev, std::abs(jm.J[i] - std::conj(jp.J[i])));
    rev = std::max(rev, std::abs(jm.K[i] - std::conj(jp.K[i])));
    imag = std::max(imag, std::abs(jp.J[i].imag()));
  }
  ASSERT_GT(mx, 1e-6);
  EXPECT_GT(imag, 1e-6 * mx) << "the field must do something";
  EXPECT_LT(rev, 1e-12 * mx) << "B -> -B must conjugate J,K";
}

// S(B) = <omega_mu|omega_nu> must be Hermitian at finite field (the London
// phases make it Hermitian, not symmetric) and genuinely complex.
TEST(GIAO, FiniteFieldOverlapHermitian) {
  auto bas = jk_basis();
  const int nao = bas.nao;
  const double Bf[3] = {0.2, -0.3, 0.7};
  auto S = intti::giao_overlap(bas, Bf);
  double mx = 0, herm = 0, imag = 0;
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j) {
      const std::size_t ij = static_cast<std::size_t>(i) * nao + j;
      const std::size_t ji = static_cast<std::size_t>(j) * nao + i;
      mx = std::max(mx, std::abs(S[ij]));
      herm = std::max(herm, std::abs(S[ij] - std::conj(S[ji])));
      imag = std::max(imag, std::abs(S[ij].imag()));
    }
  ASSERT_GT(mx, 1e-6);
  EXPECT_GT(imag, 1e-6 * mx) << "the field must make S genuinely complex";
  EXPECT_LT(herm, 1e-13 * mx) << "S(B) must be Hermitian";
}

// ---- end-to-end: the finite-field Fock matrix ------------------------------
// The capstone for the complex path: S(B), T(B), V(B), J(B), K(B) must compose
// into the complex Fock matrix a finite-field SCF consumes. Checks that the
// COMPOSITION (not just each piece) is right: at B = 0 it reproduces the real
// Fock exactly, at finite B it is Hermitian and genuinely complex, and B -> -B
// conjugates it.
namespace {
struct FockPieces {
  std::vector<C> F, S;
};
FockPieces finite_field_fock(const intti::ShellBasis<double> &bas,
                             const std::vector<intti::PointCharge<double>> &chg,
                             const std::vector<C> &D, const double B[3],
                             const intti::TGrid<double> &grid) {
  const double O[3] = {0.0, 0.0, 0.0}; // gauge origin
  auto T = intti::giao_kinetic(bas, B, O);
  auto V = intti::giao_nuclear(bas, chg, grid, B);
  auto jk = intti::giao_jk(bas, D.data(), B, grid);
  FockPieces out;
  out.S = intti::giao_overlap(bas, B);
  out.F.assign(T.size(), C(0));
  for (std::size_t i = 0; i < T.size(); ++i)
    out.F[i] = T[i] + V[i] + jk.J[i] - C(0.5) * jk.K[i];
  return out;
}
} // namespace

TEST(GIAO, FiniteFieldFockZeroFieldMatchesRealFock) {
  auto bas = jk_basis();
  const int nao = bas.nao;
  auto grid = intti::make_tgrid(intti::coulomb());
  std::vector<intti::PointCharge<double>> chg = {{-1.0, {0.1, 0.0, -0.2}},
                                                {-2.0, {-0.3, 0.4, 0.5}}};
  // real symmetric density, and its complex image
  std::vector<double> Dr(static_cast<std::size_t>(nao) * nao);
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j) Dr[i * nao + j] = 0.2 + 0.04 * (i + j);
  std::vector<C> Dc(Dr.size());
  for (std::size_t i = 0; i < Dr.size(); ++i) Dc[i] = C(Dr[i], 0.0);

  const double B0[3] = {0.0, 0.0, 0.0};
  auto got = finite_field_fock(bas, chg, Dc, B0, grid);

  // the real Fock from the ordinary builders
  auto T = intti::kinetic_matrix(bas);
  auto V = intti::nuclear_matrix(bas, chg, grid);
  std::vector<double> J(Dr.size()), K(Dr.size());
  intti::coulomb_build(bas, Dr.data(), grid, J.data());
  intti::exchange_build(bas, Dr.data(), grid, K.data(), 0.0);
  auto Sreal = intti::overlap_matrix(bas);

  double mx = 0, errF = 0, errS = 0;
  for (std::size_t i = 0; i < Dr.size(); ++i) {
    const double f = T[i] + V[i] + J[i] - 0.5 * K[i];
    mx = std::max(mx, std::abs(f));
    errF = std::max(errF, std::abs(got.F[i] - C(f, 0.0)));
    errS = std::max(errS, std::abs(got.S[i] - C(Sreal[i], 0.0)));
  }
  ASSERT_GT(mx, 1e-6);
  EXPECT_LT(errF, 1e-11 * mx) << "finite-field Fock at B=0 != real Fock";
  EXPECT_LT(errS, 1e-12 * (mx + 1)) << "finite-field S at B=0 != real overlap";
}

TEST(GIAO, FiniteFieldFockHermitianAndConjugatesUnderFieldReversal) {
  auto bas = jk_basis();
  const int nao = bas.nao;
  auto grid = intti::make_tgrid(intti::coulomb());
  std::vector<intti::PointCharge<double>> chg = {{-1.0, {0.1, 0.0, -0.2}}};
  // real density so field reversal is the clean statement
  std::vector<C> D(static_cast<std::size_t>(nao) * nao);
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j) D[i * nao + j] = C(0.2 + 0.04 * (i + j), 0.0);
  const double Bp[3] = {0.25, -0.3, 0.6}, Bm[3] = {-0.25, 0.3, -0.6};
  auto fp = finite_field_fock(bas, chg, D, Bp, grid);
  auto fm = finite_field_fock(bas, chg, D, Bm, grid);
  double mx = 0, herm = 0, rev = 0, imag = 0;
  for (int i = 0; i < nao; ++i)
    for (int j = 0; j < nao; ++j) {
      const std::size_t ij = static_cast<std::size_t>(i) * nao + j;
      const std::size_t ji = static_cast<std::size_t>(j) * nao + i;
      mx = std::max(mx, std::abs(fp.F[ij]));
      herm = std::max(herm, std::abs(fp.F[ij] - std::conj(fp.F[ji])));
      herm = std::max(herm, std::abs(fp.S[ij] - std::conj(fp.S[ji])));
      rev = std::max(rev, std::abs(fm.F[ij] - std::conj(fp.F[ij])));
      imag = std::max(imag, std::abs(fp.F[ij].imag()));
    }
  ASSERT_GT(mx, 1e-6);
  EXPECT_GT(imag, 1e-6 * mx) << "the field must make F genuinely complex";
  EXPECT_LT(herm, 1e-12 * mx) << "F(B) and S(B) must be Hermitian";
  EXPECT_LT(rev, 1e-12 * mx) << "B -> -B must conjugate F";
}

// ---- structure of the finite-B pair space (prerequisite for a CD path) ------
// A Cholesky decomposition needs a HERMITIAN POSITIVE-DEFINITE matrix. At
// finite B the matrix the real CD code would form, M_PQ = (mn|ls), is complex
// SYMMETRIC (M_PQ = M_QP by electron exchange) but NOT Hermitian, so it admits
// no Cholesky. The Hermitian Gram matrix is instead
//   H_PQ = <rho_P|rho_Q> = (nm|ls),   rho_P = omega_mu^* omega_nu,
// i.e. the ERI with the BRA PAIR SWAPPED (because rho_P^* = rho_(nu mu)).
// This test pins both facts, so a future finite-B CD knows which object to
// decompose. s-only basis, so AO index == shell index.
TEST(GIAO, FiniteFieldPairGramIsHermitianPositive) {
  const double Bf[3] = {0.2, -0.3, 0.7};
  auto grid = intti::make_tgrid(intti::coulomb());
  const double alph[3] = {0.9, 1.3, 0.6};
  const double *cen[3] = {kA, kB, kC};
  auto sh = [&](int i) { return shell(alph[i], cen[i], 0); };
  const int n = 3, np = n * n; // pairs P = (mu,nu)
  auto eri = [&](int m, int nu, int l, int s) {
    C v;
    intti::eri_quartet(intti::make_giao_pair(sh(m), sh(nu), Bf),
                       intti::make_giao_pair(sh(l), sh(s), Bf), grid, &v);
    return v;
  };
  std::vector<C> H(np * np), M(np * np);
  for (int P = 0; P < np; ++P) {
    const int m = P / n, nu = P % n;
    for (int Q = 0; Q < np; ++Q) {
      const int l = Q / n, s = Q % n;
      H[P * np + Q] = eri(nu, m, l, s); // bra pair swapped -> Gram
      M[P * np + Q] = eri(m, nu, l, s); // what the real CD would form
    }
  }
  double mx = 0, herm = 0, sym = 0, nonherm = 0, diagimag = 0, mindiag = 1e300;
  for (int P = 0; P < np; ++P) {
    mx = std::max(mx, std::abs(H[P * np + P]));
    diagimag = std::max(diagimag, std::abs(H[P * np + P].imag()));
    mindiag = std::min(mindiag, H[P * np + P].real());
    for (int Q = 0; Q < np; ++Q) {
      herm = std::max(herm, std::abs(H[P * np + Q] - std::conj(H[Q * np + P])));
      sym = std::max(sym, std::abs(M[P * np + Q] - M[Q * np + P]));
      nonherm = std::max(nonherm, std::abs(M[P * np + Q] - std::conj(M[Q * np + P])));
    }
  }
  ASSERT_GT(mx, 1e-6);
  EXPECT_LT(herm, 1e-13 * mx) << "the Gram matrix (nm|ls) must be Hermitian";
  EXPECT_LT(diagimag, 1e-14 * mx) << "its diagonal must be real";
  EXPECT_GT(mindiag, 1e-8) << "its diagonal must be positive (a CD can pivot on it)";
  EXPECT_LT(sym, 1e-13 * mx) << "(mn|ls) is complex symmetric";
  EXPECT_GT(nonherm, 1e-6 * mx) << "(mn|ls) is NOT Hermitian -- no Cholesky of it";
}

} // namespace
