// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <complex>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "intti/giao.hpp"
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

} // namespace
