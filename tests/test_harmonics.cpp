// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <cmath>
#include <complex>
#include <vector>

#include <gtest/gtest.h>

#include "intti/c2s.hpp"
#include "intti/fock.hpp"
#include "intti/harmonics.hpp"
#include "intti/oneel.hpp"

namespace {

using intti::ncart;

// A single centre carrying s, p, d and f, so every (+m, -m) pair up to l = 3 is
// exercised. One centre is the point: L about that centre is what has m as a
// good quantum number, and a second centre would destroy the very structure the
// test is checking.
intti::ShellBasis<double> one_centre(std::vector<int> &ls) {
  std::vector<intti::PrimitiveShell<double>> sh;
  ls.clear();
  for (int l = 0; l <= 3; ++l) {
    sh.push_back(intti::PrimitiveShell<double>{0.8, {0.0, 0.0, 0.0}, l});
    ls.push_back(l);
  }
  return intti::make_basis(sh);
}

// C M C^T over shell blocks: Cartesian AO matrix -> real spherical AO matrix.
std::vector<double> cart_to_sph(const intti::ShellBasis<double> &b,
                                const std::vector<double> &M, int &nsph,
                                std::vector<int> &soff) {
  const int ns = static_cast<int>(b.shells.size()), nao = b.nao;
  soff.assign(ns, 0);
  nsph = 0;
  for (int s = 0; s < ns; ++s) {
    soff[s] = nsph;
    nsph += 2 * b.shells[s].l + 1;
  }
  std::vector<double> tmp(static_cast<std::size_t>(nsph) * nao, 0.0);
  for (int s = 0; s < ns; ++s) {
    const int l = b.shells[s].l, nc = ncart(l), nm = 2 * l + 1;
    const auto C = intti::c2s_matrix<double>(l);
    for (int m = 0; m < nm; ++m)
      for (int j = 0; j < nao; ++j) {
        double acc = 0;
        for (int k = 0; k < nc; ++k)
          acc += C[static_cast<std::size_t>(m) * nc + k] *
                 M[static_cast<std::size_t>(b.ao_off[s] + k) * nao + j];
        tmp[static_cast<std::size_t>(soff[s] + m) * nao + j] = acc;
      }
  }
  std::vector<double> out(static_cast<std::size_t>(nsph) * nsph, 0.0);
  for (int s = 0; s < ns; ++s) {
    const int l = b.shells[s].l, nc = ncart(l), nm = 2 * l + 1;
    const auto C = intti::c2s_matrix<double>(l);
    for (int i = 0; i < nsph; ++i)
      for (int m = 0; m < nm; ++m) {
        double acc = 0;
        for (int k = 0; k < nc; ++k)
          acc += C[static_cast<std::size_t>(m) * nc + k] *
                 tmp[static_cast<std::size_t>(i) * nao + b.ao_off[s] + k];
        out[static_cast<std::size_t>(i) * nsph + soff[s] + m] = acc;
      }
  }
  return out;
}

} // namespace

// The r2c transform is unitary, per l.
TEST(Harmonics, TransformIsUnitary) {
  for (int l = 0; l <= 5; ++l) {
    const auto U = intti::r2c_matrix<double>(l);
    const int nm = 2 * l + 1;
    double worst = 0;
    for (int a = 0; a < nm; ++a)
      for (int b = 0; b < nm; ++b) {
        std::complex<double> acc(0);
        for (int k = 0; k < nm; ++k)
          acc += U[static_cast<std::size_t>(a) * nm + k] *
                 std::conj(U[static_cast<std::size_t>(b) * nm + k]);
        worst = std::max(worst, std::abs(acc - std::complex<double>(a == b ? 1.0 : 0.0)));
      }
    EXPECT_LT(worst, 1e-14) << "r2c matrix not unitary at l = " << l;
  }
}

// THE phase check. Unitarity above cannot see a wrong relative sign inside a
// (+m, -m) pair, and neither can any norm; the physics can. In the complex
// basis L_z is diagonal with eigenvalue m, and intti computes <mu|r x nabla|nu>
// by a completely unrelated route (oneel.hpp). r x nabla is i L, so the z
// component must come out diagonal with entries exactly i*m -- which fixes the
// Condon-Shortley phases instead of assuming them.
TEST(Harmonics, LzIsDiagonalInMWithCondonShortley) {
  std::vector<int> ls;
  auto basis = one_centre(ls);
  const double origin[3] = {0.0, 0.0, 0.0};
  auto L = intti::angular_momentum(basis, origin);
  auto S = intti::overlap_matrix(basis);

  int nsph = 0;
  std::vector<int> soff;
  auto Lz_real = cart_to_sph(basis, L[2], nsph, soff);
  int n2 = 0;
  std::vector<int> soff2;
  auto S_real = cart_to_sph(basis, S, n2, soff2);
  ASSERT_EQ(nsph, n2);

  auto Lz = intti::real_to_complex(ls, soff, nsph, Lz_real.data());
  auto Sc = intti::real_to_complex(ls, soff, nsph, S_real.data());

  // The AOs are not normalised -- the engine works with unnormalised primitives
  // -- so the statement is not L_z = i*m but the exact one for a non-orthonormal
  // basis, L_z = i*m*S. S comes from yet another builder, which makes this a
  // check against two independent quantities rather than one.
  double offdiag = 0, worst = 0, mmax = 0, sdiag = 0;
  for (std::size_t s = 0; s < ls.size(); ++s) {
    const int l = ls[s];
    for (int a = 0; a < 2 * l + 1; ++a) {
      const int m = a - l;
      const std::size_t idx = static_cast<std::size_t>(soff[s] + a) * nsph + soff[s] + a;
      const auto sv = Sc[idx];
      worst = std::max(worst, std::abs(Lz[idx] - std::complex<double>(0.0, m) * sv));
      mmax = std::max(mmax, std::abs(static_cast<double>(m)));
      sdiag = std::max(sdiag, std::abs(sv));
      EXPECT_GT(sv.real(), 1e-6) << "overlap diagonal must be real and positive";
      EXPECT_LT(std::abs(sv.imag()), 1e-13) << "overlap diagonal must be real";
    }
  }
  for (int i = 0; i < nsph; ++i)
    for (int j = 0; j < nsph; ++j)
      if (i != j)
        offdiag = std::max(offdiag, std::abs(Lz[static_cast<std::size_t>(i) * nsph + j]));

  EXPECT_GT(mmax, 2.5) << "basis must reach |m| = 3 or the check is weak";
  EXPECT_GT(sdiag, 1e-3) << "overlap trivially zero";
  EXPECT_LT(offdiag, 1e-12) << "L_z is not diagonal in the complex basis";
  EXPECT_LT(worst, 1e-12 * (sdiag + 1))
      << "L_z != i*m*S: the Condon-Shortley phases are wrong";

  // ...and L_z is NOT already diagonal in the real basis, or the transform
  // would be doing nothing and the whole check would pass vacuously.
  double realoff = 0;
  for (int i = 0; i < nsph; ++i)
    for (int j = 0; j < nsph; ++j)
      if (i != j)
        realoff = std::max(realoff, std::abs(Lz_real[static_cast<std::size_t>(i) * nsph + j]));
  EXPECT_GT(realoff, 0.5) << "L_z already diagonal in the real basis: test is vacuous";
}

// The libcint bridge is a permutation at l = 1 ONLY. Asserting both halves
// matters: that it changes something at l = 1 (or the flag would be dead), and
// that it changes nothing anywhere else (or it would silently corrupt d and
// higher, where libcint and the standard ordering agree).
TEST(Harmonics, LibcintOrderDiffersOnlyAtP) {
  for (int l = 0; l <= 5; ++l) {
    const auto A = intti::r2c_matrix<double>(l, intti::RealOrder::Standard);
    const auto B = intti::r2c_matrix<double>(l, intti::RealOrder::Libcint);
    double diff = 0;
    for (std::size_t k = 0; k < A.size(); ++k) diff = std::max(diff, std::abs(A[k] - B[k]));
    if (l == 1)
      EXPECT_GT(diff, 0.5) << "libcint p reordering must actually do something";
    else
      EXPECT_LT(diff, 1e-15) << "libcint ordering must be standard at l = " << l;
    // either way it stays unitary
    const int nm = 2 * l + 1;
    double worst = 0;
    for (int a = 0; a < nm; ++a)
      for (int b = 0; b < nm; ++b) {
        std::complex<double> acc(0);
        for (int k = 0; k < nm; ++k)
          acc += B[static_cast<std::size_t>(a) * nm + k] *
                 std::conj(B[static_cast<std::size_t>(b) * nm + k]);
        worst = std::max(worst, std::abs(acc - std::complex<double>(a == b ? 1.0 : 0.0)));
      }
    EXPECT_LT(worst, 1e-14) << "libcint-ordered r2c not unitary at l = " << l;
  }
}

// The convention layer: converting intti -> foreign -> intti must be the
// identity, and must not be the identity in one step, or the table is inert.
// The table's CORRECTNESS is established physically by the L_z test above --
// this checks that applying it to a whole matrix is consistent, which is a
// different failure mode (index bookkeeping rather than convention).
TEST(Harmonics, ConventionRoundTrip) {
  std::vector<int> ls;
  auto basis = one_centre(ls);
  int nsph = 0;
  std::vector<int> soff;
  const double origin[3] = {0.0, 0.0, 0.0};
  auto L = intti::angular_momentum(basis, origin);
  auto M = cart_to_sph(basis, L[2], nsph, soff);

  std::vector<int> nfunc;
  for (int l : ls) nfunc.push_back(2 * l + 1);
  auto fwd = [&](int s) { return intti::sph_reindex(intti::AoConvention::Libcint, ls[s]); };
  auto lc = intti::convert_matrix(soff, nfunc, nsph, M.data(), fwd);

  // the inverse map: intti_i = scale[i] * foreign[perm^{-1}[i]]
  auto inv = [&](int s) {
    const auto r = intti::sph_reindex(intti::AoConvention::Libcint, ls[s]);
    intti::ShellReindex q = intti::identity_reindex(static_cast<int>(r.perm.size()));
    for (std::size_t i = 0; i < r.perm.size(); ++i) {
      q.perm[r.perm[i]] = static_cast<int>(i);
      q.scale[r.perm[i]] = r.scale[i];
    }
    return q;
  };
  auto back = intti::convert_matrix(soff, nfunc, nsph, lc.data(), inv);

  double rt = 0, fwdiff = 0, scale = 0;
  for (std::size_t i = 0; i < M.size(); ++i) {
    rt = std::max(rt, std::abs(back[i] - M[i]));
    fwdiff = std::max(fwdiff, std::abs(lc[i] - M[i]));
    scale = std::max(scale, std::abs(M[i]));
  }
  EXPECT_GT(scale, 0.5) << "matrix trivially zero";
  EXPECT_LT(rt, 1e-14 * scale) << "convention round trip is not the identity";
  EXPECT_GT(fwdiff, 0.5) << "conversion changed nothing: the table is inert";
}
