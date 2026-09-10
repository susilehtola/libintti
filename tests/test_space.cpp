// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

// Inputs and outputs on either the host or the device, in all four
// combinations. A code that keeps its density and Fock matrices on the GPU
// should not have to copy them down to call intti and back up afterwards --
// twice per SCF iteration, for data that never needed to leave.
//
// All four must agree exactly: they run the same kernel on the same numbers,
// and the only thing that differs is where the arguments happened to live.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "intti/fock.hpp"
#include "intti/jk.hpp"
#include "intti/kernel.hpp"
#include "intti/space.hpp"
#include "intti/tgrid.hpp"

namespace {

intti::ShellBasis<double> small_basis() {
  std::vector<intti::PrimitiveShell<double>> sh{
      {1.8, {0.0, 0.0, 0.0}, 0}, {0.5, {0.0, 0.0, 0.0}, 1},
      {1.1, {0.0, 0.0, 1.4}, 0}, {0.4, {0.0, 0.9, 0.7}, 1}};
  return intti::make_basis(sh);
}

using DevView = Kokkos::View<double *>;
using HostView = Kokkos::View<double *, Kokkos::HostSpace>;

} // namespace

TEST(Space, JKAgreesAcrossAllFourCombinations) {
  auto basis = small_basis();
  const int n = basis.nao;
  const std::size_t n2 = static_cast<std::size_t>(n) * n;
  auto grid = intti::make_tgrid(intti::coulomb<double>());

  // a GENERAL density, so the general (device-digest) path is the one exercised
  std::vector<double> Dh(n2);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) Dh[i * n + j] = 0.1 + 0.3 * std::sin(0.7 * i + 1.9 * j * j);

  // reference: the existing host-pointer API
  std::vector<intti::JKRequest<double>> reqs{
      {Dh.data(), intti::DensitySymmetry::General, intti::FockTerms::CoulombExchange}};
  const auto ref = intti::jk_build(basis, reqs, grid);
  double sj = 0, sk = 0;
  for (std::size_t i = 0; i < n2; ++i) {
    sj = std::max(sj, std::abs(ref.J[0][i]));
    sk = std::max(sk, std::abs(ref.K[0][i]));
  }
  ASSERT_GT(sj, 1e-3);
  ASSERT_GT(sk, 1e-3);

  HostView Dhost("Dhost", n2);
  for (std::size_t i = 0; i < n2; ++i) Dhost(i) = Dh[i];
  DevView Ddev("Ddev", n2);
  Kokkos::deep_copy(Ddev, Dhost);

  auto check = [&](const char *what, const std::vector<double> &J,
                   const std::vector<double> &K) {
    double dj = 0, dk = 0;
    for (std::size_t i = 0; i < n2; ++i) {
      dj = std::max(dj, std::abs(J[i] - ref.J[0][i]));
      dk = std::max(dk, std::abs(K[i] - ref.K[0][i]));
    }
    EXPECT_LT(dj, 1e-13 * sj) << what << ": J differs by memory space";
    EXPECT_LT(dk, 1e-13 * sk) << what << ": K differs by memory space";
  };

  const std::vector<intti::DensitySymmetry> sym{intti::DensitySymmetry::General};
  const std::vector<intti::FockTerms> terms{intti::FockTerms::CoulombExchange};

  // (1) host in, host out
  {
    std::vector<HostView> D{Dhost}, J{HostView("J", n2)}, K{HostView("K", n2)};
    intti::jk_build_into(basis, D, sym, terms, grid, J, K);
    std::vector<double> j(n2), k(n2);
    for (std::size_t i = 0; i < n2; ++i) { j[i] = J[0](i); k[i] = K[0](i); }
    check("host in / host out", j, k);
  }
  // (2) device in, device out -- the combination that should cost nothing
  {
    std::vector<DevView> D{Ddev}, J{DevView("J", n2)}, K{DevView("K", n2)};
    intti::jk_build_into(basis, D, sym, terms, grid, J, K);
    auto hj = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, J[0]);
    auto hk = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, K[0]);
    std::vector<double> j(hj.data(), hj.data() + n2), k(hk.data(), hk.data() + n2);
    check("device in / device out", j, k);
  }
  // (3) device in, host out
  {
    std::vector<DevView> D{Ddev};
    std::vector<HostView> J{HostView("J", n2)}, K{HostView("K", n2)};
    intti::jk_build_into(basis, D, sym, terms, grid, J, K);
    std::vector<double> j(n2), k(n2);
    for (std::size_t i = 0; i < n2; ++i) { j[i] = J[0](i); k[i] = K[0](i); }
    check("device in / host out", j, k);
  }
  // (4) host in, device out
  {
    std::vector<HostView> D{Dhost};
    std::vector<DevView> J{DevView("J", n2)}, K{DevView("K", n2)};
    intti::jk_build_into(basis, D, sym, terms, grid, J, K);
    auto hj = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, J[0]);
    auto hk = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, K[0]);
    std::vector<double> j(hj.data(), hj.data() + n2), k(hk.data(), hk.data() + n2);
    check("host in / device out", j, k);
  }
}

// device_in must not copy when the argument is already device-resident: that is
// the whole contract, and a copy here would silently reintroduce the traffic
// this exists to remove.
TEST(Space, DeviceInAliasesDeviceData) {
  DevView d("d", 8);
  auto same = intti::device_in(d);
  EXPECT_EQ(same.data(), d.data()) << "device_in copied data that was already on the device";

  HostView h("h", 8);
  auto moved = intti::device_in(h);
  if (!std::is_same_v<Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>)
    EXPECT_NE(static_cast<const void *>(moved.data()), static_cast<const void *>(h.data()))
        << "host data must be staged to the device";
}

// exchange_build, the routine an SCF calls every iteration, across the same
// four combinations. Its Schwarz density bound used to be a HOST loop over D,
// which forced the density to be host-resident even though every other use of
// it was on the device; that bound is now computed where D lives.
TEST(Space, ExchangeAgreesAcrossAllFourCombinations) {
  auto basis = small_basis();
  const int n = basis.nao;
  const std::size_t n2 = static_cast<std::size_t>(n) * n;
  auto grid = intti::make_tgrid(intti::coulomb<double>());

  std::vector<double> Dh(n2);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) Dh[i * n + j] = 0.1 + 0.3 * std::sin(0.7 * i + 1.3 * j);
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) Dh[i * n + j] = Dh[j * n + i];

  std::vector<double> ref(n2, 0.0);
  intti::exchange_build(basis, Dh.data(), grid, ref.data());
  double sk = 0;
  for (double v : ref) sk = std::max(sk, std::abs(v));
  ASSERT_GT(sk, 1e-3);

  HostView Dhost("Dhost", n2);
  for (std::size_t i = 0; i < n2; ++i) Dhost(i) = Dh[i];
  DevView Ddev("Ddev", n2);
  Kokkos::deep_copy(Ddev, Dhost);

  auto check = [&](const char *what, const std::vector<double> &K) {
    double dk = 0;
    for (std::size_t i = 0; i < n2; ++i) dk = std::max(dk, std::abs(K[i] - ref[i]));
    EXPECT_LT(dk, 1e-13 * sk) << what << ": K differs by memory space";
  };
  auto grab = [&](const auto &V) {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, V);
    return std::vector<double>(h.data(), h.data() + n2);
  };

  { HostView K("K", n2);
    intti::exchange_build_into(basis, Dhost, grid, K);
    check("host in / host out", grab(K)); }
  { DevView K("K", n2);
    intti::exchange_build_into(basis, Ddev, grid, K);
    check("device in / device out", grab(K)); }
  { HostView K("K", n2);
    intti::exchange_build_into(basis, Ddev, grid, K);
    check("device in / host out", grab(K)); }
  { DevView K("K", n2);
    intti::exchange_build_into(basis, Dhost, grid, K);
    check("host in / device out", grab(K)); }
}
