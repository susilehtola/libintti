// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::ScopeGuard guard(argc, argv);
  return RUN_ALL_TESTS();
}
