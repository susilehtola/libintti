// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Susi Lehtola

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::ScopeGuard guard(argc, argv);
  return RUN_ALL_TESTS();
}
