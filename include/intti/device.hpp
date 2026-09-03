// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Small host<->device View helpers. The builders repeat the same
// create_mirror_view + fill + deep_copy dance for every array they push to the
// device (and the reverse on the way out); these collapse it to one call. Part
// of the SHARK-inspired maintainability pass (M-SHARK #2): factor the repeated
// scaffolding, not the genuinely distinct builder algorithms.

#include <cstddef>
#include <vector>

#include <Kokkos_Core.hpp>

namespace intti::detail {

/// Allocate a 1-D device View and copy a host std::vector into it.
template <class T>
Kokkos::View<T *> to_device(const std::vector<T> &v, const char *label) {
  Kokkos::View<T *> d(label, v.size());
  auto h = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
  Kokkos::deep_copy(d, h);
  return d;
}

/// Allocate a 1-D device View and copy n elements from a host pointer into it.
template <class T>
Kokkos::View<T *> to_device(const T *p, std::size_t n, const char *label) {
  Kokkos::View<T *> d(label, n);
  auto h = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < n; ++i) h(i) = p[i];
  Kokkos::deep_copy(d, h);
  return d;
}

/// Copy a 1-D device View back into a host std::vector.
template <class View> auto to_host(const View &d) {
  using T = typename View::non_const_value_type;
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d);
  std::vector<T> v(d.extent(0));
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = h(i);
  return v;
}

} // namespace intti::detail
