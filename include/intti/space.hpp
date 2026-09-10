// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Memory-space-flexible argument passing: a caller's data may live on the host
// or on the device, independently for inputs and outputs, and all four
// combinations are supported without a hidden copy when one is not needed.
//
// This exists because the engine is device-resident and the API was not. A code
// that keeps its density and Fock matrices on the GPU had to copy them down to
// call intti and copy the result back up -- twice per SCF iteration, for data
// that never needed to leave. Conversely a host-only caller must not be made to
// think about views at all.
//
// The rule is: a copy happens ONLY when the caller's data is not already in the
// space the kernel needs. Kokkos::create_mirror_view_and_copy is exactly that
// contract -- it returns the argument itself when the space already matches, so
// device-in/device-out costs nothing and host-in/host-out costs what it always
// did.
//
// Outputs cannot use the same helper, because a mirror is a COPY and writing
// into it would be discarded. DeviceOut therefore aliases the caller's view
// when it is already device-resident, and otherwise allocates a device buffer
// and copies back on commit(). commit() is explicit rather than a destructor,
// so the copy is visible at the call site and cannot happen on an exception
// path where the result is meaningless.

#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

namespace intti {

/// The space the kernels run in.
using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;

namespace detail {

/// True when `V` is a Kokkos View (as opposed to a pointer or std::vector).
template <class V> inline constexpr bool is_view_v = Kokkos::is_view<std::decay_t<V>>::value;

} // namespace detail

/// A read-only device view of `v`, without a copy when it is already there.
/// Accepts any Kokkos View in any memory space.
template <class V> auto device_in(const V &v) {
  static_assert(detail::is_view_v<V>, "device_in expects a Kokkos::View");
  return Kokkos::create_mirror_view_and_copy(DeviceSpace{}, v);
}

/// A device view of `n` elements copied from a host pointer. The host-caller
/// path, kept so raw-pointer entry points are one line.
template <class T> Kokkos::View<T *> device_in(const T *p, std::size_t n, const char *label) {
  Kokkos::View<std::remove_const_t<T> *> d(std::string(label), n);
  auto h = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < n; ++i) h(i) = p[i];
  Kokkos::deep_copy(d, h);
  return d;
}

/// Writable device staging for an output that may live in either space.
///
/// `view()` is what a kernel writes to; `commit()` publishes it. When the
/// destination is already device-resident the two are the same memory and
/// commit() does nothing, so device-out is free.
template <class Real> class DeviceOut {
public:
  using DView = Kokkos::View<Real *>;

  /// Destination is a Kokkos View, in either space.
  template <class V>
  DeviceOut(const V &dst, const char *label) : n_(dst.extent(0)) {
    if constexpr (std::is_same_v<typename std::decay_t<V>::memory_space, DeviceSpace>) {
      work_ = dst; // alias: no copy in, no copy out
      aliased_ = true;
    } else {
      work_ = DView(std::string(label), n_);
      host_dst_ = dst;
    }
  }

  /// Destination is a host pointer.
  DeviceOut(Real *dst, std::size_t n, const char *label)
      : n_(n), work_(std::string(label), n), raw_dst_(dst) {}

  const DView &view() const { return work_; }

  /// Publish the result. A no-op when the destination was already device-side.
  void commit() {
    if (aliased_) return;
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, work_);
    if (raw_dst_) {
      for (std::size_t i = 0; i < n_; ++i) raw_dst_[i] = h(i);
    } else if (host_dst_.data()) {
      Kokkos::deep_copy(host_dst_, work_);
    }
  }

private:
  std::size_t n_{0};
  DView work_;
  bool aliased_{false};
  Real *raw_dst_{nullptr};
  Kokkos::View<Real *, Kokkos::HostSpace> host_dst_{};
};

} // namespace intti
