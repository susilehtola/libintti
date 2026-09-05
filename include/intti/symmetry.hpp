// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Permutational-symmetry policy for matrix/tensor integral builds.
//
// Every many-index integral (ab|cd), (ab|cd|ef), ... carries a permutational
// symmetry group: the value is invariant under a set of index swaps, so the
// same unique integral feeds several output-matrix elements. A build can either
//   * ignore that symmetry (Symmetry::None) -- evaluate the integral separately
//     for every output block that needs it, each block owned by one work item
//     in a fixed summation order. No atomics; results are bit-reproducible and
//     the MPI-distributed build matches the serial one byte-for-byte; or
//   * exploit it (Symmetry::Full) -- evaluate each unique shell tuple ONCE and
//     scatter its contribution (via atomic adds) into every output block it
//     feeds. This cuts the number of (expensive) integral evaluations by the
//     size of the symmetry orbit (e.g. ~4x beyond the built-in 2x for the K
//     build's 8-fold ERI symmetry, ~up to 6x for a 3-electron aux integral),
//     but the atomic scatter makes the summation order nondeterministic, so a
//     Full build matches the None build only to rounding (~1e-13, not byte-for-
//     byte) and is not bit-reproducible across runs or rank counts.
//
// The flag is shared so the same opt-in applies uniformly across builds that
// support it (exchange today; the 3-electron builds are the next adopters).
// None is always the default, preserving the deterministic guarantees.

namespace intti {

enum class Symmetry { None, Full };

} // namespace intti
