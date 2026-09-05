// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// Loop/Kernel/Consumer (LKC) scaffolding -- the first slice of the SHARK
// Sec 3.10 architecture (Neese, J. Comput. Chem. 2023, 44, 381; ideas only,
// ORCA/SHARK is not open). The observation behind LKC is that every matrix-
// level builder is the same shape -- enumerate a shell-pair index space, build
// the McMurchie-Davidson factors, (optionally sweep a kernel's nodes,) and
// SCATTER the result into an AO matrix -- and only the KERNEL (operator) and
// the CONSUMER (task: which matrix, contracted against what, written how)
// vary. In libintti the kernel is already unified as a node list (tgrid.hpp /
// kernel.hpp), so the transferable work is the CONSUMER seam.
//
// This header provides the consumer for the DETERMINISTIC explicit-integral
// pair builders: the 1e property matrices (oneel.hpp) and, later, the n-center
// tensors (ncenter.hpp) and nuclear attraction. It hoists the repeated
// "enumerate the (ka,kb) Cartesian components of shell pair (a,b), compute the
// AO row-major indices, write the block, and mirror it" boilerplate behind a
// value callback, so each builder states only its per-element numerics.
//
// NOT covered (by design, not omission): the fused J/K engines (jbuild.hpp /
// kbuild.hpp) never form explicit integrals -- their digestion is fused into
// the Hermite-Coulomb coupling (density pre-contracted into per-pair Hermite
// moments for J; contracted on the fly straddling both pairs for K), and
// hoisting it behind a generic consumer would DE-FUSE the hot path and
// regress it. The n^6 three-electron builder (threeel.hpp) likewise contracts
// the density three times inside its sextet loop. Those keep their bespoke
// digestion; see roadmap M-SHARK #3 (measured: the shareable part is ~1-2% of
// the cost). LKC consolidates the write-side of the *explicit-integral*
// builders, which is where the duplication actually lives.

#include <array>
#include <cstddef>

#include "fock.hpp"
#include "gto.hpp"

namespace intti {
namespace detail {

/// LKC consumer -- deterministic AO-matrix scatter for one shell pair (a,b).
///
/// Enumerates the Cartesian components (ka with exponents a3[3], kb with b3[3])
/// of the pair, evaluates `value(ka, a3, kb, b3)` for each, writes it at the
/// (a,b) block of the row-major nao x nao matrix `M`, and applies the transpose
/// mirror per `mirror`:
///   +1  symmetric      M[b,a] = +M[a,b]  (overlap, kinetic, multipole)
///   -1  antisymmetric  M[b,a] = -M[a,b]  (angular momentum)
///    0  none           caller already loops the full a,b space (gradient,
///                      kinetic-moment) so no mirror is written
/// The diagonal pair (a == b) is written once regardless of `mirror`.
/// This is a deterministic block-ownership write (plain assignment, no atomics);
/// each pair owns disjoint AO blocks, so it is byte-exact under any threading.
template <class Real, class F>
void scatter_pair(std::vector<Real> &M, const ShellBasis<Real> &basis, int a,
                  int b, int mirror, F value) {
  const int la = basis.shells[a].l, lb = basis.shells[b].l;
  const std::size_t oa = basis.ao_off[a], ob = basis.ao_off[b];
  const std::size_t nao = static_cast<std::size_t>(basis.nao);
  const bool offdiag = (a != b);
  for (int ka = 0; ka < ncart(la); ++ka) {
    int a3[3];
    cart_comp(la, ka, a3[0], a3[1], a3[2]);
    for (int kb = 0; kb < ncart(lb); ++kb) {
      int b3[3];
      cart_comp(lb, kb, b3[0], b3[1], b3[2]);
      const Real v = value(ka, a3, kb, b3);
      M[(oa + ka) * nao + ob + kb] = v;
      if (mirror && offdiag)
        M[(ob + kb) * nao + oa + ka] = mirror > 0 ? v : -v;
    }
  }
}

/// LKC consumer for a vector of three AO matrices sharing one pair enumeration
/// (angular momentum, gradient, kinetic-moment). `value(ka, a3, kb, b3)`
/// returns the three components as a std::array<Real,3>; `mirror` applies the
/// transpose sign as in scatter_pair (0 = full a,b loop, no mirror).
template <class Real, class F>
void scatter_pair3(std::array<std::vector<Real>, 3> &M,
                   const ShellBasis<Real> &basis, int a, int b, int mirror,
                   F value) {
  const int la = basis.shells[a].l, lb = basis.shells[b].l;
  const std::size_t oa = basis.ao_off[a], ob = basis.ao_off[b];
  const std::size_t nao = static_cast<std::size_t>(basis.nao);
  const bool offdiag = (a != b);
  for (int ka = 0; ka < ncart(la); ++ka) {
    int a3[3];
    cart_comp(la, ka, a3[0], a3[1], a3[2]);
    for (int kb = 0; kb < ncart(lb); ++kb) {
      int b3[3];
      cart_comp(lb, kb, b3[0], b3[1], b3[2]);
      const std::array<Real, 3> v = value(ka, a3, kb, b3);
      const std::size_t idx = (oa + ka) * nao + ob + kb;
      const std::size_t jdx = (ob + kb) * nao + oa + ka;
      for (int c = 0; c < 3; ++c) {
        M[c][idx] = v[c];
        if (mirror && offdiag) M[c][jdx] = mirror > 0 ? v[c] : -v[c];
      }
    }
  }
}

} // namespace detail
} // namespace intti
