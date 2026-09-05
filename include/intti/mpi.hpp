// SPDX-License-Identifier: BSD-3-Clause
// Copyright (C) 2026 Susi Lehtola
#pragma once

// MPI distribution of the Fock builds (M8). Guarded by INTTI_HAVE_MPI, which
// CMake defines when INTTI_ENABLE_MPI=ON (find_package(MPI) + MPI::MPI_CXX).
//
// Distribution strategy: coulomb_build/exchange_build accept an optional
// (rank, nranks) pair (jbuild.hpp, kbuild.hpp) that partitions the work index
// space -- bra shell pairs for J, (a, b) shell blocks for K -- strided by
// rank; each rank computes only its owned indices and leaves the rest of the
// output matrix zero. Because the owned index sets are disjoint and every
// output element is produced by exactly one rank, MPI_Allreduce(MPI_SUM)
// over the full (nao x nao) matrix reproduces the serial result exactly (up
// to floating-point summation-order round-off, since the *global* result is
// the same sum just partitioned across ranks instead of accumulated
// serially -- no element receives contributions from more than one rank).

#ifdef INTTI_HAVE_MPI

#include <mpi.h>

#include <type_traits>

#include "fock.hpp"

namespace intti {

/// Rank of this process in comm.
inline int mpi_rank(MPI_Comm comm) {
  int r = 0;
  MPI_Comm_rank(comm, &r);
  return r;
}

/// Size of comm.
inline int mpi_size(MPI_Comm comm) {
  int s = 1;
  MPI_Comm_size(comm, &s);
  return s;
}

namespace detail {
template <class Real> struct mpi_real_datatype;
template <> struct mpi_real_datatype<double> {
  static MPI_Datatype get() { return MPI_DOUBLE; }
};
template <> struct mpi_real_datatype<float> {
  static MPI_Datatype get() { return MPI_FLOAT; }
};
} // namespace detail

/// In-place sum-allreduce over an n-element Real buffer. Real must be double
/// or float (MPI_DOUBLE / MPI_FLOAT); no hand-rolled reduction.
template <class Real>
void mpi_allreduce_sum(Real *buf, int n, MPI_Comm comm) {
  static_assert(std::is_same_v<Real, double> || std::is_same_v<Real, float>,
                "mpi_allreduce_sum supports double and float only");
  MPI_Allreduce(MPI_IN_PLACE, buf, n, detail::mpi_real_datatype<Real>::get(),
                MPI_SUM, comm);
}

/// MPI-distributed Coulomb build: the bra shell-pair index space is
/// partitioned across the ranks of comm (see jbuild.hpp::coulomb_build), each
/// rank fills only its owned entries of J and the rest with zero, and the
/// result is summed across ranks so every rank returns the full J matrix.
template <class Real>
void coulomb_build_mpi(const ShellBasis<Real> &basis, const Real *D,
                      const TGrid<Real> &grid, Real *J, MPI_Comm comm,
                      Real tau = Real(0)) {
  const int rank = mpi_rank(comm);
  const int nranks = mpi_size(comm);
  coulomb_build(basis, D, grid, J, tau, rank, nranks);
  mpi_allreduce_sum(J, basis.nao * basis.nao, comm);
}

/// MPI-distributed exchange build: the (a, b) shell-block index space is
/// partitioned across the ranks of comm (see kbuild.hpp), each rank fills
/// only its owned entries of K and the rest with zero, and the result is
/// summed across ranks so every rank returns the full K matrix.
template <class Real>
void exchange_build_mpi(const ShellBasis<Real> &basis, const Real *D,
                        const TGrid<Real> &grid, Real *K, MPI_Comm comm,
                        Real tau = Real(1e-12), Symmetry sym = Symmetry::None) {
  const int rank = mpi_rank(comm);
  const int nranks = mpi_size(comm);
  exchange_build(basis, D, grid, K, tau, rank, nranks, sym);
  mpi_allreduce_sum(K, basis.nao * basis.nao, comm);
}

} // namespace intti

#endif // INTTI_HAVE_MPI
