# SPDX-License-Identifier: BSD-3-Clause
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  message(STATUS "System GTest not found - fetching")
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

add_executable(intti_tests
  test_main.cpp
  test_tgrid.cpp
  test_tail.cpp
  test_boys.cpp
  test_hermite1d.cpp
  test_quartet.cpp
  test_multiprec.cpp
  test_batch.cpp
  test_bessel.cpp
  test_interaction.cpp
  test_jbuild.cpp
  test_fock.cpp
  test_c2s.cpp
  test_cdjk.cpp
  test_giao.cpp
  test_oneel.cpp
  test_contracted.cpp
  test_harmonics.cpp
  test_ngrid.cpp
  test_space.cpp
  test_nuclear.cpp
  test_ncenter.cpp
  test_localhybrid.cpp
  test_soc.cpp
  test_crossbasis.cpp
  test_scf.cpp
  test_jk.cpp
  test_deriv.cpp
  test_geoderiv.cpp
  test_erigrad.cpp
  test_erihess.cpp
  test_geohess.cpp
  test_rigrad.cpp
  test_sto.cpp
  test_threeel.cpp
  test_multipole.cpp
  test_fegrid.cpp
  test_gridri.cpp
  test_tc.cpp
  test_helmholtz.cpp
  test_blas.cpp
  test_screening.cpp)
if(INTTI_ENABLE_CHOLESKY)
  target_sources(intti_tests PRIVATE test_ri.cpp)
endif()
if(INTTI_ENABLE_CHOLESKY)
  target_sources(intti_tests PRIVATE test_threeel_ri.cpp)
endif()
if(INTTI_ENABLE_CHOLESKY)
  target_sources(intti_tests PRIVATE test_cholesky.cpp)
endif()
# The finite-element grid route lives here, not in include/intti, because it is
# a small-scale reference: validated against the analytic engine, with which it
# shares no code below the t grid, and useful precisely as an independent check.
# It is orders of magnitude slower than the analytic route on any real basis set
# (see docs/roadmap.md), so it is not shipped as API.
target_include_directories(intti_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(intti_tests PRIVATE intti::intti GTest::gtest)
include(GoogleTest)
gtest_discover_tests(intti_tests)

if(INTTI_ENABLE_MPI)
  # Separate binary (not gtest): MPI_Init must run before Kokkos::initialize,
  # and the pass/fail verdict is agreed across ranks via MPI_Allreduce before
  # anyone finalizes, so a plain main() returning 0/1 is the natural shape.
  add_executable(intti_test_mpi test_mpi.cpp)
  target_link_libraries(intti_test_mpi PRIVATE intti::intti)

  if(NOT MPIEXEC_EXECUTABLE)
    find_program(MPIEXEC_EXECUTABLE NAMES mpirun mpiexec)
  endif()

  foreach(np 1 2 4)
    add_test(NAME MPI_np${np}
      COMMAND ${MPIEXEC_EXECUTABLE} -np ${np} --oversubscribe
              $<TARGET_FILE:intti_test_mpi>)
  endforeach()
endif()
