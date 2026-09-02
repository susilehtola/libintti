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
  test_nuclear.cpp
  test_ncenter.cpp
  test_localhybrid.cpp
  test_deriv.cpp
  test_geoderiv.cpp
  test_erigrad.cpp
  test_erihess.cpp
  test_geohess.cpp
  test_rigrad.cpp
  test_sto.cpp
  test_threeel.cpp)
if(INTTI_ENABLE_CHOLESKY)
  target_sources(intti_tests PRIVATE test_ri.cpp)
endif()
if(INTTI_ENABLE_CHOLESKY)
  target_sources(intti_tests PRIVATE test_cholesky.cpp)
endif()
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
