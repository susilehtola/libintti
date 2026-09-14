# SPDX-License-Identifier: BSD-3-Clause
cmake_minimum_required(VERSION 3.23)
project(intti VERSION 0.1.0
        DESCRIPTION "Efficient quadrature code for tensor expressions"
        LANGUAGES CXX)

option(INTTI_BUILD_TESTS "Build unit tests" ON)
option(INTTI_ENABLE_MPI "Enable MPI support (placeholder, used from later milestones)" OFF)
option(INTTI_ENABLE_CHOLESKY "Enable the Cholesky decomposition module (needs LAPACK)" ON)
option(INTTI_ENABLE_QUADMATH "Enable __float128 support (needs libquadmath; GCC/clang)" OFF)

# The libcint facade (intti_cint) is a shared library. When Kokkos is fetched
# rather than found, it builds as a static archive, and linking a non-PIC
# archive into a shared object fails; build everything position-independent.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Kokkos 4.0 QUIET)
if(Kokkos_FOUND)
  message(STATUS "Using system Kokkos ${Kokkos_VERSION}")
else()
  message(STATUS "System Kokkos not found - fetching")
  include(FetchContent)
  FetchContent_Declare(kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG        4.5.01)
  FetchContent_MakeAvailable(kokkos)
endif()

# header-only: the core is templated on the scalar type (arbitrary precision)
add_library(intti INTERFACE)
add_library(intti::intti ALIAS intti)
target_include_directories(intti INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(intti INTERFACE Kokkos::kokkos)
target_compile_features(intti INTERFACE cxx_std_20)

if(INTTI_ENABLE_CHOLESKY)
  find_package(LAPACK REQUIRED)
  target_link_libraries(intti INTERFACE LAPACK::LAPACK)
endif()

if(INTTI_ENABLE_QUADMATH)
  # __float128 has std::is_floating_point_v == true but Kokkos cannot handle
  # it; traits.hpp routes it to the serial host path. libquadmath supplies
  # expq/sqrtq/fabsq and M_PIq.
  target_compile_definitions(intti INTERFACE INTTI_HAVE_QUADMATH)
  target_link_libraries(intti INTERFACE quadmath)
endif()

if(INTTI_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_link_libraries(intti INTERFACE MPI::MPI_CXX)
  target_compile_definitions(intti INTERFACE INTTI_HAVE_MPI)
endif()

# libcint-compatible C facade: the one compiled (non-header-only) piece of
# the library, so it can be dlopen()'d / ctypes-loaded as a drop-in ERI
# backend (see include/intti/cint.h, docs/roadmap.md M7).
add_library(intti_cint SHARED src/cint.cpp)
add_library(intti::cint ALIAS intti_cint)
target_link_libraries(intti_cint PRIVATE intti::intti)
target_include_directories(intti_cint PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

if(INTTI_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
  add_executable(intti_dump_eri tools/dump_eri.cpp)
  target_link_libraries(intti_dump_eri PRIVATE intti::intti)
  add_executable(intti_dump_oneel tools/dump_oneel.cpp)
  target_link_libraries(intti_dump_oneel PRIVATE intti::intti)
  add_executable(intti_dump_ri tools/dump_ri.cpp)
  target_link_libraries(intti_dump_ri PRIVATE intti::intti)
  # J/K scaling over real basis sets. Built but NOT registered with ctest: a
  # single def2-QZVPPD point runs for minutes to hours.
  add_executable(intti_scaling bench/scaling.cpp)
  target_link_libraries(intti_scaling PRIVATE intti::intti)
  target_include_directories(intti_scaling PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bench)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_FOUND)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import pyscf"
      RESULT_VARIABLE _pyscf_rc OUTPUT_QUIET ERROR_QUIET)
    if(_pyscf_rc EQUAL 0)
      add_test(NAME PySCFValidation
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/prototype/pyscf_validation.py
                --dumper $<TARGET_FILE:intti_dump_eri> --workdir ${CMAKE_CURRENT_BINARY_DIR})
      add_test(NAME PySCFValidationFacade
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/prototype/pyscf_validation.py
                --facade $<TARGET_FILE:intti_cint>)
      add_test(NAME PySCFValidationOneEl
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/prototype/pyscf_oneel_validation.py
                $<TARGET_FILE:intti_dump_oneel> ${CMAKE_CURRENT_BINARY_DIR})
      # PySCF's OWN SCF driver running on intti's J/K via intti_get_jk -- the
      # matrix-level facade entry, not the per-quartet one.
      add_test(NAME PySCFScfBackend
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/prototype/pyscf_scf_backend.py
                $<TARGET_FILE:intti_cint>)
      add_test(NAME PySCFValidationRI
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/prototype/pyscf_ri_validation.py
                $<TARGET_FILE:intti_dump_ri> ${CMAKE_CURRENT_BINARY_DIR})
      message(STATUS "PySCF found - cross-validation tests enabled")
    else()
      message(STATUS "PySCF not found - cross-validation test skipped")
    endif()
  endif()
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS intti intti_cint EXPORT inttiTargets)
install(DIRECTORY include/intti DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT inttiTargets NAMESPACE intti::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/intti)
configure_package_config_file(cmake/inttiConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/inttiConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/intti)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/inttiConfigVersion.cmake
  COMPATIBILITY SameMajorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/inttiConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/inttiConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/intti)
