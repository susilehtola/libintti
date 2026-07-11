# SPDX-License-Identifier: MPL-2.0
cmake_minimum_required(VERSION 3.23)
project(intti VERSION 0.1.0
        DESCRIPTION "Efficient quadrature code for tensor expressions"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(INTTI_BUILD_TESTS "Build unit tests" ON)
option(INTTI_ENABLE_MPI "Enable MPI support (placeholder, used from later milestones)" OFF)

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

add_library(intti
  src/tgrid.cpp
  src/quartet.cpp)
add_library(intti::intti ALIAS intti)
target_include_directories(intti PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(intti PUBLIC Kokkos::kokkos)
target_compile_features(intti PUBLIC cxx_std_20)

if(INTTI_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_link_libraries(intti PUBLIC MPI::MPI_CXX)
  target_compile_definitions(intti PUBLIC INTTI_HAVE_MPI)
endif()

if(INTTI_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS intti EXPORT inttiTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
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
