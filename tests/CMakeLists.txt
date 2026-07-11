# SPDX-License-Identifier: MPL-2.0
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
  test_multiprec.cpp)
target_link_libraries(intti_tests PRIVATE intti::intti GTest::gtest)
include(GoogleTest)
gtest_discover_tests(intti_tests)
