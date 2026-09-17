# ---------------------------------------------------------------------------
# mgkmc_cpp - C++ port of the mgkmc small-strain solver
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#
# Produces build/mgkmc (build/mgkmc.exe on Windows), which takes a single YAML
# config path as its argument.
#
# Dependencies are header-only and vendored in third_party/, so no network
# access or package manager is required:
#   * Eigen 3.4.0    - dense linear algebra
#   * pocketfft      - FFTs for the spectral solver
#
# Set -DMGKMC_USE_SYSTEM_EIGEN=ON to link against an installed Eigen instead of
# the vendored copy.
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16)

project(mgkmc_cpp
    VERSION 0.1.0
    DESCRIPTION "Mesoscale KMC/AQS plasticity solver for metallic glasses (C++ port)"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to an optimised build: the solver is compute-bound and an unoptimised
# build is roughly an order of magnitude slower.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()

option(MGKMC_USE_SYSTEM_EIGEN "Use an installed Eigen instead of the vendored copy" OFF)
option(MGKMC_BUILD_TESTS "Build the standalone reference checks" ON)

# --- dependencies ----------------------------------------------------------

add_library(mgkmc_deps INTERFACE)

if(MGKMC_USE_SYSTEM_EIGEN)
    find_package(Eigen3 3.4 REQUIRED NO_MODULE)
    target_link_libraries(mgkmc_deps INTERFACE Eigen3::Eigen)
else()
    set(MGKMC_EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen-3.4.0")
    if(NOT EXISTS "${MGKMC_EIGEN_DIR}/Eigen/Dense")
        message(FATAL_ERROR
            "Vendored Eigen not found at ${MGKMC_EIGEN_DIR}. Either restore "
            "third_party/eigen-3.4.0 or configure with -DMGKMC_USE_SYSTEM_EIGEN=ON.")
    endif()
    target_include_directories(mgkmc_deps SYSTEM INTERFACE "${MGKMC_EIGEN_DIR}")
endif()

# pocketfft is a single header and is always taken from third_party/.
target_include_directories(mgkmc_deps SYSTEM INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

# The solver itself is header-only; expose the repository root so that
# #include "kmc_solver.hpp" and friends resolve.
add_library(mgkmc_core INTERFACE)
target_link_libraries(mgkmc_core INTERFACE mgkmc_deps)
target_include_directories(mgkmc_core INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

# --- main executable -------------------------------------------------------

add_executable(mgkmc main.cpp)
target_link_libraries(mgkmc PRIVATE mgkmc_core)

# GCC/Clang on Windows (MinGW) link libstdc++ and libwinpthread dynamically by
# default, so the binary will not start unless those DLLs are on PATH. Link them
# statically to keep the executable self-contained.
if(MINGW)
    target_link_options(mgkmc PRIVATE -static -static-libgcc -static-libstdc++)
endif()

# std::filesystem needs an explicit library on GCC 8 and older.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(mgkmc PRIVATE stdc++fs)
endif()

# --- reference checks ------------------------------------------------------

# test_landau_stress is a standalone program that prints the Landau stress
# response for four hand-picked strain states, for comparison against the
# Python reference values recorded in its source comments. It is not a pass/fail
# test suite.
if(MGKMC_BUILD_TESTS)
    add_executable(test_landau_stress test_landau_stress.cpp)
    target_link_libraries(test_landau_stress PRIVATE mgkmc_core)
    if(MINGW)
        target_link_options(test_landau_stress PRIVATE -static -static-libgcc -static-libstdc++)
    endif()
endif()

if(MGKMC_USE_SYSTEM_EIGEN)
    set(MGKMC_EIGEN_SOURCE "system")
else()
    set(MGKMC_EIGEN_SOURCE "vendored (third_party/eigen-3.4.0)")
endif()
message(STATUS "mgkmc_cpp ${PROJECT_VERSION}  build=${CMAKE_BUILD_TYPE}  eigen=${MGKMC_EIGEN_SOURCE}")
