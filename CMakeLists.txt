cmake_minimum_required(VERSION 3.28)
project(tax VERSION 0.1.0 LANGUAGES CXX)

option(TAX_BUILD_UNITTESTS   "Build and enable unit tests"                  ON)
option(TAX_BUILD_REGRESSIONS "Build DACE-based regression tests"            OFF)
option(TAX_BUILD_BENCHMARK   "Build benchmark suite (Google Benchmark)"     OFF)
option(TAX_BUILD_EXAMPLES    "Build example programs (under examples/)"     OFF)
option(TAX_USE_UNROLL        "Use compile-time-unrolled M=1 Cauchy kernels" ON)
option(TAX_USE_STENCIL       "Use precomputed Cauchy stencils for M>=2"     ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wno-psabi)
endif()

add_library(tax INTERFACE)
add_library(tax::tax ALIAS tax)
target_compile_features(tax INTERFACE cxx_std_23)
target_include_directories(tax INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

if(TAX_USE_UNROLL)
    target_compile_definitions(tax INTERFACE TAX_USE_UNROLL=1)
endif()
if(TAX_USE_STENCIL)
    target_compile_definitions(tax INTERFACE TAX_USE_STENCIL=1)
endif()

find_package(Eigen3 REQUIRED)
target_link_libraries(tax INTERFACE Eigen3::Eigen)

# ---------------------------------------------------------------
# DACE dependency (only when regression suite is enabled)
# ---------------------------------------------------------------
set(TAX_DACE_TARGET "" CACHE INTERNAL "Resolved DACE target" FORCE)
if(TAX_BUILD_REGRESSIONS)
    include(FetchContent)
    if(NOT TARGET dace AND NOT TARGET dace_s
       AND NOT TARGET dace::dace AND NOT TARGET dace::dace_s)
        FetchContent_Declare(
            DACE
            GIT_REPOSITORY https://github.com/dacelib/dace.git
            GIT_TAG v2.1.0
        )
        FetchContent_MakeAvailable(DACE)
    endif()

    # Upstream-recommended aliases.
    if(TARGET dace AND NOT TARGET dace::dace)
        add_library(dace::dace ALIAS dace)
    endif()
    if(TARGET dace_s AND NOT TARGET dace::dace_s)
        add_library(dace::dace_s ALIAS dace_s)
    endif()

    if(TARGET dace::dace_s)
        set(TAX_DACE_TARGET dace::dace_s)
    elseif(TARGET dace_s)
        set(TAX_DACE_TARGET dace_s)
    elseif(TARGET dace::dace)
        set(TAX_DACE_TARGET dace::dace)
    elseif(TARGET dace)
        set(TAX_DACE_TARGET dace)
    else()
        message(FATAL_ERROR
            "TAX_BUILD_REGRESSIONS=ON but no DACE target was produced by FetchContent")
    endif()
    message(STATUS "DACE target for regression tests: ${TAX_DACE_TARGET}")
endif()

if(TAX_BUILD_UNITTESTS OR TAX_BUILD_REGRESSIONS)
    enable_testing()
endif()
if(TAX_BUILD_UNITTESTS)
    add_subdirectory(tests)
endif()
if(TAX_BUILD_REGRESSIONS)
    add_subdirectory(tests/regression)
endif()
if(TAX_BUILD_BENCHMARK)
    add_subdirectory(benchmarks)
endif()
if(TAX_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(TAX_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/tax")
install(TARGETS tax EXPORT taxTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT taxTargets FILE taxTargets.cmake NAMESPACE tax::
        DESTINATION ${TAX_INSTALL_CMAKEDIR})
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/taxConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/taxConfig.cmake"
    INSTALL_DESTINATION ${TAX_INSTALL_CMAKEDIR})
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/taxConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/taxConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/taxConfigVersion.cmake"
    DESTINATION ${TAX_INSTALL_CMAKEDIR})
