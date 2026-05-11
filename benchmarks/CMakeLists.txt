include(FetchContent)

find_package(benchmark 1.9 QUIET)

if(NOT benchmark_FOUND AND NOT TARGET benchmark::benchmark)
    message(STATUS "Google Benchmark not found, fetching via FetchContent...")

    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.4
    )

    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
endif()

if(NOT TARGET benchmark::benchmark)
    message(FATAL_ERROR
            "TAX_BUILD_BENCHMARK is ON but no benchmark::benchmark target is available.")
endif()

set(_dace_target "")
if(TAX_USE_DACE)
    if(TAX_DACE_TARGET AND TARGET ${TAX_DACE_TARGET})
        set(_dace_target ${TAX_DACE_TARGET})
    elseif(TARGET dace::dace_s)
        set(_dace_target dace::dace_s)
    elseif(TARGET dace_s)
        set(_dace_target dace_s)
    elseif(TARGET dace::dace)
        set(_dace_target dace::dace)
    elseif(TARGET dace)
        set(_dace_target dace)
    else()
        message(FATAL_ERROR
                "TAX_USE_DACE is ON but no DACE target is available for benchmarks.")
    endif()

    message(STATUS "DACE target for benchmarks: ${_dace_target}")
endif()

add_executable(bench_univariate univariate.cpp)
target_link_libraries(bench_univariate PRIVATE tax benchmark::benchmark)

if(TAX_USE_DACE)
    target_compile_definitions(bench_univariate PRIVATE TAX_BENCH_HAVE_DACE=1)
    target_link_libraries(bench_univariate PRIVATE ${_dace_target})
endif()

add_executable(bench_dynamic_vs_static dynamic_vs_static.cpp)
target_link_libraries(bench_dynamic_vs_static PRIVATE tax benchmark::benchmark)

# -----------------------------------------------------------------------------
# Profiling drivers — vanilla `int main()` programs intended to be run under
# perf / callgrind / vtune / etc.  No Google Benchmark dependency so they
# present a clean call graph dominated by the library code.
#
# Each program runs the same workload many times and prints one summary line.
# Usage:
#   perf record -g --call-graph dwarf \
#       ./build/benchmarks/profile_da_integration 100
#   perf report
# -----------------------------------------------------------------------------
add_executable(profile_da_integration profile_da_integration.cpp)
target_link_libraries(profile_da_integration PRIVATE tax)
# Frame pointers and debug info make perf call graphs vastly more readable;
# do not strip release symbols in this profiling build.
target_compile_options(profile_da_integration PRIVATE -fno-omit-frame-pointer -g)

add_executable(profile_ads_integration profile_ads_integration.cpp)
target_link_libraries(profile_ads_integration PRIVATE tax)
target_compile_options(profile_ads_integration PRIVATE -fno-omit-frame-pointer -g)

# Per-op timing across the full arithmetic + math surface — separate
# binaries for univariate and multivariate so each profile run lands on a
# single shape without cross-talk in the call graph.
add_executable(profile_operations_univariate profile_operations_univariate.cpp)
target_link_libraries(profile_operations_univariate PRIVATE tax)
target_compile_options(profile_operations_univariate PRIVATE -fno-omit-frame-pointer -g)

add_executable(profile_operations_multivariate profile_operations_multivariate.cpp)
target_link_libraries(profile_operations_multivariate PRIVATE tax)
target_compile_options(profile_operations_multivariate PRIVATE -fno-omit-frame-pointer -g)
