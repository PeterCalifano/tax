include(FetchContent)

find_package(benchmark 1.8 QUIET)

if(NOT benchmark_FOUND)
    message(STATUS "google benchmark not found, fetching via FetchContent...")

    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.4
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
endif()

add_executable(bench_ops bench_ops.cpp)
target_link_libraries(bench_ops PRIVATE tax benchmark::benchmark)
