# Tax Stage 1 — C++ Base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the `tax` C++ core from scratch on a new branch under TDD, delivering a single storage-policy-parameterized `tax::TaylorExpansion<T, N, M, Storage>` with dense and sparse variants, eager operators, the full math surface, and first-class Eigen integration.

**Architecture:** One class template with `storage::Dense` (default) and `storage::Sparse` policies. Eager operators (no expression templates). Kernels written against `TaylorPolynomial` concept with concept-based fast-path dispatch (`TAX_USE_UNROLL` for M=1, `TAX_USE_STENCIL` for M≥2). Eigen is a required dependency; public API consumes/produces `Eigen::Matrix` and `Eigen::Vector`. Internal "point" storage stays `std::array<T,M>`. Header-only, C++23.

**Tech Stack:** C++23, header-only, CMake ≥ 3.28, Eigen ≥ 3.4 (required), Google Test (tests), Google Benchmark (perf gate). Toolchain managed via `micromamba` env `tax` containing `cmake`, `gxx`, `eigen`, `benchmark`, `ninja`.

**Reference codebase:** the old code (still on `main` until we cut the branch) is the math spec for the kernels. When a task says "port from `<path>`", read that file and adapt it: change the `TaylorExpansionT` references to the new `TaylorExpansion` type, replace `std::array<T, nCoefficients>` with `tax::Coeffs<T,N,M>`, drop all `Dynamic`-shape branches, and route storage operations through the dense container's primitives.

**Spec:** `docs/superpowers/specs/2026-05-19-tax-stage1-design.md`

---

## File structure overview

What each new file owns:

| Path | Responsibility |
|---|---|
| `include/tax/tax.hpp` | Umbrella header; only file users include |
| `include/tax/core/concepts.hpp` | `Scalar`, `TaylorPolynomial`, `DensePolynomial`, `SparsePolynomial`, `StoragePolicy` |
| `include/tax/core/multi_index.hpp` | `MultiIndex<M>`, `flatIndex`, `numMonomials`, `binom`, `totalDegree`, `DegreeOf<N,M>`, `Coeffs<T,N,M>` |
| `include/tax/core/enumeration.hpp` | `forEachMonomial`, `forEachSubIndex` |
| `include/tax/core/storage/dense.hpp` | `storage::Dense` tag, `DenseContainer<T,N,M>` |
| `include/tax/core/storage/sparse.hpp` | `storage::Sparse` tag, `SparseContainer<T,N,M>`, `SparseCoeffs<T,N,M>`, `flat_index_t` |
| `include/tax/core/taylor_expansion.hpp` | `TaylorExpansion<T,N,M,Storage>` primary, aliases `TE`, `STE` |
| `include/tax/kernels/cauchy.hpp` | Dense Cauchy product loop variant + dispatch entry |
| `include/tax/kernels/cauchy_unroll.hpp` | `TAX_USE_UNROLL`-gated M=1 unrolled Cauchy variants |
| `include/tax/kernels/cauchy_stencil.hpp` | `TAX_USE_STENCIL`-gated M≥2 stencil Cauchy variants |
| `include/tax/kernels/algebra.hpp` | `seriesSquare`, `seriesCube`, `seriesSqrt`, `seriesCbrt`, `seriesReciprocal`, `seriesPow` (dense) |
| `include/tax/kernels/transcendental.hpp` | `seriesExp`, `seriesLog`, `seriesSinh`, `seriesCosh`, `seriesTanh`, inverse hyperbolics, `seriesErf` (dense) |
| `include/tax/kernels/trigonometric.hpp` | `seriesSin`, `seriesCos`, `seriesTan`, `seriesAsin`, `seriesAcos`, `seriesAtan`, `seriesAtan2` (dense) |
| `include/tax/kernels/sparse_cauchy.hpp` | Sparse Cauchy product/self-product/accumulate |
| `include/tax/kernels/sparse_subs.hpp` | Sparse sqrt, reciprocal, division, integer pow |
| `include/tax/operators/arithmetic.hpp` | `+`, `-`, `*`, `/`, scalar variants (eager free functions) |
| `include/tax/operators/math_unary.hpp` | `sin`, `cos`, `exp`, `log`, etc. (free functions wrapping kernels) |
| `include/tax/operators/math_binary.hpp` | `pow`, `atan2` |
| `include/tax/eigen.hpp` | `NumTraits`, `variables`, `gradient`, `hessian`, `jacobian`, `value`, `eval`, `derivative`, `invert` |
| `tests/CMakeLists.txt` | Test registration |
| `tests/testUtils.hpp` | `ExpectCoeffsNear`, tolerances |
| `tests/core/*.cpp` | One test executable per concern (foundations, ctor, accessors, deriv/integ) |
| `tests/kernels/*.cpp` | One test executable per kernel family |
| `tests/operators/*.cpp` | One test executable per math operator |
| `tests/sparse/*.cpp` | Sparse construction, conversion, arithmetic, kernels |
| `tests/eigen/*.cpp` | NumTraits, variables, eval, derivative, gradient, hessian, jacobian, invert |
| `benchmarks/baseline/main-<sha>.txt` | Captured baseline numbers from `main`, regression target |
| `benchmarks/ops_dense.cpp` | Dense Cauchy + math operators across (N,M) |
| `benchmarks/ops_sparse.cpp` | Sparse equivalents |
| `benchmarks/eigen_workflows.cpp` | gradient/Jacobian end-to-end |
| `.github/workflows/tests.yml` | Slimmed CI |
| `.github/workflows/sanitizers.yml` | ASAN/UBSAN/TSAN |
| `.github/workflows/bench.yml` | Perf gate |

---

## Task 0: Pre-flight — capture baseline, cut branch, scaffold

**Files:**
- Create: `benchmarks/baseline/main-<sha>.txt` (captured output)
- Delete: `include/tax/ode/`, `include/tax/ads/`, `include/tax/eigen/`, `include/tax/expr/`, `include/tax/utils/`, `include/tax/storage/sparse_tte.hpp`, `include/tax/storage/tte_static.hpp`, `include/tax/storage/tte_dynamic.hpp`, `include/tax/storage/tte_dynamic_order.hpp`, `include/tax/storage/shape.hpp`, `include/tax/kernels/`, `include/tax/operators/`, `python/`, `examples/`, `tools/`
- Modify: `CMakeLists.txt` (slim), `benchmarks/CMakeLists.txt` (slim), `tests/CMakeLists.txt` (empty stub)
- Create: `include/tax/tax.hpp` (empty stub), `include/tax/core/`, `include/tax/kernels/`, `include/tax/operators/` directories

- [ ] **Step 1: Activate env and capture current state**

```bash
eval "$(micromamba shell hook --shell bash)" && micromamba activate tax
cd /home/andrea/Documenti/Codes/tax
git status                                          # must be clean
git rev-parse HEAD                                  # record current SHA
```

- [ ] **Step 2: Build and run existing benchmarks on `main` as baseline**

```bash
git switch main || git checkout main
cmake -S . -B build-baseline -DCMAKE_BUILD_TYPE=Release -DTAX_BUILD_BENCHMARK=ON -DTAX_BUILD_TEST=OFF -DTAX_USE_DACE=ON -G Ninja
cmake --build build-baseline -j$(nproc)
mkdir -p benchmarks/baseline
SHA=$(git rev-parse --short HEAD)
./build-baseline/benchmarks/bench_univariate    --benchmark_min_time=0.5s > benchmarks/baseline/main-$SHA-univariate.txt
./build-baseline/benchmarks/bench_multivariate  --benchmark_min_time=0.5s > benchmarks/baseline/main-$SHA-multivariate.txt
./build-baseline/benchmarks/bench_vs_dace       --benchmark_min_time=0.5s > benchmarks/baseline/main-$SHA-vs-dace.txt
```

Expected: three output files with timing tables.

- [ ] **Step 3: Cut Stage 1 branch from current head**

```bash
git switch -                                        # back to claude/add-verner-integrators-vEgRF
git switch -c stage1-cpp-base
git status                                          # confirm clean
```

- [ ] **Step 4: Stash baseline files (they need to survive the deletion)**

```bash
git checkout main -- benchmarks/baseline/
git add benchmarks/baseline/ && git commit -m "bench: capture main perf baseline for Stage 1 regression gate"
```

- [ ] **Step 5: Delete out-of-scope source folders**

```bash
git rm -r include/tax/ode include/tax/ads include/tax/eigen include/tax/expr include/tax/utils
git rm include/tax/storage/sparse_tte.hpp include/tax/storage/tte_static.hpp
git rm include/tax/storage/tte_dynamic.hpp include/tax/storage/tte_dynamic_order.hpp
git rm include/tax/storage/shape.hpp
git rm -r include/tax/kernels include/tax/operators
git rm include/tax/ads.hpp include/tax/kernels.hpp include/tax/operators.hpp include/tax/utils.hpp
git rm -r python examples tools
```

- [ ] **Step 6: Delete out-of-scope test trees**

```bash
git rm -r tests/ads tests/ode tests/dace tests/dynamic tests/eigen
git rm -r tests/core tests/expr tests/foundation tests/kernels tests/sparse
git rm tests/testUtils.hpp tests/CMakeLists.txt
```

- [ ] **Step 7: Delete out-of-scope benchmark targets** (keep `baseline/` dir + reuse the CMakeLists scaffold)

```bash
git rm benchmarks/univariate.cpp benchmarks/multivariate.cpp benchmarks/dynamic_vs_static.cpp benchmarks/bench_vs_dace.cpp
git rm benchmarks/profile_*.cpp
rm -rf benchmarks/results
```

- [ ] **Step 8: Replace root `CMakeLists.txt` with the Stage 1 version**

Create at `CMakeLists.txt` (full replacement):

```cmake
cmake_minimum_required(VERSION 3.28)
project(tax VERSION 0.1.0 LANGUAGES CXX)

option(TAX_BUILD_TEST      "Build and enable unit tests"            ON)
option(TAX_BUILD_BENCHMARK "Build benchmark suite (Google Benchmark)" OFF)
option(TAX_USE_UNROLL      "Use compile-time-unrolled M=1 Cauchy kernels" ON)
option(TAX_USE_STENCIL     "Use precomputed Cauchy stencils for M>=2"     ON)

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

find_package(Eigen3 3.4 REQUIRED)
target_link_libraries(tax INTERFACE Eigen3::Eigen)

if(TAX_BUILD_TEST)
    enable_testing()
    add_subdirectory(tests)
endif()
if(TAX_BUILD_BENCHMARK)
    add_subdirectory(benchmarks)
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
```

- [ ] **Step 9: Replace `tests/CMakeLists.txt` with empty stub**

Create at `tests/CMakeLists.txt`:

```cmake
include(FetchContent)
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()
# Tests are added by subsequent slices via tax_add_test(name SOURCES files...).
function(tax_add_test name)
    cmake_parse_arguments(T "" "" "SOURCES" ${ARGN})
    add_executable(${name} ${T_SOURCES})
    target_link_libraries(${name} PRIVATE tax GTest::gtest GTest::gtest_main)
    add_test(NAME ${name} COMMAND ${name})
endfunction()
```

- [ ] **Step 10: Replace `benchmarks/CMakeLists.txt` with empty stub**

Create at `benchmarks/CMakeLists.txt`:

```cmake
include(FetchContent)
find_package(benchmark 1.9 QUIET)
if(NOT benchmark_FOUND AND NOT TARGET benchmark::benchmark)
    FetchContent_Declare(benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.4)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
endif()
function(tax_add_bench name)
    cmake_parse_arguments(B "" "" "SOURCES" ${ARGN})
    add_executable(${name} ${B_SOURCES})
    target_link_libraries(${name} PRIVATE tax benchmark::benchmark)
endfunction()
```

- [ ] **Step 11: Create directory skeleton + empty umbrella header**

```bash
mkdir -p include/tax/core/storage include/tax/kernels include/tax/operators
mkdir -p tests/core tests/kernels tests/operators tests/sparse tests/eigen
mkdir -p benchmarks
```

Create `include/tax/tax.hpp`:

```cpp
#pragma once
// Stage 1: progressively populated by each implementation slice.
// Users should include only this header.
```

Create `include/tax/eigen.hpp`:

```cpp
#pragma once
// Stage 1: populated in slices 18-19.
```

- [ ] **Step 12: Verify the empty build configures, compiles, and tests run with zero tests**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTAX_BUILD_TEST=ON -G Ninja
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

Expected: configure succeeds, build succeeds (no targets), `ctest` reports "No tests were found".

- [ ] **Step 13: Commit the scaffold**

```bash
git add -A
git commit -m "stage1: scaffold — delete out-of-scope code, empty include tree, slim CMake"
```

---

## Task 1: Foundations — combinatorics, multi-index, enumeration

**Files:**
- Create: `include/tax/core/multi_index.hpp`
- Create: `include/tax/core/enumeration.hpp`
- Create: `include/tax/core/concepts.hpp` (initial Scalar concept only)
- Create: `tests/core/test_multi_index.cpp`
- Create: `tests/core/test_enumeration.cpp`
- Modify: `tests/CMakeLists.txt` (register the two tests)
- Modify: `include/tax/tax.hpp` (include foundations)

- [ ] **Step 1: Write the failing test for `numMonomials` and `binom`**

Create `tests/core/test_multi_index.cpp`:

```cpp
#include <gtest/gtest.h>
#include <tax/tax.hpp>

TEST(NumMonomials, KnownValues) {
    // numMonomials(N, M) = C(N+M, M)
    EXPECT_EQ(tax::numMonomials(0, 1), 1u);
    EXPECT_EQ(tax::numMonomials(3, 1), 4u);     // 1 + x + x^2 + x^3
    EXPECT_EQ(tax::numMonomials(2, 2), 6u);     // 1, x, y, x^2, xy, y^2
    EXPECT_EQ(tax::numMonomials(4, 3), 35u);
}

TEST(Binom, KnownValues) {
    EXPECT_EQ(tax::detail::binom(5, 2), 10u);
    EXPECT_EQ(tax::detail::binom(7, 0), 1u);
    EXPECT_EQ(tax::detail::binom(7, 7), 1u);
    EXPECT_EQ(tax::detail::binom(-1, 0), 0u);
    EXPECT_EQ(tax::detail::binom(3, 5), 0u);
}

TEST(FlatIndex, RoundTripUni) {
    EXPECT_EQ(tax::flatIndex<1>({0}), 0u);
    EXPECT_EQ(tax::flatIndex<1>({3}), 3u);
}

TEST(FlatIndex, RoundTripBiVar) {
    // Graded-lex over (a, b): (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...
    using MI = tax::MultiIndex<2>;
    EXPECT_EQ(tax::flatIndex<2>(MI{0, 0}), 0u);
    EXPECT_EQ(tax::flatIndex<2>(MI{1, 0}), 1u);
    EXPECT_EQ(tax::flatIndex<2>(MI{0, 1}), 2u);
    EXPECT_EQ(tax::flatIndex<2>(MI{2, 0}), 3u);
    EXPECT_EQ(tax::flatIndex<2>(MI{1, 1}), 4u);
    EXPECT_EQ(tax::flatIndex<2>(MI{0, 2}), 5u);
}

TEST(TotalDegree, Sum) {
    using MI = tax::MultiIndex<3>;
    EXPECT_EQ(tax::totalDegree(MI{0, 0, 0}), 0);
    EXPECT_EQ(tax::totalDegree(MI{2, 1, 3}), 6);
}
```

- [ ] **Step 2: Register the test in `tests/CMakeLists.txt`**

Append to `tests/CMakeLists.txt`:

```cmake
tax_add_test(test_multi_index SOURCES core/test_multi_index.cpp)
```

- [ ] **Step 3: Run the test to confirm it fails to compile**

```bash
cmake --build build -j$(nproc) 2>&1 | head -20
```

Expected: `tax::numMonomials` / `tax::MultiIndex` not found.

- [ ] **Step 4: Implement `core/multi_index.hpp`**

Create `include/tax/core/multi_index.hpp` — port from `main:include/tax/utils/combinatorics.hpp` and `main:include/tax/utils/fwd.hpp`, hoisting `Coeffs<T,N,M>`, `binom`, `numMonomials`, `MultiIndex<M>`, `flatIndex<M>`, `totalDegree`, `DegreeOf<N,M>` into a single file:

```cpp
#pragma once

#include <array>
#include <cstddef>
#include <span>

namespace tax {

template <int M>
using MultiIndex = std::array<int, std::size_t(M)>;

namespace detail {
    constexpr std::size_t binom(int n, int k) noexcept {
        if (k < 0 || n < 0 || k > n) return 0;
        if (k == 0 || k == n) return 1;
        if (k > n - k) k = n - k;
        std::size_t r = 1;
        for (int i = 0; i < k; ++i) {
            r *= std::size_t(n - i);
            r /= std::size_t(i + 1);
        }
        return r;
    }
}

constexpr std::size_t numMonomials(int N, int M) noexcept {
    return detail::binom(N + M, M);
}

template <typename T, int N, int M>
using Coeffs = std::array<T, numMonomials(N, M)>;

template <int M>
constexpr int totalDegree(const MultiIndex<M>& a) noexcept {
    int d = 0;
    for (int i = 0; i < M; ++i) d += a[i];
    return d;
}

// flatIndex: map a multi-index in graded-lex order to a flat array position.
// Port from main:include/tax/utils/combinatorics.hpp lines 60-120 (the
// existing tax::detail::flatIndex implementation) and lift to tax::.
template <int M>
constexpr std::size_t flatIndex(const MultiIndex<M>& alpha) noexcept;

// Compile-time degree-of lookup: DegreeOf<N,M>::value[k] = degree of monomial k.
template <int N, int M>
struct DegreeOf;  // port from main:include/tax/utils/degree_of.hpp

}  // namespace tax
```

Fill in the `flatIndex` body and `DegreeOf` body by porting from the reference files. Drop the runtime-`std::span` overload (Stage 1 is static-only).

- [ ] **Step 5: Build and run the test**

```bash
cmake --build build -j$(nproc)
ctest --test-dir build -R test_multi_index --output-on-failure
```

Expected: PASS.

- [ ] **Step 6: Write the failing test for `forEachMonomial` / `forEachSubIndex`**

Create `tests/core/test_enumeration.cpp`:

```cpp
#include <gtest/gtest.h>
#include <tax/tax.hpp>
#include <vector>

TEST(ForEachMonomial, VisitsAllInGradedLexOrder) {
    std::vector<tax::MultiIndex<2>> visited;
    tax::forEachMonomial<2, 3>([&](const tax::MultiIndex<2>& a) {
        visited.push_back(a);
    });
    ASSERT_EQ(visited.size(), tax::numMonomials(3, 2));
    EXPECT_EQ(visited[0], (tax::MultiIndex<2>{0, 0}));
    EXPECT_EQ(visited[1], (tax::MultiIndex<2>{1, 0}));
    EXPECT_EQ(visited[2], (tax::MultiIndex<2>{0, 1}));
}

TEST(ForEachSubIndex, SumsToOuter) {
    // For an outer multi-index (2, 1), sub-indices (k, alpha-k) span all
    // componentwise <= alpha pairs.
    using MI = tax::MultiIndex<2>;
    int count = 0;
    tax::forEachSubIndex<2>(MI{2, 1}, [&](const MI& k, const MI& sub) {
        EXPECT_EQ(k[0] + sub[0], 2);
        EXPECT_EQ(k[1] + sub[1], 1);
        ++count;
    });
    EXPECT_EQ(count, (2 + 1) * (1 + 1));  // (alpha_0+1)*(alpha_1+1)
}
```

Register:

```cmake
tax_add_test(test_enumeration SOURCES core/test_enumeration.cpp)
```

- [ ] **Step 7: Implement `core/enumeration.hpp`**

Port from `main:include/tax/utils/enumeration.hpp`. Drop dynamic-shape variants. Public names: `tax::forEachMonomial<M, N>(callable)` (M first — callback receives `MultiIndex<M>`, so M anchors the first template parameter), `tax::forEachSubIndex<M>(alpha, callable)`.

- [ ] **Step 8: Build, run both tests**

```bash
cmake --build build -j$(nproc)
ctest --test-dir build -R "test_multi_index|test_enumeration" --output-on-failure
```

Expected: both PASS.

- [ ] **Step 9: Create `core/concepts.hpp` with `Scalar`**

Create `include/tax/core/concepts.hpp`:

```cpp
#pragma once

#include <concepts>

namespace tax {

template <typename T>
concept Scalar = std::floating_point<T>;

// TaylorPolynomial / DensePolynomial / SparsePolynomial concepts are filled
// in slices 2 and 9 once the types exist.

}  // namespace tax
```

- [ ] **Step 10: Wire foundations into the umbrella header**

Edit `include/tax/tax.hpp`:

```cpp
#pragma once

#include <tax/core/concepts.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/enumeration.hpp>
```

- [ ] **Step 11: Commit**

```bash
git add include/tax/core tests/core tests/CMakeLists.txt include/tax/tax.hpp
git commit -m "slice1: foundations — MultiIndex, flatIndex, Coeffs, enumeration, Scalar concept"
```

---

## Task 2: Dense storage + `TaylorExpansion` (dense path)

**Files:**
- Create: `include/tax/core/storage/dense.hpp`
- Create: `include/tax/core/taylor_expansion.hpp`
- Create: `tests/core/test_taylor_expansion_ctor.cpp`
- Create: `tests/core/test_taylor_expansion_accessors.cpp`
- Modify: `tests/CMakeLists.txt`
- Modify: `include/tax/tax.hpp`
- Modify: `include/tax/core/concepts.hpp` (add `TaylorPolynomial`, `DensePolynomial`)

- [ ] **Step 1: Write failing test for construction + variable factory**

Create `tests/core/test_taylor_expansion_ctor.cpp`:

```cpp
#include <gtest/gtest.h>
#include <tax/tax.hpp>

TEST(TaylorExpansion, ZeroCtor) {
    tax::TE<3> z;
    EXPECT_EQ(z.value(), 0.0);
    for (std::size_t k = 0; k < z.nCoefficients; ++k) EXPECT_EQ(z[k], 0.0);
}

TEST(TaylorExpansion, ConstantCtor) {
    tax::TE<3> c{2.5};
    EXPECT_EQ(c.value(), 2.5);
    for (std::size_t k = 1; k < c.nCoefficients; ++k) EXPECT_EQ(c[k], 0.0);
}

TEST(TaylorExpansion, VariableFactoryUni) {
    auto x = tax::TE<5>::variable(1.0);
    EXPECT_EQ(x.value(), 1.0);
    EXPECT_EQ(x[1], 1.0);  // d/dx coefficient
    for (std::size_t k = 2; k < x.nCoefficients; ++k) EXPECT_EQ(x[k], 0.0);
}

TEST(TaylorExpansion, VariableFactoryMulti) {
    typename tax::TE<3, 2>::Input p{1.0, 2.0};
    auto x = tax::TE<3, 2>::variable<0>(p);
    auto y = tax::TE<3, 2>::variable<1>(p);
    EXPECT_EQ(x.value(), 1.0);
    EXPECT_EQ(y.value(), 2.0);
    // x has coefficient 1 at multi-index (1,0); y at (0,1)
    EXPECT_EQ(x.coeff(tax::MultiIndex<2>{1, 0}), 1.0);
    EXPECT_EQ(x.coeff(tax::MultiIndex<2>{0, 1}), 0.0);
    EXPECT_EQ(y.coeff(tax::MultiIndex<2>{1, 0}), 0.0);
    EXPECT_EQ(y.coeff(tax::MultiIndex<2>{0, 1}), 1.0);
}
```

Register:

```cmake
tax_add_test(test_taylor_expansion_ctor SOURCES core/test_taylor_expansion_ctor.cpp)
```

- [ ] **Step 2: Run, confirm it fails to compile**

```bash
cmake --build build -j$(nproc) 2>&1 | head
```

Expected: `tax::TE` not found.

- [ ] **Step 3: Implement `core/storage/dense.hpp`**

Create `include/tax/core/storage/dense.hpp`:

```cpp
#pragma once

#include <array>
#include <cstddef>
#include <tax/core/multi_index.hpp>

namespace tax::storage {

struct Dense {};

template <typename T, int N, int M>
struct DenseContainer {
    using value_type    = T;
    using coeffs_type   = Coeffs<T, N, M>;
    static constexpr std::size_t nCoefficients = numMonomials(N, M);

    coeffs_type data{};

    constexpr T value() const noexcept { return data[0]; }
    constexpr T  operator[](std::size_t k) const noexcept { return data[k]; }
    constexpr T& operator[](std::size_t k)       noexcept { return data[k]; }

    constexpr void set(std::size_t k, T v) noexcept { data[k] = v; }
    constexpr void accumulate(std::size_t k, T v) noexcept { data[k] += v; }

    template <typename Fn>
    constexpr void forEachNonzero(Fn&& fn) const noexcept {
        for (std::size_t k = 0; k < nCoefficients; ++k) fn(k, data[k]);
    }
};

}  // namespace tax::storage
```

- [ ] **Step 4: Implement `core/taylor_expansion.hpp` (dense path)**

Create `include/tax/core/taylor_expansion.hpp`:

```cpp
#pragma once

#include <cstddef>
#include <tax/core/concepts.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/storage/dense.hpp>

namespace tax {

template <typename T, int N, int M = 1, typename Storage = storage::Dense>
class TaylorExpansion;

// Dense specialization
template <typename T, int N, int M>
class TaylorExpansion<T, N, M, storage::Dense> {
public:
    static_assert(N >= 0, "TaylorExpansion order must be non-negative");
    static_assert(M >= 1, "TaylorExpansion variable count must be at least 1");

    using scalar_type = T;
    using container_t = storage::DenseContainer<T, N, M>;
    using Input       = std::array<T, std::size_t(M)>;
    using Data        = Coeffs<T, N, M>;

    static constexpr int order_v = N;
    static constexpr int vars_v  = M;
    static constexpr std::size_t nCoefficients = numMonomials(N, M);

    // -- Constructors --------------------------------------------------------
    constexpr TaylorExpansion() noexcept = default;
    /*implicit*/ constexpr TaylorExpansion(T val) noexcept { c_.set(0, val); }
    explicit constexpr TaylorExpansion(Data c) noexcept : c_{c} {}

    // -- Factories -----------------------------------------------------------
    [[nodiscard]] static constexpr TaylorExpansion zero() noexcept { return {}; }
    [[nodiscard]] static constexpr TaylorExpansion constant(T v) noexcept { return TaylorExpansion{v}; }

    // Univariate variable.
    [[nodiscard]] static constexpr TaylorExpansion variable(T x0) noexcept
        requires (M == 1)
    {
        TaylorExpansion r{x0};
        if constexpr (N >= 1) r.c_.set(1, T{1});
        return r;
    }

    // Multivariate variable: I is the variable index.
    template <int I>
    [[nodiscard]] static constexpr TaylorExpansion variable(const Input& p) noexcept
        requires (M >= 1 && I >= 0 && I < M)
    {
        TaylorExpansion r{};
        // Constant term = expansion point's I-th coordinate.
        r.c_.set(0, p[std::size_t(I)]);
        if constexpr (N >= 1) {
            MultiIndex<M> alpha{};
            alpha[std::size_t(I)] = 1;
            r.c_.set(flatIndex<M>(alpha), T{1});
        }
        return r;
    }

    // -- Accessors -----------------------------------------------------------
    [[nodiscard]] constexpr T value() const noexcept { return c_.value(); }
    [[nodiscard]] constexpr T  operator[](std::size_t k) const noexcept { return c_[k]; }
    [[nodiscard]] constexpr T& operator[](std::size_t k)       noexcept { return c_[k]; }

    [[nodiscard]] constexpr T coeff(const MultiIndex<M>& alpha) const noexcept {
        return c_[flatIndex<M>(alpha)];
    }

    template <int... Alpha>
    [[nodiscard]] constexpr T coeff() const noexcept {
        static_assert(sizeof...(Alpha) == std::size_t(M),
                      "coeff<Alpha...>(): arity must match variable count");
        static_assert(((Alpha >= 0) && ...), "coeff<Alpha...>(): negative exponent");
        constexpr int total = (Alpha + ... + 0);
        static_assert(total <= N, "coeff<Alpha...>(): total degree exceeds N");
        constexpr MultiIndex<M> a{Alpha...};
        return c_[flatIndex<M>(a)];
    }

    // derivative(alpha) = coeff(alpha) * prod(alpha_i!).
    [[nodiscard]] constexpr T derivative(const MultiIndex<M>& alpha) const noexcept {
        std::size_t fac = 1;
        for (int i = 0; i < M; ++i)
            for (int j = 1; j <= alpha[std::size_t(i)]; ++j) fac *= std::size_t(j);
        return coeff(alpha) * T(fac);
    }

    template <int... Alpha>
    [[nodiscard]] constexpr T derivative() const noexcept {
        static_assert(sizeof...(Alpha) == std::size_t(M),
                      "derivative<Alpha...>(): arity must match variable count");
        static_assert(((Alpha >= 0) && ...));
        constexpr int total = (Alpha + ... + 0);
        static_assert(total <= N, "derivative<Alpha...>(): total degree exceeds N");
        constexpr auto factorial = [](int n) constexpr {
            std::size_t r = 1; for (int i = 2; i <= n; ++i) r *= std::size_t(i); return r;
        };
        constexpr std::size_t fac = (factorial(Alpha) * ... * 1);
        return coeff<Alpha...>() * T(fac);
    }

    // -- Container access ----------------------------------------------------
    [[nodiscard]] constexpr const container_t& container() const noexcept { return c_; }
    [[nodiscard]] constexpr       container_t& container()       noexcept { return c_; }

private:
    container_t c_{};
};

// Public aliases.
template <int N, int M = 1>  using TE = TaylorExpansion<double, N, M, storage::Dense>;

}  // namespace tax
```

- [ ] **Step 5: Add `TaylorPolynomial` and `DensePolynomial` concepts**

Edit `include/tax/core/concepts.hpp`:

```cpp
#pragma once

#include <concepts>
#include <cstddef>

namespace tax {

template <typename T> concept Scalar = std::floating_point<T>;

template <typename P>
concept TaylorPolynomial = requires(const P& p, std::size_t k) {
    typename P::scalar_type;
    typename P::container_t;
    { P::order_v } -> std::convertible_to<int>;
    { P::vars_v  } -> std::convertible_to<int>;
    { P::nCoefficients } -> std::convertible_to<std::size_t>;
    { p.value() } -> std::convertible_to<typename P::scalar_type>;
};

template <typename P>
concept DensePolynomial = TaylorPolynomial<P> && requires(const P& p, std::size_t k) {
    { p[k] } -> std::convertible_to<typename P::scalar_type>;
};

}  // namespace tax
```

- [ ] **Step 6: Wire into the umbrella header**

Edit `include/tax/tax.hpp`:

```cpp
#pragma once

#include <tax/core/concepts.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/enumeration.hpp>
#include <tax/core/storage/dense.hpp>
#include <tax/core/taylor_expansion.hpp>
```

- [ ] **Step 7: Build and run**

```bash
cmake --build build -j$(nproc)
ctest --test-dir build -R test_taylor_expansion_ctor --output-on-failure
```

Expected: PASS.

- [ ] **Step 8: Write failing test for accessors (compile-time coeff/derivative)**

Create `tests/core/test_taylor_expansion_accessors.cpp`:

```cpp
#include <gtest/gtest.h>
#include <tax/tax.hpp>

TEST(Accessors, CompileTimeCoeff) {
    typename tax::TE<3, 2>::Input p{1.0, 2.0};
    auto x = tax::TE<3, 2>::variable<0>(p);
    EXPECT_EQ((x.coeff<0, 0>()), 1.0);
    EXPECT_EQ((x.coeff<1, 0>()), 1.0);
    EXPECT_EQ((x.coeff<0, 1>()), 0.0);
}

TEST(Accessors, CompileTimeDerivativeMultipliesByFactorial) {
    // For f = x at x0=1.0 in TE<3,2>, coeff(2,0) is zero (it's a variable, not a square).
    // Test on a manually constructed polynomial: f = 1 + 2*x + 3*x^2 (univariate)
    tax::TE<3> f;
    f[0] = 1.0;
    f[1] = 2.0;
    f[2] = 3.0;
    EXPECT_EQ(f.coeff(tax::MultiIndex<1>{2}), 3.0);
    EXPECT_EQ(f.derivative(tax::MultiIndex<1>{2}), 6.0);  // 3 * 2!
    EXPECT_EQ(f.template derivative<2>(), 6.0);
}

TEST(Accessors, RuntimeDerivativeUni) {
    tax::TE<4> f;
    for (std::size_t k = 0; k < f.nCoefficients; ++k) f[k] = double(k);
    // derivative at order k = c_k * k!
    EXPECT_EQ(f.derivative(tax::MultiIndex<1>{0}), 0.0);
    EXPECT_EQ(f.derivative(tax::MultiIndex<1>{1}), 1.0);
    EXPECT_EQ(f.derivative(tax::MultiIndex<1>{2}), 4.0);    // 2 * 2!
    EXPECT_EQ(f.derivative(tax::MultiIndex<1>{3}), 18.0);   // 3 * 3!
}
```

Register:

```cmake
tax_add_test(test_taylor_expansion_accessors SOURCES core/test_taylor_expansion_accessors.cpp)
```

- [ ] **Step 9: Build and run**

```bash
cmake --build build -j$(nproc)
ctest --test-dir build -R "test_taylor_expansion_(ctor|accessors)" --output-on-failure
```

Expected: PASS.

- [ ] **Step 10: Create `testUtils.hpp`** (used by all subsequent slices)

Create `tests/testUtils.hpp`:

```cpp
#pragma once
#include <gtest/gtest.h>
#include <tax/tax.hpp>

namespace tax::test {
constexpr double kTol = 1e-10;

template <typename TE>
inline void ExpectCoeffsNear(const TE& actual, const TE& expected, double tol = kTol) {
    ASSERT_EQ(actual.nCoefficients, expected.nCoefficients);
    for (std::size_t k = 0; k < actual.nCoefficients; ++k) {
        EXPECT_NEAR(actual[k], expected[k], tol)
            << "Coefficient mismatch at flat index " << k;
    }
}
}  // namespace tax::test
```

- [ ] **Step 11: Commit**

```bash
git add include/tax tests/core tests/testUtils.hpp tests/CMakeLists.txt
git commit -m "slice2: dense TaylorExpansion — ctor, variable factories, coeff/derivative accessors"
```

---

## Task 3: Dense arithmetic — `+`, `-`, `*`, `/`, scalar variants (eager)

**Files:**
- Create: `include/tax/kernels/cauchy.hpp` (loop variant only — fast paths in Task 4)
- Create: `include/tax/operators/arithmetic.hpp`
- Create: `tests/operators/test_arith_dense.cpp`
- Create: `tests/kernels/test_cauchy_dense.cpp`
- Modify: `tests/CMakeLists.txt`
- Modify: `include/tax/tax.hpp`

- [ ] **Step 1: Write failing test for `+`, `-`, scalar variants**

Create `tests/operators/test_arith_dense.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../testUtils.hpp"

TEST(Arith, AddUni) {
    auto a = tax::TE<3>::variable(1.0);
    auto b = tax::TE<3>::variable(2.0);
    auto c = a + b;
    EXPECT_DOUBLE_EQ(c.value(), 3.0);
    EXPECT_DOUBLE_EQ(c[1], 2.0);  // both contribute +1
}

TEST(Arith, SubMulti) {
    typename tax::TE<2, 2>::Input p{1.0, 2.0};
    auto x = tax::TE<2, 2>::variable<0>(p);
    auto y = tax::TE<2, 2>::variable<1>(p);
    auto d = x - y;
    EXPECT_DOUBLE_EQ(d.value(), -1.0);
    EXPECT_DOUBLE_EQ((d.coeff<1, 0>()), 1.0);
    EXPECT_DOUBLE_EQ((d.coeff<0, 1>()), -1.0);
}

TEST(Arith, ScalarAddBothSides) {
    auto x = tax::TE<3>::variable(0.5);
    EXPECT_DOUBLE_EQ((x + 2.0).value(), 2.5);
    EXPECT_DOUBLE_EQ((2.0 + x).value(), 2.5);
    EXPECT_DOUBLE_EQ((x - 2.0).value(), -1.5);
    EXPECT_DOUBLE_EQ((2.0 - x).value(), 1.5);
}

TEST(Arith, ScalarMulBothSides) {
    auto x = tax::TE<3>::variable(0.5);
    auto m = 3.0 * x;
    EXPECT_DOUBLE_EQ(m.value(), 1.5);
    EXPECT_DOUBLE_EQ(m[1], 3.0);
}

TEST(Arith, UnaryNegate) {
    auto x = tax::TE<3>::variable(2.0);
    auto n = -x;
    EXPECT_DOUBLE_EQ(n.value(), -2.0);
    EXPECT_DOUBLE_EQ(n[1], -1.0);
}
```

Register:

```cmake
tax_add_test(test_arith_dense SOURCES operators/test_arith_dense.cpp)
```

- [ ] **Step 2: Implement `operators/arithmetic.hpp` (eager, no Cauchy yet)**

Create `include/tax/operators/arithmetic.hpp`:

```cpp
#pragma once

#include <tax/core/taylor_expansion.hpp>

namespace tax {

template <typename T, int N, int M>
[[nodiscard]] constexpr TaylorExpansion<T,N,M> operator+(
    const TaylorExpansion<T,N,M>& a, const TaylorExpansion<T,N,M>& b) noexcept {
    TaylorExpansion<T,N,M> r;
    for (std::size_t k = 0; k < a.nCoefficients; ++k) r[k] = a[k] + b[k];
    return r;
}

template <typename T, int N, int M>
[[nodiscard]] constexpr TaylorExpansion<T,N,M> operator-(
    const TaylorExpansion<T,N,M>& a, const TaylorExpansion<T,N,M>& b) noexcept {
    TaylorExpansion<T,N,M> r;
    for (std::size_t k = 0; k < a.nCoefficients; ++k) r[k] = a[k] - b[k];
    return r;
}

template <typename T, int N, int M>
[[nodiscard]] constexpr TaylorExpansion<T,N,M> operator-(
    const TaylorExpansion<T,N,M>& a) noexcept {
    TaylorExpansion<T,N,M> r;
    for (std::size_t k = 0; k < a.nCoefficients; ++k) r[k] = -a[k];
    return r;
}

template <typename T, int N, int M>
[[nodiscard]] constexpr TaylorExpansion<T,N,M> operator+(
    const TaylorExpansion<T,N,M>& a, T s) noexcept {
    TaylorExpansion<T,N,M> r = a; r[0] += s; return r;
}
template <typename T, int N, int M>
[[nodiscard]] constexpr TaylorExpansion<T,N,M> operator+(
    T s, const TaylorExpansion<T,N,M>& a) noexcept { return a + s; }

template <typename T, int N, int M>
[[nodiscard]] constexpr TaylorExpansion<T,N,M> operator-(
    const TaylorExpansion<T,N,M>& a, T s) noexcept {
    TaylorExpansion<T,N,M> r = a; r[0] -= s; return r;
}
template <typename T, int N, int M>
[[nodiscard]] constexpr TaylorExpansion<T,N,M> operator-(
    T s, const TaylorExpansion<T,N,M>& a) noexcept { return -a + s; }

template <typename T, int N, int M>
[[nodiscard]] constexpr TaylorExpansion<T,N,M> operator*(
    const TaylorExpansion<T,N,M>& a, T s) noexcept {
    TaylorExpansion<T,N,M> r;
    for (std::size_t k = 0; k < a.nCoefficients; ++k) r[k] = a[k] * s;
    return r;
}
template <typename T, int N, int M>
[[nodiscard]] constexpr TaylorExpansion<T,N,M> operator*(
    T s, const TaylorExpansion<T,N,M>& a) noexcept { return a * s; }

template <typename T, int N, int M>
[[nodiscard]] constexpr TaylorExpansion<T,N,M> operator/(
    const TaylorExpansion<T,N,M>& a, T s) noexcept { return a * (T(1)/s); }

// operator*(TE, TE) and operator/(TE, TE) are added below once the Cauchy
// kernel and reciprocal kernel exist.

}  // namespace tax
```

- [ ] **Step 3: Wire arithmetic into umbrella + build + run**

Edit `include/tax/tax.hpp` — append:

```cpp
#include <tax/operators/arithmetic.hpp>
```

```bash
cmake --build build -j$(nproc)
ctest --test-dir build -R test_arith_dense --output-on-failure
```

Expected: PASS (Mul of TE×TE not yet tested).

- [ ] **Step 4: Write failing test for Cauchy product `operator*(TE, TE)`**

Append to `tests/operators/test_arith_dense.cpp`:

```cpp
TEST(Arith, MulUniSquares) {
    auto x = tax::TE<4>::variable(0.0);
    auto y = x * x;
    // y(t) = t^2  →  coeffs [0, 0, 1, 0, 0]
    EXPECT_DOUBLE_EQ(y[0], 0.0);
    EXPECT_DOUBLE_EQ(y[2], 1.0);
    EXPECT_DOUBLE_EQ(y[4], 0.0);
}

TEST(Arith, MulMultiBilinear) {
    typename tax::TE<2, 2>::Input p{1.0, 2.0};
    auto x = tax::TE<2, 2>::variable<0>(p);
    auto y = tax::TE<2, 2>::variable<1>(p);
    auto z = x * y;
    // z(x,y) = (1 + dx)(2 + dy) = 2 + dy + 2*dx + dx*dy at order 2.
    EXPECT_DOUBLE_EQ(z.value(), 2.0);
    EXPECT_DOUBLE_EQ((z.coeff<1, 0>()), 2.0);
    EXPECT_DOUBLE_EQ((z.coeff<0, 1>()), 1.0);
    EXPECT_DOUBLE_EQ((z.coeff<1, 1>()), 1.0);
}
```

- [ ] **Step 5: Implement `kernels/cauchy.hpp` (loop variant + dispatch entry)**

Create `include/tax/kernels/cauchy.hpp` — port the math from `main:include/tax/kernels/cauchy.hpp`, adapted to use `tax::Coeffs<T,N,M>` directly. The single dispatch entry uses `#if TAX_USE_UNROLL` / `TAX_USE_STENCIL` toggles, but for now (Task 3) those `#if` blocks reference functions that don't exist; we wrap them with `__has_include` guards:

```cpp
#pragma once

#include <tax/core/multi_index.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/core/enumeration.hpp>

namespace tax::detail::kernels {

// Loop-based Cauchy product over graded-lex monomials.
template <typename T, int N, int M>
constexpr void cauchyProductLoop(Coeffs<T,N,M>& out,
                                 const Coeffs<T,N,M>& a,
                                 const Coeffs<T,N,M>& b) noexcept {
    out = {};
    tax::forEachMonomial<M, N>([&](const MultiIndex<M>& alpha) {
        const std::size_t i = flatIndex<M>(alpha);
        tax::forEachSubIndex<M>(alpha, [&](const MultiIndex<M>& k,
                                           const MultiIndex<M>& s) {
            out[i] += a[flatIndex<M>(k)] * b[flatIndex<M>(s)];
        });
    });
}

// Public dispatch entry. Fast paths added in Task 4.
template <typename T, int N, int M>
constexpr void cauchyProduct(Coeffs<T,N,M>& out,
                             const Coeffs<T,N,M>& a,
                             const Coeffs<T,N,M>& b) noexcept {
    cauchyProductLoop<T,N,M>(out, a, b);
}

}  // namespace tax::detail::kernels
```

- [ ] **Step 6: Add `operator*(TE, TE)` to `operators/arithmetic.hpp`**

Append to `include/tax/operators/arithmetic.hpp`:

```cpp
#include <tax/kernels/cauchy.hpp>

namespace tax {

template <typename T, int N, int M>
[[nodiscard]] constexpr TaylorExpansion<T,N,M> operator*(
    const TaylorExpansion<T,N,M>& a, const TaylorExpansion<T,N,M>& b) noexcept {
    TaylorExpansion<T,N,M> r;
    detail::kernels::cauchyProduct<T,N,M>(r.container().data, a.container().data, b.container().data);
    return r;
}

}  // namespace tax
```

- [ ] **Step 7: Build and run**

```bash
cmake --build build -j$(nproc)
ctest --test-dir build -R test_arith_dense --output-on-failure
```

Expected: PASS.

- [ ] **Step 8: Write direct kernel test for Cauchy product**

Create `tests/kernels/test_cauchy_dense.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <tax/kernels/cauchy.hpp>

TEST(CauchyDense, AgreesWithOperator) {
    auto x = tax::TE<5>::variable(0.3);
    auto y = tax::TE<5>::variable(0.7);
    auto z = x * y;
    tax::Coeffs<double, 5, 1> direct{};
    tax::detail::kernels::cauchyProduct<double, 5, 1>(direct, x.container().data, y.container().data);
    for (std::size_t k = 0; k < z.nCoefficients; ++k) EXPECT_DOUBLE_EQ(direct[k], z[k]);
}
```

Register:

```cmake
tax_add_test(test_cauchy_dense SOURCES kernels/test_cauchy_dense.cpp)
```

- [ ] **Step 9: Build, run all tests so far**

```bash
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

Expected: all PASS.

- [ ] **Step 10: Commit**

```bash
git add include/tax tests/operators tests/kernels tests/CMakeLists.txt
git commit -m "slice3: dense arithmetic — +, -, *, /, scalar variants, Cauchy loop kernel"
```

---

## Task 4: Cauchy fast paths — UNROLL (M=1) and STENCIL (M≥2)

**Files:**
- Create: `include/tax/kernels/cauchy_unroll.hpp`
- Create: `include/tax/kernels/cauchy_stencil.hpp`
- Create: `tests/kernels/test_cauchy_unroll_diff.cpp`
- Create: `tests/kernels/test_cauchy_stencil_diff.cpp`
- Modify: `include/tax/kernels/cauchy.hpp` (wire fast paths into dispatch entry)
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write failing diff-test for UNROLL (M=1 unrolled vs loop)**

Create `tests/kernels/test_cauchy_unroll_diff.cpp`:

```cpp
#include <gtest/gtest.h>
#include <tax/kernels/cauchy.hpp>
#include <tax/kernels/cauchy_unroll.hpp>

template <int N>
void runDiffUni(double tol = 1e-12) {
    tax::Coeffs<double, N, 1> a{}, b{}, out_loop{}, out_unroll{};
    for (int k = 0; k <= N; ++k) { a[k] = 0.3 + 0.1 * k; b[k] = -0.2 + 0.05 * k; }
    tax::detail::kernels::cauchyProductLoop<double, N, 1>(out_loop, a, b);
    tax::detail::kernels::cauchyProductUnroll<double, N, 1>(out_unroll, a, b);
    for (int k = 0; k <= N; ++k) EXPECT_NEAR(out_unroll[k], out_loop[k], tol)
        << "N=" << N << " k=" << k;
}

TEST(CauchyUnroll, DiffMatchesLoop_N3)  { runDiffUni<3>();  }
TEST(CauchyUnroll, DiffMatchesLoop_N5)  { runDiffUni<5>();  }
TEST(CauchyUnroll, DiffMatchesLoop_N10) { runDiffUni<10>(); }
```

Register:

```cmake
tax_add_test(test_cauchy_unroll_diff SOURCES kernels/test_cauchy_unroll_diff.cpp)
```

- [ ] **Step 2: Implement `kernels/cauchy_unroll.hpp`**

Create `include/tax/kernels/cauchy_unroll.hpp` — port from `main:include/tax/kernels/unroll.hpp`, namespace `tax::detail::kernels`, function name `cauchyProductUnroll`:

```cpp
#pragma once

#include <utility>
#include <tax/core/multi_index.hpp>

namespace tax::detail::kernels {

template <typename T, int N, std::size_t D, std::size_t... Ks>
constexpr T cauchyUniRow(const Coeffs<T, N, 1>& a, const Coeffs<T, N, 1>& b,
                         std::index_sequence<Ks...>) noexcept {
    return ((a[Ks] * b[D - Ks]) + ... + T{0});
}

template <typename T, int N, std::size_t... Ds>
constexpr void cauchyUniImpl(Coeffs<T, N, 1>& out, const Coeffs<T, N, 1>& a,
                             const Coeffs<T, N, 1>& b,
                             std::index_sequence<Ds...>) noexcept {
    ((out[Ds] = cauchyUniRow<T, N, Ds>(a, b, std::make_index_sequence<Ds + 1>{})), ...);
}

template <typename T, int N, int M>
constexpr void cauchyProductUnroll(Coeffs<T,N,M>& out,
                                   const Coeffs<T,N,M>& a,
                                   const Coeffs<T,N,M>& b) noexcept
    requires (M == 1)
{
    cauchyUniImpl<T, N>(out, a, b, std::make_index_sequence<std::size_t(N) + 1>{});
}

}  // namespace tax::detail::kernels
```

- [ ] **Step 3: Build and run diff test**

```bash
cmake --build build -j$(nproc)
ctest --test-dir build -R test_cauchy_unroll_diff --output-on-failure
```

Expected: PASS.

- [ ] **Step 4: Write failing diff-test for STENCIL (M≥2 stencil vs loop)**

Create `tests/kernels/test_cauchy_stencil_diff.cpp`:

```cpp
#include <gtest/gtest.h>
#include <tax/kernels/cauchy.hpp>
#include <tax/kernels/cauchy_stencil.hpp>

template <int N, int M>
void runDiffMulti(double tol = 1e-12) {
    tax::Coeffs<double, N, M> a{}, b{}, out_loop{}, out_sten{};
    for (std::size_t k = 0; k < a.size(); ++k) {
        a[k] = 0.1 * double(k + 1) - 0.4;
        b[k] = 0.2 - 0.05 * double(k);
    }
    tax::detail::kernels::cauchyProductLoop<double, N, M>(out_loop, a, b);
    tax::detail::kernels::cauchyProductStencil<double, N, M>(out_sten, a, b);
    for (std::size_t k = 0; k < a.size(); ++k)
        EXPECT_NEAR(out_sten[k], out_loop[k], tol) << "N=" << N << " M=" << M << " k=" << k;
}

TEST(CauchyStencil, Diff_N3_M2)  { runDiffMulti<3, 2>(); }
TEST(CauchyStencil, Diff_N4_M3)  { runDiffMulti<4, 3>(); }
TEST(CauchyStencil, Diff_N5_M4)  { runDiffMulti<5, 4>(); }
```

Register:

```cmake
tax_add_test(test_cauchy_stencil_diff SOURCES kernels/test_cauchy_stencil_diff.cpp)
```

- [ ] **Step 5: Implement `kernels/cauchy_stencil.hpp`**

Port from `main:include/tax/kernels/cauchy_stencil.hpp` into namespace `tax::detail::kernels`, function name `cauchyProductStencil`. Constraint: `requires (M >= 2)`. Adapt the stencil precomputation type so it consumes only `Coeffs<T,N,M>` (no `TaylorExpansionT` references) — the stencil is a `constexpr static` table of `(out_idx, a_idx, b_idx)` triples derived from `forEachMonomial`/`forEachSubIndex`.

- [ ] **Step 6: Build and run stencil diff test**

```bash
cmake --build build -j$(nproc)
ctest --test-dir build -R test_cauchy_stencil_diff --output-on-failure
```

Expected: PASS.

- [ ] **Step 7: Wire fast paths into dispatch entry**

Edit `include/tax/kernels/cauchy.hpp`, replace the dispatch entry with:

```cpp
#if TAX_USE_UNROLL
#  include <tax/kernels/cauchy_unroll.hpp>
#endif
#if TAX_USE_STENCIL
#  include <tax/kernels/cauchy_stencil.hpp>
#endif

namespace tax::detail::kernels {

template <typename T, int N, int M>
constexpr void cauchyProduct(Coeffs<T,N,M>& out,
                             const Coeffs<T,N,M>& a,
                             const Coeffs<T,N,M>& b) noexcept {
#if TAX_USE_UNROLL
    if constexpr (M == 1) { cauchyProductUnroll<T,N,M>(out, a, b); return; }
#endif
#if TAX_USE_STENCIL
    if constexpr (M >= 2) { cauchyProductStencil<T,N,M>(out, a, b); return; }
#endif
    cauchyProductLoop<T,N,M>(out, a, b);
}

}  // namespace tax::detail::kernels
```

- [ ] **Step 8: Re-run all tests so far**

```bash
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

Expected: all PASS.

- [ ] **Step 9: Run perf gate — Cauchy product**

Now build the benchmark target (after slice 5 we'll add operator benches; for now write the dense ops bench skeleton):

Create `benchmarks/ops_dense.cpp` (minimal — just Cauchy for now):

```cpp
#include <benchmark/benchmark.h>
#include <tax/tax.hpp>

template <int N>
static void BM_Mul_Uni(benchmark::State& s) {
    auto x = tax::TE<N>::variable(0.3);
    auto y = tax::TE<N>::variable(0.7);
    for (auto _ : s) {
        auto z = x * y;
        benchmark::DoNotOptimize(z);
    }
}
BENCHMARK(BM_Mul_Uni<5>);
BENCHMARK(BM_Mul_Uni<10>);
BENCHMARK(BM_Mul_Uni<15>);

template <int N, int M>
static void BM_Mul_Multi(benchmark::State& s) {
    typename tax::TE<N, M>::Input p{};
    for (int i = 0; i < M; ++i) p[i] = 0.1 * (i + 1);
    auto x = tax::TE<N, M>::template variable<0>(p);
    auto y = tax::TE<N, M>::template variable<1>(p);
    for (auto _ : s) {
        auto z = x * y;
        benchmark::DoNotOptimize(z);
    }
}
BENCHMARK(BM_Mul_Multi<4, 3>);
BENCHMARK(BM_Mul_Multi<6, 4>);

BENCHMARK_MAIN();
```

Add to `benchmarks/CMakeLists.txt`:

```cmake
tax_add_bench(bench_ops_dense SOURCES ops_dense.cpp)
```

```bash
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DTAX_BUILD_BENCHMARK=ON -DTAX_BUILD_TEST=OFF -G Ninja
cmake --build build-bench -j$(nproc)
./build-bench/benchmarks/bench_ops_dense --benchmark_min_time=0.5s > /tmp/stage1-cauchy.txt
diff <(grep BM_Mul_ benchmarks/baseline/main-*-multivariate.txt | head) /tmp/stage1-cauchy.txt | head -40
```

Compare manually. If any case is >5% slower than the baseline, investigate before continuing.

- [ ] **Step 10: Commit**

```bash
git add include/tax tests/kernels tests/CMakeLists.txt benchmarks/ops_dense.cpp benchmarks/CMakeLists.txt
git commit -m "slice4: Cauchy fast paths — UNROLL (M=1) and STENCIL (M>=2) with diff tests"
```

---

## Task 5: Algebra kernels — `seriesSquare`, `seriesCube`

**Files:**
- Create: `include/tax/kernels/algebra.hpp` (initial: square + cube + cauchySelfProduct)
- Create: `include/tax/operators/math_unary.hpp` (initial: square, cube wrappers)
- Create: `tests/operators/test_algebra_square_cube.cpp`
- Modify: `tests/CMakeLists.txt`, `include/tax/tax.hpp`

- [ ] **Step 1: Failing test**

Create `tests/operators/test_algebra_square_cube.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../testUtils.hpp"

TEST(Square, MatchesMul) {
    auto x = tax::TE<5>::variable(0.5);
    auto a = tax::square(x);
    auto b = x * x;
    tax::test::ExpectCoeffsNear(a, b);
}

TEST(Cube, MatchesMul) {
    typename tax::TE<4, 2>::Input p{0.3, -0.2};
    auto x = tax::TE<4, 2>::variable<0>(p);
    auto a = tax::cube(x);
    auto b = x * x * x;
    tax::test::ExpectCoeffsNear(a, b);
}
```

Register:

```cmake
tax_add_test(test_algebra_square_cube SOURCES operators/test_algebra_square_cube.cpp)
```

- [ ] **Step 2: Implement `kernels/algebra.hpp` (square + cube only for this task)**

Port `cauchySelfProduct`, `seriesSquare`, `seriesCube` from `main:include/tax/kernels/cauchy.hpp` and `main:include/tax/kernels/algebra.hpp`. Namespace `tax::detail::kernels`. Signatures take `Coeffs<T,N,M>&`.

For Task 5, only implement: `cauchySelfProduct`, `seriesSquare`, `seriesCube`. Other algebra ops are in Task 6/7.

- [ ] **Step 3: Implement `operators/math_unary.hpp` wrappers**

Create `include/tax/operators/math_unary.hpp`:

```cpp
#pragma once

#include <tax/core/taylor_expansion.hpp>
#include <tax/kernels/algebra.hpp>

namespace tax {

template <typename T, int N, int M>
[[nodiscard]] constexpr TaylorExpansion<T,N,M> square(const TaylorExpansion<T,N,M>& x) noexcept {
    TaylorExpansion<T,N,M> r;
    detail::kernels::seriesSquare<T,N,M>(r.container().data, x.container().data);
    return r;
}

template <typename T, int N, int M>
[[nodiscard]] constexpr TaylorExpansion<T,N,M> cube(const TaylorExpansion<T,N,M>& x) noexcept {
    TaylorExpansion<T,N,M> r;
    detail::kernels::seriesCube<T,N,M>(r.container().data, x.container().data);
    return r;
}

}  // namespace tax
```

- [ ] **Step 4: Wire into umbrella, build, run**

Edit `include/tax/tax.hpp` — append `#include <tax/operators/math_unary.hpp>`.

```bash
cmake --build build -j$(nproc)
ctest --test-dir build -R test_algebra_square_cube --output-on-failure
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add include/tax tests/operators tests/CMakeLists.txt
git commit -m "slice5a: algebra kernels — square, cube"
```

---

## Task 6: Algebra kernels — `seriesSqrt`, `seriesCbrt`, `seriesReciprocal`, `operator/(TE, TE)`

**Files:**
- Modify: `include/tax/kernels/algebra.hpp` (add sqrt, cbrt, reciprocal)
- Modify: `include/tax/operators/math_unary.hpp` (add wrappers)
- Modify: `include/tax/operators/arithmetic.hpp` (add `operator/(TE, TE)` via reciprocal)
- Create: `tests/operators/test_algebra_inverse.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Failing test**

Create `tests/operators/test_algebra_inverse.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <cmath>

TEST(Sqrt, RoundTripWithSquare) {
    auto x = tax::TE<5>::variable(4.0);
    auto s = tax::sqrt(x);
    auto r = s * s;
    tax::test::ExpectCoeffsNear(r, x, 1e-12);
}

TEST(Cbrt, RoundTripWithCube) {
    auto x = tax::TE<5>::variable(2.0);
    auto c = tax::cbrt(x);
    auto r = c * c * c;
    tax::test::ExpectCoeffsNear(r, x, 1e-12);
}

TEST(Reciprocal, MultipliesToOne) {
    auto x = tax::TE<5>::variable(2.0);
    auto i = tax::reciprocal(x);
    auto p = x * i;
    EXPECT_NEAR(p.value(), 1.0, 1e-12);
    for (std::size_t k = 1; k < p.nCoefficients; ++k) EXPECT_NEAR(p[k], 0.0, 1e-12);
}

TEST(Divide, MatchesReciprocal) {
    auto x = tax::TE<5>::variable(3.0);
    auto y = tax::TE<5>::variable(2.0);
    auto a = x / y;
    auto b = x * tax::reciprocal(y);
    tax::test::ExpectCoeffsNear(a, b, 1e-12);
}
```

Register:

```cmake
tax_add_test(test_algebra_inverse SOURCES operators/test_algebra_inverse.cpp)
```

- [ ] **Step 2: Port `seriesSqrt`, `seriesCbrt`, `seriesReciprocal` into `kernels/algebra.hpp`**

Source: `main:include/tax/kernels/algebra.hpp`. Keep the same recurrence math. Drop the `Dynamic`-shape overloads. Use `Coeffs<T,N,M>` signatures.

- [ ] **Step 3: Add wrappers in `operators/math_unary.hpp`**

Append:

```cpp
template <typename T, int N, int M>
[[nodiscard]] TaylorExpansion<T,N,M> sqrt(const TaylorExpansion<T,N,M>& x) noexcept {
    TaylorExpansion<T,N,M> r;
    detail::kernels::seriesSqrt<T,N,M>(r.container().data, x.container().data);
    return r;
}

template <typename T, int N, int M>
[[nodiscard]] TaylorExpansion<T,N,M> cbrt(const TaylorExpansion<T,N,M>& x) noexcept {
    TaylorExpansion<T,N,M> r;
    detail::kernels::seriesCbrt<T,N,M>(r.container().data, x.container().data);
    return r;
}

template <typename T, int N, int M>
[[nodiscard]] TaylorExpansion<T,N,M> reciprocal(const TaylorExpansion<T,N,M>& x) noexcept {
    TaylorExpansion<T,N,M> r;
    detail::kernels::seriesReciprocal<T,N,M>(r.container().data, x.container().data);
    return r;
}
```

- [ ] **Step 4: Add `operator/(TE, TE)` in `operators/arithmetic.hpp`**

Append:

```cpp
template <typename T, int N, int M>
[[nodiscard]] TaylorExpansion<T,N,M> operator/(
    const TaylorExpansion<T,N,M>& a, const TaylorExpansion<T,N,M>& b) noexcept {
    return a * tax::reciprocal(b);
}
```

Note: requires `<tax/operators/math_unary.hpp>` included after `arithmetic.hpp` in `tax.hpp`, OR forward-declare `reciprocal`. Use forward declaration to avoid circular include.

- [ ] **Step 5: Build, run, commit**

```bash
cmake --build build -j$(nproc)
ctest --test-dir build -R test_algebra_inverse --output-on-failure
```

Expected: PASS.

```bash
git add include/tax tests/operators tests/CMakeLists.txt
git commit -m "slice5b: algebra kernels — sqrt, cbrt, reciprocal, operator/(TE,TE)"
```

---

## Task 7: Algebra kernels — `seriesPow`, `pow(TE, int)`, `pow(TE, T)`

**Files:**
- Modify: `include/tax/kernels/algebra.hpp` (add `seriesPow`)
- Create: `include/tax/operators/math_binary.hpp` (add `pow`)
- Create: `tests/operators/test_pow.cpp`
- Modify: `tests/CMakeLists.txt`, `include/tax/tax.hpp`

- [ ] **Step 1: Failing test**

Create `tests/operators/test_pow.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../testUtils.hpp"

TEST(Pow, IntegerExponent) {
    auto x = tax::TE<6>::variable(2.0);
    auto a = tax::pow(x, 3);
    auto b = x * x * x;
    tax::test::ExpectCoeffsNear(a, b, 1e-12);
}

TEST(Pow, RealExponent) {
    auto x = tax::TE<5>::variable(4.0);
    auto a = tax::pow(x, 0.5);   // sqrt
    auto b = tax::sqrt(x);
    tax::test::ExpectCoeffsNear(a, b, 1e-12);
}

TEST(Pow, NegativeRealExponent) {
    auto x = tax::TE<5>::variable(2.0);
    auto a = tax::pow(x, -1.0);
    auto b = tax::reciprocal(x);
    tax::test::ExpectCoeffsNear(a, b, 1e-12);
}
```

Register `tax_add_test(test_pow SOURCES operators/test_pow.cpp)`.

- [ ] **Step 2: Port `seriesPow` from `main:include/tax/kernels/algebra.hpp`**

Add to `kernels/algebra.hpp`. Two variants — integer exponent (via binary exponentiation atop `cauchyProduct`) and real exponent (via standard `pow(c+ε)^p` recurrence).

- [ ] **Step 3: Implement `operators/math_binary.hpp`**

Create:

```cpp
#pragma once

#include <tax/core/taylor_expansion.hpp>
#include <tax/kernels/algebra.hpp>

namespace tax {

template <typename T, int N, int M>
[[nodiscard]] TaylorExpansion<T,N,M> pow(const TaylorExpansion<T,N,M>& x, int n) noexcept {
    TaylorExpansion<T,N,M> r;
    detail::kernels::seriesPowInt<T,N,M>(r.container().data, x.container().data, n);
    return r;
}

template <typename T, int N, int M>
[[nodiscard]] TaylorExpansion<T,N,M> pow(const TaylorExpansion<T,N,M>& x, T p) noexcept {
    TaylorExpansion<T,N,M> r;
    detail::kernels::seriesPow<T,N,M>(r.container().data, x.container().data, p);
    return r;
}

}  // namespace tax
```

- [ ] **Step 4: Wire into umbrella**

Append to `include/tax/tax.hpp`: `#include <tax/operators/math_binary.hpp>`.

- [ ] **Step 5: Build, run, commit**

```bash
cmake --build build -j$(nproc)
ctest --test-dir build -R test_pow --output-on-failure
git add include/tax tests/operators tests/CMakeLists.txt
git commit -m "slice5c: algebra kernels — seriesPow (integer + real), pow operator"
```

---

## Task 8: Transcendental kernels — `exp`, `log`

**Files:**
- Create: `include/tax/kernels/transcendental.hpp` (initial: exp, log)
- Modify: `include/tax/operators/math_unary.hpp` (add `exp`, `log`)
- Create: `tests/operators/test_exp_log.cpp`

- [ ] **Step 1: Failing test**

Create `tests/operators/test_exp_log.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <cmath>

TEST(Exp, AtZero) {
    auto x = tax::TE<5>::variable(0.0);
    auto e = tax::exp(x);
    // exp(x) at 0: coeffs [1, 1, 1/2, 1/6, 1/24, 1/120]
    EXPECT_NEAR(e[0], 1.0, 1e-15);
    EXPECT_NEAR(e[1], 1.0, 1e-15);
    EXPECT_NEAR(e[2], 0.5, 1e-15);
    EXPECT_NEAR(e[3], 1.0/6.0, 1e-15);
    EXPECT_NEAR(e[4], 1.0/24.0, 1e-15);
    EXPECT_NEAR(e[5], 1.0/120.0, 1e-15);
}

TEST(Log, RoundTripWithExp) {
    auto x = tax::TE<5>::variable(0.7);
    tax::test::ExpectCoeffsNear(tax::log(tax::exp(x)), x, 1e-12);
}

TEST(Log, ConstantOne) {
    auto one = tax::TE<5>::constant(1.0);
    auto l = tax::log(one);
    EXPECT_NEAR(l.value(), 0.0, 1e-15);
}
```

Register `tax_add_test(test_exp_log SOURCES operators/test_exp_log.cpp)`.

- [ ] **Step 2: Port `seriesExp`, `seriesLog` from `main:include/tax/kernels/transcendental.hpp`**

Strip `Dynamic`-shape branches, lift to `tax::detail::kernels::`, take `Coeffs<T,N,M>` signatures.

- [ ] **Step 3: Add wrappers**

Append to `include/tax/operators/math_unary.hpp`:

```cpp
template <typename T, int N, int M>
[[nodiscard]] TaylorExpansion<T,N,M> exp(const TaylorExpansion<T,N,M>& x) noexcept {
    TaylorExpansion<T,N,M> r;
    detail::kernels::seriesExp<T,N,M>(r.container().data, x.container().data);
    return r;
}

template <typename T, int N, int M>
[[nodiscard]] TaylorExpansion<T,N,M> log(const TaylorExpansion<T,N,M>& x) noexcept {
    TaylorExpansion<T,N,M> r;
    detail::kernels::seriesLog<T,N,M>(r.container().data, x.container().data);
    return r;
}
```

Wire `transcendental.hpp` into `tax.hpp` via `math_unary.hpp` chain.

- [ ] **Step 4: Build, run, commit**

```bash
cmake --build build -j$(nproc)
ctest --test-dir build -R test_exp_log --output-on-failure
git add include/tax tests/operators tests/CMakeLists.txt
git commit -m "slice6a: transcendentals — exp, log"
```

---

## Task 9: Transcendentals — hyperbolic + inverses + erf

**Files:**
- Modify: `include/tax/kernels/transcendental.hpp` (add sinh, cosh, tanh, asinh, acosh, atanh, erf)
- Modify: `include/tax/operators/math_unary.hpp`
- Create: `tests/operators/test_hyperbolic.cpp`, `tests/operators/test_erf.cpp`

- [ ] **Step 1: Failing tests**

Create `tests/operators/test_hyperbolic.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <cmath>

TEST(Sinh, IdentityWithCosh) {
    auto x = tax::TE<5>::variable(0.3);
    auto s = tax::sinh(x);
    auto c = tax::cosh(x);
    auto sum = c*c - s*s;          // cosh^2 - sinh^2 = 1
    EXPECT_NEAR(sum.value(), 1.0, 1e-12);
    for (std::size_t k = 1; k < sum.nCoefficients; ++k) EXPECT_NEAR(sum[k], 0.0, 1e-12);
}

TEST(Tanh, RatioSinhOverCosh) {
    auto x = tax::TE<5>::variable(0.4);
    tax::test::ExpectCoeffsNear(tax::tanh(x), tax::sinh(x) / tax::cosh(x), 1e-12);
}

TEST(Asinh, RoundTrip) {
    auto x = tax::TE<5>::variable(0.5);
    tax::test::ExpectCoeffsNear(tax::asinh(tax::sinh(x)), x, 1e-10);
}

TEST(Acosh, RoundTrip) {
    auto x = tax::TE<5>::variable(2.0);
    tax::test::ExpectCoeffsNear(tax::acosh(tax::cosh(x)), x, 1e-10);
}

TEST(Atanh, RoundTrip) {
    auto x = tax::TE<5>::variable(0.4);
    tax::test::ExpectCoeffsNear(tax::atanh(tax::tanh(x)), x, 1e-10);
}
```

Create `tests/operators/test_erf.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <cmath>

TEST(Erf, AtZero) {
    auto x = tax::TE<3>::variable(0.0);
    auto e = tax::erf(x);
    EXPECT_NEAR(e[0], 0.0, 1e-15);
    EXPECT_NEAR(e[1], 2.0/std::sqrt(M_PI), 1e-12);
}
```

Register both tests.

- [ ] **Step 2: Port `seriesSinh`, `seriesCosh`, `seriesTanh`, `seriesAsinh`, `seriesAcosh`, `seriesAtanh`, `seriesErf` from `main:include/tax/kernels/transcendental.hpp`**

Same pattern. Add the corresponding `operators/math_unary.hpp` wrappers.

- [ ] **Step 3: Build, run, commit**

```bash
cmake --build build -j$(nproc)
ctest --test-dir build -R "test_hyperbolic|test_erf" --output-on-failure
git add include/tax tests/operators tests/CMakeLists.txt
git commit -m "slice6b: transcendentals — sinh, cosh, tanh + inverses, erf"
```

---

## Task 10: Trigonometric — `sin`, `cos`, `tan`

**Files:**
- Create: `include/tax/kernels/trigonometric.hpp` (sin, cos, tan)
- Modify: `include/tax/operators/math_unary.hpp`
- Create: `tests/operators/test_trig.cpp`

- [ ] **Step 1: Failing test**

Create `tests/operators/test_trig.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <cmath>

TEST(Sin, AtZero) {
    auto x = tax::TE<5>::variable(0.0);
    auto s = tax::sin(x);
    // sin(t) at 0: [0, 1, 0, -1/6, 0, 1/120]
    EXPECT_NEAR(s[0], 0.0, 1e-15);
    EXPECT_NEAR(s[1], 1.0, 1e-15);
    EXPECT_NEAR(s[3], -1.0/6.0, 1e-15);
    EXPECT_NEAR(s[5], 1.0/120.0, 1e-15);
}

TEST(SinCos, PythagoreanIdentity) {
    auto x = tax::TE<5>::variable(0.7);
    auto p = tax::sin(x) * tax::sin(x) + tax::cos(x) * tax::cos(x);
    EXPECT_NEAR(p.value(), 1.0, 1e-12);
    for (std::size_t k = 1; k < p.nCoefficients; ++k) EXPECT_NEAR(p[k], 0.0, 1e-12);
}

TEST(Tan, RatioSinCos) {
    auto x = tax::TE<5>::variable(0.5);
    tax::test::ExpectCoeffsNear(tax::tan(x), tax::sin(x) / tax::cos(x), 1e-12);
}
```

Register.

- [ ] **Step 2: Port `seriesSin`, `seriesCos`, `seriesTan` from `main:include/tax/kernels/trigonometric.hpp`**

- [ ] **Step 3: Add `operators/math_unary.hpp` wrappers**

- [ ] **Step 4: Build, run, commit**

```bash
cmake --build build -j$(nproc)
ctest --test-dir build -R test_trig --output-on-failure
git add include/tax tests/operators tests/CMakeLists.txt
git commit -m "slice7a: trigonometric — sin, cos, tan"
```

---

## Task 11: Inverse trig — `asin`, `acos`, `atan`, `atan2`

**Files:**
- Modify: `include/tax/kernels/trigonometric.hpp` (asin, acos, atan, atan2)
- Modify: `include/tax/operators/math_unary.hpp` (asin, acos, atan)
- Modify: `include/tax/operators/math_binary.hpp` (atan2)
- Create: `tests/operators/test_inverse_trig.cpp`

- [ ] **Step 1: Failing test**

Create `tests/operators/test_inverse_trig.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <cmath>

TEST(Asin, RoundTrip) {
    auto x = tax::TE<5>::variable(0.3);
    tax::test::ExpectCoeffsNear(tax::asin(tax::sin(x)), x, 1e-10);
}

TEST(Acos, RoundTrip) {
    auto x = tax::TE<5>::variable(0.3);
    tax::test::ExpectCoeffsNear(tax::acos(tax::cos(x)), x, 1e-10);
}

TEST(Atan, RoundTrip) {
    auto x = tax::TE<5>::variable(0.4);
    tax::test::ExpectCoeffsNear(tax::atan(tax::tan(x)), x, 1e-10);
}

TEST(Atan2, ConsistentWithAtan) {
    auto x = tax::TE<5>::variable(0.6);
    auto y = tax::TE<5>::variable(0.8);
    auto a = tax::atan2(y, x);
    auto b = tax::atan(y / x);
    tax::test::ExpectCoeffsNear(a, b, 1e-12);
}
```

Register.

- [ ] **Step 2: Port `seriesAsin`, `seriesAcos`, `seriesAtan`, `seriesAtan2` from `main:include/tax/kernels/trigonometric.hpp`**

- [ ] **Step 3: Add wrappers** in `math_unary.hpp` (asin, acos, atan) and `math_binary.hpp` (atan2)

- [ ] **Step 4: Build, run, commit**

```bash
cmake --build build -j$(nproc)
ctest --test-dir build -R test_inverse_trig --output-on-failure
git add include/tax tests/operators tests/CMakeLists.txt
git commit -m "slice7b: inverse trig — asin, acos, atan, atan2"
```

### Perf gate checkpoint

After slices 5-7, re-run `bench_ops_dense` and compare to `benchmarks/baseline/main-*-vs-dace.txt` per-operator. Any operator >5% slower must be investigated before continuing.

---

## Task 12: Symbolic differentiation/integration — `deriv`, `integ`

**Files:**
- Modify: `include/tax/core/taylor_expansion.hpp` (add methods)
- Create: `tests/core/test_deriv_integ.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Failing test**

Create `tests/core/test_deriv_integ.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../testUtils.hpp"

TEST(Deriv, UnivariateMatchesDx) {
    // f = sin(x), df/dx = cos(x)
    auto x = tax::TE<5>::variable(0.3);
    auto f = tax::sin(x);
    auto df = f.deriv<0>();
    auto expected = tax::cos(x);
    // Compare up to order N-1.
    for (std::size_t k = 0; k + 1 < f.nCoefficients; ++k) EXPECT_NEAR(df[k], expected[k], 1e-12);
}

TEST(Integ, RoundTripWithDeriv) {
    // d/dx (integ f) = f
    auto x = tax::TE<5>::variable(0.0);
    auto f = tax::cos(x);
    auto F = f.integ<0>();
    auto dF = F.deriv<0>();
    for (std::size_t k = 0; k + 1 < f.nCoefficients; ++k) EXPECT_NEAR(dF[k], f[k], 1e-12);
}

TEST(Deriv, MultivariatePartial) {
    // f = x*y, df/dx = y
    typename tax::TE<3, 2>::Input p{1.0, 2.0};
    auto x = tax::TE<3, 2>::variable<0>(p);
    auto y = tax::TE<3, 2>::variable<1>(p);
    auto f = x * y;
    auto df_dx = f.deriv<0>();
    tax::test::ExpectCoeffsNear(df_dx, y, 1e-12);
}
```

Register `tax_add_test(test_deriv_integ SOURCES core/test_deriv_integ.cpp)`.

- [ ] **Step 2: Implement `deriv<I>`, `integ<I>`, runtime variants**

Port from `main:include/tax/storage/tte_static.hpp` (search for `deriv` and `integ`). Adapt: methods on the new `TaylorExpansion`, no `Dynamic` branches. Static_assert `I >= 0 && I < M`.

- [ ] **Step 3: Build, run, commit**

```bash
cmake --build build -j$(nproc)
ctest --test-dir build -R test_deriv_integ --output-on-failure
git add include/tax tests/core tests/CMakeLists.txt
git commit -m "slice8: symbolic differentiation and integration methods"
```

---

## Task 13: Sparse storage + dense↔sparse conversion

**Files:**
- Create: `include/tax/core/storage/sparse.hpp`
- Modify: `include/tax/core/taylor_expansion.hpp` (add `storage::Sparse` specialization)
- Modify: `include/tax/core/concepts.hpp` (add `SparsePolynomial`)
- Modify: `include/tax/tax.hpp`
- Create: `tests/sparse/test_sparse_ctor.cpp`
- Create: `tests/sparse/test_sparse_conversion.cpp`

- [ ] **Step 1: Failing test (construction)**

Create `tests/sparse/test_sparse_ctor.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../testUtils.hpp"

TEST(SparseCtor, ZeroIsEmpty) {
    tax::STE<3> z;
    EXPECT_EQ(z.nnz(), 0u);
    EXPECT_EQ(z.value(), 0.0);
}

TEST(SparseCtor, ConstantHasOneNonzero) {
    tax::STE<3> c{2.5};
    EXPECT_EQ(c.nnz(), 1u);
    EXPECT_EQ(c.value(), 2.5);
}

TEST(SparseCtor, VariableHasTwoNonzeros) {
    auto x = tax::STE<3>::variable(1.5);
    EXPECT_EQ(x.nnz(), 2u);
    EXPECT_EQ(x.value(), 1.5);
    EXPECT_EQ(x.coeff(tax::MultiIndex<1>{1}), 1.0);
}
```

Register.

- [ ] **Step 2: Implement `core/storage/sparse.hpp`**

Create `include/tax/core/storage/sparse.hpp` — port from `main:include/tax/storage/sparse_tte.hpp`. Hoist into the storage policy:

```cpp
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <tax/core/multi_index.hpp>

namespace tax::storage {

using flat_index_t = std::uint32_t;

struct Sparse {};

template <typename T, int N, int M>
class SparseContainer {
public:
    using value_type    = T;
    static constexpr std::size_t nCoefficientsMax = numMonomials(N, M);  // dense-equivalent size

    constexpr SparseContainer() = default;

    [[nodiscard]] std::size_t nnz() const noexcept { return idx_.size(); }
    [[nodiscard]] T value() const noexcept {
        return (!idx_.empty() && idx_.front() == 0) ? val_.front() : T{0};
    }

    [[nodiscard]] std::span<const flat_index_t> support() const noexcept {
        return {idx_.data(), idx_.size()};
    }
    [[nodiscard]] std::span<const T> values() const noexcept {
        return {val_.data(), val_.size()};
    }

    T coeffAtFlat(std::size_t k) const noexcept {
        auto it = std::lower_bound(idx_.begin(), idx_.end(), flat_index_t(k));
        if (it == idx_.end() || *it != flat_index_t(k)) return T{0};
        return val_[std::size_t(it - idx_.begin())];
    }

    void set(std::size_t k, T v) noexcept;            // port from main
    void accumulate(std::size_t k, T v) noexcept;     // port from main

    template <typename Fn>
    void forEachNonzero(Fn&& fn) const noexcept {
        for (std::size_t i = 0; i < idx_.size(); ++i) fn(std::size_t(idx_[i]), val_[i]);
    }

    template <typename Fn>
    void forEachPair(const SparseContainer& other, Fn&& fn) const noexcept;  // port merge walk

    std::vector<flat_index_t>& rawIndices() noexcept { return idx_; }
    std::vector<T>&            rawValues()  noexcept { return val_; }

private:
    std::vector<flat_index_t> idx_;
    std::vector<T>            val_;
};

}  // namespace tax::storage
```

Fill in `set`, `accumulate`, `forEachPair` by porting from the existing sparse_tte.

- [ ] **Step 3: Add `TaylorExpansion<T, N, M, storage::Sparse>` specialization**

Append to `include/tax/core/taylor_expansion.hpp`:

```cpp
namespace tax {

template <typename T, int N, int M>
class TaylorExpansion<T, N, M, storage::Sparse> {
public:
    static_assert(N >= 0); static_assert(M >= 1);
    using scalar_type = T;
    using container_t = storage::SparseContainer<T, N, M>;
    using Input       = std::array<T, std::size_t(M)>;
    using Dense       = TaylorExpansion<T, N, M, storage::Dense>;

    static constexpr int order_v = N;
    static constexpr int vars_v  = M;
    static constexpr std::size_t nCoefficients = numMonomials(N, M);  // dense-size upper bound

    constexpr TaylorExpansion() = default;
    /*implicit*/ TaylorExpansion(T c);             // store one nonzero at idx 0 if c != 0
    explicit TaylorExpansion(const Dense& d);      // drop exact-zero entries

    [[nodiscard]] static TaylorExpansion zero() noexcept { return {}; }
    [[nodiscard]] static TaylorExpansion constant(T c)   { return TaylorExpansion{c}; }

    [[nodiscard]] static TaylorExpansion variable(T x0) noexcept requires (M == 1);

    template <int I>
    [[nodiscard]] static TaylorExpansion variable(const Input& p) noexcept
        requires (M >= 1 && I >= 0 && I < M);

    [[nodiscard]] std::size_t nnz() const noexcept { return c_.nnz(); }
    [[nodiscard]] T value() const noexcept { return c_.value(); }
    [[nodiscard]] T coeff(const MultiIndex<M>& alpha) const noexcept {
        return c_.coeffAtFlat(flatIndex<M>(alpha));
    }
    template <int... Alpha> [[nodiscard]] T coeff() const noexcept;
    [[nodiscard]] T derivative(const MultiIndex<M>& alpha) const noexcept;
    template <int... Alpha> [[nodiscard]] T derivative() const noexcept;

    [[nodiscard]] std::span<const storage::flat_index_t> support() const noexcept { return c_.support(); }
    [[nodiscard]] std::span<const T> values() const noexcept { return c_.values(); }
    [[nodiscard]] Dense dense() const noexcept;       // explicit conversion

    [[nodiscard]] const container_t& container() const noexcept { return c_; }
    [[nodiscard]]       container_t& container()       noexcept { return c_; }

private:
    container_t c_{};
};

template <int N, int M = 1>
using STE = TaylorExpansion<double, N, M, storage::Sparse>;

template <typename T, int N, int M>
[[nodiscard]] TaylorExpansion<T,N,M,storage::Sparse>
    sparse(const TaylorExpansion<T,N,M,storage::Dense>& d) noexcept
{ return TaylorExpansion<T,N,M,storage::Sparse>(d); }

}  // namespace tax
```

Implement `dense()`, the `(const Dense&)` constructor, `variable`, `coeff<Alpha...>`, `derivative` — all port from main:sparse_tte.

- [ ] **Step 4: Add `SparsePolynomial` concept**

Append to `include/tax/core/concepts.hpp`:

```cpp
template <typename P>
concept SparsePolynomial = TaylorPolynomial<P> && requires(const P& p) {
    { p.nnz()     } -> std::convertible_to<std::size_t>;
    { p.support() };
    { p.values()  };
};
```

- [ ] **Step 5: Wire and build**

Append to `include/tax/tax.hpp`: `#include <tax/core/storage/sparse.hpp>`.

```bash
cmake --build build -j$(nproc)
ctest --test-dir build -R test_sparse_ctor --output-on-failure
```

Expected: PASS.

- [ ] **Step 6: Failing test (conversion)**

Create `tests/sparse/test_sparse_conversion.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../testUtils.hpp"

TEST(SparseConv, DenseToSparseToDenseRoundTrip) {
    auto d = tax::TE<3, 2>::constant(2.0);
    typename tax::TE<3, 2>::Input p{1.0, 2.0};
    auto x = tax::TE<3, 2>::variable<0>(p);
    auto f = d + x;                            // 2 nonzeros expected
    auto s = tax::sparse(f);
    EXPECT_GE(s.nnz(), 1u);
    auto back = s.dense();
    tax::test::ExpectCoeffsNear(back, f);
}

TEST(SparseConv, DropExactZeros) {
    auto d = tax::TE<3>::zero();
    d[0] = 1.0;
    d[2] = 3.0;                                // 1 + 3*x^2, two nonzeros
    auto s = tax::sparse(d);
    EXPECT_EQ(s.nnz(), 2u);
}
```

Register, build, commit.

- [ ] **Step 7: Commit**

```bash
git add include/tax tests/sparse tests/CMakeLists.txt
git commit -m "slice9: sparse storage policy + sparse <-> dense conversion"
```

---

## Task 14: Sparse arithmetic — `+`, `-`, scalar variants

**Files:**
- Modify: `include/tax/operators/arithmetic.hpp` (add sparse overloads)
- Create: `tests/sparse/test_sparse_arith.cpp`

- [ ] **Step 1: Failing test**

Create `tests/sparse/test_sparse_arith.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../testUtils.hpp"

TEST(SparseArith, AddMatchesDense) {
    auto x = tax::TE<5>::variable(0.5);
    auto y = tax::TE<5>::variable(0.3);
    auto sx = tax::sparse(x);
    auto sy = tax::sparse(y);
    auto sum = sx + sy;
    tax::test::ExpectCoeffsNear(sum.dense(), x + y);
}

TEST(SparseArith, SubMatchesDense) {
    auto x = tax::TE<5>::variable(0.5);
    auto y = tax::TE<5>::variable(0.3);
    auto d = tax::sparse(x) - tax::sparse(y);
    tax::test::ExpectCoeffsNear(d.dense(), x - y);
}

TEST(SparseArith, ScalarMul) {
    auto sx = tax::sparse(tax::TE<5>::variable(0.5));
    auto m = 3.0 * sx;
    EXPECT_NEAR(m.value(), 1.5, 1e-12);
    EXPECT_EQ(m.nnz(), sx.nnz());
}
```

Register.

- [ ] **Step 2: Implement sparse arithmetic overloads**

Append to `include/tax/operators/arithmetic.hpp`. Port the sorted-merge add/sub from `main:include/tax/storage/sparse_tte.hpp` into free functions:

```cpp
template <typename T, int N, int M>
[[nodiscard]] TaylorExpansion<T,N,M,storage::Sparse> operator+(
    const TaylorExpansion<T,N,M,storage::Sparse>& a,
    const TaylorExpansion<T,N,M,storage::Sparse>& b) noexcept;
// + variants for - and scalar combinations.
```

The body walks `a.support()` and `b.support()` simultaneously, emitting indices/values into the result via `accumulate`. Maintain sorted order.

- [ ] **Step 3: Build, run, commit**

```bash
cmake --build build -j$(nproc)
ctest --test-dir build -R test_sparse_arith --output-on-failure
git add include/tax tests/sparse tests/CMakeLists.txt
git commit -m "slice10a: sparse arithmetic — sorted-merge +, -, scalar variants"
```

---

## Task 15: Sparse kernels — Cauchy product, self-product, accumulate

**Files:**
- Create: `include/tax/kernels/sparse_cauchy.hpp`
- Modify: `include/tax/operators/arithmetic.hpp` (add sparse `operator*`)
- Create: `tests/kernels/test_sparse_cauchy.cpp`

- [ ] **Step 1: Failing test**

Create `tests/kernels/test_sparse_cauchy.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../testUtils.hpp"

TEST(SparseCauchy, MulMatchesDense) {
    typename tax::TE<4, 3>::Input p{0.3, -0.2, 0.5};
    auto x = tax::TE<4, 3>::variable<0>(p);
    auto y = tax::TE<4, 3>::variable<1>(p);
    auto sxy = tax::sparse(x) * tax::sparse(y);
    auto dxy = x * y;
    tax::test::ExpectCoeffsNear(sxy.dense(), dxy, 1e-12);
}
```

Register.

- [ ] **Step 2: Port sparse Cauchy from `main:include/tax/kernels/sparse_cauchy.hpp`**

Create `include/tax/kernels/sparse_cauchy.hpp` in `tax::detail::kernels::` namespace. Signatures take `storage::SparseContainer<T,N,M>&` for output and `const&` for inputs. The recurrence iterates the support of both inputs.

- [ ] **Step 3: Implement sparse `operator*(STE, STE)`**

Append to `include/tax/operators/arithmetic.hpp`:

```cpp
template <typename T, int N, int M>
[[nodiscard]] TaylorExpansion<T,N,M,storage::Sparse> operator*(
    const TaylorExpansion<T,N,M,storage::Sparse>& a,
    const TaylorExpansion<T,N,M,storage::Sparse>& b) noexcept {
    TaylorExpansion<T,N,M,storage::Sparse> r;
    detail::kernels::sparseCauchyProduct<T,N,M>(r.container(), a.container(), b.container());
    return r;
}
```

- [ ] **Step 4: Build, run, commit**

```bash
cmake --build build -j$(nproc)
ctest --test-dir build -R test_sparse_cauchy --output-on-failure
git add include/tax tests/kernels tests/CMakeLists.txt
git commit -m "slice10b: sparse Cauchy product"
```

---

## Task 16: Sparse subs — sqrt, reciprocal, division, integer pow

**Files:**
- Create: `include/tax/kernels/sparse_subs.hpp`
- Modify: `include/tax/operators/math_unary.hpp` (add sparse sqrt, reciprocal)
- Modify: `include/tax/operators/math_binary.hpp` (add sparse pow integer)
- Modify: `include/tax/operators/arithmetic.hpp` (add sparse `operator/`)
- Create: `tests/sparse/test_sparse_subs.cpp`

- [ ] **Step 1: Failing test**

Create `tests/sparse/test_sparse_subs.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../testUtils.hpp"

TEST(SparseSubs, SqrtMatchesDense) {
    auto x = tax::TE<5>::variable(4.0);
    auto sx = tax::sparse(x);
    tax::test::ExpectCoeffsNear(tax::sqrt(sx).dense(), tax::sqrt(x), 1e-12);
}

TEST(SparseSubs, ReciprocalMatchesDense) {
    auto x = tax::TE<5>::variable(2.0);
    auto sx = tax::sparse(x);
    tax::test::ExpectCoeffsNear(tax::reciprocal(sx).dense(), tax::reciprocal(x), 1e-12);
}

TEST(SparseSubs, DivisionMatchesDense) {
    auto x = tax::TE<5>::variable(3.0);
    auto y = tax::TE<5>::variable(2.0);
    auto sx = tax::sparse(x), sy = tax::sparse(y);
    tax::test::ExpectCoeffsNear((sx / sy).dense(), x / y, 1e-12);
}

TEST(SparseSubs, IntegerPowMatchesDense) {
    auto x = tax::TE<6>::variable(1.5);
    auto sx = tax::sparse(x);
    tax::test::ExpectCoeffsNear(tax::pow(sx, 3).dense(), tax::pow(x, 3), 1e-12);
}
```

Register.

- [ ] **Step 2: Port `seriesSqrtSparse`, `seriesReciprocalSparse`, `seriesDivSparse`, `seriesPowIntSparse` from `main:include/tax/kernels/sparse_subs.hpp`**

Keep the support-iteration optimizations from commits `a4d21b1` and `082f8d0`. Namespace `tax::detail::kernels`.

- [ ] **Step 3: Add sparse operator overloads**

```cpp
// math_unary.hpp
template <typename T, int N, int M>
[[nodiscard]] TaylorExpansion<T,N,M,storage::Sparse> sqrt(
    const TaylorExpansion<T,N,M,storage::Sparse>& x) noexcept;

template <typename T, int N, int M>
[[nodiscard]] TaylorExpansion<T,N,M,storage::Sparse> reciprocal(
    const TaylorExpansion<T,N,M,storage::Sparse>& x) noexcept;

// arithmetic.hpp
template <typename T, int N, int M>
[[nodiscard]] TaylorExpansion<T,N,M,storage::Sparse> operator/(
    const TaylorExpansion<T,N,M,storage::Sparse>& a,
    const TaylorExpansion<T,N,M,storage::Sparse>& b) noexcept;

// math_binary.hpp
template <typename T, int N, int M>
[[nodiscard]] TaylorExpansion<T,N,M,storage::Sparse> pow(
    const TaylorExpansion<T,N,M,storage::Sparse>& x, int n) noexcept;
```

- [ ] **Step 4: Build, run, commit**

```bash
cmake --build build -j$(nproc)
ctest --test-dir build -R test_sparse_subs --output-on-failure
git add include/tax tests/sparse tests/CMakeLists.txt
git commit -m "slice10c: sparse subs — sqrt, reciprocal, division, integer pow"
```

### Perf gate checkpoint

Build `bench_ops_sparse.cpp` (mirror of `bench_ops_dense.cpp` but with `STE`); compare against the baseline (capture pre-rewrite numbers from `main` for the same sparse benches). Any >5% regression blocks.

---

## Task 17: Eigen integration A — NumTraits, variables, value, eval, derivative

**Files:**
- Modify: `include/tax/eigen.hpp` (first half)
- Modify: `include/tax/tax.hpp` (include eigen.hpp)
- Create: `tests/eigen/test_eigen_numtraits.cpp`
- Create: `tests/eigen/test_eigen_variables.cpp`
- Create: `tests/eigen/test_eigen_eval.cpp`
- Create: `tests/eigen/test_eigen_derivative.cpp`

- [ ] **Step 1: Failing test — NumTraits**

Create `tests/eigen/test_eigen_numtraits.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <Eigen/Core>

TEST(EigenNumTraits, TaylorAsScalar) {
    using TE = tax::TE<3, 2>;
    Eigen::Matrix<TE, 2, 1> v;
    typename TE::Input p{1.0, 2.0};
    v(0) = TE::variable<0>(p);
    v(1) = TE::variable<1>(p);
    auto sum = v(0) + v(1);
    EXPECT_NEAR(sum.value(), 3.0, 1e-12);
}
```

Register.

- [ ] **Step 2: Implement `eigen.hpp` (NumTraits half)**

Create the first half of `include/tax/eigen.hpp`:

```cpp
#pragma once

#include <Eigen/Core>
#include <tax/core/taylor_expansion.hpp>

namespace Eigen {

template <typename T, int N, int M, typename Storage>
struct NumTraits<tax::TaylorExpansion<T, N, M, Storage>> : NumTraits<T> {
    using Self = tax::TaylorExpansion<T, N, M, Storage>;
    using Real = Self;
    using NonInteger = Self;
    using Nested = Self;
    enum { IsComplex = 0, IsInteger = 0, IsSigned = 1,
           RequireInitialization = 1,
           ReadCost = int(tax::numMonomials(N, M)),
           AddCost  = int(tax::numMonomials(N, M)),
           MulCost  = int(tax::numMonomials(N, M)) * int(tax::numMonomials(N, M)) };
};

}  // namespace Eigen
```

- [ ] **Step 3: Failing tests — variables, eval, derivative element-wise**

Create `tests/eigen/test_eigen_variables.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <tax/eigen.hpp>

TEST(EigenVariables, FromEigenVector) {
    Eigen::Vector3d x0{1.0, 2.0, 3.0};
    auto v = tax::variables<tax::TE<3, 3>>(x0);
    EXPECT_NEAR(v(0).value(), 1.0, 1e-15);
    EXPECT_NEAR(v(1).value(), 2.0, 1e-15);
    EXPECT_NEAR(v(2).value(), 3.0, 1e-15);
    EXPECT_NEAR((v(0).coeff<1,0,0>()), 1.0, 1e-15);
    EXPECT_NEAR((v(1).coeff<0,1,0>()), 1.0, 1e-15);
}
```

Create `tests/eigen/test_eigen_eval.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <tax/eigen.hpp>

TEST(EigenEval, ScalarTE) {
    auto x = tax::TE<3>::variable(1.0);
    auto f = x * x;          // (1 + dx)^2
    Eigen::Matrix<double, 1, 1> dx; dx << 0.1;
    double v = tax::eval(f, dx);
    EXPECT_NEAR(v, 1.21, 1e-12);
}

TEST(EigenEval, VectorOfTE) {
    Eigen::Vector2d x0{1.0, 2.0};
    auto v = tax::variables<tax::TE<3, 2>>(x0);
    Eigen::Matrix<tax::TE<3,2>, 2, 1> F;
    F(0) = v(0) * v(1);
    F(1) = v(0) + v(1);
    Eigen::Vector2d dx{0.1, -0.1};
    auto out = tax::eval(F, dx);
    EXPECT_NEAR(out(0), 1.1 * 1.9, 1e-12);
    EXPECT_NEAR(out(1), 1.1 + 1.9, 1e-12);
}
```

Create `tests/eigen/test_eigen_derivative.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <tax/eigen.hpp>

TEST(EigenDerivative, ElementWiseCompileTime) {
    Eigen::Vector2d x0{1.0, 2.0};
    auto v = tax::variables<tax::TE<3, 2>>(x0);
    Eigen::Matrix<tax::TE<3,2>, 2, 1> F;
    F(0) = v(0) * v(1);
    F(1) = v(0) + v(1);
    auto df_dx = tax::derivative<1, 0>(F);
    EXPECT_NEAR(df_dx(0), 2.0, 1e-12);   // d(x*y)/dx = y = 2 at x0
    EXPECT_NEAR(df_dx(1), 1.0, 1e-12);
}

TEST(EigenValue, ElementWise) {
    Eigen::Vector2d x0{1.0, 2.0};
    auto v = tax::variables<tax::TE<3, 2>>(x0);
    Eigen::Matrix<tax::TE<3,2>, 2, 1> F;
    F(0) = v(0) * v(1);
    F(1) = v(0) + v(1);
    auto val = tax::value(F);
    EXPECT_NEAR(val(0), 2.0, 1e-12);
    EXPECT_NEAR(val(1), 3.0, 1e-12);
}
```

Register all three.

- [ ] **Step 4: Implement second part of `eigen.hpp`**

Append to `include/tax/eigen.hpp`:

```cpp
namespace tax::detail::eigen {

template <typename> struct te_traits;
template <typename T, int N, int M, typename S>
struct te_traits<TaylorExpansion<T,N,M,S>> {
    using scalar_type = T;
    static constexpr int order_v = N;
    static constexpr int vars_v  = M;
    using storage_t = S;
};

template <typename T> struct is_te : std::false_type {};
template <typename T, int N, int M, typename S>
struct is_te<TaylorExpansion<T,N,M,S>> : std::true_type {};
template <typename T> inline constexpr bool is_te_v = is_te<T>::value;

}  // namespace tax::detail::eigen

namespace tax {

// variables — build a column vector of TE coordinate variables.
template <typename TE, typename Derived>
[[nodiscard]] auto variables(const Eigen::MatrixBase<Derived>& x0) {
    using tr = detail::eigen::te_traits<TE>;
    using T = typename tr::scalar_type;
    constexpr int M = tr::vars_v;
    static_assert(Derived::SizeAtCompileTime == M,
                  "variables(): Eigen input size must match TE::vars_v");
    typename TE::Input p{};
    for (int i = 0; i < M; ++i) p[std::size_t(i)] = T(x0(i));
    Eigen::Matrix<TE, M, 1> out;
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        ((out(int(I)) = TE::template variable<int(I)>(p)), ...);
    }(std::make_index_sequence<std::size_t(M)>{});
    return out;
}

// value(eigen-of-TE) -> eigen-of-scalar
template <typename Derived>
    requires (detail::eigen::is_te_v<typename Derived::Scalar>)
[[nodiscard]] auto value(const Eigen::MatrixBase<Derived>& F) {
    using TE = typename Derived::Scalar;
    using T  = typename detail::eigen::te_traits<TE>::scalar_type;
    Eigen::Matrix<T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
        out(F.rows(), F.cols());
    for (Eigen::Index i = 0; i < F.size(); ++i) out(i) = F.derived().coeff(i).value();
    return out;
}

// eval(eigen-of-TE, eigen-of-scalar-dx) -> eigen-of-scalar
template <typename Derived, typename DxDerived>
    requires (detail::eigen::is_te_v<typename Derived::Scalar>)
[[nodiscard]] auto eval(const Eigen::MatrixBase<Derived>& F,
                        const Eigen::MatrixBase<DxDerived>& dx) {
    using TE = typename Derived::Scalar;
    using T  = typename detail::eigen::te_traits<TE>::scalar_type;
    constexpr int M = detail::eigen::te_traits<TE>::vars_v;
    static_assert(DxDerived::SizeAtCompileTime == M);
    typename TE::Input p{};
    for (int i = 0; i < M; ++i) p[std::size_t(i)] = T(dx(i));
    Eigen::Matrix<T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
        out(F.rows(), F.cols());
    for (Eigen::Index i = 0; i < F.size(); ++i) out(i) = F.derived().coeff(i).eval(p);
    return out;
}

// derivative<Alpha...>(eigen-of-TE) -> eigen-of-scalar
template <int... Alpha, typename Derived>
    requires (detail::eigen::is_te_v<typename Derived::Scalar>)
[[nodiscard]] auto derivative(const Eigen::MatrixBase<Derived>& F) {
    using TE = typename Derived::Scalar;
    using T  = typename detail::eigen::te_traits<TE>::scalar_type;
    Eigen::Matrix<T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
        out(F.rows(), F.cols());
    for (Eigen::Index i = 0; i < F.size(); ++i)
        out(i) = F.derived().coeff(i).template derivative<Alpha...>();
    return out;
}

}  // namespace tax
```

Add `TaylorExpansion::eval(Eigen::Vector...)` member by wrapping the existing `eval(Input)`. Implement it inside `taylor_expansion.hpp` *only* if Eigen is available — actually, since Eigen is now required, just implement it always.

Append to `TaylorExpansion` (dense):

```cpp
template <typename Derived>
[[nodiscard]] T eval(const Eigen::MatrixBase<Derived>& dx) const noexcept {
    static_assert(Derived::SizeAtCompileTime == M);
    Input p{};
    for (int i = 0; i < M; ++i) p[std::size_t(i)] = T(dx(i));
    return eval(p);
}
```

This requires `<Eigen/Core>` to be included from `taylor_expansion.hpp`. Add it.

Also implement `eval(const Input& p)` itself by porting Horner-like evaluation from `main:include/tax/storage/tte_static.hpp`.

- [ ] **Step 5: Wire and build**

Append to `include/tax/tax.hpp`: `#include <tax/eigen.hpp>`.

```bash
cmake --build build -j$(nproc)
ctest --test-dir build -R "test_eigen_(numtraits|variables|eval|derivative)" --output-on-failure
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add include/tax tests/eigen tests/CMakeLists.txt
git commit -m "slice11: Eigen integration — NumTraits, variables, value, eval, derivative"
```

---

## Task 18: Eigen integration B — gradient, hessian, jacobian, invert

**Files:**
- Modify: `include/tax/eigen.hpp`
- Modify: `include/tax/core/taylor_expansion.hpp` (add `gradient()` and `hessian()` methods)
- Create: `tests/eigen/test_eigen_gradient.cpp`
- Create: `tests/eigen/test_eigen_hessian.cpp`
- Create: `tests/eigen/test_eigen_jacobian.cpp`
- Create: `tests/eigen/test_eigen_invert.cpp`

- [ ] **Step 1: Failing tests**

Create `tests/eigen/test_eigen_gradient.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <tax/eigen.hpp>

TEST(EigenGradient, OfQuadratic) {
    Eigen::Vector2d x0{1.0, 2.0};
    auto v = tax::variables<tax::TE<3, 2>>(x0);
    auto f = v(0) * v(0) + 2.0 * v(0) * v(1);
    auto g = tax::gradient(f);
    // df/dx = 2x + 2y; df/dy = 2x
    EXPECT_NEAR(g(0), 2.0 * 1.0 + 2.0 * 2.0, 1e-12);
    EXPECT_NEAR(g(1), 2.0 * 1.0, 1e-12);
}

TEST(EigenGradient, MethodMatchesFreeFunction) {
    Eigen::Vector2d x0{1.0, 2.0};
    auto v = tax::variables<tax::TE<3, 2>>(x0);
    auto f = v(0) * v(1);
    auto g1 = tax::gradient(f);
    auto g2 = f.gradient();
    EXPECT_NEAR((g1 - g2).norm(), 0.0, 1e-15);
}
```

Create `tests/eigen/test_eigen_hessian.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <tax/eigen.hpp>

TEST(EigenHessian, OfQuadratic) {
    Eigen::Vector2d x0{1.0, 2.0};
    auto v = tax::variables<tax::TE<3, 2>>(x0);
    auto f = v(0) * v(0) + 3.0 * v(0) * v(1) + v(1) * v(1);
    auto H = tax::hessian(f);
    EXPECT_NEAR(H(0, 0), 2.0, 1e-12);
    EXPECT_NEAR(H(0, 1), 3.0, 1e-12);
    EXPECT_NEAR(H(1, 0), 3.0, 1e-12);
    EXPECT_NEAR(H(1, 1), 2.0, 1e-12);
}
```

Create `tests/eigen/test_eigen_jacobian.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <tax/eigen.hpp>

TEST(EigenJacobian, OfLinearMap) {
    Eigen::Vector2d x0{1.0, 2.0};
    auto v = tax::variables<tax::TE<3, 2>>(x0);
    Eigen::Matrix<tax::TE<3,2>, 2, 1> F;
    F(0) = v(0) + 2.0 * v(1);
    F(1) = 3.0 * v(0) - v(1);
    auto J = tax::jacobian(F);
    EXPECT_NEAR(J(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(J(0, 1), 2.0, 1e-12);
    EXPECT_NEAR(J(1, 0), 3.0, 1e-12);
    EXPECT_NEAR(J(1, 1), -1.0, 1e-12);
}
```

Create `tests/eigen/test_eigen_invert.cpp`:

```cpp
#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <tax/eigen.hpp>

TEST(EigenInvert, OfShift) {
    // F(x) = x + 1. Inverse map: x → x - 1.
    Eigen::Matrix<double, 1, 1> x0; x0 << 0.0;
    auto v = tax::variables<tax::TE<3, 1>>(x0);
    Eigen::Matrix<tax::TE<3,1>, 1, 1> F;
    F(0) = v(0) + 1.0;
    auto Finv = tax::invert(F);
    // Finv(x) constant term should be -1.0
    EXPECT_NEAR(Finv(0).value(), -1.0, 1e-12);
    // Composing F(Finv) should give identity
    auto comp = tax::eval(F, tax::value(Finv));
    EXPECT_NEAR(comp(0), -1.0 + 1.0 + Finv(0).value(), 1e-10);
}
```

Register all four.

- [ ] **Step 2: Implement gradient/hessian methods on `TaylorExpansion`**

Append to `core/taylor_expansion.hpp` (dense specialization):

```cpp
[[nodiscard]] Eigen::Matrix<T, M, 1> gradient() const noexcept {
    Eigen::Matrix<T, M, 1> g;
    MultiIndex<M> alpha{};
    for (int i = 0; i < M; ++i) {
        alpha[std::size_t(i)] = 1;
        g(i) = derivative(alpha);
        alpha[std::size_t(i)] = 0;
    }
    return g;
}

[[nodiscard]] Eigen::Matrix<T, M, M> hessian() const noexcept {
    Eigen::Matrix<T, M, M> H;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            MultiIndex<M> alpha{};
            alpha[std::size_t(i)] += 1;
            alpha[std::size_t(j)] += 1;
            H(i, j) = derivative(alpha);
        }
    }
    return H;
}
```

- [ ] **Step 3: Implement free `gradient`, `hessian`, `jacobian`, `invert` in `eigen.hpp`**

Append to `include/tax/eigen.hpp`:

```cpp
namespace tax {

template <typename T, int N, int M, typename S>
[[nodiscard]] Eigen::Matrix<T, M, 1> gradient(const TaylorExpansion<T,N,M,S>& f) noexcept {
    return f.gradient();
}

template <typename T, int N, int M, typename S>
[[nodiscard]] Eigen::Matrix<T, M, M> hessian(const TaylorExpansion<T,N,M,S>& f) noexcept {
    return f.hessian();
}

template <typename Derived>
    requires (detail::eigen::is_te_v<typename Derived::Scalar>)
[[nodiscard]] auto jacobian(const Eigen::MatrixBase<Derived>& F) {
    using TE = typename Derived::Scalar;
    using tr = detail::eigen::te_traits<TE>;
    using T  = typename tr::scalar_type;
    constexpr int M = tr::vars_v;
    Eigen::Matrix<T, Derived::RowsAtCompileTime, M> out(F.size(), M);
    for (Eigen::Index r = 0; r < F.size(); ++r) {
        MultiIndex<M> alpha{};
        for (int j = 0; j < M; ++j) {
            alpha[std::size_t(j)] = 1;
            out(r, j) = F.derived().coeff(r).derivative(alpha);
            alpha[std::size_t(j)] = 0;
        }
    }
    return out;
}

// Polynomial map inversion — port from main:include/tax/eigen/invert_map.hpp.
template <typename Derived>
    requires (detail::eigen::is_te_v<typename Derived::Scalar>)
[[nodiscard]] auto invert(const Eigen::MatrixBase<Derived>& F);

}  // namespace tax
```

For `invert`, port the algorithm from `main:include/tax/eigen/invert_map.hpp` (the iterative inversion that uses jacobian + recursive composition). Drop dynamic-shape branches.

- [ ] **Step 4: Build, run**

```bash
cmake --build build -j$(nproc)
ctest --test-dir build -R "test_eigen_(gradient|hessian|jacobian|invert)" --output-on-failure
```

Expected: PASS.

- [ ] **Step 5: Run full perf gate**

```bash
cmake --build build-bench -j$(nproc)
./build-bench/benchmarks/bench_ops_dense --benchmark_min_time=0.5s | tee /tmp/stage1-final-dense.txt
./build-bench/benchmarks/bench_ops_sparse --benchmark_min_time=0.5s | tee /tmp/stage1-final-sparse.txt
./build-bench/benchmarks/bench_eigen_workflows --benchmark_min_time=0.5s | tee /tmp/stage1-final-eigen.txt
# Diff against benchmarks/baseline/main-*.txt manually and document any >5% regressions.
```

- [ ] **Step 6: Commit**

```bash
git add include/tax tests/eigen tests/CMakeLists.txt
git commit -m "slice12: Eigen integration — gradient, hessian, jacobian, invert"
```

---

## Task 19: CI workflows

**Files:**
- Modify: `.github/workflows/tests.yml`
- Modify: `.github/workflows/sanitizers.yml`
- Create: `.github/workflows/bench.yml`

- [ ] **Step 1: Slim `tests.yml`**

Replace the Ubuntu/macOS × GCC/Clang × Eigen 3.4/5.0 matrix block; remove the DACE-on combinations; ensure it builds `tax` with `TAX_BUILD_TEST=ON` and runs `ctest`. No Python. No DACE.

- [ ] **Step 2: Slim `sanitizers.yml`**

Keep ASAN/UBSAN/TSAN jobs; remove DACE flag from build command.

- [ ] **Step 3: Create `bench.yml`**

The workflow checks out the PR, builds `bench_ops_dense`, `bench_ops_sparse`, `bench_eigen_workflows`, runs them, and compares against `benchmarks/baseline/main-<sha>.txt`. On >5% regression, fail (unless commit message contains `[allow-perf-regression]`).

A minimal implementation:

```yaml
name: bench
on: [pull_request]
jobs:
  bench:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install deps
        run: |
          sudo apt-get update
          sudo apt-get install -y cmake g++ libeigen3-dev libbenchmark-dev
      - name: Build benches
        run: |
          cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DTAX_BUILD_BENCHMARK=ON -DTAX_BUILD_TEST=OFF
          cmake --build build-bench -j$(nproc)
      - name: Run benches
        run: |
          ./build-bench/benchmarks/bench_ops_dense --benchmark_min_time=0.5s > bench-dense.txt
          ./build-bench/benchmarks/bench_ops_sparse --benchmark_min_time=0.5s > bench-sparse.txt
      - name: Compare against baseline
        run: |
          python3 .github/scripts/perf_diff.py \
              --baseline benchmarks/baseline \
              --current bench-dense.txt bench-sparse.txt \
              --threshold 5
```

`perf_diff.py` is a small script (create at `.github/scripts/perf_diff.py`) that parses Google Benchmark output, matches benchmark names against baseline, and exits 1 on regression > threshold unless commit message contains `[allow-perf-regression]`.

- [ ] **Step 4: Commit**

```bash
git add .github
git commit -m "ci: slim tests.yml + sanitizers.yml; add bench.yml perf gate"
```

---

## Task 20: Final cleanup + README

**Files:**
- Modify: `README.md`
- Modify: `docs/index.md`, `docs/getting_started.md`
- Modify: `mkdocs.yml`

- [ ] **Step 1: Update `README.md`**

Reflect Stage 1 surface: static dense + sparse, no ODE/ADS, Eigen-first. Link to spec and plan docs.

- [ ] **Step 2: Update top-level docs**

Trim `docs/` to Stage 1 surface. Keep `getting_started.md`, `index.md`. Delete `docs/ads/`, `docs/taylor/`, `docs/python.md`, `docs/dynamic.md`, `docs/vector/`.

- [ ] **Step 3: Update `mkdocs.yml`** to match the trimmed nav.

- [ ] **Step 4: Final test + bench sanity**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTAX_BUILD_TEST=ON -G Ninja
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add README.md docs mkdocs.yml
git commit -m "stage1: trim docs to new surface; update README"
```

---

## Stage 1 complete

At this point the branch contains:

- One class `tax::TaylorExpansion<T, N, M, Storage>` with dense and sparse policies, eager operators, full math surface.
- Single-file `tax/eigen.hpp` with NumTraits + variables/value/eval/derivative + gradient/hessian/jacobian/invert.
- Tests passing across foundation, kernels, operators, sparse, eigen.
- Perf gate green or with documented `[allow-perf-regression]` annotations.
- CI: tests + sanitizers + bench.

What comes in Stage 2+ (separate plans):

- Dynamic-shape `TaylorExpansion` (slot already designed for).
- Python bindings.
- ODE / ADS integrators.
- DACE comparison harness.
