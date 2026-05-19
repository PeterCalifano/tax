# tax Stage 3 Implementation Plan — DACE regression suite

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a DACE-based coefficient-parity regression suite for the Stage 1
dense surface, gated by a new `TAX_BUILD_REGRESSIONS` CMake option, and
rename `TAX_BUILD_TEST` to `TAX_BUILD_UNITTESTS`.

**Architecture:** Four slices. Slice 1 renames the existing test flag and
scaffolds an empty `tests/regression/` subdir with a sentinel non-DACE test
behind a new flag. Slice 2 wires `FetchContent` for DACE v2.1.0, ports the
two surviving pre-Stage-1 DACE tests retargeted at the new API, and adds the
shared `prepareInput` pre-step. Slice 3 expands coverage to deriv/integ,
eval, gradient/hessian/jacobian, invert. Slice 4 adds a manual-dispatch CI
workflow.

**Tech Stack:** CMake (FetchContent), Google Test, DACE v2.1.0
(`dacelib/dace@v2.1.0`), Eigen ≥ 3.4. Source language is C++23.

**Spec:** `docs/superpowers/specs/2026-05-19-tax-stage3-design.md`.

**Reference branch (for porting):** `claude/add-verner-integrators-vEgRF`.
The pre-Stage-1 DACE tests live there at `tests/dace/testUnivariate.cpp`
and `tests/dace/testMultivariate.cpp`.

**Build env:** micromamba env `tax` (cmake, gxx, eigen, benchmark, ninja).
Activate: `eval "$(micromamba shell hook --shell bash)" && micromamba activate tax`.

---

## File Structure

Files created in this plan:

- `tests/regression/CMakeLists.txt` — register regression executables under
  `TAX_BUILD_REGRESSIONS`.
- `tests/regression/regressionUtils.hpp` — shared `expectCoeffsMatch` and
  `prepareInput` helpers.
- `tests/regression/testSentinel.cpp` — slice 1 wiring test (no DACE).
- `tests/regression/testUnivariate.cpp` — slice 2 port.
- `tests/regression/testMultivariate.cpp` — slice 2 port.
- `tests/regression/testDerivInteg.cpp` — slice 3.
- `tests/regression/testEval.cpp` — slice 3.
- `tests/regression/testEigenVectorCalc.cpp` — slice 3 (gradient, hessian,
  jacobian).
- `tests/regression/testInvert.cpp` — slice 3.
- `.github/workflows/regressions.yml` — slice 4.

Files modified:

- `CMakeLists.txt` — rename option, add `TAX_BUILD_REGRESSIONS`, wire
  DACE FetchContent, add the `tests/regression` subdir guard.
- `tests/CMakeLists.txt` — add `tax_add_regression()` helper.
- `.github/workflows/tests.yml` — option rename only.
- `.github/workflows/sanitizers.yml` — option rename only.
- `README.md` — option rename.
- `docs/getting_started.md` — option rename.
- `CLAUDE.md` — option rename + reference the regression option.

Files deleted:

- `tests/regression/testSentinel.cpp` — removed in slice 2 once real tests
  exist.

---

## Slice 1 — Flag rename + regression scaffold

Goal of slice 1: end with a working build where `TAX_BUILD_TEST` is gone
(replaced by `TAX_BUILD_UNITTESTS`), `TAX_BUILD_REGRESSIONS` exists as an
option (default OFF), and turning it on builds and runs a single trivial
non-DACE test. No DACE involvement yet.

### Task 1: Rename `TAX_BUILD_TEST` → `TAX_BUILD_UNITTESTS` in root CMake

**Files:**
- Modify: `CMakeLists.txt:4` and `CMakeLists.txt:38`

- [ ] **Step 1.1:** Edit `CMakeLists.txt` line 4 — the option declaration.

  Replace:

  ```cmake
  option(TAX_BUILD_TEST      "Build and enable unit tests"            ON)
  ```

  With:

  ```cmake
  option(TAX_BUILD_UNITTESTS "Build and enable unit tests"            ON)
  ```

- [ ] **Step 1.2:** Edit `CMakeLists.txt` line 38 — the test subdir gate.

  Replace:

  ```cmake
  if(TAX_BUILD_TEST)
      enable_testing()
      add_subdirectory(tests)
  endif()
  ```

  With:

  ```cmake
  if(TAX_BUILD_UNITTESTS)
      enable_testing()
      add_subdirectory(tests)
  endif()
  ```

- [ ] **Step 1.3:** Configure with the new flag name. Run:

  ```bash
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTAX_BUILD_UNITTESTS=ON
  ```

  Expected: succeeds; no warning about unrecognised option.

- [ ] **Step 1.4:** Build and run unit tests. Run:

  ```bash
  cmake --build build -j
  ctest --test-dir build --output-on-failure
  ```

  Expected: all existing unit tests still pass.

### Task 2: Update CI workflows for the rename

**Files:**
- Modify: `.github/workflows/tests.yml` (every occurrence of `TAX_BUILD_TEST`)
- Modify: `.github/workflows/sanitizers.yml` (every occurrence)

- [ ] **Step 2.1:** Edit `.github/workflows/tests.yml`. There are two
  occurrences of `-DTAX_BUILD_TEST=ON` (one in the `build-and-test` job's
  Configure step, one in the `coverage` job's Configure step). Replace both
  with `-DTAX_BUILD_UNITTESTS=ON`.

- [ ] **Step 2.2:** Edit `.github/workflows/sanitizers.yml`. There are two
  occurrences of `-DTAX_BUILD_TEST=ON` (one in `precheck`, one in
  `sanitizers`). Replace both with `-DTAX_BUILD_UNITTESTS=ON`.

- [ ] **Step 2.3:** Verify no stale references remain.

  ```bash
  grep -rn "TAX_BUILD_TEST" .github/
  ```

  Expected: no output.

### Task 3: Update docs for the rename

**Files:**
- Modify: `README.md` (line 118 region)
- Modify: `docs/getting_started.md` (line 23 region)
- Modify: `CLAUDE.md` (line 85 region; the line at 105 mentions
  `python_bindings`, that whole sentence describes a deferred feature — leave
  unchanged for now since the rename is the only thing in scope here)

- [ ] **Step 3.1:** Edit `README.md` line 118 — change
  `` | `TAX_BUILD_TEST` | `ON` | Build the test suite | `` to
  `` | `TAX_BUILD_UNITTESTS` | `ON` | Build the unit-test suite | ``.

- [ ] **Step 3.2:** Edit `docs/getting_started.md` line 23 — change
  `` | `TAX_BUILD_TEST` | `ON` | Build Google Test suite | `` to
  `` | `TAX_BUILD_UNITTESTS` | `ON` | Build Google Test unit-test suite | ``.

- [ ] **Step 3.3:** Edit `CLAUDE.md` line 85 — change
  `` | `TAX_BUILD_TEST` | `ON` | Build Google Test suite | `` to
  `` | `TAX_BUILD_UNITTESTS` | `ON` | Build Google Test unit-test suite | ``.

- [ ] **Step 3.4:** Verify no stale references remain outside specs/plans
  (the spec and the previous plan in `docs/superpowers/` reference the old
  name historically and should not be edited).

  ```bash
  grep -rn "TAX_BUILD_TEST" --include="*.md" --include="*.txt" --include="*.cmake" --include="*.in" --include="*.yml" \
       --exclude-dir=build --exclude-dir=.git --exclude-dir=superpowers .
  ```

  Expected: no output.

### Task 4: Add `TAX_BUILD_REGRESSIONS` option and wiring stub

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 4.1:** Edit `CMakeLists.txt` immediately after the
  `TAX_BUILD_UNITTESTS` option. Insert:

  ```cmake
  option(TAX_BUILD_REGRESSIONS "Build DACE-based regression tests"      OFF)
  ```

  The full options block should now read:

  ```cmake
  option(TAX_BUILD_UNITTESTS   "Build and enable unit tests"                  ON)
  option(TAX_BUILD_REGRESSIONS "Build DACE-based regression tests"            OFF)
  option(TAX_BUILD_BENCHMARK   "Build benchmark suite (Google Benchmark)"     OFF)
  option(TAX_USE_UNROLL        "Use compile-time-unrolled M=1 Cauchy kernels" ON)
  option(TAX_USE_STENCIL       "Use precomputed Cauchy stencils for M>=2"     ON)
  ```

- [ ] **Step 4.2:** Edit `CMakeLists.txt` near the test subdir gate. Add
  the regression subdir gate immediately after the unit-test one. Replace:

  ```cmake
  if(TAX_BUILD_UNITTESTS)
      enable_testing()
      add_subdirectory(tests)
  endif()
  if(TAX_BUILD_BENCHMARK)
      add_subdirectory(benchmarks)
  endif()
  ```

  With:

  ```cmake
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
  ```

  Rationale for the dedicated `add_subdirectory(tests/regression)`: keeps the
  regression target list isolated from `tests/CMakeLists.txt` so users
  building with `-DTAX_BUILD_UNITTESTS=OFF -DTAX_BUILD_REGRESSIONS=ON`
  do not pay the GoogleTest FetchContent inside `tests/CMakeLists.txt`
  (it is fetched only by the unit-test subdir). The regression subdir will
  re-find GoogleTest in slice 2.

### Task 5: Scaffold `tests/regression/` with a sentinel test

The point of the sentinel test is to verify that the regression subdir
builds and links **before** DACE is wired in slice 2. It is deleted in
slice 2 once real ported tests exist.

**Files:**
- Create: `tests/regression/CMakeLists.txt`
- Create: `tests/regression/testSentinel.cpp`

- [ ] **Step 5.1:** Create `tests/regression/CMakeLists.txt`:

  ```cmake
  # tests/regression/CMakeLists.txt
  #
  # Built only when TAX_BUILD_REGRESSIONS=ON. Slice 1 ships a sentinel test
  # that does not depend on DACE so the wiring can be verified in isolation.
  # Slice 2 replaces it with the real DACE-backed regression tests.

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

  # tax_add_regression(name SOURCES files...)
  #
  # Slice 1: links tax + GTest only. Slice 2 extends this helper to also
  # link the resolved DACE target.
  function(tax_add_regression name)
      cmake_parse_arguments(R "" "" "SOURCES" ${ARGN})
      add_executable(${name} ${R_SOURCES})
      target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
      target_link_libraries(${name} PRIVATE tax GTest::gtest GTest::gtest_main)
      add_test(NAME ${name} COMMAND ${name})
  endfunction()

  tax_add_regression(test_regression_sentinel SOURCES testSentinel.cpp)
  ```

- [ ] **Step 5.2:** Create `tests/regression/testSentinel.cpp`:

  ```cpp
  // tests/regression/testSentinel.cpp
  //
  // Slice 1 wiring test. Verifies that TAX_BUILD_REGRESSIONS=ON builds the
  // regression subdir, links Google Test, and finds the tax target. No DACE.
  // Replaced in slice 2 by the ported testUnivariate / testMultivariate.

  #include <gtest/gtest.h>
  #include <tax/tax.hpp>

  TEST(RegressionSentinel, TaxHeaderIncludes)
  {
      constexpr int N = 4;
      auto x = tax::TE<N>::variable(0.5);
      EXPECT_DOUBLE_EQ(x.value(), 0.5);
  }
  ```

- [ ] **Step 5.3:** Configure with the regression flag on. Run:

  ```bash
  cmake -S . -B build-reg -DCMAKE_BUILD_TYPE=Release \
        -DTAX_BUILD_UNITTESTS=OFF -DTAX_BUILD_REGRESSIONS=ON
  ```

  Expected: configures cleanly; the message log shows the
  `tests/regression` subdir being processed; no DACE fetch happens.

- [ ] **Step 5.4:** Build and run the sentinel. Run:

  ```bash
  cmake --build build-reg -j
  ctest --test-dir build-reg --output-on-failure
  ```

  Expected: one test (`test_regression_sentinel`) runs and passes.

- [ ] **Step 5.5:** Verify the unit-test build path still works (independent
  configuration; uses the existing `build/` dir from Task 1).

  ```bash
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DTAX_BUILD_UNITTESTS=ON -DTAX_BUILD_REGRESSIONS=OFF
  cmake --build build -j
  ctest --test-dir build --output-on-failure
  ```

  Expected: unit tests pass; the regression subdir is not entered.

### Task 6: Commit slice 1

- [ ] **Step 6.1:** Stage the slice 1 changes. Run:

  ```bash
  git add CMakeLists.txt \
          .github/workflows/tests.yml .github/workflows/sanitizers.yml \
          README.md docs/getting_started.md CLAUDE.md \
          tests/regression/CMakeLists.txt tests/regression/testSentinel.cpp
  ```

- [ ] **Step 6.2:** Commit. Use the heredoc form:

  ```bash
  git commit -m "$(cat <<'EOF'
  stage3-slice1: rename TAX_BUILD_TEST → TAX_BUILD_UNITTESTS, add TAX_BUILD_REGRESSIONS scaffold

  Renames the existing test option for clarity now that two test categories
  exist, and adds the new TAX_BUILD_REGRESSIONS option with an isolated
  tests/regression/ subdir and a sentinel test that does not depend on DACE.
  The sentinel verifies wiring before slice 2 introduces FetchContent for
  DACE v2.1.0.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

- [ ] **Step 6.3:** Verify with `git status` (should be clean) and
  `git log -1 --stat`.

---

## Slice 2 — DACE provisioning + ported tests

Goal of slice 2: end with the regression suite linked against DACE v2.1.0,
the two pre-Stage-1 tests ported (univariate + multivariate) and rewritten
to call `prepareInput` before every op, and `regressionUtils.hpp`
containing both `expectCoeffsMatch` and `prepareInput`.

### Task 7: Wire DACE FetchContent under `TAX_BUILD_REGRESSIONS`

The block below mirrors the pre-Stage-1 wiring on the reference branch
(`claude/add-verner-integrators-vEgRF`). DACE exposes its targets under
several names depending on how it was built; the alias dance is what makes
the downstream `target_link_libraries(... ${TAX_DACE_TARGET})` line stable.

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 7.1:** Edit `CMakeLists.txt`. After the existing
  `find_package(Eigen3 3.4...99 REQUIRED)` and
  `target_link_libraries(tax INTERFACE Eigen3::Eigen)` lines, and **before**
  the test/regression subdir gates, insert:

  ```cmake
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
  ```

- [ ] **Step 7.2:** Configure with the regression flag on (use a fresh
  build dir so FetchContent runs):

  ```bash
  rm -rf build-reg
  cmake -S . -B build-reg -DCMAKE_BUILD_TYPE=Release \
        -DTAX_BUILD_UNITTESTS=OFF -DTAX_BUILD_REGRESSIONS=ON
  ```

  Expected: the configure step fetches DACE v2.1.0 from GitHub and prints
  `DACE target for regression tests: dace::dace_s` (or one of the
  fallbacks). The fetch may take a minute on first run.

  If DACE's CMake itself fails to configure (e.g. on a very recent
  compiler), pin Ubuntu 22.04 default GCC in slice 4's CI workflow; for
  local builds, try a different compiler within the `tax` micromamba env.

### Task 8: Extend `tax_add_regression` to link DACE; replace the sentinel

**Files:**
- Modify: `tests/regression/CMakeLists.txt`

- [ ] **Step 8.1:** Replace the contents of `tests/regression/CMakeLists.txt`
  with:

  ```cmake
  # tests/regression/CMakeLists.txt
  #
  # Built only when TAX_BUILD_REGRESSIONS=ON. Each test links the resolved
  # DACE target (set in the root CMakeLists.txt under TAX_DACE_TARGET) and
  # compares tax coefficient outputs against DACE references.

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

  if(NOT TAX_DACE_TARGET)
      message(FATAL_ERROR
          "tests/regression requires TAX_DACE_TARGET to be resolved by the root CMakeLists.txt")
  endif()

  # tax_add_regression(name SOURCES files...)
  function(tax_add_regression name)
      cmake_parse_arguments(R "" "" "SOURCES" ${ARGN})
      add_executable(${name} ${R_SOURCES})
      target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
      target_link_libraries(${name} PRIVATE
          tax
          ${TAX_DACE_TARGET}
          GTest::gtest
          GTest::gtest_main)
      add_test(NAME ${name} COMMAND ${name})
  endfunction()

  tax_add_regression(test_regression_univariate   SOURCES testUnivariate.cpp)
  tax_add_regression(test_regression_multivariate SOURCES testMultivariate.cpp)
  ```

- [ ] **Step 8.2:** Delete the sentinel source:

  ```bash
  git rm tests/regression/testSentinel.cpp
  ```

  (The file will be re-staged at slice end; for now keep the source tree
  in the right shape for build-time verification.)

### Task 9: Create `tests/regression/regressionUtils.hpp`

Shared helpers used by every regression test. `expectCoeffsMatch` does
coefficient-wise equality between a tax expansion and a `DACE::DA`
reference, tol=1e-12. `prepareInput` is the shared pre-step that turns
`variable(x0)` into a polynomial with non-trivial structure (default
choice: `tax::sin(x)` univariate, `tax::sin(sum_i x_i)` multivariate).

If a specific op rejects this default (e.g. `log`, `sqrt`, `asin` all need
positive constant terms), the choice is shifted globally — never per
test. The candidate fallback documented in the spec is
`1.0 + 0.5 * sin(x)^2`, which stays in `[1, 1.5]`. Settle the final choice
in step 9.4 below after sanity-checking the affected ops.

**Files:**
- Create: `tests/regression/regressionUtils.hpp`

- [ ] **Step 9.1:** Create `tests/regression/regressionUtils.hpp` with the
  full helper surface:

  ```cpp
  // tests/regression/regressionUtils.hpp
  //
  // Shared helpers for DACE regression tests.
  //   - expectCoeffsMatch:  coefficient-wise equality between a tax
  //                         TaylorExpansion and a DACE::DA reference.
  //   - prepareInput:       shared pre-step that wraps the input variable(s)
  //                         in a fixed expression to produce a polynomial
  //                         with non-trivial structure. Applied identically
  //                         on both sides of every test.

  #pragma once

  #include <dace/dace.h>
  #include <gtest/gtest.h>

  #include <cmath>
  #include <iomanip>
  #include <vector>

  #include <tax/tax.hpp>

  namespace tax_regression
  {

  // -----------------------------------------------------------------------
  // expectCoeffsMatch — univariate
  // -----------------------------------------------------------------------
  template < int N >
  ::testing::AssertionResult expectCoeffsMatch( const tax::TE< N >& tested,
                                                const DACE::DA&      ref,
                                                double               tol = 1e-12 )
  {
      for ( unsigned int k = 0; k <= unsigned( N ); ++k )
      {
          const double c_ref = ref.getCoefficient( std::vector< unsigned int >{ k } );
          const double c_tax = static_cast< double >( tested.template coeff< int( 0 ) + 0 >() );
          // The template-coeff above is wrong for runtime k; use runtime path:
          (void)c_tax;
          const tax::MultiIndex< 1 > alpha{ int( k ) };
          const double c_tax_runtime = tested.coeff( alpha );

          if ( !( std::isfinite( c_ref ) && std::isfinite( c_tax_runtime ) ) )
          {
              return ::testing::AssertionFailure()
                  << "Non-finite coefficient at k=" << k
                  << " (DACE=" << c_ref << ", tax=" << c_tax_runtime << ")";
          }
          const double diff = std::abs( c_ref - c_tax_runtime );
          if ( diff > tol )
          {
              return ::testing::AssertionFailure()
                  << "Coefficient mismatch at k=" << k
                  << " (DACE=" << std::setprecision( 17 ) << c_ref
                  << ", tax=" << std::setprecision( 17 ) << c_tax_runtime
                  << ", |diff|=" << diff << ", tol=" << tol << ")";
          }
      }
      return ::testing::AssertionSuccess();
  }

  // -----------------------------------------------------------------------
  // expectCoeffsMatch — multivariate
  // -----------------------------------------------------------------------
  template < int N, int M >
  ::testing::AssertionResult expectCoeffsMatch( const tax::TEn< N, M >& tested,
                                                const DACE::DA&         ref,
                                                double                  tol = 1e-12 )
  {
      const std::size_t total = tax::numMonomials( N, M );
      for ( std::size_t k = 0; k < total; ++k )
      {
          const auto alpha = tax::unflatIndex< M >( k );

          std::vector< unsigned int > vindex( std::size_t( M ) );
          for ( int i = 0; i < M; ++i )
              vindex[std::size_t( i )] = static_cast< unsigned int >( alpha[std::size_t( i )] );

          const double c_ref = ref.getCoefficient( vindex );
          const double c_tax = tested.coeff( alpha );

          if ( !( std::isfinite( c_ref ) && std::isfinite( c_tax ) ) )
          {
              return ::testing::AssertionFailure()
                  << "Non-finite coefficient at k=" << k
                  << " (DACE=" << c_ref << ", tax=" << c_tax << ")";
          }
          const double diff = std::abs( c_ref - c_tax );
          if ( diff > tol )
          {
              return ::testing::AssertionFailure()
                  << "Coefficient mismatch at k=" << k
                  << " (DACE=" << std::setprecision( 17 ) << c_ref
                  << ", tax=" << std::setprecision( 17 ) << c_tax
                  << ", |diff|=" << diff << ", tol=" << tol << ")";
          }
      }
      return ::testing::AssertionSuccess();
  }

  // -----------------------------------------------------------------------
  // prepareInput — shared pre-step applied identically on both sides
  // -----------------------------------------------------------------------
  //
  // Default choice: 1.0 + 0.5 * sin(x)^2 (univariate), which yields a
  // polynomial with non-trivial structure (every other low-order coefficient
  // is zero by symmetry of sin around 0) and stays in [1, 1.5] so domain-
  // restricted ops (log, sqrt, asin) accept it.
  //
  // If a particular op needs a different range it must be addressed by
  // shifting this prep globally, not by special-casing the test.

  template < int N >
  [[nodiscard]] tax::TE< N > prepareInput( const tax::TE< N >& x ) noexcept
  {
      const auto s = tax::sin( x );
      return 1.0 + 0.5 * ( s * s );
  }

  template < int N, int M >
  [[nodiscard]] tax::TEn< N, M > prepareInput( const tax::TEn< N, M >& x ) noexcept
  {
      const auto s = tax::sin( x );
      return 1.0 + 0.5 * ( s * s );
  }

  [[nodiscard]] inline DACE::DA prepareInput( const DACE::DA& x )
  {
      const auto s = x.sin();
      return 1.0 + 0.5 * ( s * s );
  }

  }  // namespace tax_regression
  ```

  Notes for the implementer:
  - The univariate `expectCoeffsMatch` body above contains a stray
    template-coeff call that is then overridden by the runtime path; that
    is intentional — replace the body with the cleaner single-path form
    below in step 9.2.
  - DACE's `getCoefficient` takes a `std::vector<unsigned int>` of length
    M.  For univariate, the vector has length 1 (matching DACE's
    `init(N, 1)`).
  - `tax::numMonomials` and `tax::unflatIndex<M>` are in the public
    `tax::` namespace (see `include/tax/core/multi_index.hpp`).

- [ ] **Step 9.2:** Clean up the univariate `expectCoeffsMatch` to remove
  the misleading scratch line — replace lines from `for ( unsigned int k`
  through `return ::testing::AssertionSuccess(); }` of the **univariate
  overload only** with:

  ```cpp
  template < int N >
  ::testing::AssertionResult expectCoeffsMatch( const tax::TE< N >& tested,
                                                const DACE::DA&      ref,
                                                double               tol = 1e-12 )
  {
      for ( unsigned int k = 0; k <= unsigned( N ); ++k )
      {
          const double c_ref = ref.getCoefficient( std::vector< unsigned int >{ k } );
          const tax::MultiIndex< 1 > alpha{ int( k ) };
          const double c_tax = tested.coeff( alpha );

          if ( !( std::isfinite( c_ref ) && std::isfinite( c_tax ) ) )
          {
              return ::testing::AssertionFailure()
                  << "Non-finite coefficient at k=" << k
                  << " (DACE=" << c_ref << ", tax=" << c_tax << ")";
          }
          const double diff = std::abs( c_ref - c_tax );
          if ( diff > tol )
          {
              return ::testing::AssertionFailure()
                  << "Coefficient mismatch at k=" << k
                  << " (DACE=" << std::setprecision( 17 ) << c_ref
                  << ", tax=" << std::setprecision( 17 ) << c_tax
                  << ", |diff|=" << diff << ", tol=" << tol << ")";
          }
      }
      return ::testing::AssertionSuccess();
  }
  ```

- [ ] **Step 9.3:** Sanity-build `regressionUtils.hpp` by adding a
  one-test file that just includes it; transient — will be removed in step
  9.5. Create `tests/regression/testUnivariate.cpp` with a placeholder:

  ```cpp
  // Placeholder — overwritten in Task 10.
  #include "regressionUtils.hpp"
  #include <gtest/gtest.h>
  TEST(RegressionUnivariate, PlaceholderCompiles) { SUCCEED(); }
  ```

  And `tests/regression/testMultivariate.cpp`:

  ```cpp
  // Placeholder — overwritten in Task 11.
  #include "regressionUtils.hpp"
  #include <gtest/gtest.h>
  TEST(RegressionMultivariate, PlaceholderCompiles) { SUCCEED(); }
  ```

- [ ] **Step 9.4:** Configure and build to confirm `regressionUtils.hpp`
  compiles against the actual DACE headers and the Stage 1 tax surface:

  ```bash
  cmake --build build-reg -j
  ```

  Expected: both placeholder tests build cleanly. If a compile error appears
  in `prepareInput` (e.g. operator mismatch), the prep choice needs to
  change — adjust and rebuild before moving on.

- [ ] **Step 9.5:** Run the placeholder tests; both pass:

  ```bash
  ctest --test-dir build-reg --output-on-failure
  ```

  Expected: 2 tests pass.

### Task 10: Port `testUnivariate.cpp`

The pre-Stage-1 file is `tests/dace/testUnivariate.cpp` on branch
`claude/add-verner-integrators-vEgRF`. It has ~349 lines covering: Div,
MulDiv, Sin, Cos, Tan, Asin, Acos, Atan, Atan2, Exp, Log, Sqrt, Cbrt,
Reciprocal, Square, Cube, Pow (int), Pow (real), Sinh, Cosh, Tanh, Asinh,
Acosh, Atanh, Erf — checking parity at `N = 40`.

Three things change from the original:

1. Type rename: `tax::DA<N>` → `tax::TE<N>`.
2. Apply `prepareInput` to the variable before the op under test on both
   sides.
3. Drop tests for ops whose domain `prepareInput` would violate. With the
   default prep `1 + 0.5 * sin(x)^2 ∈ [1, 1.5]`, all dense ops in scope
   are valid (acosh requires ≥1; atan2 keeps both args bounded;
   log/sqrt/cbrt/reciprocal need >0 — fine).

**Files:**
- Modify (overwrite the placeholder from step 9.3):
  `tests/regression/testUnivariate.cpp`

- [ ] **Step 10.1:** Fetch the original file for reference (do not commit;
  this is a local working file):

  ```bash
  git show claude/add-verner-integrators-vEgRF:tests/dace/testUnivariate.cpp \
      > /tmp/orig_testUnivariate.cpp
  ```

- [ ] **Step 10.2:** Overwrite `tests/regression/testUnivariate.cpp` with
  the ported version. Header and a representative set of tests are spelled
  out below; the remaining ops (Cos, Tan, Asin, Acos, Atan, Exp, Log,
  Sqrt, Cbrt, Square, Cube, Pow with int, Pow with real, Sinh, Cosh,
  Tanh, Asinh, Acosh, Atanh, Erf, Atan2) follow the same shape — copy
  each from `/tmp/orig_testUnivariate.cpp`, rename `tax::DA<N>` →
  `tax::TE<N>`, and wrap the input variable in `tax_regression::prepareInput`
  on both sides.

  Header block and first three tests verbatim:

  ```cpp
  // tests/regression/testUnivariate.cpp
  //
  // Ported from claude/add-verner-integrators-vEgRF:tests/dace/testUnivariate.cpp
  // and retargeted at tax::TE<N>. Every input variable is wrapped in
  // tax_regression::prepareInput before the op under test, on both the
  // DACE and tax side, so the operators are exercised against polynomials
  // with non-trivial structure rather than the bare variable(x0).

  #include <dace/dace.h>
  #include <gtest/gtest.h>

  #include <tax/tax.hpp>

  #include "regressionUtils.hpp"

  using tax_regression::expectCoeffsMatch;
  using tax_regression::prepareInput;

  // -----------------------------------------------------------------------
  // Operators
  // -----------------------------------------------------------------------

  TEST( DaceUnivariate, Div )
  {
      constexpr int N = 40;

      DACE::DA::init( N, 1 );
      DACE::DA xr( 1 );
      DACE::DA f_ref = prepareInput( xr );
      DACE::DA yr    = 1.0 / ( 1.0 + f_ref );

      auto x = tax::TE< N >::variable( 1.0 );
      auto f = prepareInput( x );
      tax::TE< N > y = 1.0 / ( 1.0 + f );

      EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
  }

  TEST( DaceUnivariate, MulDiv )
  {
      constexpr int N = 40;

      DACE::DA::init( N, 1 );
      DACE::DA xr( 1 );
      DACE::DA f_ref = prepareInput( xr );
      DACE::DA yr    = 1.0 / ( ( 1.0 + f_ref ) * ( 1.0 + f_ref ) );

      auto x = tax::TE< N >::variable( 1.0 );
      auto f = prepareInput( x );
      tax::TE< N > y = 1.0 / ( ( 1.0 + f ) * ( 1.0 + f ) );

      EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
  }

  // -----------------------------------------------------------------------
  // Math
  // -----------------------------------------------------------------------

  TEST( DaceUnivariate, Sin )
  {
      constexpr int N = 40;

      DACE::DA::init( N, 1 );
      DACE::DA xr( 1 );
      DACE::DA f_ref = prepareInput( xr );
      DACE::DA yr    = f_ref.sin();

      auto x = tax::TE< N >::variable( 0.0 );
      auto f = prepareInput( x );
      tax::TE< N > y = tax::sin( f );

      EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
  }
  ```

  For the remaining ops, replicate this template:

  ```cpp
  TEST( DaceUnivariate, <Name> )
  {
      constexpr int N = 40;
      DACE::DA::init( N, 1 );
      DACE::DA xr( 1 );
      DACE::DA f_ref = prepareInput( xr );
      DACE::DA yr    = <DACE expression on f_ref>;

      auto x = tax::TE< N >::variable( <x0> );
      auto f = prepareInput( x );
      tax::TE< N > y = <tax expression on f>;

      EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
  }
  ```

  Specifically:
  - `Cos` / `Tan` / `Exp` / `Sinh` / `Cosh` / `Tanh` / `Erf` — direct method
    call on `f_ref` / free function on `f`. `x0 = 0.0`.
  - `Log` / `Sqrt` / `Cbrt` / `Reciprocal` — prep yields ∈ [1, 1.5] so
    these are safe.  Same `x0 = 1.0`.
  - `Square` / `Cube` / `Pow(int)` / `Pow(real)` — same.  For
    `pow(f, 3)` use both DACE's `.pow(3)` and tax's `tax::pow(f, 3)`.
  - `Asin` / `Acos` / `Atan` / `Asinh` / `Atanh` — DACE methods. The prep
    range [1, 1.5] is outside `asin`/`acos`/`atanh` domains; scale the
    operand for those three: instead of `prep(x)` use
    `(prepareInput(x) - 1.0) / 0.5 * 0.4` (gives ∈ [0, 0.4], inside the
    valid ranges).  Apply the same scaling on both sides.
  - `Acosh` — prep range [1, 1.5] is valid (acosh needs ≥1).
  - `Atan2` — port the two-arg version; pass `prepareInput(x)` as both
    arguments (or one shifted).

- [ ] **Step 10.3:** Build:

  ```bash
  cmake --build build-reg -j
  ```

  Expected: clean build. Compile errors usually mean a tax operator name
  is slightly different from the DACE method name (e.g. `tax::log` vs
  `.log()`); check `include/tax/operators/math_unary.hpp` and
  `include/tax/operators/math_binary.hpp`.

- [ ] **Step 10.4:** Run only the univariate tests:

  ```bash
  ctest --test-dir build-reg -R test_regression_univariate --output-on-failure
  ```

  Expected: every `DaceUnivariate.*` test passes. If any test fails, the
  failure prints the offending coefficient (k, DACE value, tax value, diff).
  A non-zero diff at the threshold suggests either a real Stage 1
  regression (file it) or a numerical-condition issue (raise tolerance
  per-test only if justified; never relax the default).

### Task 11: Port `testMultivariate.cpp`

The pre-Stage-1 file is `tests/dace/testMultivariate.cpp` (~464 lines).
The same shape applies — port with `prepareInput` wrapping each input
variable. Multivariate uses `tax::TEn<N, M>` and the new Eigen-form
`variables` factory.

Two API differences from the pre-Stage-1 multivariate code:

1. **Structured bindings are gone.** The old code wrote
   `auto [x, y, z] = tax::TEn<N,3>::variables({1.0, 2.0, 3.0})`. The new
   factory `tax::variables<tax::TEn<N,3>>(x0)` returns
   `Eigen::Matrix<tax::TEn<N,3>, 3, 1>`. Access elements with `v(0)`, `v(1)`,
   `v(2)`. See `include/tax/eigen.hpp:99` and the spec.
2. **DACE variables are 1-based**, tax indices are 0-based. The old test
   already handles this; keep the convention.

**Files:**
- Modify (overwrite the placeholder from step 9.3):
  `tests/regression/testMultivariate.cpp`

- [ ] **Step 11.1:** Fetch the original file:

  ```bash
  git show claude/add-verner-integrators-vEgRF:tests/dace/testMultivariate.cpp \
      > /tmp/orig_testMultivariate.cpp
  ```

- [ ] **Step 11.2:** Overwrite `tests/regression/testMultivariate.cpp`. The
  representative head and one ported test:

  ```cpp
  // tests/regression/testMultivariate.cpp
  //
  // Ported from claude/add-verner-integrators-vEgRF:tests/dace/testMultivariate.cpp
  // and retargeted at tax::TEn<N, M> with the Eigen-form variables factory.
  // Every input variable is wrapped in tax_regression::prepareInput before
  // the op under test, on both sides.

  #include <dace/dace.h>
  #include <gtest/gtest.h>

  #include <Eigen/Core>
  #include <tax/tax.hpp>

  #include "regressionUtils.hpp"

  using tax_regression::expectCoeffsMatch;
  using tax_regression::prepareInput;

  TEST( DaceMultivariate, Div )
  {
      constexpr int N = 5;

      DACE::DA::init( N, 3 );
      DACE::DA dxr( 1 ), dyr( 2 ), dzr( 3 );
      auto xr = dxr + 1.0;
      auto yr = dyr + 2.0;
      auto zr = dzr + 3.0;
      const auto fxr = prepareInput( xr );
      const auto fyr = prepareInput( yr );
      const auto fzr = prepareInput( zr );
      DACE::DA vr = 1.0 / ( fxr * fyr * fzr );

      const Eigen::Vector3d x0{ 1.0, 2.0, 3.0 };
      auto v = tax::variables< tax::TEn< N, 3 > >( x0 );
      const auto fx = prepareInput( v( 0 ) );
      const auto fy = prepareInput( v( 1 ) );
      const auto fz = prepareInput( v( 2 ) );
      tax::TEn< N, 3 > result = 1.0 / ( fx * fy * fz );

      EXPECT_TRUE( expectCoeffsMatch( result, vr ) );
  }

  TEST( DaceMultivariate, Sin )
  {
      constexpr int N = 5;

      DACE::DA::init( N, 3 );
      DACE::DA dxr( 1 ), dyr( 2 ), dzr( 3 );
      auto xr = dxr + 1.0;
      auto yr = dyr + 1.0;
      auto zr = dzr + 1.0;
      DACE::DA vr = ( prepareInput( xr ) * prepareInput( yr ) / prepareInput( zr ) ).sin();

      const Eigen::Vector3d x0{ 1.0, 1.0, 1.0 };
      auto v = tax::variables< tax::TEn< N, 3 > >( x0 );
      tax::TEn< N, 3 > result = tax::sin(
          prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) / prepareInput( v( 2 ) ) );

      EXPECT_TRUE( expectCoeffsMatch( result, vr ) );
  }
  ```

  Continue porting each remaining test from `/tmp/orig_testMultivariate.cpp`
  by:
  - Replace structured-binding `auto [x, y, z] = tax::TEn<N,3>::variables({...})`
    with `auto v = tax::variables<tax::TEn<N,3>>(Eigen::Vector3d{...}); auto x = v(0); ...`.
  - For every variable used in an expression, replace it with
    `prepareInput(<that variable>)` on both sides.
  - Where the original applied special pre-arithmetic (e.g. `xr + 1.0`),
    keep that pattern — `prepareInput` is applied to the result, so:
    `prepareInput(xr + 1.0)` mirrors `prepareInput(v(0))` on the tax side
    when the tax variable was created at `x0 = 1.0` (the addition is
    constant-shift; the per-side semantics line up).
  - Some tests use particular evaluation patterns
    (e.g. `(x*y/z).sin()`). Wrap each leaf variable in `prepareInput` then
    keep the surrounding expression.

- [ ] **Step 11.3:** Build:

  ```bash
  cmake --build build-reg -j
  ```

  Expected: clean.

- [ ] **Step 11.4:** Run only the multivariate tests:

  ```bash
  ctest --test-dir build-reg -R test_regression_multivariate --output-on-failure
  ```

  Expected: every `DaceMultivariate.*` test passes.

### Task 12: Verify slice 2 and commit

- [ ] **Step 12.1:** Run the entire regression suite:

  ```bash
  ctest --test-dir build-reg --output-on-failure
  ```

  Expected: `test_regression_univariate` + `test_regression_multivariate`
  both pass.

- [ ] **Step 12.2:** Confirm the unit-test build path still works
  unchanged:

  ```bash
  cmake --build build -j && ctest --test-dir build --output-on-failure
  ```

  Expected: green.

- [ ] **Step 12.3:** Stage and commit:

  ```bash
  git add CMakeLists.txt \
          tests/regression/CMakeLists.txt \
          tests/regression/regressionUtils.hpp \
          tests/regression/testUnivariate.cpp \
          tests/regression/testMultivariate.cpp
  git rm tests/regression/testSentinel.cpp
  git commit -m "$(cat <<'EOF'
  stage3-slice2: DACE FetchContent + ported testUnivariate/testMultivariate

  Wires DACE v2.1.0 via FetchContent under TAX_BUILD_REGRESSIONS and ports
  the two surviving pre-Stage-1 DACE tests retargeted at tax::TE<N> and
  tax::TEn<N, M> with the Stage 1 Eigen-form variables factory. Every input
  variable is wrapped in tax_regression::prepareInput (default:
  1 + 0.5 * sin(x)^2) on both sides so the operators are exercised against
  polynomials with non-trivial structure rather than the bare variable(x0).
  The slice 1 sentinel test is removed.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Slice 3 — Expand coverage to the rest of the Stage 1 dense surface

Goal of slice 3: end with regression tests covering everything Stage 1
added or formalised beyond plain math ops — symbolic derivative/integral,
displacement evaluation, gradient/hessian/jacobian extraction, polynomial
map inversion.

### Task 13: `testDerivInteg.cpp`

Stage 1 exposes `f.deriv<I>()` / `f.deriv(int)` and `f.integ<I>()` /
`f.integ(int)`. DACE exposes `da.deriv(unsigned int i)` and
`da.integ(unsigned int i)` (1-based). Compare both univariate and
multivariate cases.

**Files:**
- Create: `tests/regression/testDerivInteg.cpp`
- Modify: `tests/regression/CMakeLists.txt` — register the new test

- [ ] **Step 13.1:** Create `tests/regression/testDerivInteg.cpp`:

  ```cpp
  // tests/regression/testDerivInteg.cpp
  //
  // DACE-vs-tax parity for symbolic differentiation and integration of
  // Taylor expansions: f.deriv<I>() / f.integ<I>() against
  // DACE::DA::deriv / DACE::DA::integ.

  #include <dace/dace.h>
  #include <gtest/gtest.h>

  #include <Eigen/Core>
  #include <tax/tax.hpp>

  #include "regressionUtils.hpp"

  using tax_regression::expectCoeffsMatch;
  using tax_regression::prepareInput;

  TEST( DaceDerivInteg, UnivariateDeriv )
  {
      constexpr int N = 20;
      DACE::DA::init( N, 1 );

      DACE::DA xr( 1 );
      DACE::DA f_ref = prepareInput( xr ).sin();
      DACE::DA df_ref = f_ref.deriv( 1 );  // 1-based: derivative w.r.t. variable 1

      auto x = tax::TE< N >::variable( 0.0 );
      auto f = tax::sin( prepareInput( x ) );
      tax::TE< N > df = f.deriv< 0 >();

      EXPECT_TRUE( expectCoeffsMatch( df, df_ref ) );
  }

  TEST( DaceDerivInteg, UnivariateInteg )
  {
      constexpr int N = 20;
      DACE::DA::init( N, 1 );

      DACE::DA xr( 1 );
      DACE::DA f_ref = prepareInput( xr ).cos();
      DACE::DA F_ref = f_ref.integ( 1 );

      auto x = tax::TE< N >::variable( 0.0 );
      auto f = tax::cos( prepareInput( x ) );
      tax::TE< N > F = f.integ< 0 >();

      EXPECT_TRUE( expectCoeffsMatch( F, F_ref ) );
  }

  TEST( DaceDerivInteg, MultivariateDerivX )
  {
      constexpr int N = 6;
      DACE::DA::init( N, 2 );

      DACE::DA dxr( 1 ), dyr( 2 );
      DACE::DA fr  = ( prepareInput( dxr ) * prepareInput( dyr ) ).exp();
      DACE::DA dfr = fr.deriv( 1 );  // w.r.t. variable 1

      const Eigen::Vector2d x0{ 0.0, 0.0 };
      auto v = tax::variables< tax::TEn< N, 2 > >( x0 );
      auto f  = tax::exp( prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) );
      tax::TEn< N, 2 > df = f.deriv< 0 >();

      EXPECT_TRUE( expectCoeffsMatch( df, dfr ) );
  }

  TEST( DaceDerivInteg, MultivariateDerivY )
  {
      constexpr int N = 6;
      DACE::DA::init( N, 2 );

      DACE::DA dxr( 1 ), dyr( 2 );
      DACE::DA fr  = ( prepareInput( dxr ) * prepareInput( dyr ) ).exp();
      DACE::DA dfr = fr.deriv( 2 );

      const Eigen::Vector2d x0{ 0.0, 0.0 };
      auto v = tax::variables< tax::TEn< N, 2 > >( x0 );
      auto f  = tax::exp( prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) );
      tax::TEn< N, 2 > df = f.deriv< 1 >();

      EXPECT_TRUE( expectCoeffsMatch( df, dfr ) );
  }

  TEST( DaceDerivInteg, MultivariateInteg )
  {
      constexpr int N = 6;
      DACE::DA::init( N, 2 );

      DACE::DA dxr( 1 ), dyr( 2 );
      DACE::DA fr = ( prepareInput( dxr ) + prepareInput( dyr ) ).cos();
      DACE::DA Fr = fr.integ( 1 );

      const Eigen::Vector2d x0{ 0.0, 0.0 };
      auto v = tax::variables< tax::TEn< N, 2 > >( x0 );
      auto f = tax::cos( prepareInput( v( 0 ) ) + prepareInput( v( 1 ) ) );
      tax::TEn< N, 2 > F = f.integ< 0 >();

      EXPECT_TRUE( expectCoeffsMatch( F, Fr ) );
  }
  ```

- [ ] **Step 13.2:** Register the new test. Edit
  `tests/regression/CMakeLists.txt` and add inside the existing
  registration block:

  ```cmake
  tax_add_regression(test_regression_deriv_integ SOURCES testDerivInteg.cpp)
  ```

- [ ] **Step 13.3:** Build and run:

  ```bash
  cmake --build build-reg -j
  ctest --test-dir build-reg -R test_regression_deriv_integ --output-on-failure
  ```

  Expected: all 5 tests pass.

### Task 14: `testEval.cpp`

Stage 1 exposes `f.eval(Eigen::Matrix<T, M, 1>)` for evaluation at a
displacement. DACE evaluates a `DA` by calling
`da.eval(std::vector<double>)` (1-based; equivalent input vector).
Compare scalar outputs.

**Files:**
- Create: `tests/regression/testEval.cpp`
- Modify: `tests/regression/CMakeLists.txt`

- [ ] **Step 14.1:** Create `tests/regression/testEval.cpp`:

  ```cpp
  // tests/regression/testEval.cpp
  //
  // DACE-vs-tax parity for displacement evaluation:
  //   tax::TE<N>::eval(dx)        vs  DACE::DA::eval(...)
  //   tax::TEn<N,M>::eval(dx)     vs  DACE::DA::eval(...)

  #include <dace/dace.h>
  #include <gtest/gtest.h>

  #include <Eigen/Core>
  #include <cmath>
  #include <tax/tax.hpp>

  #include "regressionUtils.hpp"

  using tax_regression::prepareInput;

  TEST( DaceEval, UnivariateAtDx )
  {
      constexpr int N = 20;
      DACE::DA::init( N, 1 );

      DACE::DA xr( 1 );
      DACE::DA fr = prepareInput( xr ).sin();

      auto x = tax::TE< N >::variable( 0.0 );
      auto f = tax::sin( prepareInput( x ) );

      for ( const double dx : { -0.3, -0.1, 0.0, 0.1, 0.3 } )
      {
          const double ref = fr.eval( std::vector< double >{ dx } );
          const double got = f.eval( Eigen::Matrix< double, 1, 1 >( dx ) );
          EXPECT_NEAR( ref, got, 1e-12 ) << "dx=" << dx;
      }
  }

  TEST( DaceEval, MultivariateAtDx )
  {
      constexpr int N = 6;
      DACE::DA::init( N, 2 );

      DACE::DA dxr( 1 ), dyr( 2 );
      DACE::DA fr = ( prepareInput( dxr ) + prepareInput( dyr ) ).exp();

      const Eigen::Vector2d x0{ 0.0, 0.0 };
      auto v = tax::variables< tax::TEn< N, 2 > >( x0 );
      auto f = tax::exp( prepareInput( v( 0 ) ) + prepareInput( v( 1 ) ) );

      const Eigen::Vector2d displacements[] = {
          { -0.2, -0.2 }, { -0.1, 0.1 }, { 0.0, 0.0 }, { 0.1, -0.1 }, { 0.2, 0.2 }
      };
      for ( const auto& dx : displacements )
      {
          const double ref = fr.eval( std::vector< double >{ dx( 0 ), dx( 1 ) } );
          const double got = f.eval( dx );
          EXPECT_NEAR( ref, got, 1e-12 ) << "dx=" << dx.transpose();
      }
  }
  ```

- [ ] **Step 14.2:** Register the new test. Edit
  `tests/regression/CMakeLists.txt`:

  ```cmake
  tax_add_regression(test_regression_eval SOURCES testEval.cpp)
  ```

- [ ] **Step 14.3:** Build and run:

  ```bash
  cmake --build build-reg -j
  ctest --test-dir build-reg -R test_regression_eval --output-on-failure
  ```

  Expected: green.

### Task 15: `testEigenVectorCalc.cpp` (gradient, hessian, jacobian)

Stage 1 exposes:
- `tax::gradient(f) → Eigen::Matrix<T, M, 1>` for scalar f.
- `tax::hessian(f) → Eigen::Matrix<T, M, M>` for scalar f.
- `tax::jacobian(F) → Eigen::Matrix<T, K, M>` for vector F (Eigen vector
  of TE).

DACE returns each entry via `getCoefficient` for the corresponding
first-/second-order unit multi-index. Compare component-wise.

**Files:**
- Create: `tests/regression/testEigenVectorCalc.cpp`
- Modify: `tests/regression/CMakeLists.txt`

- [ ] **Step 15.1:** Create `tests/regression/testEigenVectorCalc.cpp`:

  ```cpp
  // tests/regression/testEigenVectorCalc.cpp
  //
  // DACE-vs-tax parity for vector-calculus extractors built on top of the
  // Stage 1 Eigen helpers:
  //   tax::gradient(f)   vs  DACE getCoefficient at unit multi-indices
  //   tax::hessian(f)    vs  DACE getCoefficient at 2nd-order multi-indices,
  //                          with symmetry handling (2 for off-diagonal terms)
  //   tax::jacobian(F)   vs  the row-wise gradient of each component

  #include <dace/dace.h>
  #include <gtest/gtest.h>

  #include <Eigen/Core>
  #include <tax/tax.hpp>

  #include "regressionUtils.hpp"

  using tax_regression::prepareInput;

  namespace
  {
  // DACE coefficient at a unit multi-index for variable i (0-based here).
  double daceCoeff1( const DACE::DA& f, int M, int i )
  {
      std::vector< unsigned int > idx( std::size_t( M ), 0u );
      idx[std::size_t( i )] = 1u;
      return f.getCoefficient( idx );
  }
  // Raw (un-scaled) DACE coefficient at second-order multi-index.
  double daceCoeff2( const DACE::DA& f, int M, int i, int j )
  {
      std::vector< unsigned int > idx( std::size_t( M ), 0u );
      idx[std::size_t( i )] += 1u;
      idx[std::size_t( j )] += 1u;
      return f.getCoefficient( idx );
  }
  }  // namespace

  TEST( DaceVectorCalc, GradientMatchesDace )
  {
      constexpr int N = 5;
      constexpr int M = 3;
      DACE::DA::init( N, M );

      DACE::DA d1( 1 ), d2( 2 ), d3( 3 );
      DACE::DA fr = ( prepareInput( d1 ) * prepareInput( d2 ) + prepareInput( d3 ) ).cos();

      const Eigen::Vector3d x0{ 0.0, 0.0, 0.0 };
      auto v = tax::variables< tax::TEn< N, M > >( x0 );
      tax::TEn< N, M > f =
          tax::cos( prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) + prepareInput( v( 2 ) ) );

      Eigen::Matrix< double, M, 1 > g = tax::gradient( f );
      for ( int i = 0; i < M; ++i )
      {
          const double ref = daceCoeff1( fr, M, i );          // = ∂f/∂x_i at x0
          EXPECT_NEAR( g( i ), ref, 1e-12 ) << "i=" << i;
      }
  }

  TEST( DaceVectorCalc, HessianMatchesDace )
  {
      constexpr int N = 5;
      constexpr int M = 3;
      DACE::DA::init( N, M );

      DACE::DA d1( 1 ), d2( 2 ), d3( 3 );
      DACE::DA fr = prepareInput( d1 ) * prepareInput( d2 ) * prepareInput( d3 );

      const Eigen::Vector3d x0{ 0.0, 0.0, 0.0 };
      auto v = tax::variables< tax::TEn< N, M > >( x0 );
      tax::TEn< N, M > f = prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) * prepareInput( v( 2 ) );

      Eigen::Matrix< double, M, M > H = tax::hessian( f );
      for ( int i = 0; i < M; ++i )
      {
          for ( int j = 0; j < M; ++j )
          {
              // DACE stores raw Taylor coefficients; ∂^2 f / ∂x_i ∂x_j
              //   = (i == j) ? 2 * c_2ei : c_{ei + ej}
              const double raw = daceCoeff2( fr, M, i, j );
              const double ref = ( i == j ) ? 2.0 * raw : raw;
              EXPECT_NEAR( H( i, j ), ref, 1e-12 ) << "i=" << i << " j=" << j;
          }
      }
  }

  TEST( DaceVectorCalc, JacobianMatchesDace )
  {
      constexpr int N = 4;
      constexpr int M = 2;
      DACE::DA::init( N, M );

      DACE::DA d1( 1 ), d2( 2 );
      DACE::DA fr0 = ( prepareInput( d1 ) + prepareInput( d2 ) ).sin();
      DACE::DA fr1 = ( prepareInput( d1 ) * prepareInput( d2 ) ).cos();

      const Eigen::Vector2d x0{ 0.0, 0.0 };
      auto v = tax::variables< tax::TEn< N, M > >( x0 );
      using TE = tax::TEn< N, M >;
      Eigen::Matrix< TE, 2, 1 > F;
      F( 0 ) = tax::sin( prepareInput( v( 0 ) ) + prepareInput( v( 1 ) ) );
      F( 1 ) = tax::cos( prepareInput( v( 0 ) ) * prepareInput( v( 1 ) ) );

      Eigen::Matrix< double, 2, M > J = tax::jacobian( F );
      const DACE::DA refs[2] = { fr0, fr1 };
      for ( int row = 0; row < 2; ++row )
      {
          for ( int col = 0; col < M; ++col )
          {
              const double ref = daceCoeff1( refs[row], M, col );
              EXPECT_NEAR( J( row, col ), ref, 1e-12 ) << "row=" << row << " col=" << col;
          }
      }
  }
  ```

- [ ] **Step 15.2:** Register the new test:

  ```cmake
  tax_add_regression(test_regression_vector_calc SOURCES testEigenVectorCalc.cpp)
  ```

- [ ] **Step 15.3:** Build and run:

  ```bash
  cmake --build build-reg -j
  ctest --test-dir build-reg -R test_regression_vector_calc --output-on-failure
  ```

  Expected: green. If the hessian test fails on off-diagonal entries, the
  factor-of-2 convention may need adjustment — DACE stores raw Taylor
  coefficients, tax's `hessian()` returns derivatives, so off-diagonal
  `∂² / ∂x_i ∂x_j = c_{e_i + e_j}` (no factor 2) and diagonal
  `∂² / ∂x_i² = 2 c_{2 e_i}`. The code in step 15.1 already encodes that.

### Task 16: `testInvert.cpp`

Stage 1 exposes `tax::invert(F) → Eigen::Matrix<TE, M, 1>` for a square
polynomial map represented as an Eigen vector of TE. DACE provides the
same via `DACE::AlgebraicVector<DA>::invert()` (from
`<dace/AlgebraicVector.h>`). Compare the coefficient arrays of the
inverted map component-wise.

**Files:**
- Create: `tests/regression/testInvert.cpp`
- Modify: `tests/regression/CMakeLists.txt`

- [ ] **Step 16.1:** Create `tests/regression/testInvert.cpp`:

  ```cpp
  // tests/regression/testInvert.cpp
  //
  // DACE-vs-tax parity for polynomial map inversion:
  //   tax::invert(F)                      (Eigen<TE,M,1>)
  //   DACE::AlgebraicVector<DA>::invert() (vector of DA)

  #include <dace/AlgebraicVector.h>
  #include <dace/dace.h>
  #include <gtest/gtest.h>

  #include <Eigen/Core>
  #include <tax/tax.hpp>

  #include "regressionUtils.hpp"

  using tax_regression::expectCoeffsMatch;
  using tax_regression::prepareInput;

  TEST( DaceInvert, IdentityPlusPerturbation )
  {
      constexpr int N = 5;
      constexpr int M = 2;
      DACE::DA::init( N, M );

      DACE::DA d1( 1 ), d2( 2 );
      DACE::AlgebraicVector< DACE::DA > Fr( 2 );
      // Identity + small nonlinear perturbation; linear part is invertible.
      Fr[0] = d1 + 0.1 * ( prepareInput( d2 ) - 1.0 );
      Fr[1] = d2 + 0.1 * ( prepareInput( d1 ) - 1.0 );
      DACE::AlgebraicVector< DACE::DA > Fr_inv = Fr.invert();

      const Eigen::Vector2d x0{ 0.0, 0.0 };
      auto v = tax::variables< tax::TEn< N, M > >( x0 );
      using TE = tax::TEn< N, M >;
      Eigen::Matrix< TE, 2, 1 > F;
      F( 0 ) = v( 0 ) + 0.1 * ( prepareInput( v( 1 ) ) - 1.0 );
      F( 1 ) = v( 1 ) + 0.1 * ( prepareInput( v( 0 ) ) - 1.0 );
      Eigen::Matrix< TE, 2, 1 > F_inv = tax::invert( F );

      for ( int i = 0; i < 2; ++i )
      {
          EXPECT_TRUE( expectCoeffsMatch( F_inv( i ), Fr_inv[i] ) )
              << "inverse component " << i;
      }
  }
  ```

  Notes:
  - The map must have an invertible linear part (`tax::invert` throws
    otherwise — see `include/tax/eigen.hpp:381`). The identity-plus-small-
    perturbation form is the simplest such case.
  - DACE's `AlgebraicVector` is included via `<dace/AlgebraicVector.h>`.
  - DACE indexing is 1-based on the DA side; tax uses 0-based Eigen
    indexing.

- [ ] **Step 16.2:** Register the new test:

  ```cmake
  tax_add_regression(test_regression_invert SOURCES testInvert.cpp)
  ```

- [ ] **Step 16.3:** Build and run:

  ```bash
  cmake --build build-reg -j
  ctest --test-dir build-reg -R test_regression_invert --output-on-failure
  ```

  Expected: green.

### Task 17: Verify slice 3 and commit

- [ ] **Step 17.1:** Run the full regression suite:

  ```bash
  ctest --test-dir build-reg --output-on-failure
  ```

  Expected: 6 regression test executables pass
  (`test_regression_univariate`, `_multivariate`, `_deriv_integ`, `_eval`,
  `_vector_calc`, `_invert`).

- [ ] **Step 17.2:** Stage and commit:

  ```bash
  git add tests/regression/CMakeLists.txt \
          tests/regression/testDerivInteg.cpp \
          tests/regression/testEval.cpp \
          tests/regression/testEigenVectorCalc.cpp \
          tests/regression/testInvert.cpp
  git commit -m "$(cat <<'EOF'
  stage3-slice3: expand DACE regressions to deriv/integ/eval/grad/hess/jac/invert

  Adds parity tests for the rest of the Stage 1 dense surface beyond plain
  math ops: symbolic differentiation/integration, displacement evaluation,
  gradient/hessian/jacobian extraction via the Eigen helpers, and polynomial
  map inversion via tax::invert vs DACE::AlgebraicVector::invert.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Slice 4 — `regressions.yml` CI workflow

Goal of slice 4: end with a manual-dispatch CI workflow that fetches DACE,
configures with `TAX_BUILD_REGRESSIONS=ON`, builds, and runs ctest.

### Task 18: Create `.github/workflows/regressions.yml`

**Files:**
- Create: `.github/workflows/regressions.yml`

- [ ] **Step 18.1:** Create `.github/workflows/regressions.yml`:

  ```yaml
  name: Regressions

  on:
    workflow_dispatch:

  jobs:
    regressions:
      name: regressions (ubuntu / gcc / Release)
      runs-on: ubuntu-latest
      env:
        CC: gcc
        CXX: g++

      steps:
        - name: Checkout
          uses: actions/checkout@v4

        - name: Setup CMake
          uses: jwlawson/actions-setup-cmake@v2
          with:
            cmake-version: "4.2.x"

        - name: Install toolchain and Eigen
          run: |
            sudo apt-get update
            sudo apt-get install -y g++ libeigen3-dev ninja-build

        - name: Toolchain info
          run: |
            cmake --version
            "$CC" --version
            "$CXX" --version

        - name: Configure
          run: >
            cmake -S . -B build
            -DCMAKE_BUILD_TYPE=Release
            -DTAX_BUILD_UNITTESTS=OFF
            -DTAX_BUILD_REGRESSIONS=ON
            -DTAX_BUILD_BENCHMARK=OFF
            -G Ninja

        - name: Build
          run: cmake --build build -j 2

        - name: Test
          run: ctest --test-dir build --output-on-failure
  ```

- [ ] **Step 18.2:** Stage and commit:

  ```bash
  git add .github/workflows/regressions.yml
  git commit -m "$(cat <<'EOF'
  stage3-slice4: add regressions.yml CI workflow (workflow_dispatch)

  Manual-dispatch-only workflow that builds tax with TAX_BUILD_REGRESSIONS=ON
  on Ubuntu + GCC, fetches DACE v2.1.0 via FetchContent, and runs ctest.
  Mirrors the shape of sanitizers.yml.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

- [ ] **Step 18.3:** Trigger the workflow manually once to verify it runs
  green. Either through the GitHub UI (Actions → Regressions → Run
  workflow) or via:

  ```bash
  gh workflow run regressions.yml --ref <current-branch>
  gh run watch
  ```

  Expected: a single successful run.

---

## Self-Review

**Spec coverage check (against
`docs/superpowers/specs/2026-05-19-tax-stage3-design.md`):**

- Decision: `TAX_BUILD_TEST` → `TAX_BUILD_UNITTESTS` rename — Tasks 1–3.
- Decision: `TAX_BUILD_REGRESSIONS` flag default OFF — Task 4.
- Decision: tests/regression/ added as sibling subdir — Task 5.
- Decision: single flag fetches DACE itself — Task 7.
- Decision: shared `prepareInput` pre-step — Task 9 (with default
  `1 + 0.5 * sin(x)^2`, settled in step 9.4).
- Decision: port the two surviving pre-Stage-1 files — Tasks 10, 11.
- Decision: testNorm dropped — covered by the spec edit; not represented
  as a task (intentional).
- Decision: expand to deriv/integ, eval, gradient/hessian/jacobian,
  invert — Tasks 13, 14, 15, 16.
- Decision: CI workflow `workflow_dispatch` only — Task 18.
- Decision: source of truth for ported code is branch
  `claude/add-verner-integrators-vEgRF` — referenced in steps 10.1 and
  11.1.

**Placeholder scan:** no TBDs or vague steps; every code block is complete.
The only "settle in slice 2" item is the final `prepareInput` choice
(step 9.4), which has both a default and a documented fallback so the
implementer can decide concretely.

**Type / signature consistency check:**

- `tax::TE<N>` and `tax::TEn<N, M>` used consistently across all tests.
- `tax::variables<TE>(Eigen-vector)` factory used consistently in
  multivariate tests (not the pre-Stage-1 structured-binding form).
- `expectCoeffsMatch(tax, DACE::DA, tol=1e-12)` signature is identical in
  univariate and multivariate overloads in `regressionUtils.hpp` and is
  matched at every call site.
- `prepareInput` exists for `tax::TE<N>`, `tax::TEn<N, M>`, and
  `DACE::DA`. Same default body in all three.
- DACE method names (`.sin()`, `.cos()`, `.exp()`, etc.) used as free
  methods on DA; tax math wrapped as free functions in `tax::`
  (`tax::sin(f)`, `tax::cos(f)`, etc.).
- `tax_add_regression()` defined in `tests/regression/CMakeLists.txt` and
  used by every registered test in slices 2 and 3.

No inconsistencies found.

---

**Plan complete.** Ready for execution.
