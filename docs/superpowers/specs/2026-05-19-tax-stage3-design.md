# tax — Stage 3: DACE regression suite + test flag rename — Design

**Status:** draft for approval
**Date:** 2026-05-19
**Depends on:** Stage 1 (merged on `main`)

## Goal

Add a coefficient-parity regression suite that validates the Stage 1 dense
surface against DACE (the reference Differential Algebra library), gated by a
new `TAX_BUILD_REGRESSIONS` CMake option. Rename `TAX_BUILD_TEST` →
`TAX_BUILD_UNITTESTS` for clarity, since the project now has two distinct
test categories.

Stage 3 is an infrastructure / quality-assurance stage. It does not change
the public API, the kernels, or the library code itself.

## Out of scope

- Sparse-vs-DACE parity. DACE has no sparse representation; agreement between
  sparse `STE` and dense `TE` is already a Stage 1 unit-test invariant.
- DACE-comparison benchmarks (`bench_vs_dace` from the pre-Stage-1 codebase).
  Can return later as an independent benchmark opt-in.
- ODE / ADS / dynamic-shape `TaylorExpansion` / Python bindings. Still
  deferred to later stages.

## Decisions log

These are the choices settled during brainstorming. Load-bearing for
everything below.

| Axis | Choice | Rationale |
|---|---|---|
| Test scope | Port the two pre-Stage-1 files that target the Stage 1 surface (univariate, multivariate) + expand to cover the rest of the Stage 1 dense surface (every math kernel, deriv/integ, eval, gradient/hessian/jacobian, invert) | Port establishes parity with the proven baseline; expansion covers everything Stage 1 added or formalised. The pre-Stage-1 `testNorm.cpp` is dropped — it tested `coeffsNormInf` / `coeffsNorm` / `coeffsNormEstimate`, methods that do not exist on the Stage 1 dense surface. |
| Sparse coverage | Out of scope | DACE has no sparse counterpart; dense↔sparse agreement is already a Stage 1 unit-test invariant. |
| DACE provisioning | Single flag `TAX_BUILD_REGRESSIONS`, fetches DACE v2.1.0 via FetchContent | Same provisioning that worked pre-Stage-1; simplest mental model. |
| CI policy | On-demand only — new `regressions.yml`, `workflow_dispatch` trigger | DACE build is heavy and parity is a numerical-correctness check rather than a fast-feedback signal; mirrors the `sanitizers.yml` shape. |
| Tests directory layout | Keep current layout; add `tests/regression/` as a sibling of `core/ kernels/ operators/ sparse/ eigen/` | Smallest diff; preserves existing CMake target list. |
| Flag rename | `TAX_BUILD_TEST` → `TAX_BUILD_UNITTESTS` (default ON); `TAX_BUILD_REGRESSIONS` default OFF | Disambiguates the two test categories; regression suite is opt-in because it pulls DACE. |
| Source for ported code | Branch `claude/add-verner-integrators-vEgRF` | Reference snapshot of the pre-Stage-1 implementation, including the original DACE tests. |
| Test-input fixture | Every test runs against a shared pre-step that turns `variable(x0)` into a polynomial with non-trivial / sparse structure (many zero coefficients) before applying the op under test | The pre-Stage-1 tests fed every operator the trivial input `variable(x0)`, which has only two nonzero coefficients. That exercises far less of the sparse-aware arithmetic than realistic call sites do. A consistent prep step across tests gives each operator a richer input without ballooning the test count. |

## CMake surface

After Stage 3, the root `CMakeLists.txt` exposes:

```cmake
option(TAX_BUILD_UNITTESTS   "Build and enable unit tests"                  ON)
option(TAX_BUILD_REGRESSIONS "Build DACE-based regression tests"            OFF)
option(TAX_BUILD_BENCHMARK   "Build benchmark suite (Google Benchmark)"     OFF)
option(TAX_USE_UNROLL        "Use compile-time-unrolled M=1 Cauchy kernels" ON)
option(TAX_USE_STENCIL       "Use precomputed Cauchy stencils for M>=2"     ON)
```

`TAX_BUILD_REGRESSIONS=ON` triggers a `FetchContent_Declare(DACE GIT_TAG v2.1.0)`
block at the root `CMakeLists.txt`, exposes the DACE target via the usual
alias dance (`dace::dace_s` preferred, falling back to `dace::dace` / `dace_s`
/ `dace`), then `add_subdirectory(tests/regression)`. The unit-test subdir is
added independently from `TAX_BUILD_UNITTESTS`. Both flags can be on
simultaneously; they share `enable_testing()` and `tests/testUtils.hpp` but
have separate CMake target lists.

## File layout

```
tests/
├── CMakeLists.txt                  # gates:
│                                   #   TAX_BUILD_UNITTESTS   → tax_add_test()
│                                   #   TAX_BUILD_REGRESSIONS → tax_add_regression()
├── testUtils.hpp                   # unchanged, shared
├── core/         kernels/  operators/  sparse/  eigen/   # unchanged
└── regression/
    ├── CMakeLists.txt              # registers regression executables
    ├── regressionUtils.hpp         # expectCoeffsMatch(tax, DACE::DA, tol=1e-12)
    │                               # + prepareInput(x) — shared pre-step
    ├── testUnivariate.cpp          # ported from add-verner-integrators, retargeted at tax::TE<N>
    ├── testMultivariate.cpp        # ported, retargeted at tax::TEn<N,M>
    ├── testDerivInteg.cpp          # new — DACE .deriv(i) / .integ(i) vs tax::TE::deriv / integ
    ├── testEval.cpp                # new — DACE evaluation vs tax::eval(f, dx)
    ├── testEigenVectorCalc.cpp     # new — gradient, hessian, jacobian via DACE coefficient pulls
    └── testInvert.cpp              # new — DACE invert() vs tax::invert()
```

`tax_add_regression(name SOURCES files...)` mirrors the existing
`tax_add_test()` helper but links the resolved DACE target alongside `tax`
and `GTest::gtest_main`. It is defined once in `tests/CMakeLists.txt` and is
only meaningful when `TAX_BUILD_REGRESSIONS=ON`.

## Test pattern

Adapted from the pre-Stage-1 code with two changes: the new Stage 1 type
names, and a shared pre-step that produces a polynomial with non-trivial
structure before the op under test. The shared helper does coefficient-wise
equality with `tol = 1e-12` (proven from the original suite):

```cpp
template <int N>
::testing::AssertionResult expectCoeffsMatch(const tax::TE<N>& tested,
                                             const DACE::DA& ref,
                                             double tol = 1e-12);

// Shared pre-step. Same operation applied on both sides of every test.
// Picked to produce many structural zeros (e.g. sin(x) at x=0 has only
// odd-degree coefficients), giving a richer input than bare variable(x0).
template <int N>           tax::TE<N>     prepareInput(const tax::TE<N>& x);
template <int N, int M>    tax::TEn<N,M>  prepareInput(const tax::TEn<N,M>& x);
                           DACE::DA       prepareInput(const DACE::DA&    x);

TEST(DaceUnivariate, Sin) {
    constexpr int N = 40;
    DACE::DA::init(N, 1);
    DACE::DA xr(1);
    DACE::DA f_ref = prepareInput(xr);
    DACE::DA yr    = f_ref.sin();

    auto x = tax::TE<N>::variable(0.0);
    auto f = prepareInput(x);
    auto y = tax::sin(f);

    EXPECT_TRUE(expectCoeffsMatch(y, yr));
}
```

The exact choice of `prepareInput` is settled in slice 2. Default candidate:
something like `tax::sin(x)` (univariate, x=0) and the analogous
`tax::sin(sum_i x_i)` (multivariate at origin), both of which produce
polynomials with ~50% structural zeros while remaining numerically benign
across the test surface. The choice must hold for every test in the suite —
if a specific op (e.g. `log`, which requires a positive constant term) does
not accept the default prep, the entire prep is shifted (e.g.
`1 + sin(x)^2`) rather than special-cased per test.

Multivariate uses `tax::TEn<N, M>` and the analogous DACE multi-index
coefficient pulls. The expanded files (`testDerivInteg`, `testEval`,
`testEigenVectorCalc`, `testInvert`) follow the same shape, calling
`prepareInput` once on each variable / vector entry before the op under
test.

`(N, M)` ranges follow the pre-Stage-1 defaults (univariate `N = 40`,
multivariate smaller — for example `N = 10, M = 3`) and can be tightened or
expanded inside the slice if any case proves unstable.

## CI

New `.github/workflows/regressions.yml`:

```yaml
on:
  workflow_dispatch:

jobs:
  regressions:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: install Eigen
        run: sudo apt-get update && sudo apt-get install -y libeigen3-dev
      - name: configure
        run: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
                   -DTAX_BUILD_UNITTESTS=OFF
                   -DTAX_BUILD_REGRESSIONS=ON
      - name: build
        run: cmake --build build -j
      - name: test
        run: ctest --test-dir build --output-on-failure
```

`tests.yml` and `sanitizers.yml` receive the option rename
(`-DTAX_BUILD_TEST=…` → `-DTAX_BUILD_UNITTESTS=…`) but are otherwise
untouched.

## Slice ordering

Each slice is one PR (or one merged commit). Workflow per slice: write
failing tests → implement → tests green → commit.

| # | Slice | Result |
|---|---|---|
| 1 | Flag rename + scaffold | Rename `TAX_BUILD_TEST` → `TAX_BUILD_UNITTESTS` everywhere (root CMake, `tests.yml`, `sanitizers.yml`, README, docs). Add `TAX_BUILD_REGRESSIONS` option with a no-op `tests/regression/CMakeLists.txt` and a single trivial regression test that does NOT link DACE — verifies wiring without paying the DACE fetch cost. CI green. |
| 2 | DACE provisioning + ported tests | Wire `FetchContent` (DACE v2.1.0) at the root CMake under `TAX_BUILD_REGRESSIONS`. Port `testUnivariate.cpp` and `testMultivariate.cpp` from `claude/add-verner-integrators-vEgRF`, retargeted at `tax::TE` / `tax::TEn` and rewritten to call `prepareInput` before every op. Add `regressionUtils.hpp` with `expectCoeffsMatch` and `prepareInput`. Settle the final choice of `prepareInput` here. Local `ctest` with `TAX_BUILD_REGRESSIONS=ON` green. |
| 3 | Expanded coverage | Add `testDerivInteg.cpp`, `testEval.cpp`, `testEigenVectorCalc.cpp`, `testInvert.cpp` covering everything Stage 1 added or formalised on the dense path. |
| 4 | CI workflow | Add `regressions.yml` (`workflow_dispatch` only). Verify a single manual run is green. |

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| DACE v2.1.0 build breaks on newer GCC/Clang in the Stage 1 toolchain | Pin a known-working compiler in `regressions.yml` (Ubuntu 22.04 default GCC). If upstream has a newer tag that builds cleanly, bump the `GIT_TAG` instead. |
| Coefficient ordering or sign conventions differ between `tax::TEn` and `DACE::DA` for some op | Caught in slice 2 against the ported tests, which previously passed against the old API. A new mismatch indicates a Stage 1 regression and is exactly what the suite exists to catch. |
| `FetchContent` of DACE is slow in CI | `workflow_dispatch`-only means cost is paid only when invoked. If it becomes annoying, configure a runner-side FetchContent cache. |
| `prepareInput` choice rejects a specific op (e.g. domain issue for `log`, `sqrt`, `asin`) | Shift the prep globally rather than special-case a single test. Candidate fallbacks documented in slice 2's PR; chosen prep must work for every op in the dense surface. |
| Slice 1's trivial regression test creates a `tests/regression/` subdir that builds even without DACE, but slice 2 then makes it require DACE | Slice 1's regression test does not include any DACE headers; slice 2 wires the DACE target *after* slice 1's wiring is verified. The trivial test is deleted in slice 2 once real ported tests replace it. |

## What does NOT exist at end of Stage 3

- No re-introduction of ODE / ADS / dynamic-shape / Python bindings.
- No DACE-against-benchmarks wiring (`TAX_BUILD_BENCHMARK=OFF` remains the
  default, and the benchmark subdir has no DACE target).
- No automatic gating on PRs.

## Open questions deferred to implementation

- Whether to gate the regression CI job's `actions/checkout` to fetch only
  the head commit (faster) or full history (needed if a future slice wants
  to diff against `main`). Default to head-only; revisit if needed.
- Whether `testEigenVectorCalc.cpp` is one file or three (gradient, hessian,
  jacobian). Decide at slice 3 based on file size.

---

## Implementation plan

To be produced by the `writing-plans` skill after this spec is approved.
