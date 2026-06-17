# Splitting `tax` into `tax` (core) + `tax-flow`

**Date:** 2026-06-17
**Status:** Approved design — pending implementation plan

## Goal

Split the current single project into two:

- **`tax`** — the core truncated-Taylor library: dense/sparse/named expansions,
  kernels, operators, and Eigen integration (`tax::la`). Stays in this repo at
  `/Users/andrea/Documents/Codes/tax`.
- **`tax-flow`** — the higher-level numerics built on top of `tax`: adaptive ODE
  integration (`tax::ode`) and Automatic Domain Splitting (`tax::ads`). Its own
  git repository (`git@github.com:andreapasquale94/tax-flow.git`), cloned to the
  sibling directory `/Users/andrea/Documents/Codes/tax-flow`. Independent from
  `tax`'s git history; consumes `tax` as an external dependency.

## Why the cut is clean

The header dependency graph is strictly layered with no upward edges:

| Layer | Modules | Depends on |
|-------|---------|-----------|
| Core (`tax`) | `core/`, `kernels/`, `operators/`, `la/` | each other only |
| Flow (`tax-flow`) | `ode/` | `core`, `la`, `operators` |
| Flow (`tax-flow`) | `ads/` | `ads`, `core`, `la`, `ode`, `operators` |

Verified:
- Nothing under `core/`, `kernels/`, `operators/`, `la/` includes `<tax/ode/...>`
  or `<tax/ads/...>`.
- The core unit-test suite (`tests/{core,kernels,operators,sparse,eigen}`) does
  not include `ode`/`ads`.
- The DACE regression suite includes only `<tax/tax.hpp>` and `<tax/la/...>` —
  it exercises **core** math, so it stays in `tax`.

## Decisions (locked)

1. **Topology:** `tax-flow` is its own git repo
   (`git@github.com:andreapasquale94/tax-flow.git`), cloned to the sibling dir
   `/Users/andrea/Documents/Codes/tax-flow`.
2. **Include paths & namespaces unchanged:** `tax-flow` keeps `<tax/ode.hpp>`,
   `<tax/ads.hpp>` and the `tax::ode` / `tax::ads` namespaces. Zero source churn
   in includes/namespaces. `tax-flow` installs its headers into the shared
   `tax/` include tree.
3. **Supporting material follows the code** (see inventory).
4. **`testUtils.hpp` is copied** into `tax-flow` (not installed/shared) — minor
   duplication accepted for independence.
5. **Graphify removed entirely from `tax`** — revert the merge that imported it
   (`git revert -m 1 03b68fd`), which cleanly removes `.graphify/`,
   `.claude/skills/graphify/`, the graphify hooks in `.claude/settings.json`,
   `scripts/graphify.sh`, `MEMORY.md`, and the graphify sections appended to
   `CLAUDE.md` and `.claude/CLAUDE.md`. `tax-flow` never gets graphify. (Both
   are regenerable with `/graphify` later if desired.)

## What moves to `tax-flow`

### Headers (42)
- `include/tax/ode.hpp` + `include/tax/ode/**` (28 files, incl. `detail/`,
  `steppers/`, and `ode/named.hpp` — the VectorOps adapter that lets a
  `tax::named` expansion act as an ODE state scalar).
- `include/tax/ads.hpp` + `include/tax/ads/**` (14 files).

### Tests
- `tests/ode/**` (incl. `tests/ode/CMakeLists.txt`, `events/`, `integrator/`,
  `problems/`, `steppers/`).
- `tests/ads/**` (10 files).
- **Copy** `tests/testUtils.hpp` into `tax-flow/tests/`.

### Benchmarks
- All of `benchmarks/`: `bench_ode_cr3bp.cpp`, `bench_ads_refine.cpp`,
  `baseline/`, `CMakeLists.txt`.

### Examples
- All of `examples/`: `two_body/`, `three_body/`, `wsb/`, `common/`, `plot/`,
  `CMakeLists.txt` (every example is ODE/ADS-based).

### Docs
- `docs/ode/`, `docs/ads/`, `docs/benchmarks/`.

## What stays in `tax`

- Headers: `include/tax/{core,kernels,operators,la}/` + `tax.hpp` + `la.hpp`
  (including `core/named.hpp` and `la/named.hpp`).
- Tests: `tests/{core,kernels,operators,sparse,eigen}` + `tests/regression/` +
  the original `tests/testUtils.hpp` + `tests/CMakeLists.txt` (with `ode`/`ads`
  registrations removed).
- Docs: `docs/{core,eigen,internals,tutorials}`, `getting_started.md`,
  `index.md`, plus `javascripts/`.
- `cmake/taxConfig.cmake.in`, root install/export of `tax::tax`.

## How `tax-flow` consumes `tax`

`tax-flow/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.28)
project(tax-flow VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TAXFLOW_BUILD_UNITTESTS "Build unit tests"        ON)
option(TAXFLOW_BUILD_BENCHMARK "Build benchmark suite"   OFF)
option(TAXFLOW_BUILD_EXAMPLES  "Build example programs"  OFF)

# Find an installed tax, else fall back to the sibling source tree so a
# fresh checkout of both folders builds with no install step.
set(TAX_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tax"
    CACHE PATH "Path to tax source tree (used when tax is not installed)")
find_package(tax CONFIG QUIET)
if(NOT tax_FOUND)
    add_subdirectory("${TAX_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_tax")
endif()

add_library(tax-flow INTERFACE)
add_library(tax::flow ALIAS tax-flow)
target_compile_features(tax-flow INTERFACE cxx_std_23)
target_include_directories(tax-flow INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(tax-flow INTERFACE tax::tax)
```

Notes:
- When falling back via `add_subdirectory`, `tax`'s own
  `TAX_BUILD_UNITTESTS` defaults to `ON`; `tax-flow` sets it `OFF` before the
  `add_subdirectory` call so a flow build doesn't pull in core tests.
- CMake target/package name is `tax-flow` with alias `tax::flow` — keeps the
  `tax::` brand consistent with the unchanged C++ namespaces.
- `tax-flow` links `Threads` transitively through `tax::tax`; its package config
  (`tax-flowConfig.cmake.in`) must `find_dependency(tax)` (which pulls Eigen3).

### `tax-flow` test wiring
- `tax-flow/tests/CMakeLists.txt` reproduces the `tax_add_test` helper (renamed
  `taxflow_add_test` or kept as-is) and the GTest FetchContent block, linking
  `tax-flow` instead of `tax`.
- Re-register the `ode`/`ads` tests removed from `tax` (10 ADS targets +
  `add_subdirectory(ode)`), pointing at the copied sources.

## What changes in `tax`

1. Root `CMakeLists.txt`: remove `TAX_BUILD_BENCHMARK` and `TAX_BUILD_EXAMPLES`
   options and their `add_subdirectory(benchmarks|examples)` blocks. Keep the
   `tax` INTERFACE target, unit tests, regressions, and install/export intact —
   `tax-flow` depends on `find_package(tax)`.
2. `tests/CMakeLists.txt`: delete the ADS registrations (lines for
   `test_ads_*`) and `add_subdirectory(ode)`.
3. Delete the moved directories (`include/tax/{ode,ads}`, `tests/{ode,ads}`,
   `benchmarks/`, `examples/`, `docs/{ode,ads,benchmarks}`) from `tax`.
4. **Documentation refresh** (see below), including a rewritten `CLAUDE.md`.
5. **Remove graphify** via `git revert -m 1 03b68fd` (done first — see Migration
   mechanics — so the documentation refresh edits the clean, post-revert
   `CLAUDE.md`).

## Documentation refresh (in `tax`)

- **`CLAUDE.md`** (editing the post-revert, graphify-free version): trim the
  "ODE Integration" and "Automatic Domain Splitting" sections down to a short
  "Related project" pointer to `tax-flow`; update the Repository Structure tree
  (drop `ode/`, `ads/`, `examples/`, `benchmarks/`); fix the test count and the
  CMake options table (drop EXAMPLES/BENCHMARK). The `## graphify` section is
  already gone via the revert — confirm no graphify references remain.
- **`README.md`**: scope description to the core library; add a "tax-flow"
  pointer for ODE/ADS.
- **`docs/`** (MkDocs): remove `ode`/`ads`/`benchmarks` nav entries; ensure
  `getting_started`/`index` describe the core library only.
- `tax-flow` gets its own `README.md` and `CLAUDE.md` (focused on `ode`/`ads`,
  describing the `tax::tax` dependency and the find_package/sibling-source build).

## Migration mechanics

`tax-flow` is untracked by `tax`'s git, so the "move" is **copy-out + delete**:

0. **Remove graphify first:** `git revert -m 1 03b68fd` on `main`, restoring the
   clean pre-graphify `CLAUDE.md` / `.claude/` state. (Untracked files such as
   this spec and `.claude/workflows/` do not block the revert.) Done before the
   doc refresh so CLAUDE.md edits land on the clean version.
1. Create `/Users/andrea/Documents/Codes/tax-flow/{include/tax,tests,benchmarks,examples,docs}`.
2. Copy the moving files/dirs into `tax-flow` preserving relative layout
   (`include/tax/ode`, `include/tax/ads`, umbrellas at `include/tax/`).
3. Author `tax-flow` CMake (root, tests, benchmarks, examples) + config template
   + `README.md` + `CLAUDE.md`.
4. `git rm -r` the moved paths from `tax`; edit `tax` CMake + docs.
5. Verify both build & test independently (see below).

## Verification

- **tax (core):**
  `cmake -S /Users/andrea/Documents/Codes/tax -B build && cmake --build build -j && ctest --test-dir build --output-on-failure`
  — all remaining targets pass; no `ode`/`ads` targets present.
- **tax-flow:**
  `cmake -S /Users/andrea/Documents/Codes/tax-flow -B build -DTAXFLOW_BUILD_UNITTESTS=ON && cmake --build build -j && ctest --test-dir build --output-on-failure`
  — builds against the sibling `tax` source (no install) and all `ode`/`ads`
  tests pass.
- Optional: build `tax-flow` examples & benchmarks (`-DTAXFLOW_BUILD_EXAMPLES=ON
  -DTAXFLOW_BUILD_BENCHMARK=ON`).

## Out of scope (deferred)

- Promoting `tax-flow` to its own git repository / CI.
- Renaming to a `taxflow::` namespace or `<taxflow/...>` include paths.
- Python bindings (`pyproject.toml` remains forward-looking in `tax`).
- Sharing `testUtils.hpp` via an installed test-support target.
