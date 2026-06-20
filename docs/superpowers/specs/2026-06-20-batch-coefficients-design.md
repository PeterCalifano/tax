# Batch (SIMD) coefficients — design

**Date:** 2026-06-20
**Status:** Approved for planning
**Topic:** Port the prototype `tax::Batch<T,K>` vectorised-coefficient
type onto current `main`, with full math surface, `tax::la` support, tests, and docs.

## Motivation

`tax-flow`'s ADS `refine` currently propagates independent sub-boxes with multithreading.
A batched coefficient type lets one expansion evaluate `K` independent problem instances in
a single pass through the recurrence kernels, with the inner element-wise work vectorised.
That gives `refine` a data-parallel ("batching") alternative to thread-parallelism.

This task lands the **batching capability in the `tax` library only**. Rewiring `refine`
itself lives in the separate `tax-flow` repository and is a follow-up.

## Background

The capability was prototyped on branch `claude/taylor-expansion-prototypes-hxq111`, where it
shipped as a production type (`include/tax/core/batch.hpp`) with tests and docs. That branch
forked from `main` 38 commits ago (merge base `ca4c57b`). Since then `main` has:

- heavily rewritten `kernels/transcendental.hpp` and `operators/math_binary.hpp`,
- refactored the kernel layer (`recurrence_stencil`),
- split out `tax::ode` / `tax::ads` into the `tax-flow` companion.

So this is a **re-port** of the small, well-isolated batch-specific changes onto the current
code — not a cherry-pick. The ODE-related prototype changes (`ode/vector_ops.hpp`,
`bench_ode_batch.cpp`, `test_ode_batch_kepler.cpp`) are **out of scope**: they belong to
`tax-flow`.

## Core idea

`Batch<T,K>` packs `K` independent floating-point lanes into one coefficient slot, backed by
`Eigen::Array<T, K, 1, Eigen::DontAlign>`. Every operation a dense `TaylorExpansion` needs —
the four arithmetic ops, unary minus, equality (for the deriv/integ zero-skip), and the
transcendental seeds the kernels evaluate on the constant term — is element-wise across the
lanes and found by the kernels through ADL. Substituting `Batch<double,K>` for the scalar
coefficient type therefore makes

```cpp
TaylorExpansion<Batch<double,K>, N, M, storage::Dense>   // == TE<N, M, K>
```

run `K` independent expansions in lock-step: one pass through the kernels, `K` results,
bit-for-bit identical to `K` scalar runs. Restricted to **dense** storage — sparse storage
keys off exact-zero coefficients, which is not well-defined per-lane.

This works with no kernel rewrites because the dense kernels are already generic on the
coefficient type and reach their seeds via ADL. Two small enabling hooks are needed in the
core (below).

## Components / changes

### 1. `include/tax/core/concepts.hpp` (enabling traits)

Add an opt-in trait and a real-scalar projection; widen the `Scalar` concept to use them.
Keep current file formatting (do **not** import the prototype's unrelated `requires {` reflow).

```cpp
template < typename T >
struct is_tax_scalar : std::bool_constant< std::floating_point< T > > {};

template < typename T >
concept Scalar = is_tax_scalar< T >::value;          // strict superset of floating_point

template < typename T >
struct real_scalar { using type = T; };
template < typename T >
using real_scalar_t = typename real_scalar< T >::type;
```

Existing floating-point code is unaffected. `Batch` opts in by specialising both traits
(done in `batch.hpp`).

### 2. `include/tax/core/batch.hpp` (new file)

Ported essentially verbatim from the prototype (282 lines), already matching house style:

- `Batch<T,K>` struct: `Eigen::Array<T,K,1,DontAlign>` storage; default-zero ctor, implicit
  broadcast ctor `Batch(T)`, `fromLanes`, `operator[]`/`lane`, compound and binary `+ - * /`,
  unary `-`, lane-collapsing `==`/`!=`.
- Element-wise math via ADL: `sqrt exp log sin cos tan asin acos atan sinh cosh tanh asinh
  acosh atanh abs` (Eigen array methods); `pow` (Eigen `.pow`); `cbrt erf atan2` (per-lane
  `unaryExpr`/`binaryExpr` fallbacks Eigen-core lacks).
- Trait opt-ins: `is_tax_scalar<Batch>` → `true_type`, `real_scalar<Batch>::type = T`.
- Aliases: `Batchd<K>`, `Batchf<K>` only. **No `BatchTE`** — the batched expansion is reached
  through the unified `TE<N, M, K>` alias (§2b).
- `Eigen::NumTraits<Batch<T,K>>` specialisation so Eigen matrices can hold `Batch` (and thus
  `TaylorExpansion<Batch,...>`) as their scalar.

### 2b. `include/tax/core/taylor_expansion.hpp` (unified `TE` alias)

Fold the batch lane count into the existing public `TE` alias as a trailing defaulted
parameter, so there is a single name for both scalar and batched dense expansions. Alias
templates cannot be overloaded by arity (a second `TE` is a redeclaration error — confirmed
with the compiler), so `BatchTE` is dropped in favour of this:

```cpp
// forward declaration so the alias can name Batch without including batch.hpp
// (batch.hpp includes this header — avoids a circular include)
template < typename T, int K >
struct Batch;

template < int N, int M = 1, int K = 1 >
using TE = TaylorExpansion<
    std::conditional_t< K == 1, double, Batch< double, K > >, N, M, storage::Dense >;
```

- `TE<6>` → `double` univariate, `TE<4, 3>` → `double` trivariate — **unchanged** (`K = 1`
  selects plain `double`, never names `Batch`).
- `TE<6, 1, 8>` → 8-lane batched univariate, `TE<4, 3, 8>` → 8-lane batched trivariate.

`K = 1` deliberately resolves to `double` (not `Batch<double,1>`) to preserve the existing
`constexpr` surface and exact semantics of scalar `TE`. The constraint `requires Scalar<T>` on
`TaylorExpansion` and `Batch`'s completeness are only checked when a `K > 1` instantiation is
formed — which only happens in code that has included `batch.hpp` via the umbrella, so the
forward declaration is sufficient at the alias's definition site. `TEn<N, M>` and `STE<N, M>`
are unchanged.

### 3. `include/tax/kernels/transcendental.hpp` (erf constant)

Current line:
```cpp
constexpr T two_over_sqrtpi = T{ 2 } * std::numbers::inv_sqrtpi_v< T >;
```
`std::numbers::inv_sqrtpi_v<T>` is ill-formed for `T = Batch` (not a literal floating type),
and an Eigen-backed `Batch` is not `constexpr`. Change to name the constant through the real
scalar and drop `constexpr` to `const`:
```cpp
const T two_over_sqrtpi = T{ 2 } * std::numbers::inv_sqrtpi_v< real_scalar_t< T > >;
```
`erf` is already a runtime (non-`constexpr`) kernel — `std::erf` isn't `constexpr` — so losing
`constexpr` here is immaterial for scalar `T`, and `T{2} * double` resolves via `Batch`'s
implicit broadcast ctor. Re-derive on the **current** file; verify no other kernel names a
real constant directly (`exp`, `log`, etc. are pure recurrences and need no change).

### 4. `include/tax/operators/math_binary.hpp` (pow / atan2)

The prototype generalised this file (~39 lines). Re-derive the minimum on the current version:
ensure the real-exponent `pow(TE, P)`, scalar-base `pow(s, TE)`, and `atan2` overloads compile
and dispatch for `T = Batch` (e.g. exponent/constant terms typed via `T` / `real_scalar_t<T>`
rather than a bare `double`, and the integer-vs-real `pow` kernel selection still triggers).
Driven by the `IntegerVsRealPowSelectsCorrectKernel` test below.

### 5. `include/tax/tax.hpp` (umbrella)

Add `#include <tax/core/batch.hpp>` immediately after `core/taylor_expansion.hpp` and before
the operator headers — matching the prototype ordering. `batch.hpp` includes `concepts.hpp`
and `taylor_expansion.hpp`, so it is self-sufficient if included standalone too.

### 6. `tax::la` support (verify & fix)

`tax::la` (`values.hpp`, `derivatives.hpp`) is already scalar-generic — every helper derives
`T` / `scalar_type` from traits, with no hardcoded `double`. Expectation: `variables<TE<N,M,K>>`,
`value`, `eval`, `gradient`, `hessian`, `jacobian` work once `NumTraits<Batch>` is present
(the prototype's `MultivariateGradientAndEigenInterop` test already exercises this path). Task:
instantiate each la helper with a batched `TE<N,M,K>` in tests and fix anything that breaks (most likely
none; if a helper assumes an ordered/`abs`-able scalar, address narrowly).

### 7. Tests — `tests/core/test_batch.cpp` (+ registration)

Port the prototype's 6 self-contained tests (gtest + `<tax/tax.hpp>` only):

- `ScalarLaneArithmetic`, `ScalarBroadcastAndFromLanes` — `Batch` value semantics.
- `MathSurfaceLaneEquivalence` — every math fn: each lane equals its scalar `TE` run.
- `IntegerVsRealPowSelectsCorrectKernel` — exercises the `math_binary` change (§4).
- `MultivariateGradientAndEigenInterop` — multivariate batched `TE<N,M,K>` + `tax::la` gradient +
  Eigen matrix interop (covers §6).
- `EvalPerLaneDisplacement` — `eval` with distinct per-lane displacements.

Register with `tax_add_test(test_batch SOURCES core/test_batch.cpp)` in `tests/CMakeLists.txt`.
Update the prototype's `BatchTE<N,K,M>` spellings to the unified `TE<N,M,K>` order. Add a short
la-focused assertion (value/eval/gradient over a batched `TE<N,M,K>`) if §6 surfaces a gap.

### 8. Docs — `docs/guide/batch.md` (+ nav)

Port the prototype's batch doc. Current docs have no `core/` section, so place it under
`docs/guide/` (a usage page) and add it to the `Guide` nav in `mkdocs.yml`. Update the
prototype's `BatchTE<N,K,M>` spellings to the unified `TE<N,M,K>`, drop the `BatchTE` row from
the alias table, and **remove the "Batched ODE integration" section** — `tax::ode` lives in
the `tax-flow` companion, not here.

## Data flow

```
user code / tax-flow refine
        │  builds TE<N,M,K> (K instances per lane) via tax::la::variables or directly
        ▼
TaylorExpansion<Batch<double,K>, N, M, Dense>
        │  arithmetic + math via existing dense kernels (generic on T)
        ▼   (kernels call element-wise Batch ops through ADL — one pass, K lanes)
result TE<N,M,K>  →  value()/eval()/gradient() give per-lane (Batch) results
        │  lane i == the i-th independent scalar run, bit-for-bit
```

## Error handling / invariants

- **Dense only.** No sparse `Batch` support; sparse zero-detection is per-lane-ambiguous.
- **Lane equivalence is the correctness contract:** every test asserts `lane[i]` matches the
  i-th scalar `TE` run to ~machine-eps.
- **No new heap in core:** `Batch` is a fixed `Eigen::Array` on the stack; dense storage stays
  `std::array`. `DontAlign` avoids over-aligned-load requirements for stack `std::array<Batch>`.
- **constexpr discipline:** `Batch` is intentionally non-`constexpr` (Eigen-backed). Only the
  `erf` seed loses `constexpr` (already runtime). All index arithmetic and the dense
  constexpr surface for floating `T` are untouched.
- **ODR / config macros:** unchanged. No build-system macro changes.

## Testing strategy

1. Build with the `tax` mamba env, `TAX_BUILD_UNITTESTS=ON`.
2. `ctest` green, with `test_batch` added.
3. `clang-format` clean on all touched headers.
4. No new dynamic allocation in the dense core.

## Out of scope (YAGNI)

- `tax-flow` `refine` rewiring (separate repo, follow-up).
- Sparse / named-expansion `Batch` support.
- ODE/ADS batch benchmarks and the Kepler batch test (live in `tax-flow`).
- Runtime-`K` / dynamic lane counts.

## Risks

- `math_binary` re-derivation (§4) is the only non-mechanical change; the
  `IntegerVsRealPowSelectsCorrectKernel` test pins it.
- A core member added since the merge base might assume an ordered/streamable scalar; dense
  members instantiate lazily, so this only bites if a batch test calls such a path — tests are
  scoped to avoid string-IO. Address narrowly if it surfaces.
