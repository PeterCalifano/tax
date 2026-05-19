# tax — Stage 1: C++ base (greenfield, TDD) — Design

**Status:** draft for approval
**Date:** 2026-05-19
**Branch (to be created):** `stage1-cpp-base`

## Goal

Restructure the `tax` package on a new branch by rebuilding the C++ core
ground-up under test-driven development. Stage 1 delivers:

- A single class template `tax::TaylorExpansion<T, N, M, Storage>` covering both
  dense and sparse static-shape Taylor polynomials.
- First-class Eigen integration (Eigen is a required dependency; the public API
  consumes and produces Eigen matrices/vectors).
- The full math surface of the current static path (arithmetic, algebra,
  trigonometric, transcendental).

Out of scope for Stage 1: ODE integrators, ADS/LOADS, dynamic-shape
`TaylorExpansion`, Python bindings, DACE-comparison tests, examples. These are
removed from this branch and reintroduced in later stages or re-merged from
`main`.

## Decisions log

These are the choices that were settled during brainstorming. They are
load-bearing for everything below.

| Axis | Choice | Rationale |
|---|---|---|
| Restructure mode | Greenfield rewrite | User-requested clean slate; TDD discipline mitigates regression risk. |
| Eigen integration | Hybrid: Eigen-native public API, `std::array` internals | Clean external API without surrendering `constexpr`/`noexcept` of the hot path. |
| Expression templates | Dropped in Stage 1 | Benchmarks (`benchmarks/et_vs_eager.cpp`, three seeds) show ET delivers a clear win on exactly one shape (4-term sums on medium polynomials, ~30ns absolute) and loses ~20% on the most expensive operation (Cauchy product at large `(N,M)`). Net cost: ~1200 lines + meaningful template instantiation. Can be reintroduced in a focused PR if profiling later workloads demands it. |
| Dense/sparse relationship | One class template parameterized by `Storage` policy | Eliminates duplicated class hierarchy; storage-specific fast paths still selectable via concept-based overloading. |
| Dynamic-shape types | Dropped from Stage 1; type system designed so a `storage::DynamicShape` slot can be added later without API churn | Aligned with greenfield scope. |
| Repo state at end of Stage 1 | Lean — only Stage 1 surface present | Avoids dead code rot; missing pieces re-merged or re-implemented per stage. |
| Kernel toggles | Keep `TAX_USE_UNROLL` and `TAX_USE_STENCIL` | Preserve A/B verification; replace `#ifdef` proliferation with a single dispatch entry point per kernel. |
| Type names | `TaylorExpansion<T, N, M, Storage>` with aliases `TE<N, M=1>`, `STE<N, M=1>` | Drop trailing `T` (templated-class hangover); short aliases for convenience. |
| Variables factory | Returns `Eigen::Matrix<TE, M, 1>` (Eigen vector form only) | Consistent with Eigen-vocabulary API. Structured bindings unavailable; `v(i)` indexing. |
| Polynomial map inversion | Renamed `invertMap` → `invert` | Shorter, more discoverable. |

## Architecture overview

### One class, storage-policy dispatch

```cpp
namespace tax {
    template <typename T, int N, int M, typename Storage = storage::Dense>
    class TaylorExpansion;

    template <int N, int M = 1>  using TE  = TaylorExpansion<double, N, M, storage::Dense>;
    template <int N, int M = 1>  using STE = TaylorExpansion<double, N, M, storage::Sparse>;
}
```

- `Storage` is a tag type (`storage::Dense`, `storage::Sparse`) plus an
  associated trait providing: the data container, the support iterator, and
  the in-place mutation primitives (`set(flat_index, T)`,
  `accumulate(flat_index, T)`, `forEach(callable)`).
- Kernels are written once against a `TaylorPolynomial` concept that both
  storages satisfy. Storage-specific fast paths are selected via
  concept-based overloads (`requires DensePolynomial<P>`, etc.).
- All operators are **eager**: each `operator+`, `sin(f)`, etc. returns a fresh
  `TaylorExpansion`/`SparseTaylorExpansion` by value. No expression-template
  proxies.
- C++23, header-only. Eigen ≥ 3.4 is a required dependency.

### Mixed dense↔sparse arithmetic

Operators are defined `TE op TE → TE` and `STE op STE → STE`. **Mixed types do
not compile** — the user must convert explicitly via `s.dense()` or
`tax::sparse(d)`. This keeps the type system honest about allocations and
prevents implicit storage policy upgrades in hot paths.

## File layout

```
tax/
├── CMakeLists.txt
├── cmake/
│   └── taxConfig.cmake.in
├── include/tax/
│   ├── tax.hpp                       # umbrella (only public header)
│   ├── core/
│   │   ├── concepts.hpp              # TaylorPolynomial, DensePolynomial, SparsePolynomial, Scalar
│   │   ├── multi_index.hpp           # MultiIndex<M>, flatIndex, numMonomials, DegreeOf<N,M>, Coeffs<T,N,M>
│   │   ├── enumeration.hpp           # forEachMonomial, forEachSubIndex
│   │   ├── taylor_expansion.hpp      # primary template + aliases TE, STE
│   │   └── storage/
│   │       ├── dense.hpp             # storage::Dense + DenseContainer
│   │       └── sparse.hpp            # storage::Sparse + SparseContainer (SparseCoeffs alias)
│   ├── kernels/
│   │   ├── cauchy.hpp                # dense Cauchy product (loop variant; dispatch entry)
│   │   ├── cauchy_stencil.hpp        # TAX_USE_STENCIL-gated stencil variant (M≥2)
│   │   ├── cauchy_unroll.hpp         # TAX_USE_UNROLL-gated M=1 unrolled variants
│   │   ├── algebra.hpp               # reciprocal, sqrt, cbrt, square, cube, pow
│   │   ├── trigonometric.hpp         # sin, cos, tan, asin, acos, atan, atan2
│   │   ├── transcendental.hpp        # exp, log, sinh, cosh, tanh + inverses, erf
│   │   ├── sparse_cauchy.hpp         # sparse Cauchy product, self-product, accumulate
│   │   └── sparse_subs.hpp           # sparse sqrt, reciprocal, division, integer pow
│   ├── operators/
│   │   ├── arithmetic.hpp            # +, -, *, /, scalar variants (eager)
│   │   ├── math_unary.hpp            # sin, cos, exp, log, ... free-fn forms
│   │   └── math_binary.hpp           # pow, atan2
│   └── eigen.hpp                     # NumTraits + variables + gradient + hessian
│                                     #   + jacobian + value + eval + derivative + invert
├── tests/
│   ├── CMakeLists.txt
│   ├── testUtils.hpp
│   ├── core/                         # multi-index, combinatorics, TaylorExpansion ctor/accessors
│   ├── kernels/                      # direct kernel verification (dense + sparse + diff tests)
│   ├── operators/                    # one file per math op
│   ├── sparse/                       # sparse-specific scenarios
│   └── eigen/                        # NumTraits, variables, gradient, jacobian, eval
├── benchmarks/
│   ├── CMakeLists.txt
│   ├── baseline/                     # captured numbers from main, used as regression gate
│   │   └── main-<sha>.txt
│   ├── ops_dense.cpp                 # Cauchy, sin/cos/exp, sqrt, etc. across (N,M)
│   ├── ops_sparse.cpp                # sparse equivalents
│   └── eigen_workflows.cpp           # gradient/Jacobian end-to-end
├── docs/                             # markdown notes; mkdocs config kept
└── .github/workflows/
    ├── tests.yml                     # slimmed: Ubuntu/macOS × GCC/Clang × Eigen 3.4/5.0
    ├── sanitizers.yml                # ASAN/UBSAN/TSAN
    └── bench.yml                     # perf gate; fails on >5% regression
```

Notes:

- `eigen.hpp` is single-file because the surface is small (~6 free functions +
  NumTraits) once dynamic-shape special cases and `Eigen::Tensor` rank≥3
  overloads are removed.
- `core/utils/` is gone; combinatorics/multi-index/enumeration live under
  `core/` next to the types that consume them. `Coeffs<T,N,M>` is promoted to
  the public `tax::` namespace.
- DACE integration disappears entirely from Stage 1. Comparative numbers from
  the current `bench_vs_dace` are captured to `benchmarks/baseline/` as plain
  text and used as the perf gate target during the port.

## Public type API

### Construction

```cpp
// Univariate, static
auto x = tax::TE<5>::variable(1.0);
auto c = tax::TE<5>::constant(3.0);
auto z = tax::TE<5>::zero();

// Multivariate via Eigen vector (canonical form)
Eigen::Vector3d x0{1.0, 2.0, 3.0};
Eigen::Matrix<tax::TE<5,3>, 3, 1> v = tax::variables<tax::TE<5,3>>(x0);
// users index with v(0), v(1), v(2)

// Sparse
auto s = tax::STE<5, 3>::variable(0, x0);
auto sv = tax::variables<tax::STE<5,3>>(x0);
```

### Accessors

```cpp
T            f.value();                                  // constant term
T            f.coeff(tax::MultiIndex<M>);                // runtime multi-index
template<int... A> T f.coeff();                          // compile-time, static_asserts sum(A) <= N
T            f.derivative(tax::MultiIndex<M>);           // coeff * prod(α_i!)
template<int... A> T f.derivative();                     // compile-time, static_asserts sum(A) <= N
T            f.eval(const Eigen::Matrix<T,M,1>& dx);     // evaluate at x0 + dx
```

### Symbolic operations on `TaylorExpansion`

```cpp
auto df = f.deriv<I>();                                  // ∂f/∂x_I, same shape
auto Fx = f.integ<I>();                                  // ∫ f dx_I, same shape
auto df_dyn = f.deriv(int i);
auto Fx_dyn = f.integ(int i);
auto g = f.gradient();                                   // Eigen::Matrix<T, M, 1>
auto H = f.hessian();                                    // Eigen::Matrix<T, M, M>
```

### Arithmetic and math (free functions, eager)

```cpp
auto g = a + b;                                          // also: f + 2.0, 3.0 * f, ...
auto h = a * b + sin(c) * exp(d) - sqrt(e);
auto k = pow(f, 3);
auto m = atan2(a, b);
```

### Sparse-specific surface

```cpp
tax::STE<N,M> s = ...;
s.nnz();
auto support = s.support();                              // std::span<const flat_index_t>
auto values  = s.values();                               // std::span<const T>
tax::TE<N,M> d = s.dense();                              // explicit conversion
tax::STE<N,M> s2 = tax::sparse(d);                       // dense → sparse, drops exact zeros
```

### Vector-form free functions (in `eigen.hpp`)

```cpp
Eigen::Matrix<T, K, M> tax::jacobian(const Eigen::MatrixBase<...>& F);
Eigen::Matrix<T, K, 1> tax::value(const Eigen::MatrixBase<...>& F);
Eigen::Matrix<T, K, 1> tax::eval(const Eigen::MatrixBase<...>& F,
                                 const Eigen::Matrix<T,M,1>& dx);
template<int... A>
Eigen::Matrix<T, K, 1> tax::derivative(const Eigen::MatrixBase<...>& F);
Eigen::Matrix<TE, M, 1> tax::invert(const Eigen::MatrixBase<...>& F);
```

### Concepts (for kernel writers)

```cpp
namespace tax {
    template <typename P> concept TaylorPolynomial = /* has scalar_type, order, vars, ... */;
    template <typename P> concept DensePolynomial  = TaylorPolynomial<P> &&
                                                     /* has random access by flat index */;
    template <typename P> concept SparsePolynomial = TaylorPolynomial<P> &&
                                                     /* has sorted support iteration */;
}
```

## Kernel layer

### Three layers

1. **Container-level primitives** (in `core/storage/*.hpp`):
   - `Dense`: contiguous `tax::Coeffs<T, N, M>`. Random access by flat index.
   - `Sparse`: parallel sorted `std::vector<flat_index_t>` + `std::vector<T>`.
     Iterators over support; `find(flat_index)` is `O(log nnz)`.
   - Both expose: `size()`, `value()`, `set(flat_index, T)`,
     `accumulate(flat_index, T)`, `forEachNonzero(callable)`,
     `forEachPair(other, callable)`.
   - Sparse also exposes raw `support()` and `values()` spans for kernels
     that benefit from sequential access.

2. **Algorithm-level kernels** (in `kernels/`):
   - `cauchyProduct`, `cauchyAccumulate`, `cauchySelfProduct`
   - `seriesReciprocal`, `seriesSqrt`, `seriesCbrt`, `seriesSquare`,
     `seriesCube`, `seriesPow`
   - `seriesExp`, `seriesLog`, `seriesSinh`, `seriesCosh`, `seriesTanh`,
     inverses, `seriesErf`
   - `seriesSin`, `seriesCos`, `seriesTan`, `seriesAsin`, `seriesAcos`,
     `seriesAtan`, `seriesAtan2`
   - Each has a base impl against `TaylorPolynomial`; specialized overloads for
     `DensePolynomial` (random access) and `SparsePolynomial` (support
     iteration).

3. **Operator layer** (in `operators/`):
   - Thin wrappers that allocate the output container and dispatch to the
     right kernel. Public entry points to the kernel surface.

### Dispatch by concept, not by `#ifdef`

`#if`-gates appear in **one** place per kernel — the public entry point:

```cpp
template <DensePolynomial P>
constexpr void cauchyProduct(P& out, P const& a, P const& b) noexcept
{
#if TAX_USE_UNROLL
    if constexpr (P::M == 1) { cauchyProductUnroll(out, a, b); return; }
#endif
#if TAX_USE_STENCIL
    if constexpr (P::M >= 2) { cauchyProductStencil(out, a, b); return; }
#endif
    cauchyProductLoop(out, a, b);
}
```

A/B equivalence is enforced by *dedicated diff tests* (e.g.,
`tests/kernels/test_cauchy_unroll_diff.cpp`) that always build all three
variants and assert they agree to within `1e-12`, regardless of `TAX_USE_*`.
This keeps the toggle useful (you can ship the slow path on purpose) without
doubling the test matrix.

### Kernel invariants

- No allocation. Output container is passed in pre-sized.
- No exceptions.
- No I/O.
- Sparse kernels pre-count output support and `reserve()` once; never
  `push_back` in inner loops.

## Eigen integration (single `eigen.hpp`)

```cpp
namespace Eigen {
    template <typename T, int N, int M, typename S>
    struct NumTraits<tax::TaylorExpansion<T, N, M, S>> : NumTraits<T> { /* ... */ };
}

namespace tax {
    // Variable factories — consume Eigen vectors, produce Eigen vectors of TE.
    template <typename TE, typename Derived>
    Eigen::Matrix<TE, M_of<TE>, 1> variables(const Eigen::MatrixBase<Derived>& x0);

    // Element-wise extractors on Eigen containers of TE.
    template <typename Derived, typename DxDerived>
    auto eval(const Eigen::MatrixBase<Derived>& f,
              const Eigen::MatrixBase<DxDerived>& dx);

    template <int... Alpha, typename Derived>
    auto derivative(const Eigen::MatrixBase<Derived>& f);

    template <typename Derived>
    auto value(const Eigen::MatrixBase<Derived>& f);

    // Vector calculus.
    template <typename T, int N, int M, typename S>
    Eigen::Matrix<T, M, 1> gradient(const TaylorExpansion<T,N,M,S>& f);

    template <typename T, int N, int M, typename S>
    Eigen::Matrix<T, M, M> hessian (const TaylorExpansion<T,N,M,S>& f);

    template <typename Derived>
    auto jacobian(const Eigen::MatrixBase<Derived>& F);

    // Polynomial map inversion.
    template <typename Derived>
    auto invert(const Eigen::MatrixBase<Derived>& F);
}
```

Boundary discipline: `variables`, `eval`, `invert` convert Eigen inputs to
`tax::TaylorExpansion::Input` (a `std::array<T, M>`) **once** at the
boundary. Kernels never see Eigen types.

Tradeoffs accepted:

- Including any tax header transitively pulls in `<Eigen/Core>` (was true
  only for `tax/tax.hpp` before). Compile-time cost ~+1–2s per TU. Accepted as
  the price of "Eigen as a real dependency."
- `Eigen::Tensor<TE, Rank>` rank-≥3 derivative overloads are not in Stage 1.
  Reintroduce in a focused PR if needed.

## TDD slicing and the perf gate

### Pre-flight

1. Snapshot perf baseline from `main`: run `bench_univariate`,
   `bench_multivariate`, `bench_vs_dace` (DACE on), and the
   `bench_et_vs_eager` micro. Save outputs to
   `benchmarks/baseline/main-<sha>.txt`.
2. Branch from current head as `stage1-cpp-base`.
3. Delete `include/tax/ode/`, `include/tax/ads/`,
   `include/tax/storage/tte_dynamic.hpp`,
   `include/tax/storage/tte_dynamic_order.hpp`, `include/tax/eigen/`,
   `include/tax/expr/`, `include/tax/utils/`,
   `include/tax/storage/{shape,sparse_tte,tte_static}.hpp`, and
   `include/tax/kernels/*`. Delete the entire `python/`, `examples/`,
   `tools/`, and any DACE wiring in CMake. The old code remains available
   for reference via `git show main:<path>` or
   `git checkout main -- <path>` during porting; nothing needs to be kept
   in-tree on the Stage 1 branch.
4. Scaffold: empty `tax.hpp`, empty `core/`, `kernels/`, `operators/`, empty
   `eigen.hpp`, empty `tests/` + `benchmarks/` directories. CMakeLists
   produces a no-op library; `ctest` runs zero tests successfully.

### Slice ordering

Each slice is one PR (or one merged commit, depending on size). Workflow per
slice: write failing tests → implement → tests green → run perf gate →
refactor if clean → commit.

| # | Slice | Tests ported from | New code |
|---|---|---|---|
| 1 | Foundations: `MultiIndex`, `flatIndex`, `numMonomials`, `totalDegree`, `DegreeOf`, enumeration | `tests/foundation/` | `core/multi_index.hpp`, `core/enumeration.hpp`, `core/concepts.hpp` (stubs) |
| 2 | Dense storage + `TaylorExpansion` ctor, value/coeff/derivative accessors (compile-time and runtime) | `tests/core/` (construction, accessors) | `core/storage/dense.hpp`, `core/taylor_expansion.hpp` (dense path) |
| 3 | Dense arithmetic: `+`, `-`, `*`, `/`, scalar variants; eager only | `tests/core/` (arithmetic), adapted `tests/expr/test_arith*` | `operators/arithmetic.hpp`, `kernels/cauchy.hpp` (loop variant) |
| 4 | Cauchy fast paths: M=1 unroll, M≥2 stencil, diff-tests | `tests/kernels/` (cauchy, stencil) | `kernels/cauchy_unroll.hpp`, `kernels/cauchy_stencil.hpp` |
| 5 | Algebra: square, cube, sqrt, cbrt, reciprocal, pow | `tests/expr/testExpr{Square,Sqrt,Cbrt,Reciprocal,Pow}*` | `kernels/algebra.hpp`, `operators/math_unary.hpp` |
| 6 | Transcendentals: exp, log, sinh, cosh, tanh + inverses, erf | corresponding `tests/expr/` files | `kernels/transcendental.hpp` |
| 7 | Trigonometric: sin, cos, tan, asin, acos, atan, atan2 | corresponding `tests/expr/` files | `kernels/trigonometric.hpp`, `operators/math_binary.hpp` |
| 8 | Differentiation/integration methods (`deriv<I>`, `integ<I>`, runtime variants) | `tests/core/` (deriv, integ) | methods on `TaylorExpansion` |
| 9 | Sparse storage + sparse↔dense conversion | `tests/sparse/` (construction, conversion) | `core/storage/sparse.hpp` |
| 10 | Sparse arithmetic + kernels | `tests/sparse/` (arithmetic, sparse_cauchy, sparse_subs) | `kernels/sparse_cauchy.hpp`, `kernels/sparse_subs.hpp`, sparse operator overloads |
| 11 | Eigen integration: NumTraits, variables, value, eval, derivative element-wise | `tests/eigen/` (NumTraits, variables, eval, derivative) | first half of `eigen.hpp` |
| 12 | Eigen integration: gradient, hessian, jacobian, invert | `tests/eigen/` (gradient, hessian, jacobian, invert_map) | second half of `eigen.hpp`, methods on `TaylorExpansion` |

### Perf gate

- Slices 3–4: `bench_univariate` and `bench_multivariate` Cauchy timings
  must not regress >5% vs. baseline.
- Slices 5–7: `bench_vs_dace` operator timings must not regress >5%.
- Slice 10: sparse benchmarks must not regress >5%.
- Slices 11–12: end-to-end `bench_vs_dace` must not regress >5%.

A regression blocks the slice's merge until either fixed or explicitly
accepted (commit message annotation: `[allow-perf-regression: <reason>]`).

### CI

- `tests.yml`: slimmed to Stage 1 surface. Matrix: Ubuntu/macOS × GCC/Clang
  × Eigen 3.4/5.0. Release. Google Test. No DACE, no Python.
- `sanitizers.yml`: ASAN/UBSAN/TSAN on the same test suite.
- `bench.yml`: runs the perf gate on PRs; prints comparison; fails on >5%
  regression unless `[allow-perf-regression]` is in the commit message.

### Sub-agent strategy

Slices are sequentially dependent (each depends on the previous's tests
being green and merged). Within a slice, the work fits a single focused
subagent task: "given these failing tests and these kernel signatures,
implement the kernels and operators until tests are green; do not touch
files outside the slice's scope."

## What does NOT exist at end of Stage 1

- No ODE integrators, no ADS/LOADS.
- No dynamic-shape `TaylorExpansion`.
- No Python bindings.
- No DACE-comparison tests or benchmarks.
- No `eigen/` subfolder.
- No expression-template layer.
- The C++ surface is roughly half the size of today's `include/tax/`.

## Open questions deferred to implementation

These are intentionally not pinned in the spec; they are local decisions
that surface during the slice they touch:

- Exact name of the `flat_index_t` integer type (`std::uint32_t` vs.
  `std::size_t`; affects sparse memory footprint).
- Whether the sparse drop-threshold defaults to exact-zero or a small
  epsilon (`1e-300`-class); decide when porting sparse arithmetic tests.
- Whether `TaylorExpansion::Input` stays a public type alias or becomes
  `tax::Point<T, M>`; cosmetic, settle in slice 2.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Perf regression on Cauchy hot path during the storage-policy refactor | Slice 3–4 gate; if storage-policy indirection costs >5%, fall back to keeping `Coeffs<T,N,M>` directly inside `TaylorExpansion` and reaching the policy via traits only. |
| Eigen-as-required dependency slows compile times noticeably across the test suite | Measured baseline ahead of time; acceptable cap is +20% TU compile time. If exceeded, push back to user. |
| Concept dispatch is opaque or fragile vs. simple `if constexpr` | If a kernel becomes harder to read than the current version, keep the simpler form and document the trade. |
| Diff tests for UNROLL/STENCIL bloat compile times unacceptably | Move them into a dedicated `tests/kernels/diff/` subdirectory with its own CMake target; turn off by default, on in CI. |

---

## Implementation plan

To be produced by the `writing-plans` skill after this spec is approved.
