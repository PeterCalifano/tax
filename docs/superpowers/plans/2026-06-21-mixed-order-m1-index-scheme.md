# Mixed-order Milestone 1 — `IndexScheme` + `IsotropicScheme` refactor

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce an `IndexScheme` abstraction and rewire the dense kernels + `TaylorExpansion` onto `IsotropicScheme<N,M>` — a **behavior-preserving** refactor that lets a later `MixedScheme` reuse the same kernels. No new user-facing behavior.

**Architecture:** An index scheme bundles the three things the kernels need — storage size, the graded recurrence-row walker, and the Cauchy product stencil. The existing isotropic tables become `IsotropicScheme<N,M>`. Kernels gain scheme-generic forms (`<T, Scheme>`); the old `<T,N,M>` signatures stay as thin delegating wrappers so the build is green between tasks, and call sites migrate last.

**Tech Stack:** C++23, header-only, Eigen3, Google Test, CMake. Build in the `tax` mamba/conda env (active).

**This is Milestone 1 of 5** from `docs/superpowers/specs/2026-06-21-mixed-order-named-expansions-design.md`. M2 (`MixedScheme` + mixed stencils), M3 (core `MixedExpansion`), M4 (named layer), M5 (`la::mixed` + docs) are planned just-in-time after this milestone lands and the `IndexScheme` interface is concrete.

## Global Constraints

- C++23, header-only — all code in `include/tax/`. No heap allocation in the dense core (`std::array` only; the stencil caches are pre-existing runtime statics).
- **Behavior-preserving:** the entire existing test suite must pass unchanged after every task, and bit-for-bit at the end. This refactor adds NO user-visible behavior. `TaylorExpansion`'s public API, coefficient values, and `constexpr`-ness are unchanged.
- Preserve the isotropic fast paths: the `M==1` unrolled kernels and the `M>=2` stencil specialization must remain (same codegen). `IsotropicScheme` must not regress performance.
- Keep `constexpr` discipline: the `if !consteval` stencil-vs-loop fallback in `forEachRecurrenceRow` must be preserved through the scheme.
- Kernel-config macros stay in-header (`TAX_USE_STENCIL`/`TAX_USE_UNROLL` in `cauchy.hpp`); introduce no build-system macros.
- clang-format is **v21** locally (newer than `main`'s). Format only newly added code; after `clang-format -i`, `git diff` against the task base and revert any gratuitous reflow of pre-existing constructs (notably `requires`-expression braces in `concepts.hpp` and the `tax.hpp` include order — never reorder the umbrella includes). Commit only intended logical changes.
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Work on branch `feature/mixed-order-expansions` (already checked out).

## Build & test reference (run from repo root `/Users/andrea/Documents/Codes/tax`)

- Configure (once if stale): `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
- Build one target: `cmake --build build -j --target <name>`
- Full suite (the behavior gate): `cmake --build build -j && ctest --test-dir build --output-on-failure`
- Single test exe: `ctest --test-dir build -R <name> --output-on-failure`

## File Structure

- `include/tax/core/index_scheme.hpp` — **new**: the `IndexScheme` concept + `IsotropicScheme<N,M>` adapter (its `forEachRecurrenceRow`/`cauchyProduct` members delegate to the existing free functions, so the legacy stencil headers stay untouched).
- `include/tax/kernels/cauchy.hpp` — **modify**: add a scheme-generic `cauchyProduct<T, Scheme>` forwarding to `Scheme::cauchyProduct` (keep the legacy `<N,M>` entry point).
- `include/tax/kernels/algebra.hpp`, `transcendental.hpp`, `trigonometric.hpp` — **modify**: add `<T, Scheme>` kernel forms; keep `<T,N,M>` wrappers delegating to them.
- `include/tax/core/taylor_expansion.hpp`, `include/tax/operators/*.hpp` — **unchanged in M1** (kernels reached via the preserved `<T,N,M>` wrappers); migrating them to call through `IsotropicScheme` directly and dropping the wrappers is deferred (no behavioral value yet — see Task 5 / Self-Review).
- `tests/core/test_index_scheme.cpp` — **new**: directly verify `IsotropicScheme` matches the legacy tables and that scheme-generic kernels match the public surface.

---

### Task 1: `IndexScheme` concept + `IsotropicScheme<N,M>` adapter

**Files:**
- Create: `include/tax/core/index_scheme.hpp`
- Create: `tests/core/test_index_scheme.cpp`
- Modify: `tests/CMakeLists.txt` (register `test_index_scheme`)

**Interfaces:**
- Consumes: `numMonomials`, `flatIndex`, `forEachMonomial`/`forEachSubIndex` (`core/multi_index.hpp`, `core/enumeration.hpp`); `RecurrenceEntry`, `forEachRecurrenceRow<N,M>`, `recurrenceStencil<N,M>` (`kernels/recurrence_stencil.hpp`); `CauchyStencil`/`cauchyProduct` (`kernels/cauchy*.hpp`).
- Produces:
  - concept `tax::IndexScheme<S>` requiring: `S::nCoeff` (`std::size_t`), `static constexpr bool S::isUnivariate`, `static constexpr int S::order` (max total degree any monomial can reach = the box max degree; for isotropic = `N`), and a static `S::forEachRecurrenceRow(fn)` calling `fn(std::size_t ai, int d, std::span<const detail::kernels::RecurrenceEntry>)` in graded order.
  - `template<int N, int M> struct tax::IsotropicScheme` satisfying it, with `nCoeff = numMonomials(N,M)`, `isUnivariate = (M==1)`, `order = N`, `forEachRecurrenceRow(fn)` delegating to `detail::kernels::forEachRecurrenceRow<N,M>(fn)`, plus `static std::size_t flatOf(const MultiIndex<M>&)`/`static MultiIndex<M> multiOf(std::size_t)` delegating to `flatIndex<M>`/`unflatIndex<M>` and `static constexpr int vars = M`.

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_index_scheme.cpp`:

```cpp
#include <gtest/gtest.h>

#include <tax/tax.hpp>
#include <tax/core/index_scheme.hpp>
#include <tax/kernels/recurrence_stencil.hpp>

using tax::IsotropicScheme;
using tax::detail::kernels::RecurrenceEntry;

// IsotropicScheme<N,M> must reproduce the legacy numMonomials/forEachRecurrenceRow tables exactly.
TEST( IndexScheme, IsotropicMatchesLegacyShape )
{
    using S = IsotropicScheme< 5, 2 >;
    static_assert( S::nCoeff == tax::numMonomials( 5, 2 ) );
    static_assert( S::isUnivariate == false );
    static_assert( S::order == 5 );
    static_assert( IsotropicScheme< 4, 1 >::isUnivariate == true );
}

TEST( IndexScheme, IsotropicRecurrenceRowsMatchLegacy )
{
    constexpr int N = 5, M = 2;
    using S = IsotropicScheme< N, M >;

    // Collect rows from the legacy walker.
    std::vector< std::tuple< std::size_t, int, std::vector< std::array< std::uint32_t, 3 > > > > legacy;
    tax::detail::kernels::forEachRecurrenceRow< N, M >(
        [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
            std::vector< std::array< std::uint32_t, 3 > > es;
            for ( const auto& e : row ) es.push_back( { e.b_idx, e.g_idx, e.db } );
            legacy.push_back( { ai, d, es } );
        } );

    // Collect rows from the scheme.
    std::vector< std::tuple< std::size_t, int, std::vector< std::array< std::uint32_t, 3 > > > > viaScheme;
    S::forEachRecurrenceRow(
        [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
            std::vector< std::array< std::uint32_t, 3 > > es;
            for ( const auto& e : row ) es.push_back( { e.b_idx, e.g_idx, e.db } );
            viaScheme.push_back( { ai, d, es } );
        } );

    EXPECT_EQ( legacy, viaScheme );
}
```

- [ ] **Step 2: Register the test and confirm it fails to build**

Add to `tests/CMakeLists.txt` after the `test_batch` line:

```cmake
tax_add_test(test_index_scheme SOURCES core/test_index_scheme.cpp)
```

Run: `cmake --build build -j --target test_index_scheme 2>&1 | tail -20`
Expected: FAIL — `tax/core/index_scheme.hpp` does not exist / `IsotropicScheme` undeclared.

- [ ] **Step 3: Create `include/tax/core/index_scheme.hpp`**

```cpp
#pragma once

// ---------------------------------------------------------------------------
// IndexScheme: the monomial-set abstraction the dense kernels are generic over.
// ---------------------------------------------------------------------------
// A scheme bundles what every recurrence kernel needs: the storage size, a
// graded recurrence-row walker (fn(ai, d, span<RecurrenceEntry>) in ascending
// total degree so forward substitution is causal), and the flat<->multi maps.
// IsotropicScheme<N,M> wraps exactly today's tables; a later MixedScheme<...>
// provides an anisotropic (per-axis-capped) monomial set behind the same API.

#include <cstddef>
#include <span>

#include <tax/core/multi_index.hpp>
#include <tax/kernels/recurrence_stencil.hpp>

namespace tax
{

/// Concept: a monomial-set index scheme usable by the dense kernels.
template < typename S >
concept IndexScheme = requires( const S& ) {
    { S::nCoeff } -> std::convertible_to< std::size_t >;
    { S::order } -> std::convertible_to< int >;
    { S::isUnivariate } -> std::convertible_to< bool >;
};

/// The classic single-order graded-lex scheme: total degree <= N over M vars.
template < int N, int M >
struct IsotropicScheme
{
    static constexpr std::size_t nCoeff = numMonomials( N, M );
    static constexpr int order = N;
    static constexpr int vars = M;
    static constexpr bool isUnivariate = ( M == 1 );

    [[nodiscard]] static constexpr std::size_t flatOf( const MultiIndex< M >& a ) noexcept
    {
        return flatIndex< M >( a );
    }
    [[nodiscard]] static constexpr MultiIndex< M > multiOf( std::size_t k ) noexcept
    {
        return unflatIndex< M >( k );
    }

    /// Graded recurrence-row walker (M >= 2). Delegates to the legacy table/loop.
    template < class RowFn >
    static constexpr void forEachRecurrenceRow( RowFn&& fn ) noexcept
        requires( M >= 2 )
    {
        detail::kernels::forEachRecurrenceRow< N, M >( static_cast< RowFn&& >( fn ) );
    }
};

}  // namespace tax
```

- [ ] **Step 4: Build and run the test**

Run: `cmake --build build -j --target test_index_scheme && ctest --test-dir build -R test_index_scheme --output-on-failure`
Expected: PASS — both cases green (the scheme reproduces the legacy rows exactly).

- [ ] **Step 5: clang-format and commit**

```bash
cd /Users/andrea/Documents/Codes/tax
clang-format -i include/tax/core/index_scheme.hpp tests/core/test_index_scheme.cpp
git add include/tax/core/index_scheme.hpp tests/core/test_index_scheme.cpp tests/CMakeLists.txt
git commit -m "feat(core): IndexScheme concept + IsotropicScheme<N,M> adapter

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Scheme-generic recurrence kernels (additive; wrappers preserved)

**Files:**
- Modify: `include/tax/kernels/transcendental.hpp`, `include/tax/kernels/trigonometric.hpp`, `include/tax/kernels/algebra.hpp`
- Test: existing `tests/operators/*`, `tests/kernels/*` (unchanged) remain the gate.

**Interfaces:**
- Consumes: `tax::IsotropicScheme` (Task 1), `RecurrenceEntry`.
- Produces: for each recurrence kernel `seriesX`, an additional scheme-generic overload `template<typename T, IndexScheme Scheme> void seriesX(std::array<T,Scheme::nCoeff>& out, const std::array<T,Scheme::nCoeff>& a)` (binary ones take a second input). The existing `template<typename T,int N,int M> void seriesX(Coeffs<T,N,M>&, ...)` is kept as a one-line wrapper `{ seriesX<T, IsotropicScheme<N,M>>(out, a); }` so all current call sites compile unchanged.

**The mechanical transform (apply to every `seriesX` recurrence kernel — `seriesReciprocal, seriesDivide, seriesSqrt, seriesCbrt, seriesPow, seriesExp, seriesLog, seriesTanh, seriesAsinh, seriesAcosh, seriesAtanh, seriesErf, seriesSinCos, seriesSin, seriesCos, seriesTan, seriesAsin, seriesAcos, seriesAtan, seriesAtan2`):**

1. Add a scheme-generic overload whose body is the CURRENT kernel body with these substitutions:
   - signature `Coeffs<T,N,M>& out` → `std::array<T, Scheme::nCoeff>& out` (same for inputs);
   - `if constexpr ( M == 1 )` → `if constexpr ( Scheme::isUnivariate )`, and inside that branch replace `N` with `Scheme::order`;
   - the `else` branch's `forEachRecurrenceRow<N,M>(...)` → `Scheme::forEachRecurrenceRow(...)`;
   - any other use of `N` (e.g. loop bounds) → `Scheme::order`; any `numMonomials(N,M)`/`S = ...` size → `Scheme::nCoeff`.
2. Replace the old `<T,N,M>` body with a one-line delegating wrapper.

**Worked example — `seriesExp` (`transcendental.hpp:13`).** Replace:

```cpp
template < typename T, int N, int M >
void seriesExp( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    using std::exp;
    out = {};
    out[0] = exp( a[0] );
    if constexpr ( M == 1 ) { /* univariate loop using N */ }
    else { forEachRecurrenceRow< N, M >( /* ... */ ); }
}
```

with the scheme-generic form plus a wrapper:

```cpp
/// Natural exponential series `out = exp(a)` (scheme-generic).
template < typename T, IndexScheme Scheme >
void seriesExp( std::array< T, Scheme::nCoeff >& out,
                const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::exp;
    out = {};
    out[0] = exp( a[0] );
    if constexpr ( Scheme::isUnivariate )
    {
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 0; k < d; ++k )
                rhs += T( d - k ) * a[std::size_t( d - k )] * out[std::size_t( k )];
            out[std::size_t( d )] = rhs / T( d );
        }
    }
    else
    {
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
                T rhs = T{ 0 };
                for ( const RecurrenceEntry& e : row )
                    rhs += T( e.db ) * a[e.b_idx] * out[e.g_idx];
                out[ai] = rhs / T( d );
            } );
    }
}

/// Legacy (N, M) entry point — delegates to the scheme-generic form.
template < typename T, int N, int M >
void seriesExp( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    seriesExp< T, IsotropicScheme< N, M > >( out, a );
}
```

(`Coeffs<T,N,M>` is `std::array<T, numMonomials(N,M)>` = `std::array<T, IsotropicScheme<N,M>::nCoeff>`, so the wrapper's argument binds to the generic overload's parameter type exactly.) Add `#include <tax/core/index_scheme.hpp>` to each kernel header.

> Note on multi-array kernels (`seriesSinCos`, `seriesCbrt`, `seriesTanh`, the inverse-trig/`atan2` log-style ones that build a helper `h`): the same transform applies — their internal scratch `Coeffs<T,N,M>` become `std::array<T,Scheme::nCoeff>`, and every `forEachRecurrenceRow<N,M>` becomes `Scheme::forEachRecurrenceRow`. Their math is unchanged.

- [ ] **Step 1: Transform `transcendental.hpp`**

Apply the transform to every `seriesX` in `transcendental.hpp` (`seriesExp, seriesLog, seriesSinh, seriesCosh, seriesTanh, seriesAsinh, seriesAcosh, seriesAtanh, seriesErf`). `seriesSinh`/`seriesCosh` use a post-combination flat loop `for (i=1; i<S; ++i)` — make `S = Scheme::nCoeff` and keep the loop (it is index-agnostic). Add the include.

- [ ] **Step 2: Build the dependent tests, expect green**

Run: `cmake --build build -j --target test_exp_log test_hyperbolic test_erf && ctest --test-dir build -R "test_exp_log|test_hyperbolic|test_erf" --output-on-failure`
Expected: PASS — the wrappers preserve exact behavior (these exercise `seriesExp/Log/Sinh/Cosh/Tanh/Erf`).

- [ ] **Step 3: Transform `trigonometric.hpp`**

Apply the transform to `seriesSinCos, seriesSin, seriesCos, seriesTan, seriesAsin, seriesAcos, seriesAtan, seriesAtan2`. Add the include.

- [ ] **Step 4: Build the trig tests, expect green**

Run: `cmake --build build -j --target test_trig test_inverse_trig && ctest --test-dir build -R "test_trig|test_inverse_trig" --output-on-failure`
Expected: PASS.

- [ ] **Step 5: Transform `algebra.hpp`**

Apply the transform to the recurrence kernels `seriesReciprocal, seriesDivide, seriesSqrt, seriesCbrt, seriesPow`. Leave `cauchySelfProduct`, `seriesSquare`, `seriesCube`, `seriesPowInt` unchanged for now (they are Cauchy-based; handled in Task 3). Add the include.

- [ ] **Step 6: Build the algebra/pow tests, expect green**

Run: `cmake --build build -j --target test_algebra_inverse test_algebra_square_cube test_pow && ctest --test-dir build -R "test_algebra_inverse|test_algebra_square_cube|test_pow" --output-on-failure`
Expected: PASS.

- [ ] **Step 7: Full suite gate**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`
Expected: ALL tests PASS — no behavior change (kernels delegate through `IsotropicScheme`).

- [ ] **Step 8: clang-format and commit**

```bash
cd /Users/andrea/Documents/Codes/tax
clang-format -i include/tax/kernels/transcendental.hpp include/tax/kernels/trigonometric.hpp include/tax/kernels/algebra.hpp
git add include/tax/kernels/transcendental.hpp include/tax/kernels/trigonometric.hpp include/tax/kernels/algebra.hpp
git commit -m "refactor(kernels): scheme-generic recurrence kernels (isotropic wrappers preserved)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Scheme-generic Cauchy product (additive; wrapper preserved)

**Files:**
- Modify: `include/tax/kernels/cauchy_stencil.hpp`, `include/tax/kernels/cauchy.hpp`
- Test: existing `tests/kernels/test_cauchy_*`, `tests/operators/test_arith_dense`.

**Interfaces:**
- Produces: a scheme-generic `template<typename T, IndexScheme Scheme> void cauchyProduct(std::array<T,Scheme::nCoeff>& out, const std::array<T,Scheme::nCoeff>& a, const std::array<T,Scheme::nCoeff>& b)`; for `IsotropicScheme<N,M>` it must dispatch to exactly today's `cauchyProductUnroll`(M==1) / `cauchyProductStencil`(M>=2) / loop paths. The legacy `<T,N,M>` `cauchyProduct` stays as a delegating wrapper.

- [ ] **Step 1: Add the scheme-generic dispatch**

The scheme owns its product. Add a thin scheme-generic entry point alongside the existing `cauchyProduct<T,N,M>` (in `cauchy.hpp`, after the legacy declarations) that simply forwards to the scheme:

```cpp
template < typename T, IndexScheme Scheme >
void cauchyProduct( std::array< T, Scheme::nCoeff >& out,
                    const std::array< T, Scheme::nCoeff >& a,
                    const std::array< T, Scheme::nCoeff >& b ) noexcept
{
    Scheme::template cauchyProduct< T >( out, a, b );
}
```

`IsotropicScheme::cauchyProduct` (Step 2) delegates to today's `cauchyProduct<T,N,M>`, so the isotropic path keeps the unchanged unroll(M==1)/stencil(M>=2)/loop dispatch. A later `MixedScheme` supplies its own box-filtered stencil behind the same member (M2).

- [ ] **Step 2: Add `IsotropicScheme::cauchyProduct`**

In `include/tax/core/index_scheme.hpp`, add to `IsotropicScheme`:

```cpp
template < typename T >
static void cauchyProduct( std::array< T, nCoeff >& out, const std::array< T, nCoeff >& a,
                           const std::array< T, nCoeff >& b ) noexcept
{
    detail::kernels::cauchyProduct< T, N, M >( out, a, b );
}
```

(Include `tax/kernels/cauchy.hpp` in `index_scheme.hpp`.)

- [ ] **Step 3: Build the Cauchy + arithmetic tests, expect green**

Run: `cmake --build build -j --target test_cauchy_dense test_cauchy_unroll_diff test_cauchy_stencil_diff test_arith_dense && ctest --test-dir build -R "cauchy|arith_dense" --output-on-failure`
Expected: PASS — identical product results.

- [ ] **Step 4: Full suite gate**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`
Expected: ALL PASS.

- [ ] **Step 5: clang-format and commit**

```bash
cd /Users/andrea/Documents/Codes/tax
clang-format -i include/tax/core/index_scheme.hpp include/tax/kernels/cauchy_stencil.hpp include/tax/kernels/cauchy.hpp
git add include/tax/core/index_scheme.hpp include/tax/kernels/cauchy_stencil.hpp include/tax/kernels/cauchy.hpp
git commit -m "refactor(kernels): scheme-generic Cauchy product (isotropic dispatch preserved)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Add a direct scheme-generic kernel test (lock the abstraction)

**Files:**
- Modify: `tests/core/test_index_scheme.cpp`

**Interfaces:**
- Consumes: the scheme-generic `seriesExp`/`seriesSqrt`/`cauchyProduct` (Tasks 2–3).

- [ ] **Step 1: Add a test calling kernels through the scheme directly**

Append to `tests/core/test_index_scheme.cpp` — verify the scheme-generic kernel produces the same coefficients as the public `TaylorExpansion` math surface for the isotropic case:

```cpp
TEST( IndexScheme, SchemeGenericKernelMatchesPublicSurface )
{
    constexpr int N = 6, M = 2;
    using S = tax::IsotropicScheme< N, M >;
    using TE = tax::TE< N, M >;

    typename TE::Input p{ 0.3, -0.2 };
    auto x = TE::template variable< 0 >( p );
    auto fx = exp( x );  // public surface

    std::array< double, S::nCoeff > a = x.coefficients();
    std::array< double, S::nCoeff > out{};
    tax::detail::kernels::seriesExp< double, S >( out, a );  // scheme-generic kernel

    for ( std::size_t k = 0; k < S::nCoeff; ++k )
        EXPECT_DOUBLE_EQ( out[k], fx[k] );
}
```

- [ ] **Step 2: Build and run, expect green**

Run: `cmake --build build -j --target test_index_scheme && ctest --test-dir build -R test_index_scheme --output-on-failure`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
cd /Users/andrea/Documents/Codes/tax
clang-format -i tests/core/test_index_scheme.cpp
git add tests/core/test_index_scheme.cpp
git commit -m "test(core): scheme-generic kernels match the public isotropic surface

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Final bit-for-bit gate + tidy

**Files:** none (verification only), optional wrapper-doc comment.

- [ ] **Step 1: Full suite, clean build**

Run: `rm -rf build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build --output-on-failure`
Expected: ALL tests PASS from a clean build (no stale objects).

- [ ] **Step 2: Confirm no public-surface drift**

Run: `git diff main -- include/tax/core/taylor_expansion.hpp include/tax/operators`
Expected: EMPTY — Milestone 1 did not change `TaylorExpansion` or the operators (kernels are still reached via the preserved `<T,N,M>` wrappers). The scheme is in place for M2 to consume; migrating `TaylorExpansion`/operators to instantiate kernels directly through `IsotropicScheme` (and removing the wrappers) is deferred to a later milestone since it is not needed for behavior and carries no functional value yet.

- [ ] **Step 3: Confirm formatting is churn-free**

Run: `git diff main --stat` and verify only the intended files changed (no `concepts.hpp` `requires`-brace reflow, no `tax.hpp` include reordering).

## Self-Review notes

- **Spec coverage (M1 only):** this plan covers spec Milestone 1 — the `IndexScheme` abstraction and `IsotropicScheme` with kernels rewired behind preserved wrappers, gated bit-for-bit by the existing suite. Milestones 2–5 (MixedScheme, core MixedExpansion, named layer, la+docs) are explicitly out of this plan and planned just-in-time.
- **Deferred:** migrating `TaylorExpansion`/operators to call kernels through `IsotropicScheme` directly (and deleting the `<T,N,M>` wrappers) is intentionally deferred — the wrappers are zero-cost and M2 only needs the scheme-generic kernels to exist. Folding that migration in here would touch the hot path with no behavioral benefit.
- **Bespoke ops** (`eval`/`deriv`/`integ`/`truncate`) are NOT touched in M1; they only need scheme-aware forms once `MixedScheme` exists (M3). M1 keeps them exactly as-is.

## Execution Handoff

Milestone-1 plan complete. After it lands, I'll write the Milestone-2 plan (`MixedScheme` + mixed stencils) against the now-concrete `IndexScheme` interface.
