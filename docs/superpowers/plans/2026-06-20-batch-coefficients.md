# Batch (SIMD) Coefficients Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a vectorised coefficient type `tax::Batch<T,K>` so one `TaylorExpansion` evaluates K independent problem instances in a single kernel pass, exposed through a unified `TE<N, M, K>` alias.

**Architecture:** `Batch<T,K>` wraps `Eigen::Array<T,K,1,DontAlign>` and provides element-wise arithmetic + transcendental seeds found by the existing dense kernels via ADL. Two opt-in traits (`is_tax_scalar`, `real_scalar`) let `Batch` satisfy the `Scalar` concept and let kernels name real constants generically. `K=1` resolves to plain `double`, so existing code is untouched. Dense storage only.

**Tech Stack:** C++23, header-only, Eigen3, Google Test, CMake. Build inside the `tax` mamba/conda environment (already active).

## Global Constraints

- C++23 required; header-only — all code lives in `include/tax/`.
- No heap allocation in the dense core: `Batch` is a fixed `Eigen::Array` on the stack; dense storage stays `std::array`.
- Batch is **dense storage only** — never wire it into sparse storage (sparse keys off exact-zero coefficients, undefined per-lane).
- `K=1` must resolve to `double` (not `Batch<double,1>`) to preserve the scalar `constexpr` surface and exact semantics.
- Keep existing `concepts.hpp` formatting; do **not** import the prototype's unrelated `requires {` brace reflow.
- Format every touched header with `.clang-format` (Google style, 4-space indent, 100 col, spaces inside `< >` and `( )`) before committing: `clang-format -i <files>`.
- Kernel config macros stay in-header; introduce no build-system `-D` macros.
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Work on branch `feature/batch-coefficients` (already checked out).

## File Structure

- `include/tax/core/concepts.hpp` — **modify**: add `is_tax_scalar`, `real_scalar`/`real_scalar_t`; widen `Scalar`.
- `include/tax/core/batch.hpp` — **create**: the `Batch<T,K>` type, element-wise math, trait opt-ins, `Batchd`/`Batchf` aliases, `Eigen::NumTraits<Batch>`.
- `include/tax/core/taylor_expansion.hpp` — **modify**: forward-declare `Batch`; fold `K` into the public `TE` alias.
- `include/tax/kernels/transcendental.hpp` — **modify**: name the `erf` constant via `real_scalar_t<T>`, drop `constexpr` to `const`.
- `include/tax/tax.hpp` — **modify**: include `batch.hpp` after `taylor_expansion.hpp`.
- `tests/core/test_batch.cpp` — **create**: lane-equivalence tests (built up across Tasks 2–4).
- `tests/CMakeLists.txt` — **modify**: register `test_batch`.
- `docs/guide/batch.md` — **create**: usage doc.
- `mkdocs.yml` — **modify**: add the doc to the Guide nav.

Note: `include/tax/operators/math_binary.hpp` is expected to need **no change** — current `main`'s real-exponent `pow` already takes a separate `std::floating_point P` parameter, so `pow(batch_x, 2.5)` resolves to the real kernel and `pow(batch_x, 3)` to the integer kernel with no ambiguity. Task 3 verifies this; a fallback overload is provided only if the test fails.

## Build & test reference (run from repo root `/Users/andrea/Documents/Codes/tax`)

- Configure (once, if `build/` is stale or missing): `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
- Build a single test target: `cmake --build build -j --target test_batch`
- Run one test's cases: `ctest --test-dir build -R test_batch --output-on-failure`
- Filter a single case: `./build/tests/test_batch --gtest_filter='Batch.ScalarLaneArithmetic'`
- Full suite (regression gate): `cmake --build build -j && ctest --test-dir build --output-on-failure`

---

### Task 1: Coefficient opt-in traits in `concepts.hpp`

**Files:**
- Modify: `include/tax/core/concepts.hpp:1-11`
- Test: full existing suite (regression gate) + compile-time `static_assert`s added in Task 2's test file.

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `template<typename T> struct is_tax_scalar : std::bool_constant<std::floating_point<T>> {};`
  - `template<typename T> concept Scalar = is_tax_scalar<T>::value;`
  - `template<typename T> struct real_scalar { using type = T; };`
  - `template<typename T> using real_scalar_t = typename real_scalar<T>::type;`

- [ ] **Step 1: Add the traits and widen `Scalar`**

Replace the current top of `include/tax/core/concepts.hpp`:

```cpp
#pragma once

#include <concepts>
#include <cstddef>

namespace tax
{

/// Scalar constraint for coefficients and function values.
template < typename T >
concept Scalar = std::floating_point< T >;
```

with:

```cpp
#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace tax
{

/// @brief Opt-in trait marking a type usable as a TaylorExpansion coefficient.
///
/// Every floating-point type qualifies by default. A non-floating coefficient
/// type (e.g. a SIMD batch wrapper with an element-wise scalar-like math API)
/// opts in by specialising this to `std::true_type`, keeping `Scalar` a strict
/// superset of `std::floating_point`.
template < typename T >
struct is_tax_scalar : std::bool_constant< std::floating_point< T > >
{
};

/// Scalar constraint for coefficients and function values.
template < typename T >
concept Scalar = is_tax_scalar< T >::value;

/// @brief Underlying real (floating-point) scalar of a coefficient type.
///
/// Identity for ordinary floating-point coefficients; a batch coefficient type
/// specialises this to its lane scalar so generic kernels can name real
/// constants (e.g. `std::numbers::inv_sqrtpi_v< real_scalar_t< T > >`).
template < typename T >
struct real_scalar
{
    using type = T;
};

template < typename T >
using real_scalar_t = typename real_scalar< T >::type;
```

Leave the rest of the file (the `TaylorPolynomial` / `DensePolynomial` / `SparsePolynomial` concepts) exactly as-is — do **not** reflow their `requires(...)` braces.

- [ ] **Step 2: Verify the existing suite still builds and passes**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`
Expected: all existing tests PASS (the widened `Scalar` is behaviourally identical for floating-point types).

- [ ] **Step 3: clang-format and commit**

```bash
cd /Users/andrea/Documents/Codes/tax
clang-format -i include/tax/core/concepts.hpp
git add include/tax/core/concepts.hpp
git commit -m "feat(core): opt-in is_tax_scalar / real_scalar coefficient traits

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `Batch` type, unified `TE` alias, value-semantics tests

**Files:**
- Create: `include/tax/core/batch.hpp`
- Modify: `include/tax/core/taylor_expansion.hpp:1-11` (includes), add a `Batch` forward declaration after line 13, and replace the `TE` alias at `:404-406`.
- Modify: `include/tax/tax.hpp:9` (add include after `taylor_expansion.hpp`)
- Create: `tests/core/test_batch.cpp`
- Modify: `tests/CMakeLists.txt` (register `test_batch`)

**Interfaces:**
- Consumes: `is_tax_scalar`, `real_scalar` (Task 1).
- Produces:
  - `tax::Batch<T,K>` with: `lane_type=T`, `static constexpr int lanes=K`, `array_t=Eigen::Array<T,K,1,Eigen::DontAlign>`, member `v`; ctors `Batch()` (zero), `Batch(T)` (broadcast), `Batch(const array_t&)`; `static Batch fromLanes(const std::array<T,K>&)`; `operator[](int)`, `lane(int)`; `+= -= *= /=`; free `+ - * /`, unary `-`, `== !=`; element-wise `sqrt exp log sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh abs cbrt erf pow atan2`.
  - `is_tax_scalar<Batch<T,K>> : std::true_type`, `real_scalar<Batch<T,K>>::type = T`.
  - `tax::Batchd<K> = Batch<double,K>`, `tax::Batchf<K> = Batch<float,K>`.
  - Unified `tax::TE<int N, int M=1, int K=1>` = `TaylorExpansion< conditional_t<K==1, double, Batch<double,K>>, N, M, Dense >`.
  - `Eigen::NumTraits<tax::Batch<T,K>>`.

- [ ] **Step 1: Create `include/tax/core/batch.hpp`**

Write the file exactly as below:

```cpp
#pragma once

// ---------------------------------------------------------------------------
// Vectorised ("batched" / SIMD) Taylor-expansion coefficients.
// ---------------------------------------------------------------------------
//
// `Batch< T, K >` packs K independent floating-point problem instances into one
// coefficient slot, backed by an Eigen fixed-size array. Every operation a
// TaylorExpansion needs -- the four arithmetic operators, unary minus, and the
// transcendental seeds the recurrence kernels evaluate on the constant term --
// is element-wise across the K lanes. Substituting `Batch< T, K >` for the
// scalar coefficient type therefore makes
//
//     TaylorExpansion< Batch< double, K >, N, M >   (== tax::TE< N, M, K >)
//
// evaluate K independent expansions in lock-step: one pass through the kernels,
// K results, with the inner element-wise work vectorised. Restricted to dense
// storage: sparse storage keys off exact-zero coefficients, which is not well
// defined per-lane.

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <tax/core/concepts.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <type_traits>

namespace tax
{

/**
 * @brief K-lane SIMD-friendly scalar: K independent `T` instances.
 *
 * @tparam T  Lane scalar (a floating-point type).
 * @tparam K  Number of lanes (>= 1).
 */
template < typename T, int K >
struct Batch
{
    static_assert( K >= 1, "Batch lane count must be >= 1" );
    static_assert( std::floating_point< T >, "Batch lane type must be floating-point" );

    using lane_type = T;
    static constexpr int lanes = K;
    using array_t = Eigen::Array< T, K, 1, Eigen::DontAlign >;

    array_t v;

    Batch() noexcept : v( array_t::Zero() ) {}
    /*implicit*/ Batch( T s ) noexcept : v( array_t::Constant( s ) ) {}
    /*implicit*/ Batch( const array_t& a ) noexcept : v( a ) {}

    /// @brief Build from K explicit lane values.
    [[nodiscard]] static Batch fromLanes( const std::array< T, std::size_t( K ) >& a ) noexcept
    {
        Batch r;
        for ( int i = 0; i < K; ++i ) r.v[i] = a[std::size_t( i )];
        return r;
    }

    [[nodiscard]] T operator[]( int i ) const noexcept { return v[i]; }
    [[nodiscard]] T& operator[]( int i ) noexcept { return v[i]; }
    [[nodiscard]] T lane( int i ) const noexcept { return v[i]; }

    Batch& operator+=( const Batch& o ) noexcept
    {
        v += o.v;
        return *this;
    }
    Batch& operator-=( const Batch& o ) noexcept
    {
        v -= o.v;
        return *this;
    }
    Batch& operator*=( const Batch& o ) noexcept
    {
        v *= o.v;
        return *this;
    }
    Batch& operator/=( const Batch& o ) noexcept
    {
        v /= o.v;
        return *this;
    }
};

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------
template < typename T, int K >
[[nodiscard]] inline Batch< T, K > operator+( Batch< T, K > a, const Batch< T, K >& b ) noexcept
{
    a += b;
    return a;
}
template < typename T, int K >
[[nodiscard]] inline Batch< T, K > operator-( Batch< T, K > a, const Batch< T, K >& b ) noexcept
{
    a -= b;
    return a;
}
template < typename T, int K >
[[nodiscard]] inline Batch< T, K > operator*( Batch< T, K > a, const Batch< T, K >& b ) noexcept
{
    a *= b;
    return a;
}
template < typename T, int K >
[[nodiscard]] inline Batch< T, K > operator/( Batch< T, K > a, const Batch< T, K >& b ) noexcept
{
    a /= b;
    return a;
}
template < typename T, int K >
[[nodiscard]] inline Batch< T, K > operator-( const Batch< T, K >& a ) noexcept
{
    return Batch< T, K >( ( -a.v ).eval() );
}

// Comparisons collapse all lanes (used only by the dense deriv/integ zero-skip
// paths: a coefficient is skipped only when every lane is exactly zero).
template < typename T, int K >
[[nodiscard]] inline bool operator==( const Batch< T, K >& a, const Batch< T, K >& b ) noexcept
{
    return ( a.v == b.v ).all();
}
template < typename T, int K >
[[nodiscard]] inline bool operator!=( const Batch< T, K >& a, const Batch< T, K >& b ) noexcept
{
    return !( a == b );
}

// ---------------------------------------------------------------------------
// Element-wise math (found by ADL from the kernels). Common functions use
// Eigen's vectorised array methods; the few Eigen-core lacks (cbrt, erf,
// atan2) fall back to a per-lane apply.
// ---------------------------------------------------------------------------
#define TAX_BATCH_UNARY( fn )                                                \
    template < typename T, int K >                                           \
    [[nodiscard]] inline Batch< T, K > fn( const Batch< T, K >& a ) noexcept \
    {                                                                        \
        return Batch< T, K >( a.v.fn().eval() );                             \
    }

TAX_BATCH_UNARY( sqrt )
TAX_BATCH_UNARY( exp )
TAX_BATCH_UNARY( log )
TAX_BATCH_UNARY( sin )
TAX_BATCH_UNARY( cos )
TAX_BATCH_UNARY( tan )
TAX_BATCH_UNARY( asin )
TAX_BATCH_UNARY( acos )
TAX_BATCH_UNARY( atan )
TAX_BATCH_UNARY( sinh )
TAX_BATCH_UNARY( cosh )
TAX_BATCH_UNARY( tanh )
TAX_BATCH_UNARY( asinh )
TAX_BATCH_UNARY( acosh )
TAX_BATCH_UNARY( atanh )
TAX_BATCH_UNARY( abs )
#undef TAX_BATCH_UNARY

template < typename T, int K >
[[nodiscard]] inline Batch< T, K > pow( const Batch< T, K >& a, const Batch< T, K >& b ) noexcept
{
    return Batch< T, K >( a.v.pow( b.v ).eval() );
}

template < typename T, int K >
[[nodiscard]] inline Batch< T, K > cbrt( const Batch< T, K >& a ) noexcept
{
    return Batch< T, K >( a.v.unaryExpr( []( T x ) {
                                 using std::cbrt;
                                 return cbrt( x );
                             } )
                              .eval() );
}
template < typename T, int K >
[[nodiscard]] inline Batch< T, K > erf( const Batch< T, K >& a ) noexcept
{
    return Batch< T, K >( a.v.unaryExpr( []( T x ) {
                                 using std::erf;
                                 return erf( x );
                             } )
                              .eval() );
}
template < typename T, int K >
[[nodiscard]] inline Batch< T, K > atan2( const Batch< T, K >& y, const Batch< T, K >& x ) noexcept
{
    return Batch< T, K >( y.v.binaryExpr( x.v,
                                          []( T yy, T xx ) {
                                              using std::atan2;
                                              return atan2( yy, xx );
                                          } )
                              .eval() );
}

// ---------------------------------------------------------------------------
// Coefficient-trait opt-ins
// ---------------------------------------------------------------------------
template < typename T, int K >
struct is_tax_scalar< Batch< T, K > > : std::true_type
{
};

template < typename T, int K >
struct real_scalar< Batch< T, K > >
{
    using type = T;
};

// ---------------------------------------------------------------------------
// Convenience aliases
// ---------------------------------------------------------------------------

/// @brief `Batchd<K>` — K-lane double batch.
template < int K >
using Batchd = Batch< double, K >;

/// @brief `Batchf<K>` — K-lane float batch.
template < int K >
using Batchf = Batch< float, K >;

}  // namespace tax

// ---------------------------------------------------------------------------
// Eigen scalar traits — lets Eigen matrices/vectors hold Batch (and, by
// extension, TaylorExpansion<Batch,...>) as their scalar type.
// ---------------------------------------------------------------------------
namespace Eigen
{

template < typename T, int K >
struct NumTraits< tax::Batch< T, K > > : NumTraits< T >
{
    using Self = tax::Batch< T, K >;
    using Real = Self;
    using NonInteger = Self;
    using Nested = Self;
    using Literal = Self;
    enum
    {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = K,
        AddCost = K,
        MulCost = K
    };
    static Self epsilon() { return Self( NumTraits< T >::epsilon() ); }
    static Self dummy_precision() { return Self( NumTraits< T >::dummy_precision() ); }
    static Self highest() { return Self( NumTraits< T >::highest() ); }
    static Self lowest() { return Self( NumTraits< T >::lowest() ); }
    static int digits10() { return NumTraits< T >::digits10(); }
};

}  // namespace Eigen
```

- [ ] **Step 2: Forward-declare `Batch` and unify the `TE` alias in `taylor_expansion.hpp`**

Add `#include <type_traits>` to the include block (after `#include <stdexcept>` at `:6`):

```cpp
#include <stdexcept>
#include <type_traits>
```

Add a forward declaration of `Batch` immediately after `namespace tax {` opens (after line 13, before the primary-template forward declaration):

```cpp
namespace tax
{

// Forward declaration so the public `TE` alias can name the batched coefficient
// type without including <tax/core/batch.hpp> (which includes this header).
template < typename T, int K >
struct Batch;
```

Replace the existing alias at `:404-406`:

```cpp
/// `TE<N>` — univariate `double` expansion of order N.
template < int N, int M = 1 >
using TE = TaylorExpansion< double, N, M, storage::Dense >;
```

with the unified form:

```cpp
/// `TE<N, M, K>` — order-N, M-variate dense `double` expansion.
///
/// `K == 1` (default) is a plain `double` expansion. `K > 1` makes each
/// coefficient a `Batch< double, K >`, evaluating K independent expansions in
/// lock-step. `M` defaults to 1.
template < int N, int M = 1, int K = 1 >
using TE = TaylorExpansion< std::conditional_t< K == 1, double, Batch< double, K > >, N, M,
                            storage::Dense >;
```

Leave `TEn` and `STE` unchanged.

- [ ] **Step 3: Include `batch.hpp` from the umbrella**

In `include/tax/tax.hpp`, add the include immediately after the `taylor_expansion.hpp` line:

```cpp
#include <tax/core/taylor_expansion.hpp>
#include <tax/core/batch.hpp>
#include <tax/core/named.hpp>
```

- [ ] **Step 4: Write the failing value-semantics tests**

Create `tests/core/test_batch.cpp`:

```cpp
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <tax/tax.hpp>

using tax::Batch;

namespace
{
constexpr int K = 4;

// Per-lane expansion centre for univariate tests.
double center( int k ) { return 0.25 + 0.05 * double( k ); }
}  // namespace

// Trait sanity: floating types still satisfy Scalar; real_scalar is identity
// for doubles and the lane type for batches.
static_assert( tax::Scalar< double > );
static_assert( tax::Scalar< tax::Batch< double, 4 > > );
static_assert( std::is_same_v< tax::real_scalar_t< double >, double > );
static_assert( std::is_same_v< tax::real_scalar_t< tax::Batch< double, 4 > >, double > );

TEST( Batch, ScalarLaneArithmetic )
{
    Batch< double, K > a, b;
    for ( int k = 0; k < K; ++k )
    {
        a[k] = 1.0 + k;
        b[k] = 0.5 * ( k + 1 );
    }
    const auto s = a + b;
    const auto p = a * b;
    const auto q = a / b;
    for ( int k = 0; k < K; ++k )
    {
        EXPECT_DOUBLE_EQ( s[k], a[k] + b[k] );
        EXPECT_DOUBLE_EQ( p[k], a[k] * b[k] );
        EXPECT_DOUBLE_EQ( q[k], a[k] / b[k] );
    }
    EXPECT_TRUE( a == a );
    EXPECT_TRUE( a != b );
}

TEST( Batch, ScalarBroadcastAndFromLanes )
{
    Batch< double, K > c( 3.0 );
    for ( int k = 0; k < K; ++k ) EXPECT_DOUBLE_EQ( c[k], 3.0 );
    auto d = Batch< double, K >::fromLanes( { 1.0, 2.0, 3.0, 4.0 } );
    for ( int k = 0; k < K; ++k ) EXPECT_DOUBLE_EQ( d[k], double( k + 1 ) );
}

// A batched expansion built through the unified TE<N, M, K> alias evaluates,
// per lane, exactly like the matching scalar TE<N, M>.
TEST( Batch, UnifiedAliasConstructAndValue )
{
    constexpr int N = 5;
    using TEb = tax::TE< N, 1, K >;   // batched
    using TEs = tax::TE< N, 1 >;      // scalar (K defaults to 1 -> double)

    static_assert( std::is_same_v< typename TEs::scalar_type, double > );
    static_assert( std::is_same_v< typename TEb::scalar_type, tax::Batch< double, K > > );

    typename TEb::Input pb{};
    Batch< double, K > c0;
    for ( int k = 0; k < K; ++k ) c0[k] = center( k );
    pb[0] = c0;
    TEb xb = TEb::template variable< 0 >( pb );

    for ( int k = 0; k < K; ++k )
    {
        typename TEs::Input ps{ center( k ) };
        TEs xs = TEs::template variable< 0 >( ps );
        EXPECT_DOUBLE_EQ( xb.value()[k], xs.value() );
        EXPECT_DOUBLE_EQ( xb[1][k], xs[1] );  // linear coefficient
    }
}
```

- [ ] **Step 5: Register the test target**

In `tests/CMakeLists.txt`, add after the `test_named` line:

```cmake
tax_add_test(test_named SOURCES core/test_named.cpp)
tax_add_test(test_batch SOURCES core/test_batch.cpp)
```

- [ ] **Step 6: Configure, build, and verify the tests pass**

Run:
```bash
cd /Users/andrea/Documents/Codes/tax
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target test_batch
ctest --test-dir build -R test_batch --output-on-failure
```
Expected: `test_batch` builds; `Batch.ScalarLaneArithmetic`, `Batch.ScalarBroadcastAndFromLanes`, `Batch.UnifiedAliasConstructAndValue` PASS.

- [ ] **Step 7: Verify scalar `TE` is unchanged (regression gate)**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`
Expected: the full existing suite still PASSES (the `TE` alias change is a no-op for `K=1`).

- [ ] **Step 8: clang-format and commit**

```bash
cd /Users/andrea/Documents/Codes/tax
clang-format -i include/tax/core/batch.hpp include/tax/core/taylor_expansion.hpp include/tax/tax.hpp tests/core/test_batch.cpp
git add include/tax/core/batch.hpp include/tax/core/taylor_expansion.hpp include/tax/tax.hpp tests/core/test_batch.cpp tests/CMakeLists.txt
git commit -m "feat(core): add Batch<T,K> coefficients and unified TE<N,M,K> alias

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Generalise the `erf` constant; verify the full math surface

**Files:**
- Modify: `include/tax/kernels/transcendental.hpp` (the `seriesErf` constant at `:287`; add include if needed)
- Modify: `tests/core/test_batch.cpp` (append math-surface tests)
- Possibly modify: `include/tax/operators/math_binary.hpp` (only if Step 3 fails)

**Interfaces:**
- Consumes: `Batch`, unified `TE` (Task 2); `real_scalar_t` (Task 1).
- Produces: a `seriesErf` that compiles and is correct for `T = Batch<double,K>`.

- [ ] **Step 1: Write the failing math-surface tests**

Append to `tests/core/test_batch.cpp`:

```cpp
// Compare every coefficient of a batched univariate result against the K
// independent scalar runs it should reproduce.
namespace
{
template < int N, class Fn >
double laneMaxErr( Fn f )
{
    using TEs = tax::TE< N, 1 >;
    using TEb = tax::TE< N, 1, K >;

    typename TEb::Input pb{};
    Batch< double, K > c0;
    for ( int k = 0; k < K; ++k ) c0[k] = center( k );
    pb[0] = c0;
    TEb xb = TEb::template variable< 0 >( pb );
    const TEb rb = f( xb );

    double max_err = 0.0;
    for ( int k = 0; k < K; ++k )
    {
        typename TEs::Input ps{ center( k ) };
        TEs xs = TEs::template variable< 0 >( ps );
        const TEs rs = f( xs );
        for ( std::size_t c = 0; c < TEs::nCoefficients; ++c )
        {
            const double d = std::abs( rb[c][k] - rs[c] ) / ( 1.0 + std::abs( rs[c] ) );
            max_err = std::max( max_err, d );
        }
    }
    return max_err;
}
}  // namespace

TEST( Batch, MathSurfaceLaneEquivalence )
{
    using tax::acos;
    using tax::asin;
    using tax::atan;
    using tax::atan2;
    using tax::cbrt;
    using tax::cos;
    using tax::cosh;
    using tax::erf;
    using tax::exp;
    using tax::log;
    using tax::pow;
    using tax::sin;
    using tax::sinh;
    using tax::sqrt;
    using tax::tan;
    using tax::tanh;

    constexpr double tol = 1e-12;
    EXPECT_LT( laneMaxErr< 6 >( []( auto x ) { return x * x * x + x - 2.0; } ), tol );
    EXPECT_LT( laneMaxErr< 6 >( []( auto x ) { return ( x * x + 1.0 ) / ( x + 2.0 ); } ), tol );
    EXPECT_LT( laneMaxErr< 6 >( []( auto x ) { return sin( x ) * cos( x ) + tan( x ); } ), tol );
    EXPECT_LT( laneMaxErr< 6 >( []( auto x ) { return asin( x ) + acos( x ) + atan( x ); } ), tol );
    EXPECT_LT( laneMaxErr< 6 >( []( auto x ) { return exp( x ) + log( x + 2.0 ); } ), tol );
    EXPECT_LT( laneMaxErr< 6 >( []( auto x ) { return sinh( x ) + cosh( x ) + tanh( x ); } ), tol );
    EXPECT_LT( laneMaxErr< 6 >( []( auto x ) { return sqrt( x + 1.0 ) + cbrt( x + 1.0 ); } ), tol );
    EXPECT_LT( laneMaxErr< 6 >( []( auto x ) { return erf( x ); } ), tol );
    EXPECT_LT( laneMaxErr< 6 >( []( auto x ) { return atan2( x + 1.0, x + 2.0 ); } ), tol );
    EXPECT_LT( laneMaxErr< 6 >( []( auto x ) { return pow( x + 1.0, 3 ); } ), tol );    // int
    EXPECT_LT( laneMaxErr< 6 >( []( auto x ) { return pow( x + 1.0, 2.5 ); } ), tol );  // real
}

TEST( Batch, IntegerVsRealPowSelectsCorrectKernel )
{
    using tax::pow;
    using TEb = tax::TE< 5, 1, K >;
    typename TEb::Input pb{};
    pb[0] = Batch< double, K >( 0.3 );
    auto x = TEb::template variable< 0 >( pb ) + 1.0;  // centre 1.3
    auto pi = pow( x, 2 );                             // integer exponent
    auto pr = pow( x, 2.0 );                           // real exponent, same value
    for ( int k = 0; k < K; ++k )
    {
        EXPECT_NEAR( pi[0][k], 1.3 * 1.3, 1e-12 );
        EXPECT_NEAR( pr[0][k], std::pow( 1.3, 2.0 ), 1e-12 );
        for ( std::size_t c = 0; c < TEb::nCoefficients; ++c )
            EXPECT_NEAR( pi[c][k], pr[c][k], 1e-10 );
    }
}

TEST( Batch, EvalPerLaneDisplacement )
{
    constexpr int N = 5;
    using TEb = tax::TE< N, 1, K >;
    using TEs = tax::TE< N, 1 >;
    using tax::sin;

    typename TEb::Input pb{};
    Batch< double, K > c0;
    for ( int k = 0; k < K; ++k ) c0[k] = center( k );
    pb[0] = c0;
    auto fb = sin( TEb::template variable< 0 >( pb ) );

    Batch< double, K > dxb;
    for ( int k = 0; k < K; ++k ) dxb[k] = 0.01 * ( k + 1 );
    typename TEb::Input db{ dxb };
    auto eb = fb.eval( db );

    for ( int k = 0; k < K; ++k )
    {
        typename TEs::Input ps{ center( k ) };
        auto fs = sin( TEs::template variable< 0 >( ps ) );
        typename TEs::Input ds{ 0.01 * ( k + 1 ) };
        EXPECT_NEAR( eb[k], fs.eval( ds ), 1e-12 );
    }
}
```

- [ ] **Step 2: Run to confirm the build fails on `erf`**

Run: `cmake --build build -j --target test_batch 2>&1 | head -30`
Expected: COMPILE ERROR inside `seriesErf` — `std::numbers::inv_sqrtpi_v` is ill-formed for `T = tax::Batch<double,K>` (it requires a literal floating-point type), and the `constexpr` `Batch` initialiser is not a constant expression.

- [ ] **Step 3: Generalise the `erf` constant**

In `include/tax/kernels/transcendental.hpp`, ensure the concepts header is available — add to the include block if not already present:

```cpp
#include <tax/core/concepts.hpp>
```

Then in `seriesErf`, replace:

```cpp
    constexpr T two_over_sqrtpi = T{ 2 } * std::numbers::inv_sqrtpi_v< T >;
```

with:

```cpp
    // Name the constant in the underlying real scalar so vectorised coefficient
    // types (whose lanes are floating-point) work too; broadcast into T.
    using R = real_scalar_t< T >;
    const T two_over_sqrtpi = T( R{ 2 } * std::numbers::inv_sqrtpi_v< R > );
```

(`erf` is already a runtime kernel — `std::erf` is not `constexpr` — so dropping `constexpr` to `const` costs nothing for scalar `T`; for `T=double`, `R=double` and the value is unchanged.)

- [ ] **Step 4: Build and run the math-surface tests**

Run:
```bash
cmake --build build -j --target test_batch
ctest --test-dir build -R test_batch --output-on-failure
```
Expected: all `Batch.*` cases PASS, including `MathSurfaceLaneEquivalence`, `IntegerVsRealPowSelectsCorrectKernel`, `EvalPerLaneDisplacement`.

If — and only if — `IntegerVsRealPowSelectsCorrectKernel` or the `pow(..., 2.5)` line of `MathSurfaceLaneEquivalence` fails to compile or selects the wrong kernel, add this guarded overload to `include/tax/operators/math_binary.hpp` after the existing real-exponent `pow` (and `#include <tax/core/concepts.hpp>` there):

```cpp
/// Real-exponent `pow` taking the exponent in the underlying real scalar.
/// Only for vectorised coefficient types (`real_scalar_t<T> != T`); lets a
/// plain floating literal reach the real-exponent kernel instead of binding to
/// the integer overload. For scalar `T` this overload does not exist.
template < typename T, int N, int M >
    requires( !std::is_same_v< real_scalar_t< T >, T > )
[[nodiscard]] TaylorExpansion< T, N, M > pow( const TaylorExpansion< T, N, M >& x,
                                              real_scalar_t< T > p ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesPow< T, N, M >( r.coefficients(), x.coefficients(), T( p ) );
    return r;
}
```

Then rebuild and re-run until the tests pass.

- [ ] **Step 5: Run the regression gate**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`
Expected: full suite PASSES (notably the existing `test_erf` for scalar coefficients is unaffected by the `constexpr`→`const` change).

- [ ] **Step 6: clang-format and commit**

```bash
cd /Users/andrea/Documents/Codes/tax
clang-format -i include/tax/kernels/transcendental.hpp tests/core/test_batch.cpp
# include math_binary.hpp in the line below only if you edited it in Step 4
git add include/tax/kernels/transcendental.hpp tests/core/test_batch.cpp
git commit -m "feat(kernels): make erf constant batch-generic; cover batch math surface

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `tax::la` integration for batched expansions

**Files:**
- Modify: `tests/core/test_batch.cpp` (append the la / Eigen-interop test)
- Possibly modify: a `tax::la` header (only if the test surfaces a gap)

**Interfaces:**
- Consumes: unified `TE` (Task 2); `Eigen::NumTraits<Batch>` (Task 2); `tax::la::value`, `tax::la::jacobian`, `tax::la::VecNT`, member `TaylorExpansion::gradient()`.
- Produces: confirmation that the `tax::la` surface works per-lane for batched expansions.

- [ ] **Step 1: Write the failing la / Eigen-interop test**

Append to `tests/core/test_batch.cpp`:

```cpp
TEST( Batch, MultivariateGradientAndEigenInterop )
{
    constexpr int N = 4, M = 2;
    using TEb = tax::TE< N, M, K >;
    using TEs = tax::TE< N, M >;
    using tax::exp;
    using tax::sin;

    auto lane = []( int k ) { return std::array< double, 2 >{ 0.3 + 0.05 * k, -0.2 + 0.03 * k }; };

    typename TEb::Input pb{};
    Batch< double, K > c0, c1;
    for ( int k = 0; k < K; ++k )
    {
        c0[k] = lane( k )[0];
        c1[k] = lane( k )[1];
    }
    pb[0] = c0;
    pb[1] = c1;
    auto xb = TEb::template variable< 0 >( pb );
    auto yb = TEb::template variable< 1 >( pb );

    auto fb = sin( xb * yb ) + exp( xb );

    // member gradient -> Eigen vector of Batch (needs NumTraits<Batch>)
    auto gb = fb.gradient();

    // la helpers over an Eigen vector state of batched expansions
    tax::la::VecNT< 2, TEb > Fb;
    Fb( 0 ) = fb;
    Fb( 1 ) = xb - yb;
    auto valb = tax::la::value( Fb );
    auto Jb = tax::la::jacobian( Fb );

    for ( int k = 0; k < K; ++k )
    {
        auto L = lane( k );
        typename TEs::Input ps{ L[0], L[1] };
        auto xs = TEs::template variable< 0 >( ps );
        auto ys = TEs::template variable< 1 >( ps );
        auto fs = sin( xs * ys ) + exp( xs );
        auto gs = fs.gradient();

        EXPECT_NEAR( valb( 0 )[k], fs.value(), 1e-12 );
        for ( int i = 0; i < 2; ++i )
        {
            EXPECT_NEAR( gb( i )[k], gs( i ), 1e-12 );
            EXPECT_NEAR( Jb( 0, i )[k], gs( i ), 1e-12 );
        }
    }
}
```

- [ ] **Step 2: Build and run**

Run:
```bash
cmake --build build -j --target test_batch
ctest --test-dir build -R test_batch --output-on-failure
```
Expected: `Batch.MultivariateGradientAndEigenInterop` PASSES. `tax::la` is already scalar-generic (every helper derives `T`/`scalar_type` from traits; no hardcoded `double`), so no la source change is anticipated.

If a `tax::la` helper fails to compile or run with a batched `TE`, fix it **narrowly** at the failing site — prefer deriving the scalar via `real_scalar_t`/`scalar_type` over assuming `double`, and never assume an ordered or streamable scalar. Keep scalar behaviour identical. Rebuild and re-run until it passes.

- [ ] **Step 3: Regression gate**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`
Expected: full suite PASSES (including the existing `tests/eigen/` la tests for scalar coefficients).

- [ ] **Step 4: clang-format and commit**

```bash
cd /Users/andrea/Documents/Codes/tax
clang-format -i tests/core/test_batch.cpp
# add any la header you had to touch in Step 2
git add tests/core/test_batch.cpp
git commit -m "test(eigen): verify tax::la works per-lane for batched expansions

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Documentation

**Files:**
- Create: `docs/guide/batch.md`
- Modify: `mkdocs.yml` (add the page to the Guide nav, `:110-116`)

**Interfaces:**
- Consumes: the public surface (`Batch`, `Batchd`/`Batchf`, unified `TE<N,M,K>`, the math surface, `tax::la` interop).
- Produces: user-facing documentation.

- [ ] **Step 1: Create `docs/guide/batch.md`**

Write the file:

````markdown
# Batch (SIMD) coefficients

A `TaylorExpansion` is generic in its coefficient type. Besides `double` /
`float`, the library ships a **batched** coefficient type that packs `K`
independent problem instances into every coefficient slot:

```cpp
tax::Batch<T, K>   // K lanes of floating-point scalar T (Eigen-backed)
```

Substituting it for the scalar coefficient makes a single expansion carry `K`
independent expansions that share the same monomial structure. The unified `TE`
alias selects it through a trailing lane-count parameter:

```cpp
tax::TE<N, M, K>   // = TaylorExpansion<Batch<double, K>, N, M>; K defaults to 1 (plain double)
```

All recurrence kernels run **once** and produce all `K` results, with the inner
element-wise work vectorised by Eigen / the compiler. This is ideal for
ensemble / Monte-Carlo propagation, parameter sweeps, and ADS sub-boxes that
share an expansion shape.

## Aliases

| Alias | Meaning |
|-------|---------|
| `tax::Batch<T, K>`   | `K`-lane coefficient with lane scalar `T` |
| `tax::Batchd<K>`     | `Batch<double, K>` |
| `tax::Batchf<K>`     | `Batch<float, K>` |
| `tax::TE<N, M, K>`   | `TaylorExpansion<Batch<double, K>, N, M>` (`M`, `K` default to 1) |

## Usage

```cpp
#include <tax/tax.hpp>

constexpr int N = 6, M = 1, K = 4;
using TE = tax::TE<N, M, K>;

// Per-lane expansion centre: lanes evaluate four different problems at once.
typename TE::Input p{};
tax::Batch<double, K> x0;
for (int k = 0; k < K; ++k) x0[k] = 0.2 + 0.05 * k;
p[0] = x0;

auto x = TE::variable<0>(p);
auto f = sin(x) * exp(x);          // whole math surface works

double lane2_value = f.value()[2]; // read a single lane
```

The full unary/binary math surface is supported: `+ - * /`, `sqrt`, `cbrt`,
`exp`, `log`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, the hyperbolics and
their inverses, `erf`, `pow` (integer and real exponent), and `atan2`. Every
lane is bit-for-bit identical to the equivalent scalar `TaylorExpansion`
computation.

`Eigen::NumTraits<Batch<...>>` is specialised, so batched expansions work as
Eigen scalars: `tax::la::value`, `gradient`, `jacobian`, and
`tax::la::VecNT<D, tax::TE<N, M, K>>` states all behave as expected (per lane).

## Notes & limits

- **Dense storage only.** Sparse storage keys off exact-zero coefficients,
  which is not well defined per lane.
- **Real-exponent `pow`.** `pow(x, 2.5)` selects the real-exponent kernel and
  `pow(x, 3)` the integer one, exactly as for scalar coefficients.
- **`Batch` is a runtime SIMD type** (Eigen-backed), so a batched expansion is
  not usable in `constexpr` evaluation, unlike a scalar one.
- The two enabling core hooks are the `is_tax_scalar` / `real_scalar` traits in
  `core/concepts.hpp`; any user type that presents the same element-wise math
  surface can opt in the same way.
````

- [ ] **Step 2: Add the page to the Guide nav**

In `mkdocs.yml`, add the entry under `Guide:` (after `Eigen Integration`):

```yaml
      - Eigen Integration: guide/eigen.md
      - Batch (SIMD) Coefficients: guide/batch.md
```

- [ ] **Step 3: Verify the doc builds (if mkdocs is available)**

Run: `mkdocs build --strict 2>&1 | tail -20` (skip if `mkdocs` is not installed in the environment — it is a docs-only check, not a code gate).
Expected: build succeeds with no warnings about `guide/batch.md` being missing from the nav or unreferenced.

- [ ] **Step 4: Commit**

```bash
cd /Users/andrea/Documents/Codes/tax
git add docs/guide/batch.md mkdocs.yml
git commit -m "docs(guide): document batch (SIMD) coefficients

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

- [ ] **Full suite green:** `cmake --build build -j && ctest --test-dir build --output-on-failure` — all tests PASS, including `test_batch`.
- [ ] **Formatting clean:** `clang-format -i $(git ls-files 'include/**/*.hpp' 'tests/**/*.cpp')` produces no diff (`git diff --quiet`).
- [ ] **No new core allocation:** `Batch` adds only a stack `Eigen::Array`; no `new`/`std::vector` introduced in dense headers (`git diff main -- include/tax/core include/tax/kernels include/tax/operators | grep -nE 'new |std::vector' ` returns nothing batch-related).
- [ ] **Scalar `TE` untouched:** `TE<N>` / `TE<N,M>` still resolve to `double` expansions (covered by the regression gate and `Batch.UnifiedAliasConstructAndValue`).

## Out of scope (do not implement here)

- `tax-flow` `refine` rewiring to use batching — lives in the separate `tax-flow` repo (follow-up).
- Sparse or named-expansion `Batch` support.
- ODE/ADS batch benchmarks and the Kepler batch test (live in `tax-flow`).
- Runtime / dynamic lane counts.
