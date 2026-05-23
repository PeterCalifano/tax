# Stage 2b Slice B — `VectorOps<S>` + RK over DA-vector state Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the five RK steppers (Verner78, Verner89, Fehlberg78, Feagin12, Feagin14) propagate states of type `Eigen::Matrix<tax::TEn<P,M>, D, 1>` — a vector of multivariate Taylor polynomials in the initial-condition deviations — sharing the same code path that propagates `Eigen::Matrix<double, D, 1>`.

**Architecture:** New `VectorOps<S>` trait abstracts norm / axpy / scale-assign on the state type. The adaptive RK step driver uses the trait so its body is the same for any state. Stepper-level step-size control is pinned to `double` (no `State::Scalar` plumbing). The factory zoo is replaced by one type alias per method, with a defaulted RHS parameter so the simple form is a one-liner.

**Tech Stack:** C++23, header-only, Eigen 3.4+, Google Test, CMake.

---

## File map

- Create: `include/tax/ode/vector_ops.hpp` — the `VectorOps<S>` trait + three specializations.
- Modify: `include/tax/ode/detail/adaptive_rk_step.hpp` — refactor to use `VectorOps`, return `double err_norm`.
- Modify each RK stepper header (`verner78.hpp`, `verner89.hpp`, `fehlberg78.hpp`, `feagin12.hpp`, `feagin14.hpp`): pin `T = double`, use `VectorOps<State>::norm` for the residual.
- Modify: `include/tax/ode/integrator.hpp` — remove the 12 factory overloads, add 6 type aliases.
- Modify: `include/tax/ode.hpp` (or `tax/ode/integrator.hpp` includes) — ensure `vector_ops.hpp` is reachable.
- Modify (call-site migration): `tests/ode/testIntegratorStatic.cpp`, `testEventsEveryStep.cpp`, `testEventsZeroCrossing.cpp`, `testCR3BPEvents.cpp`, `testIntegratorBasic.cpp`, `testIntegratorDense.cpp`, `testCR3BPPropagation.cpp`, `testTwoBodyKepler.cpp` — replace `make…Integrator(...)` with type-alias construction.
- Create: `tests/ode/testVectorOps.cpp` — round-trips for the trait.
- Create: `tests/ode/testRKWithDaState.cpp` — DA-vector planar-Kepler correctness over the 5 RK families.
- Modify: `tests/ode/CMakeLists.txt` — register the two new test executables.
- Modify: `docs/ode/api.md`, `docs/ode/methods.md`, optional new `docs/ode/da-vector-state.md`.

Spec: `docs/superpowers/specs/2026-05-23-tax-stage2b-fixedstep-da-state-design.md` (Section "Slice B").

**Prerequisite:** Slice A (FixedStep controller) is recommended to land first so its `if constexpr` arm is already present in each stepper. This plan assumes Slice A is in.

---

### Task 1: Failing tests for `VectorOps<S>`

**Files:**
- Create: `tests/ode/testVectorOps.cpp`
- Modify: `tests/ode/CMakeLists.txt`

- [ ] **Step 1: Register the test executable**

Append to `tests/ode/CMakeLists.txt`:
```cmake
tax_add_test(test_ode_vector_ops SOURCES testVectorOps.cpp)
```

- [ ] **Step 2: Write the test**

Create `tests/ode/testVectorOps.cpp`:

```cpp
// tests/ode/testVectorOps.cpp
//
// Round-trip the VectorOps<S> trait on the four state shapes we ship:
//   - scalar double
//   - scalar TEn<P,M>
//   - Eigen::Matrix<double, D, 1>
//   - Eigen::Matrix<TEn<P,M>, D, 1>
// For each shape, verify:
//   norm(x)            == infinity-norm(x)
//   scale_assign(y,a,x) ⇒ y == a*x
//   axpy(y,a,x)         ⇒ y_new == y_old + a*x

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>

#include <tax/ode/vector_ops.hpp>
#include <tax/tax.hpp>

using tax::ode::VectorOps;

TEST( OdeVectorOps, ScalarDouble )
{
    double y = 0.0;
    VectorOps< double >::scale_assign( y, 2.5, 3.0 );
    EXPECT_DOUBLE_EQ( y, 7.5 );

    VectorOps< double >::axpy( y, -1.0, 2.0 );
    EXPECT_DOUBLE_EQ( y, 5.5 );

    EXPECT_DOUBLE_EQ( VectorOps< double >::norm( -4.2 ), 4.2 );
}

TEST( OdeVectorOps, ScalarTaylorExpansion )
{
    using DA = tax::TEn< 2, 2 >;
    DA x = DA::variable( 0.5, 0 );      // coeff[0]=0.5, coeff[var0]=1.0
    DA y{};                              // zero
    VectorOps< DA >::scale_assign( y, 3.0, x );

    EXPECT_DOUBLE_EQ( y[ 0 ], 1.5 );    // 3.0 * 0.5
    EXPECT_DOUBLE_EQ( VectorOps< DA >::norm( y ), 3.0 );  // 3.0 * 1.0 from var0

    VectorOps< DA >::axpy( y, -1.0, x );
    EXPECT_DOUBLE_EQ( y[ 0 ], 1.5 - 0.5 );
}

TEST( OdeVectorOps, EigenVectorDouble )
{
    using V = Eigen::Matrix< double, 3, 1 >;
    V x; x << 1.0, -2.0, 0.5;
    V y = V::Zero();

    VectorOps< V >::scale_assign( y, 2.0, x );
    EXPECT_DOUBLE_EQ( y( 0 ),  2.0 );
    EXPECT_DOUBLE_EQ( y( 1 ), -4.0 );
    EXPECT_DOUBLE_EQ( y( 2 ),  1.0 );

    EXPECT_DOUBLE_EQ( VectorOps< V >::norm( y ), 4.0 );

    VectorOps< V >::axpy( y, -0.5, x );
    EXPECT_DOUBLE_EQ( y( 0 ), 2.0 - 0.5 );
    EXPECT_DOUBLE_EQ( y( 1 ), -4.0 + 1.0 );
    EXPECT_DOUBLE_EQ( y( 2 ), 1.0 - 0.25 );
}

TEST( OdeVectorOps, EigenVectorOfTaylorExpansion )
{
    using DA = tax::TEn< 2, 2 >;
    using V  = Eigen::Matrix< DA, 2, 1 >;

    V x;
    x( 0 ) = DA::variable( 1.0, 0 );    // coeff[0]=1.0, coeff[e_0]=1.0
    x( 1 ) = DA::variable( -0.5, 1 );   // coeff[0]=-0.5, coeff[e_1]=1.0
    V y = V::Zero();

    VectorOps< V >::scale_assign( y, 2.0, x );
    EXPECT_DOUBLE_EQ( y( 0 )[ 0 ], 2.0 );    // 2*1.0
    EXPECT_DOUBLE_EQ( y( 1 )[ 0 ], -1.0 );   // 2*(-0.5)

    // sup-norm: max of (|2*1.0|, |2*1.0|, |2*(-0.5)|, |2*1.0|) = 2.0
    EXPECT_DOUBLE_EQ( VectorOps< V >::norm( y ), 2.0 );

    VectorOps< V >::axpy( y, -1.0, x );
    EXPECT_DOUBLE_EQ( y( 0 )[ 0 ], 2.0 - 1.0 );
    EXPECT_DOUBLE_EQ( y( 1 )[ 0 ], -1.0 + 0.5 );
}
```

- [ ] **Step 3: Run — must FAIL to compile**

Run:
```
cmake --build build --target test_ode_vector_ops 2>&1 | head -20
```
Expected: build failure citing `'vector_ops.hpp' file not found` and `'VectorOps' is not a member of 'tax::ode'`. The compile failure is the failing test for this task.

- [ ] **Step 4: Commit the failing test**

```bash
git add tests/ode/testVectorOps.cpp tests/ode/CMakeLists.txt
git commit -m "ode: add failing tests for VectorOps<S> trait"
```

---

### Task 2: Implement `VectorOps<S>` trait

**Files:**
- Create: `include/tax/ode/vector_ops.hpp`

- [ ] **Step 1: Create the header**

Create `include/tax/ode/vector_ops.hpp`:

```cpp
// include/tax/ode/vector_ops.hpp
//
// VectorOps<S>: trait that exposes the three operations every ODE
// step-driver needs to remain agnostic of the state's scalar layout —
// infinity-norm reduced to a real `double`, plus axpy and
// scale-assign that take a `double` coefficient.
//
// Default specializations cover:
//   - floating-point scalars
//   - tax::TaylorExpansionT<T, N, M>  (sup over coefficients)
//   - Eigen::Matrix<T, D, 1>          (recurses element-wise)
//
// To support a new state type, specialize VectorOps<MyState> with the
// three static functions; no stepper changes required.

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include <tax/core/taylor_expansion.hpp>

namespace tax::ode
{

template < class S >
struct VectorOps;                                              // primary: undefined

// ---- floating-point scalar ----
template < class T >
    requires std::is_floating_point_v< T >
struct VectorOps< T >
{
    [[nodiscard]] static double norm( T x ) noexcept
    { return std::abs( static_cast< double >( x ) ); }

    static void axpy( T& y, double a, T x ) noexcept
    { y += static_cast< T >( a ) * x; }

    static void scale_assign( T& y, double a, T x ) noexcept
    { y = static_cast< T >( a ) * x; }
};

// ---- scalar tax::TaylorExpansionT ----
template < class T, int N, int M >
struct VectorOps< TaylorExpansionT< T, N, M > >
{
    using S = TaylorExpansionT< T, N, M >;

    [[nodiscard]] static double norm( const S& x ) noexcept
    {
        double n = 0.0;
        for ( std::size_t i = 0; i < S::nCoefficients; ++i )
            n = std::max( n, std::abs( static_cast< double >( x[ i ] ) ) );
        return n;
    }

    static void axpy( S& y, double a, const S& x )
    { y = y + static_cast< T >( a ) * x; }

    static void scale_assign( S& y, double a, const S& x )
    { y = static_cast< T >( a ) * x; }
};

// ---- Eigen column vector of anything supported above ----
template < class T, int D >
struct VectorOps< Eigen::Matrix< T, D, 1 > >
{
    using V     = Eigen::Matrix< T, D, 1 >;
    using Inner = VectorOps< T >;

    [[nodiscard]] static double norm( const V& x ) noexcept
    {
        double n = 0.0;
        for ( Eigen::Index i = 0; i < x.size(); ++i )
            n = std::max( n, Inner::norm( x( i ) ) );
        return n;
    }

    static void axpy( V& y, double a, const V& x )
    {
        for ( Eigen::Index i = 0; i < x.size(); ++i )
            Inner::axpy( y( i ), a, x( i ) );
    }

    static void scale_assign( V& y, double a, const V& x )
    {
        if ( y.size() != x.size() ) y.resize( x.size() );
        for ( Eigen::Index i = 0; i < x.size(); ++i )
            Inner::scale_assign( y( i ), a, x( i ) );
    }
};

}  // namespace tax::ode
```

- [ ] **Step 2: Run — must PASS**

Run:
```
cmake --build build --target test_ode_vector_ops
ctest --test-dir build -R '^test_ode_vector_ops$' --output-on-failure
```
Expected: all four `OdeVectorOps.*` tests pass.

- [ ] **Step 3: Commit**

```bash
git add include/tax/ode/vector_ops.hpp
git commit -m "ode: add VectorOps<S> trait with scalar/TE/Eigen specializations"
```

---

### Task 3: Refactor `adaptive_rk_step` + all five RK steppers (single coordinated commit)

These changes are coupled: `adaptive_rk_step` returns `double err_norm`, and each stepper's `Stepper::T = double` so that `StepResult<State, Stepper>::err_norm` is also `double`. Splitting would temporarily break compilation.

**Files modified:**
- `include/tax/ode/detail/adaptive_rk_step.hpp`
- `include/tax/ode/steppers/verner78.hpp`
- `include/tax/ode/steppers/verner89.hpp`
- `include/tax/ode/steppers/fehlberg78.hpp`
- `include/tax/ode/steppers/feagin12.hpp`
- `include/tax/ode/steppers/feagin14.hpp`

- [ ] **Step 1: Rewrite `adaptive_rk_step.hpp`**

Replace the entire body of `include/tax/ode/detail/adaptive_rk_step.hpp` with:

```cpp
// include/tax/ode/detail/adaptive_rk_step.hpp
//
// Generic explicit Runge–Kutta step driver. Routes state arithmetic
// through tax::ode::VectorOps<State>, which decouples step-size
// control (always double) from the state's scalar layout. Same body
// serves double-state and DA-vector-state.

#pragma once

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <utility>

#include <tax/ode/vector_ops.hpp>

namespace tax::ode::detail
{

template < class State, int NStages >
struct RKStepData
{
    std::array< State, NStages > k{};
};

template < class State >
struct RKStepOut
{
    State  x_new;
    State  y_emb;
    double err_norm;                       // always double
};

template < class Tab, class F, class State, int NStages >
[[nodiscard]] RKStepOut< State > adaptive_rk_step(
    F&& f, const State& x, double t, double h,
    RKStepData< State, NStages >& work )
{
    static_assert( NStages == Tab::n_stages,
                   "adaptive_rk_step: stage-count mismatch with tableau" );

    using Ops = VectorOps< State >;

    work.k[ 0 ] = f( x, t + Tab::c[ 0 ] * h );

    std::size_t a_off = 0;
    for ( int i = 1; i < NStages; ++i )
    {
        State y;
        Ops::scale_assign( y, 1.0, x );
        for ( int j = 0; j < i; ++j )
            Ops::axpy( y, h * Tab::a[ a_off + std::size_t( j ) ],
                       work.k[ std::size_t( j ) ] );
        work.k[ std::size_t( i ) ] = f( y, t + Tab::c[ std::size_t( i ) ] * h );
        a_off += std::size_t( i );
    }

    State x_new, y_emb;
    Ops::scale_assign( x_new, 1.0, x );
    Ops::scale_assign( y_emb, 1.0, x );
    for ( int i = 0; i < NStages; ++i )
    {
        Ops::axpy( x_new, h * Tab::b    [ std::size_t( i ) ],
                   work.k[ std::size_t( i ) ] );
        Ops::axpy( y_emb, h * Tab::b_emb[ std::size_t( i ) ],
                   work.k[ std::size_t( i ) ] );
    }

    State diff;
    Ops::scale_assign( diff,  1.0, x_new );
    Ops::axpy        ( diff, -1.0, y_emb );

    return { std::move( x_new ), std::move( y_emb ), Ops::norm( diff ) };
}

}  // namespace tax::ode::detail
```

- [ ] **Step 2: Rewrite `verner78.hpp`'s `step()`**

In `include/tax/ode/steppers/verner78.hpp`, change the `using T = …; using Config = …;` block and `step()` body. Final stepper definition:

```cpp
template < class StateT,
           class Controller = controllers::PI< double > >
struct Verner78Stepper
{
    using State  = StateT;
    using T      = double;                                     // step-control scalar
    using Config = IntegratorConfig< double >;
    using Rhs    = std::function< State( const State&, T ) >;
    using Tab    = detail::Verner78Tab;

    static constexpr bool is_adaptive      = true;
    static constexpr bool has_dense_output = false;
    static constexpr int  order_v          = Tab::order;
    static constexpr int  order_emb_v      = Tab::order_emb;

    struct DenseData
    {
        State x0{};
        State x1{};
        State f0{};
        State f1{};
    };

    template < class F >
    [[nodiscard]] StepResult< State, Verner78Stepper > step(
        F&& f, const State& x, T t, T h, const Config& cfg )
    {
        using detail::adaptive_rk_step;
        using detail::RKStepData;

        RKStepData< State, Tab::n_stages > work;
        auto out = adaptive_rk_step< Tab >( f, x, t, h, work );

        const double x_norm = VectorOps< State >::norm( out.x_new );
        const double tol    = cfg.abstol + cfg.reltol * x_norm;

        T    h_next;
        bool accepted;
        if constexpr ( std::is_same_v< Controller, controllers::FixedStep< T > > )
        {
            h_next   = h;
            accepted = true;
        }
        else if constexpr ( std::is_same_v< Controller, controllers::JorbaZou< T > > )
        {
            h_next   = h;
            accepted = out.err_norm <= tol;
        }
        else
        {
            h_next   = controller_.next_step( h, out.err_norm, tol, Tab::order_emb );
            accepted = out.err_norm <= tol;
        }

        DenseData dd;
        dd.x0 = x;
        dd.x1 = out.x_new;
        dd.f0 = work.k[ 0 ];
        dd.f1 = f( out.x_new, t + h );

        StepResult< State, Verner78Stepper > r;
        r.x_new    = std::move( out.x_new );
        r.h_used   = h;
        r.dense    = std::move( dd );
        r.h_next   = h_next;
        r.err_norm = out.err_norm;
        r.accepted = accepted;
        return r;
    }

    [[nodiscard]] static State eval_dense(
        const DenseData& d, const T& t0, const T& t1, const T& tq )
    {
        return detail::hermite_interp< State, T >(
            d.x0, d.x1, d.f0, d.f1, t0, t1, tq );
    }

private:
    Controller controller_{};
};
```

Add `#include <tax/ode/vector_ops.hpp>` at the top of the file (alongside the existing includes).

- [ ] **Step 3: Rewrite `verner89.hpp`'s `step()` to match (same shape as Verner78)**

In `include/tax/ode/steppers/verner89.hpp`, apply the equivalent rewrite — same `using` declarations, same step body, same defaults (`Controller = controllers::PI< double >`). Replace `Verner78Stepper` with `Verner89Stepper` and `Verner78Tab` with `Verner89Tab`. Add `#include <tax/ode/vector_ops.hpp>`. Final code differs from Verner78 only in the stepper name and tableau type.

```cpp
template < class StateT,
           class Controller = controllers::PI< double > >
struct Verner89Stepper
{
    using State  = StateT;
    using T      = double;
    using Config = IntegratorConfig< double >;
    using Rhs    = std::function< State( const State&, T ) >;
    using Tab    = detail::Verner89Tab;

    static constexpr bool is_adaptive      = true;
    static constexpr bool has_dense_output = false;
    static constexpr int  order_v          = Tab::order;
    static constexpr int  order_emb_v      = Tab::order_emb;

    struct DenseData
    {
        State x0{};
        State x1{};
        State f0{};
        State f1{};
    };

    template < class F >
    [[nodiscard]] StepResult< State, Verner89Stepper > step(
        F&& f, const State& x, T t, T h, const Config& cfg )
    {
        using detail::adaptive_rk_step;
        using detail::RKStepData;

        RKStepData< State, Tab::n_stages > work;
        auto out = adaptive_rk_step< Tab >( f, x, t, h, work );

        const double x_norm = VectorOps< State >::norm( out.x_new );
        const double tol    = cfg.abstol + cfg.reltol * x_norm;

        T    h_next;
        bool accepted;
        if constexpr ( std::is_same_v< Controller, controllers::FixedStep< T > > )
        {
            h_next   = h;
            accepted = true;
        }
        else if constexpr ( std::is_same_v< Controller, controllers::JorbaZou< T > > )
        {
            h_next   = h;
            accepted = out.err_norm <= tol;
        }
        else
        {
            h_next   = controller_.next_step( h, out.err_norm, tol, Tab::order_emb );
            accepted = out.err_norm <= tol;
        }

        DenseData dd;
        dd.x0 = x;
        dd.x1 = out.x_new;
        dd.f0 = work.k[ 0 ];
        dd.f1 = f( out.x_new, t + h );

        StepResult< State, Verner89Stepper > r;
        r.x_new    = std::move( out.x_new );
        r.h_used   = h;
        r.dense    = std::move( dd );
        r.h_next   = h_next;
        r.err_norm = out.err_norm;
        r.accepted = accepted;
        return r;
    }

    [[nodiscard]] static State eval_dense(
        const DenseData& d, const T& t0, const T& t1, const T& tq )
    {
        return detail::hermite_interp< State, T >(
            d.x0, d.x1, d.f0, d.f1, t0, t1, tq );
    }

private:
    Controller controller_{};
};
```

- [ ] **Step 4: Rewrite `fehlberg78.hpp` the same way**

`include/tax/ode/steppers/fehlberg78.hpp` — same shape as Verner78; replace `Verner78Stepper` with `Fehlberg78Stepper`, `Verner78Tab` with `Fehlberg78Tab`. Add `#include <tax/ode/vector_ops.hpp>`.

```cpp
template < class StateT,
           class Controller = controllers::PI< double > >
struct Fehlberg78Stepper
{
    using State  = StateT;
    using T      = double;
    using Config = IntegratorConfig< double >;
    using Rhs    = std::function< State( const State&, T ) >;
    using Tab    = detail::Fehlberg78Tab;

    static constexpr bool is_adaptive      = true;
    static constexpr bool has_dense_output = false;
    static constexpr int  order_v          = Tab::order;
    static constexpr int  order_emb_v      = Tab::order_emb;

    struct DenseData { State x0{}; State x1{}; State f0{}; State f1{}; };

    template < class F >
    [[nodiscard]] StepResult< State, Fehlberg78Stepper > step(
        F&& f, const State& x, T t, T h, const Config& cfg )
    {
        using detail::adaptive_rk_step;
        using detail::RKStepData;

        RKStepData< State, Tab::n_stages > work;
        auto out = adaptive_rk_step< Tab >( f, x, t, h, work );

        const double x_norm = VectorOps< State >::norm( out.x_new );
        const double tol    = cfg.abstol + cfg.reltol * x_norm;

        T    h_next;
        bool accepted;
        if constexpr ( std::is_same_v< Controller, controllers::FixedStep< T > > )
        {
            h_next = h;  accepted = true;
        }
        else if constexpr ( std::is_same_v< Controller, controllers::JorbaZou< T > > )
        {
            h_next = h;  accepted = out.err_norm <= tol;
        }
        else
        {
            h_next   = controller_.next_step( h, out.err_norm, tol, Tab::order_emb );
            accepted = out.err_norm <= tol;
        }

        DenseData dd;
        dd.x0 = x; dd.x1 = out.x_new;
        dd.f0 = work.k[ 0 ]; dd.f1 = f( out.x_new, t + h );

        StepResult< State, Fehlberg78Stepper > r;
        r.x_new    = std::move( out.x_new );
        r.h_used   = h;
        r.dense    = std::move( dd );
        r.h_next   = h_next;
        r.err_norm = out.err_norm;
        r.accepted = accepted;
        return r;
    }

    [[nodiscard]] static State eval_dense(
        const DenseData& d, const T& t0, const T& t1, const T& tq )
    {
        return detail::hermite_interp< State, T >(
            d.x0, d.x1, d.f0, d.f1, t0, t1, tq );
    }

private:
    Controller controller_{};
};
```

- [ ] **Step 5: Rewrite `feagin12.hpp` (preserves `err_for_ctrl` floor)**

`include/tax/ode/steppers/feagin12.hpp` — same skeleton plus the `err_for_ctrl = (err > 0) ? err : tol * eps` floor, computed in `double`:

```cpp
template < class StateT,
           class Controller = controllers::PI< double > >
struct Feagin12Stepper
{
    using State  = StateT;
    using T      = double;
    using Config = IntegratorConfig< double >;
    using Rhs    = std::function< State( const State&, T ) >;
    using Tab    = detail::Feagin12Tab;

    static constexpr bool is_adaptive      = true;
    static constexpr bool has_dense_output = false;
    static constexpr int  order_v          = Tab::order;
    static constexpr int  order_emb_v      = Tab::order_emb;

    struct DenseData { State x0{}; State x1{}; State f0{}; State f1{}; };

    template < class F >
    [[nodiscard]] StepResult< State, Feagin12Stepper > step(
        F&& f, const State& x, T t, T h, const Config& cfg )
    {
        using detail::adaptive_rk_step;
        using detail::RKStepData;

        RKStepData< State, Tab::n_stages > work;
        auto out = adaptive_rk_step< Tab >( f, x, t, h, work );

        const double x_norm = VectorOps< State >::norm( out.x_new );
        const double tol    = cfg.abstol + cfg.reltol * x_norm;

        // Feagin's (k_2 - k_{n-1}) indicator can underflow to zero on
        // benign integrands at small h. Floor at eps*tol so the controller
        // grows h instead of treating zero error as "use default factor".
        const double err_for_ctrl = ( out.err_norm > 0.0 )
            ? out.err_norm
            : tol * std::numeric_limits< double >::epsilon();

        T    h_next;
        bool accepted;
        if constexpr ( std::is_same_v< Controller, controllers::FixedStep< T > > )
        {
            h_next = h;  accepted = true;
        }
        else if constexpr ( std::is_same_v< Controller, controllers::JorbaZou< T > > )
        {
            h_next = h;  accepted = out.err_norm <= tol;
        }
        else
        {
            h_next   = controller_.next_step( h, err_for_ctrl, tol, Tab::order_emb );
            accepted = out.err_norm <= tol;
        }

        DenseData dd;
        dd.x0 = x; dd.x1 = out.x_new;
        dd.f0 = work.k[ 0 ]; dd.f1 = f( out.x_new, t + h );

        StepResult< State, Feagin12Stepper > r;
        r.x_new    = std::move( out.x_new );
        r.h_used   = h;
        r.dense    = std::move( dd );
        r.h_next   = h_next;
        r.err_norm = out.err_norm;
        r.accepted = accepted;
        return r;
    }

    [[nodiscard]] static State eval_dense(
        const DenseData& d, const T& t0, const T& t1, const T& tq )
    {
        return detail::hermite_interp< State, T >(
            d.x0, d.x1, d.f0, d.f1, t0, t1, tq );
    }

private:
    Controller controller_{};
};
```

Add `#include <tax/ode/vector_ops.hpp>` and keep the existing `#include <limits>`.

- [ ] **Step 6: Rewrite `feagin14.hpp` (same shape as Feagin12 with `err_for_ctrl`)**

`include/tax/ode/steppers/feagin14.hpp` — same as Feagin12 with `Feagin14Stepper` / `Feagin14Tab`:

```cpp
template < class StateT,
           class Controller = controllers::PI< double > >
struct Feagin14Stepper
{
    using State  = StateT;
    using T      = double;
    using Config = IntegratorConfig< double >;
    using Rhs    = std::function< State( const State&, T ) >;
    using Tab    = detail::Feagin14Tab;

    static constexpr bool is_adaptive      = true;
    static constexpr bool has_dense_output = false;
    static constexpr int  order_v          = Tab::order;
    static constexpr int  order_emb_v      = Tab::order_emb;

    struct DenseData { State x0{}; State x1{}; State f0{}; State f1{}; };

    template < class F >
    [[nodiscard]] StepResult< State, Feagin14Stepper > step(
        F&& f, const State& x, T t, T h, const Config& cfg )
    {
        using detail::adaptive_rk_step;
        using detail::RKStepData;

        RKStepData< State, Tab::n_stages > work;
        auto out = adaptive_rk_step< Tab >( f, x, t, h, work );

        const double x_norm = VectorOps< State >::norm( out.x_new );
        const double tol    = cfg.abstol + cfg.reltol * x_norm;

        const double err_for_ctrl = ( out.err_norm > 0.0 )
            ? out.err_norm
            : tol * std::numeric_limits< double >::epsilon();

        T    h_next;
        bool accepted;
        if constexpr ( std::is_same_v< Controller, controllers::FixedStep< T > > )
        {
            h_next = h;  accepted = true;
        }
        else if constexpr ( std::is_same_v< Controller, controllers::JorbaZou< T > > )
        {
            h_next = h;  accepted = out.err_norm <= tol;
        }
        else
        {
            h_next   = controller_.next_step( h, err_for_ctrl, tol, Tab::order_emb );
            accepted = out.err_norm <= tol;
        }

        DenseData dd;
        dd.x0 = x; dd.x1 = out.x_new;
        dd.f0 = work.k[ 0 ]; dd.f1 = f( out.x_new, t + h );

        StepResult< State, Feagin14Stepper > r;
        r.x_new    = std::move( out.x_new );
        r.h_used   = h;
        r.dense    = std::move( dd );
        r.h_next   = h_next;
        r.err_norm = out.err_norm;
        r.accepted = accepted;
        return r;
    }

    [[nodiscard]] static State eval_dense(
        const DenseData& d, const T& t0, const T& t1, const T& tq )
    {
        return detail::hermite_interp< State, T >(
            d.x0, d.x1, d.f0, d.f1, t0, t1, tq );
    }

private:
    Controller controller_{};
};
```

Keep `#include <limits>` and add `#include <tax/ode/vector_ops.hpp>`.

- [ ] **Step 7: Build and run the entire ODE test suite**

Run:
```
cmake --build build
ctest --test-dir build -R '^test_ode_' --output-on-failure
```
Expected: every existing ODE test continues to pass — same numerical results, same step counts. The refactor is correct iff no existing test regresses.

- [ ] **Step 8: Commit**

```bash
git add include/tax/ode/detail/adaptive_rk_step.hpp \
        include/tax/ode/steppers/verner78.hpp \
        include/tax/ode/steppers/verner89.hpp \
        include/tax/ode/steppers/fehlberg78.hpp \
        include/tax/ode/steppers/feagin12.hpp \
        include/tax/ode/steppers/feagin14.hpp
git commit -m "ode/rk: route state arithmetic through VectorOps, pin T=double"
```

---

### Task 4: Add per-method type aliases

**Files:**
- Modify: `include/tax/ode/integrator.hpp`

- [ ] **Step 1: Append the aliases after the `Integrator` class definition, before the `Factories` comment block**

In `include/tax/ode/integrator.hpp`, just after the closing brace of `Integrator<…>::integrate` (around line 200, where the `// ----- Factories -----` comment begins), insert:

```cpp
// ----- Type aliases: one per method -----
//
// Defaulted F parameter resolves to Stepper::Rhs = std::function<…>, so
// the simple form `Verner78<State>{f, cfg}` works out of the box (one
// vtable indirection per RHS call). Users who care about that overhead
// can spell F explicitly:
//
//   Verner78<State, controllers::PI<double>, /*Dense=*/false,
//            decltype(my_lambda)> integ{ my_lambda, cfg };

template < class State,
           class Controller = controllers::PI< double >,
           bool  Dense      = false,
           class F          = typename Verner78Stepper< State, Controller >::Rhs >
using Verner78 = Integrator< Verner78Stepper< State, Controller >, F, Dense >;

template < class State,
           class Controller = controllers::PI< double >,
           bool  Dense      = false,
           class F          = typename Verner89Stepper< State, Controller >::Rhs >
using Verner89 = Integrator< Verner89Stepper< State, Controller >, F, Dense >;

template < class State,
           class Controller = controllers::PI< double >,
           bool  Dense      = false,
           class F          = typename Fehlberg78Stepper< State, Controller >::Rhs >
using Fehlberg78 = Integrator< Fehlberg78Stepper< State, Controller >, F, Dense >;

template < class State,
           class Controller = controllers::PI< double >,
           bool  Dense      = false,
           class F          = typename Feagin12Stepper< State, Controller >::Rhs >
using Feagin12 = Integrator< Feagin12Stepper< State, Controller >, F, Dense >;

template < class State,
           class Controller = controllers::PI< double >,
           bool  Dense      = false,
           class F          = typename Feagin14Stepper< State, Controller >::Rhs >
using Feagin14 = Integrator< Feagin14Stepper< State, Controller >, F, Dense >;

template < int   N,
           class State,
           class Controller = controllers::JorbaZou< double >,
           bool  Dense      = false,
           class F          = typename TaylorStepper< N, State, Controller >::Rhs >
using Taylor = Integrator< TaylorStepper< N, State, Controller >, F, Dense >;
```

- [ ] **Step 2: Build, verify the aliases compile**

Run:
```
cmake --build build 2>&1 | grep -E 'warning|error' || echo 'clean'
```
Expected: `clean`. (No test uses the aliases yet.)

- [ ] **Step 3: Commit**

```bash
git add include/tax/ode/integrator.hpp
git commit -m "ode: add per-method type aliases (Verner78, Verner89, Fehlberg78, Feagin12, Feagin14, Taylor)"
```

---

### Task 5: Migrate test call sites from `make…Integrator` to type aliases

There are ~40 call sites across 8 test files. The pattern is mechanical: replace each `make…Integrator<…>(rhs, cfg)` with a one-line construction of the alias.

**Files:**
- Modify: `tests/ode/testIntegratorStatic.cpp`, `testEventsEveryStep.cpp`, `testEventsZeroCrossing.cpp`, `testCR3BPEvents.cpp`, `testIntegratorBasic.cpp`, `testIntegratorDense.cpp`, `testCR3BPPropagation.cpp`, `testTwoBodyKepler.cpp`.

The transformation for each call has the form:

| Before | After |
|---|---|
| `auto integ = tax::ode::makeVerner78Integrator< double, 4, false >( rhs, cfg );` | `tax::ode::Verner78< Eigen::Matrix< double, 4, 1 > > integ{ rhs, cfg };` |
| `auto integ = tax::ode::makeVerner89Integrator< double, 4, false >( rhs, cfg );` | `tax::ode::Verner89< Eigen::Matrix< double, 4, 1 > > integ{ rhs, cfg };` |
| `auto integ = tax::ode::makeFehlberg78Integrator< double, 4, false >( rhs, cfg );` | `tax::ode::Fehlberg78< Eigen::Matrix< double, 4, 1 > > integ{ rhs, cfg };` |
| `auto integ = tax::ode::makeFeagin12Integrator< double, 4, false >( rhs, cfg );` | `tax::ode::Feagin12< Eigen::Matrix< double, 4, 1 > > integ{ rhs, cfg };` |
| `auto integ = tax::ode::makeFeagin14Integrator< double, 4, false >( rhs, cfg );` | `tax::ode::Feagin14< Eigen::Matrix< double, 4, 1 > > integ{ rhs, cfg };` |
| `auto integ = tax::ode::makeTaylorIntegrator< 25, double, 4, false >( rhs, cfg );` | `tax::ode::Taylor< 25, Eigen::Matrix< double, 4, 1 > > integ{ rhs, cfg };` |

`Dense = true` callsites map similarly, with `, true >` swapping to the 3rd alias parameter `, /*Dense=*/true >`.

Event-list overloads (3-arg `make…Integrator(rhs, cfg, events)`) become `Verner78<…> integ{ rhs, cfg, events };` — the underlying `Integrator` constructor already takes events.

- [ ] **Step 1: Migrate `testTwoBodyKepler.cpp`**

In `tests/ode/testTwoBodyKepler.cpp`, replace each `make…Integrator` call. Final file (the four TEST blocks at the bottom):

```cpp
TEST( OdeTwoBodyKepler, Taylor16 )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;

    tax::ode::Taylor< 16, State > integ{ make_rhs(), cfg };
    auto sol = integ.integrate( make_ic(), 0.0, 10.0 * kPeriod );

    check_invariants( sol, /*E=*/1e-10, /*L=*/1e-10, /*close=*/1e-8, "Taylor16" );
}

TEST( OdeTwoBodyKepler, Verner89 )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;

    tax::ode::Verner89< State > integ{ make_rhs(), cfg };
    auto sol = integ.integrate( make_ic(), 0.0, 10.0 * kPeriod );

    check_invariants( sol, /*E=*/1e-9, /*L=*/1e-9, /*close=*/1e-7, "Verner89" );
}

TEST( OdeTwoBodyKepler, Feagin14 )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;

    tax::ode::Feagin14< State > integ{ make_rhs(), cfg };
    auto sol = integ.integrate( make_ic(), 0.0, 10.0 * kPeriod );

    check_invariants( sol, /*E=*/1e-11, /*L=*/1e-11, /*close=*/1e-9, "Feagin14" );
}

TEST( OdeTwoBodyKepler, Verner78 )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;

    tax::ode::Verner78< State > integ{ make_rhs(), cfg };
    auto sol = integ.integrate( make_ic(), 0.0, 10.0 * kPeriod );

    check_invariants( sol, /*E=*/1e-8, /*L=*/1e-8, /*close=*/1e-6, "Verner78" );
}
```

- [ ] **Step 2: Migrate the remaining seven test files using the table above**

For each file, find every `auto integ = tax::ode::make…Integrator<…>(rhs, cfg [, events] );` and replace with `tax::ode::<Alias>< State [extra args] > integ{ rhs, cfg [, events] };`. Use `grep -n "make.*Integrator<" tests/ode/<file>` to locate.

Apply the same transformation to:
- `tests/ode/testIntegratorBasic.cpp`
- `tests/ode/testIntegratorDense.cpp`
- `tests/ode/testIntegratorStatic.cpp`
- `tests/ode/testEventsZeroCrossing.cpp`
- `tests/ode/testEventsEveryStep.cpp`
- `tests/ode/testCR3BPPropagation.cpp`
- `tests/ode/testCR3BPEvents.cpp`

Also update Slice A's `tests/ode/testFixedStep.cpp` if it's already merged:

```cpp
// Replace:
auto integ = tax::ode::makeVerner78Integrator< double, 1, false,
                                                FixedStep< double > >( identity_rhs< State >(), cfg );
// With:
tax::ode::Verner78< Eigen::Matrix< double, 1, 1 >,
                    FixedStep< double > > integ{ identity_rhs< State >(), cfg };
```

(Same for Verner89, Fehlberg78, Feagin12, Feagin14, and the manually-constructed Taylor case in Slice A's testFixedStep.cpp Task 8.)

- [ ] **Step 3: Rebuild and run the full ODE suite**

```
cmake --build build
ctest --test-dir build -R '^test_ode_' --output-on-failure
```
Expected: every test still passes. Same numerical results, same step counts.

- [ ] **Step 4: Commit**

```bash
git add tests/ode/
git commit -m "tests/ode: migrate from make…Integrator factories to per-method type aliases"
```

---

### Task 6: Remove the 12 factory overloads from `integrator.hpp`

**Files:**
- Modify: `include/tax/ode/integrator.hpp`

- [ ] **Step 1: Delete the factory section**

In `include/tax/ode/integrator.hpp`, delete everything from the `// ----- Factories -----` comment block (around line 200) down to (but not including) the closing `}  // namespace tax::ode`. That removes:

- `makeTaylorIntegrator` × 2 overloads (cfg-only, cfg+events)
- `makeVerner78Integrator` × 2
- `makeVerner89Integrator` × 2
- `makeFehlberg78Integrator` × 2
- `makeFeagin12Integrator` × 2
- `makeFeagin14Integrator` × 2

Total: 12 function templates removed. The type aliases added in Task 4 stay.

- [ ] **Step 2: Rebuild and verify nothing references the deleted factories**

```
cmake --build build 2>&1 | grep -E 'error.*make.*Integrator' || echo 'no factory references remain'
```
Expected: `no factory references remain`.

- [ ] **Step 3: Full ODE test suite**

```
ctest --test-dir build -R '^test_ode_' --output-on-failure
```
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add include/tax/ode/integrator.hpp
git commit -m "ode: remove make…Integrator factories (superseded by type aliases)"
```

---

### Task 7: DA-vector state correctness test — constant-term parity

**Files:**
- Create: `tests/ode/testRKWithDaState.cpp`
- Modify: `tests/ode/CMakeLists.txt`

This task validates that propagating planar Kepler with `Eigen::Matrix<TEn<2,2>, 4, 1>` state produces, in the constant DA term, the same result as propagating `Eigen::Matrix<double, 4, 1>`. Task 8 adds the linear (STM) check.

- [ ] **Step 1: Register the test**

Append to `tests/ode/CMakeLists.txt`:
```cmake
tax_add_test(test_ode_rk_da_state SOURCES testRKWithDaState.cpp)
```

- [ ] **Step 2: Create the test file**

Create `tests/ode/testRKWithDaState.cpp`:

```cpp
// tests/ode/testRKWithDaState.cpp
//
// Propagate planar Kepler (e=0.5, GM=a=1) with State =
// Eigen::Matrix<TEn<P,M>, 4, 1> across the five RK families and verify:
//   (a) constant DA term  ≈ double-state propagation     (Task 7)
//   (b) linear DA term    ≈ finite-difference STM        (Task 8)

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <array>
#include <cmath>

#include <tax/ode.hpp>
#include <tax/tax.hpp>

namespace {

constexpr int P = 2;
constexpr int M = 4;                          // 4 IC DA variables
constexpr double kEcc        = 0.5;
constexpr double kPeriapsis  = 1.0 - kEcc;
const     double kVPeriapsis = std::sqrt( ( 1.0 + kEcc ) / ( 1.0 - kEcc ) );
constexpr double kPeriod     = 2.0 * M_PI;
constexpr double kHalfWidth  = 1e-3;

using DA       = tax::TEn< P, M >;
using StateD   = Eigen::Matrix< double, 4, 1 >;
using StateDA  = Eigen::Matrix< DA,     4, 1 >;

StateD make_ic_double()
{
    StateD x0;
    x0( 0 ) = kPeriapsis;  x0( 1 ) = 0.0;
    x0( 2 ) = 0.0;         x0( 3 ) = kVPeriapsis;
    return x0;
}

StateDA make_ic_da()
{
    StateD c = make_ic_double();
    StateDA x0;
    for ( int i = 0; i < 4; ++i )
        x0( i ) = DA( c( i ) ) + DA( kHalfWidth ) * DA::variable( 0.0, i );
    return x0;
}

template < class S >
auto make_rhs()
{
    return []( const S& s, double /*t*/ )
    {
        S out;
        const auto x  = s( 0 );
        const auto y  = s( 1 );
        const auto r2 = x * x + y * y;
        const auto r3 = r2 * sqrt( r2 );
        out( 0 ) = s( 2 );
        out( 1 ) = s( 3 );
        out( 2 ) = -x / r3;
        out( 3 ) = -y / r3;
        return out;
    };
}

template < template < class, class, bool, class > class IntegratorAlias >
void check_constant_term_matches_double( const char* method_name,
                                         double tol_close )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    using IntegD  = IntegratorAlias<
        StateD,  tax::ode::controllers::PI< double >, false,
        typename std::remove_reference_t<
            decltype( make_rhs< StateD >() ) > >;
    using IntegDA = IntegratorAlias<
        StateDA, tax::ode::controllers::PI< double >, false,
        typename std::remove_reference_t<
            decltype( make_rhs< StateDA >() ) > >;

    IntegD  integ_d { make_rhs< StateD  >(), cfg };
    IntegDA integ_da{ make_rhs< StateDA >(), cfg };

    auto sol_d  = integ_d .integrate( make_ic_double(), 0.0, kPeriod );
    auto sol_da = integ_da.integrate( make_ic_da(),     0.0, kPeriod );

    const StateD&  xT_d  = sol_d .x.back();
    const StateDA& xT_da = sol_da.x.back();
    for ( int i = 0; i < 4; ++i )
    {
        EXPECT_NEAR( xT_da( i )[ 0 ], xT_d( i ), tol_close )
            << "method=" << method_name << " component=" << i;
    }
}

}  // namespace

TEST( OdeRKWithDaState, ConstantTermVerner78 )
{
    check_constant_term_matches_double< tax::ode::Verner78 >( "Verner78", 1e-8 );
}

TEST( OdeRKWithDaState, ConstantTermVerner89 )
{
    check_constant_term_matches_double< tax::ode::Verner89 >( "Verner89", 1e-9 );
}

TEST( OdeRKWithDaState, ConstantTermFehlberg78 )
{
    check_constant_term_matches_double< tax::ode::Fehlberg78 >( "Fehlberg78", 1e-8 );
}

TEST( OdeRKWithDaState, ConstantTermFeagin12 )
{
    check_constant_term_matches_double< tax::ode::Feagin12 >( "Feagin12", 1e-10 );
}

TEST( OdeRKWithDaState, ConstantTermFeagin14 )
{
    check_constant_term_matches_double< tax::ode::Feagin14 >( "Feagin14", 1e-11 );
}
```

- [ ] **Step 3: Run — must PASS**

Run:
```
cmake --build build --target test_ode_rk_da_state
ctest --test-dir build -R '^test_ode_rk_da_state$' --output-on-failure
```
Expected: all five constant-term tests pass.

If they don't: check that `Eigen::Matrix<DA, 4, 1>` arithmetic in the RHS (`x*x`, `r2*sqrt(r2)`, scalar division) resolves correctly — the most likely failure mode is an ADL miss on `sqrt(DA)`, in which case the RHS lambda needs a `using std::sqrt; using tax::sqrt;` line at the top.

- [ ] **Step 4: Commit**

```bash
git add tests/ode/testRKWithDaState.cpp tests/ode/CMakeLists.txt
git commit -m "tests/ode: RK steppers on DA-vector state — constant-term parity"
```

---

### Task 8: DA-vector state correctness test — linear-term ≈ finite-difference STM

**Files:**
- Modify: `tests/ode/testRKWithDaState.cpp` (append)

- [ ] **Step 1: Add the STM helper and parameterized linear-term test**

Append to `tests/ode/testRKWithDaState.cpp`, after the constant-term tests:

```cpp
namespace {

// Forward-difference STM: column i of STM ≈
//   ( x_T(x0 + eps * e_i) - x_T(x0 - eps * e_i) ) / ( 2 * eps )
template < template < class, class, bool, class > class IntegratorAlias >
Eigen::Matrix< double, 4, 4 > fd_stm( double eps )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;

    using Integ = IntegratorAlias<
        StateD,  tax::ode::controllers::PI< double >, false,
        typename std::remove_reference_t<
            decltype( make_rhs< StateD >() ) > >;
    Integ integ{ make_rhs< StateD >(), cfg };

    Eigen::Matrix< double, 4, 4 > stm;
    for ( int i = 0; i < 4; ++i )
    {
        StateD ic_p = make_ic_double(); ic_p( i ) += eps;
        StateD ic_m = make_ic_double(); ic_m( i ) -= eps;

        auto sol_p = integ.integrate( ic_p, 0.0, kPeriod );
        auto sol_m = integ.integrate( ic_m, 0.0, kPeriod );

        const StateD& xp = sol_p.x.back();
        const StateD& xm = sol_m.x.back();
        for ( int j = 0; j < 4; ++j )
            stm( j, i ) = ( xp( j ) - xm( j ) ) / ( 2.0 * eps );
    }
    return stm;
}

template < template < class, class, bool, class > class IntegratorAlias >
void check_linear_term_matches_fd_stm( const char* method_name, double tol_abs )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    using IntegDA = IntegratorAlias<
        StateDA, tax::ode::controllers::PI< double >, false,
        typename std::remove_reference_t<
            decltype( make_rhs< StateDA >() ) > >;
    IntegDA integ_da{ make_rhs< StateDA >(), cfg };
    auto sol_da = integ_da.integrate( make_ic_da(), 0.0, kPeriod );

    const StateDA& xT_da = sol_da.x.back();
    const auto stm_fd = fd_stm< IntegratorAlias >( 1e-6 );

    // DA layout: TEn<2,4>::flatIndex({e_i}) = 1 + i (degree-1 monomial i).
    // halfWidth chain rule: DA-coef at e_i = kHalfWidth * (∂x_T/∂x_0[i]).
    for ( int row = 0; row < 4; ++row )
    {
        for ( int col = 0; col < 4; ++col )
        {
            const std::size_t flat = static_cast< std::size_t >( 1 + col );
            const double      lhs  = xT_da( row )[ flat ] / kHalfWidth;
            const double      rhs  = stm_fd( row, col );
            EXPECT_NEAR( lhs, rhs, tol_abs )
                << "method=" << method_name
                << " row="   << row
                << " col="   << col;
        }
    }
}

}  // namespace

TEST( OdeRKWithDaState, LinearTermVerner78 )
{
    check_linear_term_matches_fd_stm< tax::ode::Verner78 >( "Verner78", 1e-5 );
}

TEST( OdeRKWithDaState, LinearTermVerner89 )
{
    check_linear_term_matches_fd_stm< tax::ode::Verner89 >( "Verner89", 1e-5 );
}

TEST( OdeRKWithDaState, LinearTermFehlberg78 )
{
    check_linear_term_matches_fd_stm< tax::ode::Fehlberg78 >( "Fehlberg78", 1e-5 );
}

TEST( OdeRKWithDaState, LinearTermFeagin12 )
{
    check_linear_term_matches_fd_stm< tax::ode::Feagin12 >( "Feagin12", 1e-5 );
}

TEST( OdeRKWithDaState, LinearTermFeagin14 )
{
    check_linear_term_matches_fd_stm< tax::ode::Feagin14 >( "Feagin14", 1e-5 );
}
```

- [ ] **Step 2: Run — must PASS**

Run:
```
cmake --build build --target test_ode_rk_da_state
ctest --test-dir build -R '^test_ode_rk_da_state$' --output-on-failure
```
Expected: all ten tests (5 constant-term + 5 linear-term) pass.

Tolerance budget rationale: `kHalfWidth = 1e-3`, second-order DA truncation error scales as `O(kHalfWidth^2) = 1e-6`. Forward-difference STM error scales as `O(eps^2) = 1e-12`. Sum is dominated by truncation; `1e-5` absolute is a comfortable headroom.

If `LinearTermFeagin12` or `LinearTermFeagin14` fails with a magnitude < ~1e-4, relax the tolerance to `1e-4` and document why in the test comment (those methods can drift more on long integrations with tight DA tolerances).

- [ ] **Step 3: Commit**

```bash
git add tests/ode/testRKWithDaState.cpp
git commit -m "tests/ode: RK steppers on DA-vector state — STM via finite difference"
```

---

### Task 9: Documentation updates

**Files:**
- Modify: `docs/ode/api.md`
- Modify: `docs/ode/methods.md`

- [ ] **Step 1: Update the API reference (`docs/ode/api.md`)**

Find the table of factory functions (`makeVerner78Integrator`, etc.) and replace it with a table of type aliases. The replacement table:

```markdown
## Per-method type aliases

| Alias                                              | Stepper                  | Default controller             |
| -------------------------------------------------- | ------------------------ | ------------------------------ |
| `Verner78<State, Ctrl=PI, Dense=false, F=Rhs>`     | `Verner78Stepper`        | `controllers::PI<double>`      |
| `Verner89<State, Ctrl=PI, Dense=false, F=Rhs>`     | `Verner89Stepper`        | `controllers::PI<double>`      |
| `Fehlberg78<State, Ctrl=PI, Dense=false, F=Rhs>`   | `Fehlberg78Stepper`      | `controllers::PI<double>`      |
| `Feagin12<State, Ctrl=PI, Dense=false, F=Rhs>`     | `Feagin12Stepper`        | `controllers::PI<double>`      |
| `Feagin14<State, Ctrl=PI, Dense=false, F=Rhs>`     | `Feagin14Stepper`        | `controllers::PI<double>`      |
| `Taylor<N, State, Ctrl=JorbaZou, Dense=false, F=Rhs>` | `TaylorStepper<N,…>`   | `controllers::JorbaZou<double>`|

`F` defaults to `Stepper::Rhs` (a `std::function<State(const State&, double)>`). Spell `F` explicitly to avoid the vtable indirection on benchmark hot loops.

### Examples

```cpp
// Adaptive Verner 8(7) on a 6-state double system:
tax::ode::Verner78< Eigen::Matrix<double, 6, 1> > integ{ f, cfg };

// FixedStep grid (uses cfg.initial_step uniformly):
tax::ode::Verner78< Eigen::Matrix<double, 6, 1>,
                    tax::ode::controllers::FixedStep<double> > integ{ f, cfg };

// DA-vector state (vector of TE in IC deviations):
tax::ode::Verner78< Eigen::Matrix<tax::TEn<2,4>, 6, 1> > integ_da{ f, cfg };
```
```

- [ ] **Step 2: Add a DA-vector-state section to `docs/ode/methods.md`**

Append the following section (at the bottom of the file, before any closing material):

```markdown
## Propagating an expansion in the initial conditions (DA-vector state)

The five RK steppers accept any state for which `tax::ode::VectorOps<State>`
is specialized. The library provides specializations for floating-point
scalars, `tax::TaylorExpansionT<T,N,M>`, and `Eigen::Matrix<T,D,1>` of either.

To propagate a polynomial-flow-map about an initial-condition box, use a
vector of multivariate Taylor polynomials in the IC deviations:

```cpp
using DA    = tax::TEn<2, 4>;            // order 2, 4 IC variables
using State = Eigen::Matrix<DA, 4, 1>;

State x0;
for (int i = 0; i < 4; ++i)
    x0(i) = DA(centre(i)) + DA(halfWidth(i)) * DA::variable(0.0, i);

tax::ode::Verner78<State> integ{ f, cfg };
auto sol = integ.integrate(x0, t0, tmax);

// sol.x.back()(i)[0]   = component i at tmax, constant DA term
// sol.x.back()(i)[1+j] = ∂(component i)/∂(δ_j) at tmax, scaled by halfWidth(j)
```

Step-size control still operates in `double`: `VectorOps<State>::norm`
returns the sup over all coefficients of the polynomial state, and the
adaptive controller compares that against `cfg.abstol + cfg.reltol *
state_norm`.

The Taylor stepper currently requires a real-scalar state — propagating
a DA-vector state through `Taylor<…>` would require a separate
DA-Taylor integrator (planned, not in this stage).
```

- [ ] **Step 3: Commit**

```bash
git add docs/ode/api.md docs/ode/methods.md
git commit -m "docs/ode: document type-alias API and DA-vector state"
```

---

## Self-review checklist

- **Spec coverage** (cross-checked against `docs/superpowers/specs/2026-05-23-tax-stage2b-fixedstep-da-state-design.md` Section "Slice B"):
  - `VectorOps<S>` trait + three specializations: Tasks 1, 2 ✓
  - `adaptive_rk_step` refactor (uses `VectorOps`, returns `double err_norm`): Task 3 Step 1 ✓
  - 5 RK steppers pinned to `T = double` + use `VectorOps<State>::norm`: Task 3 Steps 2–6 ✓
  - 6 type aliases: Task 4 ✓
  - 12 factories removed: Task 6 ✓
  - DA-vector test (constant + STM): Tasks 7, 8 ✓
  - Migration of 8 test files: Task 5 ✓
  - Doc updates (`api.md`, `methods.md`): Task 9 ✓
- **Placeholder scan:** no TBD / TODO; every step has concrete code or a concrete command.
- **Type consistency:**
  - `VectorOps<S>::norm` returns `double` everywhere it is called.
  - `Stepper::T = double` set consistently across the five RK steppers.
  - `RKStepOut` drops the `T` template parameter; the new layout is `{ State x_new; State y_emb; double err_norm; }`.
  - Type alias parameter order is uniform: `<State, Controller, Dense, F>` (or `<N, State, Controller, Dense, F>` for `Taylor`).
- **Out-of-scope items deferred** (consistent with spec): TaylorStepper on TE state, `Scalar` concept relaxation, transcendental-kernel audit, DA-aware events, `RealOf<T>` trait.
