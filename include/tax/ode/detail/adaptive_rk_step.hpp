// include/tax/ode/detail/adaptive_rk_step.hpp
//
// Generic explicit Runge–Kutta step driver. Routes state arithmetic
// through tax::ode::VectorOps<State>, which decouples step-size
// control (always double) from the state's scalar layout. Same body
// serves double-state and DA-vector-state.

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <tax/ode/controllers.hpp>
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

/**
 * @brief Resolve `(h_next, accepted)` for the next RK step.
 *
 * Centralises the three-arm controller dispatch shared by every
 * Stage 2a RK stepper:
 *   - `FixedStep` ............ always accepted, h_next = h.
 *   - `JorbaZou` ............. Taylor-only; no-op fallback for RK.
 *   - any other controller ... `controller.next_step(...)` + tolerance check.
 *
 * Feagin pairs feed their floored error to the controller via
 * `err_for_ctrl` while still using the raw `err_norm` for the
 * acceptance decision; all other RK steppers pass the same value for
 * both arguments.
 *
 * @param controller     Controller instance (state-mutating for PI/H211b).
 * @param h              Step size that was just taken.
 * @param err_for_ctrl   Error magnitude handed to the controller.
 * @param err_norm       Raw error magnitude compared against `tol`.
 * @param tol            Acceptance threshold.
 * @param order_emb      Embedded estimator order (Tab::order_emb).
 */
template < class Controller >
[[nodiscard]] inline std::pair< double, bool > select_rk_step(
    Controller& controller,
    double      h,
    double      err_for_ctrl,
    double      err_norm,
    double      tol,
    int         order_emb )
{
    if constexpr ( std::is_same_v< Controller, controllers::FixedStep< double > > )
    {
        (void) controller; (void) err_for_ctrl; (void) err_norm;
        (void) tol; (void) order_emb;
        return { h, true };
    }
    else if constexpr ( std::is_same_v< Controller, controllers::JorbaZou< double > > )
    {
        (void) controller; (void) err_for_ctrl; (void) order_emb;
        // JorbaZou is Taylor-only; RK steppers fall back to identity h.
        return { h, err_norm <= tol };
    }
    else
    {
        return { controller.next_step( h, err_for_ctrl, tol, order_emb ),
                 err_norm <= tol };
    }
}

}  // namespace tax::ode::detail
