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
