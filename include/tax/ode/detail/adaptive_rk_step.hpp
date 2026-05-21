// include/tax/ode/detail/adaptive_rk_step.hpp
//
// Generic explicit Runge–Kutta step driver used by every Stage 2a RK
// stepper (Verner78, Verner89, Fehlberg78, Feagin12, Feagin14).
// Tableau is supplied as a struct exposing static-only members
// (n_stages, order, order_emb, fsal, c, a, b, b_emb).

#pragma once

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <utility>

namespace tax::ode::detail
{

template < class State, int NStages >
struct RKStepData
{
    std::array< State, NStages > k{};
};

template < class State, class T >
struct RKStepOut
{
    State x_new;
    State y_emb;
    T     err_norm;
};

template < class Tab, class F, class State, class T, int NStages >
[[nodiscard]] RKStepOut< State, T > adaptive_rk_step(
    F&& f, const State& x, T t, T h, RKStepData< State, NStages >& work )
{
    static_assert( NStages == Tab::n_stages,
                   "adaptive_rk_step: stage-count mismatch with tableau" );

    work.k[ 0 ] = f( x, t + T( Tab::c[ 0 ] ) * h );

    std::size_t a_off = 0;
    for ( int i = 1; i < NStages; ++i )
    {
        State y = x;
        for ( int j = 0; j < i; ++j )
            y += h * T( Tab::a[ a_off + std::size_t( j ) ] ) * work.k[ std::size_t( j ) ];
        work.k[ std::size_t( i ) ] = f( y, t + T( Tab::c[ std::size_t( i ) ] ) * h );
        a_off += std::size_t( i );
    }

    State x_new = x;
    State y_emb = x;
    for ( int i = 0; i < NStages; ++i )
    {
        x_new += h * T( Tab::b    [ std::size_t( i ) ] ) * work.k[ std::size_t( i ) ];
        y_emb += h * T( Tab::b_emb[ std::size_t( i ) ] ) * work.k[ std::size_t( i ) ];
    }

    T err_norm{ 0 };
    for ( Eigen::Index r = 0; r < x_new.size(); ++r )
    {
        using std::abs;
        const T d = T( abs( x_new( r ) - y_emb( r ) ) );
        if ( d > err_norm ) err_norm = d;
    }

    return { std::move( x_new ), std::move( y_emb ), err_norm };
}

}  // namespace tax::ode::detail
