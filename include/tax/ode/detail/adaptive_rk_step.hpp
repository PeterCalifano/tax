// include/tax/ode/detail/adaptive_rk_step.hpp
//
// Generic explicit Runge–Kutta step driver used by every Stage 2a RK
// stepper (Verner78, Verner89, Fehlberg78, Feagin12, Feagin14).
// Tableau is supplied as a struct exposing:
//   - static constexpr int n_stages
//   - static constexpr int order, order_emb
//   - static constexpr bool fsal           (first-same-as-last)
//   - static constexpr std::array<T, n_stages>            c
//   - static constexpr std::array<T, n_stages*(n_stages-1)/2> a  (row-major, lower-tri, no diag)
//   - static constexpr std::array<T, n_stages>            b
//   - static constexpr std::array<T, n_stages>            b_emb
// The Tableau type must be default-constructible and have all-static
// members; passes via template parameter, no instance needed.

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
    State y_emb;   // embedded estimate
    T     err_norm;
};

template < class Tab, class F, class State, class T, int NStages >
[[nodiscard]] RKStepOut< State, T > adaptive_rk_step(
    F&& f, const State& x, T t, T h, RKStepData< State, NStages >& work )
{
    static_assert( NStages == Tab::n_stages,
                   "adaptive_rk_step: stage-count mismatch with tableau" );

    // k_1 = f(x, t + c_0 * h)  (typically c_0 = 0).
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

    // Element-wise infinity norm of the difference, scaled by h.
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
