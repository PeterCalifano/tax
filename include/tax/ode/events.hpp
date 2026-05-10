#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

#include <tax/eigen/eval.hpp>
#include <tax/ode/solution.hpp>
#include <tax/tte.hpp>

namespace tax::ode
{

/**
 * @brief Specification of a single ODE event.
 *
 * @tparam N      Taylor expansion order in time.
 * @tparam State  Scalar `T` (scalar ODE) or `Eigen::Matrix<T,D,1>` (vector ODE).
 * @tparam T      Scalar coefficient type.
 *
 * @details An event is the zero set of a user-supplied function
 *   `g(x, t)`.  At each integration step the function is composed with
 *   the step's Taylor polynomial to obtain a univariate polynomial in
 *   the local time coordinate τ; a strict sign change within the step
 *   is then located by bisection.
 *
 * The callable @ref g must accept the same polynomial state and time
 * arguments produced by `step()` (a TTE for scalar ODEs, an
 * `Eigen::Matrix<TTE,D,1>` for vector ODEs) and return the scalar
 * event-function TTE.  A generic lambda
 * `[](const auto& x, const auto& t){ ... }` is the natural form.
 */
template < int N, typename State, typename T = double >
struct Event
{
    using StatePoly = detail::poly_t< N, State, T >;
    using TimePoly = TruncatedTaylorExpansionT< T, N, 1 >;
    using GFn = std::function< TimePoly( const StatePoly&, const TimePoly& ) >;

    GFn g;                                                  ///< Event function `g(x,t)`.
    EventDirection direction = EventDirection::Any;         ///< Crossing-direction filter (in t).
    bool terminal = false;                                  ///< Stop integration on detection.
};

namespace detail
{

/// @brief Locate a root of `g_poly` inside `[tau_lo, tau_hi]` by bisection.
/// @pre  `g_poly.eval(tau_lo) * g_poly.eval(tau_hi) <= 0` (strict sign change).
template < typename T, int N >
[[nodiscard]] T bisectEvent( const TruncatedTaylorExpansionT< T, N, 1 >& g_poly, T tau_lo,
                             T tau_hi ) noexcept
{
    constexpr int max_iter = 80;
    const T eps = std::numeric_limits< T >::epsilon() * T{ 16 };

    T g_lo = g_poly.eval( tau_lo );

    for ( int it = 0; it < max_iter; ++it )
    {
        const T tau_mid = T{ 0.5 } * ( tau_lo + tau_hi );
        const T width = tau_hi - tau_lo;
        if ( width < eps * ( T{ 1 } + std::abs( tau_mid ) ) ) return tau_mid;

        const T g_mid = g_poly.eval( tau_mid );
        if ( g_mid == T{ 0 } ) return tau_mid;

        if ( ( g_lo < T{ 0 } ) == ( g_mid < T{ 0 } ) )
        {
            tau_lo = tau_mid;
            g_lo = g_mid;
        }
        else
        {
            tau_hi = tau_mid;
        }
    }
    return T{ 0.5 } * ( tau_lo + tau_hi );
}

/// @brief Build the time-TTE used during a step: t(τ) = tc + τ.
template < int N, typename T >
[[nodiscard]] TruncatedTaylorExpansionT< T, N, 1 > makeTimePoly( T tc ) noexcept
{
    TruncatedTaylorExpansionT< T, N, 1 > t_da{};
    t_da[0] = tc;
    if constexpr ( N >= 1 ) t_da[1] = T{ 1 };
    return t_da;
}

/// @brief Result returned by per-step event processing.
template < typename T >
struct StepEventResult
{
    T effective_dt;   ///< Effective step displacement (truncated at first terminal hit).
    bool terminate;   ///< Whether a terminal event was detected.
};

/// @brief Detect events within one step and append records to @p out.
///
/// Iterates over all events, composes each with the step's Taylor
/// polynomials, checks for a strict sign change over `(0, dt)` filtered by
/// the requested direction (in t), and bisects the resulting univariate
/// polynomial in τ.  Records are sorted by step-progression (smallest |τ|
/// first); the earliest terminal hit truncates the step.
///
/// @tparam StepPoly  Either `TruncatedTaylorExpansionT<T,N,1>` or
///                   `Eigen::Matrix<TruncatedTaylorExpansionT<T,N,1>,D,1>`.
template < int N, typename State, typename T, typename StepPoly >
[[nodiscard]] StepEventResult< T > processStepEvents(
    const std::vector< Event< N, State, T > >& events, const StepPoly& p, T tc, T dt,
    std::vector< EventRecord< N, State, T > >& out, std::size_t step_idx )
{
    using TimePoly = TruncatedTaylorExpansionT< T, N, 1 >;

    if ( events.empty() || dt == T{ 0 } ) return { dt, false };

    auto t_da = makeTimePoly< N, T >( tc );

    struct Hit
    {
        std::size_t i;
        T tau;
        EventDirection observed;
        TimePoly g_poly;
    };

    std::vector< Hit > hits;
    hits.reserve( events.size() );

    for ( std::size_t i = 0; i < events.size(); ++i )
    {
        const auto& ev = events[i];
        if ( !ev.g ) continue;

        TimePoly g_poly = ev.g( p, t_da );
        const T g0 = g_poly[0];
        const T g1 = eval_poly( g_poly, dt );

        if ( !( g0 * g1 < T{ 0 } ) ) continue;  // require strict sign change

        const EventDirection observed = ( ( g1 - g0 ) / dt > T{ 0 } )
            ? EventDirection::Increasing
            : EventDirection::Decreasing;
        if ( ev.direction != EventDirection::Any && ev.direction != observed ) continue;

        const T tau_lo = std::min( T{ 0 }, dt );
        const T tau_hi = std::max( T{ 0 }, dt );
        const T tau_event = bisectEvent< T, N >( g_poly, tau_lo, tau_hi );

        hits.push_back( { i, tau_event, observed, std::move( g_poly ) } );
    }

    std::sort( hits.begin(), hits.end(), []( const Hit& a, const Hit& b ) {
        return std::abs( a.tau ) < std::abs( b.tau );
    } );

    T effective_dt = dt;
    bool terminate = false;
    for ( auto& hit : hits )
    {
        EventRecord< N, State, T > rec;
        rec.t = tc + hit.tau;
        rec.x = eval_poly( p, hit.tau );
        rec.tau = hit.tau;
        rec.step_idx = step_idx;
        rec.event_idx = hit.i;
        rec.direction = hit.observed;
        rec.g_poly = std::move( hit.g_poly );
        out.push_back( std::move( rec ) );

        if ( events[hit.i].terminal )
        {
            effective_dt = hit.tau;
            terminate = true;
            break;
        }
    }

    return { effective_dt, terminate };
}

}  // namespace detail

}  // namespace tax::ode
