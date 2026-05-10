#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <tax/eigen/eval.hpp>
#include <tax/ode/events.hpp>
#include <tax/ode/solution.hpp>
#include <tax/ode/step.hpp>

namespace tax::ode
{

/**
 * @brief Configuration for the plain (scalar / Eigen-vector) Taylor integrator.
 */
template < typename T = double >
struct IntegratorConfig
{
    T   abstol    = T{ 1e-14 };  ///< Absolute tolerance for adaptive step-size control.
    int max_steps = 500;         ///< Maximum number of integration steps.
};

namespace detail
{

template < typename T >
inline void validate( const IntegratorConfig< T >& cfg )
{
    if ( !( cfg.abstol > T{ 0 } ) )
        throw std::invalid_argument( "IntegratorConfig: abstol must be > 0" );
    if ( cfg.max_steps <= 0 )
        throw std::invalid_argument( "IntegratorConfig: max_steps must be > 0" );
}

}  // namespace detail

// =============================================================================
// Integrator — primary template (must specialize on a supported State)
// =============================================================================

/**
 * @brief Adaptive Taylor integrator for scalar or Eigen-vector ODEs.
 *
 * @tparam N      Taylor expansion order in time.
 * @tparam State  `T` for scalar ODEs, `Eigen::Matrix<T, D, 1>` for vector ODEs.
 * @tparam T      Scalar coefficient type (default `double`).
 *
 * @details The right-hand side, configuration, and event list are bound at
 *   construction; subsequent `integrate(x0, t0, tmax)` calls only need the
 *   initial state and time range.  The instance is read-only after
 *   construction so it can be reused (and shared across threads) safely.
 */
template < int N, typename State, typename T = double >
class Integrator;  // primary, undefined

// -----------------------------------------------------------------------------
// Scalar specialization: State == T
// -----------------------------------------------------------------------------

template < int N, typename T >
class Integrator< N, T, T >
{
public:
    using TimeTTE   = TruncatedTaylorExpansionT< T, N, 1 >;
    using Rhs       = std::function< TimeTTE( const TimeTTE&, const TimeTTE& ) >;
    using Config    = IntegratorConfig< T >;
    using EventList = std::vector< Event< N, T, T > >;
    using Solution  = TaylorSolution< N, T, T >;

    /**
     * @brief Construct an integrator bound to a given RHS, configuration, and
     *        (optional) event list.
     * @throws std::invalid_argument if the configuration is invalid.
     */
    explicit Integrator( Rhs f, Config cfg = {}, EventList events = {} )
        : f_( std::move( f ) ), cfg_( cfg ), events_( std::move( events ) )
    {
        detail::validate( cfg_ );
    }

    [[nodiscard]] const Config&    config() const noexcept { return cfg_; }
    [[nodiscard]] const EventList& events() const noexcept { return events_; }

    /// @brief Integrate `dx/dt = f(x, t)` from @p x0 at @p t0 to @p tmax.
    [[nodiscard]] Solution integrate( T x0, T t0, T tmax ) const
    {
        Solution sol;
        sol.t.reserve( std::size_t( cfg_.max_steps + 1 ) );
        sol.x.reserve( std::size_t( cfg_.max_steps + 1 ) );
        sol.p.reserve( std::size_t( cfg_.max_steps + 1 ) );
        sol.t.push_back( t0 );
        sol.x.push_back( x0 );

        const T sign = tmax >= t0 ? T{ 1 } : T{ -1 };
        T       tc   = t0;
        T       xc   = x0;

        for ( int s = 0; s < cfg_.max_steps; ++s )
        {
            if ( sign * ( tmax - tc ) <= T{} ) break;

            auto [p, h] = step< N >( f_, xc, tc, cfg_.abstol );
            if ( h <= T{} ) break;

            T dt = sign * std::min( h, std::abs( tmax - tc ) );

            const std::size_t step_idx = sol.p.size();
            const auto er = detail::processStepEvents< N, T, T >( events_, p, tc, dt, sol.events,
                                                                  step_idx );
            dt = er.effective_dt;

            xc = p.eval( dt );
            sol.p.push_back( std::move( p ) );
            tc += dt;

            sol.t.push_back( tc );
            sol.x.push_back( xc );

            if ( er.terminate ) break;
        }

        return sol;
    }

private:
    Rhs       f_;
    Config    cfg_;
    EventList events_;
};

// -----------------------------------------------------------------------------
// Vector specialization: State == Eigen::Matrix<T, D, 1>
// -----------------------------------------------------------------------------

template < int N, typename T, int D >
class Integrator< N, Eigen::Matrix< T, D, 1 >, T >
{
public:
    using State     = Eigen::Matrix< T, D, 1 >;
    using TimeTTE   = TruncatedTaylorExpansionT< T, N, 1 >;
    using VecTTE    = Eigen::Matrix< TimeTTE, D, 1 >;
    using Rhs       = std::function< void( VecTTE&, const VecTTE&, const TimeTTE& ) >;
    using Config    = IntegratorConfig< T >;
    using EventList = std::vector< Event< N, State, T > >;
    using Solution  = TaylorSolution< N, State, T >;

    explicit Integrator( Rhs f, Config cfg = {}, EventList events = {} )
        : f_( std::move( f ) ), cfg_( cfg ), events_( std::move( events ) )
    {
        detail::validate( cfg_ );
    }

    [[nodiscard]] const Config&    config() const noexcept { return cfg_; }
    [[nodiscard]] const EventList& events() const noexcept { return events_; }

    /// @brief Integrate `f(dx, x, t)` from @p x0 at @p t0 to @p tmax.
    [[nodiscard]] Solution integrate( State x0, T t0, T tmax ) const
    {
        Solution sol;
        sol.t.reserve( std::size_t( cfg_.max_steps + 1 ) );
        sol.x.reserve( std::size_t( cfg_.max_steps + 1 ) );
        sol.p.reserve( std::size_t( cfg_.max_steps + 1 ) );
        sol.t.push_back( t0 );
        sol.x.push_back( x0 );

        const T sign = tmax >= t0 ? T{ 1 } : T{ -1 };
        T       tc   = t0;
        State   xc   = x0;

        for ( int s = 0; s < cfg_.max_steps; ++s )
        {
            if ( sign * ( tmax - tc ) <= T{} ) break;

            auto [p, h] = step< N >( f_, xc, tc, cfg_.abstol );
            if ( h <= T{} ) break;

            T dt = sign * std::min( h, std::abs( tmax - tc ) );

            const std::size_t step_idx = sol.p.size();
            const auto er = detail::processStepEvents< N, State, T >( events_, p, tc, dt,
                                                                      sol.events, step_idx );
            dt = er.effective_dt;

            xc = eval( p, dt );
            sol.p.push_back( std::move( p ) );
            tc += dt;

            sol.t.push_back( tc );
            sol.x.push_back( xc );

            if ( er.terminate ) break;
        }

        return sol;
    }

private:
    Rhs       f_;
    Config    cfg_;
    EventList events_;
};

}  // namespace tax::ode
