#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
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

/**
 * @brief Adaptive Taylor integrator for scalar and Eigen-vector ODEs.
 *
 * @tparam N Taylor expansion order in time.
 * @tparam T Scalar coefficient type (default `double`).
 *
 * @details The integrator is constructed with a configuration once, then
 *   `integrate()` may be called repeatedly with different right-hand sides,
 *   initial states, and time ranges.  Optional event detection is supported
 *   via the @ref Event vector overload.
 *
 *   The class itself is stateless: every `integrate()` call produces an
 *   independent solution, so the same integrator instance is safe to reuse
 *   across threads.
 */
template < int N, typename T = double >
class Integrator
{
public:
    using Config = IntegratorConfig< T >;

    /**
     * @brief Construct with the given configuration.
     * @throws std::invalid_argument if any configuration value is invalid.
     */
    explicit Integrator( Config cfg = {} ) : cfg_( cfg ) { detail::validate( cfg_ ); }

    [[nodiscard]] const Config& config() const noexcept { return cfg_; }

    // -------------------------------------------------------------------------
    // Scalar ODE
    // -------------------------------------------------------------------------

    /**
     * @brief Integrate scalar ODE `dx/dt = f(x, t)` with adaptive step size.
     */
    template < typename F >
    [[nodiscard]] TaylorSolution< N, T, T > integrate( F&& f, T x0, T t0, T tmax ) const
    {
        return integrateImpl( std::forward< F >( f ), x0, t0, tmax,
                              std::vector< Event< N, T, T > >{} );
    }

    /// @overload Same with event detection.
    template < typename F >
    [[nodiscard]] TaylorSolution< N, T, T >
    integrate( F&& f, T x0, T t0, T tmax, const std::vector< Event< N, T, T > >& events ) const
    {
        return integrateImpl( std::forward< F >( f ), x0, t0, tmax, events );
    }

    // -------------------------------------------------------------------------
    // Vector ODE
    // -------------------------------------------------------------------------

    /**
     * @brief Integrate vector ODE `f(dx, x, t)` with adaptive step size.
     */
    template < typename F, int D >
    [[nodiscard]] TaylorSolution< N, Eigen::Matrix< T, D, 1 >, T >
    integrate( F&& f, const Eigen::Matrix< T, D, 1 >& x0, T t0, T tmax ) const
    {
        return integrateImpl< F, D >(
            std::forward< F >( f ), x0, t0, tmax,
            std::vector< Event< N, Eigen::Matrix< T, D, 1 >, T > >{} );
    }

    /// @overload Same with event detection.
    template < typename F, int D >
    [[nodiscard]] TaylorSolution< N, Eigen::Matrix< T, D, 1 >, T >
    integrate( F&& f, const Eigen::Matrix< T, D, 1 >& x0, T t0, T tmax,
               const std::vector< Event< N, Eigen::Matrix< T, D, 1 >, T > >& events ) const
    {
        return integrateImpl< F, D >( std::forward< F >( f ), x0, t0, tmax, events );
    }

private:
    Config cfg_;

    template < typename F >
    [[nodiscard]] TaylorSolution< N, T, T > integrateImpl(
        F&& f, T x0, T t0, T tmax, const std::vector< Event< N, T, T > >& events ) const
    {
        TaylorSolution< N, T, T > sol;
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

            auto [p, h] = step< N >( f, xc, tc, cfg_.abstol );
            if ( h <= T{} ) break;

            T dt = sign * std::min( h, std::abs( tmax - tc ) );

            const std::size_t step_idx = sol.p.size();
            const auto er = detail::processStepEvents< N, T, T >( events, p, tc, dt, sol.events,
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

    template < typename F, int D >
    [[nodiscard]] TaylorSolution< N, Eigen::Matrix< T, D, 1 >, T > integrateImpl(
        F&& f, const Eigen::Matrix< T, D, 1 >& x0, T t0, T tmax,
        const std::vector< Event< N, Eigen::Matrix< T, D, 1 >, T > >& events ) const
    {
        using Vec = Eigen::Matrix< T, D, 1 >;

        TaylorSolution< N, Vec, T > sol;
        sol.t.reserve( std::size_t( cfg_.max_steps + 1 ) );
        sol.x.reserve( std::size_t( cfg_.max_steps + 1 ) );
        sol.p.reserve( std::size_t( cfg_.max_steps + 1 ) );
        sol.t.push_back( t0 );
        sol.x.push_back( x0 );

        const T sign = tmax >= t0 ? T{ 1 } : T{ -1 };
        T       tc   = t0;
        Vec     xc   = x0;

        for ( int s = 0; s < cfg_.max_steps; ++s )
        {
            if ( sign * ( tmax - tc ) <= T{} ) break;

            auto [p, h] = step< N >( f, xc, tc, cfg_.abstol );
            if ( h <= T{} ) break;

            T dt = sign * std::min( h, std::abs( tmax - tc ) );

            const std::size_t step_idx = sol.p.size();
            const auto er = detail::processStepEvents< N, Vec, T >( events, p, tc, dt, sol.events,
                                                                    step_idx );
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
};

}  // namespace tax::ode
