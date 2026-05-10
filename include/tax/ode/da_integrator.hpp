#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

#include <tax/ads/box.hpp>
#include <tax/ode/integrator.hpp>  // IntegratorConfig + detail::validate
#include <tax/ode/step.hpp>
#include <tax/tte.hpp>

namespace tax::ode
{

// =============================================================================
// FlowMap: state-vector polynomial in the initial-condition deviations
// =============================================================================

/**
 * @brief Polynomial flow map from initial conditions to propagated state.
 *
 * The `state` member is a vector of multivariate Taylor polynomials in the
 * normalised initial-condition deviations δ ∈ [−1, 1]^D.  Each component
 * `state(i)` is the degree-P Taylor expansion of `x_i(tmax, x0 + halfWidth ⊙ δ)`
 * about `x0 = box.center`.
 *
 * @tparam P  DA expansion order in the initial-condition variables.
 * @tparam D  State-space dimension (= number of DA variables).
 */
template < int P, int D >
struct FlowMap
{
    using DA    = TEn< P, D >;
    using Input = std::array< double, D >;

    Eigen::Matrix< DA, D, 1 > state{};
};

// =============================================================================
// Internal helpers for DA-valued time-Taylor coefficients
// =============================================================================

namespace detail
{

/// @brief Infinity norm of a DA polynomial (max absolute coefficient).
template < typename T, int P, int M >
[[nodiscard]] double infNorm( const TruncatedTaylorExpansionT< T, P, M >& x ) noexcept
{
    double n = 0.0;
    for ( std::size_t i = 0; i < TruncatedTaylorExpansionT< T, P, M >::nCoefficients; ++i )
        n = std::max( n, std::abs( x[i] ) );
    return n;
}

/// @brief Adaptive step size from the last two DA-valued Taylor coefficients
///        (Jorba–Zou criterion generalised by replacing scalar |.| with the
///        infinity norm of polynomial-valued coefficients).
template < int P, int M, int N >
[[nodiscard]] double stepsizeDa(
    const TruncatedTaylorExpansionT< TEn< P, M >, N, 1 >& x, double abstol ) noexcept
{
    double h = std::numeric_limits< double >::infinity();

    if constexpr ( N >= 2 )
    {
        const double c = infNorm( x[N - 1] );
        if ( c > 0.0 ) h = std::min( h, std::pow( abstol / c, 1.0 / ( N - 1 ) ) );
    }
    {
        const double c = infNorm( x[N] );
        if ( c > 0.0 ) h = std::min( h, std::pow( abstol / c, 1.0 / N ) );
    }
    return h;
}

template < int P, int M, int N, int D >
[[nodiscard]] double stepsizeDa(
    const Eigen::Matrix< TruncatedTaylorExpansionT< TEn< P, M >, N, 1 >, D, 1 >& x,
    double abstol ) noexcept
{
    double h = std::numeric_limits< double >::infinity();
    for ( Eigen::Index i = 0; i < x.size(); ++i )
        h = std::min( h, stepsizeDa< P, M, N >( x( i ), abstol ) );
    return h;
}

/// @brief Evaluate a DA-valued time-TTE at a scalar displacement (Horner).
template < typename DA, int N >
[[nodiscard]] DA evalAtScalar(
    const TruncatedTaylorExpansionT< DA, N, 1 >& poly, double dt ) noexcept
{
    DA result = poly[N];
    for ( int i = N - 1; i >= 0; --i )
    {
        result *= typename DA::scalar_type( dt );
        result += poly[i];
    }
    return result;
}

template < typename DA, int N, int D >
[[nodiscard]] Eigen::Matrix< DA, D, 1 > evalAtScalar(
    const Eigen::Matrix< TruncatedTaylorExpansionT< DA, N, 1 >, D, 1 >& poly,
    double dt ) noexcept
{
    Eigen::Matrix< DA, D, 1 > result( poly.size() );
    for ( Eigen::Index i = 0; i < poly.size(); ++i )
        result( i ) = evalAtScalar< DA, N >( poly( i ), dt );
    return result;
}

/// @brief Propagate a DA state from t0 to tmax — the work routine shared by
///        DaIntegrator and AdsIntegrator.
template < int N, int P, int D, typename F >
[[nodiscard]] Eigen::Matrix< TEn< P, D >, D, 1 > propagateDa( F&& f,
                                                              Eigen::Matrix< TEn< P, D >, D, 1 > xc,
                                                              double t0, double tmax,
                                                              double abstol, int max_steps );

}  // namespace detail

// =============================================================================
// Low-level building blocks (also useful directly for testing/inspection)
// =============================================================================

/**
 * @brief Build a DA initial state from a box of initial conditions.
 *
 * Component `i` becomes `box.center[i] + box.halfWidth[i] * δ_i` where
 * δ ∈ [−1, 1]^D is the normalised deviation vector.
 */
template < int P, int D >
[[nodiscard]] Eigen::Matrix< TEn< P, D >, D, 1 > makeDaState( const Box< double, D >& box )
{
    using DA = TEn< P, D >;

    Eigen::Matrix< DA, D, 1 > x0( D );
    for ( int i = 0; i < D; ++i )
    {
        typename DA::Data c{};
        c[0] = box.center[i];
        if constexpr ( P >= 1 )
        {
            MultiIndex< D > ei{};
            ei[i] = 1;
            c[tax::detail::flatIndex< D >( ei )] = box.halfWidth[i];
        }
        x0( i ) = DA{ c };
    }
    return x0;
}

/**
 * @brief Compute one Taylor step for a vector ODE with DA-expanded state.
 */
template < int N, int P, int D, typename F >
[[nodiscard]] StepResult<
    Eigen::Matrix< TruncatedTaylorExpansionT< TEn< P, D >, N, 1 >, D, 1 >, double >
stepDa( F&& f, const Eigen::Matrix< TEn< P, D >, D, 1 >& x0, double tc, double abstol )
{
    using DA     = TEn< P, D >;
    using TTE    = TruncatedTaylorExpansionT< DA, N, 1 >;
    using VecTTE = Eigen::Matrix< TTE, D, 1 >;

    const Eigen::Index dim = x0.size();

    TTE t_da{};
    t_da[0] = DA( tc );
    if constexpr ( N >= 1 ) t_da[1] = DA( 1.0 );

    VecTTE x_da( dim );
    for ( Eigen::Index i = 0; i < dim; ++i )
    {
        x_da( i )    = TTE{};
        x_da( i )[0] = x0( i );
    }

    VecTTE dx( dim );
    for ( int k = 0; k < N; ++k )
    {
        f( dx, x_da, t_da );
        for ( Eigen::Index i = 0; i < dim; ++i )
        {
            x_da( i )[k + 1] = dx( i )[k];
            x_da( i )[k + 1] /= double( k + 1 );
        }
    }

    auto h = detail::stepsizeDa< P, D, N >( x_da, abstol );
    return { std::move( x_da ), h };
}

namespace detail
{

template < int N, int P, int D, typename F >
[[nodiscard]] Eigen::Matrix< TEn< P, D >, D, 1 > propagateDa( F&& f,
                                                              Eigen::Matrix< TEn< P, D >, D, 1 > xc,
                                                              double t0, double tmax,
                                                              double abstol, int max_steps )
{
    double       tc = t0;
    const double s  = tmax >= t0 ? 1.0 : -1.0;

    for ( int step = 0; step < max_steps; ++step )
    {
        if ( s * ( tmax - tc ) <= 0.0 ) break;

        auto [p, h] = stepDa< N, P, D >( f, xc, tc, abstol );
        if ( h <= 0.0 ) break;

        const double dt = s * std::min( h, std::abs( tmax - tc ) );
        xc              = evalAtScalar< TEn< P, D >, N >( p, dt );
        tc += dt;
    }

    return xc;
}

}  // namespace detail

// =============================================================================
// DaIntegrator class
// =============================================================================

/**
 * @brief Adaptive Taylor integrator for the DA-expanded state of a vector ODE.
 *
 * Produces the polynomial flow map `x(tmax)` as a function of the normalised
 * initial-condition deviations δ ∈ [−1, 1]^D about a box centre, with no
 * domain splitting.  The right-hand side and configuration are bound at
 * construction; subsequent `integrate(box, t0, tmax)` calls only need the
 * IC domain and time range.
 *
 * @tparam N Taylor expansion order in time.
 * @tparam P DA expansion order in the initial-condition variables.
 * @tparam D State-space dimension (= number of DA variables).
 */
template < int N, int P, int D >
class DaIntegrator
{
public:
    using DA       = TEn< P, D >;
    using TimeTTE  = TruncatedTaylorExpansionT< DA, N, 1 >;
    using VecTTE   = Eigen::Matrix< TimeTTE, D, 1 >;
    using Rhs      = std::function< void( VecTTE&, const VecTTE&, const TimeTTE& ) >;
    using Config   = IntegratorConfig< double >;
    using FlowMapT = FlowMap< P, D >;

    explicit DaIntegrator( Rhs f, Config cfg = {} )
        : f_( std::move( f ) ), cfg_( cfg )
    {
        detail::validate( cfg_ );
    }

    [[nodiscard]] const Config& config() const noexcept { return cfg_; }

    /// @brief Integrate the DA state derived from @p box from @p t0 to @p tmax.
    [[nodiscard]] FlowMapT integrate( const Box< double, D >& box, double t0, double tmax ) const
    {
        auto x0 = makeDaState< P, D >( box );
        return FlowMapT{ detail::propagateDa< N, P, D >( f_, std::move( x0 ), t0, tmax,
                                                        cfg_.abstol, cfg_.max_steps ) };
    }

private:
    Rhs    f_;
    Config cfg_;
};

}  // namespace tax::ode
