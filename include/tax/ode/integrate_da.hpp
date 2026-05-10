#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

#include <tax/ads/box.hpp>
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
 * about `x0 = box.center`, so its coefficients are the state-transition
 * tensors of the flow.
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

/// @brief Adaptive step size from the last two DA-valued Taylor coefficients.
/// Generalises the Jorba–Zou (2005) criterion to polynomial-valued
/// coefficients by replacing the scalar absolute value with the infinity norm.
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

/// @brief Vector version: minimum step size across all state components.
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
/// Each coefficient is a DA polynomial; the displacement is a plain double,
/// so the computation is polynomial-scalar multiply + polynomial addition.
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

/// @brief Vector version: evaluate each component at the same scalar dt.
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

}  // namespace detail

// =============================================================================
// Single Taylor step with DA-valued state
// =============================================================================

/**
 * @brief Compute one Taylor step for a vector ODE with DA-expanded state.
 *
 * The state components are multivariate polynomials (TEn<P,D>) representing
 * a neighbourhood of initial conditions.  Time is a plain scalar.
 *
 * @tparam N  Taylor expansion order in time.
 * @tparam P  DA expansion order in the initial-condition variables.
 * @tparam D  State-space dimension (= number of DA variables).
 * @param f      Right-hand side `f(dx, x, t)`.
 * @param x0     Current DA state vector.
 * @param tc     Current time (scalar).
 * @param abstol Absolute tolerance for step-size control.
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

    // Time variable: t(τ) = tc + τ, with constant-DA coefficients.
    TTE t_da{};
    t_da[0] = DA( tc );
    if constexpr ( N >= 1 ) t_da[1] = DA( 1.0 );

    // State variables: x_da[0] = x0 (DA polynomial per component).
    VecTTE x_da( dim );
    for ( Eigen::Index i = 0; i < dim; ++i )
    {
        x_da( i ) = TTE{};
        x_da( i )[0] = x0( i );
    }

    // Picard iteration: x_da[k+1] = f(x_da, t_da)[k] / (k+1).
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

// =============================================================================
// Build DA initial conditions from a box
// =============================================================================

/**
 * @brief Create DA-expanded initial state from a box of initial conditions.
 *
 * Component `i` is the polynomial `center[i] + halfWidth[i] * δ_i`
 * where δ ∈ [−1, 1]^D is the normalised deviation.
 */
template < int P, int D >
[[nodiscard]] Eigen::Matrix< TEn< P, D >, D, 1 >
makeDaState( const Box< double, D >& box )
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

// =============================================================================
// Propagate a single subdomain from t0 to tmax
// =============================================================================

/**
 * @brief Integrate a vector ODE with DA-expanded state over one subdomain.
 *
 * Builds the DA initial conditions from @p box and advances from @p t0 to
 * @p tmax with adaptive Taylor stepping.  Returns the final DA state vector
 * — the flow expansion x(tmax) as a degree-P polynomial in the normalised
 * initial-condition deviations δ.
 *
 * This is the entry point for users who only want the flow polynomial and
 * do not need automatic domain splitting.
 *
 * @tparam N  Taylor expansion order in time.
 * @tparam P  DA expansion order in state.
 * @tparam D  State-space dimension.
 */
template < int N, int P, int D, typename F >
[[nodiscard]] Eigen::Matrix< TEn< P, D >, D, 1 >
propagateBox( F&& f, const Box< double, D >& box, double t0, double tmax,
              double abstol, int maxsteps = 500 )
{
    auto xc        = makeDaState< P, D >( box );
    double tc      = t0;
    const double s = tmax >= t0 ? 1.0 : -1.0;

    for ( int step = 0; step < maxsteps; ++step )
    {
        if ( s * ( tmax - tc ) <= 0.0 ) break;

        auto [p, h] = stepDa< N, P, D >( f, xc, tc, abstol );
        if ( h <= 0.0 ) break;

        const double dt = s * std::min( h, std::abs( tmax - tc ) );
        xc = detail::evalAtScalar< TEn< P, D >, N >( p, dt );
        tc += dt;
    }

    return xc;
}

}  // namespace tax::ode
