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
 * @brief Polynomial flow map from initial conditions (and, optionally,
 *        parameters) to the propagated state.
 *
 * The `state` member is a vector of multivariate Taylor polynomials in the
 * normalised deviations δ ∈ [−1, 1]^{D+Q}.  The first @p D coordinates
 * correspond to initial-condition deviations and the trailing @p Q
 * coordinates (when @p Q > 0) to parameter deviations.  Each component
 * `state(i)` is the degree-P Taylor expansion of
 * `x_i(tmax, x0 + halfWidth_x ⊙ δ_x, p0 + halfWidth_p ⊙ δ_p)`
 * about `(x0, p0) = (box.center, p_box.center)`.
 *
 * @tparam P  DA expansion order.
 * @tparam D  State-space dimension (= number of IC DA variables).
 * @tparam Q  Number of parameter DA variables (0 disables parameter expansion).
 */
template < int P, int D, int Q = 0 >
struct FlowMap
{
    static constexpr int M = D + Q;
    using DA    = TEn< P, M >;
    using Input = std::array< double, M >;

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

/// @brief Propagate a DA state with constant DA parameters from t0 to tmax.
template < int N, int P, int D, int Q, typename F >
[[nodiscard]] Eigen::Matrix< TEn< P, D + Q >, D, 1 > propagateDa(
    F&& f, Eigen::Matrix< TEn< P, D + Q >, D, 1 > xc,
    const Eigen::Matrix< TEn< P, D + Q >, Q, 1 >& p, double t0, double tmax, double abstol,
    int max_steps );

}  // namespace detail

// =============================================================================
// Low-level building blocks (also useful directly for testing/inspection)
// =============================================================================

/**
 * @brief Build a DA initial state from a box of initial conditions.
 *
 * Component `i` becomes `box.center[i] + box.halfWidth[i] * δ_i` where
 * δ ∈ [−1, 1]^{D+Q} is the normalised deviation vector.  The IC variables
 * occupy slots `0..D-1`; the trailing @p Q slots are reserved for parameters
 * and remain zero here (see @ref makeDaParams).
 *
 * @tparam P  DA expansion order.
 * @tparam D  Number of state components (= number of IC DA variables).
 * @tparam Q  Number of parameter DA variables (default 0).
 */
template < int P, int D, int Q = 0 >
[[nodiscard]] Eigen::Matrix< TEn< P, D + Q >, D, 1 > makeDaState( const Box< double, D >& box )
{
    constexpr int M = D + Q;
    using DA        = TEn< P, M >;

    Eigen::Matrix< DA, D, 1 > x0( D );
    for ( int i = 0; i < D; ++i )
    {
        typename DA::Data c{};
        c[0] = box.center[i];
        if constexpr ( P >= 1 )
        {
            MultiIndex< M > ei{};
            ei[i] = 1;
            c[tax::detail::flatIndex< M >( ei )] = box.halfWidth[i];
        }
        x0( i ) = DA{ c };
    }
    return x0;
}

/**
 * @brief Build a DA parameter vector from a box of parameter values.
 *
 * Each parameter `p_i` becomes `p_box.center[i] + p_box.halfWidth[i] * δ_{D+i}`
 * — a DA polynomial that is constant in time but carries a linear term in
 * its dedicated DA variable.  Designed to be passed unchanged into the
 * user RHS at every internal step.
 *
 * @tparam P  DA expansion order.
 * @tparam D  Number of IC DA variables (occupy slots `0..D-1`).
 * @tparam Q  Number of parameter DA variables (occupy slots `D..D+Q-1`).
 */
template < int P, int D, int Q >
[[nodiscard]] Eigen::Matrix< TEn< P, D + Q >, Q, 1 > makeDaParams( const Box< double, Q >& p_box )
{
    static_assert( Q >= 1, "makeDaParams requires Q >= 1" );
    constexpr int M = D + Q;
    using DA        = TEn< P, M >;

    Eigen::Matrix< DA, Q, 1 > p0( Q );
    for ( int i = 0; i < Q; ++i )
    {
        typename DA::Data c{};
        c[0] = p_box.center[i];
        if constexpr ( P >= 1 )
        {
            MultiIndex< M > ei{};
            ei[D + i] = 1;
            c[tax::detail::flatIndex< M >( ei )] = p_box.halfWidth[i];
        }
        p0( i ) = DA{ c };
    }
    return p0;
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

/**
 * @brief Compute one Taylor step for a vector ODE with DA-expanded state and
 *        constant DA parameter vector.
 *
 * The user-supplied right-hand side has the form `f(dx, x, p, t)` where:
 *  - `x`, `dx` are length-D vectors of TTE< DA, N, 1 > polynomials in local
 *    step time, with coefficient type DA = `TEn<P, D+Q>`;
 *  - `p` is a length-Q vector of DAs (constant in time, but carrying linear
 *    terms in their dedicated DA variables);
 *  - `t` is the time TTE.
 */
template < int N, int P, int D, int Q, typename F >
[[nodiscard]] StepResult<
    Eigen::Matrix< TruncatedTaylorExpansionT< TEn< P, D + Q >, N, 1 >, D, 1 >, double >
stepDa( F&& f, const Eigen::Matrix< TEn< P, D + Q >, D, 1 >& x0,
        const Eigen::Matrix< TEn< P, D + Q >, Q, 1 >& p, double tc, double abstol )
{
    constexpr int M = D + Q;
    using DA        = TEn< P, M >;
    using TTE       = TruncatedTaylorExpansionT< DA, N, 1 >;
    using VecTTE    = Eigen::Matrix< TTE, D, 1 >;

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
        f( dx, x_da, p, t_da );
        for ( Eigen::Index i = 0; i < dim; ++i )
        {
            x_da( i )[k + 1] = dx( i )[k];
            x_da( i )[k + 1] /= double( k + 1 );
        }
    }

    auto h = detail::stepsizeDa< P, M, N >( x_da, abstol );
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

template < int N, int P, int D, int Q, typename F >
[[nodiscard]] Eigen::Matrix< TEn< P, D + Q >, D, 1 > propagateDa(
    F&& f, Eigen::Matrix< TEn< P, D + Q >, D, 1 > xc,
    const Eigen::Matrix< TEn< P, D + Q >, Q, 1 >& p, double t0, double tmax, double abstol,
    int max_steps )
{
    double       tc = t0;
    const double s  = tmax >= t0 ? 1.0 : -1.0;

    for ( int step = 0; step < max_steps; ++step )
    {
        if ( s * ( tmax - tc ) <= 0.0 ) break;

        auto [poly, h] = stepDa< N, P, D, Q >( f, xc, p, tc, abstol );
        if ( h <= 0.0 ) break;

        const double dt = s * std::min( h, std::abs( tmax - tc ) );
        xc              = evalAtScalar< TEn< P, D + Q >, N >( poly, dt );
        tc += dt;
    }

    return xc;
}

}  // namespace detail

// =============================================================================
// DaIntegrator class
// =============================================================================

/**
 * @brief Adaptive Taylor integrator for the DA-expanded state of a vector ODE,
 *        optionally expanded jointly w.r.t. a parameter box.
 *
 * Produces the polynomial flow map `x(tmax)` as a function of the normalised
 * deviations δ ∈ [−1, 1]^{D+Q} about a chosen IC centre (and parameter centre,
 * when @p Q > 0), with no domain splitting.  The right-hand side and
 * configuration are bound at construction; subsequent `integrate(...)` calls
 * supply the IC box (and, when @p Q > 0, the parameter box) and the time
 * range.
 *
 *  - When `Q == 0`, the RHS has the form `f(dx, x, t)` and `integrate(box, t0,
 *    tmax)` returns a flow map in @p D IC variables.
 *  - When `Q > 0`, the RHS has the form `f(dx, x, p, t)` (with `p` a length-Q
 *    DA-vector forwarded unchanged on every internal evaluation) and
 *    `integrate(x_box, p_box, t0, tmax)` returns a flow map in @p D IC plus
 *    @p Q parameter variables.
 *
 * @tparam N Taylor expansion order in time.
 * @tparam P DA expansion order.
 * @tparam D State-space dimension (= number of IC DA variables).
 * @tparam Q Number of parameter DA variables (default 0: no parameter expansion).
 */
template < int N, int P, int D, int Q = 0 >
class DaIntegrator
{
public:
    static constexpr int M = D + Q;

    using DA       = TEn< P, M >;
    using TimeTTE  = TruncatedTaylorExpansionT< DA, N, 1 >;
    using VecTTE   = Eigen::Matrix< TimeTTE, D, 1 >;
    using VecDaP   = Eigen::Matrix< DA, Q, 1 >;
    using Config   = IntegratorConfig< double >;
    using FlowMapT = FlowMap< P, D, Q >;

    using RhsNoParams =
        std::function< void( VecTTE&, const VecTTE&, const TimeTTE& ) >;
    using RhsWithParams =
        std::function< void( VecTTE&, const VecTTE&, const VecDaP&, const TimeTTE& ) >;
    using Rhs = std::conditional_t< ( Q == 0 ), RhsNoParams, RhsWithParams >;

    explicit DaIntegrator( Rhs f, Config cfg = {} )
        : f_( std::move( f ) ), cfg_( cfg )
    {
        detail::validate( cfg_ );
    }

    [[nodiscard]] const Config& config() const noexcept { return cfg_; }

    /// @brief Integrate the DA state derived from @p box from @p t0 to @p tmax.
    [[nodiscard]] FlowMapT integrate( const Box< double, D >& box, double t0, double tmax ) const
        requires( Q == 0 )
    {
        auto x0 = makeDaState< P, D >( box );
        return FlowMapT{ detail::propagateDa< N, P, D >( f_, std::move( x0 ), t0, tmax,
                                                        cfg_.abstol, cfg_.max_steps ) };
    }

    /// @brief Integrate the DA state derived from @p x_box, with parameters
    ///        expanded about @p p_box, from @p t0 to @p tmax.
    [[nodiscard]] FlowMapT integrate( const Box< double, D >& x_box,
                                      const Box< double, Q >& p_box, double t0,
                                      double tmax ) const
        requires( Q > 0 )
    {
        auto x0 = makeDaState< P, D, Q >( x_box );
        auto p0 = makeDaParams< P, D, Q >( p_box );
        return FlowMapT{ detail::propagateDa< N, P, D, Q >(
            f_, std::move( x0 ), p0, t0, tmax, cfg_.abstol, cfg_.max_steps ) };
    }

private:
    Rhs    f_;
    Config cfg_;
};

}  // namespace tax::ode
