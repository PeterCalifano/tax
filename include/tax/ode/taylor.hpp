#pragma once

/**
 * @file tax/ode/taylor.hpp
 * @brief Taylor-method integrator for vector ODEs dy/dt = f(t, y).
 *
 * Requires Eigen. Include this header directly; it is not part of the
 * default tax.hpp umbrella.
 *
 * Usage example:
 * @code
 *   #include <tax/ode/taylor.hpp>
 *   using namespace tax;
 *   using namespace tax::ode;
 *
 *   // Harmonic oscillator: y' = [y1; -y0]
 *   auto rhs = [](auto t, auto y) { return decltype(y){ y(1), -y(0) }; };
 *
 *   Eigen::Vector2d y0{1.0, 0.0};
 *   auto result = taylorIntegrate<10>(rhs, 0.0, 1.0, y0, 0.1);
 * @endcode
 */

#include <Eigen/Core>
#include <tax/tax.hpp>
#include <tax/eigen/num_traits.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace tax::ode
{

// ---------------------------------------------------------------------------
// Helper alias
// ---------------------------------------------------------------------------

/**
 * @brief DA column-vector matching the shape of an Eigen vector type.
 *
 * For @p Vec = `Eigen::Matrix<Scalar, Rows, 1>`, produces
 * `Eigen::Matrix<DA<N>, Rows, 1>`.
 */
template < int N, typename Vec >
using DAVec = Eigen::Matrix< DA< N >, Vec::RowsAtCompileTime, 1 >;

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/**
 * @brief Trajectory returned by taylorIntegrate.
 *
 * @c t[i] and @c y[i] are the time and state at the i-th accepted step,
 * starting from the initial condition (i = 0).
 */
struct IntegrationResult
{
    std::vector< double >           t;  ///< Time values.
    std::vector< Eigen::VectorXd >  y;  ///< State snapshots (VectorXd).
};

// ---------------------------------------------------------------------------
// taylorStep
// ---------------------------------------------------------------------------

/**
 * @brief Single N-th order Taylor step for the vector ODE dy/dt = f(t, y).
 *
 * Advances the state from @p t0 to `t0 + h` using a Taylor series of order N
 * in the time displacement δ = t − t₀.
 *
 * @tparam N   Taylor series order (compile-time).
 * @tparam Vec Eigen column-vector type (fixed or dynamic rows; any scalar).
 * @tparam F   Callable with signature
 *             `DAVec<N,Vec> f(DA<N> t, DAVec<N,Vec> y)`.
 *             A generic lambda `[](auto t, auto y){ … }` works for any N/Vec.
 *
 * @param rhs Right-hand side of the ODE.
 * @param t0  Current time.
 * @param y0  Current state.
 * @param h   Step size.
 * @return    State at time `t0 + h`.
 *
 * @note The step size @p h is fixed; use taylorIntegrate for adaptive control.
 */
template < int N, typename Vec, typename F >
Vec taylorStep( F&& rhs, double t0, const Vec& y0, double h )
{
    using DT   = DA< N >;
    using DVec = DAVec< N, Vec >;

    const Eigen::Index dim = y0.size();

    // t_da = t₀ + δ  (DA variable centred at t0)
    DT t_da = DT::variable( t0 );

    // y_da: each component starts as the constant y0[i]
    DVec y_da( dim );
    for ( Eigen::Index i = 0; i < dim; ++i )
        y_da( i ) = DT( double( y0( i ) ) );

    // Iteratively compute Taylor coefficients of y
    //
    // Causality: the k-th output coefficient of f depends only on input
    // coefficients 0..k, so filling y_da one order at a time is exact.
    //
    // ODE recurrence: c_{k+1}(y_i) = c_k(f_i) / (k+1)
    for ( int k = 0; k < N; ++k )
    {
        DVec f_da = rhs( t_da, y_da );
        for ( Eigen::Index i = 0; i < dim; ++i )
            y_da( i )[k + 1] = f_da( i )[k] / double( k + 1 );
    }

    // Evaluate each Taylor polynomial at displacement h (Horner's method)
    Vec y_new( dim );
    for ( Eigen::Index i = 0; i < dim; ++i )
        y_new( i ) = y_da( i ).eval( h );

    return y_new;
}

// ---------------------------------------------------------------------------
// taylorIntegrate
// ---------------------------------------------------------------------------

/**
 * @brief Integrate dy/dt = f(t, y) from @p t0 to @p tf with adaptive steps.
 *
 * Uses the Jorba–Zou step-size formula: after computing Taylor coefficients
 * c[0..N] for each state component, the next step size is
 * @verbatim
 *   h_opt = 0.9 * min( (atol/|c[N]|)^(1/N),  (atol/|c[N-1]|)^(1/(N-1)) )
 * @endverbatim
 * where the norm is taken as the max over all state components.
 *
 * @tparam N   Taylor series order (compile-time). Higher N allows larger steps.
 * @tparam Vec Eigen column-vector type for the state.
 * @tparam F   RHS callable (see taylorStep).
 *
 * @param rhs  Right-hand side of the ODE.
 * @param t0   Initial time.
 * @param tf   Final time.
 * @param y0   Initial state.
 * @param h0   Initial step size guess.
 * @param atol Absolute tolerance used for step-size control.
 * @param rtol Relative tolerance (currently blended into atol as a floor).
 * @return     IntegrationResult containing all (t, y) pairs.
 */
template < int N, typename Vec, typename F >
IntegrationResult taylorIntegrate(
    F&&          rhs,
    double       t0,
    double       tf,
    const Vec&   y0,
    double       h0,
    double       atol = 1e-8,
    double       rtol = 1e-8 )
{
    static_assert( N >= 2, "Taylor order N must be at least 2 for adaptive control." );

    using DT   = DA< N >;
    using DVec = DAVec< N, Vec >;

    IntegrationResult result;
    result.t.push_back( t0 );
    result.y.push_back( y0.template cast< double >() );

    const Eigen::Index dim = y0.size();
    Vec    y = y0;
    double t = t0;
    double h = std::min( h0, tf - t0 );

    while ( t < tf - 1e-14 * std::abs( tf ) )
    {
        h = std::min( h, tf - t );
        if ( h <= 0.0 ) break;

        // ── Build Taylor polynomial ──────────────────────────────────────
        DT   t_da = DT::variable( t );
        DVec y_da( dim );
        for ( Eigen::Index i = 0; i < dim; ++i )
            y_da( i ) = DT( double( y( i ) ) );

        for ( int k = 0; k < N; ++k )
        {
            DVec f_da = rhs( t_da, y_da );
            for ( Eigen::Index i = 0; i < dim; ++i )
                y_da( i )[k + 1] = f_da( i )[k] / double( k + 1 );
        }

        // ── Jorba–Zou optimal step for the NEXT iteration ───────────────
        // Use the local tolerance as the mix of atol and rtol * |y|.
        double y_norm = 0.0;
        for ( Eigen::Index i = 0; i < dim; ++i )
            y_norm = std::max( y_norm, std::abs( double( y( i ) ) ) );
        const double tol = atol + rtol * y_norm;

        double cn  = 0.0, cn1 = 0.0;
        for ( Eigen::Index i = 0; i < dim; ++i )
        {
            cn  = std::max( cn,  std::abs( y_da( i )[N] ) );
            cn1 = std::max( cn1, std::abs( y_da( i )[N - 1] ) );
        }

        double h_opt = h * 4.0;  // default: allow growth
        if ( cn  > 0.0 ) h_opt = std::min( h_opt, std::pow( tol / cn,  1.0 / N ) );
        if ( cn1 > 0.0 ) h_opt = std::min( h_opt, std::pow( tol / cn1, 1.0 / ( N - 1 ) ) );
        h_opt *= 0.9;

        // ── Advance state ────────────────────────────────────────────────
        Vec y_new( dim );
        for ( Eigen::Index i = 0; i < dim; ++i )
            y_new( i ) = y_da( i ).eval( h );

        t += h;
        y  = y_new;
        result.t.push_back( t );
        result.y.push_back( y_new.template cast< double >() );

        // Prepare next step (clamp to remaining interval)
        h = ( tf - t > 0.0 ) ? std::min( h_opt, tf - t ) : h_opt;
    }

    return result;
}

}  // namespace tax::ode
