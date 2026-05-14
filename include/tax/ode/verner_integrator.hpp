#pragma once

/**
 * @file
 * @brief Adaptive Verner Runge–Kutta integrators (orders 7/8 and 8/9).
 *
 * Two adaptive explicit RK integrators parametrised on a generic @c State type:
 *
 *  - @ref tax::ode::Verner78 — Verner 8(7) "efficient" 13-stage pair
 *  - @ref tax::ode::Verner89 — Verner 9(8) "efficient" 16-stage pair
 *
 * Both propagate at the higher order and use the embedded estimator for
 * adaptive step-size control.  Because the methods touch the state only through
 * `State + scalar*State`, the integrators accept any State that supports those
 * operations — including DA (`tax::TEn<P, M>`) and vectors of DA
 * (`Eigen::Matrix<tax::TEn<P,M>, D, 1>`), which lets the same code propagate
 * series expansions and ADS-style polynomial flow maps.
 *
 * Right-hand side:
 *
 * @code
 *   State f(const State& x, T t);
 * @endcode
 *
 * Returned by value uniformly across scalar, vector, and DA-valued states.
 *
 * Supported `State` types out of the box (via the in-namespace `verner_axpy`
 * / `verner_norm` / `verner_scale_assign` overload set in
 * `tax::ode::detail`):
 *  - Floating-point scalars (`double`, `float`).
 *  - `tax::TaylorExpansionT<T, N, M>` (DA scalar).
 *  - `Eigen::Matrix<T, D, 1>` where `T` is itself a supported state-element type.
 *
 * For custom state types, overload `tax::ode::detail::verner_axpy`,
 * `tax::ode::detail::verner_scale_assign`, and `tax::ode::detail::verner_norm`
 * in the same namespace.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <tax/ode/verner_tableaus.hpp>
#include <tax/storage/tte_static.hpp>

namespace tax::ode
{

// =============================================================================
// Configuration
// =============================================================================

/**
 * @brief Configuration for the adaptive Verner integrators.
 *
 * Step-size control compares an embedded error estimate against
 *   sc = abstol + reltol * max(||x||_inf, ||x_new||_inf)
 * (Hairer-Norsett-Wanner style).  A step is accepted when ||err||_inf <= sc.
 */
struct VernerConfig
{
    double abstol     = 1e-10;  ///< Absolute tolerance.
    double reltol     = 0.0;    ///< Relative tolerance (0 ⇒ pure abstol).
    double init_step  = 0.0;    ///< Initial step (0 ⇒ heuristic).
    double min_step   = 0.0;    ///< Minimum step (always > 0 internally).
    double max_step   = std::numeric_limits< double >::infinity();
    int    max_steps  = 10000;  ///< Maximum number of accepted steps.
    double safety     = 0.9;    ///< Step-size safety factor.
    double min_factor = 0.2;    ///< Maximum step shrink per attempt.
    double max_factor = 5.0;    ///< Maximum step growth per accepted step.
};

namespace detail
{

inline void validate( const VernerConfig& cfg )
{
    if ( !( cfg.abstol > 0.0 ) )
        throw std::invalid_argument( "VernerConfig: abstol must be > 0" );
    if ( cfg.reltol < 0.0 )
        throw std::invalid_argument( "VernerConfig: reltol must be >= 0" );
    if ( !( cfg.max_step > 0.0 ) )
        throw std::invalid_argument( "VernerConfig: max_step must be > 0" );
    if ( cfg.max_steps <= 0 )
        throw std::invalid_argument( "VernerConfig: max_steps must be > 0" );
    if ( !( cfg.safety > 0.0 && cfg.safety < 1.0 ) )
        throw std::invalid_argument( "VernerConfig: safety must be in (0, 1)" );
    if ( !( cfg.min_factor > 0.0 && cfg.min_factor < 1.0 ) )
        throw std::invalid_argument( "VernerConfig: min_factor must be in (0, 1)" );
    if ( !( cfg.max_factor > 1.0 ) )
        throw std::invalid_argument( "VernerConfig: max_factor must be > 1" );
}

// =============================================================================
// State-type customisation points
// =============================================================================
//
// Three operations parameterise the integrator over State:
//   verner_norm(x)               → infinity norm as double
//   verner_axpy(y, alpha, x)     → y += alpha * x
//   verner_scale_assign(y, a, x) → y = a * x
//
// Default overloads handle:
//   - floating-point scalars
//   - tax::TaylorExpansionT<T, N, M>
//   - Eigen::Matrix<T, D, 1> for any T supported above
//
// Users can extend by adding overloads in this namespace for custom State types.

// -- norm -------------------------------------------------------------------

template < typename T >
    requires std::is_floating_point_v< T >
[[nodiscard]] inline double verner_norm( const T& x ) noexcept
{
    return std::abs( static_cast< double >( x ) );
}

template < typename T, int N, int M >
[[nodiscard]] inline double verner_norm(
    const TaylorExpansionT< T, N, M >& x ) noexcept
{
    double n = 0.0;
    for ( std::size_t i = 0; i < TaylorExpansionT< T, N, M >::nCoefficients; ++i )
        n = std::max( n, std::abs( static_cast< double >( x[i] ) ) );
    return n;
}

template < typename T, int D >
[[nodiscard]] inline double verner_norm( const Eigen::Matrix< T, D, 1 >& x ) noexcept
{
    double n = 0.0;
    for ( Eigen::Index i = 0; i < x.size(); ++i ) n = std::max( n, verner_norm( x( i ) ) );
    return n;
}

// -- axpy: y += alpha * x ---------------------------------------------------

template < typename T >
    requires std::is_floating_point_v< T >
inline void verner_axpy( T& y, double alpha, const T& x ) noexcept
{
    y += static_cast< T >( alpha ) * x;
}

template < typename T, int N, int M >
inline void verner_axpy( TaylorExpansionT< T, N, M >& y, double alpha,
                          const TaylorExpansionT< T, N, M >& x ) noexcept
{
    y = y + static_cast< T >( alpha ) * x;
}

template < typename T, int D >
inline void verner_axpy( Eigen::Matrix< T, D, 1 >& y, double alpha,
                          const Eigen::Matrix< T, D, 1 >& x ) noexcept
{
    for ( Eigen::Index i = 0; i < y.size(); ++i ) verner_axpy( y( i ), alpha, x( i ) );
}

// -- scale_assign: y = alpha * x --------------------------------------------

template < typename T >
    requires std::is_floating_point_v< T >
inline void verner_scale_assign( T& y, double alpha, const T& x ) noexcept
{
    y = static_cast< T >( alpha ) * x;
}

template < typename T, int N, int M >
inline void verner_scale_assign( TaylorExpansionT< T, N, M >& y, double alpha,
                                   const TaylorExpansionT< T, N, M >& x ) noexcept
{
    y = static_cast< T >( alpha ) * x;
}

template < typename T, int D >
inline void verner_scale_assign( Eigen::Matrix< T, D, 1 >& y, double alpha,
                                   const Eigen::Matrix< T, D, 1 >& x ) noexcept
{
    if ( y.size() != x.size() ) y.resize( x.size() );
    for ( Eigen::Index i = 0; i < x.size(); ++i )
        verner_scale_assign( y( i ), alpha, x( i ) );
}

// =============================================================================
// Single-step kernels — one per tableau
// =============================================================================

/**
 * @brief Result of a single Verner step (before accept/reject).
 * @tparam State User state type.
 */
template < typename State >
struct VernerStepResult
{
    State  x_new;       ///< Tentative state after the step (order-p propagation).
    double err_norm;    ///< Infinity norm of the embedded error estimate.
};

/// @brief One adaptive step of Verner 8(7).
template < typename F, typename State >
[[nodiscard]] inline VernerStepResult< State > verner78_step( const F& f, const State& x,
                                                                double t, double h )
{
    using C = Verner78Coeffs;

    State tmp{};

    // Stage 1
    auto k1 = f( x, t );

    // Stage 2
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a0201, k1 );
    auto k2 = f( tmp, t + C::c2 * h );

    // Stage 3
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a0301, k1 );
    verner_axpy( tmp, h * C::a0302, k2 );
    auto k3 = f( tmp, t + C::c3 * h );

    // Stage 4
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a0401, k1 );
    verner_axpy( tmp, h * C::a0403, k3 );
    auto k4 = f( tmp, t + C::c4 * h );

    // Stage 5
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a0501, k1 );
    verner_axpy( tmp, h * C::a0503, k3 );
    verner_axpy( tmp, h * C::a0504, k4 );
    auto k5 = f( tmp, t + C::c5 * h );

    // Stage 6
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a0601, k1 );
    verner_axpy( tmp, h * C::a0604, k4 );
    verner_axpy( tmp, h * C::a0605, k5 );
    auto k6 = f( tmp, t + C::c6 * h );

    // Stage 7
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a0701, k1 );
    verner_axpy( tmp, h * C::a0704, k4 );
    verner_axpy( tmp, h * C::a0705, k5 );
    verner_axpy( tmp, h * C::a0706, k6 );
    auto k7 = f( tmp, t + C::c7 * h );

    // Stage 8
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a0801, k1 );
    verner_axpy( tmp, h * C::a0804, k4 );
    verner_axpy( tmp, h * C::a0805, k5 );
    verner_axpy( tmp, h * C::a0806, k6 );
    verner_axpy( tmp, h * C::a0807, k7 );
    auto k8 = f( tmp, t + C::c8 * h );

    // Stage 9
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a0901, k1 );
    verner_axpy( tmp, h * C::a0904, k4 );
    verner_axpy( tmp, h * C::a0905, k5 );
    verner_axpy( tmp, h * C::a0906, k6 );
    verner_axpy( tmp, h * C::a0907, k7 );
    verner_axpy( tmp, h * C::a0908, k8 );
    auto k9 = f( tmp, t + C::c9 * h );

    // Stage 10
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a1001, k1 );
    verner_axpy( tmp, h * C::a1004, k4 );
    verner_axpy( tmp, h * C::a1005, k5 );
    verner_axpy( tmp, h * C::a1006, k6 );
    verner_axpy( tmp, h * C::a1007, k7 );
    verner_axpy( tmp, h * C::a1008, k8 );
    verner_axpy( tmp, h * C::a1009, k9 );
    auto k10 = f( tmp, t + C::c10 * h );

    // Stage 11
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a1101, k1 );
    verner_axpy( tmp, h * C::a1104, k4 );
    verner_axpy( tmp, h * C::a1105, k5 );
    verner_axpy( tmp, h * C::a1106, k6 );
    verner_axpy( tmp, h * C::a1107, k7 );
    verner_axpy( tmp, h * C::a1108, k8 );
    verner_axpy( tmp, h * C::a1109, k9 );
    verner_axpy( tmp, h * C::a1110, k10 );
    auto k11 = f( tmp, t + C::c11 * h );

    // Stage 12
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a1201, k1 );
    verner_axpy( tmp, h * C::a1204, k4 );
    verner_axpy( tmp, h * C::a1205, k5 );
    verner_axpy( tmp, h * C::a1206, k6 );
    verner_axpy( tmp, h * C::a1207, k7 );
    verner_axpy( tmp, h * C::a1208, k8 );
    verner_axpy( tmp, h * C::a1209, k9 );
    verner_axpy( tmp, h * C::a1210, k10 );
    verner_axpy( tmp, h * C::a1211, k11 );
    auto k12 = f( tmp, t + C::c12 * h );

    // Stage 13 (error-only)
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a1301, k1 );
    verner_axpy( tmp, h * C::a1304, k4 );
    verner_axpy( tmp, h * C::a1305, k5 );
    verner_axpy( tmp, h * C::a1306, k6 );
    verner_axpy( tmp, h * C::a1307, k7 );
    verner_axpy( tmp, h * C::a1308, k8 );
    verner_axpy( tmp, h * C::a1309, k9 );
    verner_axpy( tmp, h * C::a1310, k10 );
    auto k13 = f( tmp, t + C::c13 * h );

    // Order-8 propagation: x_new = x + h * (b1*k1 + b6*k6 + ... + b12*k12).
    State x_new{};
    verner_scale_assign( x_new, 1.0, x );
    verner_axpy( x_new, h * C::b1,  k1 );
    verner_axpy( x_new, h * C::b6,  k6 );
    verner_axpy( x_new, h * C::b7,  k7 );
    verner_axpy( x_new, h * C::b8,  k8 );
    verner_axpy( x_new, h * C::b9,  k9 );
    verner_axpy( x_new, h * C::b10, k10 );
    verner_axpy( x_new, h * C::b11, k11 );
    verner_axpy( x_new, h * C::b12, k12 );

    // Embedded order-7 error: err = h * (e1*k1 + e6*k6 + ... + e13*k13).
    State err{};
    verner_scale_assign( err, h * C::e1, k1 );
    verner_axpy( err, h * C::e6,  k6 );
    verner_axpy( err, h * C::e7,  k7 );
    verner_axpy( err, h * C::e8,  k8 );
    verner_axpy( err, h * C::e9,  k9 );
    verner_axpy( err, h * C::e10, k10 );
    verner_axpy( err, h * C::e11, k11 );
    verner_axpy( err, h * C::e12, k12 );
    verner_axpy( err, h * C::e13, k13 );

    return { std::move( x_new ), verner_norm( err ) };
}

/// @brief One adaptive step of Verner 9(8).
template < typename F, typename State >
[[nodiscard]] inline VernerStepResult< State > verner89_step( const F& f, const State& x,
                                                                double t, double h )
{
    using C = Verner89Coeffs;

    State tmp{};

    // Stage 1
    auto k1 = f( x, t );

    // Stage 2
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a0201, k1 );
    auto k2 = f( tmp, t + C::c2 * h );

    // Stage 3
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a0301, k1 );
    verner_axpy( tmp, h * C::a0302, k2 );
    auto k3 = f( tmp, t + C::c3 * h );

    // Stage 4
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a0401, k1 );
    verner_axpy( tmp, h * C::a0403, k3 );
    auto k4 = f( tmp, t + C::c4 * h );

    // Stage 5
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a0501, k1 );
    verner_axpy( tmp, h * C::a0503, k3 );
    verner_axpy( tmp, h * C::a0504, k4 );
    auto k5 = f( tmp, t + C::c5 * h );

    // Stage 6
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a0601, k1 );
    verner_axpy( tmp, h * C::a0604, k4 );
    verner_axpy( tmp, h * C::a0605, k5 );
    auto k6 = f( tmp, t + C::c6 * h );

    // Stage 7
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a0701, k1 );
    verner_axpy( tmp, h * C::a0704, k4 );
    verner_axpy( tmp, h * C::a0705, k5 );
    verner_axpy( tmp, h * C::a0706, k6 );
    auto k7 = f( tmp, t + C::c7 * h );

    // Stage 8
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a0801, k1 );
    verner_axpy( tmp, h * C::a0806, k6 );
    verner_axpy( tmp, h * C::a0807, k7 );
    auto k8 = f( tmp, t + C::c8 * h );

    // Stage 9
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a0901, k1 );
    verner_axpy( tmp, h * C::a0906, k6 );
    verner_axpy( tmp, h * C::a0907, k7 );
    verner_axpy( tmp, h * C::a0908, k8 );
    auto k9 = f( tmp, t + C::c9 * h );

    // Stage 10
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a1001, k1 );
    verner_axpy( tmp, h * C::a1006, k6 );
    verner_axpy( tmp, h * C::a1007, k7 );
    verner_axpy( tmp, h * C::a1008, k8 );
    verner_axpy( tmp, h * C::a1009, k9 );
    auto k10 = f( tmp, t + C::c10 * h );

    // Stage 11
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a1101, k1 );
    verner_axpy( tmp, h * C::a1106, k6 );
    verner_axpy( tmp, h * C::a1107, k7 );
    verner_axpy( tmp, h * C::a1108, k8 );
    verner_axpy( tmp, h * C::a1109, k9 );
    verner_axpy( tmp, h * C::a1110, k10 );
    auto k11 = f( tmp, t + C::c11 * h );

    // Stage 12
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a1201, k1 );
    verner_axpy( tmp, h * C::a1206, k6 );
    verner_axpy( tmp, h * C::a1207, k7 );
    verner_axpy( tmp, h * C::a1208, k8 );
    verner_axpy( tmp, h * C::a1209, k9 );
    verner_axpy( tmp, h * C::a1210, k10 );
    verner_axpy( tmp, h * C::a1211, k11 );
    auto k12 = f( tmp, t + C::c12 * h );

    // Stage 13
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a1301, k1 );
    verner_axpy( tmp, h * C::a1306, k6 );
    verner_axpy( tmp, h * C::a1307, k7 );
    verner_axpy( tmp, h * C::a1308, k8 );
    verner_axpy( tmp, h * C::a1309, k9 );
    verner_axpy( tmp, h * C::a1310, k10 );
    verner_axpy( tmp, h * C::a1311, k11 );
    verner_axpy( tmp, h * C::a1312, k12 );
    auto k13 = f( tmp, t + C::c13 * h );

    // Stage 14
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a1401, k1 );
    verner_axpy( tmp, h * C::a1406, k6 );
    verner_axpy( tmp, h * C::a1407, k7 );
    verner_axpy( tmp, h * C::a1408, k8 );
    verner_axpy( tmp, h * C::a1409, k9 );
    verner_axpy( tmp, h * C::a1410, k10 );
    verner_axpy( tmp, h * C::a1411, k11 );
    verner_axpy( tmp, h * C::a1412, k12 );
    verner_axpy( tmp, h * C::a1413, k13 );
    auto k14 = f( tmp, t + C::c14 * h );

    // Stage 15
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a1501, k1 );
    verner_axpy( tmp, h * C::a1506, k6 );
    verner_axpy( tmp, h * C::a1507, k7 );
    verner_axpy( tmp, h * C::a1508, k8 );
    verner_axpy( tmp, h * C::a1509, k9 );
    verner_axpy( tmp, h * C::a1510, k10 );
    verner_axpy( tmp, h * C::a1511, k11 );
    verner_axpy( tmp, h * C::a1512, k12 );
    verner_axpy( tmp, h * C::a1513, k13 );
    verner_axpy( tmp, h * C::a1514, k14 );
    auto k15 = f( tmp, t + C::c15 * h );

    // Stage 16 (error-only)
    verner_scale_assign( tmp, 1.0, x );
    verner_axpy( tmp, h * C::a1601, k1 );
    verner_axpy( tmp, h * C::a1606, k6 );
    verner_axpy( tmp, h * C::a1607, k7 );
    verner_axpy( tmp, h * C::a1608, k8 );
    verner_axpy( tmp, h * C::a1609, k9 );
    verner_axpy( tmp, h * C::a1610, k10 );
    verner_axpy( tmp, h * C::a1611, k11 );
    verner_axpy( tmp, h * C::a1612, k12 );
    verner_axpy( tmp, h * C::a1613, k13 );
    auto k16 = f( tmp, t + C::c16 * h );

    // Order-9 propagation: stages 1, 8-15 contribute.
    State x_new{};
    verner_scale_assign( x_new, 1.0, x );
    verner_axpy( x_new, h * C::b1,  k1 );
    verner_axpy( x_new, h * C::b8,  k8 );
    verner_axpy( x_new, h * C::b9,  k9 );
    verner_axpy( x_new, h * C::b10, k10 );
    verner_axpy( x_new, h * C::b11, k11 );
    verner_axpy( x_new, h * C::b12, k12 );
    verner_axpy( x_new, h * C::b13, k13 );
    verner_axpy( x_new, h * C::b14, k14 );
    verner_axpy( x_new, h * C::b15, k15 );

    // Embedded order-8 error.
    State err{};
    verner_scale_assign( err, h * C::e1, k1 );
    verner_axpy( err, h * C::e8,  k8 );
    verner_axpy( err, h * C::e9,  k9 );
    verner_axpy( err, h * C::e10, k10 );
    verner_axpy( err, h * C::e11, k11 );
    verner_axpy( err, h * C::e12, k12 );
    verner_axpy( err, h * C::e13, k13 );
    verner_axpy( err, h * C::e14, k14 );
    verner_axpy( err, h * C::e15, k15 );
    verner_axpy( err, h * C::e16, k16 );

    return { std::move( x_new ), verner_norm( err ) };
}

}  // namespace detail

// =============================================================================
// Solution container
// =============================================================================

/**
 * @brief Step-by-step solution returned by the Verner integrators.
 *
 * Unlike @ref TaylorSolution, no per-step polynomial is stored — Verner RK
 * pairs do not provide a time-Taylor expansion of the step, so only the
 * accepted (t, x) samples are recorded.
 */
template < typename State, typename T = double >
struct VernerSolution
{
    std::vector< T >     t;             ///< Accepted step times.
    std::vector< State > x;             ///< State at each accepted step time.
    int                  n_accepted = 0;
    int                  n_rejected = 0;
};

// =============================================================================
// Adaptive integrator class
// =============================================================================

/**
 * @brief Adaptive Verner explicit Runge–Kutta integrator.
 *
 * @tparam Coeffs One of `tax::ode::detail::Verner78Coeffs` or
 *                `tax::ode::detail::Verner89Coeffs`.
 * @tparam State  User state type — any type supporting the
 *                `verner_axpy`/`verner_norm`/`verner_scale_assign` triple.
 *                Default overloads handle scalars, `TaylorExpansionT`, and
 *                `Eigen::Matrix<...,1>` of either.
 * @tparam T      Independent-variable (time) scalar type.
 *
 * Prefer the @ref Verner78 and @ref Verner89 aliases below.
 */
template < typename Coeffs, typename State, typename T = double >
class VernerIntegrator
{
public:
    using Rhs      = std::function< State( const State&, T ) >;
    using Config   = VernerConfig;
    using Solution = VernerSolution< State, T >;

    static constexpr int order     = Coeffs::order;
    static constexpr int err_order = Coeffs::err_order;

    explicit VernerIntegrator( Rhs f, Config cfg = {} )
        : f_( std::move( f ) ), cfg_( cfg )
    {
        detail::validate( cfg_ );
    }

    [[nodiscard]] const Config& config() const noexcept { return cfg_; }

    /// @brief Integrate `dx/dt = f(x, t)` from @p x0 at @p t0 to @p tmax.
    [[nodiscard]] Solution integrate( State x0, T t0, T tmax ) const
    {
        Solution sol;
        sol.t.reserve( 128 );
        sol.x.reserve( 128 );

        sol.t.push_back( t0 );
        sol.x.push_back( x0 );

        const double sign  = ( tmax >= t0 ) ? 1.0 : -1.0;
        const double total = std::abs( static_cast< double >( tmax - t0 ) );
        if ( total == 0.0 ) return sol;

        double h = cfg_.init_step > 0.0 ? cfg_.init_step
                                         : std::max( 1e-6, 1e-3 * total );
        h = std::min( h, cfg_.max_step );

        double tc = static_cast< double >( t0 );
        State  xc = std::move( x0 );

        for ( int s = 0; s < cfg_.max_steps; ++s )
        {
            const double remaining = static_cast< double >( tmax ) - tc;
            if ( sign * remaining <= 0.0 ) break;

            double h_try = std::min( h, std::abs( remaining ) );
            if ( cfg_.min_step > 0.0 ) h_try = std::max( h_try, cfg_.min_step );
            const double dt_signed = sign * h_try;

            // Take a step (with retry on rejection).
            auto step_result = takeStep( xc, tc, dt_signed );
            const auto& [x_new, err_norm] = step_result;

            const double x_scale = std::max( detail::verner_norm( xc ),
                                              detail::verner_norm( x_new ) );
            const double sc      = cfg_.abstol + cfg_.reltol * x_scale;
            const double ratio   = err_norm / sc;

            // Adaptive PI-free controller (Gustafsson-light).
            // factor = safety * ratio^{-1/(err_order+1)}
            const double exponent = 1.0 / ( static_cast< double >( err_order ) + 1.0 );
            double factor;
            if ( ratio <= 0.0 )
                factor = cfg_.max_factor;
            else
                factor = cfg_.safety * std::pow( 1.0 / ratio, exponent );

            factor = std::clamp( factor, cfg_.min_factor, cfg_.max_factor );

            if ( ratio <= 1.0 )
            {
                // Accept.
                xc = x_new;
                tc += dt_signed;
                sol.t.push_back( static_cast< T >( tc ) );
                sol.x.push_back( xc );
                ++sol.n_accepted;

                h = std::min( h_try * factor, cfg_.max_step );
            }
            else
            {
                // Reject and shrink — do not advance.
                ++sol.n_rejected;
                h = h_try * factor;
                if ( cfg_.min_step > 0.0 && h < cfg_.min_step )
                    throw std::runtime_error(
                        "VernerIntegrator: step underflow (min_step reached)" );
            }
        }

        return sol;
    }

private:
    [[nodiscard]] detail::VernerStepResult< State > takeStep( const State& x, double t,
                                                                double dt ) const
    {
        if constexpr ( std::is_same_v< Coeffs, detail::Verner78Coeffs > )
            return detail::verner78_step< Rhs, State >( f_, x, t, dt );
        else
            return detail::verner89_step< Rhs, State >( f_, x, t, dt );
    }

    Rhs    f_;
    Config cfg_;
};

// =============================================================================
// Convenience aliases
// =============================================================================

/**
 * @brief Verner 8(7) "efficient" 13-stage adaptive RK pair (a.k.a. Verner 7/8).
 *
 * Propagates at order 8 with an embedded order-7 error estimator.
 *
 * @tparam State State type — scalar, Eigen vector, DA scalar, or DA vector.
 * @tparam T     Time scalar type (default `double`).
 */
template < typename State, typename T = double >
using Verner78 = VernerIntegrator< detail::Verner78Coeffs, State, T >;

/**
 * @brief Verner 9(8) "efficient" 16-stage adaptive RK pair (a.k.a. Verner 8/9).
 *
 * Propagates at order 9 with an embedded order-8 error estimator.
 *
 * @tparam State State type — scalar, Eigen vector, DA scalar, or DA vector.
 * @tparam T     Time scalar type (default `double`).
 */
template < typename State, typename T = double >
using Verner89 = VernerIntegrator< detail::Verner89Coeffs, State, T >;

}  // namespace tax::ode
