#pragma once

#include <cmath>
#include <concepts>
#include <tax/core/taylor_expansion.hpp>
#include <tax/kernels/algebra.hpp>
#include <tax/kernels/sparse_subs.hpp>
#include <tax/kernels/trigonometric.hpp>
#include <tax/operators/math_unary.hpp>
#include <type_traits>

namespace tax
{

/**
 * @brief Compute `out = x^n` for an integer exponent via binary exponentiation.
 *
 * Handles n == 0 (returns 1), n == 1 (returns x), n < 0 (reciprocal then pow),
 * and n >= 2 (binary square-and-multiply).
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > pow( const TaylorExpansion< T, N, M >& x,
                                                        int n ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesPowInt< T, N, M >( r.coefficients(), x.coefficients(), n );
    return r;
}

/**
 * @brief Compute `out = x^p` for a real exponent via logarithmic-differentiation
 *        recurrence.
 *
 * NOT constexpr because `std::pow(a[0], p)` is not constexpr for floating-point
 * types in C++23. Requires `x.value() != 0`.
 *
 * The exponent is constrained to a floating-point type: this disambiguates calls
 * such as `pow(x, 2.0f)` (with `T == double`), which would otherwise be ambiguous
 * between this overload and the integer one — a floating-point argument now binds
 * here by exact match rather than tying with `float -> int`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M, std::floating_point P >
[[nodiscard]] TaylorExpansion< T, N, M > pow( const TaylorExpansion< T, N, M >& x, P p ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesPow< T, N, M >( r.coefficients(), x.coefficients(), T( p ) );
    return r;
}

/**
 * @brief Compute `out = a^b` for a Taylor-valued exponent: `a^b = exp(b * log(a))`.
 *
 * Requires `a.value() > 0`. NOT constexpr (relies on `log`/`exp`).
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M > pow( const TaylorExpansion< T, N, M >& a,
                                              const TaylorExpansion< T, N, M >& b ) noexcept
{
    return exp( b * log( a ) );
}

/**
 * @brief Compute `out = s^b` for a positive scalar base and Taylor exponent:
 *        `s^b = exp(b * log(s))`.
 *
 * Requires `s > 0`. NOT constexpr (relies on `exp`).
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M > pow( std::type_identity_t< T > s,
                                              const TaylorExpansion< T, N, M >& b ) noexcept
{
    using std::log;
    return exp( b * log( s ) );
}

/**
 * @brief Compute `out = atan2(y, x)` using the two-argument arctangent series kernel.
 *
 * The constant term is resolved by `std::atan2(y[0], x[0])`, giving the correct
 * quadrant. Higher-order coefficients follow the `atan(y/x)` recurrence.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M > atan2( const TaylorExpansion< T, N, M >& y,
                                                const TaylorExpansion< T, N, M >& x ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesAtan2< T, N, M >( r.coefficients(), y.coefficients(), x.coefficients() );
    return r;
}

/// @brief `atan2(y, x)` with a constant `x` (promoted to a flat expansion).
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M > atan2( const TaylorExpansion< T, N, M >& y,
                                                std::type_identity_t< T > x ) noexcept
{
    return atan2( y, TaylorExpansion< T, N, M >{ x } );
}

/// @brief `atan2(y, x)` with a constant `y` (promoted to a flat expansion).
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M > atan2( std::type_identity_t< T > y,
                                                const TaylorExpansion< T, N, M >& x ) noexcept
{
    return atan2( TaylorExpansion< T, N, M >{ y }, x );
}

/**
 * @brief Sparse `f^n` via binary exponentiation of the Cauchy product.
 *
 * Negative `n` throws `std::invalid_argument`; `n == 0` returns `1`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, storage::Sparse > pow(
    const TaylorExpansion< T, N, M, storage::Sparse >& x, int n )
{
    TaylorExpansion< T, N, M, storage::Sparse > r;
    detail::kernels::seriesPowIntSparse< T, N, M >( r.container(), x.container(), n );
    return r;
}

}  // namespace tax
