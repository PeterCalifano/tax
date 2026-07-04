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

/// Compute `out = x^n` for an integer exponent via binary exponentiation.
template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > pow( const TaylorExpansion< T, Scheme >& x,
                                                          int n ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    detail::kernels::seriesPowInt< T, Scheme >( r.coefficients(), x.coefficients(), n );
    return r;
}

/// Real-exponent power `out = x^p`. Requires `x.value() != 0`.
template < typename T, IndexScheme Scheme, std::floating_point P >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > pow( const TaylorExpansion< T, Scheme >& x,
                                                          P p ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    detail::kernels::seriesPow< T, Scheme >( r.coefficients(), x.coefficients(), T( p ) );
    return r;
}

/// Compile-time half-integer power `out = x^(K/2)`.
///
/// Even `K` dispatches to the integer-power chain (requires `x.value() != 0`
/// for negative K); odd `K` runs the single real-exponent recurrence
/// (requires `x.value() > 0`). Per the expression-template prototype's own
/// benchmarks this is the fastest spelling — one seriesPow pass beats the
/// fused sqrt/invsqrt pair + power chain whenever only one output is
/// consumed (a caller needing sqrt(x) alongside x^(-K/2) should combine
/// sqrtInvSqrt with pow instead).
template < int K, typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > halfPow(
    const TaylorExpansion< T, Scheme >& x ) noexcept
{
    if constexpr ( K % 2 == 0 )
    {
        return pow( x, K / 2 );
    } else
    {
        TaylorExpansion< T, Scheme > r;
        detail::kernels::seriesPow< T, Scheme >( r.coefficients(), x.coefficients(),
                                                 T( K ) / T( 2 ) );
        return r;
    }
}

/// Compile-time inverse square-root power `out = x^(-K/2) = 1/sqrt(x)^K`
/// (K >= 1). `invSqrtPow<3>(r2)` is the classic 1/r^3 of a squared radius.
/// Requires `x.value() > 0`.
template < int K, typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > invSqrtPow(
    const TaylorExpansion< T, Scheme >& x ) noexcept
{
    static_assert( K >= 1, "invSqrtPow<K>: K must be >= 1 (computes x^(-K/2))" );
    return halfPow< -K >( x );
}

/// Taylor-valued exponent `out = a^b = exp(b*log(a))`. Requires `a.value() > 0`.
template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > pow(
    const TaylorExpansion< T, Scheme >& a, const TaylorExpansion< T, Scheme >& b ) noexcept
{
    return exp( b * log( a ) );
}

/// Scalar base, Taylor exponent `out = s^b = exp(b*log(s))`. Requires `s > 0`.
template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > pow(
    std::type_identity_t< T > s, const TaylorExpansion< T, Scheme >& b ) noexcept
{
    return exp( b * detail::cmath::ctLog( s ) );
}

/// Compute `out = atan2(y, x)` using the two-argument arctangent series kernel.
template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > atan2(
    const TaylorExpansion< T, Scheme >& y, const TaylorExpansion< T, Scheme >& x ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    detail::kernels::seriesAtan2< T, Scheme >( r.coefficients(), y.coefficients(),
                                               x.coefficients() );
    return r;
}

/// `atan2(y, x)` with a constant `x` (promoted to a flat expansion).
template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > atan2( const TaylorExpansion< T, Scheme >& y,
                                                            std::type_identity_t< T > x ) noexcept
{
    return atan2( y, TaylorExpansion< T, Scheme >{ x } );
}

/// `atan2(y, x)` with a constant `y` (promoted to a flat expansion).
template < typename T, IndexScheme Scheme >
[[nodiscard]] constexpr TaylorExpansion< T, Scheme > atan2(
    std::type_identity_t< T > y, const TaylorExpansion< T, Scheme >& x ) noexcept
{
    return atan2( TaylorExpansion< T, Scheme >{ y }, x );
}

/// Sparse `f^n` via binary exponentiation of the Cauchy product.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > pow(
    const TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse >& x, int n )
{
    TaylorExpansion< T, IsotropicScheme< N, M >, storage::Sparse > r;
    detail::kernels::seriesPowIntSparse< T, N, M >( r.container(), x.container(), n );
    return r;
}

}  // namespace tax
