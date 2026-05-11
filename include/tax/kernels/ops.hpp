#pragma once

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include <tax/utils/combinatorics.hpp>
#include <tax/utils/enumeration.hpp>
#include <tax/utils/fwd.hpp>

namespace tax::detail
{

template < typename T, std::size_t S >
/// @brief In-place element-wise addition: `o += r`.
constexpr void addInPlace( std::array< T, S >& o, const std::array< T, S >& r ) noexcept
{
    for ( std::size_t i = 0; i < S; ++i ) o[i] += r[i];
}

template < typename T, std::size_t S >
/// @brief In-place element-wise subtraction: `o -= r`.
constexpr void subInPlace( std::array< T, S >& o, const std::array< T, S >& r ) noexcept
{
    for ( std::size_t i = 0; i < S; ++i ) o[i] -= r[i];
}

template < typename T, std::size_t S >
/// @brief In-place sign flip.
constexpr void negateInPlace( std::array< T, S >& o ) noexcept
{
    for ( auto& v : o ) v = -v;
}

template < typename T, std::size_t S >
/// @brief In-place scalar multiply.
constexpr void scaleInPlace( std::array< T, S >& o, T s ) noexcept
{
    for ( auto& v : o ) v *= s;
}

// =============================================================================
// Runtime-shape variants (used by the dynamic-shape `TaylorExpansionT`).
// =============================================================================

/// @brief Runtime overload of `addInPlace`. Caller owns `o.size() == r.size()`.
template < typename T >
inline void addInPlace( T* o, const T* r, std::size_t S ) noexcept
{
    for ( std::size_t i = 0; i < S; ++i ) o[i] += r[i];
}

/// @brief Runtime overload of `subInPlace`.
template < typename T >
inline void subInPlace( T* o, const T* r, std::size_t S ) noexcept
{
    for ( std::size_t i = 0; i < S; ++i ) o[i] -= r[i];
}

/// @brief Runtime overload of `negateInPlace`.
template < typename T >
inline void negateInPlace( T* o, std::size_t S ) noexcept
{
    for ( std::size_t i = 0; i < S; ++i ) o[i] = -o[i];
}

/// @brief Runtime overload of `scaleInPlace`.
template < typename T >
inline void scaleInPlace( T* o, T s, std::size_t S ) noexcept
{
    for ( std::size_t i = 0; i < S; ++i ) o[i] *= s;
}

namespace detail
{

/// @brief Extract the innermost scalar value for sign comparison.
template < typename U >
[[nodiscard]] constexpr auto extractValue( const U& v ) noexcept
{
    if constexpr ( requires { v.value(); } )
        return extractValue( v.value() );
    else
        return v;
}

}  // namespace detail

template < typename T, std::size_t S >
/// @brief Absolute value: `out = |a|`. Requires `a[0] != 0`.
constexpr void seriesAbs( std::array< T, S >& out, const std::array< T, S >& a ) noexcept
{
    out = a;
    if ( detail::extractValue( a[0] ) < 0 ) negateInPlace< T, S >( out );
}

/// @brief Runtime overload of `seriesAbs`. Requires `a[0] != 0`.
template < typename T >
inline void seriesAbs( T* out, const T* a, std::size_t S ) noexcept
{
    for ( std::size_t i = 0; i < S; ++i ) out[i] = a[i];
    if ( detail::extractValue( a[0] ) < 0 ) negateInPlace( out, S );
}

// =============================================================================
// Runtime symbolic calculus helpers used by the dynamic TaylorExpansionT.
// =============================================================================

/**
 * @brief Compute the symbolic partial derivative of a coefficient buffer with
 *        respect to variable `var`. `out` and `in` must both have size
 *        `numMonomials(N, M)`; the result is truncated at the same order N
 *        (no terms are lost since differentiation reduces total degree by 1).
 */
template < typename T >
inline void derivCoeffs( T* out, const T* in, std::size_t var, std::size_t N,
                         std::size_t M )
{
    const std::size_t S = numMonomials( N, M );
    for ( std::size_t i = 0; i < S; ++i ) out[i] = T{ 0 };

    std::vector< int > alpha( M, 0 );
    for ( std::size_t i = 0; i < S; ++i )
    {
        if ( in[i] == T{ 0 } ) continue;
        unflatIndex( i, std::span< int >( alpha.data(), M ) );
        const int exp = alpha[var];
        if ( exp == 0 ) continue;
        alpha[var] = exp - 1;
        const std::size_t out_i =
            flatIndex( std::span< const int >( alpha.data(), M ) );
        out[out_i] += in[i] * T( exp );
        alpha[var] = exp;  // restore for next iteration
    }
}

/**
 * @brief Compute the symbolic partial integral of a coefficient buffer with
 *        respect to variable `var`. Top-degree (|alpha| == N) terms are
 *        truncated since integrating them would exceed order N.
 */
template < typename T >
inline void integCoeffs( T* out, const T* in, std::size_t var, std::size_t N,
                         std::size_t M )
{
    const std::size_t S = numMonomials( N, M );
    for ( std::size_t i = 0; i < S; ++i ) out[i] = T{ 0 };

    std::vector< int > alpha( M, 0 );
    for ( std::size_t i = 0; i < S; ++i )
    {
        if ( in[i] == T{ 0 } ) continue;
        unflatIndex( i, std::span< int >( alpha.data(), M ) );
        if ( static_cast< std::size_t >( totalDegree(
                 std::span< const int >( alpha.data(), M ) ) ) >= N )
            continue;
        const int exp = alpha[var];
        alpha[var] = exp + 1;
        const std::size_t out_i =
            flatIndex( std::span< const int >( alpha.data(), M ) );
        out[out_i] = in[i] / T( exp + 1 );
        alpha[var] = exp;  // restore
    }
}

/// @brief Evaluate a polynomial buffer at a univariate displacement `dx`.
template < typename T >
inline T evalAtScalar( const T* c, T dx, std::size_t N ) noexcept
{
    // Horner's method on the leading run of coefficients (M = 1 layout).
    if ( N == 0 ) return c[0];
    T result = c[N];
    for ( std::size_t i = N; i-- > 0; ) result = result * dx + c[i];
    return result;
}

/// @brief Evaluate a polynomial buffer at a multivariate displacement.
template < typename T >
inline T evalAt( const T* c, std::span< const T > dx, std::size_t N, std::size_t M ) noexcept
{
    if ( M == 1 ) return evalAtScalar( c, dx[0], N );
    T result{};
    for ( int d = 0; d <= int( N ); ++d )
    {
        forEachMonomial( int( M ), d, [&]( std::span< const int > alpha, std::size_t ai ) {
            T monomial{ 1 };
            for ( std::size_t i = 0; i < M; ++i )
                for ( int j = 0; j < alpha[i]; ++j ) monomial *= dx[i];
            result += c[ai] * monomial;
        } );
    }
    return result;
}

/// @brief Factorial of a multi-index: `prod_i alpha[i]!`.
constexpr std::size_t multiIndexFactorial( std::span< const int > alpha ) noexcept
{
    std::size_t fac = 1;
    for ( int a : alpha )
        for ( int j = 1; j <= a; ++j ) fac *= std::size_t( j );
    return fac;
}

/// @brief Numerical partial derivative selected by `alpha` at the expansion point.
/// @return `c[flatIndex(alpha)] * prod_i alpha[i]!`.
template < typename T >
inline T derivativeAt( const T* c, std::span< const int > alpha ) noexcept
{
    return c[flatIndex( alpha )] * T( multiIndexFactorial( alpha ) );
}

/// @brief Fill `out` with `c[i] * (alpha_i)!` for every flat index `i`.
template < typename T >
inline void derivativesAll( T* out, const T* c, std::size_t N, std::size_t M )
{
    const std::size_t S = numMonomials( N, M );
    std::vector< int > alpha( M, 0 );
    for ( std::size_t i = 0; i < S; ++i )
    {
        unflatIndex( i, std::span< int >( alpha.data(), M ) );
        out[i] = c[i] * T( multiIndexFactorial(
                            std::span< const int >( alpha.data(), M ) ) );
    }
}

// =============================================================================
// Coefficient-vector norm helpers (used by the dynamic TaylorExpansionT).
// =============================================================================

template < typename T >
inline T coeffsNormInf( const T* c, std::size_t S ) noexcept
{
    using std::abs;
    T out{};
    for ( std::size_t i = 0; i < S; ++i )
    {
        const T mag = abs( c[i] );
        if ( mag > out ) out = mag;
    }
    return out;
}

template < typename T >
inline T coeffsNormP( const T* c, std::size_t S, unsigned int p )
{
    using std::abs;
    using std::pow;
    if ( p == 1 )
    {
        T accum{};
        for ( std::size_t i = 0; i < S; ++i ) accum += abs( c[i] );
        return accum;
    }
    T accum{};
    for ( std::size_t i = 0; i < S; ++i ) accum += pow( abs( c[i] ), T( p ) );
    return pow( accum, T{ 1 } / T( p ) );
}

}  // namespace tax::detail
