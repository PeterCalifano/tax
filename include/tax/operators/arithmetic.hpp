#pragma once

#include <tax/core/taylor_expansion.hpp>
#include <tax/kernels/algebra.hpp>
#include <tax/kernels/cauchy.hpp>
#include <tax/kernels/sparse_cauchy.hpp>

namespace tax
{

// ---------------------------------------------------------------------------
// Addition
// ---------------------------------------------------------------------------

template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > operator+(
    const TaylorExpansion< T, N, M >& a,
    const TaylorExpansion< T, N, M >& b ) noexcept
{
    TaylorExpansion< T, N, M > r;
    for ( std::size_t k = 0; k < a.nCoefficients; ++k )
        r[k] = a[k] + b[k];
    return r;
}

template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > operator+(
    const TaylorExpansion< T, N, M >& a, T s ) noexcept
{
    TaylorExpansion< T, N, M > r = a;
    r[0] += s;
    return r;
}

template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > operator+(
    T s, const TaylorExpansion< T, N, M >& a ) noexcept
{
    return a + s;
}

// ---------------------------------------------------------------------------
// Subtraction
// ---------------------------------------------------------------------------

template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > operator-(
    const TaylorExpansion< T, N, M >& a,
    const TaylorExpansion< T, N, M >& b ) noexcept
{
    TaylorExpansion< T, N, M > r;
    for ( std::size_t k = 0; k < a.nCoefficients; ++k )
        r[k] = a[k] - b[k];
    return r;
}

template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > operator-(
    const TaylorExpansion< T, N, M >& a, T s ) noexcept
{
    TaylorExpansion< T, N, M > r = a;
    r[0] -= s;
    return r;
}

template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > operator-(
    T s, const TaylorExpansion< T, N, M >& a ) noexcept
{
    TaylorExpansion< T, N, M > r;
    r[0] = s - a[0];
    for ( std::size_t k = 1; k < a.nCoefficients; ++k ) r[k] = -a[k];
    return r;
}

// ---------------------------------------------------------------------------
// Unary negation
// ---------------------------------------------------------------------------

template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > operator-(
    const TaylorExpansion< T, N, M >& a ) noexcept
{
    TaylorExpansion< T, N, M > r;
    for ( std::size_t k = 0; k < a.nCoefficients; ++k )
        r[k] = -a[k];
    return r;
}

// ---------------------------------------------------------------------------
// Scalar multiplication / division
// ---------------------------------------------------------------------------

template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > operator*(
    const TaylorExpansion< T, N, M >& a, T s ) noexcept
{
    TaylorExpansion< T, N, M > r;
    for ( std::size_t k = 0; k < a.nCoefficients; ++k )
        r[k] = a[k] * s;
    return r;
}

template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > operator*(
    T s, const TaylorExpansion< T, N, M >& a ) noexcept
{
    return a * s;
}

template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > operator/(
    const TaylorExpansion< T, N, M >& a, T s ) noexcept
{
    return a * ( T( 1 ) / s );
}

// ---------------------------------------------------------------------------
// Cauchy (TE x TE) multiplication
// ---------------------------------------------------------------------------

template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > operator*(
    const TaylorExpansion< T, N, M >& a,
    const TaylorExpansion< T, N, M >& b ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::cauchyProduct< T, N, M >(
        r.coefficients(), a.coefficients(), b.coefficients() );
    return r;
}

// ---------------------------------------------------------------------------
// TE / TE division via reciprocal
// ---------------------------------------------------------------------------

template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > operator/(
    const TaylorExpansion< T, N, M >& a,
    const TaylorExpansion< T, N, M >& b ) noexcept
{
    TaylorExpansion< T, N, M > inv_b;
    detail::kernels::seriesReciprocal< T, N, M >( inv_b.coefficients(), b.coefficients() );
    TaylorExpansion< T, N, M > r;
    detail::kernels::cauchyProduct< T, N, M >( r.coefficients(), a.coefficients(),
                                               inv_b.coefficients() );
    return r;
}

// ===========================================================================
// Sparse arithmetic:  S+S, S-S, -S, S+T, T+S, S-T, T-S, S*T, T*S, S/T
// ===========================================================================

using Sparse = storage::Sparse;

/// @brief Sparse + Sparse: two-pointer merge over sorted flat indices.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, Sparse > operator+(
    const TaylorExpansion< T, N, M, Sparse >& a,
    const TaylorExpansion< T, N, M, Sparse >& b ) noexcept
{
    TaylorExpansion< T, N, M, Sparse > r;
    a.container().forEachPair(
        b.container(), [&r]( std::size_t k, T va, T vb )
        {
            const T s = va + vb;
            if ( s != T{ 0 } ) r.container().set( k, s );
        } );
    return r;
}

/// @brief Sparse - Sparse: two-pointer merge over sorted flat indices.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, Sparse > operator-(
    const TaylorExpansion< T, N, M, Sparse >& a,
    const TaylorExpansion< T, N, M, Sparse >& b ) noexcept
{
    TaylorExpansion< T, N, M, Sparse > r;
    a.container().forEachPair(
        b.container(), [&r]( std::size_t k, T va, T vb )
        {
            const T d = va - vb;
            if ( d != T{ 0 } ) r.container().set( k, d );
        } );
    return r;
}

/// @brief Unary negation.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, Sparse > operator-(
    const TaylorExpansion< T, N, M, Sparse >& a ) noexcept
{
    TaylorExpansion< T, N, M, Sparse > r;
    a.container().forEachNonzero(
        [&r]( std::size_t k, T v ) { r.container().set( k, -v ); } );
    return r;
}

/// @brief Sparse * scalar.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, Sparse > operator*(
    const TaylorExpansion< T, N, M, Sparse >& a, T s ) noexcept
{
    if ( s == T{ 0 } ) return TaylorExpansion< T, N, M, Sparse >{};
    TaylorExpansion< T, N, M, Sparse > r;
    a.container().forEachNonzero(
        [&r, s]( std::size_t k, T v ) { r.container().set( k, v * s ); } );
    return r;
}

/// @brief Scalar * Sparse.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, Sparse > operator*(
    T s, const TaylorExpansion< T, N, M, Sparse >& a ) noexcept
{
    return a * s;
}

/// @brief Sparse / scalar.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, Sparse > operator/(
    const TaylorExpansion< T, N, M, Sparse >& a, T s ) noexcept
{
    return a * ( T{ 1 } / s );
}

/// @brief Sparse + scalar: add to constant term.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, Sparse > operator+(
    const TaylorExpansion< T, N, M, Sparse >& a, T s ) noexcept
{
    TaylorExpansion< T, N, M, Sparse > r = a;
    if ( s != T{ 0 } ) r.container().accumulate( 0, s );
    return r;
}

/// @brief Scalar + Sparse.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, Sparse > operator+(
    T s, const TaylorExpansion< T, N, M, Sparse >& a ) noexcept
{
    return a + s;
}

/// @brief Sparse - scalar.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, Sparse > operator-(
    const TaylorExpansion< T, N, M, Sparse >& a, T s ) noexcept
{
    return a + ( -s );
}

/// @brief Scalar - Sparse.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, Sparse > operator-(
    T s, const TaylorExpansion< T, N, M, Sparse >& a ) noexcept
{
    return ( -a ) + s;
}

/// @brief Sparse * Sparse: truncated Cauchy product via the sparse kernel.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, Sparse > operator*(
    const TaylorExpansion< T, N, M, Sparse >& a,
    const TaylorExpansion< T, N, M, Sparse >& b ) noexcept
{
    TaylorExpansion< T, N, M, Sparse > r;
    detail::kernels::sparseCauchyProduct< T, N, M >(
        r.container(), a.container(), b.container() );
    return r;
}

}  // namespace tax
