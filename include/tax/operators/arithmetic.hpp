#pragma once

#include <tax/core/taylor_expansion.hpp>
#include <tax/kernels/algebra.hpp>
#include <tax/kernels/cauchy.hpp>

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

}  // namespace tax
