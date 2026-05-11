#pragma once

#include <cmath>
#include <span>
#include <vector>

#include <tax/kernels/algebra.hpp>
#include <tax/kernels/cauchy.hpp>
#include <tax/kernels/ops.hpp>
#include <tax/utils/enumeration.hpp>

namespace tax::detail
{

template < typename T, int N, int M >
/**
 * @brief Natural logarithm series `out = log(a)`.
 * @details Requires `a[0] > 0` for real-valued output.
 */
constexpr void seriesLog( Coeffs< T, N, M >& out,
                          const Coeffs< T, N, M >& a ) noexcept
{
    using std::log;
    out = {};
    out[0] = log( a[0] );
    const T inv_a0 = T{ 1 } / a[0];

    if constexpr ( M == 1 )
    {
        for ( int d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 1; k < d; ++k ) rhs += T( k ) * a[d - k] * out[k];
            out[d] = ( a[d] - rhs / T( d ) ) * inv_a0;
        }
    } else
    {
        for ( int d = 1; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                T rhs = T{ 0 };
                forEachSubIndex< M >( alpha, 1, d - 1, [&]( auto bi, auto gi, int db ) {
                    rhs += T( d - db ) * a[bi] * out[gi];
                } );
                out[ai] = ( a[ai] - rhs / T( d ) ) * inv_a0;
            } );
        }
    }
}

template < typename T, int N, int M >
constexpr void seriesExp( Coeffs< T, N, M >& out,
                          const Coeffs< T, N, M >& a ) noexcept
{
    using std::exp;
    out = {};
    out[0] = exp( a[0] );

    if constexpr ( M == 1 )
    {
        for ( int d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 0; k < d; ++k ) rhs += T( d - k ) * a[d - k] * out[k];
            out[d] = rhs / T( d );
        }
    } else
    {
        for ( int d = 1; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                T rhs = T{ 0 };
                forEachSubIndex< M >( alpha, 1, d, [&]( auto bi, auto gi, int db ) {
                    rhs += T( db ) * a[bi] * out[gi];
                } );
                out[ai] = rhs / T( d );
            } );
        }
    }
}

template < typename T, int N, int M >
constexpr void seriesPow( Coeffs< T, N, M >& out,
                          const Coeffs< T, N, M >& a, T c ) noexcept
{
    using std::pow;
    out = {};
    out[0] = pow( a[0], c );
    const T inv_a0 = T{ 1 } / a[0];

    if constexpr ( M == 1 )
    {
        for ( int d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 0; k < d; ++k ) rhs += ( c * T( d - k ) - T( k ) ) * a[d - k] * out[k];
            out[d] = rhs * inv_a0 / T( d );
        }
    } else
    {
        for ( int d = 1; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                T rhs = T{ 0 };
                forEachSubIndex< M >( alpha, 1, d, [&]( auto bi, auto gi, int db ) {
                    rhs += ( c * T( db ) - T( d - db ) ) * a[bi] * out[gi];
                } );
                out[ai] = rhs * inv_a0 / T( d );
            } );
        }
    }
}

template < typename T, int N, int M >
constexpr void seriesIntPow( Coeffs< T, N, M >& out,
                             const Coeffs< T, N, M >& a, int n ) noexcept
{
    constexpr auto S = numMonomials( N, M );

    if ( n == 0 )
    {
        out = {};
        out[0] = T{ 1 };
        return;
    }
    if ( n == 1 )
    {
        out = a;
        return;
    }
    if ( n == -1 )
    {
        seriesReciprocal< T, N, M >( out, a );
        return;
    }
    if ( n < 0 )
    {
        std::array< T, S > rec{};
        seriesReciprocal< T, N, M >( rec, a );
        seriesIntPow< T, N, M >( out, rec, -n );
        return;
    }
    // n >= 2: binary exponentiation
    std::array< T, S > base = a;
    out = {};
    out[0] = T{ 1 };
    int e = n;
    while ( e > 0 )
    {
        if ( e & 1 )
        {
            std::array< T, S > tmp{};
            cauchyProduct< T, N, M >( tmp, out, base );
            out = tmp;
        }
        e >>= 1;
        if ( e > 0 )
        {
            std::array< T, S > tmp{};
            cauchyProduct< T, N, M >( tmp, base, base );
            base = tmp;
        }
    }
}

template < typename T, int N, int M >
constexpr void seriesErf( Coeffs< T, N, M >& out,
                          const Coeffs< T, N, M >& a ) noexcept
{
    using std::acos;
    using std::erf;
    using std::exp;
    using std::sqrt;
    constexpr auto S = numMonomials( N, M );
    const T two_over_sqrtpi = T{ 2 } / sqrt( acos( T{ -1 } ) );

    // h = (2/√π) · exp(-a²)
    std::array< T, S > asq{}, neg_asq{}, e{}, h{};
    cauchySelfProduct< T, N, M >( asq, a );
    neg_asq = asq;
    negateInPlace< T, S >( neg_asq );
    seriesExp< T, N, M >( e, neg_asq );
    h = e;
    scaleInPlace< T, S >( h, two_over_sqrtpi );

    out = {};
    out[0] = erf( a[0] );

    if constexpr ( M == 1 )
    {
        for ( int d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 0; k < d; ++k ) rhs += T( d - k ) * a[d - k] * h[k];
            out[d] = rhs / T( d );
        }
    } else
    {
        for ( int d = 1; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                T rhs = T{ 0 };
                forEachSubIndex< M >( alpha, 1, d, [&]( auto bi, auto gi, int db ) {
                    rhs += T( db ) * a[bi] * h[gi];
                } );
                out[ai] = rhs / T( d );
            } );
        }
    }
}

template < typename T, int N, int M >
constexpr void seriesAsin( Coeffs< T, N, M >& out,
                           const Coeffs< T, N, M >& a ) noexcept
{
    using std::asin;
    constexpr auto S = numMonomials( N, M );

    // h = sqrt(1 - a²)
    std::array< T, S > asq{}, omf{}, h{};
    cauchySelfProduct< T, N, M >( asq, a );
    omf = {};
    omf[0] = T{ 1 };
    subInPlace< T, S >( omf, asq );
    seriesSqrt< T, N, M >( h, omf );

    out = {};
    out[0] = asin( a[0] );
    const T inv_h0 = T{ 1 } / h[0];

    if constexpr ( M == 1 )
    {
        for ( int d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 1; k < d; ++k ) rhs += T( k ) * h[d - k] * out[k];
            out[d] = ( a[d] - rhs / T( d ) ) * inv_h0;
        }
    } else
    {
        for ( int d = 1; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                T rhs = T{ 0 };
                forEachSubIndex< M >( alpha, 1, d - 1, [&]( auto bi, auto gi, int db ) {
                    rhs += T( d - db ) * h[bi] * out[gi];
                } );
                out[ai] = ( a[ai] - rhs / T( d ) ) * inv_h0;
            } );
        }
    }
}

template < typename T, int N, int M >
constexpr void seriesAcos( Coeffs< T, N, M >& out,
                           const Coeffs< T, N, M >& a ) noexcept
{
    seriesAsin< T, N, M >( out, a );
    negateInPlace< T, numMonomials( N, M ) >( out );
    using std::acos;
    out[0] += acos( T{ -1 } ) / T{ 2 };  // pi/2
}

template < typename T, int N, int M >
constexpr void seriesAtan( Coeffs< T, N, M >& out,
                           const Coeffs< T, N, M >& a ) noexcept
{
    using std::atan;
    constexpr auto S = numMonomials( N, M );

    // h = 1 + a²
    std::array< T, S > h{};
    cauchySelfProduct< T, N, M >( h, a );
    h[0] += T{ 1 };

    out = {};
    out[0] = atan( a[0] );
    const T inv_h0 = T{ 1 } / h[0];

    if constexpr ( M == 1 )
    {
        for ( int d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 1; k < d; ++k ) rhs += T( k ) * h[d - k] * out[k];
            out[d] = ( a[d] - rhs / T( d ) ) * inv_h0;
        }
    } else
    {
        for ( int d = 1; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                T rhs = T{ 0 };
                forEachSubIndex< M >( alpha, 1, d - 1, [&]( auto bi, auto gi, int db ) {
                    rhs += T( d - db ) * h[bi] * out[gi];
                } );
                out[ai] = ( a[ai] - rhs / T( d ) ) * inv_h0;
            } );
        }
    }
}

template < typename T, int N, int M >
constexpr void seriesAtan2( Coeffs< T, N, M >& out,
                            const Coeffs< T, N, M >& y,
                            const Coeffs< T, N, M >& x ) noexcept
{
    using std::atan2;
    constexpr auto S = numMonomials( N, M );

    // h = x² + y²
    std::array< T, S > h{};
    cauchySelfProduct< T, N, M >( h, x );
    std::array< T, S > ysq{};
    cauchySelfProduct< T, N, M >( ysq, y );
    addInPlace< T, S >( h, ysq );

    out = {};
    out[0] = atan2( y[0], x[0] );
    const T inv_h0 = T{ 1 } / h[0];

    if constexpr ( M == 1 )
    {
        for ( int d = 1; d <= N; ++d )
        {
            T rhs = T( d ) * ( x[0] * y[d] - y[0] * x[d] );
            for ( int k = 1; k < d; ++k )
                rhs += T( k ) * ( x[d - k] * y[k] - y[d - k] * x[k] - h[d - k] * out[k] );
            out[d] = rhs * inv_h0 / T( d );
        }
    } else
    {
        for ( int d = 1; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                T rhs = T{ 0 };
                forEachSubIndex< M >( alpha, 1, d - 1, [&]( auto bi, auto gi, int db ) {
                    rhs += T( d - db ) * ( x[bi] * y[gi] - y[bi] * x[gi] - h[bi] * out[gi] );
                } );
                out[ai] = ( ( x[0] * y[ai] - y[0] * x[ai] ) + rhs / T( d ) ) * inv_h0;
            } );
        }
    }
}

template < typename T, int N, int M >
/**
 * @brief Inverse hyperbolic sine series `out = asinh(a)`.
 * @details Solves `sqrt(1 + a²) · out' = a'` degree by degree.
 */
constexpr void seriesAsinh( Coeffs< T, N, M >& out,
                            const Coeffs< T, N, M >& a ) noexcept
{
    using std::asinh;
    constexpr auto S = numMonomials( N, M );

    // h = sqrt(1 + a²)
    std::array< T, S > asq{}, opf{}, h{};
    cauchySelfProduct< T, N, M >( asq, a );
    opf = {};
    opf[0] = T{ 1 };
    addInPlace< T, S >( opf, asq );
    seriesSqrt< T, N, M >( h, opf );

    out = {};
    out[0] = asinh( a[0] );
    const T inv_h0 = T{ 1 } / h[0];

    if constexpr ( M == 1 )
    {
        for ( int d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 1; k < d; ++k ) rhs += T( k ) * h[d - k] * out[k];
            out[d] = ( a[d] - rhs / T( d ) ) * inv_h0;
        }
    } else
    {
        for ( int d = 1; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                T rhs = T{ 0 };
                forEachSubIndex< M >( alpha, 1, d - 1, [&]( auto bi, auto gi, int db ) {
                    rhs += T( d - db ) * h[bi] * out[gi];
                } );
                out[ai] = ( a[ai] - rhs / T( d ) ) * inv_h0;
            } );
        }
    }
}

template < typename T, int N, int M >
/**
 * @brief Inverse hyperbolic cosine series `out = acosh(a)`.
 * @details Solves `sqrt(a² - 1) · out' = a'` degree by degree. Requires `a[0] > 1`.
 */
constexpr void seriesAcosh( Coeffs< T, N, M >& out,
                            const Coeffs< T, N, M >& a ) noexcept
{
    using std::acosh;
    constexpr auto S = numMonomials( N, M );

    // h = sqrt(a² - 1)
    std::array< T, S > asq{}, amf{}, h{};
    cauchySelfProduct< T, N, M >( asq, a );
    amf = asq;
    amf[0] -= T{ 1 };
    seriesSqrt< T, N, M >( h, amf );

    out = {};
    out[0] = acosh( a[0] );
    const T inv_h0 = T{ 1 } / h[0];

    if constexpr ( M == 1 )
    {
        for ( int d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 1; k < d; ++k ) rhs += T( k ) * h[d - k] * out[k];
            out[d] = ( a[d] - rhs / T( d ) ) * inv_h0;
        }
    } else
    {
        for ( int d = 1; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                T rhs = T{ 0 };
                forEachSubIndex< M >( alpha, 1, d - 1, [&]( auto bi, auto gi, int db ) {
                    rhs += T( d - db ) * h[bi] * out[gi];
                } );
                out[ai] = ( a[ai] - rhs / T( d ) ) * inv_h0;
            } );
        }
    }
}

template < typename T, int N, int M >
/**
 * @brief Inverse hyperbolic tangent series `out = atanh(a)`.
 * @details Solves `(1 - a²) · out' = a'` degree by degree. Requires `|a[0]| < 1`.
 */
constexpr void seriesAtanh( Coeffs< T, N, M >& out,
                            const Coeffs< T, N, M >& a ) noexcept
{
    using std::atanh;
    constexpr auto S = numMonomials( N, M );

    // h = 1 - a²
    std::array< T, S > h{};
    cauchySelfProduct< T, N, M >( h, a );
    negateInPlace< T, S >( h );
    h[0] += T{ 1 };

    out = {};
    out[0] = atanh( a[0] );
    const T inv_h0 = T{ 1 } / h[0];

    if constexpr ( M == 1 )
    {
        for ( int d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 1; k < d; ++k ) rhs += T( k ) * h[d - k] * out[k];
            out[d] = ( a[d] - rhs / T( d ) ) * inv_h0;
        }
    } else
    {
        for ( int d = 1; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                T rhs = T{ 0 };
                forEachSubIndex< M >( alpha, 1, d - 1, [&]( auto bi, auto gi, int db ) {
                    rhs += T( d - db ) * h[bi] * out[gi];
                } );
                out[ai] = ( a[ai] - rhs / T( d ) ) * inv_h0;
            } );
        }
    }
}

// =============================================================================
// Runtime-shape variants (used by the dynamic-shape `TaylorExpansionT`).
// =============================================================================

/// @brief Runtime overload of `seriesExp`.
template < typename T >
inline void seriesExp( T* out, const T* a, std::size_t N, std::size_t M ) noexcept
{
    using std::exp;
    const std::size_t S = numMonomials( N, M );
    for ( std::size_t i = 0; i < S; ++i ) out[i] = T{ 0 };
    out[0] = exp( a[0] );

    if ( M == 1 )
    {
        for ( std::size_t d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( std::size_t k = 0; k < d; ++k ) rhs += T( d - k ) * a[d - k] * out[k];
            out[d] = rhs / T( d );
        }
    }
    else
    {
        for ( int d = 1; d <= int( N ); ++d )
        {
            forEachMonomial( int( M ), d, [&]( std::span< const int > alpha, std::size_t ai ) {
                T rhs = T{ 0 };
                forEachSubIndex( alpha, 1, d, [&]( std::size_t bi, std::size_t gi, int db ) {
                    rhs += T( db ) * a[bi] * out[gi];
                } );
                out[ai] = rhs / T( d );
            } );
        }
    }
}

/// @brief Runtime overload of `seriesLog`. Requires `a[0] > 0`.
template < typename T >
inline void seriesLog( T* out, const T* a, std::size_t N, std::size_t M ) noexcept
{
    using std::log;
    const std::size_t S = numMonomials( N, M );
    for ( std::size_t i = 0; i < S; ++i ) out[i] = T{ 0 };
    out[0] = log( a[0] );
    const T inv_a0 = T{ 1 } / a[0];

    if ( M == 1 )
    {
        for ( std::size_t d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( std::size_t k = 1; k < d; ++k ) rhs += T( k ) * a[d - k] * out[k];
            out[d] = ( a[d] - rhs / T( d ) ) * inv_a0;
        }
    }
    else
    {
        for ( int d = 1; d <= int( N ); ++d )
        {
            forEachMonomial( int( M ), d, [&]( std::span< const int > alpha, std::size_t ai ) {
                T rhs = T{ 0 };
                forEachSubIndex( alpha, 1, d - 1, [&]( std::size_t bi, std::size_t gi, int db ) {
                    rhs += T( d - db ) * a[bi] * out[gi];
                } );
                out[ai] = ( a[ai] - rhs / T( d ) ) * inv_a0;
            } );
        }
    }
}

/// @brief Runtime overload of `seriesPow`: `out = a^c` for a real exponent.
template < typename T >
inline void seriesPow( T* out, const T* a, T c, std::size_t N, std::size_t M ) noexcept
{
    using std::pow;
    const std::size_t S = numMonomials( N, M );
    for ( std::size_t i = 0; i < S; ++i ) out[i] = T{ 0 };
    out[0] = pow( a[0], c );
    const T inv_a0 = T{ 1 } / a[0];

    if ( M == 1 )
    {
        for ( std::size_t d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( std::size_t k = 0; k < d; ++k )
                rhs += ( c * T( d - k ) - T( k ) ) * a[d - k] * out[k];
            out[d] = rhs * inv_a0 / T( d );
        }
    }
    else
    {
        for ( int d = 1; d <= int( N ); ++d )
        {
            forEachMonomial( int( M ), d, [&]( std::span< const int > alpha, std::size_t ai ) {
                T rhs = T{ 0 };
                forEachSubIndex( alpha, 1, d, [&]( std::size_t bi, std::size_t gi, int db ) {
                    rhs += ( c * T( db ) - T( d - db ) ) * a[bi] * out[gi];
                } );
                out[ai] = rhs * inv_a0 / T( d );
            } );
        }
    }
}

/// @brief Runtime overload of `seriesAsin`. Requires |a[0]| < 1.
template < typename T >
inline void seriesAsin( T* out, const T* a, std::size_t N, std::size_t M )
{
    using std::asin;
    const std::size_t S = numMonomials( N, M );

    // h = sqrt(1 - a^2)
    std::vector< T > asq( S, T{ 0 } ), omf( S, T{ 0 } ), h( S, T{ 0 } );
    cauchySelfProduct( asq.data(), a, N, M );
    omf[0] = T{ 1 };
    subInPlace( omf.data(), asq.data(), S );
    seriesSqrt( h.data(), omf.data(), N, M );

    for ( std::size_t i = 0; i < S; ++i ) out[i] = T{ 0 };
    out[0] = asin( a[0] );
    const T inv_h0 = T{ 1 } / h[0];

    if ( M == 1 )
    {
        for ( std::size_t d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( std::size_t k = 1; k < d; ++k ) rhs += T( k ) * h[d - k] * out[k];
            out[d] = ( a[d] - rhs / T( d ) ) * inv_h0;
        }
    }
    else
    {
        for ( int d = 1; d <= int( N ); ++d )
        {
            forEachMonomial( int( M ), d, [&]( std::span< const int > alpha, std::size_t ai ) {
                T rhs = T{ 0 };
                forEachSubIndex( alpha, 1, d - 1, [&]( std::size_t bi, std::size_t gi, int db ) {
                    rhs += T( d - db ) * h[bi] * out[gi];
                } );
                out[ai] = ( a[ai] - rhs / T( d ) ) * inv_h0;
            } );
        }
    }
}

/// @brief Runtime overload of `seriesAcos`: acos(a) = pi/2 - asin(a).
template < typename T >
inline void seriesAcos( T* out, const T* a, std::size_t N, std::size_t M )
{
    using std::acos;
    const std::size_t S = numMonomials( N, M );
    seriesAsin( out, a, N, M );
    negateInPlace( out, S );
    out[0] += acos( T{ -1 } ) / T{ 2 };  // pi/2
}

/// @brief Runtime overload of `seriesAtan`.
template < typename T >
inline void seriesAtan( T* out, const T* a, std::size_t N, std::size_t M )
{
    using std::atan;
    const std::size_t S = numMonomials( N, M );

    // h = 1 + a^2
    std::vector< T > h( S, T{ 0 } );
    cauchySelfProduct( h.data(), a, N, M );
    h[0] += T{ 1 };

    for ( std::size_t i = 0; i < S; ++i ) out[i] = T{ 0 };
    out[0] = atan( a[0] );
    const T inv_h0 = T{ 1 } / h[0];

    if ( M == 1 )
    {
        for ( std::size_t d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( std::size_t k = 1; k < d; ++k ) rhs += T( k ) * h[d - k] * out[k];
            out[d] = ( a[d] - rhs / T( d ) ) * inv_h0;
        }
    }
    else
    {
        for ( int d = 1; d <= int( N ); ++d )
        {
            forEachMonomial( int( M ), d, [&]( std::span< const int > alpha, std::size_t ai ) {
                T rhs = T{ 0 };
                forEachSubIndex( alpha, 1, d - 1, [&]( std::size_t bi, std::size_t gi, int db ) {
                    rhs += T( d - db ) * h[bi] * out[gi];
                } );
                out[ai] = ( a[ai] - rhs / T( d ) ) * inv_h0;
            } );
        }
    }
}

/// @brief Runtime overload of `seriesAsinh`. Solves `sqrt(1+a^2) * out' = a'`.
template < typename T >
inline void seriesAsinh( T* out, const T* a, std::size_t N, std::size_t M )
{
    using std::asinh;
    const std::size_t S = numMonomials( N, M );

    // h = sqrt(1 + a^2)
    std::vector< T > asq( S, T{ 0 } ), opf( S, T{ 0 } ), h( S, T{ 0 } );
    cauchySelfProduct( asq.data(), a, N, M );
    opf[0] = T{ 1 };
    addInPlace( opf.data(), asq.data(), S );
    seriesSqrt( h.data(), opf.data(), N, M );

    for ( std::size_t i = 0; i < S; ++i ) out[i] = T{ 0 };
    out[0] = asinh( a[0] );
    const T inv_h0 = T{ 1 } / h[0];

    if ( M == 1 )
    {
        for ( std::size_t d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( std::size_t k = 1; k < d; ++k ) rhs += T( k ) * h[d - k] * out[k];
            out[d] = ( a[d] - rhs / T( d ) ) * inv_h0;
        }
    }
    else
    {
        for ( int d = 1; d <= int( N ); ++d )
        {
            forEachMonomial( int( M ), d, [&]( std::span< const int > alpha, std::size_t ai ) {
                T rhs = T{ 0 };
                forEachSubIndex( alpha, 1, d - 1, [&]( std::size_t bi, std::size_t gi, int db ) {
                    rhs += T( d - db ) * h[bi] * out[gi];
                } );
                out[ai] = ( a[ai] - rhs / T( d ) ) * inv_h0;
            } );
        }
    }
}

/// @brief Runtime overload of `seriesAcosh`. Requires `a[0] > 1`.
template < typename T >
inline void seriesAcosh( T* out, const T* a, std::size_t N, std::size_t M )
{
    using std::acosh;
    const std::size_t S = numMonomials( N, M );

    // h = sqrt(a^2 - 1)
    std::vector< T > asq( S, T{ 0 } ), amf( S, T{ 0 } ), h( S, T{ 0 } );
    cauchySelfProduct( asq.data(), a, N, M );
    for ( std::size_t i = 0; i < S; ++i ) amf[i] = asq[i];
    amf[0] -= T{ 1 };
    seriesSqrt( h.data(), amf.data(), N, M );

    for ( std::size_t i = 0; i < S; ++i ) out[i] = T{ 0 };
    out[0] = acosh( a[0] );
    const T inv_h0 = T{ 1 } / h[0];

    if ( M == 1 )
    {
        for ( std::size_t d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( std::size_t k = 1; k < d; ++k ) rhs += T( k ) * h[d - k] * out[k];
            out[d] = ( a[d] - rhs / T( d ) ) * inv_h0;
        }
    }
    else
    {
        for ( int d = 1; d <= int( N ); ++d )
        {
            forEachMonomial( int( M ), d, [&]( std::span< const int > alpha, std::size_t ai ) {
                T rhs = T{ 0 };
                forEachSubIndex( alpha, 1, d - 1, [&]( std::size_t bi, std::size_t gi, int db ) {
                    rhs += T( d - db ) * h[bi] * out[gi];
                } );
                out[ai] = ( a[ai] - rhs / T( d ) ) * inv_h0;
            } );
        }
    }
}

/// @brief Runtime overload of `seriesAtanh`. Requires `|a[0]| < 1`.
template < typename T >
inline void seriesAtanh( T* out, const T* a, std::size_t N, std::size_t M )
{
    using std::atanh;
    const std::size_t S = numMonomials( N, M );

    // h = 1 - a^2
    std::vector< T > h( S, T{ 0 } );
    cauchySelfProduct( h.data(), a, N, M );
    negateInPlace( h.data(), S );
    h[0] += T{ 1 };

    for ( std::size_t i = 0; i < S; ++i ) out[i] = T{ 0 };
    out[0] = atanh( a[0] );
    const T inv_h0 = T{ 1 } / h[0];

    if ( M == 1 )
    {
        for ( std::size_t d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( std::size_t k = 1; k < d; ++k ) rhs += T( k ) * h[d - k] * out[k];
            out[d] = ( a[d] - rhs / T( d ) ) * inv_h0;
        }
    }
    else
    {
        for ( int d = 1; d <= int( N ); ++d )
        {
            forEachMonomial( int( M ), d, [&]( std::span< const int > alpha, std::size_t ai ) {
                T rhs = T{ 0 };
                forEachSubIndex( alpha, 1, d - 1, [&]( std::size_t bi, std::size_t gi, int db ) {
                    rhs += T( d - db ) * h[bi] * out[gi];
                } );
                out[ai] = ( a[ai] - rhs / T( d ) ) * inv_h0;
            } );
        }
    }
}

/// @brief Runtime overload of `seriesErf`.
template < typename T >
inline void seriesErf( T* out, const T* a, std::size_t N, std::size_t M )
{
    using std::acos;
    using std::erf;
    using std::sqrt;
    const std::size_t S = numMonomials( N, M );
    const T two_over_sqrtpi = T{ 2 } / sqrt( acos( T{ -1 } ) );

    // h = (2/sqrt(pi)) * exp(-a^2)
    std::vector< T > asq( S, T{ 0 } ), neg_asq( S, T{ 0 } ), e( S, T{ 0 } ), h( S, T{ 0 } );
    cauchySelfProduct( asq.data(), a, N, M );
    for ( std::size_t i = 0; i < S; ++i ) neg_asq[i] = asq[i];
    negateInPlace( neg_asq.data(), S );
    seriesExp( e.data(), neg_asq.data(), N, M );
    for ( std::size_t i = 0; i < S; ++i ) h[i] = e[i];
    scaleInPlace( h.data(), two_over_sqrtpi, S );

    for ( std::size_t i = 0; i < S; ++i ) out[i] = T{ 0 };
    out[0] = erf( a[0] );

    if ( M == 1 )
    {
        for ( std::size_t d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( std::size_t k = 0; k < d; ++k ) rhs += T( d - k ) * a[d - k] * h[k];
            out[d] = rhs / T( d );
        }
    }
    else
    {
        for ( int d = 1; d <= int( N ); ++d )
        {
            forEachMonomial( int( M ), d, [&]( std::span< const int > alpha, std::size_t ai ) {
                T rhs = T{ 0 };
                forEachSubIndex( alpha, 1, d, [&]( std::size_t bi, std::size_t gi, int db ) {
                    rhs += T( db ) * a[bi] * h[gi];
                } );
                out[ai] = rhs / T( d );
            } );
        }
    }
}

/// @brief Runtime overload of `seriesAtan2`.
template < typename T >
inline void seriesAtan2( T* out, const T* y, const T* x, std::size_t N, std::size_t M )
{
    using std::atan2;
    const std::size_t S = numMonomials( N, M );

    // h = x^2 + y^2
    std::vector< T > h( S, T{ 0 } ), ysq( S, T{ 0 } );
    cauchySelfProduct( h.data(), x, N, M );
    cauchySelfProduct( ysq.data(), y, N, M );
    addInPlace( h.data(), ysq.data(), S );

    for ( std::size_t i = 0; i < S; ++i ) out[i] = T{ 0 };
    out[0] = atan2( y[0], x[0] );
    const T inv_h0 = T{ 1 } / h[0];

    if ( M == 1 )
    {
        for ( std::size_t d = 1; d <= N; ++d )
        {
            T rhs = T( d ) * ( x[0] * y[d] - y[0] * x[d] );
            for ( std::size_t k = 1; k < d; ++k )
                rhs += T( k ) * ( x[d - k] * y[k] - y[d - k] * x[k] - h[d - k] * out[k] );
            out[d] = rhs * inv_h0 / T( d );
        }
    }
    else
    {
        for ( int d = 1; d <= int( N ); ++d )
        {
            forEachMonomial( int( M ), d, [&]( std::span< const int > alpha, std::size_t ai ) {
                T rhs = T{ 0 };
                forEachSubIndex( alpha, 1, d - 1, [&]( std::size_t bi, std::size_t gi, int db ) {
                    rhs += T( d - db ) * ( x[bi] * y[gi] - y[bi] * x[gi] - h[bi] * out[gi] );
                } );
                out[ai] = ( ( x[0] * y[ai] - y[0] * x[ai] ) + rhs / T( d ) ) * inv_h0;
            } );
        }
    }
}

/// @brief Runtime overload of `seriesIntPow`: integer power via binary exponentiation.
template < typename T >
inline void seriesIntPow( T* out, const T* a, int n, std::size_t N, std::size_t M )
{
    const std::size_t S = numMonomials( N, M );

    if ( n == 0 )
    {
        for ( std::size_t i = 0; i < S; ++i ) out[i] = T{ 0 };
        out[0] = T{ 1 };
        return;
    }
    if ( n == 1 )
    {
        for ( std::size_t i = 0; i < S; ++i ) out[i] = a[i];
        return;
    }
    if ( n == -1 )
    {
        seriesReciprocal( out, a, N, M );
        return;
    }
    if ( n < 0 )
    {
        std::vector< T > rec( S, T{ 0 } );
        seriesReciprocal( rec.data(), a, N, M );
        seriesIntPow( out, rec.data(), -n, N, M );
        return;
    }
    // n >= 2: binary exponentiation.
    std::vector< T > base( a, a + S );
    for ( std::size_t i = 0; i < S; ++i ) out[i] = T{ 0 };
    out[0] = T{ 1 };
    int e = n;
    while ( e > 0 )
    {
        if ( e & 1 )
        {
            std::vector< T > tmp( S, T{ 0 } );
            cauchyProduct( tmp.data(), out, base.data(), N, M );
            for ( std::size_t i = 0; i < S; ++i ) out[i] = tmp[i];
        }
        e >>= 1;
        if ( e > 0 )
        {
            std::vector< T > tmp( S, T{ 0 } );
            cauchyProduct( tmp.data(), base.data(), base.data(), N, M );
            base.swap( tmp );
        }
    }
}

}  // namespace tax::detail
