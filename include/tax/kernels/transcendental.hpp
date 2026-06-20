#pragma once

#include <cmath>
#include <numbers>
#include <span>
#include <tax/core/concepts.hpp>
#include <tax/kernels/algebra.hpp>

namespace tax::detail::kernels
{

/// Natural exponential series `out = exp(a)`.
template < typename T, int N, int M >
void seriesExp( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    using std::exp;
    out = {};
    out[0] = exp( a[0] );

    if constexpr ( M == 1 )
    {
        for ( int d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 0; k < d; ++k )
                rhs += T( d - k ) * a[std::size_t( d - k )] * out[std::size_t( k )];
            out[std::size_t( d )] = rhs / T( d );
        }
    } else
    {
        forEachRecurrenceRow< N, M >(
            [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
                T rhs = T{ 0 };
                for ( const RecurrenceEntry& e : row ) rhs += T( e.db ) * a[e.b_idx] * out[e.g_idx];
                out[ai] = rhs / T( d );
            } );
    }
}

/// Natural logarithm series `out = log(a)`. Requires `a[0] > 0`.
template < typename T, int N, int M >
void seriesLog( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
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
            for ( int k = 1; k < d; ++k )
                rhs += T( k ) * a[std::size_t( d - k )] * out[std::size_t( k )];
            out[std::size_t( d )] = ( a[std::size_t( d )] - rhs / T( d ) ) * inv_a0;
        }
    } else
    {
        forEachRecurrenceRow< N, M >(
            [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
                T rhs = T{ 0 };
                // |beta| == d entries carry weight (d - db) == 0.
                for ( const RecurrenceEntry& e : row )
                    rhs += T( d - int( e.db ) ) * a[e.b_idx] * out[e.g_idx];
                out[ai] = ( a[ai] - rhs / T( d ) ) * inv_a0;
            } );
    }
}

/// Hyperbolic sine series `out = sinh(a)`.
template < typename T, int N, int M >
void seriesSinh( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    using std::sinh;
    constexpr std::size_t S = numMonomials( N, M );

    // compute exp(a) and exp(-a)
    Coeffs< T, N, M > ep{}, em{}, neg_a{};
    neg_a = a;
    for ( std::size_t i = 0; i < S; ++i ) neg_a[i] = -neg_a[i];
    seriesExp< T, N, M >( ep, a );
    seriesExp< T, N, M >( em, neg_a );

    out = {};
    out[0] = sinh( a[0] );
    for ( std::size_t i = 1; i < S; ++i ) out[i] = ( ep[i] - em[i] ) * T{ 0.5 };
}

/// Hyperbolic cosine series `out = cosh(a)`.
template < typename T, int N, int M >
void seriesCosh( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    using std::cosh;
    constexpr std::size_t S = numMonomials( N, M );

    // compute exp(a) and exp(-a)
    Coeffs< T, N, M > ep{}, em{}, neg_a{};
    neg_a = a;
    for ( std::size_t i = 0; i < S; ++i ) neg_a[i] = -neg_a[i];
    seriesExp< T, N, M >( ep, a );
    seriesExp< T, N, M >( em, neg_a );

    out = {};
    out[0] = cosh( a[0] );
    for ( std::size_t i = 1; i < S; ++i ) out[i] = ( ep[i] + em[i] ) * T{ 0.5 };
}

/// Hyperbolic tangent series `out = tanh(a)`.
template < typename T, int N, int M >
void seriesTanh( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    using std::cosh;
    using std::sinh;
    using std::tanh;
    constexpr std::size_t S = numMonomials( N, M );

    // h = cosh(a), s = sinh(a). Share a single exp(a)/exp(-a) pair rather than
    // calling seriesCosh + seriesSinh, each of which would recompute both exps
    // (4 seriesExp calls -> 2). The derived coefficients are bit-identical to the
    // dedicated kernels.
    Coeffs< T, N, M > ep{}, em{}, neg_a{};
    neg_a = a;
    for ( std::size_t i = 0; i < S; ++i ) neg_a[i] = -neg_a[i];
    seriesExp< T, N, M >( ep, a );
    seriesExp< T, N, M >( em, neg_a );

    Coeffs< T, N, M > h{}, s{};
    h[0] = cosh( a[0] );
    s[0] = sinh( a[0] );
    for ( std::size_t i = 1; i < S; ++i )
    {
        h[i] = ( ep[i] + em[i] ) * T{ 0.5 };
        s[i] = ( ep[i] - em[i] ) * T{ 0.5 };
    }

    out = {};
    out[0] = tanh( a[0] );
    const T inv_h0 = T{ 1 } / h[0];

    // Solve  cosh * tanh = sinh  degree-by-degree:
    //   cosh[0]*out[d] = sinh[d] - sum_{k=1}^{d} cosh[k]*out[d-k]
    if constexpr ( M == 1 )
    {
        for ( int d = 1; d <= N; ++d )
        {
            T rhs = s[std::size_t( d )];
            for ( int k = 1; k <= d; ++k ) rhs -= h[std::size_t( k )] * out[std::size_t( d - k )];
            out[std::size_t( d )] = rhs * inv_h0;
        }
    } else
    {
        forEachRecurrenceRow< N, M >(
            [&]( std::size_t ai, int, std::span< const RecurrenceEntry > row ) {
                T rhs = s[ai];
                for ( const RecurrenceEntry& e : row ) rhs -= h[e.b_idx] * out[e.g_idx];
                out[ai] = rhs * inv_h0;
            } );
    }
}

/// Inverse hyperbolic sine series `out = asinh(a)`.
template < typename T, int N, int M >
void seriesAsinh( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    using std::asinh;
    constexpr std::size_t S = numMonomials( N, M );

    // h = sqrt(1 + a^2)
    Coeffs< T, N, M > asq{}, opf{}, h{};
    cauchySelfProduct< T, N, M >( asq, a );
    opf = {};
    opf[0] = T{ 1 };
    for ( std::size_t i = 0; i < S; ++i ) opf[i] += asq[i];
    seriesSqrt< T, N, M >( h, opf );

    out = {};
    out[0] = asinh( a[0] );
    const T inv_h0 = T{ 1 } / h[0];

    if constexpr ( M == 1 )
    {
        for ( int d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 1; k < d; ++k )
                rhs += T( k ) * h[std::size_t( d - k )] * out[std::size_t( k )];
            out[std::size_t( d )] = ( a[std::size_t( d )] - rhs / T( d ) ) * inv_h0;
        }
    } else
    {
        forEachRecurrenceRow< N, M >(
            [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
                T rhs = T{ 0 };
                // |beta| == d entries carry weight (d - db) == 0.
                for ( const RecurrenceEntry& e : row )
                    rhs += T( d - int( e.db ) ) * h[e.b_idx] * out[e.g_idx];
                out[ai] = ( a[ai] - rhs / T( d ) ) * inv_h0;
            } );
    }
}

/// Inverse hyperbolic cosine series `out = acosh(a)`. Requires `a[0] > 1`.
template < typename T, int N, int M >
void seriesAcosh( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    using std::acosh;
    constexpr std::size_t S = numMonomials( N, M );

    // h = sqrt(a^2 - 1)
    Coeffs< T, N, M > asq{}, amf{}, h{};
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
            for ( int k = 1; k < d; ++k )
                rhs += T( k ) * h[std::size_t( d - k )] * out[std::size_t( k )];
            out[std::size_t( d )] = ( a[std::size_t( d )] - rhs / T( d ) ) * inv_h0;
        }
    } else
    {
        forEachRecurrenceRow< N, M >(
            [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
                T rhs = T{ 0 };
                // |beta| == d entries carry weight (d - db) == 0.
                for ( const RecurrenceEntry& e : row )
                    rhs += T( d - int( e.db ) ) * h[e.b_idx] * out[e.g_idx];
                out[ai] = ( a[ai] - rhs / T( d ) ) * inv_h0;
            } );
    }
}

/// Inverse hyperbolic tangent series `out = atanh(a)`. Requires `|a[0]| < 1`.
template < typename T, int N, int M >
void seriesAtanh( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    using std::atanh;
    constexpr std::size_t S = numMonomials( N, M );

    // h = 1 - a^2
    Coeffs< T, N, M > h{};
    cauchySelfProduct< T, N, M >( h, a );
    for ( std::size_t i = 0; i < S; ++i ) h[i] = -h[i];
    h[0] += T{ 1 };

    out = {};
    out[0] = atanh( a[0] );
    const T inv_h0 = T{ 1 } / h[0];

    if constexpr ( M == 1 )
    {
        for ( int d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 1; k < d; ++k )
                rhs += T( k ) * h[std::size_t( d - k )] * out[std::size_t( k )];
            out[std::size_t( d )] = ( a[std::size_t( d )] - rhs / T( d ) ) * inv_h0;
        }
    } else
    {
        forEachRecurrenceRow< N, M >(
            [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
                T rhs = T{ 0 };
                // |beta| == d entries carry weight (d - db) == 0.
                for ( const RecurrenceEntry& e : row )
                    rhs += T( d - int( e.db ) ) * h[e.b_idx] * out[e.g_idx];
                out[ai] = ( a[ai] - rhs / T( d ) ) * inv_h0;
            } );
    }
}

/// Error function series `out = erf(a)`.
template < typename T, int N, int M >
void seriesErf( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    using std::erf;
    constexpr std::size_t S = numMonomials( N, M );
    // Name the constant in the underlying real scalar so vectorised coefficient
    // types (whose lanes are floating-point) work too; broadcast into T.
    using R = real_scalar_t< T >;
    const T two_over_sqrtpi = T( R{ 2 } * std::numbers::inv_sqrtpi_v< R > );

    // h = (2/sqrt(pi)) * exp(-a^2)
    Coeffs< T, N, M > asq{}, neg_asq{}, e{}, h{};
    cauchySelfProduct< T, N, M >( asq, a );
    neg_asq = asq;
    for ( std::size_t i = 0; i < S; ++i ) neg_asq[i] = -neg_asq[i];
    seriesExp< T, N, M >( e, neg_asq );
    h = e;
    for ( std::size_t i = 0; i < S; ++i ) h[i] *= two_over_sqrtpi;

    out = {};
    out[0] = erf( a[0] );

    if constexpr ( M == 1 )
    {
        for ( int d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 0; k < d; ++k )
                rhs += T( d - k ) * a[std::size_t( d - k )] * h[std::size_t( k )];
            out[std::size_t( d )] = rhs / T( d );
        }
    } else
    {
        forEachRecurrenceRow< N, M >(
            [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
                T rhs = T{ 0 };
                for ( const RecurrenceEntry& e : row ) rhs += T( e.db ) * a[e.b_idx] * h[e.g_idx];
                out[ai] = rhs / T( d );
            } );
    }
}

}  // namespace tax::detail::kernels
