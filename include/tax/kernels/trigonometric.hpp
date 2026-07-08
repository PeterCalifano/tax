#pragma once

#include <cmath>
#include <span>
#include <tax/core/scheme/isotropic.hpp>
#include <tax/kernels/algebra.hpp>

namespace tax::detail::kernels
{

/// Coupled trigonometric series: jointly compute `sin(a)` and `cos(a)` (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
void seriesSinCos( std::array< T, Scheme::nCoeff >& s, std::array< T, Scheme::nCoeff >& c,
                   const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::cos;
    using std::sin;
    s = {};
    c = {};
    s[0] = sin( a[0] );
    c[0] = cos( a[0] );

    if constexpr ( Scheme::isUnivariate )
    {
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T sr = T{ 0 }, cr = T{ 0 };
            for ( int k = 0; k < d; ++k )
            {
                const T w = T( d - k ) * a[std::size_t( d - k )];
                sr += w * c[std::size_t( k )];
                cr += w * s[std::size_t( k )];
            }
            const T inv_d = T{ 1 } / T( d );
            s[std::size_t( d )] = sr * inv_d;
            c[std::size_t( d )] = -cr * inv_d;
        }
    } else
    {
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
                T sin_rhs = T{ 0 };
                T cos_rhs = T{ 0 };
                for ( const RecurrenceEntry& e : row )
                {
                    const T fg = T( e.db ) * a[e.b_idx];
                    sin_rhs += fg * c[e.g_idx];
                    cos_rhs += fg * s[e.g_idx];
                }
                const T inv_d = T{ 1 } / T( d );
                s[ai] = sin_rhs * inv_d;
                c[ai] = -cos_rhs * inv_d;
            } );
    }
}

/// Sine series `out = sin(a)` (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
void seriesSin( std::array< T, Scheme::nCoeff >& out,
                const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    std::array< T, Scheme::nCoeff > c{};
    seriesSinCos< T, Scheme >( out, c, a );
}

/// Cosine series `out = cos(a)` (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
void seriesCos( std::array< T, Scheme::nCoeff >& out,
                const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    std::array< T, Scheme::nCoeff > s{};
    seriesSinCos< T, Scheme >( s, out, a );
}

/// Tangent series `out = tan(a)` (scheme-generic): sin/cos in one substitution pass.
template < typename T, tax::IndexScheme Scheme >
void seriesTan( std::array< T, Scheme::nCoeff >& out,
                const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    constexpr std::size_t S = Scheme::nCoeff;
    std::array< T, S > s{}, c{};
    seriesSinCos< T, Scheme >( s, c, a );
    seriesDivide< T, Scheme >( out, s, c );
}

/// Inverse sine series `out = asin(a)` (scheme-generic): out' = a' / sqrt(1 - a^2).
template < typename T, tax::IndexScheme Scheme >
void seriesAsin( std::array< T, Scheme::nCoeff >& out,
                 const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::asin;
    constexpr std::size_t S = Scheme::nCoeff;

    // h = sqrt(1 - a^2)
    std::array< T, S > asq{}, h{};
    tax::cauchySelfProduct< T, Scheme >( asq, a );
    for ( T& v : asq ) v = -v;
    asq[0] += T{ 1 };
    seriesSqrt< T, Scheme >( h, asq );

    seriesDerivQuotient< 1, T, Scheme >( out, asin( a[0] ), a, h );
}

/// Inverse cosine series `out = acos(a)` (scheme-generic): out' = -a' / sqrt(1 - a^2).
template < typename T, tax::IndexScheme Scheme >
void seriesAcos( std::array< T, Scheme::nCoeff >& out,
                 const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::acos;
    constexpr std::size_t S = Scheme::nCoeff;

    // h = sqrt(1 - a^2)
    std::array< T, S > asq{}, h{};
    tax::cauchySelfProduct< T, Scheme >( asq, a );
    for ( T& v : asq ) v = -v;
    asq[0] += T{ 1 };
    seriesSqrt< T, Scheme >( h, asq );

    seriesDerivQuotient< -1, T, Scheme >( out, acos( a[0] ), a, h );
}

/// Inverse tangent series `out = atan(a)` (scheme-generic): out' = a' / (1 + a^2).
template < typename T, tax::IndexScheme Scheme >
void seriesAtan( std::array< T, Scheme::nCoeff >& out,
                 const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::atan;
    // h = 1 + a^2
    std::array< T, Scheme::nCoeff > h{};
    tax::cauchySelfProduct< T, Scheme >( h, a );
    h[0] += T{ 1 };

    seriesDerivQuotient< 1, T, Scheme >( out, atan( a[0] ), a, h );
}

/// Two-argument arctangent series `out = atan2(y, x)` (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
void seriesAtan2( std::array< T, Scheme::nCoeff >& out, const std::array< T, Scheme::nCoeff >& y,
                  const std::array< T, Scheme::nCoeff >& x ) noexcept
{
    using std::atan2;
    constexpr std::size_t S = Scheme::nCoeff;

    // r = y / x in a single forward-substitution pass, then out' = r' / (1 + r^2).
    std::array< T, S > r{}, h{};
    seriesDivide< T, Scheme >( r, y, x );
    tax::cauchySelfProduct< T, Scheme >( h, r );
    h[0] += T{ 1 };

    seriesDerivQuotient< 1, T, Scheme >( out, atan2( y[0], x[0] ), r, h );
}

}  // namespace tax::detail::kernels
