#pragma once

#include <cmath>
#include <span>
#include <tax/kernels/algebra.hpp>

namespace tax::detail::kernels
{

/**
 * @brief Coupled trigonometric series: jointly compute `sin(a)` and `cos(a)`.
 *
 * Degree-by-degree recurrence derived from differentiating `sin(a)' = cos(a)*a'`
 * and `cos(a)' = -sin(a)*a'`:
 *   d * s[d] = sum_{k=0}^{d-1} (d-k) * a[d-k] * c[k]
 *   d * c[d] = -sum_{k=0}^{d-1} (d-k) * a[d-k] * s[k]
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
void seriesSinCos( Coeffs< T, N, M >& s, Coeffs< T, N, M >& c, const Coeffs< T, N, M >& a ) noexcept
{
    using std::cos;
    using std::sin;
    s = {};
    c = {};
    s[0] = sin( a[0] );
    c[0] = cos( a[0] );

    if constexpr ( M == 1 )
    {
        for ( int d = 1; d <= N; ++d )
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
        forEachRecurrenceRow< N, M >(
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

/**
 * @brief Sine series `out = sin(a)`.
 *
 * Thin wrapper around `seriesSinCos`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
void seriesSin( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    Coeffs< T, N, M > c{};
    seriesSinCos< T, N, M >( out, c, a );
}

/**
 * @brief Cosine series `out = cos(a)`.
 *
 * Thin wrapper around `seriesSinCos`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
void seriesCos( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    Coeffs< T, N, M > s{};
    seriesSinCos< T, N, M >( s, out, a );
}

/**
 * @brief Tangent series `out = tan(a)`.
 *
 * Solves `cos(a) * out = sin(a)` degree-by-degree:
 *   cos[0] * out[d] = sin[d] - sum_{k=1}^{d} cos[k] * out[d-k]
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
void seriesTan( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    constexpr std::size_t S = numMonomials( N, M );
    Coeffs< T, N, M > s{}, c{};
    seriesSinCos< T, N, M >( s, c, a );

    out = {};
    out[0] = s[0] / c[0];
    const T inv_c0 = T{ 1 } / c[0];

    if constexpr ( M == 1 )
    {
        for ( int d = 1; d <= N; ++d )
        {
            T rhs = s[std::size_t( d )];
            for ( int k = 1; k <= d; ++k ) rhs -= c[std::size_t( k )] * out[std::size_t( d - k )];
            out[std::size_t( d )] = rhs * inv_c0;
        }
    } else
    {
        forEachRecurrenceRow< N, M >(
            [&]( std::size_t ai, int, std::span< const RecurrenceEntry > row ) {
                T rhs = s[ai];
                for ( const RecurrenceEntry& e : row ) rhs -= c[e.b_idx] * out[e.g_idx];
                out[ai] = rhs * inv_c0;
            } );
    }
}

/**
 * @brief Inverse sine series `out = asin(a)`.
 *
 * Recurrence derived from `sqrt(1-a^2) * out' = a'`:
 *   h = sqrt(1-a^2),  h[0]*d*out[d] = d*a[d] - sum_{k=1}^{d-1} k*h[d-k]*out[k]
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
void seriesAsin( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    using std::asin;
    constexpr std::size_t S = numMonomials( N, M );

    // h = sqrt(1 - a^2)
    Coeffs< T, N, M > asq{}, omf{}, h{};
    cauchySelfProduct< T, N, M >( asq, a );
    omf = {};
    omf[0] = T{ 1 };
    for ( std::size_t i = 0; i < S; ++i ) omf[i] -= asq[i];
    seriesSqrt< T, N, M >( h, omf );

    out = {};
    out[0] = asin( a[0] );
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

/**
 * @brief Inverse cosine series `out = acos(a)`.
 *
 * Recurrence derived from `sqrt(1-a^2) * out' = -a'`:
 *   h = sqrt(1-a^2),  h[0]*d*out[d] = -d*a[d] - sum_{k=1}^{d-1} k*h[d-k]*out[k]
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
void seriesAcos( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    using std::acos;
    constexpr std::size_t S = numMonomials( N, M );

    // h = sqrt(1 - a^2)
    Coeffs< T, N, M > asq{}, omf{}, h{};
    cauchySelfProduct< T, N, M >( asq, a );
    omf = {};
    omf[0] = T{ 1 };
    for ( std::size_t i = 0; i < S; ++i ) omf[i] -= asq[i];
    seriesSqrt< T, N, M >( h, omf );

    out = {};
    out[0] = acos( a[0] );
    const T inv_h0 = T{ 1 } / h[0];

    if constexpr ( M == 1 )
    {
        for ( int d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 1; k < d; ++k )
                rhs += T( k ) * h[std::size_t( d - k )] * out[std::size_t( k )];
            out[std::size_t( d )] = ( -a[std::size_t( d )] - rhs / T( d ) ) * inv_h0;
        }
    } else
    {
        forEachRecurrenceRow< N, M >(
            [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
                T rhs = T{ 0 };
                // |beta| == d entries carry weight (d - db) == 0.
                for ( const RecurrenceEntry& e : row )
                    rhs += T( d - int( e.db ) ) * h[e.b_idx] * out[e.g_idx];
                out[ai] = ( -a[ai] - rhs / T( d ) ) * inv_h0;
            } );
    }
}

/**
 * @brief Inverse tangent series `out = atan(a)`.
 *
 * Recurrence derived from `(1+a^2) * out' = a'`:
 *   h = 1 + a^2,  h[0]*d*out[d] = d*a[d] - sum_{k=1}^{d-1} k*h[d-k]*out[k]
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
void seriesAtan( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    using std::atan;
    constexpr std::size_t S = numMonomials( N, M );

    // h = 1 + a^2
    Coeffs< T, N, M > asq{}, h{};
    cauchySelfProduct< T, N, M >( asq, a );
    h = asq;
    h[0] += T{ 1 };

    out = {};
    out[0] = atan( a[0] );
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

/**
 * @brief Two-argument arctangent series `out = atan2(y, x)`.
 *
 * For Taylor series, computed as `atan(y/x)` with seed `atan2(y[0], x[0])`:
 *   Let r = y/x, then use the atan recurrence on r but with seed `atan2(y[0], x[0])`.
 *   This resolves the correct quadrant via the constant term.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
void seriesAtan2( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& y,
                  const Coeffs< T, N, M >& x ) noexcept
{
    using std::atan2;
    constexpr std::size_t S = numMonomials( N, M );

    // Compute r = y / x in a single forward-substitution pass.
    Coeffs< T, N, M > r{};
    seriesDivide< T, N, M >( r, y, x );

    // h = 1 + r^2
    Coeffs< T, N, M > rsq{}, h{};
    cauchySelfProduct< T, N, M >( rsq, r );
    h = rsq;
    h[0] += T{ 1 };

    out = {};
    out[0] = atan2( y[0], x[0] );
    const T inv_h0 = T{ 1 } / h[0];

    if constexpr ( M == 1 )
    {
        for ( int d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 1; k < d; ++k )
                rhs += T( k ) * h[std::size_t( d - k )] * out[std::size_t( k )];
            out[std::size_t( d )] = ( r[std::size_t( d )] - rhs / T( d ) ) * inv_h0;
        }
    } else
    {
        forEachRecurrenceRow< N, M >(
            [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
                T rhs = T{ 0 };
                // |beta| == d entries carry weight (d - db) == 0.
                for ( const RecurrenceEntry& e : row )
                    rhs += T( d - int( e.db ) ) * h[e.b_idx] * out[e.g_idx];
                out[ai] = ( r[ai] - rhs / T( d ) ) * inv_h0;
            } );
    }
}

}  // namespace tax::detail::kernels
