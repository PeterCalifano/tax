#pragma once

#include <cmath>
#include <numbers>

#include <tax/kernels/algebra.hpp>

namespace tax::detail::kernels
{

/**
 * @brief Natural exponential series `out = exp(a)`.
 *
 * Degree-by-degree recurrence derived from differentiating `out = exp(a)`:
 *   d * out[d] = sum_{k=0}^{d-1} (d-k) * a[d-k] * out[k]
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
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
    }
    else
    {
        for ( int d = 1; d <= N; ++d )
        {
            tax::forEachMonomialOfDegree< M >( d, [&]( const MultiIndex< M >& alpha ) {
                const std::size_t ai = flatIndex< M >( alpha );
                T rhs = T{ 0 };
                // sum over all (beta, gamma) with beta + gamma = alpha, |beta| in [1, d]
                tax::forEachSubIndex< M >( alpha, [&]( const MultiIndex< M >& beta,
                                                        const MultiIndex< M >& gamma ) {
                    int db = 0;
                    for ( int i = 0; i < M; ++i ) db += beta[std::size_t( i )];
                    if ( db == 0 ) return;  // skip beta == 0
                    const std::size_t bi = flatIndex< M >( beta );
                    const std::size_t gi = flatIndex< M >( gamma );
                    rhs += T( db ) * a[bi] * out[gi];
                } );
                out[ai] = rhs / T( d );
            } );
        }
    }
}

/**
 * @brief Natural logarithm series `out = log(a)`.
 *
 * Requires `a[0] > 0`. Degree-by-degree recurrence derived from
 * differentiating `a = exp(out)`:
 *   out[d] = (1 / a[0]) * (a[d] - (1/d) * sum_{k=1}^{d-1} k * a[d-k] * out[k])
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
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
    }
    else
    {
        for ( int d = 1; d <= N; ++d )
        {
            tax::forEachMonomialOfDegree< M >( d, [&]( const MultiIndex< M >& alpha ) {
                const std::size_t ai = flatIndex< M >( alpha );
                T rhs = T{ 0 };
                // sum over (beta, gamma) with |beta| in [1, d-1]
                tax::forEachSubIndex< M >( alpha, [&]( const MultiIndex< M >& beta,
                                                        const MultiIndex< M >& gamma ) {
                    int db = 0;
                    for ( int i = 0; i < M; ++i ) db += beta[std::size_t( i )];
                    if ( db == 0 || db == d ) return;  // skip beta==0 and beta==alpha
                    const std::size_t bi = flatIndex< M >( beta );
                    const std::size_t gi = flatIndex< M >( gamma );
                    rhs += T( d - db ) * a[bi] * out[gi];
                } );
                out[ai] = ( a[ai] - rhs / T( d ) ) * inv_a0;
            } );
        }
    }
}

/**
 * @brief Hyperbolic sine series `out = sinh(a)`.
 *
 * Computed via the identity `sinh(a) = (exp(a) - exp(-a)) / 2`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
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
    for ( std::size_t i = 1; i < S; ++i )
        out[i] = ( ep[i] - em[i] ) * T{ 0.5 };
}

/**
 * @brief Hyperbolic cosine series `out = cosh(a)`.
 *
 * Computed via the identity `cosh(a) = (exp(a) + exp(-a)) / 2`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
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
    for ( std::size_t i = 1; i < S; ++i )
        out[i] = ( ep[i] + em[i] ) * T{ 0.5 };
}

/**
 * @brief Hyperbolic tangent series `out = tanh(a)`.
 *
 * Degree-by-degree recurrence derived from the identity `cosh(a) * tanh(a) = sinh(a)`:
 *   cosh[0] * out[d] = sinh[d] - sum_{k=1}^{d} cosh[k] * out[d-k]
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
void seriesTanh( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    using std::tanh;
    constexpr std::size_t S = numMonomials( N, M );

    // h = cosh(a),  s = sinh(a)
    Coeffs< T, N, M > h{}, s{};
    seriesCosh< T, N, M >( h, a );
    seriesSinh< T, N, M >( s, a );

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
            for ( int k = 1; k <= d; ++k )
                rhs -= h[std::size_t( k )] * out[std::size_t( d - k )];
            out[std::size_t( d )] = rhs * inv_h0;
        }
    }
    else
    {
        for ( int d = 1; d <= N; ++d )
        {
            tax::forEachMonomialOfDegree< M >( d, [&]( const MultiIndex< M >& alpha ) {
                const std::size_t ai = flatIndex< M >( alpha );
                T rhs = s[ai];
                // sum over all beta with |beta| in [1, d]
                tax::forEachSubIndex< M >( alpha, [&]( const MultiIndex< M >& beta,
                                                        const MultiIndex< M >& gamma ) {
                    int db = 0;
                    for ( int i = 0; i < M; ++i ) db += beta[std::size_t( i )];
                    if ( db == 0 ) return;  // skip beta == 0
                    const std::size_t bi = flatIndex< M >( beta );
                    const std::size_t gi = flatIndex< M >( gamma );
                    rhs -= h[bi] * out[gi];
                } );
                out[ai] = rhs * inv_h0;
            } );
        }
    }
}

/**
 * @brief Inverse hyperbolic sine series `out = asinh(a)`.
 *
 * Recurrence derived from `sqrt(1+a^2) * out' = a'`:
 *   h = sqrt(1+a^2),  h[0]*d*out[d] = d*a[d] - sum_{k=1}^{d-1} k*h[d-k]*out[k]
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
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
    }
    else
    {
        for ( int d = 1; d <= N; ++d )
        {
            tax::forEachMonomialOfDegree< M >( d, [&]( const MultiIndex< M >& alpha ) {
                const std::size_t ai = flatIndex< M >( alpha );
                T rhs = T{ 0 };
                tax::forEachSubIndex< M >( alpha, [&]( const MultiIndex< M >& beta,
                                                        const MultiIndex< M >& gamma ) {
                    int db = 0;
                    for ( int i = 0; i < M; ++i ) db += beta[std::size_t( i )];
                    if ( db == 0 || db == d ) return;  // skip beta==0 and beta==alpha
                    const std::size_t bi = flatIndex< M >( beta );
                    const std::size_t gi = flatIndex< M >( gamma );
                    rhs += T( d - db ) * h[bi] * out[gi];
                } );
                out[ai] = ( a[ai] - rhs / T( d ) ) * inv_h0;
            } );
        }
    }
}

/**
 * @brief Inverse hyperbolic cosine series `out = acosh(a)`.
 *
 * Recurrence derived from `sqrt(a^2-1) * out' = a'`. Requires `a[0] > 1`.
 *   h = sqrt(a^2-1),  h[0]*d*out[d] = d*a[d] - sum_{k=1}^{d-1} k*h[d-k]*out[k]
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
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
    }
    else
    {
        for ( int d = 1; d <= N; ++d )
        {
            tax::forEachMonomialOfDegree< M >( d, [&]( const MultiIndex< M >& alpha ) {
                const std::size_t ai = flatIndex< M >( alpha );
                T rhs = T{ 0 };
                tax::forEachSubIndex< M >( alpha, [&]( const MultiIndex< M >& beta,
                                                        const MultiIndex< M >& gamma ) {
                    int db = 0;
                    for ( int i = 0; i < M; ++i ) db += beta[std::size_t( i )];
                    if ( db == 0 || db == d ) return;  // skip beta==0 and beta==alpha
                    const std::size_t bi = flatIndex< M >( beta );
                    const std::size_t gi = flatIndex< M >( gamma );
                    rhs += T( d - db ) * h[bi] * out[gi];
                } );
                out[ai] = ( a[ai] - rhs / T( d ) ) * inv_h0;
            } );
        }
    }
}

/**
 * @brief Inverse hyperbolic tangent series `out = atanh(a)`.
 *
 * Recurrence derived from `(1-a^2) * out' = a'`. Requires `|a[0]| < 1`.
 *   h = 1 - a^2,  h[0]*d*out[d] = d*a[d] - sum_{k=1}^{d-1} k*h[d-k]*out[k]
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
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
    }
    else
    {
        for ( int d = 1; d <= N; ++d )
        {
            tax::forEachMonomialOfDegree< M >( d, [&]( const MultiIndex< M >& alpha ) {
                const std::size_t ai = flatIndex< M >( alpha );
                T rhs = T{ 0 };
                tax::forEachSubIndex< M >( alpha, [&]( const MultiIndex< M >& beta,
                                                        const MultiIndex< M >& gamma ) {
                    int db = 0;
                    for ( int i = 0; i < M; ++i ) db += beta[std::size_t( i )];
                    if ( db == 0 || db == d ) return;  // skip beta==0 and beta==alpha
                    const std::size_t bi = flatIndex< M >( beta );
                    const std::size_t gi = flatIndex< M >( gamma );
                    rhs += T( d - db ) * h[bi] * out[gi];
                } );
                out[ai] = ( a[ai] - rhs / T( d ) ) * inv_h0;
            } );
        }
    }
}

/**
 * @brief Error function series `out = erf(a)`.
 *
 * Recurrence derived from `erf' = (2/sqrt(pi)) * exp(-a^2) * a'`:
 *   Let h = (2/sqrt(pi)) * exp(-a^2), then
 *   d * out[d] = sum_{k=0}^{d-1} (d-k) * a[d-k] * h[k]
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
void seriesErf( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    using std::erf;
    constexpr std::size_t S = numMonomials( N, M );
    constexpr T two_over_sqrtpi = T{ 2 } * std::numbers::inv_sqrtpi_v< T >;

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
    }
    else
    {
        for ( int d = 1; d <= N; ++d )
        {
            tax::forEachMonomialOfDegree< M >( d, [&]( const MultiIndex< M >& alpha ) {
                const std::size_t ai = flatIndex< M >( alpha );
                T rhs = T{ 0 };
                // sum over (beta, gamma) with |beta| in [1, d]
                tax::forEachSubIndex< M >( alpha, [&]( const MultiIndex< M >& beta,
                                                        const MultiIndex< M >& gamma ) {
                    int db = 0;
                    for ( int i = 0; i < M; ++i ) db += beta[std::size_t( i )];
                    if ( db == 0 ) return;  // skip beta == 0
                    const std::size_t bi = flatIndex< M >( beta );
                    const std::size_t gi = flatIndex< M >( gamma );
                    rhs += T( db ) * a[bi] * h[gi];
                } );
                out[ai] = rhs / T( d );
            } );
        }
    }
}

}  // namespace tax::detail::kernels
