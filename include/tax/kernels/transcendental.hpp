#pragma once

#include <cmath>

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

}  // namespace tax::detail::kernels
