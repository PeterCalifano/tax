#pragma once

#include <array>
#include <cmath>

#include <tax/kernels/cauchy.hpp>

namespace tax::detail::kernels
{

/**
 * @brief Symmetric self-product `out = f * f`, exploiting symmetry for ~2x fewer multiplications.
 *
 * Enumerates each unordered pair {beta, gamma} with beta+gamma=alpha only once,
 * doubling the off-diagonal contribution.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
constexpr void cauchySelfProduct( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& f ) noexcept
{
    out = {};
    if constexpr ( M == 1 )
    {
        for ( int d = 0; d <= N; ++d )
        {
            for ( int k = 0; k + k < d; ++k )
                out[std::size_t( d )] += T{ 2 } * f[std::size_t( k )] * f[std::size_t( d - k )];
            if ( d % 2 == 0 )
                out[std::size_t( d )] +=
                    f[std::size_t( d / 2 )] * f[std::size_t( d / 2 )];
        }
    } else
    {
        tax::forEachMonomial< M, N >( [&]( const MultiIndex< M >& alpha ) {
            const std::size_t ai = flatIndex< M >( alpha );
            tax::forEachSubIndex< M >( alpha, [&]( const MultiIndex< M >& beta,
                                                   const MultiIndex< M >& gamma ) {
                const std::size_t bi = flatIndex< M >( beta );
                const std::size_t gi = flatIndex< M >( gamma );
                if ( bi < gi )
                    out[ai] += T{ 2 } * f[bi] * f[gi];
                else if ( bi == gi )
                    out[ai] += f[bi] * f[bi];
            } );
        } );
    }
}

/**
 * @brief Square series `out = a^2` using the symmetric self-product.
 *
 * Uses `cauchySelfProduct` which saves ~half the multiplications vs a general
 * Cauchy product call.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
constexpr void seriesSquare( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    cauchySelfProduct< T, N, M >( out, a );
}

/**
 * @brief Cube series `out = a^3` via two Cauchy products.
 *
 * Computes `tmp = a^2` (via symmetric self-product), then `out = tmp * a`.
 * O(N^2) for M=1, O(S^2) for M>1.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
constexpr void seriesCube( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    constexpr auto S = numMonomials( N, M );
    std::array< T, S > tmp{};
    cauchySelfProduct< T, N, M >( tmp, a );
    cauchyProduct< T, N, M >( out, tmp, a );
}

/**
 * @brief Reciprocal series solve `a * out = 1`.
 *
 * Requires `a[0] != 0`. Degree-by-degree forward substitution:
 *   out[alpha] = -(1/a[0]) * sum_{0 < beta <= alpha} a[beta] * out[alpha - beta]
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
constexpr void seriesReciprocal( Coeffs< T, N, M >& out,
                                 const Coeffs< T, N, M >& a ) noexcept
{
    out = {};
    const T inv_a0 = T{ 1 } / a[0];
    out[0] = inv_a0;

    if constexpr ( M == 1 )
    {
        for ( int d = 1; d <= N; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 1; k <= d; ++k ) rhs -= a[std::size_t( k )] * out[std::size_t( d - k )];
            out[std::size_t( d )] = rhs * inv_a0;
        }
    }
    else
    {
        for ( int d = 1; d <= N; ++d )
        {
            tax::forEachMonomialOfDegree< M >( d, [&]( const MultiIndex< M >& alpha ) {
                const std::size_t ai = flatIndex< M >( alpha );
                T rhs = T{ 0 };
                // sum over all beta with 0 < |beta| <= d (i.e., beta != 0 and gamma != 0... actually
                // beta in [1..d] means we skip beta=0; gamma=alpha-beta has |gamma|=d-|beta| in [0,d-1])
                tax::forEachSubIndex< M >( alpha, [&]( const MultiIndex< M >& beta,
                                                        const MultiIndex< M >& gamma ) {
                    // skip beta == 0 (that term is a[0]*out[alpha], moved to LHS)
                    int beta_deg = 0;
                    for ( int i = 0; i < M; ++i ) beta_deg += beta[std::size_t( i )];
                    if ( beta_deg == 0 ) return;
                    const std::size_t bi = flatIndex< M >( beta );
                    const std::size_t gi = flatIndex< M >( gamma );
                    rhs -= a[bi] * out[gi];
                } );
                out[ai] = rhs * inv_a0;
            } );
        }
    }
}

/**
 * @brief Square-root series solve `out * out = a`.
 *
 * Uses the principal branch from `sqrt(a[0])`. Requires `a[0] > 0`.
 * Recurrence at degree d:
 *   out[d] = (1 / (2*out[0])) * (a[d] - sum_{0 < k < d} out[k]*out[d-k])
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
void seriesSqrt( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    using std::sqrt;
    out = {};
    out[0] = sqrt( a[0] );
    const T inv2g0 = T{ 1 } / ( T{ 2 } * out[0] );

    if constexpr ( M == 1 )
    {
        for ( int d = 1; d <= N; ++d )
        {
            T rhs = a[std::size_t( d )];
            for ( int k = 1; k + k < d; ++k )
                rhs -= T{ 2 } * out[std::size_t( k )] * out[std::size_t( d - k )];
            if ( d % 2 == 0 )
                rhs -= out[std::size_t( d / 2 )] * out[std::size_t( d / 2 )];
            out[std::size_t( d )] = rhs * inv2g0;
        }
    }
    else
    {
        for ( int d = 1; d <= N; ++d )
        {
            tax::forEachMonomialOfDegree< M >( d, [&]( const MultiIndex< M >& alpha ) {
                const std::size_t ai = flatIndex< M >( alpha );
                T rhs = a[ai];
                // Subtract off-diagonal and diagonal cross terms (beta in [1, d-1])
                tax::forEachSubIndex< M >( alpha, [&]( const MultiIndex< M >& beta,
                                                        const MultiIndex< M >& gamma ) {
                    int beta_deg = 0;
                    for ( int i = 0; i < M; ++i ) beta_deg += beta[std::size_t( i )];
                    if ( beta_deg == 0 || beta_deg == d ) return;
                    const std::size_t bi = flatIndex< M >( beta );
                    const std::size_t gi = flatIndex< M >( gamma );
                    if ( bi < gi )
                        rhs -= T{ 2 } * out[bi] * out[gi];
                    else if ( bi == gi )
                        rhs -= out[bi] * out[bi];
                } );
                out[ai] = rhs * inv2g0;
            } );
        }
    }
}

/**
 * @brief Cubic-root series solve `out * out * out = a`.
 *
 * Uses the real branch from `cbrt(a[0])`. Requires `a[0] != 0`.
 * Maintains `sq = out^2` incrementally for O(N^2) total work (M=1).
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
void seriesCbrt( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    using std::cbrt;
    constexpr auto S = numMonomials( N, M );

    out = {};
    out[0] = cbrt( a[0] );
    const T inv3g0sq = T{ 1 } / ( T{ 3 } * out[0] * out[0] );

    if constexpr ( M == 1 )
    {
        std::array< T, S > sq{};
        sq[0] = out[0] * out[0];
        for ( int d = 1; d <= N; ++d )
        {
            T sq_d_partial = T{ 0 };
            for ( int k = 1; k + k < d; ++k )
                sq_d_partial += T{ 2 } * out[std::size_t( k )] * out[std::size_t( d - k )];
            if ( d % 2 == 0 )
                sq_d_partial += out[std::size_t( d / 2 )] * out[std::size_t( d / 2 )];

            T rhs = out[0] * sq_d_partial;
            for ( int j = 1; j < d; ++j ) rhs += out[std::size_t( j )] * sq[std::size_t( d - j )];

            out[std::size_t( d )] = ( a[std::size_t( d )] - rhs ) * inv3g0sq;
            sq[std::size_t( d )] = T{ 2 } * out[0] * out[std::size_t( d )] + sq_d_partial;
        }
    }
    else
    {
        std::array< T, S > sq{};
        sq[0] = out[0] * out[0];
        for ( int d = 1; d <= N; ++d )
        {
            tax::forEachMonomialOfDegree< M >( d, [&]( const MultiIndex< M >& alpha ) {
                const std::size_t ai = flatIndex< M >( alpha );
                T rhs = a[ai];
                // beta in [1, d-1]: skip beta==0 and beta with |beta|==d
                tax::forEachSubIndex< M >( alpha, [&]( const MultiIndex< M >& beta,
                                                        const MultiIndex< M >& gamma ) {
                    int beta_deg = 0;
                    for ( int i = 0; i < M; ++i ) beta_deg += beta[std::size_t( i )];
                    if ( beta_deg == 0 || beta_deg == d ) return;
                    const std::size_t bi = flatIndex< M >( beta );
                    const std::size_t gi = flatIndex< M >( gamma );
                    rhs -= out[bi] * ( out[0] * out[gi] + sq[gi] );
                } );
                out[ai] = rhs * inv3g0sq;
            } );

            tax::forEachMonomialOfDegree< M >( d, [&]( const MultiIndex< M >& alpha ) {
                const std::size_t ai = flatIndex< M >( alpha );
                T val = T{ 0 };
                tax::forEachSubIndex< M >( alpha, [&]( const MultiIndex< M >& beta,
                                                        const MultiIndex< M >& gamma ) {
                    const std::size_t bi = flatIndex< M >( beta );
                    const std::size_t gi = flatIndex< M >( gamma );
                    if ( bi < gi )
                        val += T{ 2 } * out[bi] * out[gi];
                    else if ( bi == gi )
                        val += out[bi] * out[bi];
                } );
                sq[ai] = val;
            } );
        }
    }
}

/**
 * @brief Real-exponent power series `out = a^c` via degree-by-degree recurrence.
 *
 * Derived from logarithmic differentiation of `f = a^c`:
 *   d * a[0] * f[d] = sum_{k=0}^{d-1} (c*(d-k) - k) * a[d-k] * f[k]
 *
 * Multivariate generalisation with |alpha|=d:
 *   d * a[0] * f[alpha] = sum_{0 < |beta| <= d}
 *                           (c*|beta| - (d-|beta|)) * a[beta] * f[alpha-beta]
 *
 * Requires `a[0] != 0`. NOT constexpr because it calls `std::pow`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
void seriesPow( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a, T c ) noexcept
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
            for ( int k = 0; k < d; ++k )
                rhs += ( c * T( d - k ) - T( k ) ) * a[std::size_t( d - k )] * out[std::size_t( k )];
            out[std::size_t( d )] = rhs * inv_a0 / T( d );
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
                    // Need |beta| > 0 (skip the beta=0 term)
                    int db = 0;
                    for ( int i = 0; i < M; ++i ) db += beta[std::size_t( i )];
                    if ( db == 0 ) return;
                    const std::size_t bi = flatIndex< M >( beta );
                    const std::size_t gi = flatIndex< M >( gamma );
                    rhs += ( c * T( db ) - T( d - db ) ) * a[bi] * out[gi];
                } );
                out[ai] = rhs * inv_a0 / T( d );
            } );
        }
    }
}

/**
 * @brief Integer-exponent power series `out = a^n` via binary exponentiation.
 *
 * Special cases handled directly:
 *   - n == 0  → constant 1
 *   - n == 1  → copy of a
 *   - n == -1 → seriesReciprocal(a)
 *   - n <  0  → seriesReciprocal(a), then seriesPowInt(rec, -n)
 *   - n >= 2  → binary exponentiation using cauchyProduct
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
constexpr void seriesPowInt( Coeffs< T, N, M >& out,
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
        seriesPowInt< T, N, M >( out, rec, -n );
        return;
    }
    // n >= 2: binary exponentiation (left-to-right square-and-multiply)
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

}  // namespace tax::detail::kernels
