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

}  // namespace tax::detail::kernels
