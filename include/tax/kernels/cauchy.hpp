#pragma once

#include <tax/core/enumeration.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/storage/dense.hpp>

namespace tax::detail::kernels
{

/**
 * @brief Loop-based Cauchy (convolution) product over graded-lex monomials.
 *
 * Computes:
 *   out[alpha] = sum_{k <= alpha (componentwise)} a[k] * b[alpha - k]
 *
 * for every multi-index alpha with |alpha| <= N.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
constexpr void cauchyProductLoop( Coeffs< T, N, M >& out,
                                  const Coeffs< T, N, M >& a,
                                  const Coeffs< T, N, M >& b ) noexcept
{
    out = {};
    tax::forEachMonomial< M, N >( [&]( const MultiIndex< M >& alpha ) {
        const std::size_t i = flatIndex< M >( alpha );
        tax::forEachSubIndex< M >( alpha, [&]( const MultiIndex< M >& k,
                                               const MultiIndex< M >& s ) {
            out[i] += a[flatIndex< M >( k )] * b[flatIndex< M >( s )];
        } );
    } );
}

/**
 * @brief Public dispatch entry for the Cauchy product.
 *
 * Currently delegates to the loop variant. Fast paths (univariate,
 * symmetric self-product) will be added in Task 4.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
constexpr void cauchyProduct( Coeffs< T, N, M >& out,
                               const Coeffs< T, N, M >& a,
                               const Coeffs< T, N, M >& b ) noexcept
{
    cauchyProductLoop< T, N, M >( out, a, b );
}

}  // namespace tax::detail::kernels
