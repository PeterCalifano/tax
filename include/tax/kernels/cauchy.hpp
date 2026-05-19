#pragma once

#include <tax/core/enumeration.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/storage/dense.hpp>

#if TAX_USE_UNROLL
#    include <tax/kernels/cauchy_unroll.hpp>
#endif
#if TAX_USE_STENCIL
#    include <tax/kernels/cauchy_stencil.hpp>
#endif

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
 * Routes to the appropriate fast-path variant when enabled:
 *   - TAX_USE_UNROLL=1: uses the fully-unrolled template for M=1
 *   - TAX_USE_STENCIL=1: uses the precomputed stencil table for M>=2
 * Falls through to the loop variant when neither flag is set.
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
#if TAX_USE_UNROLL
    if constexpr ( M == 1 )
    {
        cauchyProductUnroll< T, N, M >( out, a, b );
        return;
    }
#endif
#if TAX_USE_STENCIL
    if constexpr ( M >= 2 )
    {
        cauchyProductStencil< T, N, M >( out, a, b );
        return;
    }
#endif
    cauchyProductLoop< T, N, M >( out, a, b );
}

}  // namespace tax::detail::kernels
