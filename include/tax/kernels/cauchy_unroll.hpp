#pragma once

#include <utility>
#include <tax/core/multi_index.hpp>

namespace tax::detail::kernels
{

/**
 * @brief Compute the single output coefficient at degree D for the univariate
 *        Cauchy product using compile-time pack expansion over 0..D.
 *
 * out[D] = sum_{k=0}^{D} a[k] * b[D-k]
 */
template < typename T, int N, std::size_t D, std::size_t... Ks >
constexpr T cauchyUniRow( const Coeffs< T, N, 1 >& a, const Coeffs< T, N, 1 >& b,
                          std::index_sequence< Ks... > ) noexcept
{
    return ( ( a[Ks] * b[D - Ks] ) + ... + T{} );
}

/**
 * @brief Compute all output coefficients of the univariate Cauchy product using
 *        nested compile-time pack expansion over degrees 0..N.
 */
template < typename T, int N, std::size_t... Ds >
constexpr void cauchyUniImpl( Coeffs< T, N, 1 >& out, const Coeffs< T, N, 1 >& a,
                               const Coeffs< T, N, 1 >& b,
                               std::index_sequence< Ds... > ) noexcept
{
    ( ( out[Ds] = cauchyUniRow< T, N, Ds >( a, b, std::make_index_sequence< Ds + 1 >{} ) ),
      ... );
}

/**
 * @brief Fully unrolled Cauchy product for M=1 (univariate).
 *
 * All loop bounds are compile-time constants, enabling the compiler to
 * fully unroll the inner convolution.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables (must be 1).
 */
template < typename T, int N, int M >
constexpr void cauchyProductUnroll( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a,
                                    const Coeffs< T, N, M >& b ) noexcept
    requires( M == 1 )
{
    cauchyUniImpl< T, N >( out, a, b, std::make_index_sequence< static_cast< std::size_t >( N ) + 1 >{} );
}

}  // namespace tax::detail::kernels
