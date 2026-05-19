#pragma once

#include <tax/core/taylor_expansion.hpp>
#include <tax/kernels/algebra.hpp>

namespace tax
{

/**
 * @brief Compute `out = x^2` using the symmetric self-product kernel.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > square(
    const TaylorExpansion< T, N, M >& x ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesSquare< T, N, M >( r.coefficients(), x.coefficients() );
    return r;
}

/**
 * @brief Compute `out = x^3` via two Cauchy products.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > cube(
    const TaylorExpansion< T, N, M >& x ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesCube< T, N, M >( r.coefficients(), x.coefficients() );
    return r;
}

}  // namespace tax
