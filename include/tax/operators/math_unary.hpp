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

/**
 * @brief Compute `out = sqrt(x)` using the series square-root kernel.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M > sqrt(
    const TaylorExpansion< T, N, M >& x ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesSqrt< T, N, M >( r.coefficients(), x.coefficients() );
    return r;
}

/**
 * @brief Compute `out = cbrt(x)` using the series cubic-root kernel.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M > cbrt(
    const TaylorExpansion< T, N, M >& x ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesCbrt< T, N, M >( r.coefficients(), x.coefficients() );
    return r;
}

/**
 * @brief Compute `out = 1/x` using the series reciprocal kernel.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > reciprocal(
    const TaylorExpansion< T, N, M >& x ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesReciprocal< T, N, M >( r.coefficients(), x.coefficients() );
    return r;
}

}  // namespace tax
