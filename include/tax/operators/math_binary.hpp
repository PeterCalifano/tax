#pragma once

#include <tax/core/taylor_expansion.hpp>
#include <tax/kernels/algebra.hpp>

namespace tax
{

/**
 * @brief Compute `out = x^n` for an integer exponent via binary exponentiation.
 *
 * Handles n == 0 (returns 1), n == 1 (returns x), n < 0 (reciprocal then pow),
 * and n >= 2 (binary square-and-multiply).
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > pow(
    const TaylorExpansion< T, N, M >& x, int n ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesPowInt< T, N, M >( r.coefficients(), x.coefficients(), n );
    return r;
}

/**
 * @brief Compute `out = x^p` for a real exponent via logarithmic-differentiation
 *        recurrence.
 *
 * NOT constexpr because `std::pow(a[0], p)` is not constexpr for floating-point
 * types in C++23. Requires `x.value() != 0`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M > pow(
    const TaylorExpansion< T, N, M >& x, T p ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesPow< T, N, M >( r.coefficients(), x.coefficients(), p );
    return r;
}

}  // namespace tax
