#pragma once

#include <tax/core/taylor_expansion.hpp>
#include <tax/kernels/algebra.hpp>
#include <tax/kernels/transcendental.hpp>

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

/**
 * @brief Compute `out = exp(x)` using the series exponential kernel.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M > exp(
    const TaylorExpansion< T, N, M >& x ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesExp< T, N, M >( r.coefficients(), x.coefficients() );
    return r;
}

/**
 * @brief Compute `out = log(x)` using the series logarithm kernel.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M > log(
    const TaylorExpansion< T, N, M >& x ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesLog< T, N, M >( r.coefficients(), x.coefficients() );
    return r;
}

/**
 * @brief Compute `out = sinh(x)` using the series hyperbolic-sine kernel.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M > sinh(
    const TaylorExpansion< T, N, M >& x ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesSinh< T, N, M >( r.coefficients(), x.coefficients() );
    return r;
}

/**
 * @brief Compute `out = cosh(x)` using the series hyperbolic-cosine kernel.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M > cosh(
    const TaylorExpansion< T, N, M >& x ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesCosh< T, N, M >( r.coefficients(), x.coefficients() );
    return r;
}

/**
 * @brief Compute `out = tanh(x)` using the series hyperbolic-tangent kernel.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M > tanh(
    const TaylorExpansion< T, N, M >& x ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesTanh< T, N, M >( r.coefficients(), x.coefficients() );
    return r;
}

/**
 * @brief Compute `out = asinh(x)` using the inverse-hyperbolic-sine kernel.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M > asinh(
    const TaylorExpansion< T, N, M >& x ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesAsinh< T, N, M >( r.coefficients(), x.coefficients() );
    return r;
}

/**
 * @brief Compute `out = acosh(x)` using the inverse-hyperbolic-cosine kernel.
 *
 * Requires `x.value() > 1`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M > acosh(
    const TaylorExpansion< T, N, M >& x ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesAcosh< T, N, M >( r.coefficients(), x.coefficients() );
    return r;
}

/**
 * @brief Compute `out = atanh(x)` using the inverse-hyperbolic-tangent kernel.
 *
 * Requires `|x.value()| < 1`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M > atanh(
    const TaylorExpansion< T, N, M >& x ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesAtanh< T, N, M >( r.coefficients(), x.coefficients() );
    return r;
}

/**
 * @brief Compute `out = erf(x)` using the error-function series kernel.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M > erf(
    const TaylorExpansion< T, N, M >& x ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesErf< T, N, M >( r.coefficients(), x.coefficients() );
    return r;
}

}  // namespace tax
