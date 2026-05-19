#pragma once

#include <tax/core/taylor_expansion.hpp>
#include <tax/kernels/algebra.hpp>
#include <tax/kernels/sparse_subs.hpp>
#include <tax/kernels/transcendental.hpp>
#include <tax/kernels/trigonometric.hpp>

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

/**
 * @brief Compute `out = sin(x)` using the trigonometric series kernel.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M > sin(
    const TaylorExpansion< T, N, M >& x ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesSin< T, N, M >( r.coefficients(), x.coefficients() );
    return r;
}

/**
 * @brief Compute `out = cos(x)` using the trigonometric series kernel.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M > cos(
    const TaylorExpansion< T, N, M >& x ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesCos< T, N, M >( r.coefficients(), x.coefficients() );
    return r;
}

/**
 * @brief Compute `out = tan(x)` using the trigonometric series kernel.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M > tan(
    const TaylorExpansion< T, N, M >& x ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesTan< T, N, M >( r.coefficients(), x.coefficients() );
    return r;
}

/**
 * @brief Compute `out = asin(x)` using the inverse-sine series kernel.
 *
 * Requires `|x.value()| < 1`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M > asin(
    const TaylorExpansion< T, N, M >& x ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesAsin< T, N, M >( r.coefficients(), x.coefficients() );
    return r;
}

/**
 * @brief Compute `out = acos(x)` using the inverse-cosine series kernel.
 *
 * Requires `|x.value()| < 1`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M > acos(
    const TaylorExpansion< T, N, M >& x ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesAcos< T, N, M >( r.coefficients(), x.coefficients() );
    return r;
}

/**
 * @brief Compute `out = atan(x)` using the inverse-tangent series kernel.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M > atan(
    const TaylorExpansion< T, N, M >& x ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::seriesAtan< T, N, M >( r.coefficients(), x.coefficients() );
    return r;
}

// ===========================================================================
// Sparse overloads: sqrt, reciprocal
// ===========================================================================

/**
 * @brief Sparse `sqrt(f)` via support-set forward substitution.
 *
 * Constant term must be strictly positive.
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, storage::Sparse > sqrt(
    const TaylorExpansion< T, N, M, storage::Sparse >& x )
{
    TaylorExpansion< T, N, M, storage::Sparse > r;
    detail::kernels::seriesSqrtSparse< T, N, M >( r.container(), x.container() );
    return r;
}

/**
 * @brief Sparse `1/f` via support-set forward substitution.
 *
 * Constant term must be nonzero.
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, storage::Sparse > reciprocal(
    const TaylorExpansion< T, N, M, storage::Sparse >& x )
{
    TaylorExpansion< T, N, M, storage::Sparse > r;
    detail::kernels::seriesReciprocalSparse< T, N, M >( r.container(), x.container() );
    return r;
}

}  // namespace tax
