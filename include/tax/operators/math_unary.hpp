#pragma once

#include <tax/core/taylor_expansion.hpp>
#include <tax/kernels/algebra.hpp>
#include <tax/kernels/sparse_subs.hpp>
#include <tax/kernels/transcendental.hpp>
#include <tax/kernels/trigonometric.hpp>

namespace tax
{

// ===========================================================================
// Dense unary math wrappers
//
// Every dense wrapper has the identical shape — default-construct a result, run
// the matching series kernel on the coefficient arrays, return it — so they are
// generated from a single definition. This keeps the [[nodiscard]] / noexcept
// qualifiers and the constexpr split uniform and prevents drift.
//
//   TAX_UNARY_OP_CE : constexpr-usable (pure polynomial recurrence)
//   TAX_UNARY_OP    : runtime-only (kernel evaluates std::exp/sin/... at the
//                     constant term, which is not constexpr in C++23)
//
// Domain preconditions on the constant term x.value():
//   sqrt        : x0 > 0           reciprocal : x0 != 0
//   cbrt        : x0 != 0          log        : x0 > 0
//   acosh       : x0 > 1           atanh/asin/acos : |x0| < 1
// Violating a precondition yields inf/nan (the dense kernels do not throw).
// ===========================================================================

#define TAX_UNARY_OP_CE( NAME, KERNEL )                                           \
    template < typename T, int N, int M >                                         \
    [[nodiscard]] constexpr TaylorExpansion< T, N, M > NAME(                      \
        const TaylorExpansion< T, N, M >& x ) noexcept                            \
    {                                                                             \
        TaylorExpansion< T, N, M > r;                                             \
        detail::kernels::KERNEL< T, N, M >( r.coefficients(), x.coefficients() ); \
        return r;                                                                 \
    }

#define TAX_UNARY_OP( NAME, KERNEL )                                                              \
    template < typename T, int N, int M >                                                         \
    [[nodiscard]] TaylorExpansion< T, N, M > NAME( const TaylorExpansion< T, N, M >& x ) noexcept \
    {                                                                                             \
        TaylorExpansion< T, N, M > r;                                                             \
        detail::kernels::KERNEL< T, N, M >( r.coefficients(), x.coefficients() );                 \
        return r;                                                                                 \
    }

// Pure-polynomial recurrences (constexpr).
TAX_UNARY_OP_CE( square, seriesSquare )
TAX_UNARY_OP_CE( cube, seriesCube )
TAX_UNARY_OP_CE( reciprocal, seriesReciprocal )

// Roots.
TAX_UNARY_OP( sqrt, seriesSqrt )
TAX_UNARY_OP( cbrt, seriesCbrt )

// Exponential / logarithm.
TAX_UNARY_OP( exp, seriesExp )
TAX_UNARY_OP( log, seriesLog )

// Hyperbolic and inverse-hyperbolic.
TAX_UNARY_OP( sinh, seriesSinh )
TAX_UNARY_OP( cosh, seriesCosh )
TAX_UNARY_OP( tanh, seriesTanh )
TAX_UNARY_OP( asinh, seriesAsinh )
TAX_UNARY_OP( acosh, seriesAcosh )
TAX_UNARY_OP( atanh, seriesAtanh )

// Error function.
TAX_UNARY_OP( erf, seriesErf )

// Trigonometric and inverse-trigonometric.
TAX_UNARY_OP( sin, seriesSin )
TAX_UNARY_OP( cos, seriesCos )
TAX_UNARY_OP( tan, seriesTan )
TAX_UNARY_OP( asin, seriesAsin )
TAX_UNARY_OP( acos, seriesAcos )
TAX_UNARY_OP( atan, seriesAtan )

#undef TAX_UNARY_OP
#undef TAX_UNARY_OP_CE

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
