// Vector norm (and its powers) of a vector of Taylor expansions
// (tax::la::norm, re-exported under tax::).
//
//   norm<P, Q>(v) = ||v||_P^Q = ( sum_i v_i^P )^{Q/P}
//
// Q defaults to 1 (the plain P-norm), P defaults to the Euclidean 2-norm:
//
//   norm(v)         = sqrt(sum v_i^2)          (Euclidean norm)
//   norm<3>(v)      = ( sum v_i^3 )^{1/3}      (3-norm)
//   norm<2, -3>(v)  = 1 / ||v||^3              (the gravity kernel)
//   norm<2, 2>(v)   = sum v_i^2                (no root at all)
//
// Fused: the power-sum is raised ONCE to Q/P via the compile-time rational
// power pow<Q, P> (rather than root-then-re-raise, two recurrence passes).
//
// Domain: requires sum_i v_i^P > 0 at the constant term. The result equals the
// true P-norm when the summands are non-negative (even P, or components
// positive at the expansion point); `abs` is not smooth, so the odd-P signed
// power-sum is used as-is.
//
// Accepts an Eigen column vector of expansions or any range of expansions.
// Works for dense TE, named NE and mixed-order MTE elements.

#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <ranges>
#include <tax/la/num_traits.hpp>
#include <tax/operators/math_binary.hpp>
#include <tax/operators/math_unary.hpp>
#include <tax/operators/mixed_math.hpp>
#include <tax/operators/named_math_binary.hpp>
#include <tax/operators/named_math_unary.hpp>

namespace tax::la
{

namespace detail
{

/// A Taylor-expansion-like element (TaylorExpansion, NamedTaylorExpansion,
/// MixedTaylorExpansion): has a coefficient count and a scalar type.
template < typename E >
concept ExpansionElement = requires {
    { E::nCoefficients } -> std::convertible_to< std::size_t >;
    typename E::scalar_type;
};

/// Accumulate `s += e^P` in place (P == 2 uses the square kernel directly).
template < int P, typename E >
constexpr void accumPow( E& s, const E& e ) noexcept
{
    static_assert( P >= 1, "norm order P must be >= 1" );
    if constexpr ( P == 2 )
        s += tax::square( e );
    else
        s += tax::pow< P >( e );
}

/// Accumulate `s = sum_i v_i^P` over a range of expansions.
template < int P, typename E, std::ranges::input_range R >
[[nodiscard]] constexpr E powerSum( const R& v ) noexcept
{
    E s{};
    for ( const E& e : v ) accumPow< P >( s, e );
    return s;
}

}  // namespace detail

/// `||v||_P^Q` for a range of expansions (std::array / std::vector / std::span
/// / C-array). `pow<Q, P>` reduces the exponent and binds to the cheapest
/// kernel (integer chain when `P | Q`, sqrt/invsqrt chain when the reduced
/// denominator is 2, otherwise one real-exponent recurrence).
template < int P = 2, int Q = 1, std::ranges::input_range R >
    requires detail::ExpansionElement< std::ranges::range_value_t< R > >
[[nodiscard]] auto norm( const R& v ) noexcept
{
    using E = std::ranges::range_value_t< R >;
    return tax::pow< Q, P >( detail::powerSum< P, E >( v ) );
}

/// `||v||_P^Q` for an Eigen vector of expansions (`Q` defaults to 1, `P` to 2).
template < int P = 2, int Q = 1, typename Derived >
    requires detail::ExpansionElement< typename Derived::Scalar >
[[nodiscard]] auto norm( const Eigen::MatrixBase< Derived >& v ) noexcept
{
    using E = typename Derived::Scalar;
    E s{};
    for ( Eigen::Index i = 0; i < v.size(); ++i )
    {
        const E e = v( i );
        detail::accumPow< P >( s, e );
    }
    return tax::pow< Q, P >( s );
}

}  // namespace tax::la

// The norm helper is reachable directly under `tax`.
namespace tax
{
using la::norm;
}  // namespace tax
