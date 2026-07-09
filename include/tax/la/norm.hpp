// Vector norms of a vector of Taylor expansions (tax::la::norm / normPow),
// re-exported under tax::.
//
//   norm<P>(v)       = ( sum_i v_i^P )^{1/P}          — the P-norm as a TE
//   normPow<P,Q>(v)  = ||v||_P^Q = ( sum_i v_i^P )^{Q/P}
//
// normPow is the *fused* form: it raises the accumulated power-sum ONCE to
// Q/P, rather than taking the P-th root and re-raising to Q (two recurrence
// passes collapse to one). For P == 2 it binds to halfPow<Q> / invSqrtPow, so
//
//   norm<2>(v)         = sqrt(sum v_i^2)          (Euclidean norm)
//   normPow<2,-3>(v)   = 1 / ||v||^3              (the gravity kernel)
//   normPow<2,2>(v)    = sum v_i^2                (no root at all)
//
// Domain: requires sum_i v_i^P > 0 at the expansion point (the constant term).
// The result equals the true P-norm when the summands are non-negative — even
// P, or a vector whose components are positive at the expansion point. `abs`
// is not smooth, so the odd-P signed power-sum is used as-is (document per
// call site if the true 1-norm/odd-norm is intended).
//
// Accepts an Eigen column vector of expansions (as produced by
// tax::la::variables) or any range of expansions (e.g. the std::array from
// tax::variables). Works for dense TE, named NE and mixed-order MTE elements.

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

/// A Taylor-expansion-like element: has a coefficient count and a scalar type
/// (satisfied by TaylorExpansion, NamedTaylorExpansion, MixedTaylorExpansion).
template < typename E >
concept ExpansionElement = requires {
    { E::nCoefficients } -> std::convertible_to< std::size_t >;
    typename E::scalar_type;
};

/// Accumulate `s += e^P` in place, `e^P` via the compile-time integer power
/// (P == 2 takes the symmetric square kernel directly). `+=` avoids the extra
/// temporary of `s = s + ...`.
template < int P, typename E >
constexpr void accumPow( E& s, const E& e ) noexcept
{
    static_assert( P >= 1, "norm order P must be >= 1" );
    if constexpr ( P == 2 )
        s += tax::square( e );
    else
        s += tax::pow< P >( e );
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Range overload (std::array / std::vector / std::span / C-array of expansions)
// ---------------------------------------------------------------------------

/// `||v||_P^Q` for a range of expansions. The power-sum `s = sum_i v_i^P` is
/// accumulated in place, then raised ONCE to `Q/P` via the compile-time
/// rational power `pow<Q, P>` — which reduces the exponent and binds to the
/// cheapest kernel (integer chain when `P | Q`, the sqrt/invsqrt chain when the
/// reduced denominator is 2, otherwise one real-exponent recurrence). Taking
/// the root and re-raising would cost a second recurrence pass.
template < int P, int Q, std::ranges::input_range R >
    requires detail::ExpansionElement< std::ranges::range_value_t< R > >
[[nodiscard]] auto normPow( const R& v ) noexcept
{
    using E = std::ranges::range_value_t< R >;
    E s{};
    for ( const E& e : v ) detail::accumPow< P >( s, e );
    return tax::pow< Q, P >( s );
}

/// `||v||_P` (the P-norm; P defaults to the Euclidean 2-norm) for a range.
template < int P = 2, std::ranges::input_range R >
    requires detail::ExpansionElement< std::ranges::range_value_t< R > >
[[nodiscard]] auto norm( const R& v ) noexcept
{
    return normPow< P, 1 >( v );
}

// ---------------------------------------------------------------------------
// Eigen column-vector overload
// ---------------------------------------------------------------------------

/// `||v||_P^Q` for an Eigen vector of expansions.
template < int P, int Q, typename Derived >
    requires detail::ExpansionElement< typename Derived::Scalar >
[[nodiscard]] auto normPow( const Eigen::MatrixBase< Derived >& v ) noexcept
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

/// `||v||_P` (the P-norm; P defaults to the Euclidean 2-norm) for an Eigen vector.
template < int P = 2, typename Derived >
    requires detail::ExpansionElement< typename Derived::Scalar >
[[nodiscard]] auto norm( const Eigen::MatrixBase< Derived >& v ) noexcept
{
    return normPow< P, 1 >( v );
}

}  // namespace tax::la

// The norm helpers are reachable directly under `tax`.
namespace tax
{
using la::norm;
using la::normPow;
}  // namespace tax
