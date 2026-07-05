// include/tax/la/mixed_named.hpp
//
// Eigen integration for the mixed named-axis layer (tax::named::MixedTaylorExpansion):
//
//   1. Eigen::NumTraits< tax::named::MixedTaylorExpansion<...> > — lets a mixed
//      named expansion act as a first-class Eigen scalar, so
//      Eigen::Matrix< MixedTaylorExpansion<...>, D, 1 > works.
//
//   2. Per-axis differential helpers in namespace tax::named:
//        gradient<"axis">(f)  — gradient restricted to one named axis block.
//        hessian <"axis">(f)  — Hessian restricted to one named axis block.
//        jacobian<"axis">(F)  — Jacobian of a vector of mixed named expansions
//                               with respect to one named axis block.
//
// These mirror tax::la::{gradient,hessian,jacobian} but slice the result down to
// the variables of a single named ordered axis.  The implementation mirrors
// include/tax/la/named.hpp (which does the same for NamedTaylorExpansion).

#pragma once

#include <Eigen/Core>
#include <tax/core/mixed_named.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/named.hpp>
#include <tax/la/named.hpp>

// -----------------------------------------------------------------------------
// NumTraits specialization — namespace Eigen
// -----------------------------------------------------------------------------

namespace Eigen
{

template < typename T, typename... Axes >
struct NumTraits< tax::named::MixedTaylorExpansion< T, Axes... > >
    : NumTraits< typename tax::named::MixedTaylorExpansion< T, Axes... >::Inner >
{
    using Self = tax::named::MixedTaylorExpansion< T, Axes... >;
    using Real = Self;
    using NonInteger = Self;
    using Nested = Self;

    static inline Self epsilon() { return Self( NumTraits< T >::epsilon() ); }
    static inline Self dummy_precision() { return Self( NumTraits< T >::dummy_precision() ); }
    static inline Self highest() { return Self( NumTraits< T >::highest() ); }
    static inline Self lowest() { return Self( NumTraits< T >::lowest() ); }
    static inline Self infinity() { return Self( NumTraits< T >::infinity() ); }
    static inline Self quiet_NaN() { return Self( NumTraits< T >::quiet_NaN() ); }
};

}  // namespace Eigen

namespace tax::named
{

// -----------------------------------------------------------------------------
// is_mixed — type trait used to constrain the generic jacobian overload
// -----------------------------------------------------------------------------

namespace detail
{

template < typename >
struct is_mixed : std::false_type
{
};
template < typename T, typename... Axes >
struct is_mixed< MixedTaylorExpansion< T, Axes... > > : std::true_type
{
};
template < typename T >
inline constexpr bool is_mixed_v = is_mixed< T >::value;

}  // namespace detail

// -----------------------------------------------------------------------------
// Per-axis differential helpers
//
// Thin wrappers over the shared bodies in la/named.hpp (detail::axisGradient /
// axisHessian / axisJacobian); the axisDim / axisOffset lookups are shared too.
// -----------------------------------------------------------------------------

/// Gradient of a mixed named scalar expansion with respect to one named axis.
template < FixedString Name, typename T, typename... Axes >
[[nodiscard]] auto gradient( const MixedTaylorExpansion< T, Axes... >& f ) noexcept
{
    using E = MixedTaylorExpansion< T, Axes... >;
    static_assert( detail::axisDim< E, Name > >= 1,
                   "gradient<Name>(): axis name not present in this expansion" );
    return detail::axisGradient< detail::axisDim< E, Name >, detail::axisOffset< E, Name > >( f );
}

/// Hessian of a mixed named scalar expansion restricted to one named axis.
template < FixedString Name, typename T, typename... Axes >
[[nodiscard]] auto hessian( const MixedTaylorExpansion< T, Axes... >& f ) noexcept
{
    using E = MixedTaylorExpansion< T, Axes... >;
    static_assert( detail::axisDim< E, Name > >= 1,
                   "hessian<Name>(): axis name not present in this expansion" );
    return detail::axisHessian< detail::axisDim< E, Name >, detail::axisOffset< E, Name > >( f );
}

/// Jacobian of a vector of mixed named expansions w.r.t. one named axis.
template < FixedString Name, typename Derived >
    requires( detail::is_mixed_v< typename Derived::Scalar > )
[[nodiscard]] auto jacobian( const Eigen::MatrixBase< Derived >& F )
{
    using E = typename Derived::Scalar;
    static_assert( detail::axisDim< E, Name > >= 1,
                   "jacobian<Name>(): axis name not present in the expansion" );
    return detail::axisJacobian< detail::axisDim< E, Name >, detail::axisOffset< E, Name > >( F );
}

}  // namespace tax::named

// Re-export the per-axis differential helpers under `tax`. Re-issuing the
// using-declarations here augments the `tax` overload set with the
// MixedTaylorExpansion overloads (a using-declaration only captures the
// overloads visible at its point, so the block in la/named.hpp does not pick
// them up).
namespace tax
{
using named::gradient;
using named::hessian;
using named::jacobian;
}  // namespace tax
