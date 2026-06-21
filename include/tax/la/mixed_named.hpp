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

// -----------------------------------------------------------------------------
// Axis-offset helpers for MixedTaylorExpansion
// -----------------------------------------------------------------------------

/// Dimension of axis `Name` within a MixedTaylorExpansion type `E`, or -1.
template < typename E, FixedString Name >
inline constexpr int mixedAxisDim = DimOfName< typename E::axis_list, Name >::value;

/// Variable offset of axis `Name` within a MixedTaylorExpansion type `E`, or -1.
///
/// The dimension is clamped to >= 1 to avoid instantiating an OrderedAxis with
/// dimension -1 when the name is absent (the caller's static_assert fires first).
template < typename E, FixedString Name >
inline constexpr int mixedAxisOffset = []() constexpr -> int {
    constexpr int dim = mixedAxisDim< E, Name >;
    if constexpr ( dim < 1 )
        return -1;
    else
        // The global variable index of the axis' first coordinate is exactly
        // its block offset (MixedTaylorExpansion::axisVar<Name, 0>()).
        return E::template axisVar< Name, 0 >();
}();

}  // namespace detail

// -----------------------------------------------------------------------------
// Per-axis differential helpers
// -----------------------------------------------------------------------------

/// Gradient of a mixed named scalar expansion with respect to one named axis.
template < FixedString Name, typename T, typename... Axes >
[[nodiscard]] auto gradient( const MixedTaylorExpansion< T, Axes... >& f ) noexcept
{
    using E = MixedTaylorExpansion< T, Axes... >;
    constexpr int dim = detail::mixedAxisDim< E, Name >;
    static_assert( dim >= 1, "gradient<Name>(): axis name not present in this expansion" );
    constexpr int off = detail::mixedAxisOffset< E, Name >;

    Eigen::Matrix< T, dim, 1 > g;
    MultiIndex< E::vars_v > alpha{};
    for ( int i = 0; i < dim; ++i )
    {
        alpha[std::size_t( off + i )] = 1;
        g( i ) = f.inner().derivative( alpha );
        alpha[std::size_t( off + i )] = 0;
    }
    return g;
}

/// Hessian of a mixed named scalar expansion restricted to one named axis.
template < FixedString Name, typename T, typename... Axes >
[[nodiscard]] auto hessian( const MixedTaylorExpansion< T, Axes... >& f ) noexcept
{
    using E = MixedTaylorExpansion< T, Axes... >;
    constexpr int dim = detail::mixedAxisDim< E, Name >;
    static_assert( dim >= 1, "hessian<Name>(): axis name not present in this expansion" );
    constexpr int off = detail::mixedAxisOffset< E, Name >;

    Eigen::Matrix< T, dim, dim > H;
    for ( int i = 0; i < dim; ++i )
    {
        for ( int j = 0; j < dim; ++j )
        {
            MultiIndex< E::vars_v > alpha{};
            alpha[std::size_t( off + i )] += 1;
            alpha[std::size_t( off + j )] += 1;
            H( i, j ) = f.inner().derivative( alpha );
        }
    }
    return H;
}

/// Jacobian of a vector of mixed named expansions w.r.t. one named axis.
template < FixedString Name, typename Derived >
    requires( detail::is_mixed_v< typename Derived::Scalar > )
[[nodiscard]] auto jacobian( const Eigen::MatrixBase< Derived >& F )
{
    using E = typename Derived::Scalar;
    using T = typename E::scalar_type;
    constexpr int dim = detail::mixedAxisDim< E, Name >;
    static_assert( dim >= 1, "jacobian<Name>(): axis name not present in the expansion" );
    constexpr int off = detail::mixedAxisOffset< E, Name >;
    constexpr int K = Derived::SizeAtCompileTime;

    Eigen::Matrix< T, K, dim > out( F.size(), dim );
    for ( Eigen::Index r = 0; r < F.size(); ++r )
    {
        MultiIndex< E::vars_v > alpha{};
        for ( int j = 0; j < dim; ++j )
        {
            alpha[std::size_t( off + j )] = 1;
            out( r, j ) = F.derived().coeff( r ).inner().derivative( alpha );
            alpha[std::size_t( off + j )] = 0;
        }
    }
    return out;
}

}  // namespace tax::named

// Re-export the per-axis differential helpers under `tax`. A using-declaration
// only captures the overloads visible at its point, so the `using named::...`
// block in la/named.hpp does NOT pick up the MixedTaylorExpansion overloads
// added above; re-issuing the using-declarations here extends the `tax` overload
// set with them (repeating a using-declaration for the same name is allowed and
// merely augments the set). This makes the qualified `tax::gradient<"name">(f)`
// spelling resolve for MixedTaylorExpansion, not just `tax::named::gradient`.
namespace tax
{
using named::gradient;
using named::hessian;
using named::jacobian;
}  // namespace tax
