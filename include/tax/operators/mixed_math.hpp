#pragma once

// Binary math surface for MixedTaylorExpansion (pow, atan2) and the `tax::`
// re-exports of the whole mixed math surface. Mirrors named_math_binary.hpp.
//
// The re-export block exists because a qualified call (`tax::sin(...)`)
// suppresses ADL, and the using-declarations issued by named_math_unary.hpp /
// named_math_binary.hpp only capture the overloads visible at *their* point
// of declaration — which does not include the MixedTaylorExpansion overloads
// defined in <tax/core/mixed_named.hpp>. Re-issuing them here (after both are
// visible) makes the qualified spellings work regardless of include order.

#include <tax/core/mixed_named.hpp>
#include <tax/operators/math_binary.hpp>
#include <tax/operators/math_unary.hpp>
#include <type_traits>

namespace tax::named
{

/// `x^n` for an integer exponent (axis set preserved).
template < typename T, typename... A >
[[nodiscard]] constexpr MixedTaylorExpansion< T, A... > pow(
    const MixedTaylorExpansion< T, A... >& x, int n ) noexcept
{
    return MixedTaylorExpansion< T, A... >{ tax::pow( x.inner(), n ) };
}

/// `x^p` for a real exponent (axis set preserved; requires x.value() != 0).
template < typename T, typename... A >
[[nodiscard]] constexpr MixedTaylorExpansion< T, A... > pow(
    const MixedTaylorExpansion< T, A... >& x, std::type_identity_t< T > p ) noexcept
{
    return MixedTaylorExpansion< T, A... >{ tax::pow( x.inner(), p ) };
}

/// `atan2(y, x)` over the union of the two operands' (ordered) axis sets.
template < typename T, typename... A, typename... B >
[[nodiscard]] constexpr auto atan2( const MixedTaylorExpansion< T, A... >& y,
                                    const MixedTaylorExpansion< T, B... >& x ) noexcept
{
    using R =
        detail::MergedMixedTaylorExpansion< T, detail::TypeList< A... >, detail::TypeList< B... > >;
    return R{ tax::atan2( y.template embed< R >().inner(), x.template embed< R >().inner() ) };
}

}  // namespace tax::named

namespace tax
{

#define TAX_MIXED_REEXPORT( FN ) using named::FN;

TAX_MIXED_REEXPORT( square )
TAX_MIXED_REEXPORT( cube )
TAX_MIXED_REEXPORT( reciprocal )
TAX_MIXED_REEXPORT( sqrt )
TAX_MIXED_REEXPORT( cbrt )
TAX_MIXED_REEXPORT( exp )
TAX_MIXED_REEXPORT( log )
TAX_MIXED_REEXPORT( sin )
TAX_MIXED_REEXPORT( cos )
TAX_MIXED_REEXPORT( tan )
TAX_MIXED_REEXPORT( asin )
TAX_MIXED_REEXPORT( acos )
TAX_MIXED_REEXPORT( atan )
TAX_MIXED_REEXPORT( sinh )
TAX_MIXED_REEXPORT( cosh )
TAX_MIXED_REEXPORT( tanh )
TAX_MIXED_REEXPORT( asinh )
TAX_MIXED_REEXPORT( acosh )
TAX_MIXED_REEXPORT( atanh )
TAX_MIXED_REEXPORT( erf )
TAX_MIXED_REEXPORT( pow )
TAX_MIXED_REEXPORT( atan2 )

#undef TAX_MIXED_REEXPORT

}  // namespace tax
