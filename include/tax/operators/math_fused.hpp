#pragma once

// ---------------------------------------------------------------------------
// Fused math surface: operations that compute two coupled results in a single
// recurrence pass (see <tax/kernels/fused.hpp> for the provenance and the
// benchmark evidence). Pair-returning functions order the pair as spelled in
// the name: sinCos -> {sin, cos}, sqrtInvSqrt -> {sqrt, 1/sqrt},
// expSinCos -> {exp*sin, exp*cos}.
//
//   sinCos(x)        {sin(x), cos(x)} — one coupled pass, the price of one.
//   sinhCosh(x)      {sinh(x), cosh(x)} — one shared exp(x)/exp(-x) pair.
//   sqrtInvSqrt(x)   {sqrt(x), 1/sqrt(x)} — use only when BOTH are consumed.
//   expSin(v, u)     exp(v)*sin(u)  — one coupled pass, ~1.4x vs exp(v)*sin(u).
//   expCos(v, u)     exp(v)*cos(u)  — likewise.
//   expSinCos(v, u)  {exp(v)*sin(u), exp(v)*cos(u)} — both for the price of one.
//
// Named and mixed-order named overloads live in tax::named below and are
// re-exported into tax::; two-operand forms compose in the union of the
// operands' axis sets, exactly like operator* / atan2.
// ---------------------------------------------------------------------------

#include <tax/core/mixed_named.hpp>
#include <tax/core/named.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/kernels/fused.hpp>
#include <tax/kernels/transcendental.hpp>
#include <tax/kernels/trigonometric.hpp>
#include <utility>

namespace tax
{

// ---------------------------------------------------------------------------
// Dense
// ---------------------------------------------------------------------------

/// `{sin(x), cos(x)}` from the single coupled recurrence both already share.
template < typename T, IndexScheme Scheme >
[[nodiscard]] auto sinCos( const TaylorExpansion< T, Scheme >& x ) noexcept
{
    std::pair< TaylorExpansion< T, Scheme >, TaylorExpansion< T, Scheme > > r;
    detail::kernels::seriesSinCos< T, Scheme >( r.first.coefficients(), r.second.coefficients(),
                                                x.coefficients() );
    return r;
}

/// `{sinh(x), cosh(x)}` from one shared exp(x)/exp(-x) pair.
template < typename T, IndexScheme Scheme >
[[nodiscard]] auto sinhCosh( const TaylorExpansion< T, Scheme >& x ) noexcept
{
    std::pair< TaylorExpansion< T, Scheme >, TaylorExpansion< T, Scheme > > r;
    detail::kernels::seriesSinhCosh< T, Scheme >( r.first.coefficients(), r.second.coefficients(),
                                                  x.coefficients() );
    return r;
}

/// `{sqrt(x), 1/sqrt(x)}` interleaved in one pass. Requires `x.value() > 0`.
/// Only worth calling when both results are consumed (e.g. r and 1/r^3):
/// a single-output caller should use sqrt() or pow() instead.
template < typename T, IndexScheme Scheme >
[[nodiscard]] auto sqrtInvSqrt( const TaylorExpansion< T, Scheme >& x ) noexcept
{
    std::pair< TaylorExpansion< T, Scheme >, TaylorExpansion< T, Scheme > > r;
    detail::kernels::seriesSqrtInvSqrt< T, Scheme >( r.first.coefficients(),
                                                     r.second.coefficients(), x.coefficients() );
    return r;
}

/// Fused `exp(v) * sin(u)` — one coupled recurrence instead of exp, sin/cos
/// and a Cauchy product.
template < typename T, IndexScheme Scheme >
[[nodiscard]] TaylorExpansion< T, Scheme > expSin( const TaylorExpansion< T, Scheme >& v,
                                                   const TaylorExpansion< T, Scheme >& u ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    detail::kernels::seriesExpSin< T, Scheme >( r.coefficients(), v.coefficients(),
                                                u.coefficients() );
    return r;
}

/// Fused `exp(v) * cos(u)`.
template < typename T, IndexScheme Scheme >
[[nodiscard]] TaylorExpansion< T, Scheme > expCos( const TaylorExpansion< T, Scheme >& v,
                                                   const TaylorExpansion< T, Scheme >& u ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    detail::kernels::seriesExpCos< T, Scheme >( r.coefficients(), v.coefficients(),
                                                u.coefficients() );
    return r;
}

/// `{exp(v)*sin(u), exp(v)*cos(u)}` — the coupled pass computes both anyway.
template < typename T, IndexScheme Scheme >
[[nodiscard]] auto expSinCos( const TaylorExpansion< T, Scheme >& v,
                              const TaylorExpansion< T, Scheme >& u ) noexcept
{
    std::pair< TaylorExpansion< T, Scheme >, TaylorExpansion< T, Scheme > > r;
    detail::kernels::seriesExpSinCos< T, Scheme >( r.first.coefficients(), r.second.coefficients(),
                                                   v.coefficients(), u.coefficients() );
    return r;
}

}  // namespace tax

// ---------------------------------------------------------------------------
// Named (single-order) and mixed-order named overloads
// ---------------------------------------------------------------------------

namespace tax::named
{

#define TAX_NAMED_FUSED_PAIR( FN )                                                \
    template < typename T, int N, typename... A >                                 \
    [[nodiscard]] auto FN( const NamedTaylorExpansion< T, N, A... >& a ) noexcept \
    {                                                                             \
        using R = NamedTaylorExpansion< T, N, A... >;                             \
        auto p = tax::FN( a.inner() );                                            \
        return std::pair{ R{ p.first }, R{ p.second } };                          \
    }                                                                             \
    template < typename T, typename... A >                                        \
    [[nodiscard]] auto FN( const MixedTaylorExpansion< T, A... >& a ) noexcept    \
    {                                                                             \
        using R = MixedTaylorExpansion< T, A... >;                                \
        auto p = tax::FN( a.inner() );                                            \
        return std::pair{ R{ p.first }, R{ p.second } };                          \
    }

TAX_NAMED_FUSED_PAIR( sinCos )
TAX_NAMED_FUSED_PAIR( sinhCosh )
TAX_NAMED_FUSED_PAIR( sqrtInvSqrt )

#undef TAX_NAMED_FUSED_PAIR

/// Fused `exp(v) * sin(u)` over the union of the two operands' axis sets.
template < typename T, int N, typename... A, typename... B >
[[nodiscard]] auto expSin( const NamedTaylorExpansion< T, N, A... >& v,
                           const NamedTaylorExpansion< T, N, B... >& u ) noexcept
{
    using R = detail::MergedNamedTaylorExpansion< T, N, detail::TypeList< A... >,
                                                  detail::TypeList< B... > >;
    return R{ tax::expSin( v.template embed< R >().inner(), u.template embed< R >().inner() ) };
}

/// Fused `exp(v) * cos(u)` over the union of the two operands' axis sets.
template < typename T, int N, typename... A, typename... B >
[[nodiscard]] auto expCos( const NamedTaylorExpansion< T, N, A... >& v,
                           const NamedTaylorExpansion< T, N, B... >& u ) noexcept
{
    using R = detail::MergedNamedTaylorExpansion< T, N, detail::TypeList< A... >,
                                                  detail::TypeList< B... > >;
    return R{ tax::expCos( v.template embed< R >().inner(), u.template embed< R >().inner() ) };
}

/// `{exp(v)*sin(u), exp(v)*cos(u)}` over the union of the two operands' axis sets.
template < typename T, int N, typename... A, typename... B >
[[nodiscard]] auto expSinCos( const NamedTaylorExpansion< T, N, A... >& v,
                              const NamedTaylorExpansion< T, N, B... >& u ) noexcept
{
    using R = detail::MergedNamedTaylorExpansion< T, N, detail::TypeList< A... >,
                                                  detail::TypeList< B... > >;
    auto p = tax::expSinCos( v.template embed< R >().inner(), u.template embed< R >().inner() );
    return std::pair{ R{ p.first }, R{ p.second } };
}

/// Fused `exp(v) * sin(u)` over the union of the two operands' (ordered) axis sets.
template < typename T, typename... A, typename... B >
[[nodiscard]] auto expSin( const MixedTaylorExpansion< T, A... >& v,
                           const MixedTaylorExpansion< T, B... >& u ) noexcept
{
    using R =
        detail::MergedMixedTaylorExpansion< T, detail::TypeList< A... >, detail::TypeList< B... > >;
    return R{ tax::expSin( v.template embed< R >().inner(), u.template embed< R >().inner() ) };
}

/// Fused `exp(v) * cos(u)` over the union of the two operands' (ordered) axis sets.
template < typename T, typename... A, typename... B >
[[nodiscard]] auto expCos( const MixedTaylorExpansion< T, A... >& v,
                           const MixedTaylorExpansion< T, B... >& u ) noexcept
{
    using R =
        detail::MergedMixedTaylorExpansion< T, detail::TypeList< A... >, detail::TypeList< B... > >;
    return R{ tax::expCos( v.template embed< R >().inner(), u.template embed< R >().inner() ) };
}

/// `{exp(v)*sin(u), exp(v)*cos(u)}` over the union of the two operands' (ordered) axis sets.
template < typename T, typename... A, typename... B >
[[nodiscard]] auto expSinCos( const MixedTaylorExpansion< T, A... >& v,
                              const MixedTaylorExpansion< T, B... >& u ) noexcept
{
    using R =
        detail::MergedMixedTaylorExpansion< T, detail::TypeList< A... >, detail::TypeList< B... > >;
    auto p = tax::expSinCos( v.template embed< R >().inner(), u.template embed< R >().inner() );
    return std::pair{ R{ p.first }, R{ p.second } };
}

}  // namespace tax::named

namespace tax
{
using named::expCos;
using named::expSin;
using named::expSinCos;
using named::sinCos;
using named::sinhCosh;
using named::sqrtInvSqrt;
}  // namespace tax
