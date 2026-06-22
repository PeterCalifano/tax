#pragma once

// Free-function arithmetic surface for NamedTaylorExpansion: operands over
// different axis sets are embedded into the union before the dense kernels run,
// so the result type tracks the union of axes. Mirrors operators/arithmetic.hpp
// for the unnamed dense type.

#include <tax/core/named.hpp>
#include <tax/operators/arithmetic.hpp>
#include <type_traits>

namespace tax::named
{

// ---------------------------------------------------------------------------
// Composition operators (axis sets merged into their union)
// ---------------------------------------------------------------------------

#define TAX_NAMED_BINARY_OP( OP )                                                       \
    template < typename T, int N, typename... A, typename... B >                        \
    [[nodiscard]] constexpr auto operator OP(                                           \
        const NamedTaylorExpansion< T, N, A... >& a,                                    \
        const NamedTaylorExpansion< T, N, B... >& b ) noexcept                          \
    {                                                                                   \
        using R = detail::MergedNamedTaylorExpansion< T, N, detail::TypeList< A... >,   \
                                                      detail::TypeList< B... > >;       \
        return R{ a.template embed< R >().inner() OP b.template embed< R >().inner() }; \
    }

TAX_NAMED_BINARY_OP( +)
TAX_NAMED_BINARY_OP( -)
TAX_NAMED_BINARY_OP( * )
TAX_NAMED_BINARY_OP( / )

#undef TAX_NAMED_BINARY_OP

// --- Scalar combinations (axis set unchanged) ------------------------------

#define TAX_NAMED_SCALAR_OP( OP )                                                           \
    template < typename T, int N, typename... A >                                           \
    [[nodiscard]] constexpr NamedTaylorExpansion< T, N, A... > operator OP(                 \
        const NamedTaylorExpansion< T, N, A... >& a, std::type_identity_t< T > s ) noexcept \
    {                                                                                       \
        return NamedTaylorExpansion< T, N, A... >{ a.inner() OP s };                        \
    }

TAX_NAMED_SCALAR_OP( +)
TAX_NAMED_SCALAR_OP( -)
TAX_NAMED_SCALAR_OP( * )
TAX_NAMED_SCALAR_OP( / )

#undef TAX_NAMED_SCALAR_OP

template < typename T, int N, typename... A >
[[nodiscard]] constexpr NamedTaylorExpansion< T, N, A... > operator+(
    std::type_identity_t< T > s, const NamedTaylorExpansion< T, N, A... >& a ) noexcept
{
    return a + s;
}

template < typename T, int N, typename... A >
[[nodiscard]] constexpr NamedTaylorExpansion< T, N, A... > operator*(
    std::type_identity_t< T > s, const NamedTaylorExpansion< T, N, A... >& a ) noexcept
{
    return a * s;
}

template < typename T, int N, typename... A >
[[nodiscard]] constexpr NamedTaylorExpansion< T, N, A... > operator-(
    std::type_identity_t< T > s, const NamedTaylorExpansion< T, N, A... >& a ) noexcept
{
    return NamedTaylorExpansion< T, N, A... >{ s - a.inner() };
}

template < typename T, int N, typename... A >
[[nodiscard]] constexpr NamedTaylorExpansion< T, N, A... > operator/(
    std::type_identity_t< T > s, const NamedTaylorExpansion< T, N, A... >& a ) noexcept
{
    return NamedTaylorExpansion< T, N, A... >{ s / a.inner() };
}

template < typename T, int N, typename... A >
[[nodiscard]] constexpr NamedTaylorExpansion< T, N, A... > operator-(
    const NamedTaylorExpansion< T, N, A... >& a ) noexcept
{
    return NamedTaylorExpansion< T, N, A... >{ -a.inner() };
}

}  // namespace tax::named
