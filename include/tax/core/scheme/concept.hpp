#pragma once

// ---------------------------------------------------------------------------
// IndexScheme: the monomial-set abstraction the dense kernels are generic over.
// ---------------------------------------------------------------------------
// A scheme bundles what every recurrence kernel needs: the storage size, a
// graded recurrence-row walker (fn(ai, d, span<RecurrenceEntry>) in ascending
// total degree so forward substitution is causal), and the flat<->multi maps.
// IsotropicScheme<N,M> wraps exactly today's tables; a later MixedScheme<...>
// provides an anisotropic (per-axis-capped) monomial set behind the same API.

#include <array>
#include <cstddef>

namespace tax
{

/// Concept: a monomial-set index scheme usable by the dense kernels.
template < typename S >
concept IndexScheme = requires( const S& ) {
    { S::nCoeff } -> std::convertible_to< std::size_t >;
    { S::order } -> std::convertible_to< int >;
    { S::isUnivariate } -> std::convertible_to< bool >;
};

// ---------------------------------------------------------------------------
// Scheme-generic Cauchy product entry point (free function in tax namespace)
// ---------------------------------------------------------------------------
// Selected when the second template argument satisfies IndexScheme (a type).
// For IsotropicScheme<N,M> this forwards to detail::kernels::cauchyProduct<T,N,M>,
// preserving the unroll/stencil/loop dispatch.

/// Scheme-generic Cauchy product: delegates to Scheme::cauchyProduct<T>.
template < typename T, IndexScheme Scheme >
constexpr void cauchyProduct( std::array< T, Scheme::nCoeff >& out,
                              const std::array< T, Scheme::nCoeff >& a,
                              const std::array< T, Scheme::nCoeff >& b ) noexcept
{
    Scheme::template cauchyProduct< T >( out, a, b );
}

/// Scheme-generic self-product: delegates to Scheme::cauchySelfProduct<T>.
template < typename T, IndexScheme Scheme >
constexpr void cauchySelfProduct( std::array< T, Scheme::nCoeff >& out,
                                  const std::array< T, Scheme::nCoeff >& f ) noexcept
{
    Scheme::template cauchySelfProduct< T >( out, f );
}

}  // namespace tax
