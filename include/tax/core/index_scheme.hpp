#pragma once

// ---------------------------------------------------------------------------
// IndexScheme: the monomial-set abstraction the dense kernels are generic over.
// ---------------------------------------------------------------------------
// A scheme bundles what every recurrence kernel needs: the storage size, a
// graded recurrence-row walker (fn(ai, d, span<RecurrenceEntry>) in ascending
// total degree so forward substitution is causal), and the flat<->multi maps.
// IsotropicScheme<N,M> wraps exactly today's tables; a later MixedScheme<...>
// provides an anisotropic (per-axis-capped) monomial set behind the same API.

#include <cstddef>
#include <span>
#include <tax/core/multi_index.hpp>
#include <tax/kernels/recurrence_stencil.hpp>

namespace tax
{

/// Concept: a monomial-set index scheme usable by the dense kernels.
template < typename S >
concept IndexScheme = requires( const S& ) {
    { S::nCoeff } -> std::convertible_to< std::size_t >;
    { S::order } -> std::convertible_to< int >;
    { S::isUnivariate } -> std::convertible_to< bool >;
};

/// The classic single-order graded-lex scheme: total degree <= N over M vars.
template < int N, int M >
struct IsotropicScheme
{
    static constexpr std::size_t nCoeff = numMonomials( N, M );
    static constexpr int order = N;
    static constexpr int vars = M;
    static constexpr bool isUnivariate = ( M == 1 );

    [[nodiscard]] static constexpr std::size_t flatOf( const MultiIndex< M >& a ) noexcept
    {
        return flatIndex< M >( a );
    }
    [[nodiscard]] static constexpr MultiIndex< M > multiOf( std::size_t k ) noexcept
    {
        return unflatIndex< M >( k );
    }

    /// Graded recurrence-row walker (M >= 2). Delegates to the legacy table/loop.
    template < class RowFn >
    static constexpr void forEachRecurrenceRow( RowFn&& fn ) noexcept
        requires( M >= 2 )
    {
        detail::kernels::forEachRecurrenceRow< N, M >( static_cast< RowFn&& >( fn ) );
    }
};

}  // namespace tax
