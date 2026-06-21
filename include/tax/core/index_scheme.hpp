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
#include <span>
#include <tax/core/multi_index.hpp>
#include <tax/kernels/cauchy.hpp>
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

    /// Scheme-owned Cauchy product: delegates to the legacy dispatch (unroll/stencil/loop).
    template < typename T >
    static constexpr void cauchyProduct( std::array< T, nCoeff >& out,
                                         const std::array< T, nCoeff >& a,
                                         const std::array< T, nCoeff >& b ) noexcept
    {
        detail::kernels::cauchyProduct< T, N, M >( out, a, b );
    }

    /// Scheme-owned self-product (M == 1: bespoke symmetric loop; M >= 2: cauchyProduct(f,f)).
    /// The M == 1 loop is duplicated in detail::kernels::cauchySelfProduct<T,Scheme>
    /// (algebra.hpp) because that header includes this one, not the reverse; keep
    /// the two bodies in sync.
    template < typename T >
    static constexpr void cauchySelfProduct( std::array< T, nCoeff >& out,
                                             const std::array< T, nCoeff >& f ) noexcept
    {
        if constexpr ( M == 1 )
        {
            out = {};
            for ( int d = 0; d <= N; ++d )
            {
                for ( int k = 0; k + k < d; ++k )
                    out[std::size_t( d )] += T{ 2 } * f[std::size_t( k )] * f[std::size_t( d - k )];
                if ( d % 2 == 0 )
                    out[std::size_t( d )] += f[std::size_t( d / 2 )] * f[std::size_t( d / 2 )];
            }
        } else
        {
            // Route through this scheme's own product so a future Scheme that
            // overrides cauchyProduct is honoured (not the legacy kernel directly).
            cauchyProduct< T >( out, f, f );
        }
    }
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
