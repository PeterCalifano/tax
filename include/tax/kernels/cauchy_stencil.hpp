#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <tax/core/enumeration.hpp>
#include <tax/core/multi_index.hpp>

namespace tax::detail::kernels
{

/// A single entry in the Cauchy stencil table.
struct StencilEntry
{
    std::uint32_t out_idx;
    std::uint32_t a_idx;
    std::uint32_t b_idx;
};

/// Table of all (out, a, b) Cauchy product contributions for a given (N, M) expansion, for M >= 2.
template < int N, int M >
struct CauchyStencil
{
    static_assert( M >= 2 );

    /// Exact number of contributions: |{(beta, gamma) : |beta| + |gamma| <= N}|.
    static constexpr std::size_t kEntries = numMonomials( N, 2 * M );
    static_assert( kEntries * sizeof( StencilEntry ) <= ( std::size_t{ 64 } << 20 ),
                   "CauchyStencil table exceeds 64 MB for this (N, M). "
                   "Disable TAX_USE_STENCIL or fall back to the loop kernel." );

    std::array< StencilEntry, kEntries > entries{};

    constexpr CauchyStencil() noexcept
    {
        std::size_t n = 0;
        tax::forEachMonomial< M, N >( [this, &n]( const MultiIndex< M >& alpha ) {
            const std::uint32_t out_i =
                static_cast< std::uint32_t >( flatIndex< M >( alpha ) );
            tax::forEachSubIndex< M >( alpha, [this, &n, out_i]( const MultiIndex< M >& k,
                                                                 const MultiIndex< M >& s ) {
                entries[n++] = StencilEntry{
                    out_i,
                    static_cast< std::uint32_t >( flatIndex< M >( k ) ),
                    static_cast< std::uint32_t >( flatIndex< M >( s ) ) };
            } );
        } );
        // n == kEntries by the bijection documented above.
    }
};

/// Precomputed-stencil Cauchy product for M >= 2. Not usable in constant evaluation (runtime-static table).
template < typename T, int N, int M >
void cauchyProductStencil( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a,
                           const Coeffs< T, N, M >& b ) noexcept
    requires( M >= 2 )
{
    static const CauchyStencil< N, M > stencil{};
    out = {};
    for ( const StencilEntry& e : stencil.entries )
        out[e.out_idx] += a[e.a_idx] * b[e.b_idx];
}

}  // namespace tax::detail::kernels
