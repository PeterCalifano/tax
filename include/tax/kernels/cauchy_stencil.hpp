#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <tax/core/enumeration.hpp>
#include <tax/core/multi_index.hpp>

namespace tax::detail::kernels
{

/**
 * @brief A single entry in the Cauchy stencil table.
 *
 * Encodes: out[out_idx] += a[a_idx] * b[b_idx]
 *
 * 32-bit indices keep the entry at 12 bytes (vs 24 with std::size_t),
 * halving the memory traffic of the table walk; 32 bits comfortably cover
 * any practically instantiable numMonomials(N, M).
 */
struct StencilEntry
{
    std::uint32_t out_idx;
    std::uint32_t a_idx;
    std::uint32_t b_idx;
};

/**
 * @brief Table of all (out, a, b) Cauchy product contributions for a given
 *        (N, M) expansion, for M >= 2.
 *
 * The table contains every triple (i, j, k) such that alpha_j + alpha_k = alpha_i,
 * for all multi-indices alpha_i with |alpha_i| <= N.
 *
 * The entry count is exact: contributions are the ordered pairs (beta, gamma)
 * with |beta| + |gamma| <= N, which biject with monomials of total degree <= N
 * in 2M variables, i.e. numMonomials(N, 2M) entries.
 *
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
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

/**
 * @brief Precomputed-stencil Cauchy product for M >= 2.
 *
 * Iterates the stencil table (built once, at first use) and accumulates
 * contributions, avoiding redundant multi-index arithmetic at runtime.
 *
 * Not usable in constant evaluation (the table is a runtime-initialised
 * static); `cauchyProduct` routes constant evaluation to the loop kernel.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables (must be >= 2).
 */
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
