#pragma once

#include <array>
#include <cstddef>
#include <tax/core/enumeration.hpp>
#include <tax/core/multi_index.hpp>

namespace tax::detail::kernels
{

/**
 * @brief A single entry in the Cauchy stencil table.
 *
 * Encodes: out[out_idx] += a[a_idx] * b[b_idx]
 */
struct StencilEntry
{
    std::size_t out_idx;
    std::size_t a_idx;
    std::size_t b_idx;
};

/**
 * @brief Compile-time table of all (out, a, b) Cauchy product contributions
 *        for a given (N, M) expansion, for M >= 2.
 *
 * The table contains every triple (i, j, k) such that alpha_j + alpha_k = alpha_i,
 * for all multi-indices alpha_i with |alpha_i| <= N.
 *
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < int N, int M >
struct CauchyStencil
{
    static_assert( M >= 2 );

    // Upper bound on the number of entries: sum over all monomials of
    // (d+1)^(M-1) sub-indices, which is at most numMonomials(N,M)^2.
    // We use a generous overestimate and store the actual count.
    // TODO: replace with the exact count `sum_{|alpha|<=N} prod_i (alpha_i+1)`
    //       once large (N,M) shapes (e.g. N>=8,M>=6) are actually exercised.
    //       Current overestimate factor grows ~M^2; at N=8,M=6 this would be 206 MB.
    static constexpr std::size_t kMaxEntries = numMonomials( N, M ) * numMonomials( N, M );
    static_assert( kMaxEntries < ( 1u << 22 ),
                   "CauchyStencil table exceeds 4M entries (~96 MB). "
                   "Implement the exact count before using this (N, M)." );

    std::array< StencilEntry, kMaxEntries > entries{};
    std::size_t size = 0;

    constexpr CauchyStencil() noexcept
    {
        tax::forEachMonomial< M, N >( [this]( const MultiIndex< M >& alpha ) {
            const std::size_t out_i = flatIndex< M >( alpha );
            tax::forEachSubIndex< M >( alpha, [this, out_i]( const MultiIndex< M >& k,
                                                             const MultiIndex< M >& s ) {
                entries[size++] = StencilEntry{ out_i, flatIndex< M >( k ), flatIndex< M >( s ) };
            } );
        } );
    }
};

/**
 * @brief Precomputed-stencil Cauchy product for M >= 2.
 *
 * Iterates the compile-time stencil table and accumulates contributions,
 * avoiding redundant multi-index arithmetic at runtime.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables (must be >= 2).
 */
template < typename T, int N, int M >
constexpr void cauchyProductStencil( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a,
                                     const Coeffs< T, N, M >& b ) noexcept
    requires( M >= 2 )
{
    static constexpr CauchyStencil< N, M > stencil{};
    out = {};
    for ( std::size_t i = 0; i < stencil.size; ++i )
    {
        const auto& e = stencil.entries[i];
        out[e.out_idx] += a[e.a_idx] * b[e.b_idx];
    }
}

}  // namespace tax::detail::kernels
