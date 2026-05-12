#pragma once

#include <array>
#include <cstdint>

#include <tax/utils/enumeration.hpp>

namespace tax::detail
{

/**
 * @brief Lookup table mapping a flat monomial index to its total degree.
 * @tparam N Truncation order (compile-time, non-negative).
 * @tparam M Number of variables (compile-time, `>= 1`).
 *
 * `DegreeOf<N, M>::value[k]` is the total degree of the monomial at flat
 * index `k`. Built once at compile time via the same graded-lex enumeration
 * used elsewhere. The table is small (~`numMonomials(N, M)` bytes), e.g.
 * 3003 B at `(N=8, M=6)` and 18564 B at `(N=12, M=6)`.
 *
 * Used by the sparse kernels to truncate `(beta, gamma)` pairs whose
 * degrees sum to more than `N` in O(1) per pair, instead of unpacking the
 * multi-indices and summing.
 */
template < int N, int M >
struct DegreeOf
{
    static_assert( N >= 0 && M >= 1, "DegreeOf<N, M> requires N >= 0 and M >= 1" );

    static constexpr std::size_t NC = numMonomials( N, M );
    using Table = std::array< std::uint8_t, NC >;

    static constexpr Table value = []() consteval {
        Table t{};
        for ( int d = 0; d <= N; ++d )
            forEachMonomial< M >( d, [&]( const auto& /*alpha*/, std::size_t ai ) {
                t[ai] = std::uint8_t( d );
            } );
        return t;
    }();
};

}  // namespace tax::detail
