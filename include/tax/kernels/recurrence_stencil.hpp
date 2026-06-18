#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include <tax/core/enumeration.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/kernels/cauchy.hpp>  // TAX_USE_STENCIL configuration

namespace tax::detail::kernels
{

/// A single decomposition entry of the recurrence stencil.
struct RecurrenceEntry
{
    std::uint32_t b_idx;
    std::uint32_t g_idx;
    std::uint32_t db;
};

/// Decomposition table driving the degree-by-degree recurrence kernels for M >= 2.
template < int N, int M >
struct RecurrenceStencil
{
    static_assert( M >= 2 );

    static constexpr std::size_t NC = numMonomials( N, M );
    static constexpr std::size_t kEntries = numMonomials( N, 2 * M ) - NC;
    static_assert( kEntries * sizeof( RecurrenceEntry ) <= ( std::size_t{ 64 } << 20 ),
                   "RecurrenceStencil table exceeds 64 MB for this (N, M). "
                   "Disable TAX_USE_STENCIL or fall back to the loop kernels." );

    std::array< RecurrenceEntry, kEntries > entries{};
    /// Row bounds: entries for output index ai live in [row[ai], row[ai+1]).
    std::array< std::uint32_t, NC + 1 > row{};
    /// Total degree |alpha| per output index.
    std::array< std::int32_t, NC > degree{};

    constexpr RecurrenceStencil() noexcept
    {
        std::size_t n = 0;
        std::size_t ai = 0;
        tax::forEachMonomial< M, N >( [this, &n, &ai]( const MultiIndex< M >& alpha ) {
            row[ai] = static_cast< std::uint32_t >( n );
            degree[ai] = totalDegree( alpha );
            tax::forEachSubIndex< M >( alpha, [this, &n]( const MultiIndex< M >& beta,
                                                          const MultiIndex< M >& gamma ) {
                int db = 0;
                for ( int i = 0; i < M; ++i ) db += beta[std::size_t( i )];
                if ( db == 0 ) return;
                entries[n++] = RecurrenceEntry{
                    static_cast< std::uint32_t >( flatIndex< M >( beta ) ),
                    static_cast< std::uint32_t >( flatIndex< M >( gamma ) ),
                    static_cast< std::uint32_t >( db ) };
            } );
            ++ai;
        } );
        row[NC] = static_cast< std::uint32_t >( n );
        // n == kEntries by the bijection documented above.
    }
};

/// Shared per-(N, M) table instance (kept out of the RowFn-templated
/// walker below so each kernel instantiation reuses the same static).
template < int N, int M >
[[nodiscard]] inline const RecurrenceStencil< N, M >& recurrenceStencil() noexcept
{
    static const RecurrenceStencil< N, M > s{};
    return s;
}

/// Walk all recurrence rows (M >= 2) in graded-lex order, so each output sees its lower-degree dependencies already computed. Loop and stencil paths enumerate the same rows.
template < int N, int M, class RowFn >
constexpr void forEachRecurrenceRow( RowFn&& fn ) noexcept
{
    static_assert( M >= 2 );
    constexpr std::size_t NC = numMonomials( N, M );

#if TAX_USE_STENCIL
    if !consteval
    {
        const RecurrenceStencil< N, M >& st = recurrenceStencil< N, M >();
        for ( std::size_t ai = 1; ai < NC; ++ai )
        {
            fn( ai, int( st.degree[ai] ),
                std::span< const RecurrenceEntry >( st.entries.data() + st.row[ai],
                                                    st.entries.data() + st.row[ai + 1] ) );
        }
        return;
    }
#endif
    std::array< RecurrenceEntry, NC > buf{};
    std::size_t ai = 0;
    tax::forEachMonomial< M, N >( [&]( const MultiIndex< M >& alpha ) {
        const std::size_t i = ai++;
        if ( i == 0 ) return;  // alpha == 0 has no |beta| >= 1 decompositions
        std::size_t n = 0;
        tax::forEachSubIndex< M >( alpha, [&]( const MultiIndex< M >& beta,
                                               const MultiIndex< M >& gamma ) {
            int db = 0;
            for ( int q = 0; q < M; ++q ) db += beta[std::size_t( q )];
            if ( db == 0 ) return;
            buf[n++] = RecurrenceEntry{
                static_cast< std::uint32_t >( flatIndex< M >( beta ) ),
                static_cast< std::uint32_t >( flatIndex< M >( gamma ) ),
                static_cast< std::uint32_t >( db ) };
        } );
        fn( i, totalDegree( alpha ), std::span< const RecurrenceEntry >( buf.data(), n ) );
    } );
}

}  // namespace tax::detail::kernels
