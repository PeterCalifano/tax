#pragma once

// ---------------------------------------------------------------------------
// Box-filtered Cauchy + recurrence stencils for MixedScheme.
// ---------------------------------------------------------------------------
// Provides MixedBoxCauchyStencil<Groups...> and MixedBoxRecurrenceStencil<Groups...>:
// precomputed (out, a, b) / (b, g, db) tables for the box product of a MixedScheme.
//
// Sizing (compile-time std::array — no heap):
//   kCauchyEntries = Π_g numMonomials(order_g, 2*dim_g)
//                  = |{(β,γ): |β_g|+|γ_g| ≤ order_g ∀g}|
//   kRecEntries    = kCauchyEntries - nCoeff
//                  (drops the single |β|==0 row per output, as RecurrenceStencil does)
//
// Both tables are built in graded (ascending total degree) output order by iterating
// outputs 0..nCoeff-1 (MixedScheme::multiOf visits in graded order — verified by
// the FlatRoundTripDenseAndGraded test in tests/mixed/test_mixed_scheme.cpp).
//
// NOTE: This header does NOT include mixed_scheme.hpp. Instead, mixed_scheme.hpp
// includes this header (after its own class definition), avoiding a circular dependency.
// The stencil structs reference MixedScheme<Groups...> only inside template bodies
// that are instantiated later (when the static const stencil is first accessed).

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <tax/core/enumeration.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/kernels/cauchy_stencil.hpp>
#include <tax/kernels/recurrence_stencil.hpp>

namespace tax
{
// Forward declaration — MixedScheme is fully defined in mixed_scheme.hpp, which
// includes this header after the class body. The template bodies below are only
// instantiated when the stencil static is first accessed (at runtime), at which
// point MixedScheme<Groups...> is complete.
template < typename... Groups >
struct MixedScheme;
}  // namespace tax

namespace tax::detail::kernels
{

// ---------------------------------------------------------------------------
// Compile-time sizing helpers
// ---------------------------------------------------------------------------

/// Cauchy entry count for MixedScheme<Groups...>: Π_g numMonomials(order_g, 2*dim_g).
template < typename... Groups >
inline constexpr std::size_t mixedCauchyEntries =
    ( numMonomials( Groups::order, 2 * Groups::dim ) * ... );

/// Recurrence entry count = Cauchy count − nCoeff (drop |β|==0 row per output).
template < typename... Groups >
inline constexpr std::size_t mixedRecurrenceEntries =
    mixedCauchyEntries< Groups... > - tax::MixedScheme< Groups... >::nCoeff;

// ---------------------------------------------------------------------------
// MixedBoxCauchyStencil<Groups...>
// ---------------------------------------------------------------------------

/// Precomputed (out, a, b) table for the box Cauchy product of MixedScheme<Groups...>.
/// Entries are stored in graded output order (output ai = 0 … nCoeff-1).
template < typename... Groups >
struct MixedBoxCauchyStencil
{
    static constexpr std::size_t kNCoeff = tax::MixedScheme< Groups... >::nCoeff;
    static constexpr std::size_t kEntries = mixedCauchyEntries< Groups... >;
    static_assert( kEntries * sizeof( StencilEntry ) <= ( std::size_t{ 128 } << 20 ),
                   "MixedBoxCauchyStencil exceeds 128 MB — reduce group orders." );

    std::array< StencilEntry, kEntries > entries{};

    using Scheme = tax::MixedScheme< Groups... >;
    static constexpr int V = Scheme::vars;

    constexpr MixedBoxCauchyStencil() noexcept
    {
        std::size_t n = 0;
        // Outputs in graded order: multiOf(0), multiOf(1), … multiOf(nCoeff-1).
        // Within each output α, iterate β by ascending flat index (bi = 0…nCoeff-1)
        // so the accumulation order matches the (i,j) brute-force double loop exactly.
        for ( std::size_t ai = 0; ai < kNCoeff; ++ai )
        {
            const MultiIndex< V > alpha = Scheme::multiOf( ai );
            for ( std::size_t bi = 0; bi < kNCoeff; ++bi )
            {
                const MultiIndex< V > beta = Scheme::multiOf( bi );
                // β must be componentwise ≤ α (so γ = α − β has non-negative components).
                bool ok = true;
                for ( int v = 0; v < V; ++v )
                    if ( beta[std::size_t( v )] > alpha[std::size_t( v )] )
                    {
                        ok = false;
                        break;
                    }
                if ( !ok ) continue;
                MultiIndex< V > gamma{};
                for ( int v = 0; v < V; ++v )
                    gamma[std::size_t( v )] = alpha[std::size_t( v )] - beta[std::size_t( v )];
                entries[n++] = StencilEntry{
                    static_cast< std::uint32_t >( ai ), static_cast< std::uint32_t >( bi ),
                    static_cast< std::uint32_t >( Scheme::flatOf( gamma ) ) };
            }
        }
        // n == kEntries: each (β,γ) pair with β+γ ∈ Box contributes one entry.
    }
};

// ---------------------------------------------------------------------------
// MixedBoxRecurrenceStencil<Groups...>
// ---------------------------------------------------------------------------

/// Decomposition table for the degree-by-degree recurrence over MixedScheme<Groups...>.
/// Shape mirrors RecurrenceStencil: entries[], row[] bounds, degree[] per output.
/// Built in graded output order. Drops the |β|==0 row per output (db >= 1 always).
template < typename... Groups >
struct MixedBoxRecurrenceStencil
{
    static constexpr std::size_t kNCoeff = tax::MixedScheme< Groups... >::nCoeff;
    static constexpr std::size_t kEntries = mixedRecurrenceEntries< Groups... >;
    static_assert( kEntries * sizeof( RecurrenceEntry ) <= ( std::size_t{ 128 } << 20 ),
                   "MixedBoxRecurrenceStencil exceeds 128 MB — reduce group orders." );

    std::array< RecurrenceEntry, kEntries > entries{};
    /// Row bounds: recurrence entries for output ai live in [row[ai], row[ai+1]).
    std::array< std::uint32_t, kNCoeff + 1 > row{};
    /// Total degree |α| for each output flat index.
    std::array< std::int32_t, kNCoeff > degree{};

    using Scheme = tax::MixedScheme< Groups... >;
    static constexpr int V = Scheme::vars;

    constexpr MixedBoxRecurrenceStencil() noexcept
    {
        std::size_t n = 0;
        for ( std::size_t ai = 0; ai < kNCoeff; ++ai )
        {
            row[ai] = static_cast< std::uint32_t >( n );
            const MultiIndex< V > alpha = Scheme::multiOf( ai );
            degree[ai] = totalDegree( alpha );

            // Enumerate β ≤ α, skip |β|==0.
            forEachSubIndex< V >(
                alpha, [&]( const MultiIndex< V >& beta, const MultiIndex< V >& gamma ) {
                    int db = 0;
                    for ( int v = 0; v < V; ++v ) db += beta[std::size_t( v )];
                    if ( db == 0 ) return;
                    entries[n++] =
                        RecurrenceEntry{ static_cast< std::uint32_t >( Scheme::flatOf( beta ) ),
                                         static_cast< std::uint32_t >( Scheme::flatOf( gamma ) ),
                                         static_cast< std::uint32_t >( db ) };
                } );
        }
        row[kNCoeff] = static_cast< std::uint32_t >( n );
    }
};

// ---------------------------------------------------------------------------
// Shared per-MixedScheme stencil accessors (runtime static — one per type).
// ---------------------------------------------------------------------------

template < typename... Groups >
[[nodiscard]] inline const MixedBoxCauchyStencil< Groups... >& mixedBoxCauchyStencil() noexcept
{
    static const MixedBoxCauchyStencil< Groups... > s{};
    return s;
}

template < typename... Groups >
[[nodiscard]] inline const MixedBoxRecurrenceStencil< Groups... >&
mixedBoxRecurrenceStencil() noexcept
{
    static const MixedBoxRecurrenceStencil< Groups... > s{};
    return s;
}

}  // namespace tax::detail::kernels
