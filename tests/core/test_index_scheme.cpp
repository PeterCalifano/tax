#include <gtest/gtest.h>

#include <tax/core/index_scheme.hpp>
#include <tax/kernels/recurrence_stencil.hpp>
#include <tax/tax.hpp>

using tax::IsotropicScheme;
using tax::detail::kernels::RecurrenceEntry;

// IsotropicScheme<N,M> must reproduce the legacy numMonomials/forEachRecurrenceRow tables exactly.
TEST( IndexScheme, IsotropicMatchesLegacyShape )
{
    using S = IsotropicScheme< 5, 2 >;
    static_assert( S::nCoeff == tax::numMonomials( 5, 2 ) );
    static_assert( S::isUnivariate == false );
    static_assert( S::order == 5 );
    static_assert( IsotropicScheme< 4, 1 >::isUnivariate == true );
}

TEST( IndexScheme, IsotropicRecurrenceRowsMatchLegacy )
{
    constexpr int N = 5, M = 2;
    using S = IsotropicScheme< N, M >;

    // Collect rows from the legacy walker.
    std::vector< std::tuple< std::size_t, int, std::vector< std::array< std::uint32_t, 3 > > > >
        legacy;
    tax::detail::kernels::forEachRecurrenceRow< N, M >(
        [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
            std::vector< std::array< std::uint32_t, 3 > > es;
            for ( const auto& e : row ) es.push_back( { e.b_idx, e.g_idx, e.db } );
            legacy.push_back( { ai, d, es } );
        } );

    // Collect rows from the scheme.
    std::vector< std::tuple< std::size_t, int, std::vector< std::array< std::uint32_t, 3 > > > >
        viaScheme;
    S::forEachRecurrenceRow( [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
        std::vector< std::array< std::uint32_t, 3 > > es;
        for ( const auto& e : row ) es.push_back( { e.b_idx, e.g_idx, e.db } );
        viaScheme.push_back( { ai, d, es } );
    } );

    EXPECT_EQ( legacy, viaScheme );
}
