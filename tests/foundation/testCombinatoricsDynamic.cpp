#include "testUtils.hpp"

#include <array>
#include <span>
#include <vector>

// Access detail internals directly for unit-testing the combinatorics layer.
using namespace tax::detail;

// =============================================================================
// numMonomials runtime overload
// =============================================================================

TEST( NumMonomialsRuntime, AgreesWithTemplateOverload )
{
    for ( std::size_t N = 0; N <= 8; ++N )
    {
        for ( std::size_t M = 1; M <= 6; ++M )
        {
            EXPECT_EQ( numMonomials( N, M ), numMonomials( int( N ), int( M ) ) );
        }
    }
}

// =============================================================================
// flatIndex / unflatIndex span overloads
// =============================================================================

template < int M >
static void ExpectSpanRoundTripsTo( std::size_t N )
{
    const std::size_t cnt = numMonomials( int( N ), M );
    std::array< int, std::size_t( M ) > buf{};
    std::span< int > out( buf.data(), std::size_t( M ) );
    for ( std::size_t k = 0; k < cnt; ++k )
    {
        unflatIndex( k, out );
        const auto k_back = flatIndex( std::span< const int >( buf.data(), std::size_t( M ) ) );
        EXPECT_EQ( k_back, k ) << "M=" << M << " k=" << k;
    }
}

TEST( FlatUnflatSpan, RoundTrip_M2 ) { ExpectSpanRoundTripsTo< 2 >( 7 ); }
TEST( FlatUnflatSpan, RoundTrip_M3 ) { ExpectSpanRoundTripsTo< 3 >( 5 ); }
TEST( FlatUnflatSpan, RoundTrip_M4 ) { ExpectSpanRoundTripsTo< 4 >( 4 ); }

// Verify the runtime span overload yields the SAME indices as the template
// overload (i.e. the layout is identical, not just self-consistent).
TEST( FlatUnflatSpan, AgreesWithTemplateOverload_M3 )
{
    constexpr int M = 3;
    for ( int d = 0; d <= 5; ++d )
    {
        forEachMonomial< M >( d, [&]( const tax::MultiIndex< M >& alpha, std::size_t k_tmpl ) {
            const std::span< const int > alpha_span( alpha.data(), std::size_t( M ) );
            EXPECT_EQ( flatIndex( alpha_span ), k_tmpl );
        } );
    }
}

// =============================================================================
// forEachMonomial runtime-M overload
// =============================================================================

TEST( ForEachMonomialRuntime, AgreesWithTemplateOverload_M2 )
{
    constexpr int M = 2;
    for ( int d = 0; d <= 5; ++d )
    {
        std::vector< std::size_t > tmpl_indices;
        forEachMonomial< M >(
            d, [&]( const tax::MultiIndex< M >&, std::size_t ai ) { tmpl_indices.push_back( ai ); } );

        std::vector< std::size_t > rt_indices;
        forEachMonomial( M, d,
                         [&]( std::span< const int >, std::size_t ai ) { rt_indices.push_back( ai ); } );

        EXPECT_EQ( tmpl_indices, rt_indices );
    }
}

TEST( ForEachMonomialRuntime, AgreesWithTemplateOverload_M4 )
{
    constexpr int M = 4;
    for ( int d = 0; d <= 4; ++d )
    {
        std::vector< std::size_t > tmpl_indices;
        forEachMonomial< M >(
            d, [&]( const tax::MultiIndex< M >&, std::size_t ai ) { tmpl_indices.push_back( ai ); } );

        std::vector< std::size_t > rt_indices;
        forEachMonomial( M, d,
                         [&]( std::span< const int >, std::size_t ai ) { rt_indices.push_back( ai ); } );

        EXPECT_EQ( tmpl_indices, rt_indices );
    }
}

// =============================================================================
// forEachSubIndex runtime-M overload
// =============================================================================

TEST( ForEachSubIndexRuntime, AgreesWithTemplateOverload_M3 )
{
    constexpr int M = 3;
    tax::MultiIndex< M > alpha{ 2, 1, 1 };

    std::vector< std::pair< std::size_t, std::size_t > > tmpl_pairs;
    forEachSubIndex< M >( alpha,
                          [&]( std::size_t bi, std::size_t gi ) { tmpl_pairs.emplace_back( bi, gi ); } );

    const std::span< const int > alpha_span( alpha.data(), std::size_t( M ) );
    std::vector< std::pair< std::size_t, std::size_t > > rt_pairs;
    forEachSubIndex( alpha_span,
                     [&]( std::size_t bi, std::size_t gi ) { rt_pairs.emplace_back( bi, gi ); } );

    EXPECT_EQ( tmpl_pairs, rt_pairs );
}
