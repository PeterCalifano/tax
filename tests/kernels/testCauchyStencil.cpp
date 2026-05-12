#include <algorithm>
#include <set>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <tax/kernels/cauchy_stencil.hpp>
#include <tax/utils/enumeration.hpp>

namespace
{

using Triple = std::tuple< std::size_t, std::size_t, std::size_t >;

template < int N, int M >
std::vector< Triple > enumerateLive()
{
    std::vector< Triple > out;
    for ( int d = 0; d <= N; ++d )
    {
        tax::detail::forEachMonomial< M >(
            d, [&]( const auto& alpha, std::size_t ai ) {
                tax::detail::forEachSubIndex< M >(
                    alpha, [&]( std::size_t bi, std::size_t gi ) {
                        out.emplace_back( ai, bi, gi );
                    } );
            } );
    }
    return out;
}

template < int N, int M >
std::vector< Triple > enumerateStencil()
{
    using S = tax::detail::CauchyStencil< N, M >;
    std::vector< Triple > out;
    for ( std::size_t k = 0; k < S::NC; ++k )
    {
        for ( std::size_t j = S::offsets[k]; j < S::offsets[k + 1]; ++j )
            out.emplace_back( k, std::size_t( S::col_a[j] ),
                              std::size_t( S::col_b[j] ) );
    }
    return out;
}

template < int N, int M >
void expectAsymmetricStencilMatchesLive()
{
    auto live = enumerateLive< N, M >();
    auto sten = enumerateStencil< N, M >();
    std::sort( live.begin(), live.end() );
    std::sort( sten.begin(), sten.end() );
    EXPECT_EQ( live.size(), sten.size() );
    EXPECT_EQ( live, sten );

    using S = tax::detail::CauchyStencil< N, M >;
    EXPECT_EQ( S::offsets.front(), 0u );
    EXPECT_EQ( S::offsets.back(), S::PC );
}

template < int N, int M >
std::set< Triple > enumerateLiveUnordered()
{
    std::set< Triple > out;
    for ( int d = 0; d <= N; ++d )
    {
        tax::detail::forEachMonomial< M >(
            d, [&]( const auto& alpha, std::size_t ai ) {
                tax::detail::forEachSubIndex< M >(
                    alpha, [&]( std::size_t bi, std::size_t gi ) {
                        if ( bi <= gi ) out.emplace( ai, bi, gi );
                    } );
            } );
    }
    return out;
}

template < int N, int M >
std::set< Triple > enumerateSymStencil()
{
    using S = tax::detail::CauchySymStencil< N, M >;
    std::set< Triple > out;
    for ( std::size_t k = 0; k < S::NC; ++k )
    {
        for ( std::size_t j = S::offsets[k]; j < S::offsets[k + 1]; ++j )
            out.emplace( k, std::size_t( S::col_a[j] ),
                         std::size_t( S::col_b[j] ) );
    }
    return out;
}

template < int N, int M >
void expectSymmetricStencilMatchesLive()
{
    const auto live = enumerateLiveUnordered< N, M >();
    const auto sten = enumerateSymStencil< N, M >();
    EXPECT_EQ( live, sten );

    using S = tax::detail::CauchySymStencil< N, M >;
    EXPECT_EQ( S::offsets.front(), 0u );
    EXPECT_EQ( S::offsets.back(), S::PCs );

    for ( std::size_t k = 0; k < S::NC; ++k )
    {
        for ( std::size_t j = S::offsets[k]; j < S::offsets[k + 1]; ++j )
        {
            const bool diag = ( S::col_a[j] == S::col_b[j] );
            EXPECT_EQ( S::is_diag[j], diag ? 1 : 0 )
                << "k=" << k << " j=" << j;
        }
    }
}

}  // namespace

TEST( CauchyStencil, MatchesEnumeration_N3_M2 )
{
    expectAsymmetricStencilMatchesLive< 3, 2 >();
}
TEST( CauchyStencil, MatchesEnumeration_N5_M3 )
{
    expectAsymmetricStencilMatchesLive< 5, 3 >();
}
TEST( CauchyStencil, MatchesEnumeration_N4_M4 )
{
    expectAsymmetricStencilMatchesLive< 4, 4 >();
}

TEST( CauchySymStencil, MatchesEnumeration_N3_M2 )
{
    expectSymmetricStencilMatchesLive< 3, 2 >();
}
TEST( CauchySymStencil, MatchesEnumeration_N5_M3 )
{
    expectSymmetricStencilMatchesLive< 5, 3 >();
}
TEST( CauchySymStencil, MatchesEnumeration_N4_M4 )
{
    expectSymmetricStencilMatchesLive< 4, 4 >();
}

TEST( CauchyStencil, PairCountMatchesClosedForm )
{
    using S = tax::detail::CauchyStencil< 5, 3 >;
    // numMonomials(5, 6) = C(11, 6) = 462.
    EXPECT_EQ( S::PC, tax::detail::numMonomials( 5, 6 ) );
}

// Cross-check CauchyWeightStencil::db against live |beta| values.
namespace
{
template < int N, int M >
void expectWeightStencilMatchesLive()
{
    using S = tax::detail::CauchyStencil< N, M >;
    using W = tax::detail::CauchyWeightStencil< N, M >;
    EXPECT_EQ( S::PC, W::PC );

    std::size_t pos = 0;
    for ( int d = 0; d <= N; ++d )
    {
        tax::detail::forEachMonomial< M >(
            d, [&]( const auto& alpha, std::size_t /*ai*/ ) {
                tax::detail::forEachSubIndex< M >(
                    alpha, [&]( std::size_t /*bi*/, std::size_t /*gi*/ ) {
                        // |beta| via deconstruction: same as stencil's db at this position.
                        // (We use the banded walk to get db cheaply but as a separate
                        // cross-check, recompute by summing alpha-gamma below.)
                        ++pos;  // We'll just count.
                    } );
            } );
    }
    EXPECT_EQ( pos, W::PC );

    // Now verify db[j] = totalDegree( unflatIndex( col_a[j] ) ).
    for ( std::size_t j = 0; j < W::PC; ++j )
    {
        const auto beta = tax::detail::unflatIndex< M >( std::size_t( S::col_a[j] ) );
        int dbeta = 0;
        for ( int v = 0; v < M; ++v ) dbeta += beta[v];
        EXPECT_EQ( int( W::db[j] ), dbeta ) << "j=" << j;
    }
}
}  // namespace

TEST( CauchyWeightStencil, MatchesLive_N3_M2 )
{
    expectWeightStencilMatchesLive< 3, 2 >();
}
TEST( CauchyWeightStencil, MatchesLive_N5_M3 )
{
    expectWeightStencilMatchesLive< 5, 3 >();
}
TEST( CauchyWeightStencil, MatchesLive_N4_M4 )
{
    expectWeightStencilMatchesLive< 4, 4 >();
}
