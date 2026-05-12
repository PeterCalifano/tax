#include <array>

#include "testUtils.hpp"

using tax::detail::DegreeOf;
using tax::detail::numMonomials;
using tax::detail::totalDegree;
using tax::detail::unflatIndex;

// =============================================================================
// DegreeOf<N, M>::value[k] equals totalDegree(unflatIndex<M>(k))
// =============================================================================

template < int N, int M >
static void ExpectDegreeOfTableMatchesUnflat()
{
    constexpr std::size_t NC = numMonomials( N, M );
    static_assert( DegreeOf< N, M >::NC == NC );

    for ( std::size_t k = 0; k < NC; ++k )
    {
        const auto alpha = unflatIndex< M >( k );
        const int expected = totalDegree< M >( alpha );
        EXPECT_EQ( int( DegreeOf< N, M >::value[k] ), expected ) << "N=" << N << " M=" << M
                                                                  << " k=" << k;
    }
}

TEST( DegreeOf, Univariate )
{
    ExpectDegreeOfTableMatchesUnflat< 0, 1 >();
    ExpectDegreeOfTableMatchesUnflat< 1, 1 >();
    ExpectDegreeOfTableMatchesUnflat< 5, 1 >();
    ExpectDegreeOfTableMatchesUnflat< 12, 1 >();
}

TEST( DegreeOf, Multivariate )
{
    ExpectDegreeOfTableMatchesUnflat< 2, 2 >();
    ExpectDegreeOfTableMatchesUnflat< 3, 3 >();
    ExpectDegreeOfTableMatchesUnflat< 4, 4 >();
    ExpectDegreeOfTableMatchesUnflat< 6, 5 >();
    ExpectDegreeOfTableMatchesUnflat< 8, 6 >();
}

// =============================================================================
// Spot-check: the first NC entries enumerate degrees 0, 1, 1, ..., N
// =============================================================================

TEST( DegreeOf, DegreeBlocksAreContiguous )
{
    // For graded-lex, all monomials of degree d occupy a contiguous slice.
    constexpr int N = 5;
    constexpr int M = 3;
    auto& T = DegreeOf< N, M >::value;

    for ( std::size_t k = 1; k < T.size(); ++k )
        EXPECT_LE( int( T[k - 1] ), int( T[k] ) ) << "graded-lex not monotone at k=" << k;
}

TEST( DegreeOf, FirstSlotIsConstant )
{
    EXPECT_EQ( int( DegreeOf< 0, 1 >::value[0] ), 0 );
    EXPECT_EQ( int( DegreeOf< 5, 4 >::value[0] ), 0 );
}
