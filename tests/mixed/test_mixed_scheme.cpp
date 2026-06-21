#include <gtest/gtest.h>

#include <set>
#include <tax/core/mixed_scheme.hpp>
#include <tax/core/multi_index.hpp>

using tax::Group;
using tax::MixedScheme;

// Box count = product of per-group simplex sizes; differs from the joint simplex.
TEST( MixedScheme, KeptCountIsBoxProduct )
{
    // x@2 (1 var) box t@2 (1 var): 3 * 3 = 9 (joint simplex numMonomials(2,2)=6).
    static_assert( MixedScheme< Group< 1, 2 >, Group< 1, 2 > >::nCoeff == 9 );
    // x@4 (1 var) box p@20 (1 var): 5 * 21 = 105.
    static_assert( MixedScheme< Group< 1, 4 >, Group< 1, 20 > >::nCoeff == 105 );
    // x@4 over 3 vars (numMonomials(4,3)=35) box t@20: 35 * 21 = 735.
    static_assert( MixedScheme< Group< 3, 4 >, Group< 1, 20 > >::nCoeff == 735 );
    SUCCEED();
}

// flatOf/multiOf are inverse over the whole box, dense in [0,nCoeff), graded.
TEST( MixedScheme, FlatRoundTripDenseAndGraded )
{
    using S = MixedScheme< Group< 1, 2 >, Group< 2, 3 > >;  // x(1 var)@2, y(2 vars)@3
    std::set< std::size_t > seen;
    int prev_degree = -1;
    bool graded = true;
    // Walk all flats; each maps to an in-box multi-index, round-trips, and is graded.
    for ( std::size_t k = 0; k < S::nCoeff; ++k )
    {
        auto a = S::multiOf( k );
        EXPECT_EQ( S::flatOf( a ), k );  // round trip
        // in-box: per-group degree caps
        EXPECT_LE( a[0], 2 );         // x degree
        EXPECT_LE( a[1] + a[2], 3 );  // y total degree
        int d = tax::totalDegree( a );
        if ( d < prev_degree ) graded = false;  // non-decreasing total degree
        prev_degree = d;
        seen.insert( k );
    }
    EXPECT_TRUE( graded );
    EXPECT_EQ( seen.size(), S::nCoeff );  // dense, no gaps
}

// An out-of-box multi-index is rejected.
TEST( MixedScheme, OutOfBoxRejected )
{
    using S = MixedScheme< Group< 1, 2 >, Group< 1, 2 > >;
    tax::MultiIndex< 2 > over{ 3, 0 };  // x^3, x order is 2
    EXPECT_EQ( S::flatOf( over ), S::kNotInBox );
}
