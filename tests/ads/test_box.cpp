// tests/ads/test_box.cpp
//
// Box<T, M> — construction, contains, split, denormalize, Eigen overloads.

#include <gtest/gtest.h>

#include <tax/ads/box.hpp>
#include <tax/la/types.hpp>

using tax::ads::Box;

TEST( AdsBox, DefaultCtorZero )
{
    constexpr Box< double, 2 > b{};
    EXPECT_EQ( b.center[ 0 ], 0.0 );
    EXPECT_EQ( b.center[ 1 ], 0.0 );
    EXPECT_EQ( b.halfWidth[ 0 ], 0.0 );
    EXPECT_EQ( b.halfWidth[ 1 ], 0.0 );
}

TEST( AdsBox, ArrayCtor )
{
    constexpr Box< double, 2 > b{ { 1.0, 2.0 }, { 0.5, 0.25 } };
    EXPECT_EQ( b.center[ 0 ], 1.0 );
    EXPECT_EQ( b.halfWidth[ 1 ], 0.25 );
}

TEST( AdsBox, ContainsInclusiveBoundary )
{
    constexpr Box< double, 2 > b{ { 0.0, 0.0 }, { 1.0, 1.0 } };
    EXPECT_TRUE( b.contains( { 0.5, -0.5 } ) );
    EXPECT_TRUE( b.contains( { 1.0, 1.0 } ) );    // on boundary
    EXPECT_TRUE( b.contains( { -1.0, -1.0 } ) );
    EXPECT_FALSE( b.contains( { 1.001, 0.0 } ) );
    EXPECT_FALSE( b.contains( { 0.0, -1.001 } ) );
}

TEST( AdsBox, SplitHalvesOnlyRequestedAxis )
{
    constexpr Box< double, 2 > b{ { 0.0, 0.0 }, { 1.0, 2.0 } };
    constexpr auto             pr = b.split( 0 );
    const auto&                L  = pr.first;
    const auto&                R  = pr.second;
    EXPECT_DOUBLE_EQ( L.center[ 0 ], -0.5 );
    EXPECT_DOUBLE_EQ( R.center[ 0 ],  0.5 );
    EXPECT_DOUBLE_EQ( L.halfWidth[ 0 ], 0.5 );
    EXPECT_DOUBLE_EQ( R.halfWidth[ 0 ], 0.5 );
    // Untouched axis.
    EXPECT_DOUBLE_EQ( L.center[ 1 ], 0.0 );
    EXPECT_DOUBLE_EQ( R.center[ 1 ], 0.0 );
    EXPECT_DOUBLE_EQ( L.halfWidth[ 1 ], 2.0 );
    EXPECT_DOUBLE_EQ( R.halfWidth[ 1 ], 2.0 );
}

TEST( AdsBox, Denormalize )
{
    constexpr Box< double, 2 > b{ { 1.0, 2.0 }, { 0.5, 0.25 } };
    constexpr auto             pt = b.denormalize( { 1.0, -1.0 } );
    EXPECT_DOUBLE_EQ( pt[ 0 ], 1.5 );    // 1.0 + 0.5
    EXPECT_DOUBLE_EQ( pt[ 1 ], 1.75 );   // 2.0 - 0.25
}

TEST( AdsBox, EigenCtorAndAccessors )
{
    using V = tax::la::VecNT< 2, double >;
    V c; c << 1.0, 2.0;
    V h; h << 0.5, 0.25;
    Box< double, 2 > b{ c, h };
    EXPECT_EQ( b.center[ 0 ], 1.0 );
    EXPECT_EQ( b.halfWidth[ 1 ], 0.25 );
    EXPECT_EQ( b.centerEigen()( 0 ),    1.0 );
    EXPECT_EQ( b.halfWidthEigen()( 1 ), 0.25 );
}

TEST( AdsBox, EigenContainsAndDenormalize )
{
    using V = tax::la::VecNT< 2, double >;
    Box< double, 2 > b{ { 0.0, 0.0 }, { 1.0, 1.0 } };
    V pt; pt << 0.5, -0.5;
    EXPECT_TRUE( b.contains( pt ) );
    V dn; dn << 1.0, -1.0;
    auto out = b.denormalize( dn );
    EXPECT_DOUBLE_EQ( out( 0 ),  1.0 );
    EXPECT_DOUBLE_EQ( out( 1 ), -1.0 );
}
