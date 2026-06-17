#include <gtest/gtest.h>

#include <cmath>

#include "../testUtils.hpp"

TEST( Asin, RoundTrip )
{
    auto x = tax::TE< 5 >::variable( 0.3 );
    tax::test::ExpectCoeffsNear( tax::asin( tax::sin( x ) ), x, 1e-10 );
}

TEST( Acos, RoundTrip )
{
    auto x = tax::TE< 5 >::variable( 0.3 );
    tax::test::ExpectCoeffsNear( tax::acos( tax::cos( x ) ), x, 1e-10 );
}

TEST( Atan, RoundTrip )
{
    auto x = tax::TE< 5 >::variable( 0.4 );
    tax::test::ExpectCoeffsNear( tax::atan( tax::tan( x ) ), x, 1e-10 );
}

TEST( Atan2, ConsistentWithAtan )
{
    auto x = tax::TE< 5 >::variable( 0.6 );
    auto y = tax::TE< 5 >::variable( 0.8 );
    auto a = tax::atan2( y, x );
    auto b = tax::atan( y / x );
    tax::test::ExpectCoeffsNear( a, b, 1e-12 );
}

// Mixed-scalar atan2 overloads must match promoting the scalar to a constant TE.
TEST( Atan2, MixedScalarOverloads )
{
    auto y = tax::TE< 5 >::variable( 0.8 );
    auto x = tax::TE< 5 >::variable( 0.6 );

    tax::test::ExpectCoeffsNear( tax::atan2( y, 0.6 ), tax::atan2( y, tax::TE< 5 >{ 0.6 } ),
                                 1e-12 );
    tax::test::ExpectCoeffsNear( tax::atan2( 0.8, x ), tax::atan2( tax::TE< 5 >{ 0.8 }, x ),
                                 1e-12 );
}
