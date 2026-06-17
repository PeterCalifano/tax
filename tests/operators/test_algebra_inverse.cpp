#include <gtest/gtest.h>

#include "../testUtils.hpp"

TEST( Sqrt, RoundTripWithSquare )
{
    auto x = tax::TE< 5 >::variable( 4.0 );
    auto s = tax::sqrt( x );
    auto r = s * s;
    tax::test::ExpectCoeffsNear( r, x, 1e-12 );
}

TEST( Cbrt, RoundTripWithCube )
{
    auto x = tax::TE< 5 >::variable( 2.0 );
    auto c = tax::cbrt( x );
    auto r = c * c * c;
    tax::test::ExpectCoeffsNear( r, x, 1e-12 );
}

TEST( Reciprocal, MultipliesToOne )
{
    auto x = tax::TE< 5 >::variable( 2.0 );
    auto i = tax::reciprocal( x );
    auto p = x * i;
    EXPECT_NEAR( p.value(), 1.0, 1e-12 );
    for ( std::size_t k = 1; k < p.nCoefficients; ++k ) EXPECT_NEAR( p[k], 0.0, 1e-12 );
}

TEST( Divide, MatchesReciprocal )
{
    auto x = tax::TE< 5 >::variable( 3.0 );
    auto y = tax::TE< 5 >::variable( 2.0 );
    auto a = x / y;
    auto b = x * tax::reciprocal( y );
    tax::test::ExpectCoeffsNear( a, b, 1e-12 );
}

// seriesDivide's multivariate (M >= 2) path must agree with reciprocal*multiply,
// and b * (a / b) must recover a.
TEST( Divide, MultivariateMatchesReciprocalAndRoundTrips )
{
    typename tax::TE< 4, 2 >::Input p{ 1.5, 0.7 };
    auto x = tax::TE< 4, 2 >::variable< 0 >( p );
    auto y = tax::TE< 4, 2 >::variable< 1 >( p );
    auto b = y + 2.0;  // nonzero constant term

    auto q = ( x + 1.0 ) / b;
    tax::test::ExpectCoeffsNear( q, ( x + 1.0 ) * tax::reciprocal( b ), 1e-12 );
    tax::test::ExpectCoeffsNear( b * q, x + 1.0, 1e-12 );
}
