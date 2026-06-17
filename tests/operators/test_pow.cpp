#include <gtest/gtest.h>

#include "../testUtils.hpp"

TEST( Pow, IntegerExponent )
{
    auto x = tax::TE< 6 >::variable( 2.0 );
    auto a = tax::pow( x, 3 );
    auto b = x * x * x;
    tax::test::ExpectCoeffsNear( a, b, 1e-12 );
}

TEST( Pow, RealExponent )
{
    auto x = tax::TE< 5 >::variable( 4.0 );
    auto a = tax::pow( x, 0.5 );  // sqrt
    auto b = tax::sqrt( x );
    tax::test::ExpectCoeffsNear( a, b, 1e-12 );
}

TEST( Pow, NegativeRealExponent )
{
    auto x = tax::TE< 5 >::variable( 2.0 );
    auto a = tax::pow( x, -1.0 );
    auto b = tax::reciprocal( x );
    tax::test::ExpectCoeffsNear( a, b, 1e-12 );
}

// A float exponent (with T == double) used to be ambiguous between pow(.,int)
// and pow(.,T); the floating-point-constrained overload now binds by exact match.
TEST( Pow, FloatExponentNotAmbiguous )
{
    auto x = tax::TE< 6 >::variable( 2.0 );
    auto a = tax::pow( x, 2.0f );  // must compile and pick the real-exponent overload
    auto b = x * x;
    tax::test::ExpectCoeffsNear( a, b, 1e-12 );
}

// pow(a, b) with a Taylor-valued exponent: a^b = exp(b*log(a)). A constant
// exponent must reproduce the real-scalar pow.
TEST( Pow, TaylorExponentConstantMatchesRealPow )
{
    auto a = tax::TE< 6 >::variable( 3.0 );
    tax::TE< 6 > b{ 2.5 };  // constant exponent
    auto via_te = tax::pow( a, b );
    auto via_real = tax::pow( a, 2.5 );
    tax::test::ExpectCoeffsNear( via_te, via_real, 1e-10 );
}

// pow(scalar, TE): 2^x at x0 = 3 has value 8 and derivative 8*ln(2).
TEST( Pow, ScalarBaseTaylorExponent )
{
    auto x = tax::TE< 4 >::variable( 3.0 );
    auto f = tax::pow( 2.0, x );
    EXPECT_NEAR( f.value(), 8.0, 1e-12 );
    EXPECT_NEAR( ( f.template derivative< 1 >() ), 8.0 * std::log( 2.0 ), 1e-10 );
}
