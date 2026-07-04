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

// A float exponent (with T == double) must bind the real-exponent overload by
// exact match, without ambiguity against pow(., int).
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

// ---------------------------------------------------------------------------
// halfPow<K> (x^(K/2)) and invSqrtPow<K> (x^(-K/2))
// ---------------------------------------------------------------------------

TEST( HalfPow, OddPositiveMatchesSqrtChain )
{
    auto x = tax::TE< 6 >::variable( 2.5 );
    // x^(3/2) == sqrt(x)^3
    tax::test::ExpectCoeffsNear( tax::halfPow< 3 >( x ), tax::cube( tax::sqrt( x ) ), 1e-12 );
    // x^(1/2) == sqrt(x)
    tax::test::ExpectCoeffsNear( tax::halfPow< 1 >( x ), tax::sqrt( x ), 1e-12 );
}

TEST( HalfPow, EvenDispatchesToIntegerPow )
{
    auto x = tax::TE< 6 >::variable( 1.7 );
    // Even K goes through the integer chain: bit-identical to pow(x, K/2).
    auto a = tax::halfPow< 4 >( x );
    auto b = tax::pow( x, 2 );
    for ( std::size_t k = 0; k < decltype( a )::nCoefficients; ++k ) EXPECT_EQ( a[k], b[k] );
    // ... including for a negative base, where a real exponent would NaN.
    auto n = tax::TE< 4 >::variable( -2.0 );
    tax::test::ExpectCoeffsNear( tax::halfPow< 6 >( n ), tax::cube( n ), 1e-12 );
}

TEST( HalfPow, NegativeAndZeroExponents )
{
    auto x = tax::TE< 6 >::variable( 2.0 );
    tax::test::ExpectCoeffsNear( tax::halfPow< -1 >( x ), tax::reciprocal( tax::sqrt( x ) ),
                                 1e-12 );
    tax::test::ExpectCoeffsNear( tax::halfPow< -2 >( x ), tax::reciprocal( x ), 1e-12 );
    auto one = tax::halfPow< 0 >( x );
    EXPECT_EQ( one.value(), 1.0 );
    for ( std::size_t k = 1; k < decltype( one )::nCoefficients; ++k ) EXPECT_EQ( one[k], 0.0 );
}

TEST( InvSqrtPow, GravityKernelMultivariate )
{
    using E = tax::TE< 4, 3 >;
    typename E::Input p{ 1.0, -2.0, 0.5 };
    auto x = E::variable< 0 >( p );
    auto y = E::variable< 1 >( p );
    auto z = E::variable< 2 >( p );
    auto r2 = tax::square( x ) + tax::square( y ) + tax::square( z );
    // 1/r^3 = (r^2)^(-3/2), the classic gravity kernel.
    auto a = tax::invSqrtPow< 3 >( r2 );
    auto b = tax::reciprocal( tax::cube( tax::sqrt( r2 ) ) );
    tax::test::ExpectCoeffsNear( a, b, 1e-11 );
    // Bit-identical to the pow(-1.5) spelling it binds to.
    auto c = tax::pow( r2, -1.5 );
    for ( std::size_t k = 0; k < decltype( a )::nCoefficients; ++k ) EXPECT_EQ( a[k], c[k] );
}

TEST( InvSqrtPow, NamedAndMixedPreserveAxes )
{
    auto xn = tax::variable< "x", 5 >( 3.0 );
    auto an = tax::invSqrtPow< 3 >( xn );
    static_assert( std::is_same_v< decltype( an ), decltype( xn ) > );
    EXPECT_NEAR( an.value(), std::pow( 3.0, -1.5 ), 1e-15 );
    auto hn = tax::halfPow< 5 >( xn );
    EXPECT_NEAR( hn.value(), std::pow( 3.0, 2.5 ), 1e-13 );

    auto xm = tax::mixed::variable< "x", 4 >( 3.0 );
    auto am = tax::invSqrtPow< 3 >( xm );
    static_assert( std::is_same_v< decltype( am ), decltype( xm ) > );
    EXPECT_NEAR( am.value(), std::pow( 3.0, -1.5 ), 1e-15 );
}

// Compile-time evaluation.
namespace
{

constexpr bool hpCoeffsNear( const tax::TE< 6 >& a, const tax::TE< 6 >& b, double tol )
{
    for ( std::size_t k = 0; k < tax::TE< 6 >::nCoefficients; ++k )
    {
        const double m = a[k] < 0 ? -a[k] : a[k];
        const double d = a[k] > b[k] ? a[k] - b[k] : b[k] - a[k];
        if ( d > tol * ( 1.0 + m ) ) return false;
    }
    return true;
}

constexpr auto kHx = tax::TE< 6 >::variable( 2.0 );
static_assert( hpCoeffsNear( tax::halfPow< 3 >( kHx ), tax::cube( tax::sqrt( kHx ) ), 1e-13 ) );
static_assert( hpCoeffsNear( tax::invSqrtPow< 3 >( kHx ),
                             tax::reciprocal( tax::cube( tax::sqrt( kHx ) ) ), 1e-13 ) );
static_assert( hpCoeffsNear( tax::halfPow< 4 >( kHx ), tax::square( kHx ), 1e-14 ) );

}  // namespace
