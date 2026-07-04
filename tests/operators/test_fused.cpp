#include <gtest/gtest.h>

#include <cmath>

#include "../testUtils.hpp"

// ---------------------------------------------------------------------------
// Fused pair operations: sinCos, sinhCosh, sqrtInvSqrt, expSin/expCos/expSinCos.
// The fused kernels sum the same mathematical terms in a different order than
// the compose-then-multiply path, so cross-checks use a small tolerance.
// ---------------------------------------------------------------------------

namespace
{

template < typename E >
void expectNear( const E& a, const E& b, double tol )
{
    for ( std::size_t k = 0; k < E::nCoefficients; ++k )
        EXPECT_NEAR( a[k], b[k], tol * ( 1.0 + std::abs( a[k] ) ) ) << "k = " << k;
}

}  // namespace

TEST( SinCosPair, MatchesSinAndCosExactly )
{
    auto x = tax::TE< 7 >::variable( 0.6 );
    auto [s, c] = tax::sinCos( x );
    // Same kernel as sin() / cos(): bit-identical.
    auto se = tax::sin( x );
    auto ce = tax::cos( x );
    for ( std::size_t k = 0; k < decltype( s )::nCoefficients; ++k )
    {
        EXPECT_EQ( s[k], se[k] );
        EXPECT_EQ( c[k], ce[k] );
    }
}

TEST( SinhCoshPair, MatchesSinhAndCoshExactly )
{
    auto x = tax::TE< 7 >::variable( -0.4 );
    auto [s, c] = tax::sinhCosh( x );
    auto se = tax::sinh( x );
    auto ce = tax::cosh( x );
    for ( std::size_t k = 0; k < decltype( s )::nCoefficients; ++k )
    {
        EXPECT_EQ( s[k], se[k] );
        EXPECT_EQ( c[k], ce[k] );
    }
}

TEST( SqrtInvSqrtPair, MatchesSqrtAndReciprocalSqrt )
{
    auto x = tax::TE< 9 >::variable( 2.0 );
    auto [s, r] = tax::sqrtInvSqrt( x );
    expectNear( s, tax::sqrt( x ), 1e-13 );
    expectNear( r, tax::reciprocal( tax::sqrt( x ) ), 1e-12 );
    // r * s == 1 as a series.
    auto one = r * s;
    EXPECT_NEAR( one.value(), 1.0, 1e-13 );
    for ( std::size_t k = 1; k < decltype( one )::nCoefficients; ++k )
        EXPECT_NEAR( one[k], 0.0, 1e-13 );
}

TEST( SqrtInvSqrtPair, Multivariate )
{
    using E = tax::TE< 4, 3 >;
    typename E::Input p{ 1.5, -0.25, 0.75 };
    auto x = E::variable< 0 >( p );
    auto y = E::variable< 1 >( p );
    auto z = E::variable< 2 >( p );
    auto u = tax::square( x ) + tax::square( y ) + tax::square( z );
    auto [s, r] = tax::sqrtInvSqrt( u );
    expectNear( s, tax::sqrt( u ), 1e-12 );
    expectNear( r, tax::pow( u, -0.5 ), 1e-11 );
}

TEST( ExpTrig, UnivariateMatchesComposition )
{
    auto x = tax::TE< 10 >::variable( 0.8 );
    auto v = 0.5 * x;
    auto u = tax::square( x ) - 1.0;
    expectNear( tax::expSin( v, u ), tax::exp( v ) * tax::sin( u ), 1e-11 );
    expectNear( tax::expCos( v, u ), tax::exp( v ) * tax::cos( u ), 1e-11 );
}

TEST( ExpTrig, PairMatchesSingles )
{
    auto x = tax::TE< 8 >::variable( 0.3 );
    auto [es, ec] = tax::expSinCos( x, x );
    auto s = tax::expSin( x, x );
    auto c = tax::expCos( x, x );
    for ( std::size_t k = 0; k < decltype( s )::nCoefficients; ++k )
    {
        EXPECT_EQ( es[k], s[k] );
        EXPECT_EQ( ec[k], c[k] );
    }
}

TEST( ExpTrig, MultivariateMatchesComposition )
{
    using E = tax::TE< 5, 2 >;
    typename E::Input p{ 0.4, -0.9 };
    auto x = E::variable< 0 >( p );
    auto y = E::variable< 1 >( p );
    auto v = x * y;
    auto u = x - tax::square( y );
    expectNear( tax::expSin( v, u ), tax::exp( v ) * tax::sin( u ), 1e-11 );
    expectNear( tax::expCos( v, u ), tax::exp( v ) * tax::cos( u ), 1e-11 );
}

// ---------------------------------------------------------------------------
// Named / mixed-order named surface
// ---------------------------------------------------------------------------

TEST( FusedNamed, ExpSinComposesInAxisUnion )
{
    auto x = tax::variable< "x", 6 >( 0.2 );
    auto y = tax::variable< "y", 6 >( 1.1 );
    auto f = tax::expSin( x, y );  // exp(x) * sin(y) over axes {x, y}
    auto g = tax::exp( x ) * tax::sin( y );
    static_assert( std::is_same_v< decltype( f ), decltype( g ) > );
    for ( std::size_t k = 0; k < decltype( f )::nCoefficients; ++k )
        EXPECT_NEAR( f.inner()[k], g.inner()[k], 1e-12 );
}

TEST( FusedNamed, PairFunctionsPreserveAxes )
{
    auto x = tax::variable< "x", 5 >( 0.7 );
    auto [s, c] = tax::sinCos( x );
    static_assert( std::is_same_v< decltype( s ), decltype( x ) > );
    EXPECT_NEAR( s.value(), std::sin( 0.7 ), 1e-15 );
    EXPECT_NEAR( c.value(), std::cos( 0.7 ), 1e-15 );

    auto [sq, isq] = tax::sqrtInvSqrt( x + 1.0 );
    EXPECT_NEAR( sq.value(), std::sqrt( 1.7 ), 1e-15 );
    EXPECT_NEAR( isq.value(), 1.0 / std::sqrt( 1.7 ), 1e-15 );
}

TEST( FusedMixed, ExpCosComposesInOrderedAxisUnion )
{
    auto x = tax::mixed::variable< "x", 4 >( 0.5 );
    auto y = tax::mixed::variable< "y", 3 >( -0.2 );
    auto f = tax::expCos( x, y );
    auto g = tax::exp( x ) * tax::cos( y );
    static_assert( std::is_same_v< decltype( f ), decltype( g ) > );
    for ( std::size_t k = 0; k < decltype( f )::nCoefficients; ++k )
        EXPECT_NEAR( f.inner()[k], g.inner()[k], 1e-12 );
}

// ---------------------------------------------------------------------------
// Mixed pow / atan2 (new binary surface)
// ---------------------------------------------------------------------------

TEST( MixedBinaryMath, PowAndAtan2 )
{
    auto x = tax::mixed::variable< "x", 4 >( 1.2 );
    auto y = tax::mixed::variable< "y", 4 >( 0.8 );

    auto p = tax::pow( x, 3 );
    auto pc = x * x * x;
    for ( std::size_t k = 0; k < decltype( p )::nCoefficients; ++k )
        EXPECT_NEAR( p.inner()[k], pc.inner()[k], 1e-12 );

    auto pr = tax::pow( x, 0.5 );
    auto sq = tax::sqrt( x );
    for ( std::size_t k = 0; k < decltype( pr )::nCoefficients; ++k )
        EXPECT_NEAR( pr.inner()[k], sq.inner()[k], 1e-12 );

    auto a = tax::atan2( y, x );
    auto ac = tax::atan( y / x );
    static_assert( std::is_same_v< decltype( a ), decltype( ac ) > );
    EXPECT_NEAR( a.value(), std::atan2( 0.8, 1.2 ), 1e-15 );
    for ( std::size_t k = 0; k < decltype( a )::nCoefficients; ++k )
        EXPECT_NEAR( a.inner()[k], ac.inner()[k], 1e-12 );
}

// ---------------------------------------------------------------------------
// Fused kernels in constant evaluation
// ---------------------------------------------------------------------------

namespace
{

constexpr bool ceCoeffsNear( const tax::TE< 8 >& a, const tax::TE< 8 >& b, double tol )
{
    for ( std::size_t k = 0; k < tax::TE< 8 >::nCoefficients; ++k )
    {
        const double m = a[k] < 0 ? -a[k] : a[k];
        const double d = a[k] > b[k] ? a[k] - b[k] : b[k] - a[k];
        if ( d > tol * ( 1.0 + m ) ) return false;
    }
    return true;
}

constexpr auto kCx = tax::TE< 8 >::variable( 0.5 );
static_assert( ceCoeffsNear( tax::expSin( kCx, kCx ), tax::exp( kCx ) * tax::sin( kCx ), 1e-12 ) );
static_assert( ceCoeffsNear( tax::expCos( kCx, kCx ), tax::exp( kCx ) * tax::cos( kCx ), 1e-12 ) );
static_assert( ceCoeffsNear( tax::sqrtInvSqrt( kCx + 1.0 ).first, tax::sqrt( kCx + 1.0 ),
                             1e-14 ) );
static_assert( ceCoeffsNear( tax::sqrtInvSqrt( kCx + 1.0 ).second,
                             tax::reciprocal( tax::sqrt( kCx + 1.0 ) ), 1e-13 ) );

}  // namespace
