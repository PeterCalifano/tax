#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <tax/tax.hpp>

using tax::Batch;

namespace
{
constexpr int K = 4;

// Per-lane expansion centre for univariate tests.
double center( int k ) { return 0.25 + 0.05 * double( k ); }
}  // namespace

// Trait sanity: floating types still satisfy Scalar; real_scalar is identity
// for doubles and the lane type for batches.
static_assert( tax::Scalar< double > );
static_assert( tax::Scalar< tax::Batch< double, 4 > > );
static_assert( std::is_same_v< tax::real_scalar_t< double >, double > );
static_assert( std::is_same_v< tax::real_scalar_t< tax::Batch< double, 4 > >, double > );

TEST( Batch, ScalarLaneArithmetic )
{
    Batch< double, K > a, b;
    for ( int k = 0; k < K; ++k )
    {
        a[k] = 1.0 + k;
        b[k] = 0.5 * ( k + 1 );
    }
    const auto s = a + b;
    const auto p = a * b;
    const auto q = a / b;
    for ( int k = 0; k < K; ++k )
    {
        EXPECT_DOUBLE_EQ( s[k], a[k] + b[k] );
        EXPECT_DOUBLE_EQ( p[k], a[k] * b[k] );
        EXPECT_DOUBLE_EQ( q[k], a[k] / b[k] );
    }
    EXPECT_TRUE( a == a );
    EXPECT_TRUE( a != b );
}

TEST( Batch, ScalarBroadcastAndFromLanes )
{
    Batch< double, K > c( 3.0 );
    for ( int k = 0; k < K; ++k ) EXPECT_DOUBLE_EQ( c[k], 3.0 );
    auto d = Batch< double, K >::fromLanes( { 1.0, 2.0, 3.0, 4.0 } );
    for ( int k = 0; k < K; ++k ) EXPECT_DOUBLE_EQ( d[k], double( k + 1 ) );
}

// A batched expansion built through the unified TE<N, M, K> alias evaluates,
// per lane, exactly like the matching scalar TE<N, M>.
TEST( Batch, UnifiedAliasConstructAndValue )
{
    constexpr int N = 5;
    using TEb = tax::TE< N, 1, K >;  // batched
    using TEs = tax::TE< N, 1 >;     // scalar (K defaults to 1 -> double)

    static_assert( std::is_same_v< typename TEs::scalar_type, double > );
    static_assert( std::is_same_v< typename TEb::scalar_type, tax::Batch< double, K > > );

    typename TEb::Input pb{};
    Batch< double, K > c0;
    for ( int k = 0; k < K; ++k ) c0[k] = center( k );
    pb[0] = c0;
    TEb xb = TEb::template variable< 0 >( pb );

    for ( int k = 0; k < K; ++k )
    {
        typename TEs::Input ps{ center( k ) };
        TEs xs = TEs::template variable< 0 >( ps );
        EXPECT_DOUBLE_EQ( xb.value()[k], xs.value() );
        EXPECT_DOUBLE_EQ( xb[1][k], xs[1] );  // linear coefficient
    }
}

// Compare every coefficient of a batched univariate result against the K
// independent scalar runs it should reproduce.
namespace
{
template < int N, class Fn >
double laneMaxErr( Fn f )
{
    using TEs = tax::TE< N, 1 >;
    using TEb = tax::TE< N, 1, K >;

    typename TEb::Input pb{};
    Batch< double, K > c0;
    for ( int k = 0; k < K; ++k ) c0[k] = center( k );
    pb[0] = c0;
    TEb xb = TEb::template variable< 0 >( pb );
    const TEb rb = f( xb );

    double max_err = 0.0;
    for ( int k = 0; k < K; ++k )
    {
        typename TEs::Input ps{ center( k ) };
        TEs xs = TEs::template variable< 0 >( ps );
        const TEs rs = f( xs );
        for ( std::size_t c = 0; c < TEs::nCoefficients; ++c )
        {
            const double d = std::abs( rb[c][k] - rs[c] ) / ( 1.0 + std::abs( rs[c] ) );
            max_err = std::max( max_err, d );
        }
    }
    return max_err;
}
}  // namespace

TEST( Batch, MathSurfaceLaneEquivalence )
{
    using tax::acos;
    using tax::asin;
    using tax::atan;
    using tax::atan2;
    using tax::cbrt;
    using tax::cos;
    using tax::cosh;
    using tax::erf;
    using tax::exp;
    using tax::log;
    using tax::pow;
    using tax::sin;
    using tax::sinh;
    using tax::sqrt;
    using tax::tan;
    using tax::tanh;

    constexpr double tol = 1e-12;
    EXPECT_LT( laneMaxErr< 6 >( []( auto x ) { return x * x * x + x - 2.0; } ), tol );
    EXPECT_LT( laneMaxErr< 6 >( []( auto x ) { return ( x * x + 1.0 ) / ( x + 2.0 ); } ), tol );
    EXPECT_LT( laneMaxErr< 6 >( []( auto x ) { return sin( x ) * cos( x ) + tan( x ); } ), tol );
    EXPECT_LT( laneMaxErr< 6 >( []( auto x ) { return asin( x ) + acos( x ) + atan( x ); } ), tol );
    EXPECT_LT( laneMaxErr< 6 >( []( auto x ) { return exp( x ) + log( x + 2.0 ); } ), tol );
    EXPECT_LT( laneMaxErr< 6 >( []( auto x ) { return sinh( x ) + cosh( x ) + tanh( x ); } ), tol );
    EXPECT_LT( laneMaxErr< 6 >( []( auto x ) { return sqrt( x + 1.0 ) + cbrt( x + 1.0 ); } ), tol );
    EXPECT_LT( laneMaxErr< 6 >( []( auto x ) { return erf( x ); } ), tol );
    EXPECT_LT( laneMaxErr< 6 >( []( auto x ) { return atan2( x + 1.0, x + 2.0 ); } ), tol );
    EXPECT_LT( laneMaxErr< 6 >( []( auto x ) { return pow( x + 1.0, 3 ); } ), tol );    // int
    EXPECT_LT( laneMaxErr< 6 >( []( auto x ) { return pow( x + 1.0, 2.5 ); } ), tol );  // real
}

TEST( Batch, IntegerVsRealPowSelectsCorrectKernel )
{
    using tax::pow;
    using TEb = tax::TE< 5, 1, K >;
    typename TEb::Input pb{};
    pb[0] = Batch< double, K >( 0.3 );
    auto x = TEb::template variable< 0 >( pb ) + 1.0;  // centre 1.3
    auto pi = pow( x, 2 );                             // integer exponent
    auto pr = pow( x, 2.0 );                           // real exponent, same value
    for ( int k = 0; k < K; ++k )
    {
        EXPECT_NEAR( pi[0][k], 1.3 * 1.3, 1e-12 );
        EXPECT_NEAR( pr[0][k], std::pow( 1.3, 2.0 ), 1e-12 );
        for ( std::size_t c = 0; c < TEb::nCoefficients; ++c )
            EXPECT_NEAR( pi[c][k], pr[c][k], 1e-10 );
    }
}

TEST( Batch, EvalPerLaneDisplacement )
{
    constexpr int N = 5;
    using TEb = tax::TE< N, 1, K >;
    using TEs = tax::TE< N, 1 >;
    using tax::sin;

    typename TEb::Input pb{};
    Batch< double, K > c0;
    for ( int k = 0; k < K; ++k ) c0[k] = center( k );
    pb[0] = c0;
    auto fb = sin( TEb::template variable< 0 >( pb ) );

    Batch< double, K > dxb;
    for ( int k = 0; k < K; ++k ) dxb[k] = 0.01 * ( k + 1 );
    typename TEb::Input db{ dxb };
    auto eb = fb.eval( db );

    for ( int k = 0; k < K; ++k )
    {
        typename TEs::Input ps{ center( k ) };
        auto fs = sin( TEs::template variable< 0 >( ps ) );
        typename TEs::Input ds{ 0.01 * ( k + 1 ) };
        EXPECT_NEAR( eb[k], fs.eval( ds ), 1e-12 );
    }
}
