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
