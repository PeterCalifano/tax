#include <gtest/gtest.h>

#include <tax/tax.hpp>

TEST( Accessors, CompileTimeCoeff )
{
    typename tax::TE< 3, 2 >::Input p{ 1.0, 2.0 };
    auto x = tax::TE< 3, 2 >::variable< 0 >( p );
    EXPECT_EQ( ( x.coeff< 0, 0 >() ), 1.0 );
    EXPECT_EQ( ( x.coeff< 1, 0 >() ), 1.0 );
    EXPECT_EQ( ( x.coeff< 0, 1 >() ), 0.0 );
}

TEST( Accessors, CompileTimeDerivativeMultipliesByFactorial )
{
    tax::TE< 3 > f;
    f[0] = 1.0;
    f[1] = 2.0;
    f[2] = 3.0;
    EXPECT_EQ( f.coeff( tax::MultiIndex< 1 >{ 2 } ), 3.0 );
    EXPECT_EQ( f.derivative( tax::MultiIndex< 1 >{ 2 } ), 6.0 );
    EXPECT_EQ( f.template derivative< 2 >(), 6.0 );
}

TEST( Accessors, RuntimeDerivativeUni )
{
    tax::TE< 4 > f;
    for ( std::size_t k = 0; k < f.nCoefficients; ++k ) f[k] = double( k );
    EXPECT_EQ( f.derivative( tax::MultiIndex< 1 >{ 0 } ), 0.0 );
    EXPECT_EQ( f.derivative( tax::MultiIndex< 1 >{ 1 } ), 1.0 );
    EXPECT_EQ( f.derivative( tax::MultiIndex< 1 >{ 2 } ), 4.0 );
    EXPECT_EQ( f.derivative( tax::MultiIndex< 1 >{ 3 } ), 18.0 );
}

// Regression: the k! scaling must be accumulated in T, not std::size_t.
// 21! overflows uint64, so the old size_t accumulation wrapped and produced a
// silently-wrong (and far too small) derivative for high orders.
TEST( Accessors, HighOrderDerivativeFactorialNoOverflow )
{
    constexpr int K = 25;
    tax::TE< K > f;
    f[K] = 1.0;  // coefficient of x^K is 1 => d^K/dx^K = K!

    double expected = 1.0;
    for ( int i = 2; i <= K; ++i ) expected *= double( i );  // 25! ~= 1.55e25

    // True factorial is ~1.55e25; a wrapped uint64 result would be < 1.8e19.
    EXPECT_GT( expected, 1e25 );
    EXPECT_NEAR( f.derivative( tax::MultiIndex< 1 >{ K } ), expected, expected * 1e-12 );
    EXPECT_NEAR( f.template derivative< K >(), expected, expected * 1e-12 );
}
