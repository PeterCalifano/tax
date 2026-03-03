#include <dace/dace.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <tax/tax.hpp>

template < int N >
::testing::AssertionResult expectCoeffsMatch( const tax::DA< N >& tested, const DACE::DA& ref,
                                              double tol = 1e-12 )
{
    const auto& tax_coeffs = tested.coeffs();  // expected size N+1

    for ( unsigned int k = 0; k <= N; ++k )
    {
        const double c_dace = ref.getCoefficient( { k } );  // 1D multi-index
        const double c_tax = static_cast< double >( tax_coeffs[k] );

        if ( !( std::isfinite( c_dace ) && std::isfinite( c_tax ) ) )
        {
            return ::testing::AssertionFailure() << "Non-finite coefficient at k=" << k
                                                 << " (DACE=" << c_dace << ", tax=" << c_tax << ")";
        }

        const double diff = std::abs( c_dace - c_tax );
        if ( diff > tol )
        {
            return ::testing::AssertionFailure()
                   << "Coefficient mismatch at k=" << k << " (DACE=" << std::setprecision( 17 )
                   << c_dace << ", tax=" << std::setprecision( 17 ) << c_tax << ", |diff|=" << diff
                   << ", tol=" << tol << ")";
        }
    }

    return ::testing::AssertionSuccess();
}

// Operators

TEST( DaceUnivariate, Div )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = 1 / ( 1 + xr );

    auto x = tax::DA< N >::variable( 1.0 );
    tax::DA< N > y = 1.0 / x;

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, MulDiv )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = 1 / ( ( 1 + xr ) * ( 1 + xr ) );

    auto x = tax::DA< N >::variable( 1.0 );
    tax::DA< N > y = 1.0 / ( x * x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

// Math

TEST( DaceUnivariate, Cos )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = xr.cos();

    auto x = tax::DA< N >::variable( 0.0 );
    tax::DA< N > y = tax::cos( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Sin )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = xr.sin();

    auto x = tax::DA< N >::variable( 0.0 );
    tax::DA< N > y = tax::sin( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Tan )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = xr.tan();

    auto x = tax::DA< N >::variable( 0.0 );
    tax::DA< N > y = tax::tan( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, ASin )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = xr.asin();

    auto x = tax::DA< N >::variable( 0.0 );
    tax::DA< N > y = tax::asin( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, ACos )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = xr.acos();

    auto x = tax::DA< N >::variable( 0.0 );
    tax::DA< N > y = tax::acos( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, ATan )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = xr.atan();

    auto x = tax::DA< N >::variable( 0.0 );
    tax::DA< N > y = tax::atan( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Cosh )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = xr.cosh();

    auto x = tax::DA< N >::variable( 0.0 );
    tax::DA< N > y = tax::cosh( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Sinh )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = xr.sinh();

    auto x = tax::DA< N >::variable( 0.0 );
    tax::DA< N > y = tax::sinh( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Tanh )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = xr.tanh();

    auto x = tax::DA< N >::variable( 0.0 );
    tax::DA< N > y = tax::tanh( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Exp )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = xr.exp();

    auto x = tax::DA< N >::variable( 0.0 );
    tax::DA< N > y = tax::exp( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Log )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = ( 1 + xr ).log();

    auto x = tax::DA< N >::variable( 1.0 );
    tax::DA< N > y = tax::log( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Log10 )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = ( 1 + xr ).log10();

    auto x = tax::DA< N >::variable( 1.0 );
    tax::DA< N > y = tax::log10( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Sqrt )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = ( 2 + xr ).sqrt();

    auto x = tax::DA< N >::variable( 2.0 );
    tax::DA< N > y = tax::sqrt( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Erf )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = ( 2 + xr ).erf();

    auto x = tax::DA< N >::variable( 2.0 );
    tax::DA< N > y = tax::erf( x );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, IPow )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = ( 2 + xr ).pow( 5 );

    auto x = tax::DA< N >::variable( 2.0 );
    tax::DA< N > y = tax::pow( x, 5 );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Pow )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = ( 2 + xr ).pow( 0.5 );

    auto x = tax::DA< N >::variable( 2.0 );
    tax::DA< N > y = tax::pow( x, 0.5 );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}
