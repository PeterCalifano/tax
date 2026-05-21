// tests/regression/testUnivariate.cpp
//
// Ported from claude/add-verner-integrators-vEgRF:tests/dace/testUnivariate.cpp
// and retargeted at tax::TE<N>. Every input variable is wrapped in
// tax_regression::prepareInput before the op under test, on both the
// DACE and tax side, so the operators are exercised against polynomials
// with non-trivial structure rather than the bare variable(x0).
//
// prepareInput returns values in [1, 1.5]; for ops whose domain excludes
// that range (asin, acos, atanh) the prep is scaled to [0, 0.4] on both
// sides identically.

#include <dace/dace.h>
#include <gtest/gtest.h>

#include <tax/tax.hpp>

#include "regressionUtils.hpp"

using tax_regression::expectCoeffsMatch;
using tax_regression::prepareInput;

namespace
{
// Scale prepareInput output from [1, 1.5] to [0, 0.4] for ops whose domain
// excludes the default prep range. Applied identically on both sides.
inline DACE::DA scaleToUnit( const DACE::DA& x )
{
    return ( prepareInput( x ) - 1.0 ) * 0.8;
}
template < int N >
inline tax::TE< N > scaleToUnit( const tax::TE< N >& x )
{
    return ( prepareInput( x ) - 1.0 ) * 0.8;
}
}  // namespace

// -----------------------------------------------------------------------
// Operators
// -----------------------------------------------------------------------

TEST( DaceUnivariate, Div )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA f_ref = prepareInput( xr );
    DACE::DA yr    = 1.0 / ( 1.0 + f_ref );

    auto         x = tax::TE< N >::variable( 0.0 );
    auto         f = prepareInput( x );
    tax::TE< N > y = tax::reciprocal( 1.0 + f );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, MulDiv )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA f_ref = prepareInput( xr );
    DACE::DA yr    = 1.0 / ( ( 1.0 + f_ref ) * ( 1.0 + f_ref ) );

    auto         x = tax::TE< N >::variable( 0.0 );
    auto         f = prepareInput( x );
    tax::TE< N > y = tax::reciprocal( ( 1.0 + f ) * ( 1.0 + f ) );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

// -----------------------------------------------------------------------
// Math: trig
// -----------------------------------------------------------------------

TEST( DaceUnivariate, Sin )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = prepareInput( xr ).sin();

    auto         x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::sin( prepareInput( x ) );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Cos )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = prepareInput( xr ).cos();

    auto         x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::cos( prepareInput( x ) );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Tan )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr( 1 );
    DACE::DA yr = prepareInput( xr ).tan();

    auto         x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::tan( prepareInput( x ) );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

// asin/acos: prep range [1, 1.5] is outside domain; scale to [0, 0.4]
TEST( DaceUnivariate, ASin )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr = DACE::DA( 1 );
    DACE::DA yr = scaleToUnit( xr ).asin();

    auto         x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::asin( scaleToUnit( x ) );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, ACos )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr = DACE::DA( 1 );
    DACE::DA yr = scaleToUnit( xr ).acos();

    auto         x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::acos( scaleToUnit( x ) );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, ATan )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr = DACE::DA( 1 );
    DACE::DA yr = prepareInput( xr ).atan();

    auto         x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::atan( prepareInput( x ) );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

// -----------------------------------------------------------------------
// Math: hyperbolic
// -----------------------------------------------------------------------

TEST( DaceUnivariate, Sinh )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr = DACE::DA( 1 );
    DACE::DA yr = prepareInput( xr ).sinh();

    auto         x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::sinh( prepareInput( x ) );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Cosh )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr = DACE::DA( 1 );
    DACE::DA yr = prepareInput( xr ).cosh();

    auto         x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::cosh( prepareInput( x ) );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Tanh )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr = DACE::DA( 1 );
    DACE::DA yr = prepareInput( xr ).tanh();

    auto         x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::tanh( prepareInput( x ) );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, ASinh )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr = DACE::DA( 1 );
    DACE::DA yr = prepareInput( xr ).asinh();

    auto         x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::asinh( prepareInput( x ) );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

// acosh: prep constant term is exactly 1 (sin(0)=0), which is the domain
// boundary where the derivative is singular. Shift by +1 to keep the
// expansion well inside the domain. Same shift on both sides.
TEST( DaceUnivariate, ACosh )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr = DACE::DA( 1 );
    DACE::DA yr = ( 1.0 + prepareInput( xr ) ).acosh();

    auto         x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::acosh( 1.0 + prepareInput( x ) );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

// atanh: prep range [1, 1.5] is outside (-1, 1); scale to [0, 0.4]
TEST( DaceUnivariate, ATanh )
{
    constexpr int N = 20;

    DACE::DA::init( N, 1 );
    DACE::DA xr = DACE::DA( 1 );
    DACE::DA yr = scaleToUnit( xr ).atanh();

    auto         x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::atanh( scaleToUnit( x ) );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

// -----------------------------------------------------------------------
// Math: exp / log / roots / power
// -----------------------------------------------------------------------

TEST( DaceUnivariate, Exp )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr = DACE::DA( 1 );
    DACE::DA yr = prepareInput( xr ).exp();

    auto         x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::exp( prepareInput( x ) );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Log )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr = DACE::DA( 1 );
    DACE::DA yr = prepareInput( xr ).log();

    auto         x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::log( prepareInput( x ) );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Sqrt )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr = DACE::DA( 1 );
    DACE::DA yr = prepareInput( xr ).sqrt();

    auto         x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::sqrt( prepareInput( x ) );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Cbrt )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr = DACE::DA( 1 );
    DACE::DA yr = prepareInput( xr ).cbrt();

    auto         x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::cbrt( prepareInput( x ) );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Erf )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr = DACE::DA( 1 );
    DACE::DA yr = prepareInput( xr ).erf();

    auto         x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::erf( prepareInput( x ) );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, IPow )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr = DACE::DA( 1 );
    DACE::DA yr = prepareInput( xr ).pow( 5 );

    auto         x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::pow( prepareInput( x ), 5 );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}

TEST( DaceUnivariate, Pow )
{
    constexpr int N = 40;

    DACE::DA::init( N, 1 );
    DACE::DA xr = DACE::DA( 1 );
    DACE::DA yr = prepareInput( xr ).pow( 0.5 );

    auto         x = tax::TE< N >::variable( 0.0 );
    tax::TE< N > y = tax::pow( prepareInput( x ), 0.5 );

    EXPECT_TRUE( expectCoeffsMatch( y, yr ) );
}
