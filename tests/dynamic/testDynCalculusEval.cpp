// Verification of polynomial evaluation, symbolic differentiation and
// integration, and coefficient-vector norms on the dynamic-shape Taylor
// expansions. Static-shape results are used as the reference.

#include "../testUtils.hpp"

#include <array>
#include <vector>

#include <tax/tax.hpp>

using tax::DynOrderTE;
using tax::DynTE;
using tax::TE;
using tax::TEn;

// =============================================================================
// eval(dx)
// =============================================================================

TEST( DynCalculus, EvalUnivariate )
{
    constexpr int N = 6;
    auto sx = TE< N >::variable( 1.5 );
    auto sf = TE< N >{ tax::sin( sx ) * tax::exp( sx ) };

    auto dx = DynTE<>::variable( 1.5, 0, N, 1 );
    auto df = tax::sin( dx ) * tax::exp( dx );

    const double displacement = 0.1;
    EXPECT_NEAR( df.eval( displacement ), sf.eval( displacement ), 1e-12 );
}

TEST( DynCalculus, EvalMultivariate )
{
    constexpr int N = 4, M = 2;
    auto svars = TEn< N, M >::variables( { 0.5, 0.3 } );
    auto sf = TEn< N, M >{ tax::sin( std::get< 0 >( svars ) * std::get< 1 >( svars ) ) };

    std::array< double, M > x0{ 0.5, 0.3 };
    auto dvars = DynTE<>::variables( std::span< const double >( x0 ), N );
    auto df = tax::sin( dvars[0] * dvars[1] );

    std::array< double, M > dx{ 0.05, -0.02 };
    EXPECT_NEAR( df.eval( std::span< const double >( dx ) ),
                 sf.eval( typename TEn< N, M >::Input{ 0.05, -0.02 } ), 1e-12 );
}

TEST( DynCalculus, DynOrderTteEvalUnivariate )
{
    constexpr int N = 6;
    auto x = DynOrderTE< 1 >::variable( 1.5, N );
    auto f = tax::sin( x ) * tax::exp( x );
    auto sx = TE< N >::variable( 1.5 );
    auto sf = TE< N >{ tax::sin( sx ) * tax::exp( sx ) };
    EXPECT_NEAR( f.eval( 0.1 ), sf.eval( 0.1 ), 1e-12 );
}

TEST( DynCalculus, DynOrderTteEvalMultivariate )
{
    constexpr int N = 4, M = 3;
    std::array< double, M > x0{ 0.1, 0.2, 0.3 };
    auto vars = DynOrderTE< M >::variables( x0, N );
    auto f = tax::exp( vars[0] + vars[1] + vars[2] );

    auto svars = TEn< N, M >::variables( { 0.1, 0.2, 0.3 } );
    auto sf = TEn< N, M >{ tax::exp( std::get< 0 >( svars ) + std::get< 1 >( svars )
                                     + std::get< 2 >( svars ) ) };

    std::array< double, M > dx{ 0.01, -0.02, 0.03 };
    EXPECT_NEAR( f.eval( dx ), sf.eval( typename TEn< N, M >::Input{ 0.01, -0.02, 0.03 } ),
                 1e-12 );
}

// =============================================================================
// deriv / integ (symbolic)
// =============================================================================

TEST( DynCalculus, DerivUnivariate )
{
    constexpr int N = 5;
    auto sx = TE< N >::variable( 0.3 );
    auto sf = TE< N >{ tax::sin( sx ) };
    auto sdf = sf.deriv( 0 );

    auto dx = DynTE<>::variable( 0.3, 0, N, 1 );
    auto df = tax::sin( dx );
    auto ddf = df.deriv( 0 );

    ASSERT_EQ( sdf.nCoefficients, ddf.coeffs().size() );
    for ( std::size_t i = 0; i < ddf.coeffs().size(); ++i )
        EXPECT_NEAR( ddf.coeffs()[i], sdf[i], 1e-12 );
}

TEST( DynCalculus, DerivMultivariate )
{
    constexpr int N = 4, M = 2;
    auto svars = TEn< N, M >::variables( { 0.5, 0.3 } );
    auto sf = TEn< N, M >{ tax::sin( std::get< 0 >( svars ) * std::get< 1 >( svars ) ) };
    auto sdfx = sf.deriv( 0 );
    auto sdfy = sf.deriv( 1 );

    std::array< double, M > x0{ 0.5, 0.3 };
    auto dvars = DynTE<>::variables( std::span< const double >( x0 ), N );
    auto df = tax::sin( dvars[0] * dvars[1] );
    auto ddfx = df.deriv( 0 );
    auto ddfy = df.deriv( 1 );

    for ( std::size_t i = 0; i < ddfx.coeffs().size(); ++i )
    {
        EXPECT_NEAR( ddfx.coeffs()[i], sdfx[i], 1e-12 );
        EXPECT_NEAR( ddfy.coeffs()[i], sdfy[i], 1e-12 );
    }
}

TEST( DynCalculus, IntegMultivariate )
{
    constexpr int N = 4, M = 2;
    auto svars = TEn< N, M >::variables( { 0.5, 0.3 } );
    auto sf = TEn< N, M >{ tax::exp( std::get< 0 >( svars ) ) };
    auto sigfx = sf.integ( 0 );

    std::array< double, M > x0{ 0.5, 0.3 };
    auto dvars = DynTE<>::variables( std::span< const double >( x0 ), N );
    auto df = tax::exp( dvars[0] );
    auto digfx = df.integ( 0 );

    for ( std::size_t i = 0; i < digfx.coeffs().size(); ++i )
        EXPECT_NEAR( digfx.coeffs()[i], sigfx[i], 1e-12 );
}

TEST( DynCalculus, DerivOutOfRangeThrows )
{
    auto x = DynTE<>::variable( 1.0, 0, 3, 2 );
    EXPECT_THROW( x.deriv( 5 ), std::out_of_range );
    EXPECT_THROW( x.integ( 5 ), std::out_of_range );
}

TEST( DynCalculus, DynOrderTteDerivCompileTimeAndRuntime )
{
    constexpr int N = 4, M = 2;
    auto vars = DynOrderTE< M >::variables( { 0.5, 0.3 }, N );
    auto f = tax::sin( vars[0] * vars[1] );

    auto df_ct = f.deriv< 0 >();
    auto df_rt = f.deriv( 0 );
    ASSERT_EQ( df_ct.coeffs().size(), df_rt.coeffs().size() );
    for ( std::size_t i = 0; i < df_ct.coeffs().size(); ++i )
        EXPECT_EQ( df_ct.coeffs()[i], df_rt.coeffs()[i] );
}

TEST( DynCalculus, DerivThenInteg_RoundTripsUpToDroppedTopDegree )
{
    constexpr int N = 5;
    // f(x) = exp(x) has all non-trivial coefficients. d/dx then integ should
    // return the original minus the constant of integration (we lose c[0]).
    auto x = DynTE<>::variable( 0.3, 0, N, 1 );
    auto f = tax::exp( x );
    auto g = f.deriv( 0 ).integ( 0 );

    // c[0] of g is zero (constant of integration); the rest of the leading-
    // degree-1..N-1 coefficients should match f. The top-degree N coefficient
    // is lost via integ truncation and integ then deriv re-loses one degree.
    for ( std::size_t i = 1; i < f.coeffs().size() - 1; ++i )
        EXPECT_NEAR( g.coeffs()[i], f.coeffs()[i], 1e-12 ) << "i=" << i;
}

// =============================================================================
// derivative (numerical) and derivatives()
// =============================================================================

TEST( DynCalculus, DerivativeNumerical )
{
    constexpr int N = 4, M = 2;
    auto svars = TEn< N, M >::variables( { 0.5, 0.3 } );
    auto sf = TEn< N, M >{ tax::exp( std::get< 0 >( svars ) + std::get< 1 >( svars ) ) };

    std::array< double, M > x0{ 0.5, 0.3 };
    auto dvars = DynTE<>::variables( std::span< const double >( x0 ), N );
    auto df = tax::exp( dvars[0] + dvars[1] );

    // d^2 f / (dx dy) at the expansion point
    EXPECT_NEAR( df.derivative( { 1, 1 } ), sf.derivative( tax::MultiIndex< M >{ 1, 1 } ),
                 1e-12 );
    // d^2 f / dx^2
    EXPECT_NEAR( df.derivative( { 2, 0 } ), sf.derivative( tax::MultiIndex< M >{ 2, 0 } ),
                 1e-12 );
}

TEST( DynCalculus, AllDerivatives )
{
    constexpr int N = 3, M = 2;
    auto svars = TEn< N, M >::variables( { 0.5, 0.3 } );
    auto sf = TEn< N, M >{ tax::exp( std::get< 0 >( svars ) ) };
    auto sall = sf.derivatives();

    std::array< double, M > x0{ 0.5, 0.3 };
    auto dvars = DynTE<>::variables( std::span< const double >( x0 ), N );
    auto df = tax::exp( dvars[0] );
    auto dall = df.derivatives();

    ASSERT_EQ( sall.size(), dall.size() );
    for ( std::size_t i = 0; i < dall.size(); ++i ) EXPECT_NEAR( dall[i], sall[i], 1e-12 );
}

TEST( DynCalculus, DynOrderTteDerivativeCompileTimeMultiIndex )
{
    constexpr int N = 3, M = 2;
    auto vars = DynOrderTE< M >::variables( { 0.5, 0.3 }, N );
    auto f = tax::exp( vars[0] + vars[1] );

    const double dxdy = f.derivative< 1, 1 >();
    const double dxx = f.derivative< 2, 0 >();
    // Reference values: d^2 exp(x+y)/dxdy = exp(x+y); d^2/dx^2 = exp(x+y).
    EXPECT_NEAR( dxdy, std::exp( 0.5 + 0.3 ), 1e-12 );
    EXPECT_NEAR( dxx, std::exp( 0.5 + 0.3 ), 1e-12 );
}

// =============================================================================
// Coefficient-vector norms
// =============================================================================

TEST( DynCalculus, CoeffsNormInf )
{
    constexpr int N = 5;
    auto sx = TE< N >::variable( 0.3 );
    auto sf = TE< N >{ tax::sin( sx ) };

    auto dx = DynTE<>::variable( 0.3, 0, N, 1 );
    auto df = tax::sin( dx );

    EXPECT_NEAR( df.coeffsNormInf(), sf.coeffsNormInf(), 1e-14 );
}

TEST( DynCalculus, CoeffsNorm1 )
{
    constexpr int N = 5;
    auto sx = TE< N >::variable( 0.3 );
    auto sf = TE< N >{ tax::sin( sx ) };

    auto dx = DynTE<>::variable( 0.3, 0, N, 1 );
    auto df = tax::sin( dx );

    EXPECT_NEAR( df.template coeffsNorm< 1 >(), sf.template coeffsNorm< 1 >(), 1e-14 );
    EXPECT_NEAR( df.coeffsNorm( 1 ), sf.coeffsNorm( 1 ), 1e-14 );
}

TEST( DynCalculus, CoeffsNorm2 )
{
    constexpr int N = 5;
    auto sx = TE< N >::variable( 0.3 );
    auto sf = TE< N >{ tax::exp( sx ) };

    auto dx = DynTE<>::variable( 0.3, 0, N, 1 );
    auto df = tax::exp( dx );

    EXPECT_NEAR( df.template coeffsNorm< 2 >(), sf.template coeffsNorm< 2 >(), 1e-14 );
    EXPECT_NEAR( df.coeffsNorm( 2 ), sf.coeffsNorm( 2 ), 1e-14 );
}

TEST( DynCalculus, CoeffsNormZeroThrows )
{
    auto x = DynTE<>::variable( 1.0, 0, 3, 1 );
    EXPECT_THROW( x.coeffsNorm( 0 ), std::invalid_argument );
}
