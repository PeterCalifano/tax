// Eigen integration tests for the dynamic-shape TaylorExpansionT
// specialisations: `DynOrderTE<M>` (compile-time `M`, runtime order) and
// `DynTE<T>` (runtime order and size).
//
// The static-shape reference path is used to verify numerical agreement.

#include <Eigen/Core>
#include <tax/tax.hpp>

#include "../testUtils.hpp"

using tax::DynOrderTE;
using tax::DynTE;
using tax::TE;
using tax::TEn;

// =============================================================================
// value(matrix) on dynamic TTE elements
// =============================================================================

TEST( EigenDynamic, ValueOnDynOrderTeMatrix )
{
    using TE3 = DynOrderTE< 3 >;
    std::array< double, 3 > x0{ 1.0, 2.0, 3.0 };
    auto vars = TE3::variables( x0, /*order=*/4 );

    Eigen::Matrix< TE3, 3, 1 > m;
    m( 0 ) = vars[0];
    m( 1 ) = vars[1];
    m( 2 ) = vars[2];

    auto v = tax::value( m );
    EXPECT_EQ( v( 0 ), 1.0 );
    EXPECT_EQ( v( 1 ), 2.0 );
    EXPECT_EQ( v( 2 ), 3.0 );
}

TEST( EigenDynamic, ValueOnDynTeMatrix )
{
    using DT = DynTE<>;
    std::array< double, 2 > x0{ 1.5, 2.5 };
    auto vars = DT::variables( std::span< const double >( x0 ), /*order=*/3 );

    Eigen::Matrix< DT, 2, 1 > m;
    m( 0 ) = vars[0];
    m( 1 ) = vars[1];

    auto v = tax::value( m );
    EXPECT_EQ( v( 0 ), 1.5 );
    EXPECT_EQ( v( 1 ), 2.5 );
}

// =============================================================================
// gradient / hessian / jacobian
// =============================================================================

TEST( EigenDynamic, GradientDynOrderTeMatchesStatic )
{
    constexpr int N = 4, M = 3;
    auto sx = TEn< N, M >::variables( { 0.5, 0.3, 0.2 } );
    auto sf = TEn< N, M >{ tax::sin( std::get< 0 >( sx ) * std::get< 1 >( sx ) )
                           + tax::exp( std::get< 2 >( sx ) ) };
    auto sg = tax::gradient( sf );

    auto dx = DynOrderTE< M >::variables( { 0.5, 0.3, 0.2 }, N );
    auto df = tax::sin( dx[0] * dx[1] ) + tax::exp( dx[2] );
    auto dg = tax::gradient( df );

    // Both should be `Matrix<double, M, 1>`.
    static_assert( decltype( dg )::RowsAtCompileTime == M );
    static_assert( decltype( dg )::ColsAtCompileTime == 1 );
    for ( int i = 0; i < M; ++i ) EXPECT_NEAR( dg( i ), sg( i ), 1e-12 );
}

TEST( EigenDynamic, GradientDynTe_ReturnsVectorX )
{
    constexpr int N = 4, M = 3;
    std::array< double, M > x0{ 0.5, 0.3, 0.2 };
    auto sx = TEn< N, M >::variables( { 0.5, 0.3, 0.2 } );
    auto sf = TEn< N, M >{ tax::sin( std::get< 0 >( sx ) * std::get< 1 >( sx ) )
                           + tax::exp( std::get< 2 >( sx ) ) };
    auto sg = tax::gradient( sf );

    auto dx = DynTE<>::variables( std::span< const double >( x0 ), N );
    auto df = tax::sin( dx[0] * dx[1] ) + tax::exp( dx[2] );
    auto dg = tax::gradient( df );

    // Fully-dynamic: returns Eigen::Matrix<double, Eigen::Dynamic, 1>.
    static_assert( decltype( dg )::RowsAtCompileTime == Eigen::Dynamic );
    ASSERT_EQ( dg.size(), M );
    for ( int i = 0; i < M; ++i ) EXPECT_NEAR( dg( i ), sg( i ), 1e-12 );
}

TEST( EigenDynamic, HessianDynOrderTeMatchesStatic )
{
    constexpr int N = 4, M = 2;
    auto sx = TEn< N, M >::variables( { 0.5, 0.3 } );
    auto sf = TEn< N, M >{ tax::sin( std::get< 0 >( sx ) * std::get< 1 >( sx ) ) };
    auto sH = tax::hessian( sf );

    auto dx = DynOrderTE< M >::variables( { 0.5, 0.3 }, N );
    auto df = tax::sin( dx[0] * dx[1] );
    auto dH = tax::hessian( df );

    static_assert( decltype( dH )::RowsAtCompileTime == M );
    static_assert( decltype( dH )::ColsAtCompileTime == M );
    for ( int i = 0; i < M; ++i )
        for ( int j = 0; j < M; ++j ) EXPECT_NEAR( dH( i, j ), sH( i, j ), 1e-12 );
}

TEST( EigenDynamic, HessianDynTe_ReturnsMatrixX )
{
    constexpr int N = 4, M = 2;
    auto sx = TEn< N, M >::variables( { 0.5, 0.3 } );
    auto sf = TEn< N, M >{ tax::sin( std::get< 0 >( sx ) * std::get< 1 >( sx ) ) };
    auto sH = tax::hessian( sf );

    std::array< double, M > x0{ 0.5, 0.3 };
    auto dx = DynTE<>::variables( std::span< const double >( x0 ), N );
    auto df = tax::sin( dx[0] * dx[1] );
    auto dH = tax::hessian( df );

    static_assert( decltype( dH )::RowsAtCompileTime == Eigen::Dynamic );
    static_assert( decltype( dH )::ColsAtCompileTime == Eigen::Dynamic );
    ASSERT_EQ( dH.rows(), M );
    ASSERT_EQ( dH.cols(), M );
    for ( int i = 0; i < M; ++i )
        for ( int j = 0; j < M; ++j ) EXPECT_NEAR( dH( i, j ), sH( i, j ), 1e-12 );
}

TEST( EigenDynamic, JacobianDynOrderTeMatchesStatic )
{
    constexpr int N = 4, M = 2;
    auto svars = TEn< N, M >::variables( { 1.0, 2.0 } );
    Eigen::Matrix< TEn< N, M >, 3, 1 > sf;
    sf( 0 ) = std::get< 0 >( svars );
    sf( 1 ) = std::get< 1 >( svars );
    sf( 2 ) = std::get< 0 >( svars ) * std::get< 1 >( svars );
    auto sJ = tax::jacobian( sf );

    auto dvars = DynOrderTE< M >::variables( { 1.0, 2.0 }, N );
    Eigen::Matrix< DynOrderTE< M >, 3, 1 > df;
    df( 0 ) = dvars[0];
    df( 1 ) = dvars[1];
    df( 2 ) = dvars[0] * dvars[1];
    auto dJ = tax::jacobian( df );

    static_assert( decltype( dJ )::RowsAtCompileTime == 3 );
    static_assert( decltype( dJ )::ColsAtCompileTime == M );
    for ( int r = 0; r < 3; ++r )
        for ( int c = 0; c < M; ++c ) EXPECT_NEAR( dJ( r, c ), sJ( r, c ), 1e-12 );
}

TEST( EigenDynamic, JacobianDynTe_ReturnsMatrixX )
{
    constexpr int N = 4, M = 2;
    auto svars = TEn< N, M >::variables( { 1.0, 2.0 } );
    Eigen::Matrix< TEn< N, M >, 3, 1 > sf;
    sf( 0 ) = std::get< 0 >( svars );
    sf( 1 ) = std::get< 1 >( svars );
    sf( 2 ) = std::get< 0 >( svars ) * std::get< 1 >( svars );
    auto sJ = tax::jacobian( sf );

    std::array< double, M > x0{ 1.0, 2.0 };
    auto dvars = DynTE<>::variables( std::span< const double >( x0 ), N );
    Eigen::Matrix< DynTE<>, 3, 1 > df;
    df( 0 ) = dvars[0];
    df( 1 ) = dvars[1];
    df( 2 ) = dvars[0] * dvars[1];
    auto dJ = tax::jacobian( df );

    static_assert( decltype( dJ )::RowsAtCompileTime == Eigen::Dynamic );
    static_assert( decltype( dJ )::ColsAtCompileTime == Eigen::Dynamic );
    ASSERT_EQ( dJ.rows(), 3 );
    ASSERT_EQ( dJ.cols(), M );
    for ( int r = 0; r < 3; ++r )
        for ( int c = 0; c < M; ++c ) EXPECT_NEAR( dJ( r, c ), sJ( r, c ), 1e-12 );
}

// =============================================================================
// eval(matrix, dx) on dynamic TTE
// =============================================================================

TEST( EigenDynamic, EvalDynOrderTeMatrix )
{
    constexpr int N = 4, M = 2;
    auto dvars = DynOrderTE< M >::variables( { 0.5, 0.3 }, N );
    Eigen::Matrix< DynOrderTE< M >, 2, 1 > df;
    df( 0 ) = tax::sin( dvars[0] * dvars[1] );
    df( 1 ) = tax::exp( dvars[0] );

    Eigen::Vector2d dx( 0.01, -0.02 );
    auto v = tax::eval( df, dx );

    // Static reference.
    auto svars = TEn< N, M >::variables( { 0.5, 0.3 } );
    Eigen::Matrix< TEn< N, M >, 2, 1 > sf;
    sf( 0 ) = tax::sin( std::get< 0 >( svars ) * std::get< 1 >( svars ) );
    sf( 1 ) = tax::exp( std::get< 0 >( svars ) );
    auto vs = tax::eval( sf, dx );

    EXPECT_NEAR( v( 0 ), vs( 0 ), 1e-12 );
    EXPECT_NEAR( v( 1 ), vs( 1 ), 1e-12 );
}

TEST( EigenDynamic, EvalDynTeMatrix_RuntimeSizedDisplacement )
{
    constexpr int N = 4, M = 3;
    std::array< double, M > x0{ 0.5, 0.3, 0.2 };
    auto dvars = DynTE<>::variables( std::span< const double >( x0 ), N );
    Eigen::Matrix< DynTE<>, 2, 1 > df;
    df( 0 ) = dvars[0] + dvars[1];
    df( 1 ) = tax::exp( dvars[2] );

    Eigen::VectorXd dx( 3 );
    dx << 0.01, -0.02, 0.03;
    auto v = tax::eval( df, dx );

    auto svars = TEn< N, M >::variables( { 0.5, 0.3, 0.2 } );
    Eigen::Matrix< TEn< N, M >, 2, 1 > sf;
    sf( 0 ) = std::get< 0 >( svars ) + std::get< 1 >( svars );
    sf( 1 ) = tax::exp( std::get< 2 >( svars ) );
    Eigen::Vector3d dxs( 0.01, -0.02, 0.03 );
    auto vs = tax::eval( sf, dxs );

    EXPECT_NEAR( v( 0 ), vs( 0 ), 1e-12 );
    EXPECT_NEAR( v( 1 ), vs( 1 ), 1e-12 );
}

// =============================================================================
// hessian() on static TTE matches derivative<2>(f)
// =============================================================================

TEST( EigenDynamic, HessianStaticMatchesDerivative2 )
{
    constexpr int N = 4, M = 2;
    auto vars = TEn< N, M >::variables( { 0.5, 0.3 } );
    auto f = TEn< N, M >{ tax::sin( std::get< 0 >( vars ) * std::get< 1 >( vars ) ) };

    auto H1 = tax::hessian( f );
    auto H2 = tax::derivative< 2 >( f );

    for ( int i = 0; i < M; ++i )
        for ( int j = 0; j < M; ++j ) EXPECT_NEAR( H1( i, j ), H2( i, j ), 1e-14 );
}

// =============================================================================
// derivative(matrix, alpha) span overload
// =============================================================================

TEST( EigenDynamic, DerivativeMatrixSpanAlpha )
{
    constexpr int N = 4, M = 3;
    std::array< double, M > x0{ 0.5, 0.3, 0.2 };
    auto dvars = DynTE<>::variables( std::span< const double >( x0 ), N );
    Eigen::Matrix< DynTE<>, 2, 1 > df;
    df( 0 ) = tax::sin( dvars[0] * dvars[1] );
    df( 1 ) = tax::exp( dvars[2] );

    std::array< int, M > a{ 1, 0, 0 };  // d/dx_0
    auto d = tax::derivative( df, std::span< const int >( a ) );

    auto svars = TEn< N, M >::variables( { 0.5, 0.3, 0.2 } );
    Eigen::Matrix< TEn< N, M >, 2, 1 > sf;
    sf( 0 ) = tax::sin( std::get< 0 >( svars ) * std::get< 1 >( svars ) );
    sf( 1 ) = tax::exp( std::get< 2 >( svars ) );
    auto ds = tax::derivative( sf, a );

    EXPECT_NEAR( d( 0 ), ds( 0 ), 1e-12 );
    EXPECT_NEAR( d( 1 ), ds( 1 ), 1e-12 );
}
