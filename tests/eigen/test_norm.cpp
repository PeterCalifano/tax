#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include "../testUtils.hpp"

// ---------------------------------------------------------------------------
// tax::norm<P> / tax::norm<P,Q> over a vector of Taylor expansions.
// ---------------------------------------------------------------------------

namespace
{

template < typename E >
void expectCoeffsNear( const E& a, const E& b, double tol )
{
    for ( std::size_t k = 0; k < E::nCoefficients; ++k )
        EXPECT_NEAR( a[k], b[k], tol * ( 1.0 + std::abs( a[k] ) ) ) << "k = " << k;
}

}  // namespace

TEST( Norm, EuclideanValueAndGradient )
{
    using E = tax::TE< 6, 3 >;
    typename E::Input p{ 3.0, 4.0, 12.0 };  // |p| = 13
    auto x = E::variable< 0 >( p );
    auto y = E::variable< 1 >( p );
    auto z = E::variable< 2 >( p );
    std::array< E, 3 > v{ x, y, z };

    auto r = tax::norm< 2 >( v );
    EXPECT_NEAR( r.value(), 13.0, 1e-12 );

    // grad ||v|| = v / ||v||  at the expansion point.
    auto g = r.gradient();
    EXPECT_NEAR( g( 0 ), 3.0 / 13.0, 1e-12 );
    EXPECT_NEAR( g( 1 ), 4.0 / 13.0, 1e-12 );
    EXPECT_NEAR( g( 2 ), 12.0 / 13.0, 1e-12 );
}

TEST( Norm, DefaultOrderIsEuclidean )
{
    using E = tax::TE< 4, 2 >;
    typename E::Input p{ 3.0, 4.0 };
    std::array< E, 2 > v{ E::variable< 0 >( p ), E::variable< 1 >( p ) };
    expectCoeffsNear( tax::norm( v ), tax::norm< 2 >( v ), 0.0 );  // P defaults to 2
}

TEST( Norm, PowerBindsToHalfPow )
{
    using E = tax::TE< 7, 3 >;
    typename E::Input p{ 1.0, -2.0, 0.5 };
    auto x = E::variable< 0 >( p );
    auto y = E::variable< 1 >( p );
    auto z = E::variable< 2 >( p );
    std::array< E, 3 > v{ x, y, z };
    auto s = tax::square( x ) + tax::square( y ) + tax::square( z );

    // ||v||^2 == sum of squares (no root).
    expectCoeffsNear( tax::norm< 2, 2 >( v ), s, 1e-12 );
    // 1/||v||^3 == invSqrtPow<3>(sum of squares) — the gravity kernel, bit-identical.
    auto grav = tax::norm< 2, -3 >( v );
    auto ref = tax::invSqrtPow< 3 >( s );
    for ( std::size_t k = 0; k < E::nCoefficients; ++k ) EXPECT_EQ( grav[k], ref[k] );
    // 1/||v|| == invSqrtPow<1>.
    expectCoeffsNear( tax::norm< 2, -1 >( v ), tax::reciprocal( tax::norm< 2 >( v ) ), 1e-11 );
}

TEST( Norm, PowerEqualsPowOfNorm )
{
    using E = tax::TE< 6, 3 >;
    typename E::Input p{ 0.7, 1.3, 0.9 };
    auto x = E::variable< 0 >( p );
    auto y = E::variable< 1 >( p );
    auto z = E::variable< 2 >( p );
    std::array< E, 3 > v{ x, y, z };

    // norm<P,Q> == pow(norm<P>, Q) for several (P, Q).
    expectCoeffsNear( tax::norm< 2, 3 >( v ), tax::pow( tax::norm< 2 >( v ), 3 ), 1e-11 );
    expectCoeffsNear( tax::norm< 2, 5 >( v ), tax::pow( tax::norm< 2 >( v ), 5 ), 1e-11 );
    expectCoeffsNear( tax::norm< 4, 2 >( v ), tax::pow( tax::norm< 4 >( v ), 2 ), 1e-10 );
    expectCoeffsNear( tax::norm< 3, 3 >( v ), tax::pow( tax::norm< 3 >( v ), 3 ), 1e-10 );
}

TEST( Norm, HigherOrderValue )
{
    using E = tax::TE< 3, 3 >;
    typename E::Input p{ 3.0, 4.0, 12.0 };
    std::array< E, 3 > v{ E::variable< 0 >( p ), E::variable< 1 >( p ), E::variable< 2 >( p ) };

    // 4-norm value == (3^4 + 4^4 + 12^4)^{1/4}.
    const double want4 =
        std::pow( std::pow( 3.0, 4 ) + std::pow( 4.0, 4 ) + std::pow( 12.0, 4 ), 0.25 );
    EXPECT_NEAR( tax::norm< 4 >( v ).value(), want4, 1e-10 );
}

TEST( Norm, EigenVectorInput )
{
    Eigen::Vector3d x0{ 3.0, 4.0, 12.0 };
    auto v = tax::la::variables< tax::TE< 5, 3 > >( x0 );  // Eigen column vector of TE
    auto r = tax::norm< 2 >( v );
    EXPECT_NEAR( r.value(), 13.0, 1e-12 );
    // norm<P,Q> on the Eigen overload too. (Bind first: the <2,-3> comma would
    // otherwise be read as a macro-argument separator by EXPECT_NEAR.)
    const double inv3 = tax::norm< 2, -3 >( v ).value();
    EXPECT_NEAR( inv3, 1.0 / ( 13.0 * 13.0 * 13.0 ), 1e-14 );
}

TEST( Norm, NamedVector )
{
    // A named 3-D axis "q"; its coordinate variables form a vector.
    std::array< double, 3 > q0{ 3.0, 4.0, 12.0 };
    auto q = tax::variables< "q", 5 >( q0 );  // std::array<NE, 3>
    auto r = tax::norm< 2 >( q );
    EXPECT_NEAR( r.value(), 13.0, 1e-12 );
    // Result keeps the named-axis type.
    static_assert( std::is_same_v< decltype( r ), std::decay_t< decltype( q[0] ) > > );
}
