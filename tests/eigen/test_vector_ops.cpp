#include <gtest/gtest.h>

#include <cmath>

#include "../testUtils.hpp"

// tax::la vector algebra: dot, cross, angle, unitvec, unitcross, projvec,
// projplane — over Eigen vectors of Taylor expansions.

namespace
{

using E = tax::TE< 5, 3 >;

Eigen::Matrix< E, 3, 1 > vec3( double a, double b, double c )
{
    typename E::Input p{ a, b, c };
    Eigen::Matrix< E, 3, 1 > v;
    v( 0 ) = E::variable< 0 >( p );
    v( 1 ) = E::variable< 1 >( p );
    v( 2 ) = E::variable< 2 >( p );
    return v;
}

double norm3( const Eigen::Matrix< E, 3, 1 >& v )
{
    return std::sqrt( v( 0 ).value() * v( 0 ).value() + v( 1 ).value() * v( 1 ).value() +
                      v( 2 ).value() * v( 2 ).value() );
}

}  // namespace

TEST( VectorOps, DotValueAndGradient )
{
    // Independent vectors: a uses variables 0..2, b uses variables 3..5, so
    // d(a·b)/d(a) = b and d(a·b)/d(b) = a.
    using E6 = tax::TE< 3, 6 >;
    typename E6::Input p{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Eigen::Matrix< E6, 3, 1 > a, b;
    a( 0 ) = E6::variable< 0 >( p );
    a( 1 ) = E6::variable< 1 >( p );
    a( 2 ) = E6::variable< 2 >( p );
    b( 0 ) = E6::variable< 3 >( p );
    b( 1 ) = E6::variable< 4 >( p );
    b( 2 ) = E6::variable< 5 >( p );

    auto d = tax::dot( a, b );
    EXPECT_NEAR( d.value(), 1.0 * 4 + 2 * 5 + 3 * 6, 1e-12 );
    auto g = d.gradient();              // [∂/∂a, ∂/∂b] = [b, a]
    EXPECT_NEAR( g( 0 ), 4.0, 1e-12 );  // ∂/∂a_0 = b_0
    EXPECT_NEAR( g( 1 ), 5.0, 1e-12 );
    EXPECT_NEAR( g( 2 ), 6.0, 1e-12 );
    EXPECT_NEAR( g( 3 ), 1.0, 1e-12 );  // ∂/∂b_0 = a_0
    EXPECT_NEAR( g( 5 ), 3.0, 1e-12 );
}

TEST( VectorOps, DotMatrixVectorConstantMap )
{
    auto a = vec3( 1.0, 2.0, 3.0 );
    Eigen::Matrix3d M;
    M << 2, 0, 0, 0, 3, 0, 0, 0, 1;  // constant real linear map
    auto Ma = tax::dot( M, a );
    EXPECT_NEAR( Ma( 0 ).value(), 2.0, 1e-12 );
    EXPECT_NEAR( Ma( 1 ).value(), 6.0, 1e-12 );
    EXPECT_NEAR( Ma( 2 ).value(), 3.0, 1e-12 );
    // Row 0 of the Jacobian of M·a w.r.t. a is M's row 0.
    EXPECT_NEAR( ( Ma( 0 ).template derivative< 1, 0, 0 >() ), 2.0, 1e-12 );

    // Same result with an expansion matrix.
    Eigen::Matrix< E, 3, 3 > Me = M.cast< E >();
    auto Mea = tax::dot( Me, a );
    for ( int i = 0; i < 3; ++i ) tax::test::ExpectCoeffsNear( Ma( i ), Mea( i ), 1e-12 );
}

// dot(A, x) with a genuinely variable-valued TE matrix: both A and x carry
// Taylor dependence, so the product is a full Cauchy product per entry.
TEST( VectorOps, DotMatrixVectorExpansionMatrix )
{
    using E6 = tax::TE< 3, 6 >;  // A entries in vars 0..3, x in vars 4..5
    typename E6::Input p{ 2.0, 3.0, 4.0, 5.0, 10.0, 20.0 };
    Eigen::Matrix< E6, 2, 2 > A;
    A( 0, 0 ) = E6::variable< 0 >( p );
    A( 0, 1 ) = E6::variable< 1 >( p );
    A( 1, 0 ) = E6::variable< 2 >( p );
    A( 1, 1 ) = E6::variable< 3 >( p );
    Eigen::Matrix< E6, 2, 1 > x;
    x( 0 ) = E6::variable< 4 >( p );
    x( 1 ) = E6::variable< 5 >( p );

    auto y = tax::dot( A, x );  // y_i = Σ_j A_ij x_j
    EXPECT_NEAR( y( 0 ).value(), 2 * 10 + 3 * 20, 1e-12 );
    EXPECT_NEAR( y( 1 ).value(), 4 * 10 + 5 * 20, 1e-12 );
    // ∂y_0/∂A_00 = x_0, ∂y_0/∂x_0 = A_00, and the bilinear cross term = 1.
    EXPECT_NEAR( ( y( 0 ).template derivative< 1, 0, 0, 0, 0, 0 >() ), 10.0, 1e-12 );
    EXPECT_NEAR( ( y( 0 ).template derivative< 0, 0, 0, 0, 1, 0 >() ), 2.0, 1e-12 );
    EXPECT_NEAR( ( y( 0 ).template derivative< 1, 0, 0, 0, 1, 0 >() ), 1.0, 1e-12 );
}

TEST( VectorOps, CrossIsAntisymmetricAndOrthogonal )
{
    auto a = vec3( 1.0, 2.0, 3.0 );
    auto b = vec3( 4.0, 5.0, 6.0 );
    auto c = tax::cross( a, b );
    EXPECT_NEAR( c( 0 ).value(), -3.0, 1e-12 );
    EXPECT_NEAR( c( 1 ).value(), 6.0, 1e-12 );
    EXPECT_NEAR( c( 2 ).value(), -3.0, 1e-12 );
    // a × b ⟂ a and ⟂ b (as full series).
    tax::test::ExpectCoeffsNear( tax::dot( c, a ), E{ 0.0 }, 1e-11 );
    tax::test::ExpectCoeffsNear( tax::dot( c, b ), E{ 0.0 }, 1e-11 );
    // a × b == −(b × a).
    auto c2 = tax::cross( b, a );
    for ( int i = 0; i < 3; ++i ) tax::test::ExpectCoeffsNear( c( i ), -c2( i ), 1e-12 );
}

TEST( VectorOps, AngleValueAndSymmetry )
{
    auto a = vec3( 1.0, 0.0, 0.0 );
    auto b = vec3( 1.0, 1.0, 0.0 );
    auto ang = tax::angle( a, b );
    EXPECT_NEAR( ang.value(), M_PI / 4.0, 1e-12 );  // 45°
    // Symmetric.
    tax::test::ExpectCoeffsNear( tax::angle( a, b ), tax::angle( b, a ), 1e-11 );
    // General triple: acos of the normalised dot.
    auto u = vec3( 1.0, 2.0, 3.0 );
    auto w = vec3( 4.0, 5.0, 6.0 );
    const double cosv = ( 4.0 + 10 + 18 ) / ( std::sqrt( 14.0 ) * std::sqrt( 77.0 ) );
    EXPECT_NEAR( tax::angle( u, w ).value(), std::acos( cosv ), 1e-12 );
}

TEST( VectorOps, UnitVecHasUnitLengthAndDirection )
{
    auto a = vec3( 3.0, 4.0, 12.0 );  // |a| = 13
    auto u = tax::unitvec( a );
    EXPECT_NEAR( norm3( u ), 1.0, 1e-12 );
    EXPECT_NEAR( u( 0 ).value(), 3.0 / 13.0, 1e-12 );
    EXPECT_NEAR( u( 2 ).value(), 12.0 / 13.0, 1e-12 );
    // |unitvec|² == 1 as a full series (constant 1, higher terms 0).
    auto n2 = tax::dot( u, u );
    EXPECT_NEAR( n2.value(), 1.0, 1e-12 );
    for ( std::size_t k = 1; k < E::nCoefficients; ++k ) EXPECT_NEAR( n2[k], 0.0, 1e-10 );
}

TEST( VectorOps, UnitCrossIsUnitAndOrthogonal )
{
    auto a = vec3( 1.0, 2.0, 3.0 );
    auto b = vec3( -2.0, 0.5, 1.0 );
    auto n = tax::unitcross( a, b );
    EXPECT_NEAR( norm3( n ), 1.0, 1e-12 );
    tax::test::ExpectCoeffsNear( tax::dot( n, a ), E{ 0.0 }, 1e-10 );
    tax::test::ExpectCoeffsNear( tax::dot( n, b ), E{ 0.0 }, 1e-10 );
}

TEST( VectorOps, ProjVecAlongDirection )
{
    auto a = vec3( 2.0, 1.0, 0.0 );
    auto d = vec3( 1.0, 0.0, 0.0 );  // x-axis
    auto p = tax::projvec( a, d );
    EXPECT_NEAR( p( 0 ).value(), 2.0, 1e-12 );  // projection onto x keeps a_x
    EXPECT_NEAR( p( 1 ).value(), 0.0, 1e-12 );
    EXPECT_NEAR( p( 2 ).value(), 0.0, 1e-12 );
    // projvec is parallel to d and (a − p) ⟂ d.
    Eigen::Matrix< E, 3, 1 > rem;
    for ( int i = 0; i < 3; ++i ) rem( i ) = a( i ) - p( i );
    tax::test::ExpectCoeffsNear( tax::dot( rem, d ), E{ 0.0 }, 1e-11 );
}

TEST( VectorOps, ProjPlaneRemovesNormalComponent )
{
    auto a = vec3( 1.0, 2.0, 3.0 );
    auto nrm = vec3( 0.0, 0.0, 2.0 );  // z-normal (unnormalised)
    auto p = tax::projplane( a, nrm );
    // In-plane component keeps x, y; drops z.
    EXPECT_NEAR( p( 0 ).value(), 1.0, 1e-12 );
    EXPECT_NEAR( p( 1 ).value(), 2.0, 1e-12 );
    EXPECT_NEAR( p( 2 ).value(), 0.0, 1e-12 );
    // Result is orthogonal to the normal (as a full series).
    tax::test::ExpectCoeffsNear( tax::dot( p, nrm ), E{ 0.0 }, 1e-11 );
    // projvec + projplane reconstruct a.
    auto pv = tax::projvec( a, nrm );
    for ( int i = 0; i < 3; ++i )
        tax::test::ExpectCoeffsNear( E{ p( i ) + pv( i ) }, a( i ), 1e-11 );
}

// Works for named expansions too (element type carries its axis).
TEST( VectorOps, NamedElements )
{
    std::array< double, 3 > r0{ 3.0, 4.0, 12.0 };
    auto q = tax::variables< "q", 4 >( r0 );  // std::array<NE, 3>
    Eigen::Matrix< std::decay_t< decltype( q[0] ) >, 3, 1 > v;
    v << q[0], q[1], q[2];
    auto len = tax::dot( v, v );
    EXPECT_NEAR( len.value(), 169.0, 1e-11 );
    auto u = tax::unitvec( v );
    EXPECT_NEAR( u( 0 ).value(), 3.0 / 13.0, 1e-12 );
    static_assert(
        std::is_same_v< std::decay_t< decltype( u( 0 ) ) >, std::decay_t< decltype( q[0] ) > > );
}
