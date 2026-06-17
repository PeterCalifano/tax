// tests/eigen/test_named_la.cpp
//
// Eigen integration for tax::named: NumTraits (NamedTaylorExpansion usable as an Eigen
// scalar) and the per-axis gradient/hessian/jacobian helpers.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <array>
#include <tax/tax.hpp>

// Exercise the public API through the tax:: re-exports (no tax::named here).
using namespace tax;

namespace
{
constexpr int O = 3;
using XAxis = Axis< "x", 2 >;
using PAxis = Axis< "p", 1 >;
using NEpx = NamedTaylorExpansion< double, O, PAxis, XAxis >;
}  // namespace

TEST( NamedLa, ExpansionIsUsableAsEigenScalar )
{
    auto x = variables< "x", O >( std::array< double, 2 >{ 0.0, 0.0 } );
    auto p = variables< "p", O >( std::array< double, 1 >{ 0.0 } );

    // Build an Eigen vector whose scalar is a named expansion and exercise
    // element-wise Eigen arithmetic (relies on Eigen::NumTraits<NamedTaylorExpansion>).
    Eigen::Matrix< NEpx, 2, 1 > v;
    v( 0 ) = x[0] * p[0];
    v( 1 ) = x[1] + 1.0;  // {x}-only value promotes implicitly into the {p, x} slot

    Eigen::Matrix< NEpx, 2, 1 > w = v + v;  // element-wise scalar operator+
    EXPECT_DOUBLE_EQ( w( 1 ).value(), 2.0 );
    EXPECT_DOUBLE_EQ( ( w( 0 ).inner().coeff< 1, 1, 0 >() ), 2.0 );  // 2 * (p0*x0)
}

TEST( NamedLa, VariablesFromEigenVector )
{
    Eigen::Vector2d x0{ 1.0, 2.0 };
    auto x = variables< "x", O >( x0 );  // Eigen vector of named expansions
    static_assert(
        std::is_same_v< decltype( x ),
                        Eigen::Matrix< NamedTaylorExpansion< double, O, XAxis >, 2, 1 > > );

    EXPECT_DOUBLE_EQ( x( 0 ).value(), 1.0 );
    EXPECT_DOUBLE_EQ( x( 1 ).value(), 2.0 );
    EXPECT_DOUBLE_EQ( ( x( 0 ).inner().coeff< 1, 0 >() ), 1.0 );
    EXPECT_DOUBLE_EQ( ( x( 1 ).inner().coeff< 0, 1 >() ), 1.0 );

    // The Eigen overload feeds straight into named composition.
    auto p = variables< "p", O >( std::array< double, 1 >{ 0.0 } );
    auto f = x( 0 ) * p[0];
    static_assert( std::is_same_v< decltype( f ), NEpx > );
}

TEST( NamedLa, OneVariableExpansionOfVectorAndMatrixQuantity )
{
    // Expansion of a vector/matrix-valued quantity in a *single* variable
    // (here "t" = time): the quantity is an Eigen container whose scalar is a
    // 1-axis named expansion, built from the single time variable.
    using NEt = NamedTaylorExpansion< double, O, Axis< "t", 1 > >;
    const double t0 = 0.3;
    auto t = variable< "t", O >( t0 );  // single scalar time variable

    // Vector quantity r(t) = [cos t, sin t]^T expanded about t0.
    Eigen::Matrix< NEt, 2, 1 > r;
    r( 0 ) = cos( t );
    r( 1 ) = sin( t );
    static_assert( std::is_same_v< decltype( r )::Scalar, NEt > );

    // Each component matches the anonymous univariate Taylor series of the
    // same function coefficient-for-coefficient.
    using TE = tax::TE< O, 1 >;
    auto at = TE::variable( t0 );
    auto rc = tax::cos( at );
    auto rs = tax::sin( at );
    for ( std::size_t k = 0; k < TE::nCoefficients; ++k )
    {
        EXPECT_DOUBLE_EQ( r( 0 ).inner()[k], rc[k] ) << "cos coeff " << k;
        EXPECT_DOUBLE_EQ( r( 1 ).inner()[k], rs[k] ) << "sin coeff " << k;
    }

    // A constant vector/matrix lifts into the named scalar via Eigen's cast
    // (uses the implicit double -> NamedTaylorExpansion conversion).
    Eigen::Vector2d c0{ 2.0, -1.0 };
    Eigen::Matrix< NEt, 2, 1 > cN = c0.cast< NEt >();
    EXPECT_DOUBLE_EQ( cN( 0 ).value(), 2.0 );
    EXPECT_DOUBLE_EQ( cN( 1 ).value(), -1.0 );

    // Matrix-valued quantity over the single time axis (a rotation matrix).
    Eigen::Matrix< NEt, 2, 2 > M;
    M( 0, 0 ) = cos( t );
    M( 0, 1 ) = -sin( t );
    M( 1, 0 ) = sin( t );
    M( 1, 1 ) = cos( t );
    EXPECT_DOUBLE_EQ( M( 0, 0 ).value(), std::cos( t0 ) );
    EXPECT_DOUBLE_EQ( M( 1, 0 ).value(), std::sin( t0 ) );
    // det(M) = cos^2 + sin^2 = 1 identically: constant term 1, all higher 0.
    auto det = M( 0, 0 ) * M( 1, 1 ) - M( 0, 1 ) * M( 1, 0 );
    EXPECT_NEAR( det.value(), 1.0, 1e-14 );
    for ( std::size_t k = 1; k < NEt::Inner::nCoefficients; ++k )
        EXPECT_NEAR( det.inner()[k], 0.0, 1e-14 ) << "det coeff " << k;
}

TEST( NamedLa, GradientByAxis )
{
    // f = x0^2 * p0 + 2*x1 at x=(3,5), p=(2).
    auto x = variables< "x", O >( std::array< double, 2 >{ 3.0, 5.0 } );
    auto p = variables< "p", O >( std::array< double, 1 >{ 2.0 } );
    auto f = x[0] * x[0] * p[0] + 2.0 * x[1];

    auto gx = gradient< "x" >( f );  // [df/dx0, df/dx1] = [2*x0*p0, 2] = [12, 2]
    ASSERT_EQ( gx.rows(), 2 );
    EXPECT_DOUBLE_EQ( gx( 0 ), 12.0 );
    EXPECT_DOUBLE_EQ( gx( 1 ), 2.0 );

    auto gp = gradient< "p" >( f );  // df/dp0 = x0^2 = 9
    ASSERT_EQ( gp.rows(), 1 );
    EXPECT_DOUBLE_EQ( gp( 0 ), 9.0 );
}

TEST( NamedLa, HessianByAxis )
{
    // f = x0^2 * p0 ; Hessian over x = [[2*p0, 0], [0, 0]] = [[4,0],[0,0]].
    auto x = variables< "x", O >( std::array< double, 2 >{ 3.0, 5.0 } );
    auto p = variables< "p", O >( std::array< double, 1 >{ 2.0 } );
    auto f = x[0] * x[0] * p[0];

    auto H = hessian< "x" >( f );
    ASSERT_EQ( H.rows(), 2 );
    ASSERT_EQ( H.cols(), 2 );
    EXPECT_DOUBLE_EQ( H( 0, 0 ), 4.0 );
    EXPECT_DOUBLE_EQ( H( 0, 1 ), 0.0 );
    EXPECT_DOUBLE_EQ( H( 1, 0 ), 0.0 );
    EXPECT_DOUBLE_EQ( H( 1, 1 ), 0.0 );
}

TEST( NamedLa, JacobianByAxis )
{
    // F = [ x0 * p0 , x1^2 ] at x=(0,5), p=(2); rows over the "x" axis.
    auto x = variables< "x", O >( std::array< double, 2 >{ 0.0, 5.0 } );
    auto p = variables< "p", O >( std::array< double, 1 >{ 2.0 } );

    Eigen::Matrix< NEpx, 2, 1 > F;
    F( 0 ) = x[0] * p[0];
    F( 1 ) = x[1] * x[1];  // {x}-only value promotes implicitly into the {p, x} slot

    auto J = jacobian< "x" >( F );  // [[dF0/dx0, dF0/dx1],[dF1/dx0, dF1/dx1]]
    ASSERT_EQ( J.rows(), 2 );
    ASSERT_EQ( J.cols(), 2 );
    EXPECT_DOUBLE_EQ( J( 0, 0 ), 2.0 );  // p0
    EXPECT_DOUBLE_EQ( J( 0, 1 ), 0.0 );
    EXPECT_DOUBLE_EQ( J( 1, 0 ), 0.0 );
    EXPECT_DOUBLE_EQ( J( 1, 1 ), 10.0 );  // 2*x1
}

TEST( NamedLa, ValueAndEval )
{
    // F = [ x0 + p0 , x1 ] centered at x=(1,4), p=(2).
    auto x = variables< "x", O >( std::array< double, 2 >{ 1.0, 4.0 } );
    auto p = variables< "p", O >( std::array< double, 1 >{ 2.0 } );

    Eigen::Matrix< NEpx, 2, 1 > F;
    F( 0 ) = x[0] + p[0];
    F( 1 ) = x[1] + 0.0 * p[0];  // depends on the joint (p, x) space

    // value() returns the constant terms.
    auto v0 = value( F );
    EXPECT_DOUBLE_EQ( v0( 0 ), 3.0 );  // 1 + 2
    EXPECT_DOUBLE_EQ( v0( 1 ), 4.0 );

    // eval() at a joint displacement; joint var order is (p, x) for NEpx.
    Eigen::Vector3d dx;  // (dp0, dx0, dx1)
    dx << 0.5, 0.1, -0.2;
    auto fv = eval( F, dx );
    EXPECT_NEAR( fv( 0 ), 3.0 + 0.1 + 0.5, 1e-12 );  // x0 + p0 shift
    EXPECT_NEAR( fv( 1 ), 4.0 - 0.2, 1e-12 );

    // Scalar-overload value()/eval() on a single named expansion.
    EXPECT_DOUBLE_EQ( value( F( 0 ) ), 3.0 );
    EXPECT_NEAR( eval( F( 0 ), dx ), 3.0 + 0.1 + 0.5, 1e-12 );
}

// NumTraits<NamedTaylorExpansion>::epsilon()/dummy_precision() must return the
// Real type (the expansion), not a bare scalar.
TEST( NamedLa, NumTraitsRealReturningValueFunctions )
{
    using NT = Eigen::NumTraits< NEpx >;
    static_assert( std::is_same_v< decltype( NT::epsilon() ), NEpx > );
    static_assert( std::is_same_v< decltype( NT::dummy_precision() ), NEpx > );
    EXPECT_DOUBLE_EQ( NT::epsilon().value(), Eigen::NumTraits< double >::epsilon() );
}
