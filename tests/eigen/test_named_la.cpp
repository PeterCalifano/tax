// tests/eigen/test_named_la.cpp
//
// Eigen integration for tax::named: NumTraits (Expansion usable as an Eigen
// scalar) and the per-axis gradient/hessian/jacobian helpers.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <array>
#include <tax/tax.hpp>

using namespace tax::named;

namespace
{
constexpr int O = 3;
using XAxis = Axis< "x", 2 >;
using PAxis = Axis< "p", 1 >;
using NEpx = Expansion< double, O, PAxis, XAxis >;
}  // namespace

TEST( NamedLa, ExpansionIsUsableAsEigenScalar )
{
    auto x = variables< "x", O >( std::array< double, 2 >{ 0.0, 0.0 } );
    auto p = variables< "p", O >( std::array< double, 1 >{ 0.0 } );

    // Build an Eigen vector whose scalar is a named expansion and exercise
    // element-wise Eigen arithmetic (relies on Eigen::NumTraits<Expansion>).
    Eigen::Matrix< NEpx, 2, 1 > v;
    v( 0 ) = x[0] * p[0];
    v( 1 ) = x[1] + 1.0;  // {x}-only value promotes implicitly into the {p, x} slot

    Eigen::Matrix< NEpx, 2, 1 > w = v + v;  // element-wise scalar operator+
    EXPECT_DOUBLE_EQ( w( 1 ).value(), 2.0 );
    EXPECT_DOUBLE_EQ( ( w( 0 ).inner().coeff< 1, 1, 0 >() ), 2.0 );  // 2 * (p0*x0)
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
