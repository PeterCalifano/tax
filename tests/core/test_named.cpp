#include <gtest/gtest.h>

#include <array>
#include <tax/tax.hpp>
#include <type_traits>

using namespace tax::named;

namespace
{

// Order-2 expansions over a 2-dim state "x" and a 1-dim parameter block "p".
constexpr int N = 2;
using XAxis = Axis< "x", 2 >;
using PAxis = Axis< "p", 1 >;

}  // namespace

TEST( Named, VariablesCarrySingleAxis )
{
    auto x = variables< "x", N >( std::array< double, 2 >{ 1.0, 2.0 } );
    static_assert( std::is_same_v< decltype( x )::value_type, Expansion< double, N, XAxis > > );

    // x[0] = 1 + dx0 ; x[1] = 2 + dx1
    EXPECT_DOUBLE_EQ( x[0].value(), 1.0 );
    EXPECT_DOUBLE_EQ( x[1].value(), 2.0 );
    EXPECT_DOUBLE_EQ( ( x[0].inner().coeff< 1, 0 >() ), 1.0 );
    EXPECT_DOUBLE_EQ( ( x[1].inner().coeff< 0, 1 >() ), 1.0 );
}

TEST( Named, ComposeTakesUnionOfAxes )
{
    auto x = variables< "x", N >( std::array< double, 2 >{ 0.0, 0.0 } );
    auto p = variables< "p", N >( std::array< double, 1 >{ 0.0 } );

    // f = x0 * p0  — depends on both axes -> type is over {p, x} (canonical sort).
    auto f = x[0] * p[0];
    using F = decltype( f );
    static_assert( std::is_same_v< F, Expansion< double, N, PAxis, XAxis > > );
    static_assert( F::vars_v == 3 );

    // In the joint {p(0), x(1,2)} layout the cross term p0*x0 has coefficient 1.
    EXPECT_DOUBLE_EQ( ( f.inner().coeff< 1, 1, 0 >() ), 1.0 );  // p^1 x0^1 x1^0
    EXPECT_DOUBLE_EQ( ( f.inner().coeff< 1, 0, 0 >() ), 0.0 );
}

TEST( Named, CompositionIsCommutativeInType )
{
    auto x = variables< "x", N >( std::array< double, 2 >{ 0.0, 0.0 } );
    auto p = variables< "p", N >( std::array< double, 1 >{ 0.0 } );

    static_assert( std::is_same_v< decltype( x[0] * p[0] ), decltype( p[0] * x[0] ) > );
    static_assert( std::is_same_v< decltype( x[0] + p[0] ), decltype( p[0] + x[0] ) > );
}

TEST( Named, EmbedIsValuePreserving )
{
    auto x = variables< "x", N >( std::array< double, 2 >{ 3.0, 0.0 } );
    auto p = variables< "p", N >( std::array< double, 1 >{ 0.0 } );

    // Promote x[0] (over {x}) into the joint {p,x} space; value & x-coeffs survive.
    using Joint = Expansion< double, N, PAxis, XAxis >;
    auto x0_joint = x[0].embed< Joint >();
    EXPECT_DOUBLE_EQ( x0_joint.value(), 3.0 );
    EXPECT_DOUBLE_EQ( ( x0_joint.inner().coeff< 0, 1, 0 >() ), 1.0 );  // dx0
    EXPECT_DOUBLE_EQ( ( x0_joint.inner().coeff< 1, 0, 0 >() ), 0.0 );  // no p dependence
    (void)p;
}

TEST( Named, SliceProjectsOntoSubset )
{
    auto x = variables< "x", N >( std::array< double, 2 >{ 1.0, 1.0 } );
    auto p = variables< "p", N >( std::array< double, 1 >{ 5.0 } );

    // g = x0 + p0 ; slicing to {x} restricts p to its expansion point (p0 = 5).
    auto g = x[0] + p[0];
    auto gx = g.template slice< "x" >();
    static_assert( std::is_same_v< decltype( gx ), Expansion< double, N, XAxis > > );

    // g(x, p0=5) = (1 + dx0) + 5  -> value 6, dx0 coeff 1, no p term remains.
    EXPECT_DOUBLE_EQ( gx.value(), 6.0 );
    EXPECT_DOUBLE_EQ( ( gx.inner().coeff< 1, 0 >() ), 1.0 );
}

TEST( Named, SliceDropsTermsDependingOnRemovedAxis )
{
    auto x = variables< "x", N >( std::array< double, 2 >{ 0.0, 0.0 } );
    auto p = variables< "p", N >( std::array< double, 1 >{ 0.0 } );

    // h = x0 * p0 (pure cross term). Projecting onto {x} drops it entirely.
    auto h = x[0] * p[0];
    auto hx = h.template slice< "x" >();
    EXPECT_DOUBLE_EQ( hx.value(), 0.0 );
    EXPECT_DOUBLE_EQ( ( hx.inner().coeff< 1, 0 >() ), 0.0 );
    EXPECT_DOUBLE_EQ( ( hx.inner().coeff< 0, 1 >() ), 0.0 );
}

TEST( Named, ScalarOpsKeepAxisSet )
{
    auto x = variables< "x", N >( std::array< double, 2 >{ 0.0, 0.0 } );
    auto f = 2.0 * x[0] + 1.0;
    static_assert( std::is_same_v< decltype( f ), Expansion< double, N, XAxis > > );
    EXPECT_DOUBLE_EQ( f.value(), 1.0 );
    EXPECT_DOUBLE_EQ( ( f.inner().coeff< 1, 0 >() ), 2.0 );
}

TEST( Named, ComposedNumericMatchesAnonymous )
{
    // f = (x0 + 2*x1) * (1 + p0) evaluated against a hand-built anonymous TE.
    auto x = variables< "x", N >( std::array< double, 2 >{ 0.5, -0.5 } );
    auto p = variables< "p", N >( std::array< double, 1 >{ 0.25 } );

    auto f = ( x[0] + 2.0 * x[1] ) * ( 1.0 + p[0] );

    // Reference in anonymous {p, x0, x1} layout (same canonical order as f).
    using TE = tax::TE< N, 3 >;
    typename TE::Input pt{ 0.25, 0.5, -0.5 };
    auto p0 = TE::variable< 0 >( pt );
    auto a0 = TE::variable< 1 >( pt );
    auto a1 = TE::variable< 2 >( pt );
    auto ref = ( a0 + 2.0 * a1 ) * ( 1.0 + p0 );

    for ( std::size_t k = 0; k < TE::nCoefficients; ++k )
        EXPECT_DOUBLE_EQ( f.inner()[k], ref[k] ) << "coeff " << k;
}

TEST( Named, UnaryMathPreservesAxesAndMatchesAnonymous )
{
    // exp(x0 + p0): named result keeps the {p, x} axis set and matches the
    // anonymous expansion coefficient-for-coefficient.
    auto x = variables< "x", N >( std::array< double, 2 >{ 0.3, 0.0 } );
    auto p = variables< "p", N >( std::array< double, 1 >{ -0.2 } );

    auto g = exp( x[0] + p[0] );
    static_assert( std::is_same_v< decltype( g ), Expansion< double, N, PAxis, XAxis > > );

    using TE = tax::TE< N, 3 >;
    typename TE::Input pt{ -0.2, 0.3, 0.0 };
    auto ref = tax::exp( TE::variable< 1 >( pt ) + TE::variable< 0 >( pt ) );

    for ( std::size_t k = 0; k < TE::nCoefficients; ++k )
        EXPECT_DOUBLE_EQ( g.inner()[k], ref[k] ) << "coeff " << k;
}

TEST( Named, DerivByAxisName )
{
    // f = x0 * x0 + 3 * p0 ; d/dx0 = 2*x0, d/dp0 = 3 (axis set preserved).
    auto x = variables< "x", N >( std::array< double, 2 >{ 0.0, 0.0 } );
    auto p = variables< "p", N >( std::array< double, 1 >{ 0.0 } );

    auto f = x[0] * x[0] + 3.0 * p[0];
    auto fx = f.template deriv< "x", 0 >();
    auto fp = f.template deriv< "p" >();
    static_assert( std::is_same_v< decltype( fx ), Expansion< double, N, PAxis, XAxis > > );

    // d/dx0 (x0^2) has linear coefficient 2 in x0.
    EXPECT_DOUBLE_EQ( ( fx.inner().coeff< 0, 1, 0 >() ), 2.0 );
    // d/dp0 (3 p0) is the constant 3.
    EXPECT_DOUBLE_EQ( fp.value(), 3.0 );
}

TEST( Named, SubDerivativeViaDerivThenSlice )
{
    // The "sub-derivative slice": p-derivative viewed as a function of x at p0.
    // f = x0 * p0 + x1 ; df/dp0 = x0 ; sliced to {x} -> the x0 coordinate.
    auto x = variables< "x", N >( std::array< double, 2 >{ 0.0, 0.0 } );
    auto p = variables< "p", N >( std::array< double, 1 >{ 4.0 } );

    auto f = x[0] * p[0] + x[1];
    auto dfx = f.template deriv< "p" >().template slice< "x" >();
    static_assert( std::is_same_v< decltype( dfx ), Expansion< double, N, XAxis > > );

    // df/dp0 = x0 ; at p0 it is exactly the x0 coordinate variable.
    EXPECT_DOUBLE_EQ( dfx.value(), 0.0 );
    EXPECT_DOUBLE_EQ( ( dfx.inner().coeff< 1, 0 >() ), 1.0 );  // dx0
    EXPECT_DOUBLE_EQ( ( dfx.inner().coeff< 0, 1 >() ), 0.0 );  // no dx1
}

TEST( Named, DerivIntegRoundTrip )
{
    // integ then deriv along the same axis recovers the original (orders < N).
    auto x = variables< "x", N >( std::array< double, 2 >{ 0.0, 0.0 } );
    auto p = variables< "p", N >( std::array< double, 1 >{ 0.0 } );

    auto f = 1.0 + 2.0 * x[0] + p[0];  // degree 1 -> safe under integ (raises to <= N)
    auto rt = f.template integ< "x", 0 >().template deriv< "x", 0 >();

    for ( std::size_t k = 0; k < decltype( f )::Inner::nCoefficients; ++k )
        EXPECT_DOUBLE_EQ( rt.inner()[k], f.inner()[k] ) << "coeff " << k;
}
