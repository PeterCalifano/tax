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
