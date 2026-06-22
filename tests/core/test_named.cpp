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
    static_assert(
        std::is_same_v< decltype( x )::value_type, NamedTaylorExpansion< double, N, XAxis > > );

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
    static_assert( std::is_same_v< F, NamedTaylorExpansion< double, N, PAxis, XAxis > > );
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
    using Joint = NamedTaylorExpansion< double, N, PAxis, XAxis >;
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
    static_assert( std::is_same_v< decltype( gx ), NamedTaylorExpansion< double, N, XAxis > > );

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
    static_assert( std::is_same_v< decltype( f ), NamedTaylorExpansion< double, N, XAxis > > );
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
    static_assert(
        std::is_same_v< decltype( g ), NamedTaylorExpansion< double, N, PAxis, XAxis > > );

    using TE = tax::TE< N, 3 >;
    typename TE::Input pt{ -0.2, 0.3, 0.0 };
    auto ref = tax::exp( TE::variable< 1 >( pt ) + TE::variable< 0 >( pt ) );

    for ( std::size_t k = 0; k < TE::nCoefficients; ++k )
        EXPECT_DOUBLE_EQ( g.inner()[k], ref[k] ) << "coeff " << k;
}

TEST( Named, PowIntegerPreservesAxesAndMatchesAnonymous )
{
    auto x = variables< "x", N >( std::array< double, 2 >{ 0.2, -0.1 } );
    auto p = variables< "p", N >( std::array< double, 1 >{ 0.3 } );

    auto f = pow( x[0] * p[0] + 1.0, 3 );
    static_assert(
        std::is_same_v< decltype( f ), NamedTaylorExpansion< double, N, PAxis, XAxis > > );

    using TE = tax::TE< N, 3 >;
    typename TE::Input pt{ 0.3, 0.2, -0.1 };
    auto ref = tax::pow( TE::variable< 1 >( pt ) * TE::variable< 0 >( pt ) + 1.0, 3 );
    for ( std::size_t k = 0; k < TE::nCoefficients; ++k )
        EXPECT_DOUBLE_EQ( f.inner()[k], ref[k] ) << "coeff " << k;
}

TEST( Named, PowRealMatchesAnonymous )
{
    auto x = variables< "x", N >( std::array< double, 2 >{ 0.1, 0.0 } );
    auto p = variables< "p", N >( std::array< double, 1 >{ 0.2 } );

    auto f = pow( 2.0 + x[0] + p[0], 0.5 );  // value 2.3 != 0
    static_assert(
        std::is_same_v< decltype( f ), NamedTaylorExpansion< double, N, PAxis, XAxis > > );

    using TE = tax::TE< N, 3 >;
    typename TE::Input pt{ 0.2, 0.1, 0.0 };
    auto ref = tax::pow( 2.0 + TE::variable< 1 >( pt ) + TE::variable< 0 >( pt ), 0.5 );
    for ( std::size_t k = 0; k < TE::nCoefficients; ++k )
        EXPECT_DOUBLE_EQ( f.inner()[k], ref[k] ) << "coeff " << k;
}

TEST( Named, Atan2TakesUnionAndMatchesAnonymous )
{
    auto x = variables< "x", N >( std::array< double, 2 >{ 0.4, 0.0 } );
    auto p = variables< "p", N >( std::array< double, 1 >{ 0.7 } );

    // y over {x}, x over {p} -> result over {p, x}.
    auto f = atan2( x[0], p[0] );
    static_assert(
        std::is_same_v< decltype( f ), NamedTaylorExpansion< double, N, PAxis, XAxis > > );

    using TE = tax::TE< N, 3 >;
    typename TE::Input pt{ 0.7, 0.4, 0.0 };
    auto ref = tax::atan2( TE::variable< 1 >( pt ), TE::variable< 0 >( pt ) );
    for ( std::size_t k = 0; k < TE::nCoefficients; ++k )
        EXPECT_DOUBLE_EQ( f.inner()[k], ref[k] ) << "coeff " << k;
}

TEST( Named, DerivByAxisName )
{
    // f = x0 * x0 + 3 * p0 ; d/dx0 = 2*x0, d/dp0 = 3 (axis set preserved).
    auto x = variables< "x", N >( std::array< double, 2 >{ 0.0, 0.0 } );
    auto p = variables< "p", N >( std::array< double, 1 >{ 0.0 } );

    auto f = x[0] * x[0] + 3.0 * p[0];
    auto fx = f.template deriv< "x", 0 >();
    auto fp = f.template deriv< "p" >();
    static_assert(
        std::is_same_v< decltype( fx ), NamedTaylorExpansion< double, N, PAxis, XAxis > > );

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
    static_assert( std::is_same_v< decltype( dfx ), NamedTaylorExpansion< double, N, XAxis > > );

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

// ===========================================================================
// Deep core-layer coverage
// ===========================================================================

namespace
{
using AAxis = Axis< "a", 1 >;
using BAxis = Axis< "b", 1 >;
using CAxis = Axis< "c", 1 >;
}  // namespace

TEST( NamedCore, ThreeAxisCanonicalOrderingAllPermutations )
{
    auto a = variables< "a", N >( std::array< double, 1 >{ 0.0 } );
    auto b = variables< "b", N >( std::array< double, 1 >{ 0.0 } );
    auto c = variables< "c", N >( std::array< double, 1 >{ 0.0 } );

    // Every association/order of the three single-axis variables yields the
    // same canonical type NamedTaylorExpansion<double, N, a, b, c>.
    using Canon = NamedTaylorExpansion< double, N, AAxis, BAxis, CAxis >;
    static_assert( std::is_same_v< decltype( a[0] * b[0] * c[0] ), Canon > );
    static_assert( std::is_same_v< decltype( c[0] * a[0] * b[0] ), Canon > );
    static_assert( std::is_same_v< decltype( b[0] * c[0] * a[0] ), Canon > );
    static_assert( std::is_same_v< decltype( ( c[0] + a[0] ) * b[0] ), Canon > );
    static_assert( std::is_same_v< decltype( c[0] + b[0] + a[0] ), Canon > );
    static_assert( Canon::vars_v == 3 );
}

TEST( NamedCore, MultiDimAxisEmbedPreservesInternalOrder )
{
    // axis "x" has 3 coordinates; embedding into {p, x} must keep their order.
    using X3 = Axis< "x", 3 >;
    auto x = variables< "x", N >( std::array< double, 3 >{ 0.0, 0.0, 0.0 } );

    auto f = x[0] + 10.0 * x[1] + 100.0 * x[2];  // over {x}
    using Joint = NamedTaylorExpansion< double, N, PAxis, X3 >;
    auto fe = f.embed< Joint >();  // layout: p(0), x0(1), x1(2), x2(3)

    EXPECT_DOUBLE_EQ( ( fe.inner().coeff< 0, 1, 0, 0 >() ), 1.0 );
    EXPECT_DOUBLE_EQ( ( fe.inner().coeff< 0, 0, 1, 0 >() ), 10.0 );
    EXPECT_DOUBLE_EQ( ( fe.inner().coeff< 0, 0, 0, 1 >() ), 100.0 );
    EXPECT_DOUBLE_EQ( ( fe.inner().coeff< 1, 0, 0, 0 >() ), 0.0 );  // no p dependence
}

TEST( NamedCore, SharedAxisUnifiedNotDuplicated )
{
    // Both operands live over {x}; the product stays over {x} (no duplication).
    auto x = variables< "x", N >( std::array< double, 2 >{ 0.0, 0.0 } );

    auto g = ( 1.0 + x[0] ) * ( 1.0 + x[1] );
    static_assert( std::is_same_v< decltype( g ), NamedTaylorExpansion< double, N, XAxis > > );
    static_assert( decltype( g )::vars_v == 2 );

    using TE = tax::TE< N, 2 >;
    typename TE::Input pt{ 0.0, 0.0 };
    auto ref = ( 1.0 + TE::variable< 0 >( pt ) ) * ( 1.0 + TE::variable< 1 >( pt ) );
    for ( std::size_t k = 0; k < TE::nCoefficients; ++k )
        EXPECT_DOUBLE_EQ( g.inner()[k], ref[k] ) << "coeff " << k;
}

TEST( NamedCore, SharedAxisAcrossThreeWayMix )
{
    // b already spans {p, x}; multiplying by another x coordinate keeps {p, x}.
    auto x = variables< "x", N >( std::array< double, 2 >{ 0.0, 0.0 } );
    auto p = variables< "p", N >( std::array< double, 1 >{ 0.0 } );

    auto b = x[0] * p[0];  // {p, x}
    auto h = b * x[1];     // still {p, x}
    static_assert(
        std::is_same_v< decltype( h ), NamedTaylorExpansion< double, N, PAxis, XAxis > > );
    // p^1 x0^1 x1^1 is degree 3 > N==2, so it truncates to zero — a good check
    // that the shared-axis path respects the order ceiling.
    EXPECT_DOUBLE_EQ( h.value(), 0.0 );
    for ( std::size_t k = 0; k < decltype( h )::Inner::nCoefficients; ++k )
        EXPECT_DOUBLE_EQ( h.inner()[k], 0.0 ) << "coeff " << k;
}

TEST( NamedCore, DivisionMatchesAnonymous )
{
    auto x = variables< "x", N >( std::array< double, 2 >{ 0.2, 0.0 } );
    auto p = variables< "p", N >( std::array< double, 1 >{ -0.1 } );

    auto f = ( 1.0 + x[0] ) / ( 2.0 + p[0] );
    static_assert(
        std::is_same_v< decltype( f ), NamedTaylorExpansion< double, N, PAxis, XAxis > > );

    using TE = tax::TE< N, 3 >;
    typename TE::Input pt{ -0.1, 0.2, 0.0 };
    auto ref = ( 1.0 + TE::variable< 1 >( pt ) ) / ( 2.0 + TE::variable< 0 >( pt ) );
    for ( std::size_t k = 0; k < TE::nCoefficients; ++k )
        EXPECT_DOUBLE_EQ( f.inner()[k], ref[k] ) << "coeff " << k;
}

TEST( NamedCore, ScalarDividedByNamed )
{
    auto x = variables< "x", N >( std::array< double, 2 >{ 0.2, 0.0 } );

    auto f = 3.0 / ( 2.0 + x[0] );  // scalar / named
    static_assert( std::is_same_v< decltype( f ), NamedTaylorExpansion< double, N, XAxis > > );

    // Matches `s / a = s * (1 / a)` on the anonymous inner expansion.
    using TE = tax::TE< N, 2 >;
    typename TE::Input pt{ 0.2, 0.0 };
    auto ref = 3.0 / ( 2.0 + TE::variable< 0 >( pt ) );
    for ( std::size_t k = 0; k < TE::nCoefficients; ++k )
        EXPECT_DOUBLE_EQ( f.inner()[k], ref[k] ) << "coeff " << k;
}

TEST( NamedCore, SubtractionAndUnaryMinus )
{
    auto x = variables< "x", N >( std::array< double, 2 >{ 0.0, 0.0 } );
    auto p = variables< "p", N >( std::array< double, 1 >{ 0.0 } );

    auto d = x[0] - p[0];
    static_assert(
        std::is_same_v< decltype( d ), NamedTaylorExpansion< double, N, PAxis, XAxis > > );
    EXPECT_DOUBLE_EQ( ( d.inner().coeff< 0, 1, 0 >() ), 1.0 );   // +dx0
    EXPECT_DOUBLE_EQ( ( d.inner().coeff< 1, 0, 0 >() ), -1.0 );  // -dp0

    auto n = -d;
    EXPECT_DOUBLE_EQ( ( n.inner().coeff< 0, 1, 0 >() ), -1.0 );
    EXPECT_DOUBLE_EQ( ( n.inner().coeff< 1, 0, 0 >() ), 1.0 );

    auto r = 3.0 - x[0];  // scalar - named
    EXPECT_DOUBLE_EQ( r.value(), 3.0 );
    EXPECT_DOUBLE_EQ( ( r.inner().coeff< 1, 0 >() ), -1.0 );
}

TEST( NamedCore, SliceMultipleNamesKeepsCrossTerms )
{
    auto a = variables< "a", N >( std::array< double, 1 >{ 1.0 } );
    auto b = variables< "b", N >( std::array< double, 1 >{ 1.0 } );
    auto c = variables< "c", N >( std::array< double, 1 >{ 7.0 } );

    auto f = a[0] + 2.0 * b[0] + 3.0 * c[0] + a[0] * b[0];  // over {a, b, c}
    static_assert(
        std::is_same_v< decltype( f ), NamedTaylorExpansion< double, N, AAxis, BAxis, CAxis > > );

    // Drop c (restrict to c0 = 7); keep the a*b cross term.
    auto fab = f.template slice< "a", "b" >();
    static_assert(
        std::is_same_v< decltype( fab ), NamedTaylorExpansion< double, N, AAxis, BAxis > > );

    EXPECT_DOUBLE_EQ( fab.value(), 25.0 );                     // 1 + 2 + 21 + 1
    EXPECT_DOUBLE_EQ( ( fab.inner().coeff< 1, 0 >() ), 2.0 );  // da: a0 + a0*b0
    EXPECT_DOUBLE_EQ( ( fab.inner().coeff< 0, 1 >() ), 3.0 );  // db: 2*b0 + a0*b0
    EXPECT_DOUBLE_EQ( ( fab.inner().coeff< 1, 1 >() ), 1.0 );  // da*db cross term
}

TEST( NamedCore, SliceNameOrderIsCanonical )
{
    auto a = variables< "a", N >( std::array< double, 1 >{ 0.0 } );
    auto b = variables< "b", N >( std::array< double, 1 >{ 0.0 } );
    auto c = variables< "c", N >( std::array< double, 1 >{ 0.0 } );
    auto f = a[0] + b[0] + c[0];

    // Requesting axes in a non-sorted order still yields the canonical type.
    auto s1 = f.template slice< "c", "a" >();
    auto s2 = f.template slice< "a", "c" >();
    static_assert(
        std::is_same_v< decltype( s1 ), NamedTaylorExpansion< double, N, AAxis, CAxis > > );
    static_assert( std::is_same_v< decltype( s1 ), decltype( s2 ) > );
}

TEST( NamedCore, ConstantExpansionAndValue )
{
    auto x = variables< "x", N >( std::array< double, 2 >{ 0.0, 0.0 } );
    NamedTaylorExpansion< double, N, XAxis > k = 5.0;  // implicit constant
    EXPECT_DOUBLE_EQ( k.value(), 5.0 );

    auto f = k + x[0];  // same axis set
    static_assert( std::is_same_v< decltype( f ), NamedTaylorExpansion< double, N, XAxis > > );
    EXPECT_DOUBLE_EQ( f.value(), 5.0 );
    EXPECT_DOUBLE_EQ( ( f.inner().coeff< 1, 0 >() ), 1.0 );
}

TEST( NamedCore, PublicNamesReachableDirectlyUnderTax )
{
    // The public API is re-exported into namespace tax, so the fully
    // qualified tax:: spellings name the same entities as tax::named::.
    static_assert( std::is_same_v< tax::NamedTaylorExpansion< double, N, XAxis >,
                                   NamedTaylorExpansion< double, N, XAxis > > );
    static_assert( std::is_same_v< tax::NE< N, XAxis >, NE< N, XAxis > > );
    static_assert( std::is_same_v< tax::Axis< "x", 2 >, XAxis > );

    // tax::variables resolves to the named factory.
    auto x = tax::variables< "x", N >( std::array< double, 2 >{ 1.0, 0.0 } );
    static_assert( std::is_same_v< decltype( x )::value_type,
                                   tax::NamedTaylorExpansion< double, N, XAxis > > );
    EXPECT_DOUBLE_EQ( x[0].value(), 1.0 );
}

TEST( NamedCore, VariableScalarFactory )
{
    // A scalar expansion point yields a single named expansion over a 1-D axis
    // (singular `variable`, not the plural collection factory).
    auto a = variable< "a", N >( 0.7 );
    static_assert( std::is_same_v< decltype( a ), NamedTaylorExpansion< double, N, AAxis > > );
    EXPECT_DOUBLE_EQ( a.value(), 0.7 );
    EXPECT_DOUBLE_EQ( ( a.inner().coeff< 1 >() ), 1.0 );

    // It composes with variables of other axes just like the array form.
    auto x = variables< "x", N >( std::array< double, 2 >{ 0.0, 0.0 } );
    auto f = a * x[0];
    static_assert(
        std::is_same_v< decltype( f ), NamedTaylorExpansion< double, N, AAxis, XAxis > > );
}

TEST( NamedCore, ImplicitPromotionFromSubsetAxes )
{
    auto x = variables< "x", N >( std::array< double, 2 >{ 0.0, 0.0 } );
    auto p = variables< "p", N >( std::array< double, 1 >{ 0.0 } );

    using Joint = NamedTaylorExpansion< double, N, PAxis, XAxis >;

    // An {x}-only value is implicitly convertible to the wider {p, x} type;
    // no explicit "+ 0 * p" padding required.
    static_assert( std::is_convertible_v< decltype( x[1] + 1.0 ), Joint > );
    static_assert( !std::is_convertible_v< Joint, decltype( x[1] ) > );  // not the other way

    Joint j = x[1] + 1.0;  // promotes {x} -> {p, x}
    EXPECT_DOUBLE_EQ( j.value(), 1.0 );
    EXPECT_DOUBLE_EQ( ( j.inner().coeff< 0, 0, 1 >() ), 1.0 );  // dx1 survives
    EXPECT_DOUBLE_EQ( ( j.inner().coeff< 1, 0, 0 >() ), 0.0 );  // no p dependence

    // Mixed arithmetic where one side is already wide also works without help.
    Joint k = ( x[0] * p[0] ) + x[1];                           // {p,x} + {x} -> {p,x}
    EXPECT_DOUBLE_EQ( ( k.inner().coeff< 1, 1, 0 >() ), 1.0 );  // p0*x0
    EXPECT_DOUBLE_EQ( ( k.inner().coeff< 0, 0, 1 >() ), 1.0 );  // x1
}

TEST( NamedCore, DerivLocalIndexWithinMultiDimAxis )
{
    using X3 = Axis< "x", 3 >;
    auto x = variables< "x", N >( std::array< double, 3 >{ 0.0, 0.0, 0.0 } );

    auto f = x[2] * x[2];  // depends only on the third x coordinate
    auto d2 = f.template deriv< "x", 2 >();
    static_assert( std::is_same_v< decltype( d2 ), NamedTaylorExpansion< double, N, X3 > > );
    EXPECT_DOUBLE_EQ( ( d2.inner().coeff< 0, 0, 1 >() ), 2.0 );  // 2*x2
    // Differentiating along an untouched coordinate gives zero.
    auto d0 = f.template deriv< "x", 0 >();
    for ( std::size_t k = 0; k < decltype( d0 )::Inner::nCoefficients; ++k )
        EXPECT_DOUBLE_EQ( d0.inner()[k], 0.0 ) << "coeff " << k;
}

TEST( NamedCore, HighOrderThreeAxisMatchesAnonymous )
{
    // Strong property test: a mixed expression over three named axes at order
    // 4, including transcendental + rational pieces, must agree with the
    // anonymous expansion coefficient-for-coefficient.
    constexpr int O = 4;
    using QAxis = Axis< "q", 1 >;
    using X2 = Axis< "x", 2 >;

    auto x = variables< "x", O >( std::array< double, 2 >{ 0.30, -0.20 } );
    auto p = variables< "p", O >( std::array< double, 1 >{ 0.10 } );
    auto q = variables< "q", O >( std::array< double, 1 >{ -0.05 } );

    auto f = exp( x[0] ) * ( 1.0 + p[0] * x[1] ) - sin( q[0] ) / ( 2.0 + x[0] );
    static_assert(
        std::is_same_v< decltype( f ), NamedTaylorExpansion< double, O, PAxis, QAxis, X2 > > );

    // Anonymous reference in canonical layout p(0), q(1), x0(2), x1(3).
    using TE = tax::TE< O, 4 >;
    typename TE::Input pt{ 0.10, -0.05, 0.30, -0.20 };
    auto ap = TE::variable< 0 >( pt );
    auto aq = TE::variable< 1 >( pt );
    auto ax0 = TE::variable< 2 >( pt );
    auto ax1 = TE::variable< 3 >( pt );
    auto ref = tax::exp( ax0 ) * ( 1.0 + ap * ax1 ) - tax::sin( aq ) / ( 2.0 + ax0 );

    for ( std::size_t k = 0; k < TE::nCoefficients; ++k )
        EXPECT_DOUBLE_EQ( f.inner()[k], ref[k] ) << "coeff " << k;
}

TEST( NamedCore, EvalThroughInnerMatchesComposition )
{
    // The named expansion's inner polynomial evaluates like the anonymous one:
    // f(x0+dx, p0+dp) via inner().eval matches direct numeric substitution.
    constexpr int O = 3;
    auto x = variables< "x", O >( std::array< double, 2 >{ 0.0, 0.0 } );
    auto p = variables< "p", O >( std::array< double, 1 >{ 0.0 } );

    auto f = ( 1.0 + x[0] ) * ( 1.0 + p[0] ) + x[1];     // {p, x}, layout p,x0,x1
    typename decltype( f )::Input dx{ 0.1, 0.2, -0.3 };  // dp, dx0, dx1
    const double got = f.inner().eval( dx );
    // Truncated product to order 3 is exact for this bilinear+linear form.
    const double expected = ( 1.0 + 0.2 ) * ( 1.0 + 0.1 ) + ( -0.3 );
    EXPECT_NEAR( got, expected, 1e-14 );
}
