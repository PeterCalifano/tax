// Tests for the dynamic-order / static-size partial specialisation
// `tax::TaylorExpansionT<T, Dynamic, M>` (alias `tax::DynOrderTE<M, T>`).
//
// Construction / shape queries, arithmetic, and math functions are verified
// against the fully-static `tax::TEn<N, M>` reference path.

#include "../testUtils.hpp"

#include <array>

#include <tax/tax.hpp>

using tax::DynOrderTE;
using tax::Dynamic;
using tax::TE;
using tax::TEn;

// =============================================================================
// Shape / sizeof
// =============================================================================

TEST( DynOrderTte, ShapeQueries )
{
    DynOrderTE< 3 > a( /*order=*/5 );
    EXPECT_EQ( a.order(), 5u );
    EXPECT_EQ( a.size(), 3u );
    EXPECT_EQ( a.coeffsSize(), tax::detail::numMonomials( 5, 3 ) );

    // size_ct is M (static); order_ct is Dynamic (= -1).
    static_assert( decltype( a )::size_ct == 3 );
    static_assert( decltype( a )::order_ct == Dynamic );
}

TEST( DynOrderTte, DefaultConstructIsEmpty )
{
    DynOrderTE< 4 > a;
    EXPECT_EQ( a.order(), 0u );
    EXPECT_EQ( a.size(), 4u );  // static M shines through even for an empty object
    EXPECT_EQ( a.coeffs().size(), 0u );  // no allocation until an explicit-order ctor is called
}

TEST( DynOrderTte, SizeofMatchesShapeBasePlusVector )
{
    // ShapeBase<Dynamic, M> for static M is OrderHolder<Dynamic> + VarsHolder<M>
    // -> sizeof == sizeof(std::size_t) (the runtime order_ member).
    // Plus the std::vector<T> coefficient buffer.
    using TTE = DynOrderTE< 3 >;
    EXPECT_GE( sizeof( TTE ), sizeof( std::size_t ) + sizeof( std::vector< double > ) );
}

// =============================================================================
// Factories
// =============================================================================

TEST( DynOrderTte, Factories )
{
    auto z = DynOrderTE< 3 >::zero( 2 );
    auto o = DynOrderTE< 3 >::one( 2 );
    auto c = DynOrderTE< 3 >::constant( 4.5, 2 );

    EXPECT_EQ( z.value(), 0.0 );
    EXPECT_EQ( o.value(), 1.0 );
    EXPECT_EQ( c.value(), 4.5 );
    EXPECT_EQ( z.order(), 2u );
    EXPECT_EQ( o.size(), 3u );
}

TEST( DynOrderTte, UnivariateVariable )
{
    auto x = DynOrderTE< 1 >::variable( 2.5, /*order=*/4 );
    EXPECT_EQ( x.value(), 2.5 );
    EXPECT_EQ( x.order(), 4u );
    EXPECT_EQ( x.coeffs()[1], 1.0 );  // dx coefficient
}

TEST( DynOrderTte, MultivariateVariableCompileTimeIndex )
{
    std::array< double, 3 > x0_vals{ 1.0, 2.0, 3.0 };
    auto x = DynOrderTE< 3 >::variable< 0 >( x0_vals, /*order=*/3 );
    auto y = DynOrderTE< 3 >::variable< 1 >( x0_vals, /*order=*/3 );
    auto z = DynOrderTE< 3 >::variable< 2 >( x0_vals, /*order=*/3 );

    EXPECT_EQ( x.value(), 1.0 );
    EXPECT_EQ( y.value(), 2.0 );
    EXPECT_EQ( z.value(), 3.0 );

    EXPECT_EQ( x.coeff( { 1, 0, 0 } ), 1.0 );
    EXPECT_EQ( y.coeff( { 0, 1, 0 } ), 1.0 );
    EXPECT_EQ( z.coeff( { 0, 0, 1 } ), 1.0 );
}

TEST( DynOrderTte, MultivariateVariableRuntimeIndex )
{
    auto x = DynOrderTE< 3 >::variable( /*x0=*/5.0, /*var_idx=*/1, /*order=*/2 );
    EXPECT_EQ( x.value(), 5.0 );
    EXPECT_EQ( x.coeff( { 0, 1, 0 } ), 1.0 );
}

TEST( DynOrderTte, RuntimeIndexOutOfRangeThrows )
{
    EXPECT_THROW( DynOrderTE< 2 >::variable( 0.0, /*var_idx=*/5, 3 ), std::out_of_range );
}

TEST( DynOrderTte, VariablesArrayStructuredBinding )
{
    std::array< double, 3 > x0_vals{ 1.0, 2.0, 3.0 };
    auto vars = DynOrderTE< 3 >::variables( x0_vals, /*order=*/4 );

    static_assert( vars.size() == 3u );  // compile-time array length
    EXPECT_EQ( vars[0].value(), 1.0 );
    EXPECT_EQ( vars[1].value(), 2.0 );
    EXPECT_EQ( vars[2].value(), 3.0 );

    for ( const auto& v : vars )
    {
        EXPECT_EQ( v.order(), 4u );
        EXPECT_EQ( v.size(), 3u );
    }
}

// =============================================================================
// Arithmetic / math against static reference
// =============================================================================

template < int N_static, int M, typename StaticFn, typename DynFn >
static void ExpectMatchesStatic( StaticFn static_op, DynFn dyn_op, double tol = 1e-12 )
{
    auto s = static_op();
    auto d = dyn_op();
    ASSERT_EQ( s.nCoefficients, d.coeffs().size() );
    for ( std::size_t i = 0; i < d.coeffs().size(); ++i )
        EXPECT_NEAR( d.coeffs()[i], s[i], tol ) << "i=" << i;
}

TEST( DynOrderTte, Add )
{
    constexpr int N_static = 5, M = 2;
    ExpectMatchesStatic< N_static, M >(
        []() {
            auto vars = TEn< N_static, M >::variables( { 1.0, 2.0 } );
            return TEn< N_static, M >{ std::get< 0 >( vars ) + std::get< 1 >( vars ) };
        },
        []() {
            auto vars = DynOrderTE< M >::variables( { 1.0, 2.0 }, /*order=*/N_static );
            return vars[0] + vars[1];
        } );
}

TEST( DynOrderTte, Mul )
{
    constexpr int N_static = 5, M = 2;
    ExpectMatchesStatic< N_static, M >(
        []() {
            auto vars = TEn< N_static, M >::variables( { 1.0, 2.0 } );
            return TEn< N_static, M >{ std::get< 0 >( vars ) * std::get< 1 >( vars ) };
        },
        []() {
            auto vars = DynOrderTE< M >::variables( { 1.0, 2.0 }, /*order=*/N_static );
            return vars[0] * vars[1];
        } );
}

TEST( DynOrderTte, Div )
{
    constexpr int N_static = 4, M = 2;
    ExpectMatchesStatic< N_static, M >(
        []() {
            auto vars = TEn< N_static, M >::variables( { 1.0, 2.0 } );
            return TEn< N_static, M >{ std::get< 0 >( vars ) / std::get< 1 >( vars ) };
        },
        []() {
            auto vars = DynOrderTE< M >::variables( { 1.0, 2.0 }, N_static );
            return vars[0] / vars[1];
        } );
}

TEST( DynOrderTte, Sin )
{
    constexpr int N_static = 5, M = 1;
    ExpectMatchesStatic< N_static, M >(
        []() {
            auto x = TE< N_static >::variable( 0.4 );
            return TE< N_static >{ tax::sin( x ) };
        },
        []() {
            auto x = DynOrderTE< 1 >::variable( 0.4, N_static );
            return tax::sin( x );
        } );
}

TEST( DynOrderTte, ExpMultivariate )
{
    constexpr int N_static = 4, M = 3;
    ExpectMatchesStatic< N_static, M >(
        []() {
            auto vars = TEn< N_static, M >::variables( { 0.1, 0.2, 0.3 } );
            return TEn< N_static, M >{ tax::exp( std::get< 0 >( vars ) + std::get< 1 >( vars )
                                                 + std::get< 2 >( vars ) ) };
        },
        []() {
            auto vars = DynOrderTE< M >::variables( { 0.1, 0.2, 0.3 }, N_static );
            return tax::exp( vars[0] + vars[1] + vars[2] );
        } );
}

TEST( DynOrderTte, CompositeAgainstStatic )
{
    constexpr int N_static = 5, M = 2;
    ExpectMatchesStatic< N_static, M >(
        []() {
            auto vars = TEn< N_static, M >::variables( { 0.5, 0.3 } );
            auto x = std::get< 0 >( vars );
            auto y = std::get< 1 >( vars );
            return TEn< N_static, M >{ tax::sin( x * y ) + tax::exp( x + y ) - tax::log( x + 1.0 ) };
        },
        []() {
            auto vars = DynOrderTE< M >::variables( { 0.5, 0.3 }, N_static );
            const auto& x = vars[0];
            const auto& y = vars[1];
            return tax::sin( x * y ) + tax::exp( x + y ) - tax::log( x + 1.0 );
        } );
}

// =============================================================================
// Order chosen at runtime, varied independently per object
// =============================================================================

TEST( DynOrderTte, OrderChosenAtRuntime )
{
    // Build the same operation at different orders and verify the lower-order
    // result matches the truncation of the higher-order one.
    std::array< double, 2 > x0_vals{ 0.5, 0.3 };

    auto low = DynOrderTE< 2 >::variables( x0_vals, /*order=*/3 );
    auto high = DynOrderTE< 2 >::variables( x0_vals, /*order=*/6 );

    auto f_low = tax::sin( low[0] * low[1] ) + tax::exp( low[0] );
    auto f_high = tax::sin( high[0] * high[1] ) + tax::exp( high[0] );

    EXPECT_EQ( f_low.order(), 3u );
    EXPECT_EQ( f_high.order(), 6u );
    EXPECT_NEAR( f_low.value(), f_high.value(), 1e-14 );

    // The order-3 truncation of f_high equals f_low (for the same operation).
    const auto S_low = f_low.coeffs().size();
    for ( std::size_t i = 0; i < S_low; ++i )
        EXPECT_NEAR( f_low.coeffs()[i], f_high.coeffs()[i], 1e-12 ) << "i=" << i;
}
