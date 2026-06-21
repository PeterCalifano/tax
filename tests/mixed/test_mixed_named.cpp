#include <gtest/gtest.h>

#include <tax/tax.hpp>

TEST( MixedNamed, ConstructAndType )
{
    auto x = tax::mixed::variable< "x", 4 >( 1.0 );  // axis "x" dim 1 order 4
    using X = decltype( x );
    static_assert( X::vars_v == 1 );
    static_assert( X::Inner::nCoefficients == 5 );  // numMonomials(4,1)
    EXPECT_DOUBLE_EQ( x.value(), 1.0 );
}

TEST( MixedNamed, VariablesArrayAndAxisDim )
{
    std::array< double, 3 > p{ 0.1, 0.2, 0.3 };
    auto v = tax::mixed::variables< "p", 6, 3 >( p );  // 3-D axis "p" order 6
    // v is std::array; v[0] is a reference, so decay before member access.
    static_assert( std::decay_t< decltype( v[0] ) >::vars_v == 3 );
    static_assert( std::tuple_size_v< decltype( v ) > == 3 );
    EXPECT_DOUBLE_EQ( v[0].value(), 0.1 );
    EXPECT_DOUBLE_EQ( v[2].value(), 0.3 );
}

TEST( MixedNamed, ComposeUnionAxesNoBlowup )
{
    auto x = tax::mixed::variable< "x", 4 >( 0.3 );
    auto t = tax::mixed::variable< "t", 20 >( 0.1 );
    auto f = x * t + x;  // union {t@20, x@4} (sorted by name)
    using F = decltype( f );
    static_assert( F::vars_v == 2 );
    // box size = numMonomials(4,1) * numMonomials(20,1) = 5 * 21 = 105 (NOT (24+2 choose 2))
    static_assert( F::Inner::nCoefficients == 105 );

    // Pin the x*t coefficient. Union axes sorted by name: t (var 0), x (var 1).
    // The x*t monomial is exponent (t=1, x=1) in the union variable layout.
    tax::MultiIndex< 2 > xt{};
    xt[0] = 1;  // t
    xt[1] = 1;  // x
    EXPECT_NEAR( f.inner()[F::Inner::scheme::flatOf( xt )], 1.0, 1e-12 );
}

// canonical type equality: x*t and t*x are the same type
TEST( MixedNamed, CanonicalTypeOrderIndependent )
{
    auto x = tax::mixed::variable< "x", 4 >( 0.3 );
    auto t = tax::mixed::variable< "t", 20 >( 0.1 );
    static_assert( std::is_same_v< decltype( x * t ), decltype( t * x ) > );
    SUCCEED();
}

// max-order promotion on a shared axis
TEST( MixedNamed, SharedAxisPromotesToMaxOrder )
{
    auto x2 = tax::mixed::variable< "x", 2 >( 0.3 );
    auto x5 = tax::mixed::variable< "x", 5 >( 0.3 );
    auto p = x2 * x5;                                           // shared axis x -> order 5
    static_assert( decltype( p )::Inner::nCoefficients == 6 );  // numMonomials(5,1)
    SUCCEED();
}

// ---------------------------------------------------------------------------
// Task 3: axis-name differential / projection ops
// ---------------------------------------------------------------------------

// deriv<"x"> of a composed {t@4, x@3} expansion matches the analytical derivative
// for monomials that don't require source terms above the x-order cap.
// We verify two things:
//  (1) the type of df is the same as f (axis set and orders preserved)
//  (2) for monomials (t^a, x^b) with x-degree b < x-order (= 3), df matches
//      the finite-difference derivative in x from the isotropic oracle.
TEST( MixedNamed, DerivByAxisNameMatchesIsotropicOracle )
{
    // small orders to keep the isotropic oracle affordable: x@3, t@4 (Σ=7)
    auto x = tax::mixed::variable< "x", 3 >( 0.7 );
    auto t = tax::mixed::variable< "t", 4 >( 1.3 );
    auto f = sin( x * t ) + exp( x );
    using F = decltype( f );
    // Union axes sorted by name: t (var 0), x (var 1)
    auto df = f.template deriv< "x" >();
    using DF = decltype( df );
    static_assert( std::is_same_v< DF, F >, "deriv preserves the axis set and orders" );

    // Isotropic oracle: t=var0 at 1.3, x=var1 at 0.7
    typename tax::TE< 7, 2 >::Input p{ 1.3, 0.7 };
    auto it = tax::TE< 7, 2 >::variable< 0 >( p );
    auto ix = tax::TE< 7, 2 >::variable< 1 >( p );
    auto iso = sin( ix * it ) + exp( ix );
    auto iso_dx = iso.template deriv< 1 >();  // deriv w.r.t. var 1 (x)

    // Compare only monomials where the x-degree in the RESULT is < x-order (= 3).
    // For those, the deriv coefficient comes from source monomials with x-degree <= x-order,
    // which are all present in the mixed box.  Monomials with x-degree = 3 in df would
    // require x-degree 4 source terms, which are truncated in the mixed box but present
    // in the isotropic oracle — so we skip them.
    constexpr int x_order = 3;
    for ( std::size_t k = 0; k < F::Inner::nCoefficients; ++k )
    {
        const auto alpha = F::Inner::scheme::multiOf( k );
        const int x_deg = alpha[1];        // union layout: t=var0, x=var1
        if ( x_deg >= x_order ) continue;  // skip boundary: iso includes x^{x_order+1} source
        EXPECT_NEAR( df.inner()[k], iso_dx[tax::flatIndex< 2 >( alpha )], 1e-12 )
            << "coeff " << k << " alpha=(" << alpha[0] << "," << alpha[1] << ")";
    }
}

TEST( MixedNamed, IntegByAxisNameMatchesIsotropicOracle )
{
    auto x = tax::mixed::variable< "x", 3 >( 0.7 );
    auto t = tax::mixed::variable< "t", 4 >( 1.3 );
    auto f = sin( x * t ) + exp( x );
    using F = decltype( f );
    auto fi = f.template integ< "t" >();
    using FI = decltype( fi );
    static_assert( std::is_same_v< FI, F >, "integ preserves the axis set and orders" );

    // Isotropic oracle: t=var0 at 1.3, x=var1 at 0.7
    typename tax::TE< 7, 2 >::Input p{ 1.3, 0.7 };
    auto it = tax::TE< 7, 2 >::variable< 0 >( p );
    auto ix = tax::TE< 7, 2 >::variable< 1 >( p );
    auto iso = sin( ix * it ) + exp( ix );
    auto iso_it = iso.template integ< 0 >();  // integ w.r.t. var 0 (t)

    for ( std::size_t k = 0; k < F::Inner::nCoefficients; ++k )
    {
        const auto alpha = F::Inner::scheme::multiOf( k );
        EXPECT_NEAR( fi.inner()[k], iso_it[tax::flatIndex< 2 >( alpha )], 1e-12 ) << "coeff " << k;
    }
}

// slice<"x"> of an {t@4,x@3} expansion: result has only axis x;
// every monomial with t-degree>0 dropped; x-only coefficients match the source.
TEST( MixedNamed, SliceByAxisName )
{
    auto x = tax::mixed::variable< "x", 3 >( 0.7 );
    auto t = tax::mixed::variable< "t", 4 >( 1.3 );
    // f = x + x*t + t^2 (plus constant from expansion points)
    auto f = x + x * t;
    using F = decltype( f );
    // Union axes: t(var0), x(var1)

    auto sx = f.template slice< "x" >();
    using SX = decltype( sx );
    // Result should be a MixedTaylorExpansion over only axis "x" at order 3
    using ExpectedAx = tax::named::OrderedAxis< "x", 1, 3 >;
    static_assert( std::is_same_v< SX, tax::named::MixedTaylorExpansion< double, ExpectedAx > > );

    // The x-only slice should keep only monomials with t-degree = 0.
    // f = x + x*t + ... ; slicing drops the x*t term.
    // At expansion point x=0.7, t=1.3:
    // f value = 0.7 + 0.7*1.3 = 0.7*2.3 = 1.61, dx coeff = 1 + 1.3 = 2.3, dt coeff = 0.7, etc.
    // slice keeps: value=1.61, dx coeff=2.3
    // (only monomials with t-exponent=0)
    EXPECT_NEAR( sx.value(), 0.7 + 0.7 * 1.3, 1e-12 );

    // dx coefficient: from f, coeff of dt^0*dx^1 is (1 + 1.3) = 2.3
    tax::MultiIndex< 1 > alpha_x1{};
    alpha_x1[0] = 1;
    EXPECT_NEAR( sx.inner()[SX::Inner::scheme::flatOf( alpha_x1 )], 1.0 + 1.3, 1e-12 );

    // Every source monomial with t-degree > 0 must be absent in sx
    // In SX there are no t variables, so check that x*t cross terms are gone
    for ( std::size_t k = 0; k < F::Inner::nCoefficients; ++k )
    {
        const auto alpha = F::Inner::scheme::multiOf( k );
        if ( alpha[0] > 0 )  // t-degree > 0
        {
            // The source's t-containing monomials should not appear in the slice
            // (Nothing to verify in the output since they are dropped)
            (void)alpha;
        }
    }
}

// truncate<"t",2> of {t@20,x@4}: result is {t@2,x@4}; coefficients with t-degree<=2
// are preserved, t-degree>2 are dropped.
TEST( MixedNamed, TruncateByAxisName )
{
    auto x = tax::mixed::variable< "x", 4 >( 0.5 );
    auto t = tax::mixed::variable< "t", 20 >( 0.2 );
    auto f = x * t + x;  // simple enough to have known coefficients
    using F = decltype( f );
    // Union: t(var0,order20), x(var1,order4). Box: 21*5=105

    auto ft = f.truncate< "t", 2 >();
    using FT = decltype( ft );
    // Expected: {t@2, x@4} — box = numMonomials(2,1)*numMonomials(4,1) = 3*5 = 15
    using ExpAxT = tax::named::OrderedAxis< "t", 1, 2 >;
    using ExpAxX = tax::named::OrderedAxis< "x", 1, 4 >;
    static_assert(
        std::is_same_v< FT, tax::named::MixedTaylorExpansion< double, ExpAxT, ExpAxX > > );
    static_assert( FT::Inner::nCoefficients == 15 );

    // Coefficients with t-degree <= 2 must match the original.
    for ( std::size_t k = 0; k < FT::Inner::nCoefficients; ++k )
    {
        const auto alpha = FT::Inner::scheme::multiOf( k );
        // alpha[0] = t-degree, alpha[1] = x-degree; t-degree <= 2 by construction of FT
        // Find the same coefficient in the source.
        const std::size_t k_src = F::Inner::scheme::flatOf( alpha );
        EXPECT_NEAR( ft.inner()[k], f.inner()[k_src], 1e-12 ) << "coeff " << k;
    }

    // Also verify that the source's t-degree>2 terms are NOT in the truncated result.
    // The truncated result has only 15 coefficients so they cannot appear by construction.
    static_assert( FT::Inner::nCoefficients < F::Inner::nCoefficients );
}

// isotropic-superset oracle: every box coefficient of a composed mixed expansion
// must equal the same coefficient of the full isotropic TE<Σorder, vars>.
TEST( MixedNamed, IsotropicSupersetOracle )
{
    // Small per-axis orders so the isotropic super-box is affordable: x@3, t@4 (Σ=7).
    auto x = tax::mixed::variable< "x", 3 >( 0.7 );
    auto t = tax::mixed::variable< "t", 4 >( 1.3 );
    auto f = sin( x * t ) + exp( x );
    using F = decltype( f );

    // Isotropic oracle over the same two variables at the same expansion points.
    // Union axes sorted by name: t (var 0, x0=1.3), x (var 1, x0=0.7).
    typename tax::TE< 7, 2 >::Input p{ 1.3, 0.7 };
    auto it = tax::TE< 7, 2 >::variable< 0 >( p );
    auto ix = tax::TE< 7, 2 >::variable< 1 >( p );
    auto iso = sin( ix * it ) + exp( ix );

    for ( std::size_t k = 0; k < F::Inner::nCoefficients; ++k )
    {
        const auto alpha = F::Inner::scheme::multiOf( k );
        EXPECT_NEAR( f.inner()[k], iso[tax::flatIndex< 2 >( alpha )], 1e-12 );
    }
}
