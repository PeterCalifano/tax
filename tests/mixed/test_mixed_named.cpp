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
