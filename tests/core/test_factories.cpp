// Free-function variable/variables factories for unnamed isotropic expansions.
//
// tax::variable / tax::variables forward to the static-member factories on
// TaylorExpansion and must produce identical coefficients, preserve constexpr,
// and deduce the right result types.

#include <gtest/gtest.h>

#include <array>
#include <tax/tax.hpp>
#include <type_traits>

// --- Univariate: tax::variable<N>(x0) ---
TEST( Factories, IsoUnivariate )
{
    auto x = tax::variable< 5 >( 1.0 );
    static_assert( std::is_same_v< decltype( x ), tax::TE< 5 > > );
    EXPECT_EQ( x.value(), 1.0 );
    EXPECT_EQ( x[1], 1.0 );  // d/dx coefficient
    for ( std::size_t k = 2; k < x.nCoefficients; ++k ) EXPECT_EQ( x[k], 0.0 );
}

// --- Multivariate: tax::variable<I, N, M>(p) matches the static member exactly ---
TEST( Factories, IsoMultivariate )
{
    std::array< double, 2 > p{ 1.0, 2.0 };
    auto x = tax::variable< 0, 3, 2 >( p );
    auto y = tax::variable< 1, 3, 2 >( p );
    static_assert( std::is_same_v< decltype( x ), tax::TE< 3, 2 > > );

    const auto x_ref = tax::TE< 3, 2 >::variable< 0 >( p );
    const auto y_ref = tax::TE< 3, 2 >::variable< 1 >( p );
    for ( std::size_t k = 0; k < tax::TE< 3, 2 >::nCoefficients; ++k )
    {
        EXPECT_EQ( x[k], x_ref[k] ) << "x mismatch at " << k;
        EXPECT_EQ( y[k], y_ref[k] ) << "y mismatch at " << k;
    }
    EXPECT_EQ( x.value(), 1.0 );
    EXPECT_EQ( y.value(), 2.0 );
    // Independent (non-forwarding) pin: the right coordinate's unit derivative is set.
    EXPECT_EQ( x.coeff( tax::MultiIndex< 2 >{ 1, 0 } ), 1.0 );
    EXPECT_EQ( x.coeff( tax::MultiIndex< 2 >{ 0, 1 } ), 0.0 );
    EXPECT_EQ( y.coeff( tax::MultiIndex< 2 >{ 1, 0 } ), 0.0 );
    EXPECT_EQ( y.coeff( tax::MultiIndex< 2 >{ 0, 1 } ), 1.0 );
}

// --- Braced-init-list ergonomics: T defaults to double ---
TEST( Factories, IsoMultivariateBracedInit )
{
    auto x = tax::variable< 0, 3, 2 >( { 1.0, 2.0 } );
    static_assert( std::is_same_v< decltype( x ), tax::TE< 3, 2 > > );
    EXPECT_EQ( x.value(), 1.0 );
    EXPECT_EQ( x.coeff( tax::MultiIndex< 2 >{ 1, 0 } ), 1.0 );
    EXPECT_EQ( x.coeff( tax::MultiIndex< 2 >{ 0, 1 } ), 0.0 );
}

// --- Plural: tax::variables<N, M>(p) builds all coordinate variables ---
TEST( Factories, IsoVariablesPlural )
{
    std::array< double, 3 > p{ 1.0, 2.0, 3.0 };
    auto v = tax::variables< 4, 3 >( p );
    static_assert( std::is_same_v< decltype( v ), std::array< tax::TE< 4, 3 >, 3 > > );

    const auto r0 = tax::variable< 0, 4, 3 >( p );
    const auto r1 = tax::variable< 1, 4, 3 >( p );
    const auto r2 = tax::variable< 2, 4, 3 >( p );
    for ( std::size_t k = 0; k < tax::TE< 4, 3 >::nCoefficients; ++k )
    {
        EXPECT_EQ( v[0][k], r0[k] ) << "v[0] mismatch at " << k;
        EXPECT_EQ( v[1][k], r1[k] ) << "v[1] mismatch at " << k;
        EXPECT_EQ( v[2][k], r2[k] ) << "v[2] mismatch at " << k;
    }
}

// --- Forwarding preserves constexpr ---
TEST( Factories, IsoConstexpr )
{
    constexpr std::array< double, 2 > p{ 1.0, 2.0 };
    constexpr auto x = tax::variable< 0, 2, 2 >( p );
    static_assert( x.value() == 1.0 );
    static_assert( x.coeff( tax::MultiIndex< 2 >{ 1, 0 } ) == 1.0 );
    static_assert( x.coeff( tax::MultiIndex< 2 >{ 0, 1 } ) == 0.0 );
    SUCCEED();
}

// --- Non-default scalar type deduces through the factory ---
TEST( Factories, IsoFloatScalar )
{
    std::array< float, 2 > p{ 1.0F, 2.0F };
    auto x = tax::variable< 1, 3, 2 >( p );
    static_assert( std::is_same_v< decltype( x ),
                                   tax::TaylorExpansion< float, tax::IsotropicScheme< 3, 2 > > > );
    EXPECT_FLOAT_EQ( x.value(), 2.0F );
}

// --- Plural factory stays usable in constant evaluation ---
TEST( Factories, IsoVariablesPluralConstexpr )
{
    constexpr std::array< double, 2 > p{ 1.0, 2.0 };
    constexpr auto v = tax::variables< 2, 2 >( p );
    static_assert(
        std::is_same_v< std::remove_const_t< decltype( v ) >, std::array< tax::TE< 2, 2 >, 2 > > );
    static_assert( v[0].value() == 1.0 );
    static_assert( v[1].value() == 2.0 );
    static_assert( v[0].coeff( tax::MultiIndex< 2 >{ 1, 0 } ) == 1.0 );
    static_assert( v[1].coeff( tax::MultiIndex< 2 >{ 0, 1 } ) == 1.0 );
    SUCCEED();
}

// --- Order-0 (constant-only) expansion carries just the constant term ---
TEST( Factories, IsoUnivariateOrder0 )
{
    auto c = tax::variable< 0 >( 2.0 );
    static_assert( std::is_same_v< decltype( c ), tax::TE< 0 > > );
    EXPECT_EQ( c.value(), 2.0 );
    EXPECT_EQ( c.nCoefficients, 1u );
}
