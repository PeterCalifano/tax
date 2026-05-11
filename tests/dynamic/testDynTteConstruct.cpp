#include "../testUtils.hpp"

#include <array>
#include <vector>

#include <tax/tax.hpp>

using tax::DynTE;
using tax::Dynamic;

// =============================================================================
// Shape queries / EBO layout
// =============================================================================

TEST( DynTteConstruct, DefaultConstructIsZeroShape )
{
    DynTE<> a;
    EXPECT_EQ( a.order(), 0u );
    EXPECT_EQ( a.size(), 0u );
    EXPECT_EQ( a.coeffs().size(), 0u );  // numMonomials(0, 0) == 0 by our overload
}

TEST( DynTteConstruct, ShapeCtor )
{
    DynTE<> a( 5, 3 );
    EXPECT_EQ( a.order(), 5u );
    EXPECT_EQ( a.size(), 3u );
    EXPECT_EQ( a.coeffsSize(), tax::detail::numMonomials( 5, 3 ) );
    EXPECT_EQ( a.coeffs().size(), tax::detail::numMonomials( 5, 3 ) );
    EXPECT_EQ( a.value(), 0.0 );
}

TEST( DynTteConstruct, ConstantFactory )
{
    auto x = DynTE<>::constant( 3.5, 4, 2 );
    EXPECT_EQ( x.order(), 4u );
    EXPECT_EQ( x.size(), 2u );
    EXPECT_EQ( x.value(), 3.5 );
    // All non-constant coefficients are zero.
    for ( std::size_t i = 1; i < x.coeffs().size(); ++i ) EXPECT_EQ( x.coeffs()[i], 0.0 );
}

TEST( DynTteConstruct, ZeroAndOneFactories )
{
    auto z = DynTE<>::zero( 3, 2 );
    auto o = DynTE<>::one( 3, 2 );
    EXPECT_EQ( z.value(), 0.0 );
    EXPECT_EQ( o.value(), 1.0 );
}

TEST( DynTteConstruct, VariableFactory_FirstCoord )
{
    auto x = DynTE<>::variable( 2.0, /*var_idx=*/0, /*order=*/3, /*size=*/3 );
    EXPECT_EQ( x.value(), 2.0 );
    // First-degree e_0 coefficient is at flat index 1 in the graded-lex layout.
    EXPECT_EQ( x.coeffs()[1], 1.0 );  // e_0 coefficient
    EXPECT_EQ( x.coeffs()[2], 0.0 );  // e_1 coefficient
    EXPECT_EQ( x.coeffs()[3], 0.0 );  // e_2 coefficient
}

TEST( DynTteConstruct, VariableFactory_OutOfRangeThrows )
{
    EXPECT_THROW( DynTE<>::variable( 0.0, 3, 2, 3 ), std::out_of_range );
}

TEST( DynTteConstruct, VariablesVector )
{
    std::array< double, 3 > x0{ 1.0, 2.0, 3.0 };
    auto vars = DynTE<>::variables( std::span< const double >( x0 ), /*order=*/2 );
    ASSERT_EQ( vars.size(), 3u );
    for ( std::size_t i = 0; i < 3; ++i )
    {
        EXPECT_EQ( vars[i].value(), x0[i] );
        EXPECT_EQ( vars[i].order(), 2u );
        EXPECT_EQ( vars[i].size(), 3u );
    }
}
