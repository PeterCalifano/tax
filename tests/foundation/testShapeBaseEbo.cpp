#include "testUtils.hpp"

#include <tax/storage/shape.hpp>

using tax::Dynamic;

// =============================================================================
// ShapeBase EBO: fully-static configuration must be empty (sizeof == 1).
// =============================================================================

TEST( ShapeBaseEbo, FullyStaticIsEmptyBase )
{
    using S = tax::detail::ShapeBase< 5, 3 >;
    static_assert( S::fully_static );
    static_assert( !S::any_dynamic );
    static_assert( !S::fully_dynamic );
    // Empty struct has sizeof 1 by the standard; EBO will collapse it when
    // used as a base class.
    EXPECT_EQ( sizeof( S ), 1u );
}

TEST( ShapeBaseEbo, OrderDynamicCarriesOneSizeT )
{
    using S = tax::detail::ShapeBase< Dynamic, 3 >;
    static_assert( !S::fully_static );
    static_assert( S::any_dynamic );
    static_assert( !S::fully_dynamic );
    EXPECT_EQ( sizeof( S ), sizeof( std::size_t ) );
}

TEST( ShapeBaseEbo, VarsDynamicCarriesOneSizeT )
{
    using S = tax::detail::ShapeBase< 5, Dynamic >;
    static_assert( !S::fully_static );
    static_assert( S::any_dynamic );
    EXPECT_EQ( sizeof( S ), sizeof( std::size_t ) );
}

TEST( ShapeBaseEbo, FullyDynamicCarriesTwoSizeT )
{
    using S = tax::detail::ShapeBase< Dynamic, Dynamic >;
    static_assert( !S::fully_static );
    static_assert( S::any_dynamic );
    static_assert( S::fully_dynamic );
    EXPECT_EQ( sizeof( S ), 2 * sizeof( std::size_t ) );
}

// =============================================================================
// TaylorExpansionT sizeof must match the raw coefficient std::array — EBO must
// collapse the ShapeBase + Expr + ExprLeaf bases to zero added size.
// =============================================================================

template < int N, int M >
static void ExpectTteFootprintMatchesArray()
{
    using TTE = tax::TaylorExpansionT< double, N, M >;
    constexpr std::size_t expected =
        sizeof( std::array< double, tax::detail::numMonomials( N, M ) > );
    EXPECT_EQ( sizeof( TTE ), expected ) << "N=" << N << " M=" << M;
}

TEST( ShapeBaseEbo, TteSizeofMatchesUnderlyingArray_TE10 )
{
    ExpectTteFootprintMatchesArray< 10, 1 >();
}

TEST( ShapeBaseEbo, TteSizeofMatchesUnderlyingArray_TEn52 )
{
    ExpectTteFootprintMatchesArray< 5, 2 >();
}

TEST( ShapeBaseEbo, TteSizeofMatchesUnderlyingArray_TEn54 )
{
    ExpectTteFootprintMatchesArray< 5, 4 >();
}

TEST( ShapeBaseEbo, TteSizeofMatchesUnderlyingArray_TEn104 )
{
    ExpectTteFootprintMatchesArray< 10, 4 >();
}

// =============================================================================
// Static order() / size() / coeffsSize() accessors are constexpr.
// =============================================================================

TEST( ShapeBaseEbo, StaticAccessorsAreConstexpr )
{
    constexpr tax::TE< 5 > x{};
    static_assert( x.order() == 5 );
    static_assert( x.size() == 1 );
    static_assert( x.coeffsSize() == 6 );

    constexpr tax::TEn< 3, 2 > y{};
    static_assert( y.order() == 3 );
    static_assert( y.size() == 2 );
    static_assert( y.coeffsSize() == tax::detail::numMonomials( 3, 2 ) );
}
