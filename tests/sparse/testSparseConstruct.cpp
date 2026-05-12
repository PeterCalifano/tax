#include "testUtils.hpp"

// Construction, factories, and dense ↔ sparse round-trip for
// SparseTaylorExpansionT<T, N, M>.

using tax::SparseTaylorExpansionT;
using tax::STE;
using tax::STEn;

template < typename Sparse, typename Dense >
static void ExpectRoundTripIdentity( const Dense& dense )
{
    Sparse sp{ dense };
    auto back = sp.toDense();
    ExpectCoeffsNear< Dense >( back, dense );
}

// =============================================================================
// Default / zero construction
// =============================================================================

TEST( SparseConstruct, DefaultIsZero )
{
    STE< 5 > s;
    EXPECT_EQ( s.nnz(), 0u );
    EXPECT_EQ( s.value(), 0.0 );
    EXPECT_EQ( s.coeff( 3 ), 0.0 );
}

TEST( SparseConstruct, ZeroFactoryIsZero )
{
    auto s = STEn< 4, 3 >::zero();
    EXPECT_EQ( s.nnz(), 0u );
}

TEST( SparseConstruct, OneFactoryIsOne )
{
    auto s = STEn< 4, 3 >::one();
    EXPECT_EQ( s.nnz(), 1u );
    EXPECT_EQ( s.value(), 1.0 );
}

TEST( SparseConstruct, ConstantNonzero )
{
    auto s = STEn< 4, 3 >::constant( 7.0 );
    EXPECT_EQ( s.nnz(), 1u );
    EXPECT_EQ( s.value(), 7.0 );
    EXPECT_EQ( s.coeff( 1 ), 0.0 );
}

TEST( SparseConstruct, ConstantZeroEmits_NoStorage )
{
    auto s = STEn< 4, 3 >::constant( 0.0 );
    EXPECT_EQ( s.nnz(), 0u );
}

// =============================================================================
// Variable factories
// =============================================================================

TEST( SparseConstruct, UnivariateVariable )
{
    auto x = STE< 5 >::variable( 2.5 );
    EXPECT_EQ( x.nnz(), 2u );
    EXPECT_EQ( x.value(), 2.5 );
    EXPECT_EQ( x.coeff( 1 ), 1.0 );

    auto dense = x.toDense();
    EXPECT_EQ( dense[0], 2.5 );
    EXPECT_EQ( dense[1], 1.0 );
    for ( std::size_t k = 2; k < dense.nCoefficients; ++k ) EXPECT_EQ( dense[k], 0.0 );
}

TEST( SparseConstruct, UnivariateVariableAtZero )
{
    auto x = STE< 5 >::variable( 0.0 );
    // The constant term is zero → only e_1 slot stored.
    EXPECT_EQ( x.nnz(), 1u );
    EXPECT_EQ( x.value(), 0.0 );
    EXPECT_EQ( x.coeff( 1 ), 1.0 );
}

TEST( SparseConstruct, MultivariateVariableCompileTime )
{
    using DA = STEn< 4, 3 >;
    DA::Input x0{ 1.0, 2.0, 3.0 };
    auto y = DA::variable< 1 >( x0 );
    EXPECT_EQ( y.value(), 2.0 );
    EXPECT_EQ( y.coeff( tax::MultiIndex< 3 >{ 0, 1, 0 } ), 1.0 );
    EXPECT_EQ( y.coeff( tax::MultiIndex< 3 >{ 1, 0, 0 } ), 0.0 );
    EXPECT_EQ( y.nnz(), 2u );
}

TEST( SparseConstruct, VariablesTupleExpansion )
{
    using DA = STEn< 3, 3 >;
    DA::Input x0{ 1.0, 2.0, 3.0 };
    auto [x, y, z] = DA::variables( x0 );
    EXPECT_EQ( x.value(), 1.0 );
    EXPECT_EQ( y.value(), 2.0 );
    EXPECT_EQ( z.value(), 3.0 );
    EXPECT_EQ( x.coeff( tax::MultiIndex< 3 >{ 1, 0, 0 } ), 1.0 );
    EXPECT_EQ( y.coeff( tax::MultiIndex< 3 >{ 0, 1, 0 } ), 1.0 );
    EXPECT_EQ( z.coeff( tax::MultiIndex< 3 >{ 0, 0, 1 } ), 1.0 );
}

// =============================================================================
// Indices are stored sorted
// =============================================================================

TEST( SparseConstruct, IndicesSortedAfterFactories )
{
    using DA = STEn< 4, 3 >;
    DA::Input x0{ 5.0, 6.0, 7.0 };
    auto z = DA::variable< 2 >( x0 );
    auto ids = z.indices();
    for ( std::size_t k = 1; k < ids.size(); ++k ) EXPECT_LT( ids[k - 1], ids[k] );
}

// =============================================================================
// Dense → sparse → dense round-trip identity
// =============================================================================

TEST( SparseConstruct, RoundTripUnivariate )
{
    tax::TE< 6 > dense;
    dense[0] = 1.0;
    dense[3] = -2.5;
    dense[6] = 0.5;
    ExpectRoundTripIdentity< STE< 6 > >( dense );
}

TEST( SparseConstruct, RoundTripMultivariateSmall )
{
    tax::TEn< 3, 2 > dense;
    dense[0] = 1.0;
    dense[1] = 2.0;
    dense[5] = 3.0;
    dense[9] = 4.0;
    ExpectRoundTripIdentity< STEn< 3, 2 > >( dense );
}

TEST( SparseConstruct, RoundTripMultivariateLarge )
{
    tax::TEn< 8, 6 > dense;
    // Sprinkle ~10 nonzeros throughout the storage.
    dense[0] = 1.0;
    dense[7] = 2.0;
    dense[42] = -3.0;
    dense[100] = 4.5;
    dense[500] = -1.5;
    dense[1000] = 0.25;
    dense[3000] = 7.7;
    ExpectRoundTripIdentity< STEn< 8, 6 > >( dense );
}

TEST( SparseConstruct, FullyDenseRoundTrip )
{
    tax::TEn< 4, 3 > dense;
    for ( std::size_t k = 0; k < dense.nCoefficients; ++k ) dense[k] = double( k ) + 0.5;
    ExpectRoundTripIdentity< STEn< 4, 3 > >( dense );
}

// =============================================================================
// coeff() lookup: hit and miss
// =============================================================================

TEST( SparseConstruct, CoeffLookupBinarySearch )
{
    tax::TEn< 5, 3 > dense;
    dense[0] = 1.0;
    dense[3] = -2.0;
    dense[7] = 3.0;
    dense[55] = 4.0;
    STEn< 5, 3 > sp{ dense };

    EXPECT_EQ( sp.coeff( std::size_t{ 0 } ), 1.0 );
    EXPECT_EQ( sp.coeff( std::size_t{ 1 } ), 0.0 );  // miss
    EXPECT_EQ( sp.coeff( std::size_t{ 3 } ), -2.0 );
    EXPECT_EQ( sp.coeff( std::size_t{ 7 } ), 3.0 );
    EXPECT_EQ( sp.coeff( std::size_t{ 55 } ), 4.0 );
    EXPECT_EQ( sp.coeff( std::size_t{ 4 } ), 0.0 );  // miss between hits
    EXPECT_EQ( sp.coeff( std::size_t{ 56 } ), 0.0 );  // miss past last
}

// =============================================================================
// nCoefficients agrees with dense
// =============================================================================

TEST( SparseConstruct, nCoefficientsMatchesDense )
{
    EXPECT_EQ( STE< 5 >::nCoefficients, tax::TE< 5 >::nCoefficients );
    EXPECT_EQ( ( STEn< 4, 3 >::nCoefficients ), ( tax::TEn< 4, 3 >::nCoefficients ) );
    EXPECT_EQ( ( STEn< 8, 6 >::nCoefficients ), ( tax::TEn< 8, 6 >::nCoefficients ) );
}
