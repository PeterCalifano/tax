#include "testUtils.hpp"

// Verifies sparse Cauchy kernels against dense Cauchy kernels (the source
// of truth).  For every sparse op, we round-trip operands to dense,
// compute the same op on dense, and compare the dense projection of the
// sparse result.

using tax::SparseTaylorExpansionT;
using tax::STE;
using tax::STEn;
using tax::TaylorExpansionT;
using tax::TE;
using tax::TEn;

static constexpr double kCauchyTol = 1e-12;

template < typename Sparse >
static void ExpectSparseSortedAndAllNonzero( const Sparse& s )
{
    auto ids = s.indices();
    auto vs = s.values();
    ASSERT_EQ( ids.size(), vs.size() );
    for ( std::size_t k = 0; k < vs.size(); ++k ) EXPECT_NE( vs[k], 0.0 ) << "k=" << k;
    for ( std::size_t k = 1; k < ids.size(); ++k )
        EXPECT_LT( ids[k - 1], ids[k] ) << "k=" << k;
}

template < typename Sparse, typename Dense >
static void ExpectProductAgrees( const Dense& da, const Dense& db )
{
    Sparse sa{ da };
    Sparse sb{ db };
    auto sresult = sa * sb;
    Dense dresult = da * db;
    ExpectCoeffsNear< Dense >( sresult.toDense(), dresult, kCauchyTol );
    ExpectSparseSortedAndAllNonzero< Sparse >( sresult );
}

// =============================================================================
// Univariate sparse product
// =============================================================================

TEST( SparseCauchy, UnivariateLinear )
{
    auto x = STE< 5 >::variable( 2.0 );
    auto y = STE< 5 >::variable( 3.0 );
    auto sresult = x * y;

    auto dx = TE< 5 >::variable( 2.0 );
    auto dy = TE< 5 >::variable( 3.0 );
    auto dresult = dx * dy;
    ExpectCoeffsNear< TE< 5 > >( sresult.toDense(), dresult, kCauchyTol );
}

TEST( SparseCauchy, UnivariateGeneric )
{
    TE< 6 > da;
    da[0] = 1.5;
    da[2] = -2.0;
    da[5] = 0.25;
    TE< 6 > db;
    db[1] = 3.0;
    db[3] = -1.5;
    db[6] = 4.0;
    ExpectProductAgrees< STE< 6 > >( da, db );
}

TEST( SparseCauchy, UnivariateDenseOperands )
{
    TE< 5 > da;
    TE< 5 > db;
    for ( std::size_t k = 0; k < da.nCoefficients; ++k )
    {
        da[k] = double( k ) + 0.25;
        db[k] = double( k ) - 0.5;
    }
    ExpectProductAgrees< STE< 5 > >( da, db );
}

// =============================================================================
// Multivariate sparse product
// =============================================================================

TEST( SparseCauchy, MV_LinearVarMulVar )
{
    using DA = STEn< 4, 3 >;
    DA::Input x0{ 1.0, 2.0, 3.0 };
    auto [x, y, z] = DA::variables( x0 );
    auto sresult = x * y;

    using DenseDA = TEn< 4, 3 >;
    DenseDA::Input dx0{ 1.0, 2.0, 3.0 };
    auto [dx, dy, dz] = DenseDA::variables( dx0 );
    auto dresult = dx * dy;
    ExpectCoeffsNear< DenseDA >( sresult.toDense(), dresult, kCauchyTol );
}

TEST( SparseCauchy, MV_TwoSparseLinear_N6_M4 )
{
    // Models the workload where DACE used to beat tax-dense by orders of
    // magnitude: a 2-term linear operand in a high-M, high-N space.
    // Tests use (N=6, M=4) to keep CauchyStencil consteval evaluation
    // within the default budget; the bench exercises (N=8, M=6).
    using DA = STEn< 6, 4 >;
    DA::Input x0{ 1.0, 0.0, 0.0, 0.0 };

    auto a = DA::variable< 1 >( x0 );  // 0 + x_1
    auto b = a + 0.7;                  // 0.7 + x_1
    auto sresult = a * b;

    using DenseDA = TEn< 6, 4 >;
    DenseDA::Input dx0{ 1.0, 0.0, 0.0, 0.0 };
    auto da = DenseDA::variable< 1 >( dx0 );
    auto db = da + 0.7;
    auto dresult = da * db;
    ExpectCoeffsNear< DenseDA >( sresult.toDense(), dresult, kCauchyTol );

    // Sparse result should have at most 3 slots (constant·x_1, x_1^2,
    // possibly the constant term but it's 0·0.7 = 0).
    EXPECT_LE( sresult.nnz(), 3u );
}

TEST( SparseCauchy, MV_StructuredOperands )
{
    using DA = STEn< 5, 3 >;
    DA::Input x0{ 0.5, 0.5, 0.5 };
    auto [x, y, z] = DA::variables( x0 );
    auto sa = x * y + 2.0 * z;
    auto sb = x + y;
    auto sresult = sa * sb;

    using DenseDA = TEn< 5, 3 >;
    DenseDA::Input dx0{ 0.5, 0.5, 0.5 };
    auto [dx, dy, dz] = DenseDA::variables( dx0 );
    DenseDA da = dx * dy + 2.0 * dz;
    DenseDA db = dx + dy;
    auto dresult = da * db;
    ExpectCoeffsNear< DenseDA >( sresult.toDense(), dresult, kCauchyTol );
}

TEST( SparseCauchy, MV_DenseOperands )
{
    // Sanity check: even when both operands are fully dense, sparse Cauchy
    // must produce the same result as dense Cauchy.
    using DenseDA = TEn< 4, 3 >;
    DenseDA da;
    DenseDA db;
    for ( std::size_t k = 0; k < da.nCoefficients; ++k )
    {
        da[k] = double( k ) + 0.5;
        db[k] = -double( k ) + 0.3;
    }
    ExpectProductAgrees< STEn< 4, 3 > >( da, db );
}

// =============================================================================
// Truncation
// =============================================================================

TEST( SparseCauchy, TruncatesHighDegreeTerms )
{
    // x^3 * x^3 at order 5 should be exactly zero (degree 6 > order).
    auto x = STE< 5 >::variable( 0.0 );
    auto x3 = x * x * x;
    auto out = x3 * x3;
    EXPECT_EQ( out.nnz(), 0u );
}

// =============================================================================
// Multiplicative identity / zero
// =============================================================================

TEST( SparseCauchy, MultiplyByOneIsIdentity )
{
    using DA = STEn< 4, 3 >;
    DA::Input x0{ 1.0, 2.0, 3.0 };
    auto [x, y, z] = DA::variables( x0 );
    auto a = x * y + z;
    auto one = DA::one();
    auto result = a * one;
    ExpectCoeffsNear< TEn< 4, 3 > >( result.toDense(), a.toDense(), kCauchyTol );
}

TEST( SparseCauchy, MultiplyByZeroIsZero )
{
    using DA = STEn< 4, 3 >;
    DA::Input x0{ 1.0, 2.0, 3.0 };
    auto [x, y, z] = DA::variables( x0 );
    auto a = x * y + z;
    auto zero = DA::zero();
    auto result = a * zero;
    EXPECT_EQ( result.nnz(), 0u );
}

// =============================================================================
// Self product via sparseCauchySelfProduct
// =============================================================================

TEST( SparseCauchy, SelfProduct_MatchesGeneralProduct )
{
    using DA = STEn< 5, 3 >;
    DA::Input x0{ 0.5, 0.5, 0.5 };
    auto [x, y, z] = DA::variables( x0 );
    auto a = x * y + 2.0 * z + 1.0;

    auto sq_general = a * a;
    auto sq_self = tax::detail::sparseCauchySelfProduct< double, 5, 3 >( a );

    auto general_dense = sq_general.toDense();
    auto self_dense = sq_self.toDense();
    ExpectCoeffsNear< TEn< 5, 3 > >( self_dense, general_dense, kCauchyTol );
}

TEST( SparseCauchy, SelfProduct_OfTwoSparseLinear_N6_M4 )
{
    using DA = STEn< 6, 4 >;
    DA::Input x0{ 1.0, 0.0, 0.0, 0.0 };
    auto a = DA::variable< 1 >( x0 ) + 0.7;       // 0.7 + x_1

    auto sq_self = tax::detail::sparseCauchySelfProduct< double, 6, 4 >( a );

    using DenseDA = TEn< 6, 4 >;
    DenseDA::Input dx0{ 1.0, 0.0, 0.0, 0.0 };
    DenseDA da = DenseDA::variable< 1 >( dx0 ) + 0.7;
    DenseDA dresult = da * da;
    ExpectCoeffsNear< DenseDA >( sq_self.toDense(), dresult, kCauchyTol );
}

// =============================================================================
// sparseCauchyAccumulate
// =============================================================================

TEST( SparseCauchy, Accumulate )
{
    using DA = STEn< 4, 3 >;
    DA::Input x0{ 0.5, 0.5, 0.5 };
    auto [x, y, z] = DA::variables( x0 );

    DA acc = z;
    tax::detail::sparseCauchyAccumulate< double, 4, 3 >( acc, x, y );

    auto expected = z + ( x * y );
    ExpectCoeffsNear< TEn< 4, 3 > >( acc.toDense(), expected.toDense(), kCauchyTol );
}
