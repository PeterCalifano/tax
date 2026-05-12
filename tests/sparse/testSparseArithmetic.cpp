#include "testUtils.hpp"

// Verifies sparse arithmetic against dense arithmetic.  For every sparse
// op, we round-trip operands to dense, compute the same op on dense, and
// compare the dense projection of the sparse result.

using tax::SparseTaylorExpansionT;
using tax::STE;
using tax::STEn;
using tax::TaylorExpansionT;
using tax::TE;
using tax::TEn;

template < typename Sparse, typename Dense >
static void ExpectSparseSortedAndAllNonzero( const Sparse& s )
{
    auto ids = s.indices();
    auto vs = s.values();
    ASSERT_EQ( ids.size(), vs.size() );
    for ( std::size_t k = 0; k < vs.size(); ++k ) EXPECT_NE( vs[k], 0.0 ) << "k=" << k;
    for ( std::size_t k = 1; k < ids.size(); ++k )
        EXPECT_LT( ids[k - 1], ids[k] ) << "k=" << k;
}

template < typename Sparse, typename Dense, typename Op >
static void ExpectAgreesWithDense( const Dense& da, const Dense& db, Op&& op )
{
    Sparse sa{ da };
    Sparse sb{ db };
    auto sresult = op( sa, sb );
    auto dresult = op( da, db );
    auto dproj = sresult.toDense();
    ExpectCoeffsNear< Dense >( dproj, dresult );
    ExpectSparseSortedAndAllNonzero< Sparse, Dense >( sresult );
}

// =============================================================================
// Helpers to build representative dense operands
// =============================================================================

static auto makeDenseSmallUni()
{
    TE< 5 > d;
    d[0] = 1.0;
    d[2] = -2.0;
    d[5] = 0.5;
    return d;
}

static auto makeDenseFullUni()
{
    TE< 5 > d;
    for ( std::size_t k = 0; k < d.nCoefficients; ++k ) d[k] = double( k ) + 0.25;
    return d;
}

static auto makeDenseSparseMV()
{
    TEn< 5, 3 > d;
    d[0] = 1.0;
    d[3] = 2.0;
    d[10] = -3.0;
    d[35] = 4.5;
    return d;
}

// =============================================================================
// Addition
// =============================================================================

TEST( SparseArith, Add_Univariate )
{
    ExpectAgreesWithDense< STE< 5 >, TE< 5 > >(
        makeDenseSmallUni(), makeDenseFullUni(),
        []( auto& a, auto& b ) { return a + b; } );
}

TEST( SparseArith, Add_Multivariate )
{
    auto a = makeDenseSparseMV();
    auto b = a;
    b[3] = -2.0;   // overlap that cancels out a's slot 3
    b[20] = 7.0;   // new slot
    b[0] = 0.5;    // shifts constant term
    ExpectAgreesWithDense< STEn< 5, 3 >, TEn< 5, 3 > >(
        a, b, []( auto& x, auto& y ) { return x + y; } );
}

TEST( SparseArith, Add_OverlappingCancellation )
{
    // a + (-a) should drop all entries (true sparse zero).
    auto a = makeDenseSparseMV();
    TEn< 5, 3 > b;
    for ( std::size_t k = 0; k < b.nCoefficients; ++k ) b[k] = -a[k];
    STEn< 5, 3 > sa{ a };
    STEn< 5, 3 > sb{ b };
    auto sresult = sa + sb;
    EXPECT_EQ( sresult.nnz(), 0u );
}

TEST( SparseArith, Add_ZeroIdentity )
{
    auto a = makeDenseSparseMV();
    STEn< 5, 3 > sa{ a };
    STEn< 5, 3 > z;
    auto sresult = sa + z;
    ExpectCoeffsNear< TEn< 5, 3 > >( sresult.toDense(), a );
}

// =============================================================================
// Subtraction
// =============================================================================

TEST( SparseArith, Sub_Univariate )
{
    ExpectAgreesWithDense< STE< 5 >, TE< 5 > >(
        makeDenseSmallUni(), makeDenseFullUni(),
        []( auto& a, auto& b ) { return a - b; } );
}

TEST( SparseArith, Sub_SelfIsZero )
{
    auto a = makeDenseSparseMV();
    STEn< 5, 3 > sa{ a };
    auto sresult = sa - sa;
    EXPECT_EQ( sresult.nnz(), 0u );
}

// =============================================================================
// Unary minus
// =============================================================================

TEST( SparseArith, UnaryNeg )
{
    auto a = makeDenseSparseMV();
    STEn< 5, 3 > sa{ a };
    auto neg = -sa;
    auto expected = -a;
    ExpectCoeffsNear< TEn< 5, 3 > >( neg.toDense(), expected );
    ExpectSparseSortedAndAllNonzero< STEn< 5, 3 >, TEn< 5, 3 > >( neg );
}

// =============================================================================
// Scalar multiplication
// =============================================================================

TEST( SparseArith, ScalarMul_Nonzero )
{
    auto a = makeDenseSparseMV();
    STEn< 5, 3 > sa{ a };
    auto sresult = 2.5 * sa;
    auto dresult = 2.5 * a;
    ExpectCoeffsNear< TEn< 5, 3 > >( sresult.toDense(), dresult );
}

TEST( SparseArith, ScalarMul_Zero_BecomesEmpty )
{
    auto a = makeDenseSparseMV();
    STEn< 5, 3 > sa{ a };
    auto sresult = sa * 0.0;
    EXPECT_EQ( sresult.nnz(), 0u );
}

TEST( SparseArith, ScalarDiv )
{
    auto a = makeDenseSparseMV();
    STEn< 5, 3 > sa{ a };
    auto sresult = sa / 2.0;
    auto dresult = a;
    for ( std::size_t k = 0; k < dresult.nCoefficients; ++k ) dresult[k] *= 0.5;
    ExpectCoeffsNear< TEn< 5, 3 > >( sresult.toDense(), dresult );
}

// =============================================================================
// Scalar add / sub
// =============================================================================

TEST( SparseArith, ScalarAdd_NoConstantTerm )
{
    // Build a sparse polynomial with no constant term.
    STEn< 5, 3 >::Input x0{ 1.0, 2.0, 3.0 };
    auto y = STEn< 5, 3 >::variable< 1 >( x0 );  // has constant term = 2.0
    auto z = y - 2.0;                            // strip the constant term
    EXPECT_EQ( z.value(), 0.0 );

    auto w = z + 7.0;
    EXPECT_EQ( w.value(), 7.0 );
}

TEST( SparseArith, ScalarAdd_CancelsConstantTerm )
{
    auto a = makeDenseSparseMV();    // value() = 1.0
    STEn< 5, 3 > sa{ a };
    auto sresult = sa + ( -1.0 );    // constant term should drop entirely
    EXPECT_EQ( sresult.value(), 0.0 );
    EXPECT_EQ( sresult.coeff( std::size_t{ 0 } ), 0.0 );
    // Other slots unchanged
    EXPECT_EQ( sresult.coeff( std::size_t{ 3 } ), 2.0 );
}

TEST( SparseArith, ScalarSub )
{
    auto a = makeDenseSparseMV();
    STEn< 5, 3 > sa{ a };
    auto sresult = sa - 0.5;
    auto dresult = a;
    dresult[0] -= 0.5;
    ExpectCoeffsNear< TEn< 5, 3 > >( sresult.toDense(), dresult );
}

TEST( SparseArith, ScalarMinusPolynomial )
{
    auto a = makeDenseSparseMV();
    STEn< 5, 3 > sa{ a };
    auto sresult = 1.0 - sa;
    auto dresult = a;
    for ( std::size_t k = 0; k < dresult.nCoefficients; ++k ) dresult[k] = -dresult[k];
    dresult[0] += 1.0;
    ExpectCoeffsNear< TEn< 5, 3 > >( sresult.toDense(), dresult );
}
