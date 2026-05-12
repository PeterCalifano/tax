#include "testUtils.hpp"

// Cross-checks sparse integer-power against repeated dense multiplication.

using tax::SparseTaylorExpansionT;
using tax::STE;
using tax::STEn;
using tax::TE;
using tax::TEn;

static constexpr double kIntPowTol = 1e-11;

// Reference: build dense f^n by repeated dense multiplication.
template < typename Dense >
static Dense dense_ipow( const Dense& f, int n )
{
    if ( n == 0 )
    {
        Dense one;
        one[0] = 1.0;
        return one;
    }
    Dense out = f;
    for ( int k = 1; k < n; ++k ) out = out * f;
    return out;
}

template < typename Sparse, typename Dense >
static void ExpectIPowAgrees( const Dense& da, int n )
{
    Sparse sa{ da };
    auto sresult = tax::ipow( sa, n );
    auto dresult = dense_ipow< Dense >( da, n );
    ExpectCoeffsNear< Dense >( sresult.toDense(), dresult, kIntPowTol );
}

// =============================================================================
// Univariate
// =============================================================================

TEST( SparseIntPow, Univariate_n0 )
{
    auto x = STE< 5 >::variable( 1.5 );
    auto one = tax::ipow( x, 0 );
    EXPECT_EQ( one.value(), 1.0 );
    EXPECT_EQ( one.nnz(), 1u );
}

TEST( SparseIntPow, Univariate_n1 )
{
    auto x = STE< 5 >::variable( 1.5 );
    auto p = tax::ipow( x, 1 );
    ExpectCoeffsNear< TE< 5 > >( p.toDense(), x.toDense(), kIntPowTol );
}

TEST( SparseIntPow, Univariate_n2 )
{
    TE< 6 > da;
    da[0] = 2.0;
    da[1] = -1.0;
    da[3] = 0.5;
    ExpectIPowAgrees< STE< 6 > >( da, 2 );
}

TEST( SparseIntPow, Univariate_n3 )
{
    TE< 6 > da;
    da[0] = 2.0;
    da[1] = -1.0;
    da[3] = 0.5;
    ExpectIPowAgrees< STE< 6 > >( da, 3 );
}

TEST( SparseIntPow, Univariate_n5 )
{
    TE< 6 > da;
    da[0] = 1.0;
    da[1] = 0.3;
    da[2] = -0.1;
    ExpectIPowAgrees< STE< 6 > >( da, 5 );
}

TEST( SparseIntPow, Univariate_n7_BinaryExpVsLinearChain )
{
    // n=7 = 0b111 exercises the squaring + multiply path.
    TE< 6 > da;
    da[0] = 1.5;
    da[1] = 0.2;
    da[2] = -0.1;
    da[3] = 0.05;
    ExpectIPowAgrees< STE< 6 > >( da, 7 );
}

TEST( SparseIntPow, Univariate_n8_PureSquaring )
{
    // n=8 = 0b1000 exercises only the squaring branch.
    TE< 6 > da;
    da[0] = 1.5;
    da[1] = 0.2;
    ExpectIPowAgrees< STE< 6 > >( da, 8 );
}

// =============================================================================
// Multivariate
// =============================================================================

TEST( SparseIntPow, MV_LinearOperand_Squared )
{
    using DA = STEn< 5, 3 >;
    DA::Input x0{ 1.0, 0.0, 0.0 };
    auto a = DA::variable< 0 >( x0 ) + 0.5;     // 0.5 + x_0
    auto sq = tax::ipow( a, 2 );

    using DenseDA = TEn< 5, 3 >;
    DenseDA::Input dx0{ 1.0, 0.0, 0.0 };
    DenseDA da = DenseDA::variable< 0 >( dx0 ) + 0.5;  // materialise eagerly
    DenseDA dsq = da * da;
    ExpectCoeffsNear< DenseDA >( sq.toDense(), dsq, kIntPowTol );
}

TEST( SparseIntPow, MV_LinearOperand_n5 )
{
    using DA = STEn< 6, 4 >;
    DA::Input x0{ 1.0, 0.0, 0.0, 0.0 };
    auto a = DA::variable< 1 >( x0 ) + 0.7;     // 0.7 + x_1
    auto p = tax::ipow( a, 5 );

    using DenseDA = TEn< 6, 4 >;
    DenseDA::Input dx0{ 1.0, 0.0, 0.0, 0.0 };
    DenseDA da = DenseDA::variable< 1 >( dx0 ) + 0.7;
    auto dp = dense_ipow< DenseDA >( da, 5 );
    ExpectCoeffsNear< DenseDA >( p.toDense(), dp, kIntPowTol );
}

TEST( SparseIntPow, MV_Structured_n3 )
{
    using DA = STEn< 5, 3 >;
    DA::Input x0{ 0.5, 0.5, 0.5 };
    auto [x, y, z] = DA::variables( x0 );
    auto a = x * y + 2.0 * z;
    auto p = tax::ipow( a, 3 );

    using DenseDA = TEn< 5, 3 >;
    DenseDA::Input dx0{ 0.5, 0.5, 0.5 };
    auto [dx, dy, dz] = DenseDA::variables( dx0 );
    DenseDA da = dx * dy + 2.0 * dz;
    auto dp = dense_ipow< DenseDA >( da, 3 );
    ExpectCoeffsNear< DenseDA >( p.toDense(), dp, kIntPowTol );
}

// =============================================================================
// Truncation
// =============================================================================

TEST( SparseIntPow, TruncationProducesZeroForHighDegree )
{
    auto x = STE< 4 >::variable( 0.0 );  // pure x, degree 1
    auto p = tax::ipow( x, 5 );          // x^5 at order 4 truncates to 0
    EXPECT_EQ( p.nnz(), 0u );
}

// =============================================================================
// Error handling
// =============================================================================

TEST( SparseIntPow, NegativeExponentThrows )
{
    auto x = STE< 4 >::variable( 1.0 );
    EXPECT_THROW( tax::ipow( x, -1 ), std::invalid_argument );
    EXPECT_THROW( tax::ipow( x, -5 ), std::invalid_argument );
}
