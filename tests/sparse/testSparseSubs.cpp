#include "testUtils.hpp"

// Cross-checks sparse sqrt / reciprocal / division against the dense
// forward-substitution path.  Sparse kernels here densify internally
// (the Taylor series of sqrt and 1/x are generally fully dense), so
// "agreement" here means the dense kernel result, just stored as
// SparseTaylorExpansionT.

using tax::SparseTaylorExpansionT;
using tax::STE;
using tax::STEn;
using tax::TE;
using tax::TEn;

static constexpr double kSubsTol = 1e-11;

// =============================================================================
// sqrt
// =============================================================================

TEST( SparseSubs, Sqrt_Univariate )
{
    auto sx = STE< 5 >::variable( 2.0 ) + 0.0;  // sparse 2.0 + x
    auto sresult = tax::sqrt( sx );

    TE< 5 > dx = TE< 5 >::variable( 2.0 );  // dense
    TE< 5 > dresult = tax::sqrt( dx );
    ExpectCoeffsNear< TE< 5 > >( sresult.toDense(), dresult, kSubsTol );
}

TEST( SparseSubs, Sqrt_MV_LinearOperand )
{
    using DA = STEn< 5, 3 >;
    DA::Input x0{ 1.0, 0.0, 0.0 };
    auto sx = DA::variable< 0 >( x0 ) + 0.0;  // 1.0 + x_0  (sparse, nnz=2)
    auto sresult = tax::sqrt( sx );

    using Dense = TEn< 5, 3 >;
    Dense::Input dx0{ 1.0, 0.0, 0.0 };
    Dense dx = Dense::variable< 0 >( dx0 );
    Dense dresult = tax::sqrt( dx );
    ExpectCoeffsNear< Dense >( sresult.toDense(), dresult, kSubsTol );
}

TEST( SparseSubs, Sqrt_OutputDensifies_AlongPerturbedAxes )
{
    // sqrt only densifies along the variables that actually appear in the
    // operand: sqrt(c + x_0) keeps x_1's exponents at zero, so the output
    // has just N+1 nonzero terms (one per power of x_0).
    using DA = STEn< 4, 2 >;
    DA::Input x0{ 1.0, 0.0 };
    auto sx = DA::variable< 0 >( x0 ) + 0.0;
    EXPECT_EQ( sx.nnz(), 2u );

    auto sresult = tax::sqrt( sx );
    EXPECT_EQ( sresult.nnz(), 5u );  // {1, x_0, x_0^2, x_0^3, x_0^4}

    // When all variables appear, the output is genuinely fully dense.
    DA::Input x0_full{ 1.0, 1.0 };  // both variables perturbed
    auto sy = DA::variable< 0 >( x0_full ) + DA::variable< 1 >( x0_full ) - 1.0;
    auto sresult_full = tax::sqrt( sy );
    EXPECT_EQ( sresult_full.nnz(), 15u );  // C(N+M, M) = C(6, 2) = 15
}

// =============================================================================
// reciprocal
// =============================================================================

TEST( SparseSubs, Reciprocal_Univariate )
{
    auto sx = STE< 5 >::variable( 2.0 ) + 0.0;
    auto sresult = tax::reciprocal( sx );

    TE< 5 > dx = TE< 5 >::variable( 2.0 );
    TE< 5 > dresult = TE< 5 >( 1.0 ) / dx;
    ExpectCoeffsNear< TE< 5 > >( sresult.toDense(), dresult, kSubsTol );
}

TEST( SparseSubs, Reciprocal_MV_LinearOperand )
{
    using DA = STEn< 4, 3 >;
    DA::Input x0{ 1.0, 0.0, 0.0 };
    auto sx = DA::variable< 0 >( x0 ) + 0.5;  // 1.5 + x_0
    auto sresult = tax::reciprocal( sx );

    using Dense = TEn< 4, 3 >;
    Dense::Input dx0{ 1.0, 0.0, 0.0 };
    Dense dx = Dense::variable< 0 >( dx0 ) + 0.5;
    Dense dresult = Dense( 1.0 ) / dx;
    ExpectCoeffsNear< Dense >( sresult.toDense(), dresult, kSubsTol );
}

// =============================================================================
// operator/ — sparse / sparse, scalar / sparse
// =============================================================================

TEST( SparseSubs, SparseDivSparse_Univariate )
{
    auto sa = STE< 5 >::variable( 1.0 );        // 1 + x
    auto sb = STE< 5 >::variable( 2.0 );        // 2 + x
    auto sresult = sa / sb;

    TE< 5 > da = TE< 5 >::variable( 1.0 );
    TE< 5 > db = TE< 5 >::variable( 2.0 );
    TE< 5 > dresult = da / db;
    ExpectCoeffsNear< TE< 5 > >( sresult.toDense(), dresult, kSubsTol );
}

TEST( SparseSubs, SparseDivSparse_MV )
{
    using DA = STEn< 5, 3 >;
    DA::Input x0{ 1.0, 0.5, 0.0 };
    auto [x, y, z] = DA::variables( x0 );
    auto sa = x + y;                  // sparse
    auto sb = x * y + 2.0;            // less sparse but still bounded
    auto sresult = sa / sb;

    using Dense = TEn< 5, 3 >;
    Dense::Input dx0{ 1.0, 0.5, 0.0 };
    auto [dx, dy, dz] = Dense::variables( dx0 );
    Dense da = dx + dy;
    Dense db = dx * dy + 2.0;
    Dense dresult = da / db;
    ExpectCoeffsNear< Dense >( sresult.toDense(), dresult, kSubsTol );
}

TEST( SparseSubs, ScalarDivSparse )
{
    using DA = STEn< 4, 3 >;
    DA::Input x0{ 1.0, 0.0, 0.0 };
    auto sx = DA::variable< 0 >( x0 ) + 0.5;
    auto sresult = 3.0 / sx;

    using Dense = TEn< 4, 3 >;
    Dense::Input dx0{ 1.0, 0.0, 0.0 };
    Dense dx = Dense::variable< 0 >( dx0 ) + 0.5;
    Dense dresult = Dense( 3.0 ) / dx;
    ExpectCoeffsNear< Dense >( sresult.toDense(), dresult, kSubsTol );
}

// =============================================================================
// Composition: sqrt and / together
// =============================================================================

TEST( SparseSubs, Composition_SqrtThenDivide )
{
    using DA = STEn< 4, 3 >;
    DA::Input x0{ 1.0, 0.5, 0.0 };
    auto [x, y, z] = DA::variables( x0 );
    auto sa = x + y;
    auto sb = x * y + 2.0;
    auto sresult = tax::sqrt( sa * sa + 1.0 ) / sb;

    using Dense = TEn< 4, 3 >;
    Dense::Input dx0{ 1.0, 0.5, 0.0 };
    auto [dx, dy, dz] = Dense::variables( dx0 );
    Dense da = dx + dy;
    Dense db = dx * dy + 2.0;
    Dense dresult = tax::sqrt( da * da + 1.0 ) / db;
    ExpectCoeffsNear< Dense >( sresult.toDense(), dresult, kSubsTol );
}
