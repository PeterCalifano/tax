#include "testUtils.hpp"
#include <tax/ads.hpp>

#include <array>
#include <span>

// ---------------------------------------------------------------------------
// Unit tests for the nonlinearity index of Losacco, Fossà, Armellin (2024).
// ---------------------------------------------------------------------------

namespace
{

template < int N, int M >
auto asSpan( const TEn< N, M >& f )
{
    return std::span< const TEn< N, M > >{ &f, 1 };
}

}  // namespace

// ---------------------------------------------------------------------------
// Affine maps have zero nonlinearity index.
// ---------------------------------------------------------------------------

TEST( NonlinearityIndex, AffineHasZero )
{
    auto x = TE< 2 >::variable( 0.0 );
    TE< 2 > f = 3.0 + 2.0 * x;   // f(δ) = 3 + 2δ

    EXPECT_DOUBLE_EQ( nonlinearityIndex( f ), 0.0 );
    const double dj = jacobianVariationNorm< double, 2, 1 >( f );
    const double j0 = centralJacobianNorm< double, 2, 1 >( f );
    EXPECT_DOUBLE_EQ( dj, 0.0 );
    EXPECT_DOUBLE_EQ( j0, 2.0 );
}

// ---------------------------------------------------------------------------
// Pure quadratic.  f(δ) = δ^2  on δ ∈ [-1,1].
//   ∂f/∂δ(δ) = 2δ,  ∂f/∂δ(0) = 0
//   bound = max_{|δ|≤1} |2δ| = 2
//   central Jacobian = 0 → ν is +∞.
// ---------------------------------------------------------------------------

TEST( NonlinearityIndex, PureQuadraticZeroCentralJacobian )
{
    auto x = TE< 2 >::variable( 0.0 );
    TE< 2 > f = x * x;

    const double j0 = centralJacobianNorm< double, 2, 1 >( f );
    const double dj = jacobianVariationNorm< double, 2, 1 >( f );
    EXPECT_DOUBLE_EQ( j0, 0.0 );
    EXPECT_DOUBLE_EQ( dj, 2.0 );
    EXPECT_TRUE( std::isinf( nonlinearityIndex( f ) ) );
}

// ---------------------------------------------------------------------------
// f(δ) = δ + (1/2) δ^2.
//   ∂f/∂δ(0) = 1,  ∂f/∂δ(δ) − 1 = δ,  bound = 1
//   ν = 1.0
// ---------------------------------------------------------------------------

TEST( NonlinearityIndex, LinearPlusHalfQuadratic )
{
    auto x = TE< 2 >::variable( 0.0 );
    TE< 2 > f = x + 0.5 * x * x;

    const double j0 = centralJacobianNorm< double, 2, 1 >( f );
    const double dj = jacobianVariationNorm< double, 2, 1 >( f );
    EXPECT_DOUBLE_EQ( j0, 1.0 );
    EXPECT_DOUBLE_EQ( dj, 1.0 );
    EXPECT_DOUBLE_EQ( nonlinearityIndex( f ), 1.0 );
}

// ---------------------------------------------------------------------------
// Halving the half-width quarters the index (linear in coefficients,
// quadratic in box size).
// ---------------------------------------------------------------------------

TEST( NonlinearityIndex, BoxRescalingHalvesQuadratic )
{
    // Build f(x) = x^2 in two boxes [-1,1] and [-0.5,0.5].
    auto build = [&]( double half ) {
        TE< 2 >::Data c{};
        c[0] = 0.0;          // f(0) = 0
        c[1] = 0.0;          // ∂f/∂x|0 = 0  →  scaled coefficient = 0
        c[2] = half * half;  // (1/2!) · 2 · half^2 = half^2  (degree-2 scaled coef)
        return TE< 2 >{ c };
    };

    TE< 2 > f_big   = build( 1.0 );    // x^2 over δ ∈ [-1,1]
    TE< 2 > f_small = build( 0.5 );    // x^2 over δ ∈ [-0.5, 0.5]

    // Both have ν = inf (zero central Jacobian).  Compare the *bound*.
    const double big   = jacobianVariationNorm< double, 2, 1 >( f_big );
    const double small = jacobianVariationNorm< double, 2, 1 >( f_small );
    EXPECT_DOUBLE_EQ( big,   2.0 );
    EXPECT_DOUBLE_EQ( small, 0.5 );
}

// ---------------------------------------------------------------------------
// Multivariate: f(δ_0, δ_1) = δ_0 + δ_1 + δ_0 · δ_1.
//   J(0) = (1, 1) → row sum = 2.
//   ∂f/∂δ_0(δ) = 1 + δ_1; variation bound = 1.
//   ∂f/∂δ_1(δ) = 1 + δ_0; variation bound = 1.
//   row-sum bound = 2 → ν = 1.
// ---------------------------------------------------------------------------

TEST( NonlinearityIndex, BilinearTwoVariables )
{
    auto [x, y] = TEn< 2, 2 >::variables( 0.0, 0.0 );
    TEn< 2, 2 > f = x + y + x * y;

    const double j0 = centralJacobianNorm< double, 2, 2 >( f );
    const double dj = jacobianVariationNorm< double, 2, 2 >( f );
    EXPECT_DOUBLE_EQ( j0, 2.0 );
    EXPECT_DOUBLE_EQ( dj, 2.0 );
    EXPECT_DOUBLE_EQ( nonlinearityIndex( f ), 1.0 );

    const auto scores = nliPerVariable( f );
    // |c_{e_0+e_1}| · |α| = 1 · 2 = 2 contributes to both variables.
    EXPECT_DOUBLE_EQ( scores[0], 2.0 );
    EXPECT_DOUBLE_EQ( scores[1], 2.0 );
}

// ---------------------------------------------------------------------------
// Split dimension picks the variable that contributes most to the bound.
// f(δ_0, δ_1) = 2δ_0 + δ_1 + 10 · δ_0^2 + 0.01 · δ_1^2
//   δ_0 contributes: 10·2 = 20.   δ_1 contributes: 0.01·2 = 0.02.
//   → split on δ_0.
// ---------------------------------------------------------------------------

TEST( NonlinearityIndex, SplitDimPicksDominantVariable )
{
    auto [x, y] = TEn< 2, 2 >::variables( 0.0, 0.0 );
    TEn< 2, 2 > f = 2.0 * x + y + 10.0 * x * x + 0.01 * y * y;

    EXPECT_EQ( nliSplitDim( f ), 0 );

    const auto scores = nliPerVariable( f );
    EXPECT_DOUBLE_EQ( scores[0], 20.0 );
    EXPECT_DOUBLE_EQ( scores[1], 0.02 );
}

// ---------------------------------------------------------------------------
// Vector-valued: identity Jacobian, only off-diagonal quadratic term.
// f_0(δ) = δ_0 + δ_1^2
// f_1(δ) = δ_1
//   J(0) = I.  Row 0 bound = |c_{0,2e_1}| · 2 = 2.  Row 1 bound = 0.
//   max row sum bound = 2.   max central row sum = 1.   → ν = 2.
// ---------------------------------------------------------------------------

TEST( NonlinearityIndex, VectorOutputs )
{
    auto [x, y] = TEn< 2, 2 >::variables( 0.0, 0.0 );

    TEn< 2, 2 > f0 = x + y * y;
    TEn< 2, 2 > f1 = y;
    std::array< TEn< 2, 2 >, 2 > out{ f0, f1 };

    const auto sp = std::span< const TEn< 2, 2 > >( out.data(), out.size() );
    EXPECT_DOUBLE_EQ( nonlinearityIndex( sp ), 2.0 );
}
