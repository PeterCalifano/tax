#include "testUtils.hpp"
#include <tax/ads.hpp>

#include <cmath>

// ---------------------------------------------------------------------------
// LowOrderAdsRunner: NLI-driven ADS at low Taylor order (Losacco et al. 2024).
// ---------------------------------------------------------------------------

namespace
{

constexpr int    N = 2;
constexpr int    M = 1;
constexpr double TOL = 1e-2;

using TTE = TEn< N, M >;

// f(x) = exp(x) — strongly nonlinear over a wide interval but locally
// near-affine on small subdomains.  The central Jacobian never vanishes,
// so the nonlinearity index decreases monotonically with the box width.
//
// For exp(x) over a box of centre c and half-width h, the order-2 TTE
// has J(0) = h·exp(c) and ∂²/∂δ² = h²·exp(c), so ν = h.  Hence
// driving ν ≤ TOL bisects the domain until h ≤ TOL.
static auto expFn = []( const auto& x ) { return exp( x ); };

}  // namespace

TEST( LowOrderAds, AffineFunctionIsNotSplit )
{
    // f(x) = 2 + 3x: the central Jacobian is 3, no quadratic part, ν = 0.
    auto affine = []( const auto& x ) { return 2.0 + 3.0 * x; };

    auto runner = makeLowOrderAdsRunner< N, M >( affine, TOL );
    auto tree   = runner.run( Box< double, M >{ { 0.0 }, { 5.0 } } );

    EXPECT_EQ( tree.numDone(), 1 );
    EXPECT_EQ( tree.numActive(), 0 );
}

TEST( LowOrderAds, ExpSplitsIntoMultipleSubdomains )
{
    auto runner = makeLowOrderAdsRunner< N, M >( expFn, TOL, /*maxDepth=*/30 );
    auto tree   = runner.run( Box< double, M >{ { 0.0 }, { 1.0 } } );

    // Strongly curved on [-1,1]; must split at least once.
    EXPECT_GT( tree.numDone(), 1 );
    EXPECT_EQ( tree.numActive(), 0 );

    // Every done leaf must satisfy ν ≤ TOL (max depth never reached here).
    for ( int idx : tree.doneLeaves() )
    {
        const auto& lf = tree.node( idx ).leaf();
        const double nu = nonlinearityIndex( lf.tte );
        EXPECT_LE( nu, TOL + 1e-12 ) << "  leaf " << idx;
        // For exp(x), ν reduces exactly to the half-width.
        EXPECT_LE( lf.box.halfWidth[0], TOL + 1e-12 );
    }
}

TEST( LowOrderAds, SubdomainsCoverInitialDomain )
{
    auto runner = makeLowOrderAdsRunner< N, M >( expFn, TOL );
    auto tree   = runner.run( Box< double, M >{ { 0.0 }, { 1.0 } } );

    // Every interior probe maps to a done leaf.
    for ( double x : { -0.95, -0.4, -0.05, 0.1, 0.6, 0.95 } )
    {
        const int idx = tree.findLeaf( { x } );
        ASSERT_GE( idx, 0 ) << "  no leaf for x=" << x;
        EXPECT_TRUE( tree.node( idx ).leaf().done );
    }
}

// ---------------------------------------------------------------------------
// 2D test: function with anisotropic curvature picks the high-curvature axis.
// f(x, y) = exp(-x^2) + 0.001 * y^2.
// Most of the nonlinearity comes from x; the algorithm should split x first.
// ---------------------------------------------------------------------------

TEST( LowOrderAds, AnisotropicCurvature )
{
    auto f = []( const auto& x, const auto& y ) {
        return exp( -x * x ) + 0.001 * y * y;
    };

    auto runner = makeLowOrderAdsRunner< 2, 2 >( f, 1e-2, /*maxDepth=*/10 );
    auto tree   = runner.run(
        Box< double, 2 >{ { 0.0, 0.0 }, { 2.0, 2.0 } } );

    ASSERT_GE( tree.numNodes(), 3 );  // at least one split happened

    // The first split (root → two children) must be along dim 0 (the
    // dominant curvature axis).
    ASSERT_TRUE( tree.node( 0 ).isInternal() );
    EXPECT_EQ( tree.node( 0 ).internal().splitDim, 0 );
}
