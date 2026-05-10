#include "testUtils.hpp"
#include <tax/ads.hpp>

#include <cmath>

// ---------------------------------------------------------------------------
// Merging stage of the low-order ADS algorithm (Losacco, Fossà, Armellin 2024).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// AdsTree::merge — direct API exercise.
// Build a manual tree with one root split into two leaves and merge them back.
// ---------------------------------------------------------------------------

TEST( AdsTreeMerge, RoundTripSplitMerge )
{
    using TTE = TEn< 2, 1 >;
    AdsTree< TTE > tree;

    // Root leaf over [-2, 2].
    const int root = tree.addLeaf( TTE{ 7.0 }, Box< double, 1 >{ { 0.0 }, { 2.0 } } );
    (void) tree.pop();   // drain the work queue for tidiness

    auto [lidx, ridx] = tree.split( root, /*dim=*/0,
                                    TTE{ 11.0 }, TTE{ 13.0 } );
    EXPECT_TRUE( tree.node( root ).isInternal() );
    EXPECT_EQ( tree.numLeaves(), 2 );

    // Drain the work queue and mark both children done.
    (void) tree.pop();
    (void) tree.pop();
    tree.markDone( lidx );
    tree.markDone( ridx );
    EXPECT_EQ( tree.numDone(), 2 );

    // Now merge — provide a brand-new TTE for the merged leaf.
    tree.merge( root, TTE{ 42.0 }, /*markDone=*/true );

    EXPECT_TRUE( tree.node( root ).isLeaf() );
    EXPECT_EQ( tree.numLeaves(), 1 );
    EXPECT_EQ( tree.numDone(), 1 );
    EXPECT_EQ( tree.numActive(), 0 );
    EXPECT_DOUBLE_EQ( tree.node( root ).leaf().tte.value(), 42.0 );

    // Parent box is reconstructed.
    EXPECT_DOUBLE_EQ( tree.node( root ).leaf().box.center[0],    0.0 );
    EXPECT_DOUBLE_EQ( tree.node( root ).leaf().box.halfWidth[0], 2.0 );

    // Point lookup goes straight to the merged leaf.
    EXPECT_EQ( tree.findLeaf( { 0.5 } ), root );
}

// ---------------------------------------------------------------------------
// Affine functions are split-then-merged into a single leaf.
// ---------------------------------------------------------------------------

TEST( MergeAds, AffineCollapsesToSingleLeaf )
{
    auto affine = []( const auto& x ) { return 2.0 + 3.0 * x; };

    // Force an artificial split by setting the tolerance to a vanishingly
    // small (but positive) value; affine functions have ν = 0 exactly so
    // we instead pre-split manually.
    auto runner = makeLowOrderAdsRunner< 2, 1 >( affine, 1e-3 );
    auto tree   = runner.run( Box< double, 1 >{ { 0.0 }, { 1.0 } } );
    ASSERT_EQ( tree.numDone(), 1 );  // already optimal: no splits

    // Manually split to mimic an over-conservative initial tree.
    const int root          = tree.roots()[0];
    auto&     root_leaf     = tree.node( root ).leaf();
    auto      box           = root_leaf.box;
    root_leaf.done          = false;
    auto [lb, rb]           = box.split( 0 );
    auto      lt            = runner.evaluate( lb );
    auto      rt            = runner.evaluate( rb );

    // Pop root from doneLeaves manually to model a fresh, un-done state.
    // (We re-mark the children done after the synthetic split.)
    auto [li, ri] = tree.split( root, 0, std::move( lt ), std::move( rt ) );
    (void) tree.pop();
    (void) tree.pop();
    tree.markDone( li );
    tree.markDone( ri );

    const int n_before = tree.numDone();
    const int merges   = mergeAds( tree, affine, 1e-3 );

    EXPECT_GE( merges, 1 );
    EXPECT_LT( tree.numDone(), n_before );
    EXPECT_TRUE( tree.node( root ).isLeaf() );
    EXPECT_EQ( tree.numLeaves(), 1 );
}

// ---------------------------------------------------------------------------
// Realistic test: split a Gaussian with a tight tolerance, then merge with
// a loose tolerance.  The post-merge tree must remain a valid covering
// (every probe still resolves to a done leaf) and the leaf count must
// shrink.
// ---------------------------------------------------------------------------

TEST( MergeAds, ShrinksTreeWhileRemainingValid )
{
    auto f = []( const auto& x ) { return exp( x ); };

    auto runner = makeLowOrderAdsRunner< 2, 1 >( f, /*nliTol=*/1e-3,
                                                 /*maxDepth=*/30 );
    auto tree   = runner.run( Box< double, 1 >{ { 0.0 }, { 1.0 } } );

    const int leaves_before = tree.numDone();
    ASSERT_GT( leaves_before, 2 );  // splitting must have happened

    const int merges = mergeAds( tree, f, /*nliTol=*/0.1 );
    EXPECT_GT( merges, 0 );
    EXPECT_LT( tree.numDone(), leaves_before );

    // The tree still covers [-1, 1]: every probe resolves to a done leaf.
    for ( double x : { -0.95, -0.4, 0.0, 0.4, 0.95 } )
    {
        const int idx = tree.findLeaf( { x } );
        ASSERT_GE( idx, 0 ) << "  no leaf for x=" << x;
        EXPECT_TRUE( tree.node( idx ).leaf().done );
    }
}
