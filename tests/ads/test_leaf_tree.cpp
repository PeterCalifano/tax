// tests/ads/test_leaf_tree.cpp
//
// AdsTree<Payload, M, T> — arena layout, BFS work queue, sibling links,
// point-lookup linear scan, sibling merge.

#include <gtest/gtest.h>

#include <tax/ads/box.hpp>
#include <tax/ads/leaf.hpp>
#include <tax/ads/tree.hpp>

using tax::ads::AdsTree;
using tax::ads::Box;
using tax::ads::Leaf;

namespace
{
using Tree = AdsTree< int, 2, double >;       // Payload = int (cheap to copy)
using BoxT = Box< double, 2 >;

BoxT unitBox()
{
    return BoxT{ { 0.0, 0.0 }, { 1.0, 1.0 } };
}
}  // namespace

TEST( AdsTree, InitMakesActiveLeaf )
{
    Tree tree;
    int idx = tree.init( unitBox(), /*payload=*/42, /*tEntry=*/0.0 );
    EXPECT_EQ( idx, 0 );
    EXPECT_EQ( tree.roots().size(),  1u );
    EXPECT_EQ( tree.active().size(), 1u );
    EXPECT_EQ( tree.done().size(),   0u );
    EXPECT_FALSE( tree.empty() );
    EXPECT_EQ( tree.leaf( idx ).payload,    42 );
    EXPECT_EQ( tree.leaf( idx ).depth,      0 );
    EXPECT_EQ( tree.leaf( idx ).parentIdx, -1 );
    EXPECT_FALSE( tree.leaf( idx ).done );
    EXPECT_FALSE( tree.leaf( idx ).retired );
}

TEST( AdsTree, PopFrontIsBfsOrder )
{
    Tree tree;
    const int a = tree.init( unitBox(), 1 );
    const int b = tree.init( unitBox(), 2 );
    EXPECT_EQ( tree.front(), a );        // peek matches first popFront
    EXPECT_EQ( tree.popFront(), a );
    EXPECT_EQ( tree.popFront(), b );
    EXPECT_TRUE( tree.empty() );
}

TEST( AdsTree, SplitRetiresParentAndAppendsChildren )
{
    Tree tree;
    const int root = tree.init( unitBox(), 7 );
    (void)tree.popFront();   // simulate driver dequeue

    auto pr = tree.split( root, /*dim=*/0, /*splitValue=*/0.0,
                          /*leftPayload=*/10, /*rightPayload=*/20,
                          /*tEntry=*/1.0 );
    const int L = pr.first;
    const int R = pr.second;

    EXPECT_TRUE( tree.leaf( root ).retired );
    EXPECT_EQ( tree.leaf( L ).parentIdx,  root );
    EXPECT_EQ( tree.leaf( R ).parentIdx,  root );
    EXPECT_EQ( tree.leaf( L ).siblingIdx, R    );
    EXPECT_EQ( tree.leaf( R ).siblingIdx, L    );
    EXPECT_EQ( tree.leaf( L ).splitDim,   0    );
    EXPECT_EQ( tree.leaf( R ).splitDim,   0    );
    EXPECT_EQ( tree.leaf( L ).depth,      1    );
    EXPECT_EQ( tree.leaf( R ).depth,      1    );

    // Active list now holds L and R; root is no longer active.
    EXPECT_EQ( tree.active().size(), 2u );

    // BFS order: L came first.
    EXPECT_EQ( tree.popFront(), L );
    EXPECT_EQ( tree.popFront(), R );
}

TEST( AdsTree, FinalizeMovesToDoneList )
{
    Tree tree;
    const int root = tree.init( unitBox(), 7 );
    (void)tree.popFront();
    tree.finalize( root );
    EXPECT_TRUE(  tree.leaf( root ).done );
    EXPECT_FALSE( tree.leaf( root ).retired );
    EXPECT_EQ( tree.active().size(), 0u );
    EXPECT_EQ( tree.done().size(),   1u );
    EXPECT_EQ( tree.done()[ 0 ],  root );
}

TEST( AdsTree, LeafLookupSkipsRetired )
{
    Tree tree;
    const int root = tree.init( unitBox(), 7 );
    (void)tree.popFront();
    auto pr = tree.split( root, 0, 0.0, 10, 20, 0.0 );
    const int L = pr.first;
    const int R = pr.second;

    auto fl = tree.leaf( std::array< double, 2 >{ -0.5, 0.0 } );
    auto fr = tree.leaf( std::array< double, 2 >{  0.5, 0.0 } );
    ASSERT_TRUE( fl.has_value() );
    ASSERT_TRUE( fr.has_value() );
    EXPECT_EQ( *fl, L );
    EXPECT_EQ( *fr, R );
}

TEST( AdsTree, LeafLookupNoneOutside )
{
    Tree tree;
    (void)tree.init( unitBox(), 7 );
    auto miss = tree.leaf( std::array< double, 2 >{ 2.0, 0.0 } );
    EXPECT_FALSE( miss.has_value() );
}

TEST( AdsTree, MergeRevivesParent )
{
    Tree tree;
    const int root = tree.init( unitBox(), 7 );
    (void)tree.popFront();
    auto pr = tree.split( root, 0, 0.0, 10, 20, 0.0 );
    (void)tree.popFront();   // dequeue L
    tree.finalize( pr.first );
    (void)tree.popFront();   // dequeue R
    tree.finalize( pr.second );

    tree.merge( pr.first, pr.second, /*mergedPayload=*/99 );

    EXPECT_FALSE( tree.leaf( root ).retired );
    EXPECT_TRUE(  tree.leaf( root ).done );
    EXPECT_EQ( tree.leaf( root ).payload, 99 );
    EXPECT_TRUE(  tree.leaf( pr.first  ).retired );
    EXPECT_TRUE(  tree.leaf( pr.second ).retired );

    // Done list now contains only the revived parent.
    EXPECT_EQ( tree.done().size(), 1u );
    EXPECT_EQ( tree.done()[ 0 ],   root );
}
