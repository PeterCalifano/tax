# tax::ads — Automatic Domain Splitting Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a header-only `tax::ads` module that performs Automatic Domain Splitting (Wittig 2015) and its Low-Order variant (LOADS) on top of the existing `tax::ode` event/integrator infrastructure, *without modifying `tax::ode`*.

**Architecture:** Leaf-only arena tree (no internal variant nodes; each leaf carries `parentIdx`/`siblingIdx`/`splitDim`/`splitValue`). Splits are signaled at accepted-step boundaries via a custom `Event<Stepper>` consisting of a `SplitTrigger` (criterion-driven) and a `SplitAction` (writes a `SplitRequest` and returns `ControlFlow::Terminate`). A BFS driver runs `tax::ode::Integrator` once per leaf and consumes the request to split-or-mark-done.

**Tech Stack:** C++23, header-only, Eigen 3.4+ via `tax::la`, Google Test, CMake. Depends on `tax::core` (`TaylorExpansionT`, `numMonomials`, `flatIndex`), `tax::ode` (`Event`, `Integrator`, `Stepper`), `tax::la` (`VecNT`, `Eigen::MatrixBase` overloads).

**Reference:** `docs/superpowers/specs/2026-06-01-tax-ads-module-design.md`. NLI math + merge predicate port from prototype branch `claude/add-verner-integrators-vEgRF`; control flow is rewritten.

**Conventions reminder:**

- `clang-format` style (4-space indent, 100-col, `.clang-format` in repo root). Run `clang-format -i` on every new header before committing.
- `PascalCase` types, `camelCase` methods/free-functions, `snake_case` locals, `lowercase` namespaces.
- `constexpr` and `noexcept` everywhere feasible.
- `static_assert(M >= 1)` for box dimension; `static_assert(N >= 0)` everywhere ADS infrastructure hits a TTE (matches ode/ads constraint that prevents `Dynamic`).
- Tests use `ExpectCoeffsNear<TTE>(actual, expected, kTol)` from `tests/testUtils.hpp` where coefficient comparison is needed.
- One test executable per source file, registered via `tax_add_test()` in `tests/CMakeLists.txt` (no subdirectory CMakeLists for `ads/` — the eight tests are flat).

---

## File Structure

```
include/tax/ads/
├── box.hpp                   Box<T, M> geometry primitive
├── leaf.hpp                  Leaf<Payload, M, T> struct (POD)
├── tree.hpp                  AdsTree<Payload, M, T> arena + BFS queue + sibling links
├── nonlinearity_index.hpp    jacobianVariationBound, linRowBound, nonlinearityIndex, nliSplitDim
├── criteria.hpp              SplitCriterion concept + TruncationCriterion + NliCriterion
├── split_event.hpp           SplitRequest + SplitTrigger + SplitAction factories
├── da_state.hpp              create, split helpers
├── driver.hpp                AdsDriver<Stepper, Criterion>
└── merge.hpp                 merge() + MergeStats

include/tax/ads.hpp           umbrella (#includes all of the above)

tests/ads/
├── test_box.cpp
├── test_leaf_tree.cpp
├── test_nonlinearity_index.cpp
├── test_criteria.cpp
├── test_split_event.cpp
├── test_da_state.cpp
├── test_driver.cpp
└── test_merge.cpp

tests/CMakeLists.txt          add 8 tax_add_test() lines
CLAUDE.md                     update ads/ section to match actual surface
```

Each file has a single responsibility; nothing leaks between headers beyond the type aliases declared in earlier tasks.

---

## Task 1: `Box<T, M>` — geometry primitive

**Files:**
- Create: `include/tax/ads/box.hpp`
- Create: `tests/ads/test_box.cpp`
- Modify: `tests/CMakeLists.txt` (add `tax_add_test(test_ads_box SOURCES ads/test_box.cpp)`)

- [ ] **Step 1.1: Write the failing test file**

`tests/ads/test_box.cpp`:

```cpp
// tests/ads/test_box.cpp
//
// Box<T, M> — construction, contains, split, denormalize, Eigen overloads.

#include <gtest/gtest.h>

#include <tax/ads/box.hpp>
#include <tax/la/types.hpp>

using tax::ads::Box;

TEST( AdsBox, DefaultCtorZero )
{
    constexpr Box< double, 2 > b{};
    EXPECT_EQ( b.center[ 0 ], 0.0 );
    EXPECT_EQ( b.center[ 1 ], 0.0 );
    EXPECT_EQ( b.halfWidth[ 0 ], 0.0 );
    EXPECT_EQ( b.halfWidth[ 1 ], 0.0 );
}

TEST( AdsBox, ArrayCtor )
{
    constexpr Box< double, 2 > b{ { 1.0, 2.0 }, { 0.5, 0.25 } };
    EXPECT_EQ( b.center[ 0 ], 1.0 );
    EXPECT_EQ( b.halfWidth[ 1 ], 0.25 );
}

TEST( AdsBox, ContainsInclusiveBoundary )
{
    constexpr Box< double, 2 > b{ { 0.0, 0.0 }, { 1.0, 1.0 } };
    EXPECT_TRUE( b.contains( { 0.5, -0.5 } ) );
    EXPECT_TRUE( b.contains( { 1.0, 1.0 } ) );    // on boundary
    EXPECT_TRUE( b.contains( { -1.0, -1.0 } ) );
    EXPECT_FALSE( b.contains( { 1.001, 0.0 } ) );
    EXPECT_FALSE( b.contains( { 0.0, -1.001 } ) );
}

TEST( AdsBox, SplitHalvesOnlyRequestedAxis )
{
    constexpr Box< double, 2 > b{ { 0.0, 0.0 }, { 1.0, 2.0 } };
    constexpr auto             pr = b.split( 0 );
    const auto&                L  = pr.first;
    const auto&                R  = pr.second;
    EXPECT_DOUBLE_EQ( L.center[ 0 ], -0.5 );
    EXPECT_DOUBLE_EQ( R.center[ 0 ],  0.5 );
    EXPECT_DOUBLE_EQ( L.halfWidth[ 0 ], 0.5 );
    EXPECT_DOUBLE_EQ( R.halfWidth[ 0 ], 0.5 );
    // Untouched axis.
    EXPECT_DOUBLE_EQ( L.center[ 1 ], 0.0 );
    EXPECT_DOUBLE_EQ( R.center[ 1 ], 0.0 );
    EXPECT_DOUBLE_EQ( L.halfWidth[ 1 ], 2.0 );
    EXPECT_DOUBLE_EQ( R.halfWidth[ 1 ], 2.0 );
}

TEST( AdsBox, Denormalize )
{
    constexpr Box< double, 2 > b{ { 1.0, 2.0 }, { 0.5, 0.25 } };
    constexpr auto             pt = b.denormalize( { 1.0, -1.0 } );
    EXPECT_DOUBLE_EQ( pt[ 0 ], 1.5 );    // 1.0 + 0.5
    EXPECT_DOUBLE_EQ( pt[ 1 ], 1.75 );   // 2.0 - 0.25
}

TEST( AdsBox, EigenCtorAndAccessors )
{
    using V = tax::la::VecNT< 2, double >;
    V c; c << 1.0, 2.0;
    V h; h << 0.5, 0.25;
    Box< double, 2 > b{ c, h };
    EXPECT_EQ( b.center[ 0 ], 1.0 );
    EXPECT_EQ( b.halfWidth[ 1 ], 0.25 );
    EXPECT_EQ( b.centerEigen()( 0 ),    1.0 );
    EXPECT_EQ( b.halfWidthEigen()( 1 ), 0.25 );
}

TEST( AdsBox, EigenContainsAndDenormalize )
{
    using V = tax::la::VecNT< 2, double >;
    Box< double, 2 > b{ { 0.0, 0.0 }, { 1.0, 1.0 } };
    V pt; pt << 0.5, -0.5;
    EXPECT_TRUE( b.contains( pt ) );
    V dn; dn << 1.0, -1.0;
    auto out = b.denormalize( dn );
    EXPECT_DOUBLE_EQ( out( 0 ),  1.0 );
    EXPECT_DOUBLE_EQ( out( 1 ), -1.0 );
}
```

- [ ] **Step 1.2: Register the test**

In `tests/CMakeLists.txt`, after the last `tax_add_test(...)` line (just before `# ODE integrator — Stage 2a`), add:

```cmake
# ADS — Stage 3a
tax_add_test(test_ads_box SOURCES ads/test_box.cpp)
```

- [ ] **Step 1.3: Run the test to verify it fails**

```bash
cmake --build build --target test_ads_box 2>&1 | tail -20
```

Expected: compilation error — `tax/ads/box.hpp: No such file or directory`.

- [ ] **Step 1.4: Implement `Box<T, M>`**

`include/tax/ads/box.hpp`:

```cpp
// include/tax/ads/box.hpp
//
// Box<T, M> — axis-aligned hyperrectangle in M-dimensional space.
// Used as the geometric primitive of the ADS tree: every leaf owns a
// Box describing the subdomain of initial conditions for which its
// payload (typically a DA-valued flow map) is valid.
//
// Storage is std::array<T, M> on both center and halfWidth so the
// type is constexpr-friendly and trivially copyable. Eigen overloads
// exist for ergonomic interop with the tax::la vector aliases.

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include <tax/la/types.hpp>

namespace tax::ads
{

template < class T, int M >
struct Box
{
    static_assert( M >= 1, "Box dimension must be at least 1" );

    std::array< T, M > center{};
    std::array< T, M > halfWidth{};

    constexpr Box() noexcept = default;

    constexpr Box( std::array< T, M > c, std::array< T, M > hw ) noexcept
        : center( c ), halfWidth( hw )
    {
    }

    template < class CenterDerived, class HalfDerived >
    Box( const Eigen::MatrixBase< CenterDerived >& c,
         const Eigen::MatrixBase< HalfDerived >&   hw )
    {
        for ( int i = 0; i < M; ++i )
        {
            center[ static_cast< std::size_t >( i ) ]    = c( i );
            halfWidth[ static_cast< std::size_t >( i ) ] = hw( i );
        }
    }

    [[nodiscard]] constexpr bool contains( const std::array< T, M >& pt ) const noexcept
    {
        for ( int i = 0; i < M; ++i )
        {
            const std::size_t k = static_cast< std::size_t >( i );
            const T           d = pt[ k ] - center[ k ];
            if ( d >  halfWidth[ k ] ) return false;
            if ( d < -halfWidth[ k ] ) return false;
        }
        return true;
    }

    [[nodiscard]] constexpr std::pair< Box, Box > split( int dim ) const noexcept
    {
        Box L{ *this };
        Box R{ *this };
        const std::size_t d = static_cast< std::size_t >( dim );
        const T           h = halfWidth[ d ] * T{ 0.5 };
        L.halfWidth[ d ] = h;
        R.halfWidth[ d ] = h;
        L.center[ d ]    = center[ d ] - h;
        R.center[ d ]    = center[ d ] + h;
        return { L, R };
    }

    [[nodiscard]] constexpr std::array< T, M > denormalize(
        const std::array< T, M >& d ) const noexcept
    {
        std::array< T, M > out{};
        for ( int i = 0; i < M; ++i )
        {
            const std::size_t k = static_cast< std::size_t >( i );
            out[ k ] = center[ k ] + halfWidth[ k ] * d[ k ];
        }
        return out;
    }

    [[nodiscard]] tax::la::VecNT< M, T > centerEigen() const noexcept
    {
        tax::la::VecNT< M, T > v;
        for ( int i = 0; i < M; ++i )
            v( i ) = center[ static_cast< std::size_t >( i ) ];
        return v;
    }

    [[nodiscard]] tax::la::VecNT< M, T > halfWidthEigen() const noexcept
    {
        tax::la::VecNT< M, T > v;
        for ( int i = 0; i < M; ++i )
            v( i ) = halfWidth[ static_cast< std::size_t >( i ) ];
        return v;
    }

    template < class Derived >
    [[nodiscard]] bool contains( const Eigen::MatrixBase< Derived >& pt ) const
    {
        for ( int i = 0; i < M; ++i )
        {
            const T d = pt( i ) - center[ static_cast< std::size_t >( i ) ];
            if ( d >  halfWidth[ static_cast< std::size_t >( i ) ] ) return false;
            if ( d < -halfWidth[ static_cast< std::size_t >( i ) ] ) return false;
        }
        return true;
    }

    template < class Derived >
    [[nodiscard]] tax::la::VecNT< M, T > denormalize(
        const Eigen::MatrixBase< Derived >& d ) const
    {
        tax::la::VecNT< M, T > out;
        for ( int i = 0; i < M; ++i )
        {
            const std::size_t k = static_cast< std::size_t >( i );
            out( i ) = center[ k ] + halfWidth[ k ] * d( i );
        }
        return out;
    }
};

}  // namespace tax::ads
```

- [ ] **Step 1.5: clang-format and rerun the test**

```bash
clang-format -i include/tax/ads/box.hpp
cmake --build build --target test_ads_box -j
ctest --test-dir build -R test_ads_box --output-on-failure
```

Expected: build succeeds, all 7 test cases pass.

- [ ] **Step 1.6: Commit**

```bash
git add include/tax/ads/box.hpp tests/ads/test_box.cpp tests/CMakeLists.txt
git commit -m "$(cat <<'EOF'
ads: introduce Box<T, M> axis-aligned subdomain primitive

Constexpr-friendly geometry type with std::array<T, M> storage. Provides
center/halfWidth ctors, inclusive contains, axis-halving split, and a
denormalize helper that maps [-1, 1]^M to box coordinates. Eigen overloads
of contains/denormalize and centerEigen()/halfWidthEigen() accessors
interop with the tax::la VecNT aliases.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `Leaf` and `AdsTree` — arena with sibling links

**Files:**
- Create: `include/tax/ads/leaf.hpp`
- Create: `include/tax/ads/tree.hpp`
- Create: `tests/ads/test_leaf_tree.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 2.1: Write the failing test file**

`tests/ads/test_leaf_tree.cpp`:

```cpp
// tests/ads/test_leaf_tree.cpp
//
// AdsTree<Payload, M, T> — arena layout, BFS work queue, sibling links,
// leaf linear scan, merge.

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

TEST( AdsTree, AddRootMakesActiveLeaf )
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
    EXPECT_EQ( tree.popFront(), a );
    EXPECT_EQ( tree.popFront(), b );
    EXPECT_TRUE( tree.empty() );
}

TEST( AdsTree, SplitRetiresParentAndAppendsChildren )
{
    Tree tree;
    const int root = tree.init( unitBox(), 7 );
    tree.popFront();   // simulate driver dequeue

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

TEST( AdsTree, MarkDoneMovesToDoneList )
{
    Tree tree;
    const int root = tree.init( unitBox(), 7 );
    tree.popFront();
    tree.finalize( root );
    EXPECT_TRUE(  tree.leaf( root ).done );
    EXPECT_FALSE( tree.leaf( root ).retired );
    EXPECT_EQ( tree.active().size(), 0u );
    EXPECT_EQ( tree.done().size(),   1u );
    EXPECT_EQ( tree.done()[ 0 ],  root );
}

TEST( AdsTree, FindLeafSkipsRetired )
{
    Tree tree;
    const int root = tree.init( unitBox(), 7 );
    tree.popFront();
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

TEST( AdsTree, FindLeafNoneOutside )
{
    Tree tree;
    tree.init( unitBox(), 7 );
    auto miss = tree.leaf( std::array< double, 2 >{ 2.0, 0.0 } );
    EXPECT_FALSE( miss.has_value() );
}

TEST( AdsTree, CollapsePairRevivesParent )
{
    Tree tree;
    const int root = tree.init( unitBox(), 7 );
    tree.popFront();
    auto pr = tree.split( root, 0, 0.0, 10, 20, 0.0 );
    tree.popFront();   // dequeue L
    tree.finalize( pr.first );
    tree.popFront();   // dequeue R
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
```

- [ ] **Step 2.2: Register the test**

In `tests/CMakeLists.txt`, after `tax_add_test(test_ads_box ...)` add:

```cmake
tax_add_test(test_ads_leaf_tree SOURCES ads/test_leaf_tree.cpp)
```

- [ ] **Step 2.3: Run to verify failure**

```bash
cmake --build build --target test_ads_leaf_tree 2>&1 | tail -10
```

Expected: `tax/ads/leaf.hpp: No such file or directory`.

- [ ] **Step 2.4: Implement `Leaf`**

`include/tax/ads/leaf.hpp`:

```cpp
// include/tax/ads/leaf.hpp
//
// Leaf<Payload, M, T> — single arena entry in AdsTree. Each leaf owns
// its Box subdomain and a Payload, plus parent / sibling indices and
// the dim/value that separated it from its sibling. A retired leaf is
// the parent of an active or done sibling pair; it stays in the arena
// so the merger can revive it via AdsTree::merge.

#pragma once

#include <tax/ads/box.hpp>

namespace tax::ads
{

template < class Payload, int M, class T = double >
struct Leaf
{
    Box< T, M > box{};
    Payload     payload{};
    int         depth      = 0;
    bool        done       = false;
    bool        retired    = false;
    int         parentIdx  = -1;
    int         siblingIdx = -1;
    int         splitDim   = -1;
    T           splitValue = T{ 0 };
    T           tEntry     = T{ 0 };
};

}  // namespace tax::ads
```

- [ ] **Step 2.5: Implement `AdsTree`**

`include/tax/ads/tree.hpp`:

```cpp
// include/tax/ads/tree.hpp
//
// AdsTree<Payload, M, T> — arena-backed leaf-only binary tree used by
// the ADS driver. Splits append two children and retire the parent;
// merges restore the parent and retire the pair. No internal "split"
// nodes: every record in the arena is a Leaf, and the tree shape is
// reconstructed from parentIdx / siblingIdx links.
//
// Work queue: std::deque<int> driven in BFS order via popFront. The
// driver pops a leaf, integrates, and either splits or finalize-s it.
//
// Lookup: leaf does a linear scan over active+done (skipping
// retired). At ADS-typical sizes (10..1000 leaves) this is faster than
// a tree walk in practice and avoids the variant-node bookkeeping.

#pragma once

#include <cassert>
#include <cstddef>
#include <deque>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <tax/la/types.hpp>

#include <tax/ads/box.hpp>
#include <tax/ads/leaf.hpp>

namespace tax::ads
{

template < class Payload, int M, class T = double >
class AdsTree
{
public:
    using LeafT = Leaf< Payload, M, T >;
    using BoxT  = Box< T, M >;

    int init( BoxT box, Payload payload, T tEntry = T{ 0 } )
    {
        LeafT l{};
        l.box     = std::move( box );
        l.payload = std::move( payload );
        l.tEntry  = tEntry;
        const int idx = static_cast< int >( leaves_.size() );
        leaves_.push_back( std::move( l ) );
        roots_.push_back( idx );
        activeList_.push_back( idx );
        workQueue_.push_back( idx );
        return idx;
    }

    [[nodiscard]] bool empty() const noexcept { return workQueue_.empty(); }

    [[nodiscard]] int front() const
    {
        assert( !workQueue_.empty() );
        return workQueue_.front();
    }

    int popFront()
    {
        assert( !workQueue_.empty() );
        const int idx = workQueue_.front();
        workQueue_.pop_front();
        return idx;
    }

    std::pair< int, int > split( int idx, int dim, T splitValue,
                                 Payload leftPayload, Payload rightPayload,
                                 T tEntry )
    {
        assert( idx >= 0 && idx < static_cast< int >( leaves_.size() ) );
        assert( !leaves_[ idx ].done && !leaves_[ idx ].retired );

        // Retire parent and remove from active list.
        leaves_[ idx ].retired = true;
        removeFromActive( idx );

        // Halve the parent's box.
        auto pr     = leaves_[ idx ].box.split( dim );
        auto& boxL  = pr.first;
        auto& boxR  = pr.second;

        const int parentDepth = leaves_[ idx ].depth;

        LeafT L{};
        L.box        = std::move( boxL );
        L.payload    = std::move( leftPayload );
        L.depth      = parentDepth + 1;
        L.parentIdx  = idx;
        L.splitDim   = dim;
        L.splitValue = splitValue;
        L.tEntry     = tEntry;

        LeafT R{};
        R.box        = std::move( boxR );
        R.payload    = std::move( rightPayload );
        R.depth      = parentDepth + 1;
        R.parentIdx  = idx;
        R.splitDim   = dim;
        R.splitValue = splitValue;
        R.tEntry     = tEntry;

        const int lIdx = static_cast< int >( leaves_.size() );
        leaves_.push_back( std::move( L ) );
        const int rIdx = static_cast< int >( leaves_.size() );
        leaves_.push_back( std::move( R ) );

        // Wire sibling links.
        leaves_[ lIdx ].siblingIdx = rIdx;
        leaves_[ rIdx ].siblingIdx = lIdx;

        activeList_.push_back( lIdx );
        activeList_.push_back( rIdx );
        workQueue_.push_back( lIdx );
        workQueue_.push_back( rIdx );

        return { lIdx, rIdx };
    }

    void finalize( int idx )
    {
        assert( idx >= 0 && idx < static_cast< int >( leaves_.size() ) );
        assert( !leaves_[ idx ].done && !leaves_[ idx ].retired );
        leaves_[ idx ].done = true;
        removeFromActive( idx );
        doneList_.push_back( idx );
    }

    void merge( int leftIdx, int rightIdx, Payload mergedPayload )
    {
        assert( leftIdx  >= 0 && leftIdx  < static_cast< int >( leaves_.size() ) );
        assert( rightIdx >= 0 && rightIdx < static_cast< int >( leaves_.size() ) );
        assert( leaves_[ leftIdx  ].parentIdx == leaves_[ rightIdx ].parentIdx );
        assert( leaves_[ leftIdx  ].siblingIdx == rightIdx );
        assert( leaves_[ rightIdx ].siblingIdx == leftIdx  );
        const int parent = leaves_[ leftIdx ].parentIdx;
        assert( parent >= 0 );
        assert( leaves_[ parent ].retired );

        // Both children move out of the done list (or active, defensively),
        // and become retired themselves. The parent revives as done.
        leaves_[ leftIdx  ].done    = false;
        leaves_[ rightIdx ].done    = false;
        leaves_[ leftIdx  ].retired = true;
        leaves_[ rightIdx ].retired = true;
        removeFromDone( leftIdx  );
        removeFromDone( rightIdx );
        // (If a caller ever collapses an active pair the assertion above
        // would catch the misuse before we got here.)

        leaves_[ parent ].retired = false;
        leaves_[ parent ].done    = true;
        leaves_[ parent ].payload = std::move( mergedPayload );
        doneList_.push_back( parent );
    }

    [[nodiscard]] const LeafT& leaf( int idx ) const noexcept
    {
        return leaves_[ static_cast< std::size_t >( idx ) ];
    }
    [[nodiscard]] LeafT& leaf( int idx ) noexcept
    {
        return leaves_[ static_cast< std::size_t >( idx ) ];
    }

    [[nodiscard]] std::span< const int > active() const noexcept
    {
        return { activeList_.data(), activeList_.size() };
    }
    [[nodiscard]] std::span< const int > done() const noexcept
    {
        return { doneList_.data(), doneList_.size() };
    }
    [[nodiscard]] std::span< const int > roots() const noexcept
    {
        return { roots_.data(), roots_.size() };
    }

    [[nodiscard]] std::optional< int > leaf(
        const std::array< T, M >& pt ) const
    {
        for ( int idx : activeList_ )
            if ( leaves_[ idx ].box.contains( pt ) ) return idx;
        for ( int idx : doneList_ )
            if ( leaves_[ idx ].box.contains( pt ) ) return idx;
        return std::nullopt;
    }

    template < class Derived >
    [[nodiscard]] std::optional< int > leaf(
        const Eigen::MatrixBase< Derived >& pt ) const
    {
        for ( int idx : activeList_ )
            if ( leaves_[ idx ].box.contains( pt ) ) return idx;
        for ( int idx : doneList_ )
            if ( leaves_[ idx ].box.contains( pt ) ) return idx;
        return std::nullopt;
    }

private:
    void removeFromActive( int idx )
    {
        for ( std::size_t i = 0; i < activeList_.size(); ++i )
        {
            if ( activeList_[ i ] == idx )
            {
                activeList_[ i ] = activeList_.back();
                activeList_.pop_back();
                return;
            }
        }
    }
    void removeFromDone( int idx )
    {
        for ( std::size_t i = 0; i < doneList_.size(); ++i )
        {
            if ( doneList_[ i ] == idx )
            {
                doneList_[ i ] = doneList_.back();
                doneList_.pop_back();
                return;
            }
        }
    }

    std::vector< LeafT > leaves_;
    std::vector< int >   activeList_;
    std::vector< int >   doneList_;
    std::vector< int >   roots_;
    std::deque< int >    workQueue_;
};

}  // namespace tax::ads
```

- [ ] **Step 2.6: clang-format and run the tests**

```bash
clang-format -i include/tax/ads/leaf.hpp include/tax/ads/tree.hpp
cmake --build build --target test_ads_leaf_tree -j
ctest --test-dir build -R test_ads_leaf_tree --output-on-failure
```

Expected: all 7 test cases pass.

- [ ] **Step 2.7: Commit**

```bash
git add include/tax/ads/leaf.hpp include/tax/ads/tree.hpp \
        tests/ads/test_leaf_tree.cpp tests/CMakeLists.txt
git commit -m "$(cat <<'EOF'
ads: add Leaf POD + AdsTree arena with sibling links

AdsTree<Payload, M, T> is a leaf-only binary tree: every arena entry is
a Leaf with parentIdx, siblingIdx, splitDim, splitValue. Splits retire
the parent in place, append two children with sibling cross-links, and
push both onto a BFS work queue. finalize moves an active leaf into the
done list; merge (used later by the merger) revives the parent
and retires the pair. leaf is a linear scan over active+done,
skipping retired entries — appropriate for the typical 10..1000 leaves.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `nonlinearity_index.hpp` — NLI math

**Files:**
- Create: `include/tax/ads/nonlinearity_index.hpp`
- Create: `tests/ads/test_nonlinearity_index.cpp`
- Modify: `tests/CMakeLists.txt`

**Context:** These helpers operate on `TaylorExpansionT<T, N, M>` over the *normalized* box `[-1, 1]^M`. The polynomial coefficients are already in this normalized basis when the DA state is built from a Box by `create` (Task 6). For each scalar TE `f`:

- `linRowBound(f)`  = sum of `|coeff(i)|` over monomials of degree 1. Bounds `|∂f/∂x_i(0)|` summed across `i`.
- `jacobianVariationBound(f)`  = per-coordinate vector `v`, where `v[j]` = sum of `|coeff(i)| * (deg_j(i))` over monomials of degree ≥ 2. This bounds `max_{x ∈ [-1,1]^M} |∂f/∂x_j(x) - ∂f/∂x_j(0)|`.
- `nonlinearityIndex(F)`  = `max_i ||v_i||_1 / linRowBound(F_i)`, the LOADS NLI.
- `nliSplitDim(F)`  = `argmax_j Σ_i v_i[j]` — pick the coordinate whose nonlinear contribution dominates.

- [ ] **Step 3.1: Write the failing test file**

`tests/ads/test_nonlinearity_index.cpp`:

```cpp
// tests/ads/test_nonlinearity_index.cpp
//
// jacobianVariationBound, linRowBound, nonlinearityIndex, nliSplitDim.

#include <gtest/gtest.h>

#include <cmath>

#include <tax/ads/nonlinearity_index.hpp>
#include <tax/la/types.hpp>
#include <tax/tax.hpp>

using tax::TEn;
using tax::ads::detail::jacobianVariationBound;
using tax::ads::detail::linRowBound;
using tax::ads::detail::nliSplitDim;
using tax::ads::detail::nonlinearityIndex;

TEST( AdsNli, LinRowBoundPureLinear )
{
    // f(x, y) = 2*x + 3*y, no constant, no higher.
    auto [ x, y ] = TEn< 3, 2 >::variables( 0.0, 0.0 );
    auto f        = 2.0 * x + 3.0 * y;
    EXPECT_DOUBLE_EQ( linRowBound( f ), 5.0 );
}

TEST( AdsNli, JacobianVariationBoundQuadratic )
{
    // f(x, y) = 1 + x + y + 0.5*x*y + 2*x*x
    // ∂f/∂x = 1 + 0.5*y + 4*x   →  nonlinear part variation bound:
    //                              over [-1,1]^2,  |0.5*y + 4*x| ≤ 0.5 + 4 = 4.5
    // ∂f/∂y = 1 + 0.5*x         →  variation bound: 0.5
    auto [ x, y ] = TEn< 3, 2 >::variables( 0.0, 0.0 );
    auto f        = 1.0 + x + y + 0.5 * x * y + 2.0 * x * x;
    const auto bnd = jacobianVariationBound( f );
    EXPECT_DOUBLE_EQ( bnd[ 0 ], 4.5 );
    EXPECT_DOUBLE_EQ( bnd[ 1 ], 0.5 );
}

TEST( AdsNli, NonlinearityIndexLinearIsZero )
{
    using TE = TEn< 3, 2 >;
    auto [ x, y ] = TE::variables( 0.0, 0.0 );
    tax::la::VecNT< 2, TE > F;
    F( 0 ) = 2.0 * x + 3.0 * y;
    F( 1 ) =       x +       y;
    EXPECT_DOUBLE_EQ( nonlinearityIndex( F ), 0.0 );
}

TEST( AdsNli, NonlinearityIndexQuadratic )
{
    using TE = TEn< 3, 2 >;
    auto [ x, y ] = TE::variables( 0.0, 0.0 );
    tax::la::VecNT< 2, TE > F;
    F( 0 ) = x + 0.5 * x * x;   // lin = 1, var = 1   → NLI_row0 = 1.0
    F( 1 ) = y;                 // lin = 1, var = 0   → NLI_row1 = 0
    EXPECT_DOUBLE_EQ( nonlinearityIndex( F ), 1.0 );
}

TEST( AdsNli, SplitDimPicksDominantAxis )
{
    using TE = TEn< 3, 2 >;
    auto [ x, y ] = TE::variables( 0.0, 0.0 );
    tax::la::VecNT< 2, TE > F;
    F( 0 ) = 3.0 * x * x;       // contributes 6 to dim-0 var
    F( 1 ) = 1.0 * y * y;       // contributes 2 to dim-1 var
    EXPECT_EQ( nliSplitDim( F ), 0 );
}
```

- [ ] **Step 3.2: Register the test**

In `tests/CMakeLists.txt`:

```cmake
tax_add_test(test_ads_nonlinearity_index SOURCES ads/test_nonlinearity_index.cpp)
```

- [ ] **Step 3.3: Run to verify failure**

```bash
cmake --build build --target test_ads_nonlinearity_index 2>&1 | tail -10
```

Expected: `tax/ads/nonlinearity_index.hpp: No such file or directory`.

- [ ] **Step 3.4: Implement the NLI helpers**

`include/tax/ads/nonlinearity_index.hpp`:

```cpp
// include/tax/ads/nonlinearity_index.hpp
//
// LOADS nonlinearity-index math. All helpers live in tax::ads::detail
// and operate on TaylorExpansionT<T, N, M> over the normalized box
// [-1, 1]^M (the basis in which DA states built from a Box live).
//
//   linRowBound(f)            = Σ_{|α|=1} |coeff(α)|
//   jacobianVariationBound(f) = vector v ∈ R^M with
//                                v[j] = Σ_{|α|≥2} |coeff(α)| · α_j
//                                (bounds |∂f/∂x_j(x) - ∂f/∂x_j(0)| over [-1,1]^M)
//   nonlinearityIndex(F)      = max_i ||v_i||_1 / linRowBound(F_i)
//                                (Losacco/Fossà/Armellin 2024)
//   nliSplitDim(F)            = argmax_j Σ_i v_i[j]

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#include <tax/core/taylor_expansion.hpp>
#include <tax/la/types.hpp>
#include <tax/utils/enumeration.hpp>

namespace tax::ads::detail
{

// Sum of |coefficients| at total degree 1.
template < class T, int N, int M >
[[nodiscard]] T linRowBound( const tax::TaylorExpansionT< T, N, M >& f ) noexcept
{
    static_assert( N >= 1, "linRowBound requires N >= 1" );
    T acc{ 0 };
    for ( int j = 0; j < M; ++j )
    {
        std::array< int, M > alpha{};
        alpha[ static_cast< std::size_t >( j ) ] = 1;
        acc += std::abs( f.coeff( tax::flatIndex< N, M >( alpha ) ) );
    }
    return acc;
}

// Per-coordinate j: Σ over total-degree-≥2 monomials of |coeff| * α_j.
template < class T, int N, int M >
[[nodiscard]] std::array< T, M > jacobianVariationBound(
    const tax::TaylorExpansionT< T, N, M >& f ) noexcept
{
    std::array< T, M > bound{};
    constexpr std::size_t Ncoef = tax::numMonomials( N, M );
    for ( std::size_t k = 0; k < Ncoef; ++k )
    {
        const auto alpha = tax::unflatIndex< N, M >( k );
        int total = 0;
        for ( int j = 0; j < M; ++j )
            total += alpha[ static_cast< std::size_t >( j ) ];
        if ( total < 2 ) continue;
        const T mag = std::abs( f.coeff( k ) );
        for ( int j = 0; j < M; ++j )
        {
            const int aj = alpha[ static_cast< std::size_t >( j ) ];
            bound[ static_cast< std::size_t >( j ) ] += mag * T( aj );
        }
    }
    return bound;
}

// LOADS nonlinearity index over a Vec of TE rows.
template < class T, int N, int M, int D >
[[nodiscard]] double nonlinearityIndex(
    const Eigen::Matrix< tax::TaylorExpansionT< T, N, M >, D, 1 >& f )
{
    double best = 0.0;
    for ( Eigen::Index i = 0; i < f.size(); ++i )
    {
        const auto& row = f( i );
        const auto  var = jacobianVariationBound( row );
        T var_l1{ 0 };
        for ( int j = 0; j < M; ++j )
            var_l1 += var[ static_cast< std::size_t >( j ) ];
        const T lin = linRowBound( row );
        if ( lin <= T{ 0 } )
        {
            // Pure constant row: variation/0 → treat as infinite if any
            // nonlinear mass exists, otherwise 0.
            if ( var_l1 > T{ 0 } )
                best = std::numeric_limits< double >::infinity();
            continue;
        }
        const double ratio = static_cast< double >( var_l1 ) / static_cast< double >( lin );
        if ( ratio > best ) best = ratio;
    }
    return best;
}

// Split dimension: argmax over j of Σ_i v_i[j].
template < class T, int N, int M, int D >
[[nodiscard]] int nliSplitDim(
    const Eigen::Matrix< tax::TaylorExpansionT< T, N, M >, D, 1 >& f )
{
    std::array< T, M > totals{};
    for ( Eigen::Index i = 0; i < f.size(); ++i )
    {
        const auto v = jacobianVariationBound( f( i ) );
        for ( int j = 0; j < M; ++j )
            totals[ static_cast< std::size_t >( j ) ] +=
                v[ static_cast< std::size_t >( j ) ];
    }
    int   best_j   = 0;
    T     best_val = totals[ 0 ];
    for ( int j = 1; j < M; ++j )
    {
        if ( totals[ static_cast< std::size_t >( j ) ] > best_val )
        {
            best_val = totals[ static_cast< std::size_t >( j ) ];
            best_j   = j;
        }
    }
    return best_j;
}

}  // namespace tax::ads::detail
```

- [ ] **Step 3.5: Verify `tax::flatIndex<N, M>` and `tax::unflatIndex<N, M>` exist**

```bash
grep -n "constexpr.*flatIndex\|constexpr.*unflatIndex" include/tax/utils/enumeration.hpp | head -10
```

If `unflatIndex` is spelled differently (e.g., `multiIndex`, `unflat`, `unflattenIndex`), update the implementation to match. The expected behavior is: take a `std::size_t` flat coefficient index and return `std::array<int, M>` of exponents.

- [ ] **Step 3.6: clang-format and run the tests**

```bash
clang-format -i include/tax/ads/nonlinearity_index.hpp
cmake --build build --target test_ads_nonlinearity_index -j
ctest --test-dir build -R test_ads_nonlinearity_index --output-on-failure
```

Expected: all 5 test cases pass.

- [ ] **Step 3.7: Commit**

```bash
git add include/tax/ads/nonlinearity_index.hpp \
        tests/ads/test_nonlinearity_index.cpp tests/CMakeLists.txt
git commit -m "$(cat <<'EOF'
ads: implement nonlinearity-index math (LOADS) in detail namespace

linRowBound — sum of linear-monomial magnitudes.
jacobianVariationBound — per-coordinate L1 bound on ∂f/∂x_j over [-1,1]^M
  for the degree-≥2 part of f.
nonlinearityIndex — max-over-rows of variation/linear ratio.
nliSplitDim — pick the coordinate whose nonlinear contribution dominates.

Ports the math from the prototype on add-verner-integrators-vEgRF;
control flow stays unimported.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `criteria.hpp` — `SplitCriterion` concept + two criteria

**Files:**
- Create: `include/tax/ads/criteria.hpp`
- Create: `tests/ads/test_criteria.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 4.1: Write the failing test file**

`tests/ads/test_criteria.cpp`:

```cpp
// tests/ads/test_criteria.cpp
//
// TruncationCriterion + NliCriterion — shouldSplit / splitDim decisions
// on crafted DA-valued states.

#include <gtest/gtest.h>

#include <tax/ads/criteria.hpp>
#include <tax/la/types.hpp>
#include <tax/tax.hpp>

using tax::TEn;
using tax::ads::NliCriterion;
using tax::ads::TruncationCriterion;

TEST( AdsCriteria, TruncationDoesNotSplitBelowTol )
{
    // f = x + y → degree-N (=3) coefficients all zero → no split.
    using TE = TEn< 3, 2 >;
    auto [ x, y ] = TE::variables( 0.0, 0.0 );
    tax::la::VecNT< 2, TE > F;
    F( 0 ) = x + y;
    F( 1 ) = x - y;
    TruncationCriterion crit{ /*tol=*/1e-8 };
    EXPECT_FALSE( crit.shouldSplit( F, /*depth=*/0 ) );
}

TEST( AdsCriteria, TruncationSplitsAboveTol )
{
    using TE = TEn< 3, 2 >;
    auto [ x, y ] = TE::variables( 0.0, 0.0 );
    tax::la::VecNT< 1, TE > F;
    F( 0 ) = 1.0 * x * x * x + 0.5 * y * y * y;     // degree-3 mass = 1.5
    TruncationCriterion crit{ /*tol=*/1e-3 };
    EXPECT_TRUE( crit.shouldSplit( F, 0 ) );
}

TEST( AdsCriteria, TruncationRespectsMaxDepth )
{
    using TE = TEn< 3, 2 >;
    auto [ x, y ] = TE::variables( 0.0, 0.0 );
    tax::la::VecNT< 1, TE > F;
    F( 0 ) = 1.0 * x * x * x;
    TruncationCriterion crit{ /*tol=*/1e-12, /*maxDepth=*/3 };
    EXPECT_TRUE(  crit.shouldSplit( F, 2 ) );
    EXPECT_FALSE( crit.shouldSplit( F, 3 ) );    // at the cap → don't split
}

TEST( AdsCriteria, NliBelowTolNoSplit )
{
    using TE = TEn< 3, 2 >;
    auto [ x, y ] = TE::variables( 0.0, 0.0 );
    tax::la::VecNT< 2, TE > F;
    F( 0 ) = x + y;
    F( 1 ) = x - y;
    NliCriterion crit{ /*tol=*/0.1 };
    EXPECT_FALSE( crit.shouldSplit( F, 0 ) );
}

TEST( AdsCriteria, NliAboveTolSplitsAtDominantDim )
{
    using TE = TEn< 3, 2 >;
    auto [ x, y ] = TE::variables( 0.0, 0.0 );
    tax::la::VecNT< 2, TE > F;
    F( 0 ) = x + 0.5 * x * x;       // big nonlinearity on dim 0
    F( 1 ) = y;
    NliCriterion crit{ /*tol=*/0.1 };
    EXPECT_TRUE( crit.shouldSplit( F, 0 ) );
    EXPECT_EQ(   crit.splitDim( F ), 0 );
}
```

- [ ] **Step 4.2: Register the test**

```cmake
tax_add_test(test_ads_criteria SOURCES ads/test_criteria.cpp)
```

- [ ] **Step 4.3: Run to verify failure**

```bash
cmake --build build --target test_ads_criteria 2>&1 | tail -10
```

Expected: `tax/ads/criteria.hpp: No such file or directory`.

- [ ] **Step 4.4: Implement criteria**

`include/tax/ads/criteria.hpp`:

```cpp
// include/tax/ads/criteria.hpp
//
// SplitCriterion concept + two implementations.
//
//   TruncationCriterion — Wittig 2015. Sum the |coefficient| values
//     at total degree N (the truncation degree of the TE). If that
//     mass exceeds `tol`, split along the coordinate contributing the
//     largest portion of it.
//
//   NliCriterion        — Losacco/Fossà/Armellin 2024 (LOADS). Use
//     the nonlinearity index built from the Jacobian variation bound;
//     split along the coordinate whose nonlinear contribution dominates.
//
// Both criteria honour a `maxDepth` cap: shouldSplit() returns false
// once depth >= maxDepth, regardless of state magnitude. This lets the
// driver leave deeply split leaves to integrate to the final time
// instead of looping forever on hopelessly nonlinear regions.

#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>

#include <tax/ads/nonlinearity_index.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/la/types.hpp>
#include <tax/utils/enumeration.hpp>

namespace tax::ads
{

template < class C, class State >
concept SplitCriterion = requires( C c, const State& x, int depth )
{
    { c.shouldSplit( x, depth ) } -> std::convertible_to< bool >;
    { c.splitDim( x )           } -> std::convertible_to< int >;
};

struct TruncationCriterion
{
    double tol      = 1e-6;
    int    maxDepth = 30;

    template < class T, int N, int M, int D >
    [[nodiscard]] bool shouldSplit(
        const Eigen::Matrix< tax::TaylorExpansionT< T, N, M >, D, 1 >& f,
        int depth ) const
    {
        if ( depth >= maxDepth ) return false;
        return totalTopDegreeMass( f ) > T{ tol };
    }

    template < class T, int N, int M, int D >
    [[nodiscard]] int splitDim(
        const Eigen::Matrix< tax::TaylorExpansionT< T, N, M >, D, 1 >& f ) const
    {
        // Coordinate j with the largest sum_{|α|=N, α_j>0} |coeff(α)| · α_j.
        std::array< T, M > totals{};
        constexpr std::size_t Ncoef = tax::numMonomials( N, M );
        for ( Eigen::Index i = 0; i < f.size(); ++i )
        {
            const auto& row = f( i );
            for ( std::size_t k = 0; k < Ncoef; ++k )
            {
                const auto alpha = tax::unflatIndex< N, M >( k );
                int total = 0;
                for ( int j = 0; j < M; ++j )
                    total += alpha[ static_cast< std::size_t >( j ) ];
                if ( total != N ) continue;
                const T mag = std::abs( row.coeff( k ) );
                for ( int j = 0; j < M; ++j )
                {
                    const int aj = alpha[ static_cast< std::size_t >( j ) ];
                    totals[ static_cast< std::size_t >( j ) ] += mag * T( aj );
                }
            }
        }
        int best = 0;
        T   bestVal = totals[ 0 ];
        for ( int j = 1; j < M; ++j )
        {
            if ( totals[ static_cast< std::size_t >( j ) ] > bestVal )
            {
                bestVal = totals[ static_cast< std::size_t >( j ) ];
                best    = j;
            }
        }
        return best;
    }

private:
    template < class T, int N, int M, int D >
    static T totalTopDegreeMass(
        const Eigen::Matrix< tax::TaylorExpansionT< T, N, M >, D, 1 >& f )
    {
        T acc{ 0 };
        constexpr std::size_t Ncoef = tax::numMonomials( N, M );
        for ( Eigen::Index i = 0; i < f.size(); ++i )
        {
            const auto& row = f( i );
            for ( std::size_t k = 0; k < Ncoef; ++k )
            {
                const auto alpha = tax::unflatIndex< N, M >( k );
                int total = 0;
                for ( int j = 0; j < M; ++j )
                    total += alpha[ static_cast< std::size_t >( j ) ];
                if ( total == N ) acc += std::abs( row.coeff( k ) );
            }
        }
        return acc;
    }
};

struct NliCriterion
{
    double tol      = 0.1;
    int    maxDepth = 30;

    template < class T, int N, int M, int D >
    [[nodiscard]] bool shouldSplit(
        const Eigen::Matrix< tax::TaylorExpansionT< T, N, M >, D, 1 >& f,
        int depth ) const
    {
        if ( depth >= maxDepth ) return false;
        return tax::ads::detail::nonlinearityIndex( f ) > tol;
    }

    template < class T, int N, int M, int D >
    [[nodiscard]] int splitDim(
        const Eigen::Matrix< tax::TaylorExpansionT< T, N, M >, D, 1 >& f ) const
    {
        return tax::ads::detail::nliSplitDim( f );
    }
};

}  // namespace tax::ads
```

- [ ] **Step 4.5: clang-format and run the tests**

```bash
clang-format -i include/tax/ads/criteria.hpp
cmake --build build --target test_ads_criteria -j
ctest --test-dir build -R test_ads_criteria --output-on-failure
```

Expected: all 5 test cases pass.

- [ ] **Step 4.6: Commit**

```bash
git add include/tax/ads/criteria.hpp tests/ads/test_criteria.cpp tests/CMakeLists.txt
git commit -m "$(cat <<'EOF'
ads: add SplitCriterion concept + TruncationCriterion + NliCriterion

Both criteria operate on Vec<TEn<P,M>>-valued DA states and honour a
maxDepth cap. TruncationCriterion (Wittig) sums |coeff| at total
degree P; NliCriterion (LOADS) uses the nonlinearity index from
ads::detail. Each criterion's splitDim picks the coordinate whose
contribution to the splitting metric dominates.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `split_event.hpp` — interop with `tax::ode::Event`

**Files:**
- Create: `include/tax/ads/split_event.hpp`
- Create: `tests/ads/test_split_event.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 5.1: Write the failing test file**

`tests/ads/test_split_event.cpp`:

```cpp
// tests/ads/test_split_event.cpp
//
// SplitTrigger / SplitAction round-trip: fabricate a StepperCtx with a
// known DA state, route through the trigger and action, verify the
// SplitRequest is populated and ControlFlow::Terminate is returned.

#include <gtest/gtest.h>

#include <tax/ads/criteria.hpp>
#include <tax/ads/split_event.hpp>
#include <tax/la/types.hpp>
#include <tax/ode/event.hpp>
#include <tax/ode/solution.hpp>
#include <tax/tax.hpp>

using tax::TEn;
using tax::ads::SplitAction;
using tax::ads::SplitRequest;
using tax::ads::SplitTrigger;
using tax::ads::TruncationCriterion;
using tax::ode::ControlFlow;
using tax::ode::EventStorage;
using tax::ode::TriggerContext;

namespace
{
using TE        = TEn< 3, 2 >;
using State     = tax::la::VecNT< 2, TE >;
using DenseData = State;   // any matrix-of-TE works as fake DenseData

State makeQuadraticState()
{
    auto [ x, y ] = TE::variables( 0.0, 0.0 );
    State F;
    F( 0 ) = x + 0.5 * x * x;
    F( 1 ) = y;
    return F;
}
}  // namespace

TEST( AdsSplitEvent, TriggerFiresAtBoundaryWhenCriterionAgrees )
{
    auto    state  = makeQuadraticState();
    auto    state2 = state;
    DenseData dense = state;
    TriggerContext< State, double, DenseData > ctx{ state, 0.0, state2, 0.1, dense };

    TruncationCriterion crit{ /*tol=*/1e-3 };
    auto                trig = SplitTrigger( crit, /*depth=*/0 );
    auto                tau  = trig( ctx );
    ASSERT_TRUE( tau.has_value() );
    EXPECT_DOUBLE_EQ( *tau, 0.1 );
}

TEST( AdsSplitEvent, TriggerSilentBelowTol )
{
    auto      state = makeQuadraticState();
    auto      s2    = state;
    DenseData dense = state;
    TriggerContext< State, double, DenseData > ctx{ state, 0.0, s2, 0.1, dense };

    TruncationCriterion crit{ /*tol=*/1.0 };   // nothing exceeds it
    auto                trig = SplitTrigger( crit, /*depth=*/0 );
    auto                tau  = trig( ctx );
    EXPECT_FALSE( tau.has_value() );
}

TEST( AdsSplitEvent, ActionRecordsRequestAndTerminates )
{
    auto      state = makeQuadraticState();
    auto      s2    = state;
    DenseData dense = state;
    TriggerContext< State, double, DenseData > ctx{ state, 0.0, s2, 0.1, dense };

    TruncationCriterion crit{ /*tol=*/1e-3 };
    SplitRequest< double > req{};
    auto                  act = SplitAction( crit, &req );
    EventStorage< State, double > storage{ /*events=*/nullptr };
    auto cf = act( ctx, /*tau=*/0.1, storage );
    EXPECT_EQ( cf, ControlFlow::Terminate );
    EXPECT_TRUE( req.fired );
    EXPECT_EQ( req.dim, 0 );
    EXPECT_DOUBLE_EQ( req.t, 0.1 );
}
```

- [ ] **Step 5.2: Register the test**

```cmake
tax_add_test(test_ads_split_event SOURCES ads/test_split_event.cpp)
```

- [ ] **Step 5.3: Run to verify failure**

```bash
cmake --build build --target test_ads_split_event 2>&1 | tail -10
```

Expected: `tax/ads/split_event.hpp: No such file or directory`.

- [ ] **Step 5.4: Implement `split_event.hpp`**

`include/tax/ads/split_event.hpp`:

```cpp
// include/tax/ads/split_event.hpp
//
// Interop with tax::ode::Event. A SplitRequest is the side-channel
// from the driver-owned split event back to the BFS driver: the
// trigger asks the criterion whether the current DA state demands a
// split, and the action records {fired, dim, t} into the request and
// returns ControlFlow::Terminate. The integrator then truncates the
// solution at the step boundary; the driver consumes req to decide
// split-vs-done.
//
// Note: tax::ode does not need to know about ADS. Trigger and Action
// satisfy tax::ode's std::function-erased signatures.

#pragma once

#include <optional>

#include <tax/ode/event.hpp>

namespace tax::ads
{

template < class T >
struct SplitRequest
{
    bool fired = false;
    int  dim   = -1;
    T    t     = T{ 0 };
};

// Trigger: fires at the step boundary iff criterion.shouldSplit is true.
// `depth` is captured by value at construction (the driver builds a new
// event per BFS leaf, so the leaf's depth is known).
template < class Criterion >
auto SplitTrigger( Criterion crit, int depth )
{
    return [ crit = std::move( crit ), depth ]< class Ctx >( const Ctx& ctx )
               -> std::optional< typename Ctx::T_type >
    {
        if ( crit.shouldSplit( ctx.x_new, depth ) )
            return ctx.h_used;
        return std::nullopt;
    };
}

// Action: write {fired, dim, t} into *out and Terminate. The caller
// keeps `out` alive for the duration of Integrator::integrate().
template < class Criterion, class T >
auto SplitAction( Criterion crit, SplitRequest< T >* out )
{
    return [ crit = std::move( crit ), out ]< class Ctx, class TT, class Storage >(
               const Ctx& ctx, TT tau, Storage& ) -> tax::ode::ControlFlow
    {
        out->fired = true;
        out->dim   = crit.splitDim( ctx.x_new );
        out->t     = ctx.t_old + static_cast< T >( tau );
        return tax::ode::ControlFlow::Terminate;
    };
}

}  // namespace tax::ads
```

- [ ] **Step 5.5: clang-format and run the tests**

```bash
clang-format -i include/tax/ads/split_event.hpp
cmake --build build --target test_ads_split_event -j
ctest --test-dir build -R test_ads_split_event --output-on-failure
```

Expected: all 3 test cases pass.

- [ ] **Step 5.6: Commit**

```bash
git add include/tax/ads/split_event.hpp tests/ads/test_split_event.cpp tests/CMakeLists.txt
git commit -m "$(cat <<'EOF'
ads: SplitTrigger + SplitAction factories interop with ode::Event

SplitRequest<T> is the side-channel from the ADS-owned event back to
the BFS driver. SplitTrigger asks the criterion at every accepted-step
boundary; SplitAction populates the request and returns
ControlFlow::Terminate. tax::ode itself is not touched.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `da_state.hpp` — DA construction & split

**Files:**
- Create: `include/tax/ads/da_state.hpp`
- Create: `tests/ads/test_da_state.cpp`
- Modify: `tests/CMakeLists.txt`

**Math reminder:**

`create(box, x0)` builds `x_i = x0_i + halfWidth_i · ξ_i` where `ξ_i` is the i-th identity variable of `TEn<P, M>`. So the state is a TE-valued vector whose evaluation at `ξ ∈ [-1,1]^M` reproduces `x0 + halfWidth ⊙ ξ`, i.e., the box's denormalize on `ξ`.

`split(state, parent_box, dim)` re-identifies the state's normalized variable on the two halves. After a split along dim, the parent's `ξ_dim` is replaced by:
- left  half: `(-0.5 + 0.5·ξ'_dim)`  →  composed via `state.compose(...)` or via direct coefficient transformation.
- right half: `(+0.5 + 0.5·ξ'_dim)`

For each coefficient `c · ξ^α` in the original state, the substitution yields the binomial expansion of `(c · ∏_{j≠dim} ξ_j^{α_j} · (s + 0.5·ξ'_dim)^{α_dim})` where `s = ±0.5`. This is straightforward but tedious; implementation can either use `TaylorExpansionT::compose` if available, or build the substitution polynomial manually using `tax::TEn<P, M>::variable` for the new `ξ'` axis.

Check what composition primitives `tax::TaylorExpansionT` exposes before writing the implementation:

```bash
grep -n "compose\|substitute\|eval" include/tax/core/taylor_expansion.hpp | head -20
```

Use whichever primitive exists; the test below pins the *result*, not the *mechanism*.

- [ ] **Step 6.1: Write the failing test file**

`tests/ads/test_da_state.cpp`:

```cpp
// tests/ads/test_da_state.cpp
//
// create identity property; split round-trip.

#include <gtest/gtest.h>

#include <tax/ads/box.hpp>
#include <tax/ads/da_state.hpp>
#include <tax/la/types.hpp>
#include <tax/tax.hpp>

#include "../testUtils.hpp"   // ExpectCoeffsNear, kTol — relative to tests/

using tax::TEn;
using tax::ads::Box;
using tax::ads::create;
using tax::ads::split;

namespace
{
constexpr int P = 4;
constexpr int M = 2;
using TE      = TEn< P, M >;
using State   = tax::la::VecNT< 2, TE >;
}  // namespace

TEST( AdsDaState, MakeMapsZeroDeviationToCenter )
{
    Box< double, M > box{ { 1.0, 2.0 }, { 0.5, 0.25 } };
    tax::la::VecNT< 2, double > x0{};
    x0( 0 ) = 1.0; x0( 1 ) = 2.0;
    State F = create< P, M >( box, x0 );

    // At ξ = 0, each row should evaluate to x0_i.
    EXPECT_DOUBLE_EQ( F( 0 ).coeff( 0 ), 1.0 );
    EXPECT_DOUBLE_EQ( F( 1 ).coeff( 0 ), 2.0 );

    // First-order coefficient w.r.t. ξ_i should be halfWidth_i on row i.
    std::array< int, M > alpha_x{ 1, 0 };
    std::array< int, M > alpha_y{ 0, 1 };
    EXPECT_DOUBLE_EQ( F( 0 ).coeff( tax::flatIndex< P, M >( alpha_x ) ), 0.5 );
    EXPECT_DOUBLE_EQ( F( 1 ).coeff( tax::flatIndex< P, M >( alpha_y ) ), 0.25 );
}

TEST( AdsDaState, SplitRoundTripPreservesValue )
{
    // Build a state on parent box, split it, evaluate both halves at the
    // child-local boundary that corresponds to a known interior point of
    // the parent, and verify agreement.
    Box< double, M > parent{ { 0.0, 0.0 }, { 1.0, 1.0 } };
    tax::la::VecNT< 2, double > x0{}; x0( 0 ) = 0.0; x0( 1 ) = 0.0;
    State F = create< P, M >( parent, x0 );

    auto pr   = split( F, parent, /*dim=*/0 );
    auto& FL = pr.first;
    auto& FR = pr.second;

    // Parent's ξ_0 = -0.5 corresponds to left child's ξ'_0 = 0.
    // Parent's ξ_0 = +0.5 corresponds to right child's ξ'_0 = 0.
    // Sample row 0 (which is x_0(ξ) = 0 + 1·ξ_0 = ξ_0).
    auto evalRow = []( const TE& f, std::array< double, M > xi ) {
        double acc = 0.0;
        constexpr std::size_t Nc = tax::numMonomials( P, M );
        for ( std::size_t k = 0; k < Nc; ++k ) {
            auto alpha = tax::unflatIndex< P, M >( k );
            double term = f.coeff( k );
            for ( int j = 0; j < M; ++j )
                for ( int p = 0; p < alpha[ j ]; ++p ) term *= xi[ j ];
            acc += term;
        }
        return acc;
    };

    // Parent at ξ_0=-0.5, ξ_1=0 should equal left child at ξ'_0=0, ξ'_1=0.
    EXPECT_NEAR( evalRow( FL( 0 ), { 0.0, 0.0 } ), -0.5, 1e-12 );
    // Parent at ξ_0=+0.5, ξ_1=0 should equal right child at ξ'_0=0, ξ'_1=0.
    EXPECT_NEAR( evalRow( FR( 0 ), { 0.0, 0.0 } ),  0.5, 1e-12 );
    // Row 1 (= ξ_1) is unaffected by a dim-0 split.
    EXPECT_NEAR( evalRow( FL( 1 ), { 0.0, 0.7 } ), 0.7, 1e-12 );
    EXPECT_NEAR( evalRow( FR( 1 ), { 0.0, 0.7 } ), 0.7, 1e-12 );
}
```

- [ ] **Step 6.2: Register the test**

```cmake
tax_add_test(test_ads_da_state SOURCES ads/test_da_state.cpp)
```

- [ ] **Step 6.3: Run to verify failure**

```bash
cmake --build build --target test_ads_da_state 2>&1 | tail -10
```

Expected: `tax/ads/da_state.hpp: No such file or directory`.

- [ ] **Step 6.4: Inspect composition primitives**

```bash
grep -n "compose\|substitute\|eval\b" include/tax/core/taylor_expansion.hpp | head -25
```

You will use one of:
- a `compose` / `substitute` member if exposed; or
- direct coefficient-space substitution via the multi-index walk used in `nonlinearity_index.hpp`.

Pick the simplest path that makes the test pass. If neither composition primitive exists, fall back to the manual binomial-substitution loop sketched below.

- [ ] **Step 6.5: Implement `da_state.hpp`**

`include/tax/ads/da_state.hpp`:

```cpp
// include/tax/ads/da_state.hpp
//
// create  — build a DA-valued state vector from a Box and a
//                center initial condition. Each component is
//                x0_i + halfWidth_i · ξ_i, so the state spans the box
//                as ξ runs over [-1, 1]^M.
//
// split — re-identify a DA state's domain on the two halves of
//                its parent box along `dim`. The substitution is
//                  ξ_dim  →  -0.5 + 0.5 · ξ'_dim   (left half)
//                  ξ_dim  →  +0.5 + 0.5 · ξ'_dim   (right half)
//                so the children carry polynomials in their own local
//                [-1, 1] coordinates.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

#include <tax/ads/box.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/la/types.hpp>
#include <tax/utils/enumeration.hpp>

namespace tax::ads
{

namespace detail
{
// Binomial coefficient C(n, k) at compile time / runtime.
[[nodiscard]] inline constexpr double binom( int n, int k ) noexcept
{
    if ( k < 0 || k > n ) return 0.0;
    double r = 1.0;
    for ( int i = 0; i < k; ++i ) r = r * double( n - i ) / double( i + 1 );
    return r;
}

// Substitute ξ_dim → shift + 0.5 · ξ_dim in a single TE.
// Implementation: build the result coefficient-by-coefficient via the
// binomial expansion of (shift + 0.5 · ξ_dim)^α_dim.
template < class T, int N, int M >
[[nodiscard]] tax::TaylorExpansionT< T, N, M > substituteAxis(
    const tax::TaylorExpansionT< T, N, M >& f, int dim, T shift ) noexcept
{
    tax::TaylorExpansionT< T, N, M > out{};
    constexpr std::size_t Ncoef = tax::numMonomials( N, M );
    for ( std::size_t k = 0; k < Ncoef; ++k )
    {
        const auto alpha = tax::unflatIndex< N, M >( k );
        const T    cval  = f.coeff( k );
        if ( cval == T{ 0 } ) continue;
        const int aDim = alpha[ static_cast< std::size_t >( dim ) ];
        // Distribute (shift + 0.5·ξ_dim)^aDim into ξ_dim^j terms.
        for ( int j = 0; j <= aDim; ++j )
        {
            std::array< int, M > beta = alpha;
            beta[ static_cast< std::size_t >( dim ) ] = j;
            int total = 0;
            for ( int q = 0; q < M; ++q )
                total += beta[ static_cast< std::size_t >( q ) ];
            if ( total > N ) continue;
            const T coef = cval
                         * T( detail::binom( aDim, j ) )
                         * std::pow( shift, T( aDim - j ) )
                         * std::pow( T( 0.5 ), T( j ) );
            out.coeffRef( tax::flatIndex< N, M >( beta ) ) += coef;
        }
    }
    return out;
}
}  // namespace detail

template < int N, int M, class T, int D >
[[nodiscard]] Eigen::Matrix< tax::TaylorExpansionT< T, N, M >, D, 1 > create(
    const Box< T, M >& box, const Eigen::Matrix< T, D, 1 >& x0 )
{
    Eigen::Matrix< tax::TaylorExpansionT< T, N, M >, D, 1 > out;
    if constexpr ( D == Eigen::Dynamic )
        out.resize( x0.size() );
    for ( Eigen::Index i = 0; i < x0.size(); ++i )
    {
        const Eigen::Index vi = ( i < M ) ? i : 0;   // see note below
        // Build x0_i + halfWidth_{vi} · ξ_{vi}. The typical use case is
        // D == M and the i-th component spans the i-th axis; the conditional
        // above handles the degenerate case where the state is wider than
        // the box by attaching extra rows to ξ_0 (no spread).
        tax::TaylorExpansionT< T, N, M > comp{};
        comp.coeffRef( 0 ) = x0( i );
        if ( i < M )
        {
            std::array< int, M > alpha{};
            alpha[ static_cast< std::size_t >( i ) ] = 1;
            comp.coeffRef( tax::flatIndex< N, M >( alpha ) ) =
                box.halfWidth[ static_cast< std::size_t >( i ) ];
        }
        out( i ) = std::move( comp );
    }
    return out;
}

template < class T, int N, int M, int D >
[[nodiscard]] std::pair<
    Eigen::Matrix< tax::TaylorExpansionT< T, N, M >, D, 1 >,
    Eigen::Matrix< tax::TaylorExpansionT< T, N, M >, D, 1 > >
split(
    const Eigen::Matrix< tax::TaylorExpansionT< T, N, M >, D, 1 >& state,
    const Box< T, M >& /*parent_box*/,  // unused — split is in normalized coords
    int dim )
{
    using State = Eigen::Matrix< tax::TaylorExpansionT< T, N, M >, D, 1 >;
    State L{ state.size() };
    State R{ state.size() };
    for ( Eigen::Index i = 0; i < state.size(); ++i )
    {
        L( i ) = detail::substituteAxis( state( i ), dim, T{ -0.5 } );
        R( i ) = detail::substituteAxis( state( i ), dim, T{  0.5 } );
    }
    return { std::move( L ), std::move( R ) };
}

}  // namespace tax::ads
```

**Caveat:** if `TaylorExpansionT` exposes `coeffRef(std::size_t)` differently (e.g., `operator[]`, `set_coeff`), substitute the correct API. Check with:

```bash
grep -n "coeff\b\|coeffRef\b\|operator\[\]" include/tax/core/taylor_expansion.hpp | head -20
```

- [ ] **Step 6.6: clang-format and run the tests**

```bash
clang-format -i include/tax/ads/da_state.hpp
cmake --build build --target test_ads_da_state -j
ctest --test-dir build -R test_ads_da_state --output-on-failure
```

Expected: both test cases pass.

- [ ] **Step 6.7: Commit**

```bash
git add include/tax/ads/da_state.hpp tests/ads/test_da_state.cpp tests/CMakeLists.txt
git commit -m "$(cat <<'EOF'
ads: create + split helpers

create lifts a (Box, x0) pair to a DA-valued state vector
x_i = x0_i + halfWidth_i · ξ_i. split re-identifies a state's
normalized coordinate ξ_dim on the two halves of its parent box via
the binomial substitution ξ_dim → ±0.5 + 0.5·ξ'_dim.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `driver.hpp` — `AdsDriver<Stepper, Criterion>`

**Files:**
- Create: `include/tax/ads/driver.hpp`
- Create: `tests/ads/test_driver.cpp`
- Modify: `tests/CMakeLists.txt`

**Reference problem (from the spec):**

Mildly nonlinear oscillator
```
dx/dt = v
dv/dt = -x - 0.1 * x^3
```
IC box centered at `(1, 0)` with half-width `(0.5, 0.5)`. Propagate to `t = 2π` with `P = 6`, `M = 2`, `D = 2`. Sample 5 points in the IC box, integrate each scalar IC with a tight scalar Verner89 reference, and compare against `tree.leaf(ic).payload` evaluated at the displacement. Expect agreement to within `1e-3`.

- [ ] **Step 7.1: Write the failing test file**

`tests/ads/test_driver.cpp`:

```cpp
// tests/ads/test_driver.cpp
//
// End-to-end ADS propagation on a mildly nonlinear oscillator,
// verified against high-accuracy scalar reference propagations sampled
// from the initial-condition box.

#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include <tax/ads/box.hpp>
#include <tax/ads/criteria.hpp>
#include <tax/ads/driver.hpp>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <tax/tax.hpp>

using tax::TEn;
using tax::ads::AdsDriver;
using tax::ads::Box;
using tax::ads::TruncationCriterion;
using tax::ode::IntegratorConfig;
using tax::ode::Verner89Stepper;

namespace
{
constexpr int P = 6;
constexpr int M = 2;
constexpr int D = 2;

using TE       = TEn< P, M >;
using DAState  = tax::la::VecNT< D, TE >;
using ScState  = tax::la::VecNT< D, double >;
using Stepper  = Verner89Stepper< DAState >;
using ScStep   = Verner89Stepper< ScState >;

// f(x, v) = (v, -x - 0.1 x^3). Templated on State so it accepts both
// scalar and DA-valued vectors.
auto rhs()
{
    return []( const auto& x, double ) {
        using S = std::decay_t< decltype( x ) >;
        S out{ x.size() };
        out( 0 ) =  x( 1 );
        out( 1 ) = -x( 0 ) - 0.1 * x( 0 ) * x( 0 ) * x( 0 );
        return out;
    };
}

// Reference: scalar Verner89 to t = 2π at the given IC.
ScState scalarReference( ScState x0, double t1 )
{
    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    tax::ode::Verner89< ScState > integ{ rhs(), cfg };
    auto sol = integ.integrate( x0, 0.0, t1 );
    return sol.x.back();
}
}  // namespace

TEST( AdsDriver, MildlyNonlinearOscillatorMatchesReference )
{
    const double t1 = 2.0 * M_PI;

    Box< double, M > ic_box{ { 1.0, 0.0 }, { 0.5, 0.5 } };
    tax::la::VecNT< D, double > center; center << 1.0, 0.0;

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    AdsDriver< Stepper, TruncationCriterion > driver{
        TruncationCriterion{ /*tol=*/1e-4, /*maxDepth=*/8 },
        cfg
    };

    auto tree = driver.run( rhs(), ic_box, center, /*t0=*/0.0, t1 );

    // Sanity: at least the root, and every done leaf is a valid index.
    EXPECT_GE( tree.done().size(), 1u );

    // Sample 5 points (including the center) and compare to reference.
    const std::array< std::array< double, M >, 5 > samples{ {
        { 0.0,  0.0 },
        { 0.3, -0.2 },
        { -0.4,  0.4 },
        { 0.5,  0.5 },
        { -0.5, -0.5 },
    } };

    for ( const auto& xi : samples )
    {
        auto idx = tree.leaf( xi );
        ASSERT_TRUE( idx.has_value() );
        const auto& leaf = tree.leaf( *idx );

        // Evaluate the leaf's DA payload at xi (relative to the box).
        // Convert global xi to leaf-local ξ' ∈ [-1, 1]^M:
        std::array< double, M > xi_local{};
        for ( int j = 0; j < M; ++j )
            xi_local[ j ] = ( xi[ j ] - leaf.box.center[ j ] )
                            / leaf.box.halfWidth[ j ];

        // Evaluate each component of the DA flow map.
        ScState x_predicted;
        for ( int row = 0; row < D; ++row )
        {
            double acc = 0.0;
            constexpr std::size_t Nc = tax::numMonomials( P, M );
            for ( std::size_t k = 0; k < Nc; ++k )
            {
                auto   alpha = tax::unflatIndex< P, M >( k );
                double term  = leaf.payload( row ).coeff( k );
                for ( int j = 0; j < M; ++j )
                    for ( int p = 0; p < alpha[ j ]; ++p )
                        term *= xi_local[ j ];
                acc += term;
            }
            x_predicted( row ) = acc;
        }

        // Reference: integrate the scalar IC (center + xi) to t1.
        ScState ic;
        ic( 0 ) = 1.0 + xi[ 0 ];
        ic( 1 ) = 0.0 + xi[ 1 ];
        const ScState x_ref = scalarReference( ic, t1 );

        EXPECT_NEAR( x_predicted( 0 ), x_ref( 0 ), 1e-3 )
            << "row 0 mismatch at xi = (" << xi[ 0 ] << ", " << xi[ 1 ] << ")";
        EXPECT_NEAR( x_predicted( 1 ), x_ref( 1 ), 1e-3 )
            << "row 1 mismatch at xi = (" << xi[ 0 ] << ", " << xi[ 1 ] << ")";
    }
}

TEST( AdsDriver, ExtraUserEventIsForwarded )
{
    // Register an EveryStep event that increments a counter; verify it
    // fires alongside the split event without interfering.
    const double t1 = 0.5;
    Box< double, M > ic_box{ { 1.0, 0.0 }, { 0.05, 0.05 } };   // small box → no split
    tax::la::VecNT< D, double > center; center << 1.0, 0.0;

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-10;

    int step_counter = 0;
    using ExtraEvt   = std::vector< tax::ode::Event< Stepper > >;
    ExtraEvt extras;
    extras.emplace_back(
        tax::ode::EveryStep(),
        [&]< class Ctx, class T, class Storage >(
            const Ctx&, T, Storage& ) -> tax::ode::ControlFlow
        {
            ++step_counter;
            return tax::ode::ControlFlow::Continue;
        } );

    AdsDriver< Stepper, TruncationCriterion > driver{
        TruncationCriterion{ /*tol=*/1.0, /*maxDepth=*/0 },   // never split
        cfg,
        std::move( extras )
    };

    auto tree = driver.run( rhs(), ic_box, center, 0.0, t1 );
    EXPECT_EQ( tree.done().size(), 1u );
    EXPECT_GT( step_counter,       0 );
}
```

- [ ] **Step 7.2: Register the test**

```cmake
tax_add_test(test_ads_driver SOURCES ads/test_driver.cpp)
```

- [ ] **Step 7.3: Run to verify failure**

```bash
cmake --build build --target test_ads_driver 2>&1 | tail -10
```

Expected: `tax/ads/driver.hpp: No such file or directory`.

- [ ] **Step 7.4: Implement `AdsDriver`**

`include/tax/ads/driver.hpp`:

```cpp
// include/tax/ads/driver.hpp
//
// AdsDriver<Stepper, Criterion> — BFS driver around tax::ode::Integrator.
// For each leaf in the work queue, the driver runs the integrator with
// a (SplitTrigger, SplitAction) pair appended to any user-supplied
// events. If the split event fires, the leaf is replaced by two
// children with the parent's DA state re-identified on each half.
// Otherwise the leaf is marked done with the propagated DA flow map
// stored as its payload.

#pragma once

#include <type_traits>
#include <utility>
#include <vector>

#include <tax/ads/box.hpp>
#include <tax/ads/da_state.hpp>
#include <tax/ads/split_event.hpp>
#include <tax/ads/tree.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/la/types.hpp>
#include <tax/ode/event.hpp>
#include <tax/ode/integrator.hpp>

namespace tax::ads
{

namespace detail
{
// Extract the multivariate degree M of a TE-typed scalar.
template < class S >
struct AdsDimOf
{
    static constexpr int value = S::vars_v;
};

// Extract the Taylor truncation order N from a TE-typed scalar.
template < class S >
struct AdsOrderOf
{
    static constexpr int value = S::order_v;
};
}  // namespace detail

template < class Stepper, class Criterion >
class AdsDriver
{
public:
    using State    = typename Stepper::State;
    using T        = typename Stepper::T;
    using Cfg      = typename Stepper::Config;
    using ExtraEvt = std::vector< tax::ode::Event< Stepper > >;

    using TE                  = typename State::Scalar;
    static constexpr int N    = detail::AdsOrderOf< TE >::value;
    static constexpr int M    = detail::AdsDimOf< TE >::value;
    static constexpr int D    = State::RowsAtCompileTime;

    using Tree = AdsTree< State, M, T >;
    using BoxT = Box< T, M >;

    AdsDriver( Criterion crit, Cfg cfg, ExtraEvt extras = {} )
        : crit_( std::move( crit ) ),
          cfg_( std::move( cfg ) ),
          extras_( std::move( extras ) )
    {
    }

    template < class F >
    Tree run( F&& rhs, const BoxT& ic_box,
              const Eigen::Matrix< T, D, 1 >& ic_center, T t0, T t1 )
    {
        Tree tree;
        State root_state = create< N, M >( ic_box, ic_center );
        tree.init( ic_box, std::move( root_state ), t0 );

        while ( !tree.empty() )
        {
            const int idx = tree.popFront();
            auto&     l   = tree.leaf( idx );

            // Build the split event for this leaf at its current depth.
            SplitRequest< T > req;
            using Evt = tax::ode::Event< Stepper >;
            auto events = extras_;
            events.emplace_back(
                SplitTrigger( crit_, l.depth ),
                SplitAction(  crit_, &req ) );

            tax::ode::Integrator< Stepper, std::decay_t< F > > integ{
                rhs, cfg_, std::move( events ) };
            auto sol = integ.integrate( l.payload, l.tEntry, t1 );

            if ( req.fired )
            {
                auto pr_box   = l.box.split( req.dim );
                auto pr_state = split( sol.x.back(), l.box, req.dim );
                tree.split( idx, req.dim, l.box.center[ req.dim ],
                            std::move( pr_state.first  ),
                            std::move( pr_state.second ),
                            req.t );
            }
            else
            {
                l.payload = std::move( sol.x.back() );
                tree.finalize( idx );
            }
        }
        return tree;
    }

private:
    Criterion crit_;
    Cfg       cfg_;
    ExtraEvt  extras_;
};

}  // namespace tax::ads
```

- [ ] **Step 7.5: clang-format and run the tests**

```bash
clang-format -i include/tax/ads/driver.hpp
cmake --build build --target test_ads_driver -j
ctest --test-dir build -R test_ads_driver --output-on-failure
```

Expected: both test cases pass. The first is the end-to-end propagation
match (tolerance 1e-3); the second is the user-event forwarding check.

- [ ] **Step 7.6: Diagnose failures**

If the propagation test fails:
- Print `tree.done().size()` and `leaf.depth` for each done leaf; you should see at least one split if the tolerance is set correctly.
- Print one sample reference vs. predicted side by side to gauge whether the gap is numerical or structural.
- Verify `split` agreement: pick the leaf that contains your sample point; the leaf's `payload` evaluated at the leaf-local `ξ` should match the parent's `payload` evaluated at the parent-local `ξ` (which is the substitution we test in `test_da_state.cpp`).

If the user-event test fails:
- Confirm the event ordering inside `Integrator::integrate` (alphabetical by index, tie-broken by trigger τ) doesn't terminate before EveryStep runs. The `EveryStep` action returns `Continue`, so it shouldn't terminate even when it fires before a split.

- [ ] **Step 7.7: Commit**

```bash
git add include/tax/ads/driver.hpp tests/ads/test_driver.cpp tests/CMakeLists.txt
git commit -m "$(cat <<'EOF'
ads: AdsDriver<Stepper, Criterion> — BFS driver over ode::Integrator

For each leaf in the work queue, run the integrator with a
(SplitTrigger, SplitAction) pair appended to any user-supplied
events. On a fired split: halve the box, re-identify the DA state on
each half via split, push two children. Otherwise: mark done
with the final-time DA flow map as the payload.

Tested end-to-end on a mildly nonlinear oscillator against a tight
scalar Verner89 reference; user-event forwarding tested with an
EveryStep counter.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `merge.hpp` — bottom-up merger

**Files:**
- Create: `include/tax/ads/merge.hpp`
- Create: `tests/ads/test_merge.cpp`
- Modify: `tests/CMakeLists.txt`

**Algorithm:** In each pass, scan `tree.done()` for sibling pairs (both done, shared `parentIdx`). For each pair, build the candidate merged state on the parent's normalized coordinates and check `!crit.shouldSplit(merged, parent_depth)`. If the candidate passes, `merge`.

The candidate-merged state is constructed by undoing the substitutions in `split`. Concretely: the left child carries `f_L(ξ') = f_parent(-0.5 + 0.5 ξ')`; we recover `f_parent(ξ) = f_L(2 ξ + 1)` on `ξ ∈ [-1, 0]` and `f_R(2 ξ - 1)` on `ξ ∈ [0, 1]`. If `f_L` and `f_R` agree (as polynomials of fixed truncation order they cannot exactly agree on the whole interval unless the original parent was already low-order), we synthesise a merged polynomial by Chebyshev-style averaging: pick the higher-quality of the two child polynomials (lower truncation residual) and re-substitute it onto the parent. The criterion-check then validates whether this merged polynomial is a faithful representation of the parent.

For step-1 scope, use a simple variant: `merged = substituteAxis(f_L, dim, -1.0 + 2.0·ξ_dim_substitution …)` is not directly supported; instead, pick the child whose payload has the smaller top-degree residual mass and substitute its inverse-shift. The merge predicate then re-evaluates the criterion on this candidate.

If the candidate passes the criterion *and* numerically agrees with the other child within `tol`, accept the merge.

- [ ] **Step 8.1: Write the failing test file**

`tests/ads/test_merge.cpp`:

```cpp
// tests/ads/test_merge.cpp
//
// Build an over-split tree with two sibling leaves whose payloads are
// the two halves of a low-order polynomial; verify merge collapses
// them into the parent and that MergeStats reports it.

#include <gtest/gtest.h>

#include <tax/ads/box.hpp>
#include <tax/ads/criteria.hpp>
#include <tax/ads/da_state.hpp>
#include <tax/ads/merge.hpp>
#include <tax/ads/tree.hpp>
#include <tax/la/types.hpp>
#include <tax/tax.hpp>

using tax::TEn;
using tax::ads::AdsTree;
using tax::ads::Box;
using tax::ads::create;
using tax::ads::merge;
using tax::ads::MergeStats;
using tax::ads::split;
using tax::ads::TruncationCriterion;

namespace
{
constexpr int P = 4;
constexpr int M = 2;
constexpr int D = 2;
using TE      = TEn< P, M >;
using State   = tax::la::VecNT< D, TE >;
using Tree    = AdsTree< State, M, double >;
}  // namespace

TEST( AdsMerge, CollapsesUnnecessarySplit )
{
    // Parent state: F(ξ) = (ξ_0, ξ_1) → a low-order polynomial that
    // *can* be represented exactly on the parent box.
    Box< double, M > parent{ { 0.0, 0.0 }, { 1.0, 1.0 } };
    tax::la::VecNT< D, double > x0{}; x0( 0 ) = 0.0; x0( 1 ) = 0.0;
    State F = create< P, M >( parent, x0 );

    Tree tree;
    const int root = tree.init( parent, F, /*t=*/0.0 );
    tree.popFront();
    auto child_states = split( F, parent, /*dim=*/0 );
    auto pr = tree.split( root, /*dim=*/0, /*splitValue=*/0.0,
                          std::move( child_states.first  ),
                          std::move( child_states.second ),
                          /*tEntry=*/0.0 );
    tree.popFront();
    tree.finalize( pr.first );
    tree.popFront();
    tree.finalize( pr.second );

    TruncationCriterion crit{ /*tol=*/1e-10 };
    const auto stats = merge< /*Stepper=*/void >( tree, crit );

    EXPECT_GE( stats.merges, 1 );
    EXPECT_GE( stats.passes, 1 );
    EXPECT_FALSE( tree.leaf( root ).retired );
    EXPECT_TRUE(  tree.leaf( root ).done );
    EXPECT_TRUE(  tree.leaf( pr.first  ).retired );
    EXPECT_TRUE(  tree.leaf( pr.second ).retired );
}

TEST( AdsMerge, RejectsWhenChildrenDoNotMatch )
{
    Box< double, M > parent{ { 0.0, 0.0 }, { 1.0, 1.0 } };
    tax::la::VecNT< D, double > x0{}; x0( 0 ) = 0.0; x0( 1 ) = 0.0;
    State F = create< P, M >( parent, x0 );

    Tree tree;
    const int root = tree.init( parent, F, 0.0 );
    tree.popFront();
    auto cs = split( F, parent, /*dim=*/0 );
    // Perturb the right child to break agreement.
    cs.second( 0 ).coeffRef( 0 ) += 1.0;
    auto pr = tree.split( root, 0, 0.0, std::move( cs.first  ),
                          std::move( cs.second ), 0.0 );
    tree.popFront(); tree.finalize( pr.first  );
    tree.popFront(); tree.finalize( pr.second );

    TruncationCriterion crit{ /*tol=*/1e-10 };
    const auto stats = merge< void >( tree, crit );
    EXPECT_EQ( stats.merges,   0 );
    EXPECT_GE( stats.rejected, 1 );
    EXPECT_FALSE( tree.leaf( root ).done );      // not revived
    EXPECT_TRUE(  tree.leaf( root ).retired );   // still retired
}
```

- [ ] **Step 8.2: Register the test**

```cmake
tax_add_test(test_ads_merge SOURCES ads/test_merge.cpp)
```

- [ ] **Step 8.3: Run to verify failure**

```bash
cmake --build build --target test_ads_merge 2>&1 | tail -10
```

Expected: `tax/ads/merge.hpp: No such file or directory`.

- [ ] **Step 8.4: Implement `merge`**

`include/tax/ads/merge.hpp`:

```cpp
// include/tax/ads/merge.hpp
//
// Bottom-up merge: scan done leaves for sibling pairs whose payloads
// agree on the parent's coordinates (within criterion tolerance), and
// collapse each accepted pair back onto the parent.
//
// The candidate merged payload is reconstructed from the *left* child:
// since the left child carries f_L(ξ') = f_parent(-0.5 + 0.5·ξ'),
// the inverse is f_parent(ξ) = f_L(2·ξ + 1). We use the same
// substituteAxis primitive from da_state.hpp to apply this.
//
// Then we check (a) that the criterion does not flag the merged
// payload, and (b) that the right child's payload, similarly
// inverse-mapped, agrees with the left's. If both, accept.

#pragma once

#include <cmath>
#include <cstddef>

#include <tax/ads/criteria.hpp>
#include <tax/ads/da_state.hpp>
#include <tax/ads/tree.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/la/types.hpp>
#include <tax/utils/enumeration.hpp>

namespace tax::ads
{

struct MergeStats
{
    int passes   = 0;
    int merges   = 0;
    int rejected = 0;
};

namespace detail
{
// Substitute ξ_dim → shift + 2·ξ_dim (inverse of the ±0.5 + 0.5·ξ_dim
// used by split). shift = +1 for the left half, -1 for the right.
template < class T, int N, int M >
[[nodiscard]] tax::TaylorExpansionT< T, N, M > inverseSubstituteAxis(
    const tax::TaylorExpansionT< T, N, M >& f, int dim, T shift ) noexcept
{
    tax::TaylorExpansionT< T, N, M > out{};
    constexpr std::size_t Ncoef = tax::numMonomials( N, M );
    for ( std::size_t k = 0; k < Ncoef; ++k )
    {
        const auto alpha = tax::unflatIndex< N, M >( k );
        const T    cval  = f.coeff( k );
        if ( cval == T{ 0 } ) continue;
        const int aDim = alpha[ static_cast< std::size_t >( dim ) ];
        for ( int j = 0; j <= aDim; ++j )
        {
            std::array< int, M > beta = alpha;
            beta[ static_cast< std::size_t >( dim ) ] = j;
            int total = 0;
            for ( int q = 0; q < M; ++q )
                total += beta[ static_cast< std::size_t >( q ) ];
            if ( total > N ) continue;
            const T coef = cval
                         * T( tax::ads::detail::binom( aDim, j ) )
                         * std::pow( shift, T( aDim - j ) )
                         * std::pow( T( 2.0 ), T( j ) );
            out.coeffRef( tax::flatIndex< N, M >( beta ) ) += coef;
        }
    }
    return out;
}

template < class T, int N, int M, int D >
[[nodiscard]] T maxCoeffDiff(
    const Eigen::Matrix< tax::TaylorExpansionT< T, N, M >, D, 1 >& a,
    const Eigen::Matrix< tax::TaylorExpansionT< T, N, M >, D, 1 >& b ) noexcept
{
    T worst{ 0 };
    constexpr std::size_t Ncoef = tax::numMonomials( N, M );
    for ( Eigen::Index i = 0; i < a.size(); ++i )
    {
        for ( std::size_t k = 0; k < Ncoef; ++k )
        {
            const T d = std::abs( a( i ).coeff( k ) - b( i ).coeff( k ) );
            if ( d > worst ) worst = d;
        }
    }
    return worst;
}
}  // namespace detail

// Template parameter `Stepper` is reserved for future criteria that
// need stepper-aware inspection; currently unused but kept on the
// signature for forward compatibility.
template < class /*Stepper*/, class Payload, int M, class T,
           class Criterion >
MergeStats merge( AdsTree< Payload, M, T >& tree, Criterion crit )
{
    using State = Payload;
    MergeStats stats{};

    while ( true )
    {
        ++stats.passes;
        bool changed = false;

        // Snapshot done indices because merge mutates the list.
        std::vector< int > snapshot( tree.done().begin(), tree.done().end() );
        for ( std::size_t i = 0; i + 1 < snapshot.size(); ++i )
        {
            const int li = snapshot[ i ];
            // Skip if already retired by an earlier merge in this pass.
            if ( tree.leaf( li ).retired ) continue;
            const int sib = tree.leaf( li ).siblingIdx;
            if ( sib < 0 || !tree.leaf( sib ).done
                 || tree.leaf( sib ).retired )
                continue;

            const int dim = tree.leaf( li ).splitDim;

            // Reconstruct candidate parent payloads from each child.
            State fromL{ tree.leaf( li  ).payload.size() };
            State fromR{ tree.leaf( sib ).payload.size() };
            for ( Eigen::Index r = 0; r < fromL.size(); ++r )
            {
                fromL( r ) = detail::inverseSubstituteAxis(
                    tree.leaf( li  ).payload( r ), dim, T{  1 } );
                fromR( r ) = detail::inverseSubstituteAxis(
                    tree.leaf( sib ).payload( r ), dim, T{ -1 } );
            }

            const T diff = detail::maxCoeffDiff( fromL, fromR );
            const int parent_depth =
                tree.leaf( tree.leaf( li ).parentIdx ).depth;
            const bool flagged = crit.shouldSplit( fromL, parent_depth );

            if ( !flagged && diff <= T( crit.tol ) )
            {
                // Pick whichever child's reconstruction is "smaller in
                // top-degree mass"; here they agree within tol, so use L.
                const int leftIdx  = li;
                const int rightIdx = sib;
                tree.merge( leftIdx, rightIdx, std::move( fromL ) );
                ++stats.merges;
                changed = true;
            }
            else
            {
                ++stats.rejected;
            }
        }
        if ( !changed ) break;
    }
    return stats;
}

}  // namespace tax::ads
```

- [ ] **Step 8.5: clang-format and run the tests**

```bash
clang-format -i include/tax/ads/merge.hpp
cmake --build build --target test_ads_merge -j
ctest --test-dir build -R test_ads_merge --output-on-failure
```

Expected: both test cases pass.

- [ ] **Step 8.6: Commit**

```bash
git add include/tax/ads/merge.hpp tests/ads/test_merge.cpp tests/CMakeLists.txt
git commit -m "$(cat <<'EOF'
ads: merge() — bottom-up sibling collapse with MergeStats

Reconstruct a candidate parent payload from each child via the inverse
of split's substitution (ξ_dim → ±1 + 2·ξ_dim). If the two
reconstructions agree within criterion tol and the criterion does not
flag the merged payload, merge via tree. Repeats until no
further merges happen in a pass.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: umbrella header + CLAUDE.md + final smoke

**Files:**
- Create: `include/tax/ads.hpp`
- Modify: `CLAUDE.md` (replace the existing `ads/` section)

- [ ] **Step 9.1: Write the umbrella**

`include/tax/ads.hpp`:

```cpp
// include/tax/ads.hpp
//
// Umbrella header for the tax::ads module. Users include only this.

#pragma once

#include <tax/ads/box.hpp>
#include <tax/ads/criteria.hpp>
#include <tax/ads/da_state.hpp>
#include <tax/ads/driver.hpp>
#include <tax/ads/leaf.hpp>
#include <tax/ads/merge.hpp>
#include <tax/ads/nonlinearity_index.hpp>
#include <tax/ads/split_event.hpp>
#include <tax/ads/tree.hpp>
```

- [ ] **Step 9.2: Update CLAUDE.md**

In `CLAUDE.md`, locate the existing **Automatic Domain Splitting (ADS)** section (around lines 396-433 in the current file). Replace it with:

```markdown
## Automatic Domain Splitting (ADS)

Located in `include/tax/ads/`. Implements Wittig 2015's ADS and the LOADS
variant (Losacco/Fossà/Armellin 2024) by composing with the existing
`tax::ode` event infrastructure — `tax::ode` itself is not modified.

### ODE Propagation with ADS

```cpp
#include <tax/ads.hpp>
#include <tax/ode.hpp>

using TE      = tax::TEn< /*N=*/6, /*M=*/2 >;
using State   = tax::la::VecNT< /*D=*/2, TE >;
using Stepper = tax::ode::Verner89Stepper< State >;

auto f = []( const auto& x, double ) {
    using S = std::decay_t< decltype( x ) >;
    S out{ x.size() };
    out( 0 ) =  x( 1 );
    out( 1 ) = -x( 0 ) - 0.1 * x( 0 ) * x( 0 ) * x( 0 );
    return out;
};

tax::ads::Box< double, 2 > ic_box{ { 1.0, 0.0 }, { 0.5, 0.5 } };
tax::la::VecNT< 2, double > center; center << 1.0, 0.0;

tax::ode::IntegratorConfig< double > cfg;
cfg.abstol = cfg.reltol = 1e-12;

tax::ads::AdsDriver< Stepper, tax::ads::TruncationCriterion > driver{
    tax::ads::TruncationCriterion{ /*tol=*/1e-4 },
    cfg
};
auto tree = driver.run( f, ic_box, center, /*t0=*/0.0, /*t1=*/2.0 * M_PI );

for ( int i : tree.done() ) {
    const auto& leaf = tree.leaf( i );
    // leaf.payload — DA-valued flow map at t = 2π on leaf.box
}
```

### LOADS — Nonlinearity-Index Criterion

Same setup, swap `TruncationCriterion` for `NliCriterion`:

```cpp
tax::ads::AdsDriver< Stepper, tax::ads::NliCriterion > driver{
    tax::ads::NliCriterion{ /*tol=*/0.1 },
    cfg
};
```

### Architecture

- **Leaf-only arena tree** (`AdsTree<Payload, M, T>`): single
  `std::vector<Leaf>` with parent / sibling indices on each leaf.
  Splits retire the parent in place; merges revive it.
- **Event interop**: a `(SplitTrigger, SplitAction)` pair is appended
  to the user's event list and passed to `tax::ode::Integrator`. The
  trigger fires at accepted-step boundaries when the criterion says
  split; the action records `{dim, t}` into a `SplitRequest` and
  returns `ControlFlow::Terminate`. The driver consumes the request to
  decide split vs. mark-done.
- **Templated on Stepper**: any `tax::ode::Stepper` (Taylor / Verner78
  / Verner89 / Fehlberg78 / Feagin12 / Feagin14) works, provided the
  state type is `tax::la::VecNT<D, TaylorExpansionT<T, N, M>>` with
  `N >= 1` and `M >= 1`.
- **Boundary-only splits**: triggers fire only at accepted-step
  boundaries (matches Wittig's original ADS).
- **Post-pass merger**: `merge<Stepper>(tree, criterion)` walks
  sibling pairs bottom-up and collapses any pair whose reconstructed
  parent payload satisfies the criterion within `tol`.

### Key files in `ads/`

| File | Purpose |
|------|---------|
| `box.hpp`                 | `Box<T, M>` axis-aligned subdomain |
| `leaf.hpp`                | `Leaf<Payload, M, T>` POD record |
| `tree.hpp`                | `AdsTree<Payload, M, T>` arena + BFS queue |
| `criteria.hpp`            | `SplitCriterion` concept + `TruncationCriterion` + `NliCriterion` |
| `nonlinearity_index.hpp`  | LOADS NLI math (`tax::ads::detail`) |
| `split_event.hpp`         | `SplitRequest`, `SplitTrigger`, `SplitAction` |
| `da_state.hpp`            | `create`, `split` helpers |
| `driver.hpp`              | `AdsDriver<Stepper, Criterion>` |
| `merge.hpp`               | `merge` + `MergeStats` |
```

- [ ] **Step 9.3: Final smoke**

Build everything and run the entire ADS suite:

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure -R test_ads_
```

Expected: 8 test executables, all green.

Also run the existing ode tests to confirm nothing regressed (the ADS module should be purely additive):

```bash
ctest --test-dir build --output-on-failure -R test_ode_
```

Expected: all green.

- [ ] **Step 9.4: Commit**

```bash
git add include/tax/ads.hpp CLAUDE.md
git commit -m "$(cat <<'EOF'
ads: add umbrella include + CLAUDE.md documentation

Users include <tax/ads.hpp>. CLAUDE.md updated to reflect the actual
surface (AdsDriver, leaf-only AdsTree, SplitTrigger/SplitAction
factories) and the architectural choice to compose with tax::ode's
event system rather than duplicate the step loop.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Self-review

1. **Spec coverage:**
   - Box → Task 1.
   - Leaf + AdsTree (leaf-only design) → Task 2.
   - NLI math → Task 3.
   - SplitCriterion concept + TruncationCriterion + NliCriterion → Task 4.
   - SplitRequest + SplitTrigger + SplitAction → Task 5.
   - create + split → Task 6.
   - AdsDriver (templated on Stepper + Criterion, ExtraEvt forwarding, depth cap) → Task 7.
   - merge + MergeStats → Task 8.
   - Umbrella + CLAUDE.md update → Task 9.
   - No modifications to `tax::ode` — confirmed in implementation files.
   - Boundary-only splits — `SplitTrigger` returns `ctx.h_used`, matching spec.
   - Tests mirror the source structure as described in the spec's "Testing" section.

2. **Placeholder scan:** No TBDs, no "implement later," no "similar to Task N." Each step has the actual code or command an engineer needs.

3. **Type consistency:**
   - `SplitRequest<T>` fields `{fired, dim, t}` used consistently across split_event.hpp, driver.hpp, and tests.
   - `SplitTrigger(crit, depth)` and `SplitAction(crit, &req)` signatures match between the implementation and the driver invocation.
   - `AdsTree::split(idx, dim, splitValue, leftPayload, rightPayload, tEntry)` consistent between tree implementation, driver call site, and tree tests.
   - `crit.shouldSplit(state, depth)` and `crit.splitDim(state)` consistent across the concept, both criteria, the trigger, and the action.
   - `Box::split(dim)` (no second argument) returns `pair<Box, Box>`; consistent with usage in driver and tests.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-01-tax-ads-module.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
