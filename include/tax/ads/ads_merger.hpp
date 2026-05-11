#pragma once

#include <tax/ads/ads_tree.hpp>
#include <tax/ads/low_order_ads_runner.hpp>
#include <tax/ads/nonlinearity_index.hpp>
#include <tax/storage/tte_static.hpp>

#include <utility>

namespace tax
{

/**
 * @file
 * @brief Bottom-up merging of ADS leaves driven by the nonlinearity index.
 *
 * Implements the merging stage of Losacco, Fossà, Armellin (J. Guid.
 * Control Dyn. 2024; arXiv:2303.05791): adjacent sibling leaves that
 * jointly satisfy the linearity constraint are recombined into a
 * single, larger leaf.  This counter-balances the natural growth of
 * the ADS tree, recovering compact representations when the dynamics
 * become less nonlinear over time (or when the initial domain was
 * conservatively over-split).
 *
 * The implementation is generic: any callable that produced the tree
 * (or that can re-evaluate it on a candidate parent box) can drive the
 * merger.  The function is re-evaluated on each candidate parent box
 * and the merge is accepted only if the resulting TTE has
 * nonlinearity index ≤ @p nliTol.
 */

/**
 * @brief One bottom-up merging pass over the tree.
 *
 * Walks every Internal node whose children are both *done* Leaves.
 * For each candidate, re-evaluates @p func on the parent box; if the
 * polynomial satisfies the nonlinearity threshold, the children are
 * discarded and the internal node becomes a done leaf.  A second
 * mergeable pair may form upstream as a result; the pass therefore
 * iterates until a full sweep produces no further merges.
 *
 * @tparam N      DA order of the leaf TTEs.
 * @tparam M      Number of variables.
 * @tparam F      Callable used to re-evaluate the function on a box;
 *                same convention as @ref LowOrderAdsRunner.
 *
 * @param tree    ADS tree to mutate in place.
 * @param func    Function approximated by the tree's leaves.
 * @param nliTol  Acceptance threshold on the merged leaf's
 *                nonlinearity index.
 *
 * @return Number of successful merges across all passes.
 */
template < int N, int M, typename F >
int mergeAds( AdsTree< TEn< N, M > >& tree, F func, double nliTol )
{
    using TTE = TEn< N, M >;

    int total_merges = 0;

    for ( ;; )
    {
        bool progressed = false;

        // Snapshot the candidate set: indices of Internal nodes whose
        // children are both done leaves.  Computing this on every sweep
        // keeps the algorithm robust to arena growth from re-evaluation.
        std::vector< int > candidates;
        candidates.reserve( std::size_t( tree.numNodes() ) );
        for ( int i = 0; i < tree.numNodes(); ++i )
        {
            const auto& n = tree.node( i );
            if ( !n.isInternal() ) continue;
            const auto& in = n.internal();
            const auto& ln = tree.node( in.leftIdx );
            const auto& rn = tree.node( in.rightIdx );
            if ( ln.isLeaf() && rn.isLeaf() && ln.leaf().done && rn.leaf().done )
                candidates.push_back( i );
        }

        // Visit candidates deepest-first so a successful merge can chain
        // upward in the same sweep.  We approximate "deepest" with arena
        // index: children are appended after their parents, so larger
        // arena indices correspond to deeper nodes.
        std::sort( candidates.begin(), candidates.end(), std::greater< int >{} );

        for ( int idx : candidates )
        {
            // Skip if the candidate was consumed by an earlier merge
            // in this sweep — e.g. a child of the current candidate
            // was itself a merge target and the candidate is no longer
            // an Internal node.
            if ( !tree.node( idx ).isInternal() ) continue;
            const auto& in = tree.node( idx ).internal();
            const auto& ln = tree.node( in.leftIdx );
            const auto& rn = tree.node( in.rightIdx );
            if ( !ln.isLeaf() || !rn.isLeaf() ) continue;
            if ( !ln.leaf().done || !rn.leaf().done ) continue;

            // Reconstruct the parent box (same logic as AdsTree::merge).
            Box< double, M > parent_box           = ln.leaf().box;
            parent_box.center[in.splitDim]          = in.splitValue;
            parent_box.halfWidth[in.splitDim]       =
                parent_box.halfWidth[in.splitDim] * 2.0;

            // Re-evaluate the function on the parent box and check the
            // nonlinearity threshold.
            TTE merged = detail::evaluateOnBox< N, M, F >( func, parent_box );
            const double nu = nonlinearityIndex< double, N, M >( merged );
            if ( nu > nliTol ) continue;

            tree.merge( idx, std::move( merged ), /*markDone=*/true );
            ++total_merges;
            progressed = true;
        }

        if ( !progressed ) break;
    }

    return total_merges;
}

}  // namespace tax
