#pragma once

#include <tax/ads/ads_tree.hpp>
#include <tax/ads/nonlinearity_index.hpp>
#include <tax/tte.hpp>
#include <tax/utils/combinatorics.hpp>

#include <array>
#include <tuple>
#include <utility>
#include <vector>

namespace tax
{

/**
 * @file
 * @brief Low-order Automatic Domain Splitting driven by a nonlinearity index.
 *
 * Implements the algorithm of Losacco, Fossà, Armellin (J. Guid. Control
 * Dyn. 2024; arXiv:2303.05791): instead of estimating the truncation
 * error from the highest-degree coefficients (Wittig et al. 2015), a
 * @ref nonlinearityIndex computed from the polynomial bound of the
 * Jacobian variation is used to decide when to split.  The method is
 * tailored to *low* Taylor orders (N = 2 in the paper) and produces
 * fewer, larger subdomains while still controlling local linearity.
 */

namespace detail
{

/**
 * @brief Build the M DA variables of a leaf box and evaluate @p func on them.
 *
 * Creates one variable per box dimension as
 *   x_k = center[k] + halfWidth[k] · δ_k,  δ_k ∈ [-1, 1]
 * and materialises the resulting expression in a single pass into the
 * returned TTE.  Variables are kept alive on the stack frame so the
 * expression's by-reference leaves are valid until evaluation completes.
 */
template < int N, int M, typename F >
[[nodiscard]] TEn< N, M > evaluateOnBox( F& func, const Box< double, M >& box )
{
    using TTE = TEn< N, M >;

    auto make_var = [&]( std::size_t k ) -> TTE {
        typename TTE::Data c{};
        c[0] = box.center[k];
        if constexpr ( N >= 1 )
        {
            MultiIndex< M > ek{};
            ek[k] = 1;
            c[tax::detail::flatIndex< M >( ek )] = box.halfWidth[k];
        }
        return TTE{ c };
    };

    auto xs = [&]< std::size_t... I >( std::index_sequence< I... > ) {
        return std::tuple< decltype( make_var( I ) )... >{ make_var( I )... };
    }( std::make_index_sequence< M >{} );

    typename TTE::Data c{};
    std::apply( func, xs ).evalTo( c );
    return TTE{ c };
}

}  // namespace detail

/**
 * @brief Low-order ADS runner using a nonlinearity index as splitting criterion.
 *
 * The driving loop mirrors @ref AdsRunner: pop a leaf, decide split-or-done,
 * bisect and re-enqueue if needed.  Differences from the classical Wittig
 * algorithm:
 *
 *   - Splitting trigger is @ref nonlinearityIndex ν > @p nliTol, not the
 *     degree-N truncation norm.  ν is a normalised measure of Jacobian
 *     variation over the leaf's normalised box δ ∈ [-1,1]^M; ν → 0 means
 *     the polynomial is locally close to affine.
 *   - Split dimension is the input with the largest @ref nliPerVariable
 *     contribution to the Jacobian-variation bound.
 *
 * The method is intended for low Taylor orders (N = 2 captures the bulk
 * of the nonlinearity information used by the index); higher orders are
 * supported but the criterion only inspects degree-≥-2 coefficients.
 *
 * @tparam N  DA order.  Must satisfy N ≥ 2 (a degree-1 polynomial has no
 *            measurable Jacobian variation and the criterion would
 *            always trigger 0 splits).
 * @tparam M  Number of variables.
 * @tparam F  Callable `(const TEn<N,M>&)... → Expr`.  Arguments **must**
 *            be taken by const reference; expression nodes hold leaves
 *            by reference and a by-value copy would dangle.
 */
template < int N, int M, typename F >
class LowOrderAdsRunner
{
   public:
    static_assert( N >= 2,
                   "LowOrderAdsRunner requires N >= 2 (the nonlinearity "
                   "index is defined from degree-≥-2 coefficients)" );

    using TTE  = TEn< N, M >;
    using Tree = AdsTree< TTE >;

    /**
     * @param func     Function to approximate (takes M DA variables).
     * @param nliTol   Nonlinearity-index threshold above which a leaf is
     *                 split.  Typical values ~1e-3 (paper).
     * @param maxDepth Maximum number of bisections from root to any leaf.
     */
    LowOrderAdsRunner( F func, double nliTol, int maxDepth = 30 )
        : func_( std::move( func ) ), nliTol_( nliTol ), maxDepth_( maxDepth )
    {
    }

    [[nodiscard]] double nliTol()   const noexcept { return nliTol_; }
    [[nodiscard]] int    maxDepth() const noexcept { return maxDepth_; }

    /**
     * @brief Evaluate @c func_ on @p box and return the resulting TTE.
     *
     * Exposed so post-processing passes (e.g. @ref mergeAds) can re-evaluate
     * the function on a candidate parent box without rebuilding the variables.
     */
    [[nodiscard]] TTE evaluate( const Box< double, M >& box )
    {
        return detail::evaluateOnBox< N, M, F >( func_, box );
    }

    Tree run( Box< double, M > initial_box )
    {
        Tree tree;
        tree.addLeaf( evaluate( initial_box ), initial_box );

        std::vector< int > depth( 1, 0 );

        while ( !tree.empty() )
        {
            const int    idx = tree.pop();
            const double nu  = nonlinearityIndex< double, N, M >(
                tree.node( idx ).leaf().tte );
            const int    d   = depth[idx];

            if ( nu <= nliTol_ || d >= maxDepth_ )
            {
                tree.markDone( idx );
            }
            else
            {
                const auto& box = tree.node( idx ).leaf().box;
                const int   dim = nliSplitDim< double, N, M >(
                    tree.node( idx ).leaf().tte );

                auto [lb, rb] = box.split( dim );
                auto lt       = evaluate( lb );
                auto rt       = evaluate( rb );

                auto [li, ri] = tree.split( idx, dim, std::move( lt ), std::move( rt ) );

                if ( int( depth.size() ) <= ri ) depth.resize( ri + 1, 0 );
                depth[li] = d + 1;
                depth[ri] = d + 1;
            }
        }
        return tree;
    }

   private:
    F      func_;
    double nliTol_;
    int    maxDepth_;
};

/// Convenience factory — deduces N, M, F from arguments.
template < int N, int M, typename F >
[[nodiscard]] LowOrderAdsRunner< N, M, F >
makeLowOrderAdsRunner( F func, double nliTol, int maxDepth = 30 )
{
    return LowOrderAdsRunner< N, M, F >( std::move( func ), nliTol, maxDepth );
}

}  // namespace tax
