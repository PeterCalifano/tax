#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include <tax/ads/ads_tree.hpp>
#include <tax/ads/box.hpp>
#include <tax/ode/integrate_da.hpp>
#include <tax/tte.hpp>
#include <tax/utils/combinatorics.hpp>

namespace tax::ode
{

// =============================================================================
// Internal helpers: ADS splitting criteria
// =============================================================================

namespace detail
{

/// @brief Truncation error of a DA state vector.
///
/// Returns the infinity norm of all degree-P coefficients across every
/// component of the state.  Large values indicate that the polynomial
/// approximation of the flow map is degrading.
template < int P, int D >
[[nodiscard]] double truncationError(
    const Eigen::Matrix< TEn< P, D >, D, 1 >& state ) noexcept
{
    double err = 0.0;
    for ( Eigen::Index i = 0; i < state.size(); ++i )
    {
        const auto& poly = state( i );
        for ( std::size_t j = 0; j < TEn< P, D >::nCoefficients; ++j )
        {
            const auto alpha = tax::detail::unflatIndex< D >( j );
            if ( tax::detail::totalDegree< D >( alpha ) == P )
                err = std::max( err, std::abs( poly[j] ) );
        }
    }
    return err;
}

/// @brief Choose the initial-condition variable that contributes most to the
///        truncation error, following Wittig et al. (2015).
template < int P, int D >
[[nodiscard]] int bestSplitDim(
    const Eigen::Matrix< TEn< P, D >, D, 1 >& state ) noexcept
{
    std::array< double, D > scores{};
    for ( Eigen::Index i = 0; i < state.size(); ++i )
    {
        const auto& poly = state( i );
        for ( std::size_t j = 0; j < TEn< P, D >::nCoefficients; ++j )
        {
            const auto alpha = tax::detail::unflatIndex< D >( j );
            if ( tax::detail::totalDegree< D >( alpha ) == P )
                for ( int k = 0; k < D; ++k )
                    if ( alpha[k] > 0 )
                        scores[k] += std::abs( poly[j] );
        }
    }
    return static_cast< int >(
        std::max_element( scores.begin(), scores.end() ) - scores.begin() );
}

}  // namespace detail

// =============================================================================
// ADS-integrated ODE propagation
// =============================================================================

/**
 * @brief Integrate a vector ODE with Automatic Domain Splitting.
 *
 * The initial-condition domain @p x0_box is propagated from @p t0 to @p tmax.
 * If the DA approximation of the flow map degrades beyond @p ads_tol, the
 * domain is bisected along the variable that contributes most to the
 * truncation error (Wittig et al. 2015), and each half is re-propagated
 * independently.
 *
 * @details The driver processes the work queue in waves: at each iteration it
 * drains every currently-pending leaf, decides which ones need splitting,
 * builds the list of child boxes that have to be propagated, and runs all
 * those `propagateBox` calls **in parallel** with OpenMP when available.  At
 * deep levels the wave contains many independent splits, so the speed-up
 * scales with the available cores.  Tree-modification calls (`markDone`,
 * `split`) are performed serially after each parallel batch.
 *
 * The right-hand side @p f must be safe to invoke concurrently on distinct
 * argument sets; a stateless lambda or a function over `const`-captured data
 * satisfies this.
 *
 * @tparam N  Taylor expansion order in time.
 * @tparam P  DA expansion order in the initial-condition variables.
 * @tparam D  State-space dimension (= number of DA variables).
 * @param f             Right-hand side `f(dx, x, t)` (must be re-entrant).
 * @param x0_box        Initial-condition domain (centre + half-widths).
 * @param t0            Initial time.
 * @param tmax          Final time.
 * @param step_tol      Absolute tolerance for adaptive time stepping.
 * @param ads_tol       Truncation-error tolerance for ADS splitting.
 * @param ads_max_depth Maximum number of recursive bisections from root.
 * @param maxsteps      Maximum integration steps per subdomain.
 * @return ADS tree whose done leaves contain the piecewise flow map.
 */
template < int N, int P, typename F, int D >
[[nodiscard]] AdsTree< FlowMap< P, D > > integrateAds(
    F&& f, const Box< double, D >& x0_box, double t0, double tmax,
    double step_tol, double ads_tol, int ads_max_depth = 30, int maxsteps = 500 )
{
    using FM   = FlowMap< P, D >;
    using Tree = AdsTree< FM >;

    auto evaluate_box = [&]( const Box< double, D >& box ) -> FM {
        return FM{ propagateBox< N, P, D >( f, box, t0, tmax, step_tol, maxsteps ) };
    };

    Tree tree;
    tree.addLeaf( evaluate_box( x0_box ), x0_box );

    std::vector< int > depth( 1, 0 );

    // Per-iteration buffer of pending splits; reused across waves.
    struct PendingSplit
    {
        int             parent_idx;
        int             dim;
        int             parent_depth;
        Box< double, D > lb, rb;
        FM              lt, rt;
    };
    std::vector< PendingSplit > splits;

    while ( !tree.empty() )
    {
        // 1. Drain the current wave from the work queue.
        std::vector< int > wave;
        while ( !tree.empty() ) wave.push_back( tree.pop() );

        // 2. Decide split-or-done for each leaf and stage child boxes.
        splits.clear();
        for ( int idx : wave )
        {
            const auto&  lf  = tree.node( idx ).leaf();
            const double err = detail::truncationError< P, D >( lf.tte.state );
            const int    d   = depth[idx];

            if ( err < ads_tol || d >= ads_max_depth )
            {
                tree.markDone( idx );
            }
            else
            {
                const int dim   = detail::bestSplitDim< P, D >( lf.tte.state );
                auto [lb, rb]   = lf.box.split( dim );
                splits.push_back( { idx, dim, d, lb, rb, FM{}, FM{} } );
            }
        }

        // 3. Propagate every child box in parallel (each wave produces 2N
        //    independent propagateBox calls; OpenMP distributes them).
        const int n2 = static_cast< int >( splits.size() ) * 2;
#pragma omp parallel for schedule( dynamic ) if ( n2 > 1 )
        for ( int i = 0; i < n2; ++i )
        {
            const int  s       = i / 2;
            const bool is_left = ( i & 1 ) == 0;
            const auto& box    = is_left ? splits[s].lb : splits[s].rb;
            FM          result = evaluate_box( box );
            if ( is_left )
                splits[s].lt = std::move( result );
            else
                splits[s].rt = std::move( result );
        }

        // 4. Apply the splits to the tree (serial; tree mutation isn't
        //    thread-safe and the cost here is dominated by step 3 anyway).
        for ( auto& s : splits )
        {
            auto [li, ri] =
                tree.split( s.parent_idx, s.dim, std::move( s.lt ), std::move( s.rt ) );
            if ( static_cast< int >( depth.size() ) <= ri ) depth.resize( ri + 1, 0 );
            depth[li] = s.parent_depth + 1;
            depth[ri] = s.parent_depth + 1;
        }
    }

    return tree;
}

}  // namespace tax::ode
