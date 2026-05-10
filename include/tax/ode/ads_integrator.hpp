#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <tax/ads/ads_tree.hpp>
#include <tax/ads/box.hpp>
#include <tax/ode/da_integrator.hpp>
#include <tax/tte.hpp>
#include <tax/utils/combinatorics.hpp>

namespace tax::ode
{

// =============================================================================
// Configuration
// =============================================================================

/**
 * @brief Configuration for the ADS integrator.
 */
struct AdsConfig
{
    double step_tol  = 1e-14;  ///< Adaptive-step-size tolerance for `propagate`.
    double ads_tol   = 1e-3;   ///< Truncation-error tolerance triggering a split.
    int    max_depth = 30;     ///< Maximum number of recursive bisections from the root.
    int    max_steps = 500;    ///< Maximum integration steps per subdomain.
};

namespace detail
{

inline void validate( const AdsConfig& cfg )
{
    if ( !( cfg.step_tol > 0.0 ) )
        throw std::invalid_argument( "AdsConfig: step_tol must be > 0" );
    if ( !( cfg.ads_tol > 0.0 ) )
        throw std::invalid_argument( "AdsConfig: ads_tol must be > 0" );
    if ( cfg.max_depth < 0 )
        throw std::invalid_argument( "AdsConfig: max_depth must be >= 0" );
    if ( cfg.max_steps <= 0 )
        throw std::invalid_argument( "AdsConfig: max_steps must be > 0" );
}

/// @brief Truncation error of a DA state vector (max degree-P coefficient).
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

/// @brief Pick the IC variable that contributes most to the degree-P
///        truncation error (Wittig et al. 2015).
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
                    if ( alpha[k] > 0 ) scores[k] += std::abs( poly[j] );
        }
    }
    return static_cast< int >(
        std::max_element( scores.begin(), scores.end() ) - scores.begin() );
}

}  // namespace detail

// =============================================================================
// SplitEvent — payload for the on_split callback
// =============================================================================

/**
 * @brief Information about a single ADS bisection.
 *
 * Passed to @ref AdsIntegrator::on_split each time a leaf is bisected.  The
 * callback is invoked from the serial tree-update phase, after the parallel
 * propagation of the two children, so it observes a consistent tree state.
 */
template < int P, int D >
struct SplitEvent
{
    int                parent_idx;       ///< Arena index of the parent (now Internal).
    int                left_idx;         ///< Arena index of the left  child leaf.
    int                right_idx;        ///< Arena index of the right child leaf.
    int                split_dim;        ///< IC variable on which the split was made.
    int                parent_depth;     ///< Depth of the parent (root = 0).
    double             truncation_error; ///< Truncation error that triggered the split.
    Box< double, D >   parent_box;       ///< Parent's IC box.
    Box< double, D >   left_box;         ///< Left  child's IC box.
    Box< double, D >   right_box;        ///< Right child's IC box.
};

// =============================================================================
// AdsIntegrator class
// =============================================================================

/**
 * @brief Adaptive Domain Splitting integrator for vector ODEs in DA form.
 *
 * Drives `propagateBox`-style DA integrations and adaptively bisects the
 * initial-condition box wherever the multivariate-Taylor flow truncation
 * exceeds @ref AdsConfig::ads_tol (Wittig et al. 2015).
 *
 * @details Internally the work queue is processed in waves.  At each wave the
 * driver:
 *   1. drains every currently pending leaf;
 *   2. decides split-or-done on each (cheap, serial);
 *   3. propagates every child box of every split in parallel via OpenMP — the
 *      natural parallelism of ADS, since sub-boxes are fully independent;
 *   4. applies the splits to the tree serially.
 *
 *  The user-supplied @p f must be safe to invoke concurrently (a stateless
 *  lambda, or one over `const`-captured data, satisfies this).  The
 *  @ref on_split callback is invoked serially from step 4 and may freely
 *  read shared state.
 *
 * @tparam N  Taylor expansion order in time.
 * @tparam P  DA expansion order in the initial-condition variables.
 * @tparam D  State-space dimension (= number of DA variables).
 */
template < int N, int P, int D >
class AdsIntegrator
{
public:
    using Config      = AdsConfig;
    using FlowMapT    = FlowMap< P, D >;
    using TreeT       = AdsTree< FlowMapT >;
    using SplitEventT = SplitEvent< P, D >;
    using OnSplitFn   = std::function< void( const SplitEventT& ) >;

    /**
     * @brief Construct with the given configuration.
     * @throws std::invalid_argument on invalid configuration.
     */
    explicit AdsIntegrator( Config cfg = {} ) : cfg_( cfg ) { detail::validate( cfg_ ); }

    [[nodiscard]] const Config& config() const noexcept { return cfg_; }

    /**
     * @brief Optional callback fired once per ADS split.
     *
     * Assign any callable matching `void(const SplitEvent<P,D>&)`.  Leave as
     * default-constructed `std::function` to disable.
     */
    OnSplitFn on_split{};

    /**
     * @brief Integrate the IC domain @p x0_box from @p t0 to @p tmax with
     *        adaptive domain splitting.
     */
    template < typename F >
    [[nodiscard]] TreeT
    integrate( F&& f, const Box< double, D >& x0_box, double t0, double tmax ) const
    {
        TreeT tree;

        // Root leaf.
        {
            auto root = FlowMapT{ detail::propagateDa< N, P, D >(
                f, makeDaState< P, D >( x0_box ), t0, tmax, cfg_.step_tol,
                cfg_.max_steps ) };
            tree.addLeaf( std::move( root ), x0_box );
        }

        std::vector< int > depth( 1, 0 );

        struct PendingSplit
        {
            int               parent_idx;
            int               dim;
            int               parent_depth;
            double            err;
            Box< double, D >  parent_box;
            Box< double, D >  lb, rb;
            FlowMapT          lt, rt;
        };
        std::vector< PendingSplit > splits;

        while ( !tree.empty() )
        {
            // 1. Drain the current wave.
            std::vector< int > wave;
            while ( !tree.empty() ) wave.push_back( tree.pop() );

            // 2. Decide split-or-done; stage child boxes.
            splits.clear();
            for ( int idx : wave )
            {
                const auto&  lf  = tree.node( idx ).leaf();
                const double err = detail::truncationError< P, D >( lf.tte.state );
                const int    d   = depth[idx];

                if ( err < cfg_.ads_tol || d >= cfg_.max_depth )
                {
                    tree.markDone( idx );
                }
                else
                {
                    const int dim   = detail::bestSplitDim< P, D >( lf.tte.state );
                    auto [lb, rb]   = lf.box.split( dim );
                    splits.push_back(
                        { idx, dim, d, err, lf.box, lb, rb, FlowMapT{}, FlowMapT{} } );
                }
            }

            // 3. Propagate every child box in parallel.
            const int n2 = static_cast< int >( splits.size() ) * 2;
#pragma omp parallel for schedule( dynamic ) if ( n2 > 1 )
            for ( int i = 0; i < n2; ++i )
            {
                const int  s       = i / 2;
                const bool is_left = ( i & 1 ) == 0;
                const auto& box    = is_left ? splits[s].lb : splits[s].rb;
                FlowMapT    result{ detail::propagateDa< N, P, D >(
                    f, makeDaState< P, D >( box ), t0, tmax, cfg_.step_tol,
                    cfg_.max_steps ) };
                if ( is_left )
                    splits[s].lt = std::move( result );
                else
                    splits[s].rt = std::move( result );
            }

            // 4. Apply splits to the tree serially; emit on_split.
            for ( auto& s : splits )
            {
                auto [li, ri] =
                    tree.split( s.parent_idx, s.dim, std::move( s.lt ), std::move( s.rt ) );
                if ( static_cast< int >( depth.size() ) <= ri ) depth.resize( ri + 1, 0 );
                depth[li] = s.parent_depth + 1;
                depth[ri] = s.parent_depth + 1;

                if ( on_split )
                    on_split( SplitEventT{ s.parent_idx, li, ri, s.dim, s.parent_depth, s.err,
                                           s.parent_box, s.lb, s.rb } );
            }
        }

        return tree;
    }

private:
    Config cfg_;
};

}  // namespace tax::ode
