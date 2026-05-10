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
template < int P, int D, int M = D >
[[nodiscard]] double truncationError(
    const Eigen::Matrix< TEn< P, M >, D, 1 >& state ) noexcept
{
    double err = 0.0;
    for ( Eigen::Index i = 0; i < state.size(); ++i )
    {
        const auto& poly = state( i );
        for ( std::size_t j = 0; j < TEn< P, M >::nCoefficients; ++j )
        {
            const auto alpha = tax::detail::unflatIndex< M >( j );
            if ( tax::detail::totalDegree< M >( alpha ) == P )
                err = std::max( err, std::abs( poly[j] ) );
        }
    }
    return err;
}

/// @brief Pick the DA variable that contributes most to the degree-P
///        truncation error (Wittig et al. 2015).
template < int P, int D, int M = D >
[[nodiscard]] int bestSplitDim(
    const Eigen::Matrix< TEn< P, M >, D, 1 >& state ) noexcept
{
    std::array< double, M > scores{};
    for ( Eigen::Index i = 0; i < state.size(); ++i )
    {
        const auto& poly = state( i );
        for ( std::size_t j = 0; j < TEn< P, M >::nCoefficients; ++j )
        {
            const auto alpha = tax::detail::unflatIndex< M >( j );
            if ( tax::detail::totalDegree< M >( alpha ) == P )
                for ( int k = 0; k < M; ++k )
                    if ( alpha[k] > 0 ) scores[k] += std::abs( poly[j] );
        }
    }
    return static_cast< int >(
        std::max_element( scores.begin(), scores.end() ) - scores.begin() );
}

/// @brief Build a combined Box<D+Q> by stacking an IC box and a parameter box.
template < int D, int Q >
[[nodiscard]] Box< double, D + Q > combineBoxes( const Box< double, D >& xb,
                                                  const Box< double, Q >& pb ) noexcept
{
    Box< double, D + Q > c{};
    for ( int i = 0; i < D; ++i )
    {
        c.center[i]    = xb.center[i];
        c.halfWidth[i] = xb.halfWidth[i];
    }
    for ( int i = 0; i < Q; ++i )
    {
        c.center[D + i]    = pb.center[i];
        c.halfWidth[D + i] = pb.halfWidth[i];
    }
    return c;
}

/// @brief Extract the leading D coordinates of a combined IC+param box.
template < int D, int Q >
[[nodiscard]] Box< double, D > stateSubBox( const Box< double, D + Q >& c ) noexcept
{
    Box< double, D > b{};
    for ( int i = 0; i < D; ++i )
    {
        b.center[i]    = c.center[i];
        b.halfWidth[i] = c.halfWidth[i];
    }
    return b;
}

/// @brief Extract the trailing Q coordinates of a combined IC+param box.
template < int D, int Q >
[[nodiscard]] Box< double, Q > paramSubBox( const Box< double, D + Q >& c ) noexcept
{
    Box< double, Q > b{};
    for ( int i = 0; i < Q; ++i )
    {
        b.center[i]    = c.center[D + i];
        b.halfWidth[i] = c.halfWidth[D + i];
    }
    return b;
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
 *
 * For parameter-aware integrators (`Q > 0`), boxes span `D + Q` dimensions:
 * the first @p D coordinates are IC deviations and the trailing @p Q
 * coordinates are parameter deviations.
 */
template < int P, int D, int Q = 0 >
struct SplitEvent
{
    static constexpr int M = D + Q;
    int                  parent_idx;        ///< Arena index of the parent (now Internal).
    int                  left_idx;          ///< Arena index of the left  child leaf.
    int                  right_idx;         ///< Arena index of the right child leaf.
    int                  split_dim;         ///< Variable on which the split was made.
    int                  parent_depth;      ///< Depth of the parent (root = 0).
    double               truncation_error;  ///< Truncation error that triggered the split.
    Box< double, M >     parent_box;        ///< Parent's IC (+ param) box.
    Box< double, M >     left_box;          ///< Left  child's IC (+ param) box.
    Box< double, M >     right_box;         ///< Right child's IC (+ param) box.
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
template < int N, int P, int D, int Q = 0 >
class AdsIntegrator
{
public:
    static constexpr int M = D + Q;

    using DA          = TEn< P, M >;
    using TimeTTE     = TruncatedTaylorExpansionT< DA, N, 1 >;
    using VecTTE      = Eigen::Matrix< TimeTTE, D, 1 >;
    using VecDaP      = Eigen::Matrix< DA, Q, 1 >;
    using Config      = AdsConfig;
    using FlowMapT    = FlowMap< P, D, Q >;
    using TreeT       = AdsTree< FlowMapT >;
    using SplitEventT = SplitEvent< P, D, Q >;
    using OnSplitFn   = std::function< void( const SplitEventT& ) >;

    using RhsNoParams =
        std::function< void( VecTTE&, const VecTTE&, const TimeTTE& ) >;
    using RhsWithParams =
        std::function< void( VecTTE&, const VecTTE&, const VecDaP&, const TimeTTE& ) >;
    using Rhs = std::conditional_t< ( Q == 0 ), RhsNoParams, RhsWithParams >;

    /**
     * @brief Construct with the given right-hand side and configuration.
     * @throws std::invalid_argument on invalid configuration.
     */
    explicit AdsIntegrator( Rhs f, Config cfg = {} )
        : f_( std::move( f ) ), cfg_( cfg )
    {
        detail::validate( cfg_ );
    }

    [[nodiscard]] const Config& config() const noexcept { return cfg_; }

    /**
     * @brief Optional callback fired once per ADS split.
     *
     * Assign any callable matching `void(const SplitEvent<P,D,Q>&)`.  Leave as
     * default-constructed `std::function` to disable.
     */
    OnSplitFn on_split{};

    /// @brief Integrate the IC domain @p x0_box from @p t0 to @p tmax with
    ///        adaptive domain splitting.
    [[nodiscard]] TreeT integrate( const Box< double, D >& x0_box, double t0,
                                   double tmax ) const
        requires( Q == 0 )
    {
        return integrateImpl( x0_box, t0, tmax );
    }

    /// @brief Integrate the IC domain @p x0_box with parameters expanded
    ///        about @p p_box, from @p t0 to @p tmax, with adaptive splitting
    ///        across both IC and parameter directions.
    [[nodiscard]] TreeT integrate( const Box< double, D >& x0_box,
                                   const Box< double, Q >& p_box, double t0,
                                   double tmax ) const
        requires( Q > 0 )
    {
        return integrateImpl( detail::combineBoxes< D, Q >( x0_box, p_box ), t0, tmax );
    }

private:
    [[nodiscard]] FlowMapT propagateRoot( const Box< double, M >& box, double t0,
                                          double tmax ) const
    {
        if constexpr ( Q == 0 )
        {
            return FlowMapT{ detail::propagateDa< N, P, D >(
                f_, makeDaState< P, D >( box ), t0, tmax, cfg_.step_tol, cfg_.max_steps ) };
        }
        else
        {
            const auto x_box = detail::stateSubBox< D, Q >( box );
            const auto p_box = detail::paramSubBox< D, Q >( box );
            return FlowMapT{ detail::propagateDa< N, P, D, Q >(
                f_, makeDaState< P, D, Q >( x_box ), makeDaParams< P, D, Q >( p_box ), t0, tmax,
                cfg_.step_tol, cfg_.max_steps ) };
        }
    }

    [[nodiscard]] TreeT integrateImpl( const Box< double, M >& root_box, double t0,
                                        double tmax ) const
    {
        TreeT tree;

        // Root leaf.
        tree.addLeaf( propagateRoot( root_box, t0, tmax ), root_box );

        std::vector< int > depth( 1, 0 );

        struct PendingSplit
        {
            int                parent_idx;
            int                dim;
            int                parent_depth;
            double             err;
            Box< double, M >   parent_box;
            Box< double, M >   lb, rb;
            FlowMapT           lt, rt;
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
                const double err = detail::truncationError< P, D, M >( lf.tte.state );
                const int    d   = depth[idx];

                if ( err < cfg_.ads_tol || d >= cfg_.max_depth )
                {
                    tree.markDone( idx );
                }
                else
                {
                    const int dim = detail::bestSplitDim< P, D, M >( lf.tte.state );
                    auto [lb, rb] = lf.box.split( dim );
                    splits.push_back(
                        { idx, dim, d, err, lf.box, lb, rb, FlowMapT{}, FlowMapT{} } );
                }
            }

            // 3. Propagate every child box in parallel.
            const int n2 = static_cast< int >( splits.size() ) * 2;
#pragma omp parallel for schedule( dynamic ) if ( n2 > 1 )
            for ( int i = 0; i < n2; ++i )
            {
                const int   s       = i / 2;
                const bool  is_left = ( i & 1 ) == 0;
                const auto& box     = is_left ? splits[s].lb : splits[s].rb;
                FlowMapT    result  = propagateRoot( box, t0, tmax );
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

    Rhs    f_;
    Config cfg_;
};

}  // namespace tax::ode
