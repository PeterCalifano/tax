#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <tax/ads/ads_tree.hpp>
#include <tax/ads/box.hpp>
#include <tax/ads/nonlinearity_index.hpp>
#include <tax/ode/da_integrator.hpp>
#include <tax/storage/tte_static.hpp>

namespace tax::ode
{

// =============================================================================
// Configuration
// =============================================================================

/**
 * @brief Configuration for the low-order ADS integrator.
 *
 * The splitting criterion is the polynomial bound on the Jacobian
 * variation of the flow map over the unit IC box, normalised by the
 * central Jacobian (Losacco, Fossà, Armellin, J. Guid. Control Dyn.
 * 2024; arXiv:2303.05791).
 */
struct LowOrderAdsConfig
{
    double step_tol  = 1e-14;  ///< Adaptive-step-size tolerance for the inner Taylor propagator.
    double nli_tol   = 1e-3;   ///< Nonlinearity-index threshold above which a leaf is split.
    int    max_depth = 30;     ///< Maximum number of recursive bisections from the root.
    int    max_steps = 500;    ///< Maximum integration steps per subdomain.
};

namespace detail
{

inline void validate( const LowOrderAdsConfig& cfg )
{
    if ( !( cfg.step_tol > 0.0 ) )
        throw std::invalid_argument( "LowOrderAdsConfig: step_tol must be > 0" );
    if ( !( cfg.nli_tol > 0.0 ) )
        throw std::invalid_argument( "LowOrderAdsConfig: nli_tol must be > 0" );
    if ( cfg.max_depth < 0 )
        throw std::invalid_argument( "LowOrderAdsConfig: max_depth must be >= 0" );
    if ( cfg.max_steps <= 0 )
        throw std::invalid_argument( "LowOrderAdsConfig: max_steps must be > 0" );
}

/// @brief Nonlinearity index of a propagated DA flow-state vector.
template < int P, int D, int M = D >
[[nodiscard]] inline double stateNonlinearityIndex(
    const Eigen::Matrix< TEn< P, M >, D, 1 >& state ) noexcept
{
    return tax::nonlinearityIndex< double, P, M >(
        std::span< const TEn< P, M > >( state.data(),
                                        std::size_t( state.size() ) ) );
}

/// @brief Argmax of the per-variable nonlinearity contribution.
template < int P, int D, int M = D >
[[nodiscard]] inline int stateNliSplitDim(
    const Eigen::Matrix< TEn< P, M >, D, 1 >& state ) noexcept
{
    return tax::nliSplitDim< double, P, M >(
        std::span< const TEn< P, M > >( state.data(),
                                        std::size_t( state.size() ) ) );
}

}  // namespace detail

// =============================================================================
// LowOrderSplitEvent — payload for the on_split callback
// =============================================================================

/**
 * @brief Information about a single low-order ADS bisection.
 *
 * Mirrors @ref SplitEvent but carries the nonlinearity index that
 * triggered the split instead of a truncation-error norm.  For
 * parameter-aware integrators (`Q > 0`), boxes span `D + Q` dimensions.
 */
template < int P, int D, int Q = 0 >
struct LowOrderSplitEvent
{
    static constexpr int M = D + Q;
    int                  parent_idx;          ///< Arena index of the parent (now Internal).
    int                  left_idx;            ///< Arena index of the left  child leaf.
    int                  right_idx;           ///< Arena index of the right child leaf.
    int                  split_dim;           ///< Variable on which the split was made.
    int                  parent_depth;        ///< Depth of the parent (root = 0).
    double               nonlinearity_index;  ///< NLI value that triggered the split.
    Box< double, M >     parent_box;          ///< Parent's IC (+ param) box.
    Box< double, M >     left_box;            ///< Left  child's IC (+ param) box.
    Box< double, M >     right_box;           ///< Right child's IC (+ param) box.
};

// =============================================================================
// LowOrderAdsIntegrator class
// =============================================================================

/**
 * @brief Low-order Automatic Domain Splitting integrator for vector ODEs.
 *
 * Drives DA flow propagation and adaptively bisects the initial-condition
 * box wherever the @ref nonlinearityIndex of the propagated state
 * exceeds @ref LowOrderAdsConfig::nli_tol.  This is the orbit-mapping
 * variant of the algorithm of Losacco, Fossà, Armellin (J. Guid. Control
 * Dyn. 2024; arXiv:2303.05791) and is intended to be used at low DA
 * orders (P ≥ 2, with P = 2 capturing the bulk of the nonlinearity
 * information used by the index).
 *
 * The control flow mirrors @ref AdsIntegrator — including the parallel
 * wave-by-wave propagation and on_split callback — so the two classes
 * are drop-in interchangeable when comparing splitting heuristics.
 *
 * @tparam N  Taylor expansion order in time.
 * @tparam P  DA expansion order in the initial-condition variables.
 *            Must satisfy P ≥ 2.
 * @tparam D  State-space dimension (= number of DA variables).
 */
template < int N, int P, int D, int Q = 0 >
class LowOrderAdsIntegrator
{
public:
    static_assert( P >= 2,
                   "LowOrderAdsIntegrator requires P >= 2 (the nonlinearity "
                   "index is defined from degree-≥-2 coefficients)" );

    static constexpr int M = D + Q;

    using DA          = TEn< P, M >;
    using TimeTTE     = TaylorExpansionT< DA, N, 1 >;
    using VecTTE      = Eigen::Matrix< TimeTTE, D, 1 >;
    using VecDaP      = Eigen::Matrix< DA, Q, 1 >;
    using Config      = LowOrderAdsConfig;
    using FlowMapT    = FlowMap< P, D, Q >;
    using TreeT       = AdsTree< FlowMapT >;
    using SplitEventT = LowOrderSplitEvent< P, D, Q >;
    using OnSplitFn   = std::function< void( const SplitEventT& ) >;

    using RhsNoParams =
        std::function< void( VecTTE&, const VecTTE&, const TimeTTE& ) >;
    using RhsWithParams =
        std::function< void( VecTTE&, const VecTTE&, const VecDaP&, const TimeTTE& ) >;
    using Rhs = std::conditional_t< ( Q == 0 ), RhsNoParams, RhsWithParams >;

    explicit LowOrderAdsIntegrator( Rhs f, Config cfg = {} )
        : f_( std::move( f ) ), cfg_( cfg )
    {
        detail::validate( cfg_ );
    }

    [[nodiscard]] const Config& config() const noexcept { return cfg_; }

    /**
     * @brief Optional callback fired once per ADS split.
     *
     * Assign any callable matching `void(const LowOrderSplitEvent<P,D,Q>&)`.
     * Leave as default-constructed `std::function` to disable.
     */
    OnSplitFn on_split{};

    /// @brief Integrate the IC domain @p x0_box from @p t0 to @p tmax
    ///        with NLI-driven domain splitting.
    [[nodiscard]] TreeT integrate( const Box< double, D >& x0_box, double t0,
                                   double tmax ) const
        requires( Q == 0 )
    {
        return integrateImpl( x0_box, t0, tmax );
    }

    /// @brief Integrate the IC domain @p x0_box with parameters expanded
    ///        about @p p_box, from @p t0 to @p tmax, with NLI-driven
    ///        splitting across both IC and parameter directions.
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

        tree.addLeaf( propagateRoot( root_box, t0, tmax ), root_box );

        std::vector< int > depth( 1, 0 );

        struct PendingSplit
        {
            int                parent_idx;
            int                dim;
            int                parent_depth;
            double             nu;
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
                const auto&  lf = tree.node( idx ).leaf();
                const double nu =
                    detail::stateNonlinearityIndex< P, D, M >( lf.tte.state );
                const int d = depth[idx];

                if ( nu <= cfg_.nli_tol || d >= cfg_.max_depth )
                {
                    tree.markDone( idx );
                }
                else
                {
                    const int dim = detail::stateNliSplitDim< P, D, M >( lf.tte.state );
                    auto [lb, rb] = lf.box.split( dim );
                    splits.push_back(
                        { idx, dim, d, nu, lf.box, lb, rb, FlowMapT{}, FlowMapT{} } );
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
                    on_split( SplitEventT{ s.parent_idx, li, ri, s.dim, s.parent_depth,
                                           s.nu, s.parent_box, s.lb, s.rb } );
            }
        }

        return tree;
    }

    Rhs    f_;
    Config cfg_;
};

}  // namespace tax::ode
