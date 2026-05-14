#pragma once

/**
 * @file
 * @brief ADS-driven Verner Runge–Kutta integrator for DA-expanded flows.
 *
 * Mirrors @ref tax::ode::AdsIntegrator but uses the embedded Verner pairs
 * @ref tax::ode::Verner78 / @ref tax::ode::Verner89 for the per-subdomain
 * propagation, rather than the Taylor-method integrator.
 *
 * Right-hand side signature:
 *
 * @code
 *   Eigen::Matrix<tax::TEn<P, D>, D, 1>
 *     f(const Eigen::Matrix<tax::TEn<P, D>, D, 1>& x, double t);
 * @endcode
 *
 * No parameter expansion (Q) is provided in this header; users wanting joint
 * IC + parameter expansion can compose `tax::ode::makeDaParams` with a
 * generic `VernerIntegrator` themselves.
 */

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <tax/ads/ads_tree.hpp>
#include <tax/ads/box.hpp>
#include <tax/ode/ads_integrator.hpp>   // SplitEvent, detail::truncationError, etc.
#include <tax/ode/da_integrator.hpp>    // FlowMap, makeDaState
#include <tax/ode/verner_integrator.hpp>

namespace tax::ode
{

/**
 * @brief Configuration for the Verner-based ADS integrator.
 */
struct VernerAdsConfig
{
    VernerConfig step{};        ///< Per-subdomain Verner step-size control.
    double       ads_tol  = 1e-3;  ///< Truncation-error tolerance triggering a split.
    int          max_depth = 30;   ///< Maximum number of recursive bisections.
};

namespace detail
{

inline void validate( const VernerAdsConfig& cfg )
{
    detail::validate( cfg.step );
    if ( !( cfg.ads_tol > 0.0 ) )
        throw std::invalid_argument( "VernerAdsConfig: ads_tol must be > 0" );
    if ( cfg.max_depth < 0 )
        throw std::invalid_argument( "VernerAdsConfig: max_depth must be >= 0" );
}

}  // namespace detail

/**
 * @brief ADS-integrated propagation of a DA initial-condition box using a
 *        Verner Runge–Kutta pair on each subdomain.
 *
 * @tparam Coeffs  `tax::ode::detail::Verner78Coeffs` or
 *                 `tax::ode::detail::Verner89Coeffs`.
 * @tparam P       DA expansion order in the initial-condition variables.
 * @tparam D       State-space dimension (= number of DA variables).
 *
 * @details Propagation per subdomain is performed by an internal
 *   `VernerIntegrator<Coeffs, Eigen::Matrix<TEn<P,D>, D, 1>>`.  Splitting,
 *   truncation-error measurement, and tree management follow the same arena
 *   model as @ref AdsIntegrator.
 */
template < typename Coeffs, int P, int D >
class VernerAdsIntegrator
{
public:
    using DA          = TEn< P, D >;
    using VecDa       = Eigen::Matrix< DA, D, 1 >;
    using Rhs         = std::function< VecDa( const VecDa&, double ) >;
    using Config      = VernerAdsConfig;
    using FlowMapT    = FlowMap< P, D >;
    using TreeT       = AdsTree< FlowMapT >;
    using SplitEventT = SplitEvent< P, D >;
    using OnSplitFn   = std::function< void( const SplitEventT& ) >;

    explicit VernerAdsIntegrator( Rhs f, Config cfg = {} )
        : f_( std::move( f ) ), cfg_( cfg )
    {
        detail::validate( cfg_ );
    }

    [[nodiscard]] const Config& config() const noexcept { return cfg_; }

    /// @brief Optional callback fired once per ADS split.
    OnSplitFn on_split{};

    /// @brief Integrate the IC domain @p x0_box from @p t0 to @p tmax.
    [[nodiscard]] TreeT integrate( const Box< double, D >& x0_box, double t0,
                                   double tmax ) const
    {
        TreeT tree;
        tree.addLeaf( propagateRoot( x0_box, t0, tmax ), x0_box );

        std::vector< int > depth( 1, 0 );

        struct PendingSplit
        {
            int              parent_idx;
            int              dim;
            int              parent_depth;
            double           err;
            Box< double, D > parent_box;
            Box< double, D > lb, rb;
            FlowMapT         lt, rt;
        };
        std::vector< PendingSplit > splits;

        while ( !tree.empty() )
        {
            std::vector< int > wave;
            while ( !tree.empty() ) wave.push_back( tree.pop() );

            splits.clear();
            for ( int idx : wave )
            {
                const auto&  lf  = tree.node( idx ).leaf();
                const double err = detail::truncationError< P, D, D >( lf.tte.state );
                const int    d   = depth[idx];

                if ( err < cfg_.ads_tol || d >= cfg_.max_depth )
                {
                    tree.markDone( idx );
                }
                else
                {
                    const int dim = detail::bestSplitDim< P, D, D >( lf.tte.state );
                    auto [lb, rb] = lf.box.split( dim );
                    splits.push_back(
                        { idx, dim, d, err, lf.box, lb, rb, FlowMapT{}, FlowMapT{} } );
                }
            }

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
    [[nodiscard]] FlowMapT propagateRoot( const Box< double, D >& box, double t0,
                                          double tmax ) const
    {
        VecDa x0  = makeDaState< P, D >( box );
        VecDa xc  = std::move( x0 );

        const double sign  = tmax >= t0 ? 1.0 : -1.0;
        double       tc    = t0;
        const double total = std::abs( tmax - t0 );
        if ( total == 0.0 ) return FlowMapT{ xc };

        const auto& scfg = cfg_.step;
        double      h    = scfg.init_step > 0.0 ? scfg.init_step
                                                : std::max( 1e-6, 1e-3 * total );
        h = std::min( h, scfg.max_step );

        for ( int s = 0; s < scfg.max_steps; ++s )
        {
            const double remaining = tmax - tc;
            if ( sign * remaining <= 0.0 ) break;

            double h_try = std::min( h, std::abs( remaining ) );
            if ( scfg.min_step > 0.0 ) h_try = std::max( h_try, scfg.min_step );
            const double dt = sign * h_try;

            detail::VernerStepResult< VecDa > sr = [&] {
                if constexpr ( std::is_same_v< Coeffs, detail::Verner78Coeffs > )
                    return detail::verner78_step< Rhs, VecDa >( f_, xc, tc, dt );
                else
                    return detail::verner89_step< Rhs, VecDa >( f_, xc, tc, dt );
            }();

            const double x_scale = std::max( detail::verner_norm( xc ),
                                              detail::verner_norm( sr.x_new ) );
            const double sc       = scfg.abstol + scfg.reltol * x_scale;
            const double ratio    = sr.err_norm / sc;

            const double exponent = 1.0 / ( static_cast< double >( Coeffs::err_order ) + 1.0 );
            double factor;
            if ( ratio <= 0.0 )
                factor = scfg.max_factor;
            else
                factor = scfg.safety * std::pow( 1.0 / ratio, exponent );
            factor = std::clamp( factor, scfg.min_factor, scfg.max_factor );

            if ( ratio <= 1.0 )
            {
                xc = std::move( sr.x_new );
                tc += dt;
                h = std::min( h_try * factor, scfg.max_step );
            }
            else
            {
                h = h_try * factor;
                if ( scfg.min_step > 0.0 && h < scfg.min_step )
                    throw std::runtime_error(
                        "VernerAdsIntegrator: step underflow (min_step reached)" );
            }
        }

        return FlowMapT{ xc };
    }

    Rhs    f_;
    Config cfg_;
};

// =============================================================================
// Convenience aliases
// =============================================================================

/// @brief ADS-driven Verner 8(7) flow propagation (a.k.a. Verner 7/8 + ADS).
template < int P, int D >
using Verner78AdsIntegrator = VernerAdsIntegrator< detail::Verner78Coeffs, P, D >;

/// @brief ADS-driven Verner 9(8) flow propagation (a.k.a. Verner 8/9 + ADS).
template < int P, int D >
using Verner89AdsIntegrator = VernerAdsIntegrator< detail::Verner89Coeffs, P, D >;

}  // namespace tax::ode
