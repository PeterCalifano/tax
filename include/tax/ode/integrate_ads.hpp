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
 * @tparam N  Taylor expansion order in time.
 * @tparam P  DA expansion order in the initial-condition variables.
 * @tparam D  State-space dimension (= number of DA variables).
 * @param f             Right-hand side `f(dx, x, t)`.
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

    while ( !tree.empty() )
    {
        const int    idx = tree.pop();
        const auto&  lf  = tree.node( idx ).leaf();
        const double err = detail::truncationError< P, D >( lf.tte.state );
        const int    d   = depth[idx];

        if ( err < ads_tol || d >= ads_max_depth )
        {
            tree.markDone( idx );
        }
        else
        {
            const int dim = detail::bestSplitDim< P, D >( lf.tte.state );
            auto [lb, rb] = lf.box.split( dim );
            auto lt        = evaluate_box( lb );
            auto rt        = evaluate_box( rb );

            auto [li, ri] = tree.split( idx, dim, std::move( lt ), std::move( rt ) );

            if ( static_cast< int >( depth.size() ) <= ri )
                depth.resize( ri + 1, 0 );
            depth[li] = d + 1;
            depth[ri] = d + 1;
        }
    }

    return tree;
}

}  // namespace tax::ode
