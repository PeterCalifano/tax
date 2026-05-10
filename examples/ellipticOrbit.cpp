// =============================================================================
// Planar elliptic orbit: plain Taylor → flow expansion → ADS
//
// Three runs of the same Kepler problem (μ = 1, planar) propagated for one
// orbital period of an a=1, e=0.5 ellipse:
//
//   1. Plain Taylor integration of a single trajectory (the reference orbit).
//   2. propagateBox(): a single multivariate-Taylor flow map x(tmax; x0+δ)
//      around the reference IC, valid only while the polynomial truncation
//      remains accurate.
//   3. integrateAds(): the same domain split into pieces wherever the
//      truncation error exceeds a tolerance, producing a piecewise polynomial
//      flow map.
//
// All three runs share the same right-hand side and the same final time.  The
// program writes four CSV files consumed by the companion plotting script:
//
//   orbit_reference.csv     — reference trajectory (t, x, y, vx, vy)
//   orbits_perturbed.csv    — a handful of perturbed trajectories
//   endpoint_compare.csv    — endpoint of every method at tmax across δ ∈ [-1,1]
//   ads_leaves.csv          — sub-domain bounds of each ADS leaf
// =============================================================================

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numbers>
#include <vector>

#include <tax/tax.hpp>
#include <tax/ode/taylor_integrator.hpp>

using namespace tax;

namespace
{

constexpr int kN = 14;  ///< Taylor expansion order in time.
constexpr int kP = 4;   ///< DA expansion order in initial conditions.
constexpr int kD = 4;   ///< 2 position + 2 velocity components.

/// Planar two-body RHS with normalised gravitational parameter μ = 1.
constexpr auto kepler = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t )
{
    using std::sqrt;
    auto r2 = x( 0 ) * x( 0 ) + x( 1 ) * x( 1 );
    auto r  = sqrt( r2 );
    auto r3 = r2 * r;
    dx( 0 ) = x( 2 );
    dx( 1 ) = x( 3 );
    dx( 2 ) = -x( 0 ) / r3;
    dx( 3 ) = -x( 1 ) / r3;
};

}  // namespace

int main()
{
    // -------------------------------------------------------------------------
    // Reference IC: periapsis of an a = 1, e = 0.5 ellipse.
    //   r_p = a(1 - e),    v_p = sqrt(μ/a) · sqrt((1+e)/(1-e))
    //   period T = 2π · sqrt(a³/μ) = 2π
    // -------------------------------------------------------------------------
    constexpr double a    = 1.0;
    constexpr double e    = 0.5;
    const double     rp   = a * ( 1.0 - e );
    const double     vp   = std::sqrt( ( 1.0 + e ) / ( 1.0 - e ) );
    const double     tmax = std::numbers::pi;  // periapsis → apoapsis (half period)

    Eigen::Vector< double, kD > x0;
    x0 << rp, 0.0, 0.0, vp;

    // -------------------------------------------------------------------------
    // 1) Plain Taylor integration of the reference orbit.
    // -------------------------------------------------------------------------
    auto sol = ode::integrate< kN >( kepler, x0, 0.0, tmax, 1e-16 );
    {
        std::ofstream out( "orbit_reference.csv" );
        out << "t,x,y,vx,vy\n";
        for ( std::size_t i = 0; i < sol.t.size(); ++i )
            out << sol.t[i] << ',' << sol.x[i]( 0 ) << ',' << sol.x[i]( 1 ) << ','
                << sol.x[i]( 2 ) << ',' << sol.x[i]( 3 ) << '\n';
    }
    std::cout << "Plain integration:    " << sol.t.size() - 1 << " steps to t = " << sol.t.back()
              << '\n';

    // -------------------------------------------------------------------------
    // IC uncertainty: a 1-D box in periapsis tangential velocity v_y(0).
    // The box is large enough that a single Taylor flow visibly degrades by
    // tmax (one full period), motivating the ADS split.
    // -------------------------------------------------------------------------
    Box< double, kD > box{ { rp, 0.0, 0.0, vp }, { 0.0, 0.0, 0.0, 0.10 } };

    // -------------------------------------------------------------------------
    // 2) Single flow expansion (propagateBox, no splitting).
    // -------------------------------------------------------------------------
    auto flow = ode::propagateBox< kN, kP, kD >( kepler, box, 0.0, tmax, 1e-14 );
    std::cout << "Flow expansion:       single polynomial, order P = " << kP << '\n';

    // -------------------------------------------------------------------------
    // 3) ADS-integrated flow expansion.
    // -------------------------------------------------------------------------
    constexpr double ads_tol = 1e-4;
    auto             tree    = ode::integrateAds< kN, kP >( kepler, box, 0.0, tmax, 1e-14, ads_tol,
                                                            12 );
    std::cout << "ADS:                  " << tree.numDone() << " leaves (tol = " << ads_tol << ")\n";

    // -------------------------------------------------------------------------
    // Robust leaf lookup: tree.findLeaf walks via strict comparisons, so a
    // query exactly on a split boundary can fall outside the matched leaf in
    // floating-point.  Fall back to a linear scan with a tiny tolerance.
    // -------------------------------------------------------------------------
    auto find_leaf_robust = [&]( const std::array< double, kD >& q ) -> int {
        const int idx = tree.findLeaf( q );
        if ( idx >= 0 ) return idx;
        constexpr double eps = 1e-9;
        for ( int li : tree.doneLeaves() )
        {
            const auto& b      = tree.node( li ).leaf().box;
            bool        inside = true;
            for ( int k = 0; k < kD; ++k )
                if ( std::abs( q[k] - b.center[k] ) > b.halfWidth[k] + eps )
                {
                    inside = false;
                    break;
                }
            if ( inside ) return li;
        }
        return -1;
    };

    // -------------------------------------------------------------------------
    // Sample the IC box and compare predicted vs true endpoint at tmax.
    // -------------------------------------------------------------------------
    constexpr int n_samples = 41;
    std::ofstream cmp( "endpoint_compare.csv" );
    cmp << "delta,vy0,truex,truey,truevx,truevy,flowx,flowy,flowvx,flowvy,adsx,adsy,adsvx,adsvy\n";

    double max_flow_err = 0.0, max_ads_err = 0.0;
    for ( int i = 0; i < n_samples; ++i )
    {
        // Sample δ in the open interval (−1, 1) — exact box endpoints round
        // ambiguously across leaf boundaries in findLeaf().
        const double delta = -1.0 + 2.0 * ( double( i ) + 0.5 ) / double( n_samples );
        const double vy0   = vp + box.halfWidth[3] * delta;

        // Truth: integrate the perturbed IC directly.
        Eigen::Vector< double, kD > x0p;
        x0p << rp, 0.0, 0.0, vy0;
        auto        sol_p = ode::integrate< kN >( kepler, x0p, 0.0, tmax, 1e-16 );
        const auto& xt    = sol_p.x.back();

        // Single flow polynomial at δ (only δ_3 = δ varies).
        const std::array< double, kD > d_full{ 0.0, 0.0, 0.0, delta };
        Eigen::Vector< double, kD >    xf;
        for ( int k = 0; k < kD; ++k ) xf( k ) = flow( k ).eval( d_full );

        // ADS leaf containing the perturbed IC, then evaluate at the leaf-local δ.
        const std::array< double, kD > query{ rp, 0.0, 0.0, vy0 };
        const int                      leaf_idx = find_leaf_robust( query );
        Eigen::Vector< double, kD >    xa       = Eigen::Vector< double, kD >::Zero();
        if ( leaf_idx >= 0 )
        {
            const auto&              leaf = tree.node( leaf_idx ).leaf();
            std::array< double, kD > local_delta{};
            for ( int k = 0; k < kD; ++k )
                local_delta[k] = leaf.box.halfWidth[k] > 0.0
                                     ? ( query[k] - leaf.box.center[k] ) / leaf.box.halfWidth[k]
                                     : 0.0;
            for ( int k = 0; k < kD; ++k ) xa( k ) = leaf.tte.state( k ).eval( local_delta );
        }

        const double err_flow = ( xt - xf ).norm();
        const double err_ads  = ( xt - xa ).norm();
        max_flow_err          = std::max( max_flow_err, err_flow );
        max_ads_err           = std::max( max_ads_err, err_ads );

        cmp << delta << ',' << vy0 << ',' << xt( 0 ) << ',' << xt( 1 ) << ',' << xt( 2 ) << ','
            << xt( 3 ) << ',' << xf( 0 ) << ',' << xf( 1 ) << ',' << xf( 2 ) << ',' << xf( 3 )
            << ',' << xa( 0 ) << ',' << xa( 1 ) << ',' << xa( 2 ) << ',' << xa( 3 ) << '\n';
    }

    std::cout << "Max endpoint error vs truth across δ ∈ [-1, 1]:\n";
    std::cout << "  single flow polynomial: " << max_flow_err << '\n';
    std::cout << "  ADS piecewise:          " << max_ads_err << '\n';

    // -------------------------------------------------------------------------
    // A handful of full perturbed trajectories for the orbit-plane plot.
    // -------------------------------------------------------------------------
    const std::vector< double > deltas_traj = { -1.0, -0.5, 0.0, 0.5, 1.0 };
    std::ofstream               traj( "orbits_perturbed.csv" );
    traj << "delta,t,x,y\n";
    for ( double d : deltas_traj )
    {
        Eigen::Vector< double, kD > x0p;
        x0p << rp, 0.0, 0.0, vp + box.halfWidth[3] * d;
        auto sp = ode::integrate< kN >( kepler, x0p, 0.0, tmax, 1e-16 );
        for ( std::size_t i = 0; i < sp.t.size(); ++i )
            traj << d << ',' << sp.t[i] << ',' << sp.x[i]( 0 ) << ',' << sp.x[i]( 1 ) << '\n';
    }

    // -------------------------------------------------------------------------
    // ADS leaf summary (sub-domain bounds in v_y, the only varying axis).
    // -------------------------------------------------------------------------
    std::ofstream lf( "ads_leaves.csv" );
    lf << "leaf_idx,vy_lo,vy_hi\n";
    for ( int li : tree.doneLeaves() )
    {
        const auto&  leaf = tree.node( li ).leaf();
        const double lo   = leaf.box.center[3] - leaf.box.halfWidth[3];
        const double hi   = leaf.box.center[3] + leaf.box.halfWidth[3];
        lf << li << ',' << lo << ',' << hi << '\n';
    }

    std::cout << "Wrote: orbit_reference.csv, orbits_perturbed.csv, "
                 "endpoint_compare.csv, ads_leaves.csv\n";
    return 0;
}
