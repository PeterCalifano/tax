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
// program writes the following CSV files consumed by the companion plotting
// scripts:
//
//   orbit_reference.csv     — reference trajectory (t, x, y, vx, vy)
//   orbits_perturbed.csv    — a handful of perturbed trajectories
//   endpoint_compare.csv    — endpoint of every method at tmax across δ ∈ [-1,1]
//   ads_leaves.csv          — sub-domain bounds of each 1-D ADS leaf
//   ads_box_snapshots.csv   — boundary of each ADS leaf at each snapshot time
//   flow_box_snapshots.csv  — boundary of the single-flow polygon at each
//                             snapshot time
//   ads_box_leaves.csv      — IC-space bounds of every ADS leaf at each snapshot
//
// Companion scripts:
//   plotEllipticOrbit.py     — orbit, endpoint scatter, error-vs-δ figure
//   plotBoxEvolution.py      — ADS leaves vs single-flow polygon at each
//                              snapshot time (the "box pushed forward" plot)
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

constexpr int kN = 12;  ///< Taylor expansion order in time.
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
    const double     tmax_orbit = 2.0 * std::numbers::pi; // full period
    const double     tmax       = 0.5 * std::numbers::pi; // 1-D analysis (faster)

    Eigen::Vector< double, kD > x0;
    x0 << rp, 0.0, 0.0, vp;

    // -------------------------------------------------------------------------
    // 1) Plain Taylor integration of the reference orbit.
    // -------------------------------------------------------------------------
    auto sol = ode::integrate< kN >( kepler, x0, 0.0, tmax_orbit, 1e-16 );
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
    Box< double, kD > box{ { rp, 0.0, 0.0, vp }, { 0.0, 0.0, 0.0, 0.08 } };

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
                                                            6 );
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
        auto sp = ode::integrate< kN >( kepler, x0p, 0.0, tmax_orbit, 1e-16 );
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

    // -------------------------------------------------------------------------
    // 2-D IC box pushed forward in time: the classic ADS visualisation.
    //
    // Perturb both y(0) and v_y(0) about the periapsis IC.  At a handful of
    // snapshot times we run propagateBox (single polynomial) and integrateAds
    // (piecewise) and sample the boundary of the unit square in normalised
    // δ ∈ [-1, 1]^2.  Mapping that boundary through each leaf's polynomial
    // gives a deformed quadrilateral in the (x, y) plane: the original IC box
    // pushed forward to the snapshot time, partitioned into ADS leaves.
    // -------------------------------------------------------------------------
    Box< double, kD > box2D{ { rp, 0.0, 0.0, vp }, { 0.0, 0.020, 0.0, 0.030 } };

    // Ten snapshots equally spaced along one full orbital period.  Equal time
    // spacing (not equal anomaly) puts more snapshots near apoapsis where the
    // orbit is slow.  Each snapshot triggers a fresh propagateBox + integrateAds
    // run from t=0; the late ones (near apoapsis return) dominate the runtime.
    constexpr int         n_snapshots = 10;
    std::vector< double > snapshots( n_snapshots );
    for ( int k = 0; k < n_snapshots; ++k )
        snapshots[k] = ( double( k + 1 ) / double( n_snapshots ) ) * tmax_orbit;

    constexpr int n_per_edge = 24;  ///< boundary samples per side of the unit square

    auto unit_square_boundary = []( int n ) {
        std::vector< std::array< double, 2 > > pts;
        pts.reserve( std::size_t( 4 * n + 1 ) );
        for ( int e = 0; e < 4; ++e )
            for ( int i = 0; i < n; ++i )
            {
                const double s = double( i ) / double( n );
                double       dy = 0.0, dvy = 0.0;
                switch ( e )
                {
                case 0: dy = -1.0 + 2.0 * s; dvy = +1.0; break;            // top
                case 1: dy = +1.0; dvy = +1.0 - 2.0 * s; break;            // right
                case 2: dy = +1.0 - 2.0 * s; dvy = -1.0; break;            // bottom
                case 3: dy = -1.0; dvy = -1.0 + 2.0 * s; break;            // left
                }
                pts.push_back( { dy, dvy } );
            }
        pts.push_back( pts.front() );  // close the loop
        return pts;
    };

    const auto bnd = unit_square_boundary( n_per_edge );

    std::ofstream sf( "ads_box_snapshots.csv" );
    sf << "snapshot,t,leaf_idx,sample_idx,x,y\n";
    std::ofstream sff( "flow_box_snapshots.csv" );
    sff << "snapshot,t,sample_idx,x,y\n";
    std::ofstream sfl( "ads_box_leaves.csv" );
    sfl << "snapshot,t,leaf_idx,dy_lo,dy_hi,dvy_lo,dvy_hi\n";

    for ( std::size_t s = 0; s < snapshots.size(); ++s )
    {
        const double t_snap = snapshots[s];

        auto tree2 = ode::integrateAds< kN, kP >( kepler, box2D, 0.0, t_snap, 1e-13, 1e-3, 4 );
        auto flow2 = ode::propagateBox< kN, kP, kD >( kepler, box2D, 0.0, t_snap, 1e-14 );

        std::cout << "Snapshot t = " << t_snap << ":  ADS leaves = " << tree2.numDone() << '\n';

        // ADS: sample the unit-square boundary in each leaf's local δ.
        for ( int li : tree2.doneLeaves() )
        {
            const auto& leaf = tree2.node( li ).leaf();
            for ( std::size_t i = 0; i < bnd.size(); ++i )
            {
                const std::array< double, kD > d{ 0.0, bnd[i][0], 0.0, bnd[i][1] };
                const double                   x = leaf.tte.state( 0 ).eval( d );
                const double                   y = leaf.tte.state( 1 ).eval( d );
                sf << s << ',' << t_snap << ',' << li << ',' << i << ',' << x << ',' << y << '\n';
            }
            // Leaf bounds in the original normalised δ-space (for the IC-side panel).
            const double dy_lo = ( ( leaf.box.center[1] - leaf.box.halfWidth[1] ) -
                                   box2D.center[1] ) / box2D.halfWidth[1];
            const double dy_hi = ( ( leaf.box.center[1] + leaf.box.halfWidth[1] ) -
                                   box2D.center[1] ) / box2D.halfWidth[1];
            const double dvy_lo = ( ( leaf.box.center[3] - leaf.box.halfWidth[3] ) -
                                    box2D.center[3] ) / box2D.halfWidth[3];
            const double dvy_hi = ( ( leaf.box.center[3] + leaf.box.halfWidth[3] ) -
                                    box2D.center[3] ) / box2D.halfWidth[3];
            sfl << s << ',' << t_snap << ',' << li << ',' << dy_lo << ',' << dy_hi << ','
                << dvy_lo << ',' << dvy_hi << '\n';
        }

        // Single flow: sample the unit-square boundary in box2D-relative δ.
        for ( std::size_t i = 0; i < bnd.size(); ++i )
        {
            const std::array< double, kD > d{ 0.0, bnd[i][0], 0.0, bnd[i][1] };
            const double                   x = flow2( 0 ).eval( d );
            const double                   y = flow2( 1 ).eval( d );
            sff << s << ',' << t_snap << ',' << i << ',' << x << ',' << y << '\n';
        }
    }

    std::cout << "Wrote: orbit_reference.csv, orbits_perturbed.csv, endpoint_compare.csv,\n"
              << "       ads_leaves.csv, ads_box_snapshots.csv, flow_box_snapshots.csv,\n"
              << "       ads_box_leaves.csv\n";
    return 0;
}
