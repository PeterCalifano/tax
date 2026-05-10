// =============================================================================
// Two-body ADS comparison
//
// Side-by-side benchmark of the two ADS strategies provided by the library on
// the planar Kepler problem (μ = 1, 4D phase space):
//
//   1. AdsIntegrator         — split when the degree-P truncation error of
//                              the flow polynomial exceeds the tolerance
//                              (Wittig et al. 2015).
//   2. LowOrderAdsIntegrator — split when the *nonlinearity index* of the
//                              flow polynomial — a polynomial bound on the
//                              Jacobian variation over the IC box — exceeds
//                              the tolerance (Losacco, Fossà, Armellin,
//                              J. Guid. Control Dyn. 2024;
//                              arXiv:2303.05791).
//
// Both methods are run on the same reference orbit (a = 1, e = 0.5, propagated
// over one orbital period from periapsis) with the same IC box: a 2 %
// perturbation of the tangential periapsis velocity v_y(0) combined with a
// position offset along x.  The two methods receive *the same tolerance
// number* — interpreted as a truncation-error norm or a Jacobian-variation
// ratio, respectively.  The headline result is a comparison of:
//   - the number of subdomains produced;
//   - the maximum end-point error vs a high-precision scalar integration of
//     the same IC, sampled on a regular grid over δ ∈ [-1, 1]^2.
//
// Output:
//   twoBody_ads_comparison.csv    — per-sample errors of both methods
//   twoBody_te_leaves.csv         — IC-space bounds of the AdsIntegrator leaves
//   twoBody_lo_leaves.csv         — IC-space bounds of the LowOrderAdsIntegrator leaves
//   twoBody_te_leaf_orbit.csv     — leaf boundaries pushed forward to (x, y)
//   twoBody_lo_leaf_orbit.csv     — same for the LowOrderAdsIntegrator leaves
//   twoBody_orbit_reference.csv   — full elliptic orbit of the central IC
//   twoBody_orbit_endpoint.csv    — central-IC position at t = tmax
//   twoBody_snapshots.csv         — leaf boundaries at each snapshot time, both methods
//   twoBody_snapshots_meta.csv    — snapshot time and per-method leaf count
//
// Companion plotting scripts:
//   plotTwoBodyAdsComparison.py  — IC-space + endpoint error figure
//   plotTwoBodyAdsOrbit.py       — splitting projected onto the orbit
//   plotTwoBodyAdsSnapshots.py   — time evolution of the partition through one orbit
// =============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>

#include <tax/tax.hpp>
#include <tax/ode/taylor_integrator.hpp>

using namespace tax;

namespace
{

constexpr int kN = 12;  ///< Taylor order in time.
constexpr int kP = 2;   ///< DA order in initial conditions (low-order ADS uses P = 2).
constexpr int kD = 4;   ///< Phase-space dimension: 2 position + 2 velocity.

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

using Vec = Eigen::Vector< double, kD >;

/// Look up the leaf containing @p q, falling back to a tolerant linear scan
/// when the binary-tree walk lands on a split boundary.
template < typename Tree >
int findLeafRobust( const Tree& tree, const std::array< double, kD >& q )
{
    if ( int idx = tree.findLeaf( q ); idx >= 0 ) return idx;
    constexpr double eps = 1e-9;
    for ( int li : tree.doneLeaves() )
    {
        const auto& b      = tree.node( li ).leaf().box;
        bool        inside = true;
        for ( int k = 0; k < kD; ++k )
            if ( std::abs( q[k] - b.center[k] ) > b.halfWidth[k] + eps ) { inside = false; break; }
        if ( inside ) return li;
    }
    return -1;
}

/// Evaluate the polynomial flow stored in the leaf containing @p q.
template < typename Tree >
Vec evalAtPoint( const Tree& tree, const std::array< double, kD >& q )
{
    const int idx = findLeafRobust( tree, q );
    if ( idx < 0 ) return Vec::Constant( std::nan( "" ) );
    const auto& leaf = tree.node( idx ).leaf();
    std::array< double, kD > local{};
    for ( int k = 0; k < kD; ++k )
        local[k] = leaf.box.halfWidth[k] > 0.0
                       ? ( q[k] - leaf.box.center[k] ) / leaf.box.halfWidth[k]
                       : 0.0;
    Vec out;
    for ( int k = 0; k < kD; ++k ) out( k ) = leaf.tte.state( k ).eval( local );
    return out;
}

}  // namespace

int main()
{
    // -------------------------------------------------------------------------
    // Reference orbit: a = 1, e = 0.5 ellipse, starting at periapsis.
    // -------------------------------------------------------------------------
    constexpr double a    = 1.0;
    constexpr double e    = 0.5;
    const double     rp   = a * ( 1.0 - e );
    const double     vp   = std::sqrt( ( 1.0 + e ) / ( 1.0 - e ) );
    const double     tmax = std::numbers::pi;  // half orbital period (apoapsis pass)

    // 2-D IC uncertainty along x(0) and v_y(0).  Sized so the IC partition is
    // visible when pushed forward to (x, y) at t_max while still letting both
    // criteria converge with a moderate split budget.
    Box< double, kD > box{ { rp,    0.0, 0.0, vp   },
                           { 0.01,  0.0, 0.0, 0.02 } };

    std::cout << "Two-body ADS comparison\n"
              << "  Orbit:        a = " << a << ", e = " << e << "\n"
              << "  Time span:    [0, " << tmax << "]\n"
              << "  IC box:       δx ∈ ±" << box.halfWidth[0]
              << ",  δv_y ∈ ±"            << box.halfWidth[3] << "\n"
              << "  Tolerance:    tol = 1e-3 for both criteria\n"
              << "  N (time) = " << kN << ", P (IC) = " << kP << "\n\n";

    constexpr double tol = 1e-3;

    // -------------------------------------------------------------------------
    // 1) Truncation-error ADS (classical Wittig criterion).
    // -------------------------------------------------------------------------
    ode::AdsIntegrator< kN, kP, kD > te_ig{
        kepler, ode::AdsConfig{ .step_tol = 1e-14, .ads_tol = tol, .max_depth = 8 } };
    int te_splits = 0;
    te_ig.on_split = [&]( const ode::SplitEvent< kP, kD >& ) { ++te_splits; };
    auto te_tree = te_ig.integrate( box, 0.0, tmax );

    // -------------------------------------------------------------------------
    // 2) Low-order NLI-driven ADS.
    // -------------------------------------------------------------------------
    ode::LowOrderAdsIntegrator< kN, kP, kD > lo_ig{
        kepler, ode::LowOrderAdsConfig{ .step_tol = 1e-14, .nli_tol = tol, .max_depth = 8 } };
    int lo_splits = 0;
    lo_ig.on_split = [&]( const ode::LowOrderSplitEvent< kP, kD >& ) { ++lo_splits; };
    auto lo_tree = lo_ig.integrate( box, 0.0, tmax );

    // -------------------------------------------------------------------------
    // Dump the IC-space partition produced by each method along the two
    // *active* axes: x(0) (dim 0) and v_y(0) (dim 3).  The other two axes are
    // degenerate (halfWidth = 0) and need not be plotted.
    // -------------------------------------------------------------------------
    auto dumpLeaves = [&]( const auto& tree, const char* path ) {
        std::ofstream out( path );
        out << "x_lo,x_hi,vy_lo,vy_hi\n";
        for ( int li : tree.doneLeaves() )
        {
            const auto& b = tree.node( li ).leaf().box;
            const double xlo = b.center[0] - b.halfWidth[0];
            const double xhi = b.center[0] + b.halfWidth[0];
            const double vlo = b.center[3] - b.halfWidth[3];
            const double vhi = b.center[3] + b.halfWidth[3];
            out << xlo << ',' << xhi << ',' << vlo << ',' << vhi << '\n';
        }
    };
    dumpLeaves( te_tree, "twoBody_te_leaves.csv" );
    dumpLeaves( lo_tree, "twoBody_lo_leaves.csv" );

    // -------------------------------------------------------------------------
    // Push every leaf's IC-box boundary forward to (x, y) at t = tmax by
    // evaluating its flow polynomial at uniformly spaced points along the
    // perimeter of δ ∈ [-1, 1]² (the two active dimensions).  This is the
    // "split on orbit" view: each method's IC partition becomes a tiling of
    // phase space around the reference endpoint.
    // -------------------------------------------------------------------------
    constexpr int n_perim = 24;  // points per box edge

    // Walk every leaf's perimeter in δ-space and stream the (x, y) image to
    // @p sink.  @p prefix is prepended verbatim to each line (used to attach
    // snapshot/method metadata in the multi-snapshot CSV).
    auto streamLeafBoundaries = [&]( const auto& tree, std::ostream& sink,
                                     std::string_view prefix ) {
        for ( int li : tree.doneLeaves() )
        {
            const auto& lf = tree.node( li ).leaf();
            auto emit = [&]( double dx, double dv, int seg ) {
                std::array< double, kD > local{ dx, 0.0, 0.0, dv };
                const double x_pred = lf.tte.state( 0 ).eval( local );
                const double y_pred = lf.tte.state( 1 ).eval( local );
                sink << prefix << li << ',' << seg << ',' << dx << ',' << dv << ','
                     << x_pred << ',' << y_pred << '\n';
            };
            for ( int i = 0; i < n_perim; ++i )
                emit( -1.0 + 2.0 * double( i ) / n_perim, -1.0, 0 );  // bottom
            for ( int i = 0; i < n_perim; ++i )
                emit( 1.0, -1.0 + 2.0 * double( i ) / n_perim, 1 );   // right
            for ( int i = 0; i < n_perim; ++i )
                emit( 1.0 - 2.0 * double( i ) / n_perim, 1.0, 2 );    // top
            for ( int i = 0; i < n_perim; ++i )
                emit( -1.0, 1.0 - 2.0 * double( i ) / n_perim, 3 );   // left
            emit( -1.0, -1.0, 4 );                                    // close loop
        }
    };

    auto dumpLeafBoundaries = [&]( const auto& tree, const char* path ) {
        std::ofstream out( path );
        out << "leaf,segment,delta_x,delta_vy,x,y\n";
        streamLeafBoundaries( tree, out, "" );
    };
    dumpLeafBoundaries( te_tree, "twoBody_te_leaf_orbit.csv" );
    dumpLeafBoundaries( lo_tree, "twoBody_lo_leaf_orbit.csv" );

    // -------------------------------------------------------------------------
    // Reference orbit of the central IC over a full period (for the "split
    // on orbit" plot backdrop).  The flow-map view uses tmax; the orbit
    // backdrop uses 2π so the ellipse is shown in full.
    // -------------------------------------------------------------------------
    {
        ode::Integrator< kN, Vec > ref_ig{
            kepler, ode::IntegratorConfig< double >{ .abstol = 1e-14 } };
        Vec x0_center;
        for ( int k = 0; k < kD; ++k ) x0_center( k ) = box.center[k];
        auto ref_sol = ref_ig.integrate( x0_center, 0.0, 2.0 * std::numbers::pi );
        std::ofstream ref( "twoBody_orbit_reference.csv" );
        ref << "t,x,y\n";
        for ( std::size_t i = 0; i < ref_sol.t.size(); ++i )
            ref << ref_sol.t[i] << ',' << ref_sol.x[i]( 0 ) << ','
                << ref_sol.x[i]( 1 ) << '\n';
        // Also record where the reference is at t = tmax so the plot can mark it.
        Vec x_at_tmax = ref_ig.integrate( x0_center, 0.0, tmax ).x.back();
        std::ofstream ep( "twoBody_orbit_endpoint.csv" );
        ep << "x,y\n" << x_at_tmax( 0 ) << ',' << x_at_tmax( 1 ) << '\n';
    }

    // -------------------------------------------------------------------------
    // Snapshots over one full orbital period.
    //
    // For each snapshot time t_k in (0, T_orbit] we run a fresh ADS for both
    // criteria and dump the pushed-forward leaf boundaries.  This gives a
    // movie-frame view of how the IC partition deforms (and how each method
    // refines its tessellation) as the orbit progresses.  Snapshot ADS runs
    // use a shallower max_depth to keep cumulative runtime bounded — the
    // headline analysis above already used the deeper budget.
    // -------------------------------------------------------------------------
    {
        constexpr int    n_snap   = 6;
        const double     T_period = 2.0 * std::numbers::pi;
        constexpr int    snap_depth = 6;

        ode::AdsIntegrator< kN, kP, kD > te_snap_ig{
            kepler, ode::AdsConfig{
                        .step_tol = 1e-14, .ads_tol = tol, .max_depth = snap_depth } };
        ode::LowOrderAdsIntegrator< kN, kP, kD > lo_snap_ig{
            kepler, ode::LowOrderAdsConfig{
                        .step_tol = 1e-14, .nli_tol = tol, .max_depth = snap_depth } };

        std::ofstream snaps( "twoBody_snapshots.csv" );
        snaps << "snapshot,t,method,leaf,segment,delta_x,delta_vy,x,y\n";

        std::ofstream snap_ic( "twoBody_snapshots_ic.csv" );
        snap_ic << "snapshot,t,method,leaf,x_lo,x_hi,vy_lo,vy_hi\n";

        std::ofstream snap_meta( "twoBody_snapshots_meta.csv" );
        snap_meta << "snapshot,t,te_leaves,lo_leaves\n";

        auto dumpIc = [&]( const auto& tree, int k_snap, double tk,
                           const char* tag ) {
            for ( int li : tree.doneLeaves() )
            {
                const auto& b   = tree.node( li ).leaf().box;
                const double xlo = b.center[0] - b.halfWidth[0];
                const double xhi = b.center[0] + b.halfWidth[0];
                const double vlo = b.center[3] - b.halfWidth[3];
                const double vhi = b.center[3] + b.halfWidth[3];
                snap_ic << k_snap << ',' << tk << ',' << tag << ',' << li << ','
                        << xlo << ',' << xhi << ',' << vlo << ',' << vhi << '\n';
            }
        };

        std::cout << "\nSnapshots (depth = " << snap_depth << "):\n";
        for ( int k = 1; k <= n_snap; ++k )
        {
            const double tk = T_period * double( k ) / double( n_snap );

            auto te_k = te_snap_ig.integrate( box, 0.0, tk );
            auto lo_k = lo_snap_ig.integrate( box, 0.0, tk );

            char buf[64];
            std::snprintf( buf, sizeof( buf ), "%d,%.10g,te,", k, tk );
            streamLeafBoundaries( te_k, snaps, buf );
            std::snprintf( buf, sizeof( buf ), "%d,%.10g,lo,", k, tk );
            streamLeafBoundaries( lo_k, snaps, buf );

            dumpIc( te_k, k, tk, "te" );
            dumpIc( lo_k, k, tk, "lo" );

            snap_meta << k << ',' << tk << ',' << te_k.numDone() << ','
                      << lo_k.numDone() << '\n';

            std::cout << "  t = " << tk << ":  te = " << te_k.numDone()
                      << " leaves,  lo = " << lo_k.numDone() << " leaves\n";
        }
    }

    // -------------------------------------------------------------------------
    // 3) Reference: scalar integrator at tight tolerance for every sample.
    //   1e-12 is comfortably below the 1e-3 comparison tolerance, so the
    //   reference dominates the error budget while staying tractable on a
    //   sample grid.
    // -------------------------------------------------------------------------
    ode::Integrator< kN, Vec > scalar_ig{
        kepler, ode::IntegratorConfig< double >{ .abstol = 1e-12 } };

    // -------------------------------------------------------------------------
    // Sample δ ∈ [-1, 1]^2 along the active IC dimensions (x and v_y).
    // -------------------------------------------------------------------------
    constexpr int n_grid = 11;
    std::ofstream csv( "twoBody_ads_comparison.csv" );
    csv << "delta_x,delta_vy,err_te,err_lo\n";

    double max_te = 0.0;
    double max_lo = 0.0;
    int    nan_te = 0;
    int    nan_lo = 0;

    for ( int i = 0; i < n_grid; ++i )
    {
        const double dx = -1.0 + 2.0 * double( i ) / double( n_grid - 1 );
        for ( int j = 0; j < n_grid; ++j )
        {
            const double dv = -1.0 + 2.0 * double( j ) / double( n_grid - 1 );

            Vec x0p;
            x0p << rp + box.halfWidth[0] * dx, 0.0, 0.0, vp + box.halfWidth[3] * dv;
            auto       sol = scalar_ig.integrate( x0p, 0.0, tmax );
            const auto truth = sol.x.back();

            const std::array< double, kD > q{ x0p( 0 ), x0p( 1 ), x0p( 2 ), x0p( 3 ) };
            const Vec te_pred = evalAtPoint( te_tree, q );
            const Vec lo_pred = evalAtPoint( lo_tree, q );

            const double err_te = ( truth - te_pred ).norm();
            const double err_lo = ( truth - lo_pred ).norm();

            if ( std::isnan( err_te ) )
                ++nan_te;
            else
                max_te = std::max( max_te, err_te );
            if ( std::isnan( err_lo ) )
                ++nan_lo;
            else
                max_lo = std::max( max_lo, err_lo );

            csv << dx << ',' << dv << ',' << err_te << ',' << err_lo << '\n';
        }
    }

    // -------------------------------------------------------------------------
    // Print a compact summary table.
    // -------------------------------------------------------------------------
    auto pad = []( const char* s, int w ) {
        std::cout << std::left << std::setw( w ) << s;
    };
    std::cout << std::setprecision( 4 ) << std::scientific;
    pad( "Method",                26 ); pad( "leaves", 10 ); pad( "splits", 10 );
    pad( "max endpoint error",    24 ); std::cout << "unresolved samples\n";
    std::cout << std::string( 80, '-' ) << '\n';
    pad( "AdsIntegrator (trunc)", 26 ); std::cout << std::setw( 10 ) << te_tree.numDone()
                                                  << std::setw( 10 ) << te_splits;
    std::cout << std::setw( 24 ) << max_te << nan_te << '\n';
    pad( "LowOrderAdsIntegrator", 26 ); std::cout << std::setw( 10 ) << lo_tree.numDone()
                                                  << std::setw( 10 ) << lo_splits;
    std::cout << std::setw( 24 ) << max_lo << nan_lo << '\n';

    std::cout << "\nWrote per-sample errors to twoBody_ads_comparison.csv\n";
    return 0;
}
