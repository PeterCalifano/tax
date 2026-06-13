// =============================================================================
// examples/two_body/refine.cpp
//
// Step 4 — "Propagate-then-assess" ADS refinement (tax::ads::refine).
//
// Where ads.cpp splits a box mid-flight the instant its flow polynomial
// stops converging, refine() instead carries *every* box all the way to the
// final time and only then judges its quality by bisecting it, propagating
// both halves to the end as well, and comparing the result. Here the verdict
// is the CoefficientMatchCriterion: re-identify the parent map on each half
// and check it reproduces the independently propagated child to a relative
// tolerance. (AreaRatioCriterion — the parent-vs-children image-area ratio —
// is a drop-in alternative; see the tutorial for the trade-off.) Because no
// box ever needs another box's partial state, the whole refinement fans out
// in parallel.
//
// This driver runs the refinement at increasing depth caps k = 0, 1, 2, ...
// Iteration 0 is the single box; each iteration roughly doubles the number
// of sub-boxes until the partition converges. For every iteration we record
// the box images at a sweep of snapshot times (for the animation) and the
// RMS error of the piecewise-polynomial prediction against a Monte-Carlo
// reference cloud (to show that more boxes ⇒ better matching).
//
// Here the IC box varies the initial *position* (x, y) — two DA variables —
// while velocity is pinned at the periapsis value.
//
// Run:    ./two_body_refine
// Writes: refine.json   (animate with examples/two_body/plot_refine.py)
// =============================================================================

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <tax/ads.hpp>
#include <tax/ode.hpp>
#include <tax/ode/io.hpp>
#include <vector>

#include "common.hpp"

namespace
{
using namespace example;
using namespace example::two_body;
using namespace tax::ode::methods;

constexpr int P = 6;  // DA truncation order
constexpr int M = 2;  // DA variables: initial (x, y) position
constexpr int D = 4;  // state dimension (x, y, vx, vy)

using TE = tax::TE< P, M >;
using DAState = tax::la::VecNT< D, TE >;
using BoxT = tax::ads::Box< double, M >;

// IC box: vary the initial position only (axes 0 and 1).
constexpr double kHx = 0.06;
constexpr double kHy = 0.06;

BoxT positionBox() { return BoxT{ { kPeriapsis, 0.0 }, { kHx, kHy } }; }

// The two boundary coordinates map straight onto the two DA variables.
std::array< double, M > toBox( double a, double b ) { return { a, b }; }

// Reconstruct a leaf's identity (t0) DA state from its sub-box: the active
// axes carry the (shifted) center and (halved) half-width; the pinned axes
// hold the reference IC. Re-propagating this densely recovers the leaf's
// flow map at every time, not just the final one.
DAState leafInit( const BoxT& box )
{
    const auto center = icCenter();
    DAState s;
    for ( int i = 0; i < D; ++i )
    {
        TE c{};
        c[0] = ( i < M ) ? box.center( i ) : center( i );
        if ( i < M )
        {
            tax::MultiIndex< M > alpha{};
            alpha[static_cast< std::size_t >( i )] = 1;
            c[tax::flatIndex< M >( alpha )] = box.halfWidth( i );
        }
        s( i ) = std::move( c );
    }
    return s;
}

// Shoelace area of a closed polygon.
double polygonArea( const Polygon& p )
{
    double twice = 0.0;
    const std::size_t n = p.x.size();
    for ( std::size_t i = 0; i + 1 < n; ++i ) twice += p.x[i] * p.y[i + 1] - p.x[i + 1] * p.y[i];
    return 0.5 * std::abs( twice );
}

struct McSample
{
    double ic_x, ic_y;              // initial position in the box
    std::vector< double > truth_x;  // (x, y) along the snapshot times
    std::vector< double > truth_y;
};

struct Iteration
{
    int max_depth = 0;
    int n_boxes = 0;
    double area = 0.0;  // total covered (x, y) area at the final time
    double rms = 0.0;   // RMS prediction error vs Monte Carlo at the final time
    std::vector< Snapshot > snapshots;
};
}  // namespace

int main()
{
    constexpr int kNSnaps = 13;  // animation frames per iteration
    constexpr int kNPerEdge = 20;
    constexpr int kMaxIter = 6;
    constexpr int kNMonte = 350;
    const double t_final = kPeriod;

    const BoxT box = positionBox();

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    const auto snap_times = tax::ode::linspace( 0.0, t_final, kNSnaps );
    const auto boundary = unitSquareBoundary( kNPerEdge );

    // ---- Scalar centerpoint orbit (plot underlay) ----------------------------
    auto ref_sol = tax::ode::propagate< /*Dense=*/true >( Taylor< 16 >{}, rhs(), icCenter(), 0.0,
                                                          t_final, cfg );
    const auto reference = sampleOrbit( ref_sol, tax::ode::linspace( 0.0, t_final, 200 ), D );

    // ---- Monte-Carlo reference cloud -----------------------------------------
    // Uniform samples of the IC box, each propagated densely so we can show
    // the true set at every snapshot time and score each iteration against it.
    std::mt19937 rng( 12345u );
    std::uniform_real_distribution< double > unit( -1.0, 1.0 );
    std::vector< McSample > monte;
    monte.reserve( kNMonte );
    for ( int s = 0; s < kNMonte; ++s )
    {
        const double u = unit( rng );
        const double v = unit( rng );
        tax::la::VecNT< D, double > ic = icCenter();
        ic( 0 ) += kHx * u;
        ic( 1 ) += kHy * v;
        auto sol =
            tax::ode::propagate< /*Dense=*/true >( Verner89{}, rhs(), ic, 0.0, t_final, cfg );

        McSample m;
        m.ic_x = ic( 0 );
        m.ic_y = ic( 1 );
        m.truth_x.reserve( snap_times.size() );
        m.truth_y.reserve( snap_times.size() );
        for ( double t : snap_times )
        {
            const auto x = sol( t );
            m.truth_x.push_back( x( 0 ) );
            m.truth_y.push_back( x( 1 ) );
        }
        monte.push_back( std::move( m ) );
    }

    // ---- Refinement sweep: increasing depth cap ------------------------------
    Stopwatch clock;
    std::vector< Iteration > iters;
    std::string box_counts;
    for ( int k = 0; k <= kMaxIter; ++k )
    {
        // Swap in tax::ads::AreaRatioCriterion{ 0.01, k, 0, 1, kNPerEdge } to
        // drive refinement by the parent-vs-children image-area ratio instead.
        const tax::ads::CoefficientMatchCriterion crit{ /*tol=*/2e-3, /*maxDepth=*/k };
        auto tree = tax::ads::refine< P >( Verner89{}, crit, rhs(), box, icCenter(), 0.0, t_final,
                                           cfg, adsThreads() );

        Iteration it;
        it.max_depth = k;
        it.snapshots.assign( snap_times.size(), Snapshot{} );
        for ( std::size_t si = 0; si < snap_times.size(); ++si )
            it.snapshots[si].t = snap_times[si];

        int id = 0;
        for ( int li : tree.done() )
        {
            const auto& leaf = tree.leaf( li );
            auto sol = tax::ode::propagate< /*Dense=*/true >(
                Verner89{}, rhs(), leafInit( leaf.box ), 0.0, t_final, cfg );
            for ( std::size_t si = 0; si < snap_times.size(); ++si )
            {
                auto poly = evalPolygon( sol( snap_times[si] ), boundary, toBox, id, leaf.depth );
                if ( si + 1 == snap_times.size() ) it.area += polygonArea( poly );
                it.snapshots[si].leaves.push_back( std::move( poly ) );
            }
            ++id;
        }
        it.n_boxes = id;

        // RMS error of the piecewise-polynomial prediction at the final time.
        double sq = 0.0;
        int counted = 0;
        const std::size_t last = snap_times.size() - 1;
        for ( const auto& m : monte )
        {
            tax::la::VecNT< M, double > pt;
            pt << m.ic_x, m.ic_y;
            auto idx = tree.leaf( pt );
            if ( !idx.has_value() ) continue;
            const auto& leaf = tree.leaf( *idx );
            std::array< double, M > local{};
            for ( int j = 0; j < M; ++j )
                local[static_cast< std::size_t >( j )] =
                    ( pt( j ) - leaf.box.center( j ) ) / leaf.box.halfWidth( j );
            const double dx = leaf.payload( 0 ).eval( local ) - m.truth_x[last];
            const double dy = leaf.payload( 1 ).eval( local ) - m.truth_y[last];
            sq += dx * dx + dy * dy;
            ++counted;
        }
        it.rms = counted > 0 ? std::sqrt( sq / counted ) : 0.0;

        box_counts += ( box_counts.empty() ? "" : ", " ) + std::to_string( it.n_boxes );
        iters.push_back( std::move( it ) );
    }
    const double elapsed_ms = clock.ms();

    // ---- Write JSON (custom nested schema: iterations -> snapshots -> leaves) -
    std::ofstream out( "refine.json" );
    out << std::setprecision( 14 );
    out << "{\n  \"method\": \"refine\",\n";
    out << "  \"params\": {\n";
    out << "    \"P\": " << P << ", \"M\": " << M << ", \"D\": " << D << ",\n";
    out << "    \"t_final\": " << jsonNumber( t_final ) << ", \"ecc\": " << jsonNumber( kEcc )
        << ",\n";
    out << "    \"criterion\": \"coefficient_match\", \"tol\": 0.002,\n";
    out << "    \"ic_center\": " << jsonArray( box.center )
        << ", \"ic_half_width\": " << jsonArray( box.halfWidth ) << ",\n";
    out << "    \"n_monte\": " << kNMonte << "\n  },\n";
    out << "  \"timing\": { \"elapsed_ms\": " << elapsed_ms << " },\n";

    out << "  \"reference_orbit\": {\n    \"t\": ";
    writeJsonArray( out, reference.t );
    out << ",\n    \"x0\": ";
    writeJsonArray( out, reference.cols[0] );
    out << ",\n    \"x1\": ";
    writeJsonArray( out, reference.cols[1] );
    out << "\n  },\n";

    out << "  \"snap_times\": ";
    writeJsonArray( out, snap_times );
    out << ",\n";

    // Monte-Carlo cloud per snapshot time.
    out << "  \"monte_carlo\": [\n";
    for ( std::size_t si = 0; si < snap_times.size(); ++si )
    {
        std::vector< double > xs, ys;
        xs.reserve( monte.size() );
        ys.reserve( monte.size() );
        for ( const auto& m : monte )
        {
            xs.push_back( m.truth_x[si] );
            ys.push_back( m.truth_y[si] );
        }
        out << "    { \"t\": " << snap_times[si] << ", \"x\": ";
        writeJsonArray( out, xs );
        out << ", \"y\": ";
        writeJsonArray( out, ys );
        out << " }" << ( si + 1 < snap_times.size() ? "," : "" ) << "\n";
    }
    out << "  ],\n";

    // Iterations.
    out << "  \"iterations\": [\n";
    for ( std::size_t ki = 0; ki < iters.size(); ++ki )
    {
        const auto& it = iters[ki];
        out << "    { \"iter\": " << it.max_depth << ", \"n_boxes\": " << it.n_boxes
            << ", \"area\": " << jsonNumber( it.area ) << ", \"rms\": " << jsonNumber( it.rms )
            << ", \"snapshots\": [\n";
        for ( std::size_t si = 0; si < it.snapshots.size(); ++si )
        {
            const auto& snap = it.snapshots[si];
            out << "      { \"t\": " << snap.t << ", \"leaves\": [";
            for ( std::size_t l = 0; l < snap.leaves.size(); ++l )
            {
                out << "\n        { \"id\": " << snap.leaves[l].id
                    << ", \"depth\": " << snap.leaves[l].depth << ", \"x\": ";
                writeJsonArray( out, snap.leaves[l].x );
                out << ", \"y\": ";
                writeJsonArray( out, snap.leaves[l].y );
                out << " }" << ( l + 1 < snap.leaves.size() ? "," : "" );
            }
            out << "\n      ] }" << ( si + 1 < it.snapshots.size() ? "," : "" ) << "\n";
        }
        out << "    ] }" << ( ki + 1 < iters.size() ? "," : "" ) << "\n";
    }
    out << "  ]\n}\n";

    printBanner( "two_body/refine — propagate-then-assess ADS (coefficient-match criterion)",
                 { { "P, M", std::to_string( P ) + ", " + std::to_string( M ) },
                   { "iterations", std::to_string( kMaxIter + 1 ) },
                   { "boxes per iter", box_counts },
                   { "final RMS", jsonNumber( iters.back().rms ) },
                   { "elapsed", std::to_string( elapsed_ms ) + " ms" },
                   { "output", "refine.json" } } );
    return 0;
}
