// =============================================================================
// examples/three_body/ads.cpp
//
// Step 2 — Automatic Domain Splitting on the L1 neighbourhood IC box
// (Earth-Moon planar CR3BP).
//
// For each of the kNSnaps snapshot times we run a fresh ADS
// propagation from t = 0 to t_snap using the truncation criterion. At
// each snapshot we dump every done leaf's (x, y) boundary image; the
// collection grows from a single leaf at small times to several leaves
// as the unstable manifold stretches the IC box into a long thin
// streak that one polynomial can no longer represent.
//
// Run:    ./three_body_ads
// Writes: cr3bp_ads.json
// =============================================================================

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <tax/ads.hpp>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <tax/ode/io.hpp>

#include "common.hpp"

int main()
{
    using namespace example::three_body;
    using namespace tax::ode::methods;

    constexpr int P = 6;
    constexpr int M = 4;
    constexpr int D = 4;

    constexpr int    kNSnaps   = 13;     // every 0.25 time units
    constexpr int    kNPerEdge = 24;
    constexpr double tFinal    = 3.0;

    auto ic_box = icBox();

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;
    cfg.max_steps           = 100000;

    const tax::ads::TruncationCriterion criterion{ /*tol=*/1e-4, /*maxDepth=*/8 };

    // Scalar centerpoint reference.
    auto ref_sol = tax::ode::propagate< /*Dense=*/true >(
        Verner89{}, rhs(), icCenter(), 0.0, tFinal, cfg );

    const auto times    = tax::ode::linspace( 0.0, tFinal, kNSnaps );
    const auto boundary = unitSquareBoundary( kNPerEdge );

    std::ofstream out( "cr3bp_ads.json" );
    out << std::setprecision( 14 );
    out << "{\n";
    out << "  \"method\": \"ads\",\n";
    out << "  \"problem\": \"planar_cr3bp_earth_moon\",\n";
    out << "  \"criterion\": { \"type\": \"truncation\", \"tol\": " << criterion.tol
        << ", \"maxDepth\": " << criterion.maxDepth << " },\n";
    out << "  \"config\": {\n";
    out << "    \"mu\":       " << kCR3BPMu << ",\n";
    out << "    \"x_L1\":     " << kCR3BPL1 << ",\n";
    out << "    \"earth_x\":  " << ( -kCR3BPMu )       << ",\n";
    out << "    \"moon_x\":   " << ( 1.0 - kCR3BPMu )  << ",\n";
    out << "    \"P\": " << P << ", \"M\": " << M << ", \"D\": " << D << ",\n";
    out << "    \"t_final\":  " << tFinal << ",\n";
    out << "    \"ic_box\": {\n";
    out << "      \"center\":    "; writeJsonArray( out, ic_box.center );    out << ",\n";
    out << "      \"halfWidth\": "; writeJsonArray( out, ic_box.halfWidth ); out << "\n";
    out << "    }\n";
    out << "  },\n";

    constexpr int kNRefSnaps = 200;
    const auto    ref_times  = tax::ode::linspace( 0.0, tFinal, kNRefSnaps );
    out << "  \"reference_orbit\": {\n";
    out << "    \"t\":  "; writeJsonArray( out, ref_times ); out << ",\n";
    std::vector< double > col( ref_times.size() );
    for ( int j = 0; j < D; ++j )
    {
        for ( std::size_t i = 0; i < ref_times.size(); ++i )
            col[ i ] = ref_sol( ref_times[ i ] )( j );
        out << "    \"x" << j << "\": "; writeJsonArray( out, col );
        out << ( j + 1 < D ? ",\n" : "\n" );
    }
    out << "  },\n";

    out << "  \"polygons\": [\n";
    std::vector< double > xs( boundary.size() ), ys( boundary.size() );
    std::vector< int >    leaves_per_snap;
    double                ads_total_ms = 0.0;
    for ( int s = 0; s < kNSnaps; ++s )
    {
        const double t_snap = times[ static_cast< std::size_t >( s ) ];
        out << "    { \"t\": " << t_snap << ", \"leaves\": [";

        if ( t_snap <= 0.0 )
        {
            for ( std::size_t v = 0; v < boundary.size(); ++v )
            {
                const tax::la::VecNT< M, double > d{
                    boundary[ v ][ 0 ], 0.0, 0.0, boundary[ v ][ 1 ]
                };
                const auto pt = ic_box.denormalize( d );
                xs[ v ] = pt( 0 );
                ys[ v ] = pt( 1 );
            }
            out << "\n      { \"id\": 0, \"depth\": 0, \"x\": ";
            writeJsonArray( out, xs );
            out << ", \"y\": ";
            writeJsonArray( out, ys );
            out << " }\n    ";
            leaves_per_snap.push_back( 1 );
        }
        else
        {
            const auto t_a   = std::chrono::high_resolution_clock::now();
            auto       tree  = tax::ads::propagate< P >(
                Verner89{}, criterion, rhs(),
                ic_box, icCenter(), 0.0, t_snap, cfg );
            const auto t_b   = std::chrono::high_resolution_clock::now();
            ads_total_ms += std::chrono::duration< double, std::milli >( t_b - t_a ).count();

            bool first = true;
            for ( int li : tree.done() )
            {
                const auto& leaf = tree.leaf( li );
                for ( std::size_t v = 0; v < boundary.size(); ++v )
                {
                    const auto d = boundaryToBox( boundary[ v ][ 0 ], boundary[ v ][ 1 ] );
                    xs[ v ] = leaf.payload( 0 ).eval( d );
                    ys[ v ] = leaf.payload( 1 ).eval( d );
                }
                if ( !first ) out << ",";
                first = false;
                out << "\n      { \"id\": " << li << ", \"depth\": " << leaf.depth
                    << ", \"x\": "; writeJsonArray( out, xs );
                out << ", \"y\": "; writeJsonArray( out, ys );
                out << " }";
            }
            out << "\n    ";
            leaves_per_snap.push_back( static_cast< int >( tree.done().size() ) );
        }

        out << "] }" << ( s + 1 < kNSnaps ? "," : "" ) << "\n";
    }
    out << "  ],\n";
    out << "  \"timing\": { \"elapsed_ms\": " << ads_total_ms << " }\n";
    out << "}\n";

    std::string leaves_str;
    for ( std::size_t i = 0; i < leaves_per_snap.size(); ++i )
    {
        if ( i ) leaves_str += ", ";
        leaves_str += std::to_string( leaves_per_snap[ i ] );
    }
    const std::vector< std::pair< std::string, std::string > > rows{
        { "P, M, D",         std::to_string( P ) + ", " + std::to_string( M ) + ", " + std::to_string( D ) },
        { "criterion",       "truncation (tol=1e-4, depth<=8)" },
        { "t_final",         std::to_string( tFinal ) },
        { "snapshots",       std::to_string( kNSnaps ) },
        { "leaves per snap", leaves_str },
        { "elapsed",         std::to_string( ads_total_ms / 1e3 ) + " s" },
        { "output",          "cr3bp_ads.json" }
    };
    printBanner( "CR3BP ads (piecewise polynomial flow)",
                 std::span< const std::pair< std::string, std::string > >{ rows } );
    return 0;
}
