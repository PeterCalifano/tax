// =============================================================================
// examples/three_body/loads.cpp
//
// Step 3 — Low-Order Automatic Domain Splitting (LOADS) on the L1
// neighbourhood IC box. Same structure as ads.cpp but with the
// nonlinearity-index criterion (Losacco/Fossà/Armellin) instead of
// the truncation criterion.
//
// Run:    ./three_body_loads
// Writes: cr3bp_loads.json
// =============================================================================

#include <chrono>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <iostream>
#include <thread>

#include <tax/ads.hpp>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <tax/ode/io.hpp>

#include "common.hpp"

int main()
{
    using namespace example::three_body;
    using namespace tax::ode::methods;

    constexpr int P = 2;
    constexpr int M = 4;
    constexpr int D = 4;

    constexpr int    kNSnaps   = 13;     // every 0.25 time units
    constexpr int    kNPerEdge = 24;
    constexpr double tFinal    = 3.0;

    auto ic_box = icBox();

    const int kThreads = [] {
        if ( const char* e = std::getenv( "TAX_ADS_THREADS" ) )
        {
            const int n = std::atoi( e );
            if ( n > 0 ) return n;
        }
        const unsigned hc = std::thread::hardware_concurrency();
        return hc > 0 ? static_cast< int >( hc ) : 1;
    }();

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;
    cfg.max_steps           = 100000;

    const tax::ads::NliCriterion criterion{ /*tol=*/0.3, /*maxDepth=*/12 };

    auto ref_sol = tax::ode::propagate< /*Dense=*/true >(
        Verner89{}, rhs(), icCenter(), 0.0, tFinal, cfg );

    const auto times    = tax::ode::linspace( 0.0, tFinal, kNSnaps );
    const auto boundary = unitSquareBoundary( kNPerEdge );

    std::ofstream out( "cr3bp_loads.json" );
    out << std::setprecision( 14 );
    out << "{\n";
    out << "  \"method\": \"loads\",\n";
    out << "  \"problem\": \"planar_cr3bp_earth_moon\",\n";
    out << "  \"criterion\": { \"type\": \"nli\", \"tol\": " << criterion.tol
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

    std::cout << "[loads] running per-snapshot LOADS propagations..." << std::flush;

    out << "  \"polygons\": [\n";
    std::vector< double > xs( boundary.size() ), ys( boundary.size() );
    std::vector< int >    leaves_per_snap;
    double                total_ms = 0.0;
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
                ic_box, icCenter(), 0.0, t_snap, cfg, kThreads );
            const auto t_b   = std::chrono::high_resolution_clock::now();
            total_ms += std::chrono::duration< double, std::milli >( t_b - t_a ).count();

            bool first = true;
            int  rank  = 0;
            for ( int li : tree.done() )
            {
                const auto& leaf = tree.leaf( li );
                const int   id   = rank++;
                for ( std::size_t v = 0; v < boundary.size(); ++v )
                {
                    const auto d = boundaryToBox( boundary[ v ][ 0 ], boundary[ v ][ 1 ] );
                    xs[ v ] = leaf.payload( 0 ).eval( d );
                    ys[ v ] = leaf.payload( 1 ).eval( d );
                }
                if ( !first ) out << ",";
                first = false;
                out << "\n      { \"id\": " << id << ", \"depth\": " << leaf.depth
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
    out << "  \"timing\": { \"elapsed_ms\": " << total_ms << " }\n";
    out << "}\n";

    std::cout << "\r" << std::string( 50, ' ' ) << "\r";

    std::string leaves_str;
    for ( std::size_t i = 0; i < leaves_per_snap.size(); ++i )
    {
        if ( i ) leaves_str += ", ";
        leaves_str += std::to_string( leaves_per_snap[ i ] );
    }
    const std::vector< std::pair< std::string, std::string > > rows{
        { "P, M, D",         std::to_string( P ) + ", " + std::to_string( M ) + ", " + std::to_string( D ) },
        { "criterion",       "nli (tol=0.3, depth<=6)" },
        { "t_final",         std::to_string( tFinal ) },
        { "snapshots",       std::to_string( kNSnaps ) },
        { "leaves per snap", leaves_str },
        { "elapsed",         std::to_string( total_ms / 1e3 ) + " s" },
        { "output",          "cr3bp_loads.json" }
    };
    printBanner( "CR3BP loads (NLI piecewise polynomial flow)",
                 std::span< const std::pair< std::string, std::string > >{ rows } );
    return 0;
}
