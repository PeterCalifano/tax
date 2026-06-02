// =============================================================================
// examples/two_body/loads.cpp
//
// Step 3 — Low-Order Automatic Domain Splitting (LOADS).
//
// Identical structure to ads.cpp; the only change is the splitting
// criterion. LOADS (Losacco/Fossà/Armellin 2024) splits on the
// "nonlinearity index" — a ratio between the L1 mass of the degree-≥2
// Jacobian-variation bound and the L1 mass of the linear Jacobian. NLI
// is more sensitive to swirl-type nonlinearities (think: periapsis
// passages) than the truncation residual is.
//
// Run:    ./two_body_loads
// Writes: loads.json
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
    using namespace example::two_body;
    using namespace tax::ode::methods;

    constexpr int P = 6;
    constexpr int M = 4;
    constexpr int D = 4;

    constexpr int    kNOrbits  = 1;
    constexpr int    kNSnaps   = 9;
    constexpr int    kNPerEdge = 24;
    const     double tFinal    = kNOrbits * kPeriod;

    // ---- IC box (configured in common.hpp) ---------------------------------
    auto ic_box = icBox();

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    const tax::ads::NliCriterion criterion{ /*tol=*/1, /*maxDepth=*/6 };

    tax::ode::IntegratorConfig< double > ref_cfg;
    ref_cfg.abstol = ref_cfg.reltol = 1e-13;
    auto ref_sol = tax::ode::propagate< /*Dense=*/true >(
        Taylor< 16 >{}, rhs(), icCenter(), 0.0, tFinal, ref_cfg );

    const auto times    = tax::ode::linspace( 0.0, tFinal, kNSnaps );
    const auto boundary = unitSquareBoundary( kNPerEdge );

    std::ofstream out( "loads.json" );
    out << std::setprecision( 12 );
    out << "{\n";
    out << "  \"method\": \"loads\",\n";
    out << "  \"criterion\": { \"type\": \"nli\", \"tol\": " << criterion.tol
        << ", \"maxDepth\": " << criterion.maxDepth << " },\n";
    out << "  \"config\": {\n";
    out << "    \"P\": " << P << ", \"M\": " << M << ", \"D\": " << D << ",\n";
    out << "    \"n_orbits\": " << kNOrbits << ", \"t_final\": " << tFinal << ",\n";
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

    std::cout << "[loads] running 9 per-snapshot LOADS propagations..." << std::flush;

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
                    0.0, boundary[ v ][ 0 ], 0.0, boundary[ v ][ 1 ]
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
                Verner89{}, criterion, rhs(), ic_box, icCenter(), 0.0, t_snap, cfg );
            const auto t_b   = std::chrono::high_resolution_clock::now();
            total_ms += std::chrono::duration< double, std::milli >( t_b - t_a ).count();

            bool first = true;
            for ( int li : tree.done() )
            {
                const auto& leaf = tree.leaf( li );
                for ( std::size_t v = 0; v < boundary.size(); ++v )
                {
                    const std::array< double, M > d{
                        0.0, boundary[ v ][ 0 ], 0.0, boundary[ v ][ 1 ]
                    };
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
        { "criterion",       "nli (tol=0.1, depth<=6)" },
        { "orbits",          std::to_string( kNOrbits ) },
        { "snapshots",       std::to_string( kNSnaps ) },
        { "leaves per snap", leaves_str },
        { "elapsed",         std::to_string( total_ms / 1e3 ) + " s" },
        { "output",          "loads.json" }
    };
    printBanner( "loads (piecewise polynomial, NLI criterion)",
                 std::span< const std::pair< std::string, std::string > >{ rows } );
    return 0;
}
