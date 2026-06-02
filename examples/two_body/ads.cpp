// =============================================================================
// examples/two_body/ads.cpp
//
// Step 2 — Automatic Domain Splitting on the IC box.
//
// A single Taylor flow polynomial (taylor.cpp) becomes inaccurate as the
// IC box deforms. ADS (Wittig 2015) subdivides the box whenever the
// polynomial's top-degree residual exceeds a tolerance. Each leaf of the
// resulting tree carries its own polynomial flow map on a sub-domain of
// the original box.
//
// For the box-evolution figure, we run ADS independently to each of 8
// snapshot times (every 45 degrees after t=0). The first snapshot is
// just the IC box; later snapshots show progressively finer partitions
// as nonlinearity grows.
//
// Run:    ./two_body_ads
// Writes: ads.json
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

    // ---- Problem dimensions -------------------------------------------------
    constexpr int P = 6;
    constexpr int M = 4;
    constexpr int D = 4;

    // ---- Time grid + boundary sampling -------------------------------------
    constexpr int    kNOrbits  = 1;
    constexpr int    kNSnaps   = 9;       // including t=0 (trivial polygon)
    constexpr int    kNPerEdge = 24;
    const     double tFinal    = kNOrbits * kPeriod;

    // ---- IC box (only y and vy vary) ---------------------------------------
    tax::ads::Box< double, M > ic_box{
        icCenter(),
        tax::la::VecNT< M, double >{ 0.0, 5e-3, 0.0, 8e-3 }
    };

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    // Truncation criterion (Wittig): split when sum |coeff(alpha)| over
    // alpha at total degree P exceeds `tol`.
    const tax::ads::TruncationCriterion criterion{ /*tol=*/1e-4, /*maxDepth=*/8 };

    // ---- Scalar centerpoint reference --------------------------------------
    tax::ode::IntegratorConfig< double > ref_cfg;
    ref_cfg.abstol = ref_cfg.reltol = 1e-13;
    auto ref_sol = tax::ode::propagate< /*Dense=*/true >(
        Taylor< 16 >{}, rhs(), icCenter(), 0.0, tFinal, ref_cfg );

    const auto times    = tax::ode::linspace( 0.0, tFinal, kNSnaps );
    const auto boundary = unitSquareBoundary( kNPerEdge );

    // ---- JSON: open --------------------------------------------------------
    std::ofstream out( "ads.json" );
    out << std::setprecision( 12 );
    out << "{\n";
    out << "  \"method\": \"ads\",\n";
    out << "  \"criterion\": { \"type\": \"truncation\", \"tol\": " << criterion.tol
        << ", \"maxDepth\": " << criterion.maxDepth << " },\n";
    out << "  \"config\": {\n";
    out << "    \"P\": " << P << ", \"M\": " << M << ", \"D\": " << D << ",\n";
    out << "    \"n_orbits\": " << kNOrbits << ", \"t_final\": " << tFinal << ",\n";
    out << "    \"ic_box\": {\n";
    out << "      \"center\":    "; writeJsonArray( out, ic_box.center );    out << ",\n";
    out << "      \"halfWidth\": "; writeJsonArray( out, ic_box.halfWidth ); out << "\n";
    out << "    }\n";
    out << "  },\n";

    //   reference orbit
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

    // ---- One ADS run per snapshot ------------------------------------------
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
            // Trivial: one polygon equal to the IC box image (the box itself).
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
            ads_total_ms += std::chrono::duration< double, std::milli >( t_b - t_a ).count();

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

    out << "  \"timing\": { \"elapsed_ms\": " << ads_total_ms << " }\n";
    out << "}\n";

    // ---- Terminal banner ---------------------------------------------------
    std::string leaves_str;
    for ( std::size_t i = 0; i < leaves_per_snap.size(); ++i )
    {
        if ( i ) leaves_str += ", ";
        leaves_str += std::to_string( leaves_per_snap[ i ] );
    }

    const std::vector< std::pair< std::string, std::string > > rows{
        { "P, M, D",         std::to_string( P ) + ", " + std::to_string( M ) + ", " + std::to_string( D ) },
        { "criterion",       "truncation (tol=1e-4, depth<=8)" },
        { "orbits",          std::to_string( kNOrbits ) },
        { "snapshots",       std::to_string( kNSnaps ) },
        { "leaves per snap", leaves_str },
        { "elapsed",         std::to_string( ads_total_ms / 1e3 ) + " s" },
        { "output",          "ads.json" }
    };
    printBanner( "ads (piecewise polynomial flow)",
                 std::span< const std::pair< std::string, std::string > >{ rows } );
    return 0;
}
