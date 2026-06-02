// =============================================================================
// examples/three_body/taylor.cpp
//
// Step 1 — Single multivariate Taylor flow polynomial over the L1
// neighbourhood IC box (Earth-Moon planar CR3BP).
//
// We propagate one DA-valued state Vec<4, TE<P, M>> built from icBox()
// with Dense=true. At kNSnaps evenly spaced times in [0, t_final] we
// evaluate the (x, y) components of the flow polynomial along the IC
// box boundary; the resulting closed polygons show how the box is
// sheared along the unstable manifold direction.
//
// Run:    ./three_body_taylor
// Writes: cr3bp_taylor.json
// =============================================================================

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <tax/ads/box.hpp>
#include <tax/ads/da_state.hpp>
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

    using TE      = tax::TE< P, M >;
    using DAState = tax::la::VecNT< D, TE >;

    constexpr int    kNSnaps   = 9;
    constexpr int    kNPerEdge = 24;
    constexpr double tFinal    = 3.0;     // ~1.1 Lyapunov times

    auto ic_box = icBox();

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;
    cfg.max_steps           = 100000;

    DAState    x0_da   = tax::ads::create< P, M >( ic_box, icCenter() );
    const auto t_start = std::chrono::high_resolution_clock::now();
    auto       sol     = tax::ode::propagate< /*Dense=*/true >(
        Verner89{}, rhs(), x0_da, 0.0, tFinal, cfg );
    const auto t_end   = std::chrono::high_resolution_clock::now();
    const double elapsed_ms =
        std::chrono::duration< double, std::milli >( t_end - t_start ).count();

    // Scalar centerpoint reference orbit.
    auto ref_sol = tax::ode::propagate< /*Dense=*/true >(
        Verner89{}, rhs(), icCenter(), 0.0, tFinal, cfg );

    const auto times    = tax::ode::linspace( 0.0, tFinal, kNSnaps );
    const auto boundary = unitSquareBoundary( kNPerEdge );

    std::ofstream out( "cr3bp_taylor.json" );
    out << std::setprecision( 14 );
    out << "{\n";
    out << "  \"method\": \"taylor\",\n";
    out << "  \"problem\": \"planar_cr3bp_earth_moon\",\n";
    out << "  \"config\": {\n";
    out << "    \"mu\":       " << kCR3BPMu << ",\n";
    out << "    \"x_L1\":     " << kCR3BPL1 << ",\n";
    out << "    \"earth_x\":  " << ( -kCR3BPMu )    << ",\n";
    out << "    \"moon_x\":   " << ( 1.0 - kCR3BPMu )<< ",\n";
    out << "    \"P\": "        << P << ", \"M\": " << M << ", \"D\": " << D << ",\n";
    out << "    \"t_final\":  " << tFinal << ",\n";
    out << "    \"lambda_unstable\": " << linL1().lambda_unstable << ",\n";
    out << "    \"v_unstable\":      ["
        << linL1().v_unstable( 0 ) << ", " << linL1().v_unstable( 1 ) << ", "
        << linL1().v_unstable( 2 ) << ", " << linL1().v_unstable( 3 ) << "],\n";
    out << "    \"ic_box\": {\n";
    out << "      \"center\":    "; writeJsonArray( out, ic_box.center );    out << ",\n";
    out << "      \"halfWidth\": "; writeJsonArray( out, ic_box.halfWidth ); out << "\n";
    out << "    }\n";
    out << "  },\n";
    out << "  \"timing\": { \"elapsed_ms\": " << elapsed_ms << " },\n";

    // Reference orbit (200 samples for the underlay).
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
    for ( int s = 0; s < kNSnaps; ++s )
    {
        const double t      = times[ static_cast< std::size_t >( s ) ];
        const auto   x_at_t = sol( t );
        for ( std::size_t v = 0; v < boundary.size(); ++v )
        {
            const auto d = boundaryToBox( boundary[ v ][ 0 ], boundary[ v ][ 1 ] );
            xs[ v ] = x_at_t( 0 ).eval( d );
            ys[ v ] = x_at_t( 1 ).eval( d );
        }
        out << "    { \"t\": " << t << ", \"x\": ";
        writeJsonArray( out, xs );
        out << ", \"y\": ";
        writeJsonArray( out, ys );
        out << " }" << ( s + 1 < kNSnaps ? "," : "" ) << "\n";
    }
    out << "  ]\n";
    out << "}\n";

    const std::vector< std::pair< std::string, std::string > > rows{
        { "P, M, D",         std::to_string( P ) + ", " + std::to_string( M ) + ", " + std::to_string( D ) },
        { "t_final",         std::to_string( tFinal )   },
        { "lambda_unstable", std::to_string( linL1().lambda_unstable ) },
        { "snapshots",       std::to_string( kNSnaps )  },
        { "elapsed",         std::to_string( elapsed_ms / 1e3 ) + " s" },
        { "output",          "cr3bp_taylor.json" }
    };
    printBanner( "CR3BP taylor (single Taylor flow polynomial)",
                 std::span< const std::pair< std::string, std::string > >{ rows } );
    return 0;
}
