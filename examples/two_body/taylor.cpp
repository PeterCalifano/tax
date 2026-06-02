// =============================================================================
// examples/two_body/taylor.cpp
//
// Step 1 — One multivariate Taylor flow polynomial over the IC box.
//
// We build a DA-valued state Vec<4, TE<P, M>> = ic_center + halfWidth * xi
// on the initial-condition box and integrate it once with Dense=true.
// The result is a single polynomial flow map: at any time t, sol(t) is a
// Vec<4, TE> whose value at xi is the propagated state of the IC point
// ic_center + halfWidth * xi.
//
// At 9 snapshot times spaced every 45 degrees along the first orbit, we
// evaluate the (x, y) components on the boundary of the IC box's
// (y, vy)-face. The resulting closed polygons show how the IC box
// distorts under a single flow polynomial — gradually banana-shaped as
// the orbit wraps around.
//
// Run:    ./two_body_taylor
// Writes: taylor.json
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
    using namespace example::two_body;
    using namespace tax::ode::methods;

    // ---- Problem dimensions -------------------------------------------------
    constexpr int P = 6;     // DA truncation order
    constexpr int M = 4;     // IC dimension
    constexpr int D = 4;     // state dimension

    using TE      = tax::TE< P, M >;
    using DAState = tax::la::VecNT< D, TE >;

    // ---- Time grid + boundary sampling -------------------------------------
    constexpr int    kNOrbits  = 1;        // one revolution
    constexpr int    kNSnaps   = 9;        // 0, pi/4, ..., 2 pi (every 45 deg)
    constexpr int    kNPerEdge = 24;       // boundary samples per square edge
    const     double tFinal    = kNOrbits * kPeriod;

    // ---- IC box (configured in common.hpp) ---------------------------------
    auto ic_box = icBox();

    // ---- Stepper config -----------------------------------------------------
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    // ---- One dense DA propagation over [0, tFinal] -------------------------
    DAState    x0_da   = tax::ads::create< P, M >( ic_box, icCenter() );
    const auto t_start = std::chrono::high_resolution_clock::now();
    auto       sol     = tax::ode::propagate< /*Dense=*/true >(
        Verner89{}, rhs(), x0_da, 0.0, tFinal, cfg );
    const auto t_end   = std::chrono::high_resolution_clock::now();
    const double elapsed_ms =
        std::chrono::duration< double, std::milli >( t_end - t_start ).count();

    // ---- Scalar centerpoint reference orbit (for the plot underlay) --------
    tax::ode::IntegratorConfig< double > ref_cfg;
    ref_cfg.abstol = ref_cfg.reltol = 1e-13;
    auto ref_sol = tax::ode::propagate< /*Dense=*/true >(
        Taylor< 16 >{}, rhs(), icCenter(), 0.0, tFinal, ref_cfg );

    const auto times    = tax::ode::linspace( 0.0, tFinal, kNSnaps );
    const auto boundary = unitSquareBoundary( kNPerEdge );

    // ---- JSON output --------------------------------------------------------
    std::ofstream out( "taylor.json" );
    out << std::setprecision( 12 );
    out << "{\n";
    out << "  \"method\": \"taylor\",\n";
    out << "  \"config\": {\n";
    out << "    \"P\": " << P << ", \"M\": " << M << ", \"D\": " << D << ",\n";
    out << "    \"n_orbits\": " << kNOrbits << ", \"t_final\": " << tFinal << ",\n";
    out << "    \"ic_box\": {\n";
    out << "      \"center\":    "; writeJsonArray( out, ic_box.center );    out << ",\n";
    out << "      \"halfWidth\": "; writeJsonArray( out, ic_box.halfWidth ); out << "\n";
    out << "    }\n";
    out << "  },\n";
    out << "  \"timing\": { \"elapsed_ms\": " << elapsed_ms << " },\n";

    //   reference orbit (centerpoint trajectory)
    constexpr int kNRefSnaps = 200;
    const auto    ref_times  = tax::ode::linspace( 0.0, tFinal, kNRefSnaps );
    out << "  \"reference_orbit\": {\n";
    out << "    \"t\":  "; writeJsonArray( out, ref_times );    out << ",\n";
    std::vector< double > col( ref_times.size() );
    for ( int j = 0; j < D; ++j )
    {
        for ( std::size_t i = 0; i < ref_times.size(); ++i )
            col[ i ] = ref_sol( ref_times[ i ] )( j );
        out << "    \"x" << j << "\": "; writeJsonArray( out, col );
        out << ( j + 1 < D ? ",\n" : "\n" );
    }
    out << "  },\n";

    //   polygon per snapshot
    out << "  \"polygons\": [\n";
    std::vector< double > xs( boundary.size() ), ys( boundary.size() );
    for ( int s = 0; s < kNSnaps; ++s )
    {
        const double t      = times[ static_cast< std::size_t >( s ) ];
        const auto   x_at_t = sol( t );
        for ( std::size_t v = 0; v < boundary.size(); ++v )
        {
            const std::array< double, M > d{
                0.0, boundary[ v ][ 0 ], 0.0, boundary[ v ][ 1 ]
            };
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

    // ---- Terminal banner ----------------------------------------------------
    const std::vector< std::pair< std::string, std::string > > rows{
        { "P, M, D",     std::to_string( P ) + ", " + std::to_string( M ) + ", " + std::to_string( D ) },
        { "orbits",      std::to_string( kNOrbits ) },
        { "elapsed",     std::to_string( elapsed_ms / 1e3 ) + " s" },
        { "snapshots",   std::to_string( kNSnaps ) },
        { "output",      "taylor.json" }
    };
    printBanner( "taylor (single Taylor flow polynomial)",
                 std::span< const std::pair< std::string, std::string > >{ rows } );
    return 0;
}
