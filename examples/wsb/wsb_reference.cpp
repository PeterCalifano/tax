// =============================================================================
// examples/wsb/wsb_reference.cpp
//
// Weak Stability Boundary lunar transfer — first-draft reference
// trajectory in the planar Sun-Earth CR3BP.
//
// Canonical units (Sun-Earth):
//   length unit  = 1 AU                  = 149,597,870.7 km
//   time unit    = 1 / sqrt(GM_sun / a^3) = year / (2 pi)
//   velocity     = AU / time unit         ≈ 29.785 km/s
//   mass param   = M_Earth / (M_Sun + M_Earth) ≈ 3.0035e-6
//
// Synodic rotating frame with the Sun-Earth barycentre at the origin:
//   Sun   at (-mu_SE, 0)            (essentially the origin)
//   Earth at (1 - mu_SE, 0)         (just inside x = 1)
//   L2    at ≈ (1 - mu_SE + r_H, 0), r_H = (mu_SE/3)^(1/3) ≈ 0.01
//
// First-draft IC: parking just above LEO altitude on the anti-Sun side
// of Earth, prograde (in inertial) — slightly above local escape so
// the spacecraft drifts outward and gets shepherded by Sun gravity.
// Numbers chosen so they show *something* on the figure; not yet a
// real WSB transfer.
//
// Run:    ./wsb_reference
// Writes: wsb_reference.json
// =============================================================================

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <tax/ode/io.hpp>

namespace
{

using Vec4 = tax::la::VecNT< 4, double >;

// ---- Sun-Earth CR3BP constants ----------------------------------------------
inline constexpr double kSunEarthMu = 3.00348959632E-6;   // M_Earth / (M_Sun + M_Earth)
inline constexpr double kAU_km      = 149597870.7;
inline constexpr double kVelU_kms   = 29.78469183;        // 1 AU / (year/(2 pi))
inline constexpr double kTimeU_days = 58.13235;           // 1 / (2 pi/year) in days

// Earth reference + L2 offset.
inline constexpr double kEarthX     = 1.0 - kSunEarthMu;
inline           double earthHillR()                       // (mu_SE/3)^(1/3)
{
    return std::cbrt( kSunEarthMu / 3.0 );
}

// Moon's mean orbital radius around Earth (just used as a reference
// circle on the plot — the Moon does NOT live in the Sun-Earth CR3BP).
inline constexpr double kMoonOrbitKm = 384400.0;
inline constexpr double kMoonOrbitR  = kMoonOrbitKm / kAU_km;

// ---- Right-hand side --------------------------------------------------------
inline auto rhs( double mu = kSunEarthMu )
{
    return [ mu ]( const auto& s, const auto& /*t*/ )
    {
        using S = std::decay_t< decltype( s ) >;
        using V = typename S::Scalar;

        S out;
        const V x  = s( 0 );
        const V y  = s( 1 );
        const V vx = s( 2 );
        const V vy = s( 3 );

        const V x1   = x + V( mu );
        const V x2   = x - V( 1.0 - mu );
        const V r1_2 = x1 * x1 + y * y;
        const V r2_2 = x2 * x2 + y * y;
        const V r1_3 = r1_2 * sqrt( r1_2 );
        const V r2_3 = r2_2 * sqrt( r2_2 );

        out( 0 ) = vx;
        out( 1 ) = vy;
        out( 2 ) =  V( 2.0 ) * vy + x
                   - V( 1.0 - mu ) * x1 / r1_3
                   - V( mu )       * x2 / r2_3;
        out( 3 ) = -V( 2.0 ) * vx + y
                   - V( 1.0 - mu ) * y  / r1_3
                   - V( mu )       * y  / r2_3;
        return out;
    };
}

}  // namespace

int main()
{
    using namespace tax::ode::methods;

    // ---- Initial condition (draft) -----------------------------------------
    //
    // Parameterise by the Earth-orbital characteristic energy
    //   C3 = v_inertial^2 - 2 GM_Earth / r
    // C3 < 0 -> bound Keplerian ellipse around Earth (apogee inside
    //          Earth's sphere of influence). For a Belbruno-style WSB
    //          transfer, C3 ~ -0.5 km^2/s^2 puts apogee right at the
    //          Hill-sphere boundary (~1.5 x 10^6 km).
    // C3 = 0  parabolic; C3 > 0 hyperbolic (just shoots away).
    constexpr double kEarthR_km = 6378.0;
    constexpr double h_km       = 200.0;             // LEO altitude
    const double     r_park     = ( kEarthR_km + h_km ) / kAU_km;   // ~4.4e-5 AU

    constexpr double GM_E_km3s2 = 398600.4418;
    const double     v_esc_kms  = std::sqrt( 2.0 * GM_E_km3s2 / ( kEarthR_km + h_km ) );

    // ---- WSB knob -----------------------------------------------------------
    // Belbruno's WSB lobe sits at C3 in (-1.0, -0.3) km^2/s^2 for
    // Earth-Moon transfers; -0.5 is a textbook starting point.
    constexpr double C3_kms2 = -0.5;
    const double v0_kms = std::sqrt( v_esc_kms * v_esc_kms + C3_kms2 );
    const double v0     = v0_kms / kVelU_kms;        // canonical

    // Predicted Earth-Kepler apogee (helpful for calibration):
    //   1/a = 2/r - v^2/GM,  r_a = 2a - r_p.
    const double a_earth_km = 1.0 / ( 2.0 / ( kEarthR_km + h_km )
                                     - v0_kms * v0_kms / GM_E_km3s2 );
    const double r_apogee_km = 2.0 * a_earth_km - ( kEarthR_km + h_km );
    const double T_earth_orbit_days =
        ( 2.0 * M_PI * std::sqrt( std::pow( a_earth_km, 3.0 ) / GM_E_km3s2 ) ) / 86400.0;

    // Inertial perigee on the anti-Sun side of Earth (+x in synodic).
    // Prograde tangential burn → inertial velocity in +y. Convert to
    // synodic via v_rot = v_inertial - omega x r with omega = +1 zhat.
    const double r_x_earth = r_park;                 // anti-Sun side
    const double vy_syn    = v0 - ( 1.0 * r_x_earth );

    Vec4 ic;
    ic << kEarthX + r_x_earth, 0.0, 0.0, vy_syn;

    // ---- Integration setup --------------------------------------------------
    constexpr double tFinal_days = 90.0;
    const     double tFinal       = tFinal_days / kTimeU_days;   // canonical

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol      = cfg.reltol = 1e-13;
    cfg.max_steps   = 5000000;
    cfg.initial_step = 1e-5;

    // ---- Propagate ----------------------------------------------------------
    const auto t_start = std::chrono::high_resolution_clock::now();
    auto       sol     = tax::ode::propagate< /*Dense=*/true >(
        Verner89{}, rhs(), ic, 0.0, tFinal, cfg );
    const auto t_end   = std::chrono::high_resolution_clock::now();
    const double elapsed_ms =
        std::chrono::duration< double, std::milli >( t_end - t_start ).count();

    // ---- Sample the trajectory and write JSON -------------------------------
    constexpr int   n_snaps = 4000;
    const auto      times   = tax::ode::linspace( 0.0, tFinal, n_snaps );

    std::vector< double > xs, ys, vxs, vys, xs_e, ys_e;
    xs.reserve( n_snaps ); ys.reserve( n_snaps );
    vxs.reserve( n_snaps ); vys.reserve( n_snaps );
    xs_e.reserve( n_snaps ); ys_e.reserve( n_snaps );
    for ( double t : times )
    {
        const Vec4 s = sol( t );
        xs.push_back(  s( 0 ) );
        ys.push_back(  s( 1 ) );
        vxs.push_back( s( 2 ) );
        vys.push_back( s( 3 ) );
        // Earth-centred (subtract Earth position).
        xs_e.push_back( s( 0 ) - kEarthX );
        ys_e.push_back( s( 1 ) );
    }

    auto writeArray = [ & ]( std::ostream& o, const std::vector< double >& v )
    {
        o << '[';
        for ( std::size_t i = 0; i < v.size(); ++i )
        {
            if ( i ) o << ", ";
            o << v[ i ];
        }
        o << ']';
    };

    std::ofstream out( "wsb_reference.json" );
    out << std::setprecision( 15 );
    out << "{\n";
    out << "  \"problem\": \"sun_earth_cr3bp\",\n";
    out << "  \"config\": {\n";
    out << "    \"mu_SE\":             " << kSunEarthMu  << ",\n";
    out << "    \"earth_x\":           " << kEarthX      << ",\n";
    out << "    \"sun_x\":             " << ( -kSunEarthMu ) << ",\n";
    out << "    \"earth_hill_radius\": " << earthHillR() << ",\n";
    out << "    \"L2_x\":              " << ( kEarthX + earthHillR() ) << ",\n";
    out << "    \"L1_x\":              " << ( kEarthX - earthHillR() ) << ",\n";
    out << "    \"moon_orbit_AU\":     " << kMoonOrbitR  << ",\n";
    out << "    \"AU_km\":             " << kAU_km       << ",\n";
    out << "    \"velocity_unit_kms\": " << kVelU_kms    << ",\n";
    out << "    \"time_unit_days\":    " << kTimeU_days  << ",\n";
    out << "    \"t_final\":           " << tFinal       << ",\n";
    out << "    \"t_final_days\":      " << tFinal_days  << "\n";
    out << "  },\n";
    out << "  \"ic\": {\n";
    out << "    \"r_park_AU\":          " << r_park     << ",\n";
    out << "    \"r_park_km\":          " << ( r_park * kAU_km ) << ",\n";
    out << "    \"v_inertial_kms\":     " << v0_kms     << ",\n";
    out << "    \"C3_kms2\":            " << C3_kms2    << ",\n";
    out << "    \"earth_kepler_apogee_km\": " << r_apogee_km << ",\n";
    out << "    \"earth_kepler_T_days\":    " << T_earth_orbit_days << ",\n";
    out << "    \"v_synodic\":          " << vy_syn     << ",\n";
    out << "    \"state\":              [" << ic( 0 ) << ", " << ic( 1 )
        << ", "                            << ic( 2 ) << ", " << ic( 3 ) << "]\n";
    out << "  },\n";
    out << "  \"timing\": { \"elapsed_ms\": " << elapsed_ms
        << ", \"n_steps\": " << ( sol.size() - 1 ) << " },\n";
    out << "  \"trajectory\": {\n";
    out << "    \"t\":       "; writeArray( out, std::vector< double >( times.begin(), times.end() ) ); out << ",\n";
    out << "    \"x_syn\":   "; writeArray( out, xs );  out << ",\n";
    out << "    \"y_syn\":   "; writeArray( out, ys );  out << ",\n";
    out << "    \"vx_syn\":  "; writeArray( out, vxs ); out << ",\n";
    out << "    \"vy_syn\":  "; writeArray( out, vys ); out << ",\n";
    out << "    \"x_earth\": "; writeArray( out, xs_e ); out << ",\n";
    out << "    \"y_earth\": "; writeArray( out, ys_e ); out << "\n";
    out << "  }\n";
    out << "}\n";

    // ---- Terminal banner ----------------------------------------------------
    constexpr std::size_t label_w = 24;
    auto bannerRow = []( std::string_view label, std::string_view value )
    {
        const std::size_t pad = label.size() < label_w ? label_w - label.size() : 0;
        std::cout << "  " << std::string( pad, ' ' ) << label << " : " << value << '\n';
    };

    std::cout << "\n=== Sun-Earth CR3BP : WSB reference (draft) ===\n";
    bannerRow( "mu_SE",             std::to_string( kSunEarthMu ) );
    bannerRow( "Earth Hill radius", std::to_string( earthHillR() ) + " AU = "
                                  + std::to_string( earthHillR() * kAU_km / 1.0e6 ) + " e6 km" );
    bannerRow( "Park altitude",     std::to_string( h_km ) + " km" );
    bannerRow( "Park r (AU)",       std::to_string( r_park ) );
    bannerRow( "Escape v",          std::to_string( v_esc_kms ) + " km/s" );
    bannerRow( "C3 (km^2/s^2)",     std::to_string( C3_kms2 )  );
    bannerRow( "Inertial v_0",      std::to_string( v0_kms ) + " km/s" );
    bannerRow( "Earth-Kepler r_a",  std::to_string( r_apogee_km / 1e6 ) + " e6 km" );
    bannerRow( "Earth-Kepler T",    std::to_string( T_earth_orbit_days ) + " days" );
    bannerRow( "Synodic v_y_0",     std::to_string( vy_syn ) + " (canonical)" );
    bannerRow( "t_final",           std::to_string( tFinal_days ) + " days" );
    bannerRow( "elapsed",           std::to_string( elapsed_ms / 1e3 ) + " s ("
                                  + std::to_string( sol.size() - 1 ) + " steps)" );
    bannerRow( "output",            "wsb_reference.json" );
    std::cout << '\n';
    return 0;
}
