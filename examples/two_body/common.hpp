// =============================================================================
// examples/two_body/common.hpp
//
// Shared problem definition + small IO helpers for the three two-body
// examples (taylor, ads, loads).
//
// The model is the planar Kepler problem in canonical units:
//
//    d/dt (x, y)   = (vx, vy)
//    d/dt (vx, vy) = -(x, y) / r^3 ,    r = sqrt(x^2 + y^2)
//
// with GM = 1 and semi-major axis a = 1. The reference IC is at periapsis
// of an eccentricity-0.5 ellipse:
//
//    x  = a(1 - e) = 0.5
//    vy = sqrt((1 + e)/(1 - e)) = sqrt(3)
//
// One full orbital period is T = 2*pi.
//
// Helpers exposed:
//   * rhs()                  — generic RHS lambda (scalar + DA-valued state)
//   * icCenter()             — periapsis IC vector
//   * unitSquareBoundary(n)  — closed-loop boundary of [-1, 1]^2 for
//                              evaluating the DA flow polygon
//   * printBanner(...)       — tidy per-example terminal summary
//   * writeJsonArray(...)    — inline JSON array writer for numeric ranges
// =============================================================================

#pragma once

#include <array>
#include <cmath>
#include <iostream>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <tax/ads/box.hpp>
#include <tax/la/types.hpp>
#include <tax/tax.hpp>

namespace example::two_body
{

// ---- Orbit constants -------------------------------------------------------
inline constexpr double kEcc        = 0.5;
inline constexpr double kPeriapsis  = 1.0 - kEcc;                 // x(0)
inline const     double kVPeriapsis = std::sqrt( ( 1.0 + kEcc ) / ( 1.0 - kEcc ) );
inline const     double kPeriod     = 2.0 * M_PI;

// ---- Right-hand side -------------------------------------------------------
//
// Generic over the state type so the same lambda accepts:
//   * tax::la::VecNT<4, double>            (scalar reference path)
//   * tax::la::VecNT<4, tax::TE<P, M>>     (DA-valued state, used everywhere
//                                          else here).
// ADL picks up tax::sqrt for TE; <cmath> provides ::sqrt for double.
inline auto rhs()
{
    return []( const auto& s, const auto& /*t*/ )
    {
        using S       = std::decay_t< decltype( s ) >;
        const auto x  = s( 0 );
        const auto y  = s( 1 );
        const auto r2 = x * x + y * y;
        const auto r3 = r2 * sqrt( r2 );    // r^3 = r^2 * r

        S out;
        out( 0 ) =  s( 2 );      // dx/dt  = vx
        out( 1 ) =  s( 3 );      // dy/dt  = vy
        out( 2 ) = -x / r3;      // dvx/dt = -x/r^3
        out( 3 ) = -y / r3;      // dvy/dt = -y/r^3
        return out;
    };
}

inline tax::la::VecNT< 4, double > icCenter()
{
    return tax::la::VecNT< 4, double >{ kPeriapsis, 0.0, 0.0, kVPeriapsis };
}

// ---- IC box configuration --------------------------------------------------
//
// Edit kIcBoxHalfWidth to change the size of the initial-condition box used
// by all three examples (taylor / ads / loads). Zero-half-width components
// pin the corresponding state axis to its centerpoint value.
//
// The defaults vary only the y position and the y-velocity — that's enough
// to produce visible distortion in one orbit at e = 0.5 without triggering
// excessive ADS subdivisions.
inline const tax::la::VecNT< 4, double > kIcBoxHalfWidth{ 0.0, 8e-3, 0.0, 2e-2 };

inline tax::ads::Box< double, 4 > icBox()
{
    return tax::ads::Box< double, 4 >{ icCenter(), kIcBoxHalfWidth };
}

// ---- Boundary of [-1, 1]^2 -------------------------------------------------
//
// Returns 4 * n_per_edge + 1 samples tracing the perimeter
// counter-clockwise. The first vertex is repeated at the end so the
// polygon closes.
inline std::vector< std::array< double, 2 > > unitSquareBoundary( int n_per_edge )
{
    std::vector< std::array< double, 2 > > pts;
    pts.reserve( static_cast< std::size_t >( 4 * n_per_edge + 1 ) );
    for ( int edge = 0; edge < 4; ++edge )
    {
        for ( int i = 0; i < n_per_edge; ++i )
        {
            const double s = static_cast< double >( i ) / static_cast< double >( n_per_edge );
            double a = 0.0, b = 0.0;
            switch ( edge )
            {
                case 0: a = -1.0 + 2.0 * s; b = +1.0;             break;
                case 1: a = +1.0;            b = +1.0 - 2.0 * s; break;
                case 2: a = +1.0 - 2.0 * s; b = -1.0;             break;
                case 3: a = -1.0;            b = -1.0 + 2.0 * s; break;
            }
            pts.push_back( { a, b } );
        }
    }
    pts.push_back( pts.front() );
    return pts;
}

// ---- Pretty terminal banner ------------------------------------------------
inline void printBanner( std::string_view                                            title,
                         std::span< const std::pair< std::string, std::string > > rows )
{
    constexpr std::size_t label_w = 18;
    std::cout << "\n=== " << title << " ===\n";
    for ( const auto& [ label, value ] : rows )
    {
        const std::size_t pad = label.size() < label_w ? label_w - label.size() : 0;
        std::cout << "  " << std::string( pad, ' ' ) << label << " : " << value << '\n';
    }
    std::cout << '\n';
}

// ---- Inline JSON array writer ----------------------------------------------
//
// All structural JSON (objects, keys, ordering) is hand-written in each
// example so a reader can see the data shape directly. This helper just
// flattens a numeric range into "[a, b, c, ...]".
template < class Range >
inline void writeJsonArray( std::ostream& out, const Range& v )
{
    out << '[';
    bool first = true;
    for ( auto x : v )
    {
        if ( !first ) out << ", ";
        out << x;
        first = false;
    }
    out << ']';
}

}  // namespace example::two_body
