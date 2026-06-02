// =============================================================================
// examples/three_body/common.hpp
//
// Shared planar Earth-Moon CR3BP problem definition + small IO helpers
// for the three-body example(s).
//
// Synodic rotating frame with the Earth-Moon barycentre at the origin.
// Primaries: Earth at (-mu, 0), Moon at (1 - mu, 0). State = (x, y, vx, vy).
//
//   d/dt x  =  vx
//   d/dt y  =  vy
//   d/dt vx =  2 vy + x - (1-mu)(x+mu)/r1^3 - mu (x-1+mu)/r2^3
//   d/dt vy = -2 vx + y - (1-mu) y    /r1^3 - mu  y      /r2^3
//
// with  r1 = sqrt((x+mu)^2 + y^2),  r2 = sqrt((x-1+mu)^2 + y^2).
//
// Jacobi constant (conserved):
//   C = 2*Omega(x, y) - vx^2 - vy^2,
//   Omega = 1/2 (x^2 + y^2) + (1-mu)/r1 + mu/r2 + 1/2 mu (1-mu).
//
// L1 is the libration point between the primaries; its numeric value
// kCR3BPL1 is the root of the 5th-order Lagrange equation.
// =============================================================================

#pragma once

#include <cmath>
#include <iostream>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <tax/la/types.hpp>
#include <tax/tax.hpp>

namespace example::three_body
{

// ---- Problem constants -----------------------------------------------------
inline constexpr double kCR3BPMu = 0.01215058560962404;   // Earth-Moon
inline constexpr double kCR3BPL1 = 0.8369180073407246;

// ---- Right-hand side -------------------------------------------------------
//
// Generic over the state type — accepts scalar VecNT<4, double> and
// DA-valued VecNT<4, TE>. `using V = typename S::Scalar` keeps the
// arithmetic in the appropriate field.
inline auto rhs( double mu = kCR3BPMu )
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

inline double jacobi( const tax::la::VecNT< 4, double >& s, double mu = kCR3BPMu )
{
    const double x  = s( 0 ), y  = s( 1 );
    const double vx = s( 2 ), vy = s( 3 );
    const double r1 = std::hypot( x + mu, y );
    const double r2 = std::hypot( x - 1.0 + mu, y );
    const double Omega = 0.5 * ( x * x + y * y )
                       + ( 1.0 - mu ) / r1 + mu / r2
                       + 0.5 * mu * ( 1.0 - mu );
    return 2.0 * Omega - ( vx * vx + vy * vy );
}

// ---- Terminal banner -------------------------------------------------------
inline void printBanner( std::string_view                                            title,
                         std::span< const std::pair< std::string, std::string > > rows )
{
    constexpr std::size_t label_w = 20;
    std::cout << "\n=== " << title << " ===\n";
    for ( const auto& [ label, value ] : rows )
    {
        const std::size_t pad = label.size() < label_w ? label_w - label.size() : 0;
        std::cout << "  " << std::string( pad, ' ' ) << label << " : " << value << '\n';
    }
    std::cout << '\n';
}

// ---- Inline JSON array writer ----------------------------------------------
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

}  // namespace example::three_body
