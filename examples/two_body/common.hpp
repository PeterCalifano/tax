// examples/two_body/common.hpp
//
// Shared problem definition for the three two-body examples:
// planar Kepler orbit with eccentricity e = 0.5, canonical units
// (GM = 1, a = 1). IC at periapsis, period T = 2π. The RHS is a
// generic lambda so it accepts both scalar VecNT<4, double> states
// (for the simple-Taylor example and IC-centerpoint reference) and
// DA-valued VecNT<4, TE<P, M>> states (for ADS / LOADS).

#pragma once

#include <array>
#include <cmath>

#include <tax/la/types.hpp>
#include <tax/tax.hpp>

namespace example::two_body
{

inline constexpr double kEcc        = 0.5;
inline constexpr double kPeriapsis  = 1.0 - kEcc;   // a*(1-e), a=1
inline const     double kVPeriapsis = std::sqrt( ( 1.0 + kEcc ) / ( 1.0 - kEcc ) );
inline const     double kPeriod     = 2.0 * M_PI;

inline auto rhs()
{
    return []( const auto& s, const auto& /*t*/ )
    {
        using S       = std::decay_t< decltype( s ) >;
        const auto x  = s( 0 );
        const auto y  = s( 1 );
        const auto r2 = x * x + y * y;
        const auto r3 = r2 * sqrt( r2 );
        S out;
        out( 0 ) = s( 2 );
        out( 1 ) = s( 3 );
        out( 2 ) = -x / r3;
        out( 3 ) = -y / r3;
        return out;
    };
}

inline tax::la::VecNT< 4, double > icCenter()
{
    tax::la::VecNT< 4, double > x0;
    x0 << kPeriapsis, 0.0, 0.0, kVPeriapsis;
    return x0;
}

inline std::array< double, 4 > icCenterArray()
{
    return { kPeriapsis, 0.0, 0.0, kVPeriapsis };
}

}  // namespace example::two_body
