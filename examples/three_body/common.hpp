// =============================================================================
// examples/three_body/common.hpp
//
// Shared planar Earth-Moon CR3BP fixture for the three-body examples
// (taylor / ads / loads).
//
// Synodic rotating frame with the Earth-Moon barycentre at the origin.
// Primaries: Earth at (-mu, 0), Moon at (1 - mu, 0). State = (x, y, vx, vy).
//
//   d/dt x  =  vx
//   d/dt y  =  vy
//   d/dt vx =  2 vy + x - (1-mu)(x+mu)/r1^3 - mu (x-1+mu)/r2^3
//   d/dt vy = -2 vx + y - (1-mu) y    /r1^3 - mu  y      /r2^3
//
// All three examples propagate the same small IC box centred *near*
// L1, offset along the unstable eigendirection so the entire box
// drifts toward the Moon under forward propagation:
//
//   ic_center = (x_L1, 0, 0, 0) + kManifoldOffset * v_unstable
//
// The IC box halfwidth varies along two state-space axes (x and vy
// by default). The offset is chosen so all corners of the box have
// strictly positive projection on v_unstable — the trajectories
// stay on the Moon branch of the L1 unstable manifold throughout.
// Flip the sign of kManifoldOffset (or center on L1 itself) if you
// instead want the Earth branch / both branches.
//
// The unstable eigenvector v_unstable is computed numerically with
// Eigen::EigenSolver from the linearised 4x4 dynamics at L1; it is
// cached at first call so all examples and snapshots see identical
// values.
//
// Configurable knobs (edit and rebuild):
//   * kIcBoxHalfWidth  — IC box halfwidth. Default spreads in
//                        (x, vy); change indices to use other axes.
// =============================================================================

#pragma once

#include <Eigen/Eigenvalues>

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

namespace example::three_body
{

// ---- Problem constants -----------------------------------------------------
inline constexpr double kCR3BPMu = 0.01215058560962404;   // Earth-Moon
inline constexpr double kCR3BPL1 = 0.8369180073407246;

// ---- Right-hand side -------------------------------------------------------
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

// ---- L1 linearised eigenvalues + unstable eigenvector ----------------------
struct LinearisationL1
{
    double                       sigma;
    double                       lambda_unstable;
    double                       T_lyapunov;
    tax::la::VecNT< 4, double >  v_unstable;
};

inline LinearisationL1 linearisationL1( double mu = kCR3BPMu, double x_L1 = kCR3BPL1 )
{
    const double r1    = x_L1 + mu;
    const double r2    = 1.0 - mu - x_L1;
    const double sigma = ( 1.0 - mu ) / ( r1 * r1 * r1 )
                       + mu / ( r2 * r2 * r2 );

    Eigen::Matrix4d A;
    A <<     0.0,             0.0,        1.0,  0.0,
             0.0,             0.0,        0.0,  1.0,
         1.0 + 2.0 * sigma,    0.0,        0.0,  2.0,
             0.0,         1.0 - sigma,   -2.0,  0.0;

    Eigen::EigenSolver< Eigen::Matrix4d > es( A );
    const auto& vals = es.eigenvalues();
    const auto& vecs = es.eigenvectors();

    int    idx_u    = -1;
    double lambda_u = 0.0;
    for ( int i = 0; i < 4; ++i )
    {
        const double re = vals( i ).real();
        const double im = vals( i ).imag();
        if ( std::abs( im ) < 1e-9 && re > lambda_u )
        {
            lambda_u = re;
            idx_u    = i;
        }
    }

    LinearisationL1 out{};
    out.sigma           = sigma;
    out.lambda_unstable = lambda_u;

    const double u_minus = 0.5 * ( ( sigma - 2.0 )
                                 - std::sqrt( 9.0 * sigma * sigma - 8.0 * sigma ) );
    out.T_lyapunov       = 2.0 * M_PI / std::sqrt( -u_minus );

    for ( int i = 0; i < 4; ++i )
        out.v_unstable( i ) = vecs( i, idx_u ).real();
    if ( out.v_unstable( 0 ) < 0.0 ) out.v_unstable = -out.v_unstable;
    out.v_unstable /= out.v_unstable.norm();
    return out;
}

inline const LinearisationL1& linL1()
{
    static const LinearisationL1 cached = linearisationL1();
    return cached;
}

// ---- Configurable knobs (edit, rebuild) ------------------------------------
//
// Offset of the IC box centre from L1, along the unstable
// eigendirection. Positive value = Moon branch; negative = Earth
// branch. Magnitude must exceed the box's projection on v_unstable
// (about 2e-5 for the default halfwidth) for the whole box to lie
// strictly on one side of the saddle.
inline constexpr double kManifoldOffset = 4.0e-5;

// IC box halfwidth. Defaults to a (x, vy) face of state space; edit
// the four entries to spread the box on a different 2D face.
inline const tax::la::VecNT< 4, double > kIcBoxHalfWidth{
    3.0e-5, 0.0, 0.0, 3.0e-5
};

// ---- IC + box --------------------------------------------------------------
inline tax::la::VecNT< 4, double > icCenter()
{
    tax::la::VecNT< 4, double > base{ kCR3BPL1, 0.0, 0.0, 0.0 };
    return base + kManifoldOffset * linL1().v_unstable;
}

inline tax::ads::Box< double, 4 > icBox()
{
    return tax::ads::Box< double, 4 >{ icCenter(), kIcBoxHalfWidth };
}

// ---- Boundary of [-1, 1]^2 (closed loop, 4*n + 1 points) -------------------
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

// ---- Map a boundary point (xi_a, xi_b) to a 4D normalised vector -----------
//
// Maps the two active variation axes to their box positions; the two
// pinned axes get 0. Defaults match kIcBoxHalfWidth (x and vy active);
// adjust the index pattern if you change kIcBoxHalfWidth.
inline std::array< double, 4 > boundaryToBox( double a, double b )
{
    return { a, 0.0, 0.0, b };
}

// ---- Pretty terminal banner + JSON helpers ---------------------------------
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
