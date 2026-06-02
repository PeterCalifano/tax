// =============================================================================
// examples/three_body/manifolds.cpp
//
// Earth-Moon planar CR3BP: L1 unstable manifold globalization.
//
// At a collinear libration point the planar linearised dynamics has a
// 4x4 Jacobian
//
//     A = [   0          0         1    0
//             0          0         0    1
//          1 + 2 sigma   0         0    2
//             0       1 - sigma   -2    0  ]
//
// with sigma = (1 - mu)/r1^3 + mu/r2^3 (r1, r2 = distances from L1 to
// the two primaries). Its eigenvalues come in pairs:
//
//     +/- lambda    (real,  hyperbolic) -> unstable / stable manifolds
//     +/- i omega   (pure imag, elliptic) -> Lyapunov family
//
// The unstable eigenvector v_u is the direction along which arbitrary
// perturbations grow as e^{lambda t}. The L1 unstable manifold is
// globalised by taking small perturbations of the equilibrium along
// v_u and propagating forward:
//
//     x0(epsilon) = (x_L1, 0, 0, 0) + epsilon * v_u,    epsilon > 0  (toward Moon)
//     x0(epsilon) = (x_L1, 0, 0, 0) + epsilon * v_u,    epsilon < 0  (toward Earth)
//
// We propagate four samples on each side, in geometric progression of
// |epsilon|, to render the manifold as a fan of trajectories whose
// initial points cluster near L1 but whose final points sweep out the
// non-linear manifold.
//
// Run:    ./three_body_manifolds
// Writes: manifolds.json
// =============================================================================

#include <Eigen/Eigenvalues>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <tax/ode/io.hpp>

#include "common.hpp"

namespace
{

using State = tax::la::VecNT< 4, double >;

struct Trajectory
{
    double                epsilon;
    std::string           branch;     // "moon" or "earth"
    std::vector< double > t, x, y, vx, vy, jacobi;
};

}  // namespace

int main()
{
    using namespace example::three_body;
    using namespace tax::ode::methods;

    // ---- Problem constants -----------------------------------------------
    const double mu   = kCR3BPMu;
    const double x_L1 = kCR3BPL1;

    // ---- Linearisation at L1 ---------------------------------------------
    const double r1     = x_L1 + mu;            // distance from L1 to Earth
    const double r2     = 1.0 - mu - x_L1;      // distance from L1 to Moon
    const double sigma  = ( 1.0 - mu ) / ( r1 * r1 * r1 )
                        + mu / ( r2 * r2 * r2 );

    Eigen::Matrix4d A;
    A <<     0.0,             0.0,        1.0,  0.0,
             0.0,             0.0,        0.0,  1.0,
         1.0 + 2.0 * sigma,    0.0,        0.0,  2.0,
             0.0,         1.0 - sigma,   -2.0,  0.0;

    Eigen::EigenSolver< Eigen::Matrix4d > es( A );
    const auto& eigvals = es.eigenvalues();
    const auto& eigvecs = es.eigenvectors();

    // ---- Pick the real positive eigenvalue (unstable mode) ----------------
    int    idx_u    = -1;
    double lambda_u = 0.0;
    for ( int i = 0; i < 4; ++i )
    {
        const double re = eigvals( i ).real();
        const double im = eigvals( i ).imag();
        if ( std::abs( im ) < 1e-9 && re > lambda_u )
        {
            lambda_u = re;
            idx_u    = i;
        }
    }
    if ( idx_u < 0 )
    {
        std::cerr << "no real positive eigenvalue found at L1\n";
        return 1;
    }

    State v_u;
    for ( int i = 0; i < 4; ++i ) v_u( i ) = eigvecs( i, idx_u ).real();
    if ( v_u( 0 ) < 0.0 ) v_u = -v_u;
    v_u /= v_u.norm();

    // ---- Lyapunov half-period (linear estimate) --------------------------
    const double u_minus     = 0.5 * ( ( sigma - 2.0 )
                                     - std::sqrt( 9.0 * sigma * sigma - 8.0 * sigma ) );
    const double omega_lyap  = std::sqrt( -u_minus );  // u_minus < 0 at L1
    const double T_lyapunov  = 2.0 * std::numbers::pi / omega_lyap;

    // ---- Integration setup -----------------------------------------------
    const double t_final = 5.0;          // ~1.9 Lyapunov times
    const int    n_snaps = 400;

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol      = cfg.reltol = 1e-13;
    cfg.max_steps   = 100000;

    auto rhs_fn = rhs( mu );

    // ---- Propagate per epsilon ------------------------------------------
    State x_L1_state;
    x_L1_state << x_L1, 0.0, 0.0, 0.0;

    auto integrateAtEps = [ & ]( double eps, const std::string& branch ) -> Trajectory
    {
        const State ic  = x_L1_state + eps * v_u;
        auto        sol = tax::ode::propagate< /*Dense=*/true >(
            Verner89{}, rhs_fn, ic, 0.0, t_final, cfg );

        Trajectory tr;
        tr.epsilon = eps;
        tr.branch  = branch;
        const auto times = tax::ode::linspace( 0.0, t_final, n_snaps );
        tr.t.reserve( n_snaps );
        tr.x.reserve( n_snaps );
        tr.y.reserve( n_snaps );
        tr.vx.reserve( n_snaps );
        tr.vy.reserve( n_snaps );
        tr.jacobi.reserve( n_snaps );
        for ( double t : times )
        {
            const State x = sol( t );
            tr.t.push_back( t );
            tr.x.push_back( x( 0 ) );
            tr.y.push_back( x( 1 ) );
            tr.vx.push_back( x( 2 ) );
            tr.vy.push_back( x( 3 ) );
            tr.jacobi.push_back( jacobi( x, mu ) );
        }
        return tr;
    };

    const auto t_start = std::chrono::high_resolution_clock::now();

    std::vector< Trajectory > trajectories;
    // Geometric sweep of |epsilon| produces a visual fan.
    for ( double eps : { 1e-7, 3e-7, 1e-6, 3e-6 } )
        trajectories.push_back( integrateAtEps(  eps, "moon"  ) );
    for ( double eps : { 1e-7, 3e-7, 1e-6, 3e-6 } )
        trajectories.push_back( integrateAtEps( -eps, "earth" ) );

    const auto t_end = std::chrono::high_resolution_clock::now();
    const double elapsed_ms =
        std::chrono::duration< double, std::milli >( t_end - t_start ).count();

    // ---- JSON output -----------------------------------------------------
    std::ofstream out( "manifolds.json" );
    out << std::setprecision( 14 );
    out << "{\n";
    out << "  \"problem\": \"planar_cr3bp_earth_moon\",\n";
    out << "  \"config\": {\n";
    out << "    \"mu\":      " << mu       << ",\n";
    out << "    \"x_L1\":    " << x_L1     << ",\n";
    out << "    \"earth_x\": " << ( -mu ) << ",\n";
    out << "    \"moon_x\":  " << ( 1.0 - mu ) << ",\n";
    out << "    \"t_final\": " << t_final  << ",\n";
    out << "    \"n_snaps\": " << n_snaps  << "\n";
    out << "  },\n";
    out << "  \"linearization\": {\n";
    out << "    \"sigma\":           " << sigma        << ",\n";
    out << "    \"lambda_unstable\": " << lambda_u     << ",\n";
    out << "    \"T_lyapunov\":      " << T_lyapunov   << ",\n";
    out << "    \"v_unstable\":      ["
        << v_u( 0 ) << ", " << v_u( 1 ) << ", "
        << v_u( 2 ) << ", " << v_u( 3 ) << "]\n";
    out << "  },\n";
    out << "  \"timing\": { \"elapsed_ms\": " << elapsed_ms << " },\n";
    out << "  \"trajectories\": [\n";
    for ( std::size_t i = 0; i < trajectories.size(); ++i )
    {
        const auto& tr = trajectories[ i ];
        out << "    {\n";
        out << "      \"branch\":  \"" << tr.branch << "\",\n";
        out << "      \"epsilon\": " << tr.epsilon << ",\n";
        out << "      \"t\":      "; writeJsonArray( out, tr.t      ); out << ",\n";
        out << "      \"x\":      "; writeJsonArray( out, tr.x      ); out << ",\n";
        out << "      \"y\":      "; writeJsonArray( out, tr.y      ); out << ",\n";
        out << "      \"vx\":     "; writeJsonArray( out, tr.vx     ); out << ",\n";
        out << "      \"vy\":     "; writeJsonArray( out, tr.vy     ); out << ",\n";
        out << "      \"jacobi\": "; writeJsonArray( out, tr.jacobi ); out << "\n";
        out << "    }" << ( i + 1 < trajectories.size() ? "," : "" ) << "\n";
    }
    out << "  ]\n";
    out << "}\n";

    // ---- Terminal banner -------------------------------------------------
    std::ostringstream v_u_str;
    v_u_str << std::scientific << std::setprecision( 3 )
            << "(" << v_u( 0 ) << ", " << v_u( 1 ) << ", "
                   << v_u( 2 ) << ", " << v_u( 3 ) << ")";

    const std::vector< std::pair< std::string, std::string > > rows{
        { "mu",                std::to_string( mu )         },
        { "x_L1",              std::to_string( x_L1 )       },
        { "sigma",             std::to_string( sigma )      },
        { "lambda_unstable",   std::to_string( lambda_u )   },
        { "Lyapunov period",   std::to_string( T_lyapunov ) },
        { "v_unstable",        v_u_str.str()                },
        { "trajectories",      std::to_string( trajectories.size() ) + "  (4 Moon + 4 Earth)" },
        { "t_final",           std::to_string( t_final )    },
        { "elapsed",           std::to_string( elapsed_ms / 1e3 ) + " s" },
        { "output",            "manifolds.json"             }
    };
    printBanner( "L1 unstable manifolds (Earth-Moon CR3BP)",
                 std::span< const std::pair< std::string, std::string > >{ rows } );
    return 0;
}
