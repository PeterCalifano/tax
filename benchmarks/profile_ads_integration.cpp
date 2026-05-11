// SPDX-License-Identifier: BSD-3-Clause
//
// Profiling driver for `tax::ode::AdsIntegrator` — planar Kepler over one
// orbital period with a 2 % tangential-velocity wedge around periapsis of
// an a=1, e=0.5 ellipse.  Exercises the full ADS pipeline: DA flow
// propagation, truncation-error estimation, leaf bisection, work-queue
// management, and tree growth.  Intentionally minimal: no timing, no
// Google Benchmark — `perf` / `callgrind` / `vtune` does the measuring.
//
//   perf record -g --call-graph dwarf -- \
//       ./build/benchmarks/profile_ads_integration 20
//   perf report
//
// argv[1]   iterations (default 5)
// argv[2]   ads_tol    (default 1e-3)

#include <cstdio>
#include <cstdlib>
#include <numbers>

#include <tax/tax.hpp>

int main( int argc, char** argv )
{
    int iters = 5;
    double ads_tol = 1e-3;
    if ( argc >= 2 ) iters = std::atoi( argv[1] );
    if ( iters <= 0 ) iters = 5;
    if ( argc >= 3 ) ads_tol = std::atof( argv[2] );
    if ( ads_tol <= 0.0 ) ads_tol = 1e-3;

    constexpr int N = 12;
    constexpr int P = 4;
    constexpr int D = 4;

    constexpr auto kepler =
        []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
            using std::sqrt;
            auto r2 = x( 0 ) * x( 0 ) + x( 1 ) * x( 1 );
            auto r = sqrt( r2 );
            auto r3 = r2 * r;
            dx( 0 ) = x( 2 );
            dx( 1 ) = x( 3 );
            dx( 2 ) = -x( 0 ) / r3;
            dx( 3 ) = -x( 1 ) / r3;
        };

    constexpr double a = 1.0;
    constexpr double e = 0.5;
    constexpr double rp = a * ( 1.0 - e );
    const double vp = std::sqrt( ( 1.0 + e ) / rp );
    constexpr double T_orbit = 2.0 * std::numbers::pi_v< double > * a * std::sqrt( a );

    const tax::Box< double, D > ic_box{
        .center = { rp, 0.0, 0.0, vp },
        .halfWidth = { 0.0, 0.0, 0.0, 0.02 * vp }
    };

    tax::ode::AdsIntegrator< N, P, D > integrator(
        kepler,
        { .step_tol = 1e-14,
          .ads_tol = ads_tol,
          .max_depth = 30,
          .max_steps = 5000 } );

    volatile double sink = 0.0;
    for ( int i = 0; i < iters; ++i )
    {
        auto tree = integrator.integrate( ic_box, 0.0, T_orbit );
        for ( int j : tree.doneLeaves() )
            sink += tree.node( j ).leaf().tte.state( 0 ).value();
    }

    std::printf( "profile_ads_integration: iters=%d ads_tol=%g sink=%g\n",
                 iters, ads_tol, ( double )sink );
    return 0;
}
