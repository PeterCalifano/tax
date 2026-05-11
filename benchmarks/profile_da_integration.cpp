// SPDX-License-Identifier: BSD-3-Clause
//
// Profiling driver for `tax::ode::DaIntegrator` — Kepler over one orbital
// period from periapsis of an a=1, e=0.5 ellipse, order-12 Taylor in time,
// order-4 in DA, 4-D state.  Intentionally minimal: no timing, no
// Google Benchmark — `perf` / `callgrind` / `vtune` does the measuring.
//
//   perf record -g --call-graph dwarf -- \
//       ./build/benchmarks/profile_da_integration 100
//   perf report
//
// argv[1]   iterations (default 50)

#include <cstdio>
#include <cstdlib>
#include <numbers>

#include <tax/tax.hpp>

int main( int argc, char** argv )
{
    int iters = 50;
    if ( argc >= 2 ) iters = std::atoi( argv[1] );
    if ( iters <= 0 ) iters = 50;

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
        .halfWidth = { 1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3 }
    };

    tax::ode::DaIntegrator< N, P, D > integrator(
        kepler, { .abstol = 1e-14, .max_steps = 5000 } );

    volatile double sink = 0.0;
    for ( int i = 0; i < iters; ++i )
    {
        auto sol = integrator.integrate( ic_box, 0.0, T_orbit );
        sink += sol.state( 0 ).value() + sol.state( 1 ).value();
    }

    std::printf( "profile_da_integration: iters=%d sink=%g\n", iters, ( double )sink );
    return 0;
}
