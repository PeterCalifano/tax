// SPDX-License-Identifier: BSD-3-Clause
//
// Profiling driver for `tax::ode::DaIntegrator` — a flow polynomial in the
// initial conditions, integrated without adaptive domain splitting.
//
// Designed for use with `perf record`, `callgrind`, `vtune`, etc.:
//   - No Google Benchmark dependency (vanilla `int main()`).
//   - Single hot loop runs the same DA integration N times so a long-enough
//     sample lands on the actual hot paths (Cauchy products, sin/cos kernels,
//     Taylor step, stepsize control, etc.).
//   - Iteration count is configurable via argv[1] (default 50) so you can
//     dial up sample density for a given run length.
//   - Minimal output — one summary line — so it never interleaves with
//     profiler output.
//
// Workload: planar Kepler problem (μ = 1) propagated over one orbital period
// from periapsis of an a=1, e=0.5 ellipse. Order-12 Taylor in time, order-4
// in DA, 4-dimensional state. This is a realistic mid-sized DA workload —
// the same problem the `examples/` programs use for their figures.
//
// Build:
//   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTAX_BUILD_BENCHMARK=ON
//   cmake --build build --target profile_da_integration -j
//
// Run:
//   ./build/benchmarks/profile_da_integration          # default 50 iters
//   ./build/benchmarks/profile_da_integration 500      # 500 iters
//
// Profile (example with perf):
//   perf record -g --call-graph dwarf \
//       ./build/benchmarks/profile_da_integration 100
//   perf report

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <string>

#include <tax/tax.hpp>

namespace
{

constexpr int kN = 12;  ///< Taylor expansion order in time.
constexpr int kP = 4;   ///< DA expansion order in initial conditions.
constexpr int kD = 4;   ///< State dimension (2 position + 2 velocity).

/// Planar two-body RHS with normalised gravitational parameter μ = 1.
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

[[nodiscard]] int parseIters( int argc, char** argv, int default_iters )
{
    if ( argc < 2 ) return default_iters;
    try
    {
        const int n = std::stoi( argv[1] );
        return ( n > 0 ) ? n : default_iters;
    }
    catch ( ... )
    {
        return default_iters;
    }
}

}  // namespace

int main( int argc, char** argv )
{
    using clock = std::chrono::steady_clock;
    using namespace std::chrono;

    const int iters = parseIters( argc, argv, 50 );

    // --- Reference IC: periapsis of an a=1, e=0.5 ellipse. -----------------
    constexpr double a = 1.0;
    constexpr double e = 0.5;
    constexpr double rp = a * ( 1.0 - e );
    const double vp = std::sqrt( ( 1.0 + e ) / rp );
    constexpr double T_orbit = 2.0 * std::numbers::pi_v< double > * a * std::sqrt( a );

    // Tight IC box around periapsis — small enough that a single flow
    // polynomial covers it without need for splitting (this is what makes
    // `DaIntegrator` the right tool — see profile_ads_integration.cpp for
    // the splitting case).
    const tax::Box< double, kD > ic_box{
        .center = { rp, 0.0, 0.0, vp },
        .halfWidth = { 1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3 }
    };

    tax::ode::DaIntegrator< kN, kP, kD > integrator(
        kepler, { .abstol = 1e-14, .max_steps = 5000 } );

    // Sink to make sure the optimiser doesn't elide the integration body.
    volatile double sink = 0.0;

    // --- Warm-up (don't count) ---------------------------------------------
    {
        auto sol = integrator.integrate( ic_box, /*t0=*/0.0, T_orbit );
        sink += sol.state( 0 ).value();
    }

    // --- Timed loop --------------------------------------------------------
    const auto t_start = clock::now();
    for ( int i = 0; i < iters; ++i )
    {
        auto sol = integrator.integrate( ic_box, 0.0, T_orbit );
        // Touch the result so the compiler can't optimise away the call.
        sink += sol.state( 0 ).value() + sol.state( 1 ).value();
    }
    const auto t_end = clock::now();

    const auto total_ms = duration_cast< microseconds >( t_end - t_start ).count();
    const double per_iter_ms = double( total_ms ) / double( iters ) / 1000.0;

    std::printf( "profile_da_integration: N=%d P=%d D=%d  iters=%d  "
                 "total=%lld us  per_iter=%.3f ms  sink=%g\n",
                 kN, kP, kD, iters, ( long long )total_ms, per_iter_ms,
                 ( double )sink );
    return 0;
}
