// SPDX-License-Identifier: BSD-3-Clause
//
// Profiling driver for `tax::ode::AdsIntegrator` — adaptive Domain Splitting
// integration. Exercises the full ADS pipeline: flow propagation in DA
// (the heavy kernels — Cauchy products, sin/cos, sqrt), truncation-error
// estimation, leaf bisection, work-queue management, and tree growth.
//
// Designed for use with `perf`, `callgrind`, `vtune`, ...; same conventions
// as `profile_da_integration.cpp`.
//
// Workload: planar Kepler (μ = 1) over one orbital period with a `2 %`
// tangential-velocity perturbation box around periapsis of an a=1, e=0.5
// ellipse. The default tolerance `ads_tol = 1e-3` is loose enough to
// finish in a few seconds but tight enough to force several splits — the
// exact regime where the splitter / queue management cost should dominate
// over per-leaf propagation.
//
// Build:
//   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTAX_BUILD_BENCHMARK=ON
//   cmake --build build --target profile_ads_integration -j
//
// Run:
//   ./build/benchmarks/profile_ads_integration              # default 5 iters
//   ./build/benchmarks/profile_ads_integration 50 1e-2      # 50 iters, ads_tol=1e-2
//
// Profile (example):
//   perf record -g --call-graph dwarf \
//       ./build/benchmarks/profile_ads_integration 20
//   perf report

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numbers>

#include <tax/tax.hpp>

namespace
{

constexpr int kN = 12;  ///< Taylor expansion order in time.
constexpr int kP = 4;   ///< DA expansion order in initial conditions.
constexpr int kD = 4;   ///< State dimension.

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

struct CliArgs
{
    int iters = 5;
    double ads_tol = 1e-3;
};

[[nodiscard]] CliArgs parseCli( int argc, char** argv )
{
    CliArgs a;
    if ( argc >= 2 )
    {
        try
        {
            const int n = std::stoi( argv[1] );
            if ( n > 0 ) a.iters = n;
        }
        catch ( ... )
        {
        }
    }
    if ( argc >= 3 )
    {
        try
        {
            const double t = std::stod( argv[2] );
            if ( t > 0.0 ) a.ads_tol = t;
        }
        catch ( ... )
        {
        }
    }
    return a;
}

}  // namespace

int main( int argc, char** argv )
{
    using clock = std::chrono::steady_clock;
    using namespace std::chrono;

    const auto args = parseCli( argc, argv );

    // --- Reference IC: periapsis of an a=1, e=0.5 ellipse. -----------------
    constexpr double a = 1.0;
    constexpr double e = 0.5;
    constexpr double rp = a * ( 1.0 - e );
    const double vp = std::sqrt( ( 1.0 + e ) / rp );
    constexpr double T_orbit = 2.0 * std::numbers::pi_v< double > * a * std::sqrt( a );

    // IC box wide enough to force splits: 2 % tangential-velocity wedge.
    const tax::Box< double, kD > ic_box{
        .center = { rp, 0.0, 0.0, vp },
        .halfWidth = { 0.0, 0.0, 0.0, 0.02 * vp }
    };

    tax::ode::AdsIntegrator< kN, kP, kD > integrator(
        kepler,
        { .step_tol = 1e-14,
          .ads_tol = args.ads_tol,
          .max_depth = 30,
          .max_steps = 5000 } );

    volatile double sink = 0.0;
    long long total_leaves = 0;

    // --- Warm-up (don't count) ---------------------------------------------
    {
        auto tree = integrator.integrate( ic_box, /*t0=*/0.0, T_orbit );
        for ( int i : tree.doneLeaves() )
            sink += tree.node( i ).leaf().tte.state( 0 ).value();
    }

    // --- Timed loop --------------------------------------------------------
    const auto t_start = clock::now();
    for ( int it = 0; it < args.iters; ++it )
    {
        auto tree = integrator.integrate( ic_box, 0.0, T_orbit );
        const auto& done = tree.doneLeaves();
        total_leaves += static_cast< long long >( done.size() );
        for ( int i : done )
            sink += tree.node( i ).leaf().tte.state( 0 ).value();
    }
    const auto t_end = clock::now();

    const auto total_ms = duration_cast< microseconds >( t_end - t_start ).count();
    const double per_iter_ms = double( total_ms ) / double( args.iters ) / 1000.0;
    const double avg_leaves = double( total_leaves ) / double( args.iters );

    std::printf( "profile_ads_integration: N=%d P=%d D=%d  ads_tol=%g  iters=%d  "
                 "total=%lld us  per_iter=%.3f ms  avg_leaves=%.1f  sink=%g\n",
                 kN, kP, kD, args.ads_tol, args.iters, ( long long )total_ms,
                 per_iter_ms, avg_leaves, ( double )sink );
    return 0;
}
