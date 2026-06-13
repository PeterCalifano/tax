// =============================================================================
// benchmarks/bench_ads_refine.cpp
//
// Runtime comparison of the two ADS strategies on the same task — propagate a
// box of Kepler initial conditions over one orbit and partition it:
//
//   * classic in-flight ADS   (tax::ads::propagate, TruncationCriterion)
//   * propagate-then-assess    (tax::ads::refine, CoefficientMatch / Volume)
//
// Each is run with 1, 2, 3 and 4 worker threads (the driver's num_threads
// argument; the thread count is benchmark arg 0) so the parallel scaling of
// the two approaches can be compared. The "leaves" counter reports the size
// of the resulting partition, since the two criteria do not split to exactly
// the same resolution — divide time by leaves for a per-box cost.
//
// Build:  cmake -S . -B build -DTAX_BUILD_BENCHMARK=ON && cmake --build build -j
// Run:    ./build/benchmarks/bench_ads_refine
// =============================================================================

#include <benchmark/benchmark.h>

#include <cmath>
#include <tax/ads.hpp>
#include <tax/ode.hpp>

namespace
{
using namespace tax::ode::methods;

constexpr int P = 6;  // DA truncation order (same for both methods)

// Planar Kepler problem (GM = 1), generic over scalar / DA state.
auto rhs()
{
    return []( const auto& s, const auto& ) {
        using S = std::decay_t< decltype( s ) >;
        const auto x = s( 0 );
        const auto y = s( 1 );
        const auto r2 = x * x + y * y;
        const auto r3 = r2 * sqrt( r2 );
        S o;
        o( 0 ) = s( 2 );
        o( 1 ) = s( 3 );
        o( 2 ) = -x / r3;
        o( 3 ) = -y / r3;
        return o;
    };
}

tax::la::VecNT< 4, double > center() { return { 0.5, 0.0, 0.0, std::sqrt( 3.0 ) }; }

// IC box varying y and vy (×3 the small tutorial box, so the partition is big
// enough to exercise the thread pool).
tax::ads::Box< double, 4 > icBox()
{
    constexpr double s = 3.0;
    return { center(), tax::la::VecNT< 4, double >{ 0.0, 8e-3 * s, 0.0, 2e-2 * s } };
}

tax::ode::IntegratorConfig< double > cfg()
{
    tax::ode::IntegratorConfig< double > c;
    c.abstol = c.reltol = 1e-12;
    return c;
}

constexpr double kTfinal = 2.0 * M_PI;
}  // namespace

static void BM_ClassicAds( benchmark::State& state )
{
    const int threads = static_cast< int >( state.range( 0 ) );
    std::size_t leaves = 0;
    for ( auto _ : state )
    {
        auto tree =
            tax::ads::propagate< P >( Verner89{}, tax::ads::TruncationCriterion{ 1e-4, 6 }, rhs(),
                                      icBox(), center(), 0.0, kTfinal, cfg(), threads );
        benchmark::DoNotOptimize( tree );
        leaves = tree.done().size();
    }
    state.counters["leaves"] = static_cast< double >( leaves );
}
BENCHMARK( BM_ClassicAds )
    ->Arg( 1 )
    ->Arg( 2 )
    ->Arg( 3 )
    ->Arg( 4 )
    ->Unit( benchmark::kMillisecond )
    ->UseRealTime();

static void BM_RefineCoeff( benchmark::State& state )
{
    const int threads = static_cast< int >( state.range( 0 ) );
    std::size_t leaves = 0;
    for ( auto _ : state )
    {
        auto tree =
            tax::ads::refine< P >( Verner89{}, tax::ads::CoefficientMatchCriterion{ 1e-6, 6 },
                                   rhs(), icBox(), center(), 0.0, kTfinal, cfg(), threads );
        benchmark::DoNotOptimize( tree );
        leaves = tree.done().size();
    }
    state.counters["leaves"] = static_cast< double >( leaves );
}
BENCHMARK( BM_RefineCoeff )
    ->Arg( 1 )
    ->Arg( 2 )
    ->Arg( 3 )
    ->Arg( 4 )
    ->Unit( benchmark::kMillisecond )
    ->UseRealTime();

static void BM_RefineVolume( benchmark::State& state )
{
    const int threads = static_cast< int >( state.range( 0 ) );
    std::size_t leaves = 0;
    for ( auto _ : state )
    {
        auto tree = tax::ads::refine< P >( Verner89{},
                                           tax::ads::VolumeRatioCriterion{ 1e-6, 6, { 1, 3 }, 8 },
                                           rhs(), icBox(), center(), 0.0, kTfinal, cfg(), threads );
        benchmark::DoNotOptimize( tree );
        leaves = tree.done().size();
    }
    state.counters["leaves"] = static_cast< double >( leaves );
}
BENCHMARK( BM_RefineVolume )
    ->Arg( 1 )
    ->Arg( 2 )
    ->Arg( 3 )
    ->Arg( 4 )
    ->Unit( benchmark::kMillisecond )
    ->UseRealTime();

BENCHMARK_MAIN();
