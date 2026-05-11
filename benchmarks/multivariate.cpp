// Google Benchmark sweep for the multivariate static path — the only place
// the precomputed CauchyStencil / CauchySymStencil actually fires.
//
// Coverage: pure Cauchy product (`a * b`), self-product (`square`), cube
// (one self + one general Cauchy), integer power 5 (binary-exponentiation
// chain of Cauchy multiplies), and a couple of transcendentals whose
// recurrence opens with a `cauchySelfProduct` call (`asin`, `atan`, `atan2`,
// `erf`). All those should track the stencil's performance.
//
// Shapes chosen to cover both small (L1-fit) and large (L3-fit) stencils:
//   (N=5, M=3) — profile_operations_multivariate default, ~5 KiB
//   (N=4, M=4) — DA outer order for ODE/ADS,            ~4 KiB
//   (N=5, M=4) — wider DA box,                         ~30 KiB
//   (N=6, M=3) — slightly higher order,                ~15 KiB

#include <benchmark/benchmark.h>

#include <tax/tax.hpp>

namespace
{

template < int N, int M >
auto makeVars()
{
    typename tax::TEn< N, M >::Input x0{};
    for ( int i = 0; i < M; ++i ) x0[i] = 0.1 * double( i + 1 );
    return tax::TEn< N, M >::variables( x0 );
}

template < int N, int M >
void BM_MV_Mul( benchmark::State& s )
{
    auto vars = makeVars< N, M >();
    const auto x = std::get< 0 >( vars );
    const auto y = std::get< 1 >( vars );
    for ( auto _ : s )
    {
        tax::TEn< N, M > z = x * y;
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

template < int N, int M >
void BM_MV_Square( benchmark::State& s )
{
    auto vars = makeVars< N, M >();
    const auto x = std::get< 0 >( vars );
    for ( auto _ : s )
    {
        tax::TEn< N, M > z = tax::square( x );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

template < int N, int M >
void BM_MV_Cube( benchmark::State& s )
{
    auto vars = makeVars< N, M >();
    const auto x = std::get< 0 >( vars );
    for ( auto _ : s )
    {
        tax::TEn< N, M > z = tax::cube( x );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

template < int N, int M >
void BM_MV_IPow5( benchmark::State& s )
{
    auto vars = makeVars< N, M >();
    const auto x = std::get< 0 >( vars );
    for ( auto _ : s )
    {
        tax::TEn< N, M > z = tax::pow( x, 5 );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

template < int N, int M >
void BM_MV_Asin( benchmark::State& s )
{
    auto vars = makeVars< N, M >();
    const auto x = std::get< 0 >( vars );
    for ( auto _ : s )
    {
        tax::TEn< N, M > z = tax::asin( x );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

template < int N, int M >
void BM_MV_Atan( benchmark::State& s )
{
    auto vars = makeVars< N, M >();
    const auto x = std::get< 0 >( vars );
    for ( auto _ : s )
    {
        tax::TEn< N, M > z = tax::atan( x );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

template < int N, int M >
void BM_MV_Atan2( benchmark::State& s )
{
    auto vars = makeVars< N, M >();
    const auto x = std::get< 0 >( vars );
    const auto y = std::get< 1 >( vars );
    for ( auto _ : s )
    {
        tax::TEn< N, M > z = tax::atan2( x, y + 1.0 );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

template < int N, int M >
void BM_MV_Erf( benchmark::State& s )
{
    auto vars = makeVars< N, M >();
    const auto x = std::get< 0 >( vars );
    for ( auto _ : s )
    {
        tax::TEn< N, M > z = tax::erf( x );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

#define TAX_REG_SHAPE( N, M )                                                  \
    do                                                                         \
    {                                                                          \
        auto reg = []( const char* name, auto fn ) {                           \
            benchmark::RegisterBenchmark( name, fn )->Unit( benchmark::kNanosecond ); \
        };                                                                     \
        reg( "MV/Mul/N" #N "_M" #M, &BM_MV_Mul< N, M > );                      \
        reg( "MV/Square/N" #N "_M" #M, &BM_MV_Square< N, M > );                \
        reg( "MV/Cube/N" #N "_M" #M, &BM_MV_Cube< N, M > );                    \
        reg( "MV/IPow5/N" #N "_M" #M, &BM_MV_IPow5< N, M > );                  \
        reg( "MV/Asin/N" #N "_M" #M, &BM_MV_Asin< N, M > );                    \
        reg( "MV/Atan/N" #N "_M" #M, &BM_MV_Atan< N, M > );                    \
        reg( "MV/Atan2/N" #N "_M" #M, &BM_MV_Atan2< N, M > );                  \
        reg( "MV/Erf/N" #N "_M" #M, &BM_MV_Erf< N, M > );                      \
    } while ( 0 )

void registerBenchmarks()
{
    TAX_REG_SHAPE( 5, 3 );
    TAX_REG_SHAPE( 4, 4 );
    TAX_REG_SHAPE( 5, 4 );
    TAX_REG_SHAPE( 6, 3 );
}

}  // namespace

int main( int argc, char** argv )
{
    benchmark::Initialize( &argc, argv );
    if ( benchmark::ReportUnrecognizedArguments( argc, argv ) ) return 1;
    registerBenchmarks();
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
