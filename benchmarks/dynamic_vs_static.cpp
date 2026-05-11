// Compare the runtime overhead of the dynamic TaylorExpansionT<T,Dynamic,Dynamic>
// against the static TaylorExpansionT<T,N,M> on identical workloads. Both run
// through their respective evaluation paths (static: expression-template fusion;
// dynamic: eager kernel calls with std::vector storage).

#include <benchmark/benchmark.h>
#include <array>
#include <span>
#include <tax/tax.hpp>

namespace
{

// =============================================================================
// Univariate workloads
// =============================================================================

template < int N >
void BM_Static_Uni_Sin( benchmark::State& s )
{
    const auto x = tax::TE< N >::variable( 0.1 );
    for ( auto _ : s )
    {
        tax::TE< N > y = tax::sin( x );
        benchmark::DoNotOptimize( y );
        benchmark::ClobberMemory();
    }
}

template < int N >
void BM_Dynamic_Uni_Sin( benchmark::State& s )
{
    const auto x = tax::DynTE<>::variable( 0.1, /*var_idx=*/0, /*order=*/N, /*size=*/1 );
    for ( auto _ : s )
    {
        auto y = tax::sin( x );
        benchmark::DoNotOptimize( y );
        benchmark::ClobberMemory();
    }
}

template < int N >
void BM_Static_Uni_Composite( benchmark::State& s )
{
    const auto x = tax::TE< N >::variable( 0.1 );
    for ( auto _ : s )
    {
        tax::TE< N > y = tax::sin( x ) * tax::exp( x ) + tax::log( x + 1.0 );
        benchmark::DoNotOptimize( y );
        benchmark::ClobberMemory();
    }
}

template < int N >
void BM_Dynamic_Uni_Composite( benchmark::State& s )
{
    const auto x = tax::DynTE<>::variable( 0.1, 0, N, 1 );
    for ( auto _ : s )
    {
        auto y = tax::sin( x ) * tax::exp( x ) + tax::log( x + 1.0 );
        benchmark::DoNotOptimize( y );
        benchmark::ClobberMemory();
    }
}

// =============================================================================
// Multivariate workloads
// =============================================================================

template < int N, int M >
void BM_Static_MV_Mul( benchmark::State& s )
{
    typename tax::TEn< N, M >::Input x0{};
    for ( int i = 0; i < M; ++i ) x0[i] = 0.1 * i;
    auto vars = tax::TEn< N, M >::variables( x0 );
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
void BM_Dynamic_MV_Mul( benchmark::State& s )
{
    std::array< double, M > x0{};
    for ( int i = 0; i < M; ++i ) x0[i] = 0.1 * i;
    auto vars = tax::DynTE<>::variables( std::span< const double >( x0 ), N );
    const auto x = vars[0];
    const auto y = vars[1];
    for ( auto _ : s )
    {
        auto z = x * y;
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

template < int N, int M >
void BM_Static_MV_Composite( benchmark::State& s )
{
    typename tax::TEn< N, M >::Input x0{};
    for ( int i = 0; i < M; ++i ) x0[i] = 0.1 * i;
    auto vars = tax::TEn< N, M >::variables( x0 );
    const auto x = std::get< 0 >( vars );
    const auto y = std::get< 1 >( vars );
    for ( auto _ : s )
    {
        tax::TEn< N, M > z = tax::sin( x * y ) + tax::exp( x + y );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

template < int N, int M >
void BM_Dynamic_MV_Composite( benchmark::State& s )
{
    std::array< double, M > x0{};
    for ( int i = 0; i < M; ++i ) x0[i] = 0.1 * i;
    auto vars = tax::DynTE<>::variables( std::span< const double >( x0 ), N );
    const auto x = vars[0];
    const auto y = vars[1];
    for ( auto _ : s )
    {
        auto z = tax::sin( x * y ) + tax::exp( x + y );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void registerBenchmarks()
{
    auto reg = []( const char* name, auto fn ) {
        benchmark::RegisterBenchmark( name, fn )->Unit( benchmark::kMicrosecond );
    };

    reg( "Static/Uni/Sin/N10", &BM_Static_Uni_Sin< 10 > );
    reg( "Dynamic/Uni/Sin/N10", &BM_Dynamic_Uni_Sin< 10 > );
    reg( "Static/Uni/Sin/N40", &BM_Static_Uni_Sin< 40 > );
    reg( "Dynamic/Uni/Sin/N40", &BM_Dynamic_Uni_Sin< 40 > );

    reg( "Static/Uni/Composite/N10", &BM_Static_Uni_Composite< 10 > );
    reg( "Dynamic/Uni/Composite/N10", &BM_Dynamic_Uni_Composite< 10 > );
    reg( "Static/Uni/Composite/N40", &BM_Static_Uni_Composite< 40 > );
    reg( "Dynamic/Uni/Composite/N40", &BM_Dynamic_Uni_Composite< 40 > );

    reg( "Static/MV/Mul/N5_M2", &BM_Static_MV_Mul< 5, 2 > );
    reg( "Dynamic/MV/Mul/N5_M2", &BM_Dynamic_MV_Mul< 5, 2 > );
    reg( "Static/MV/Mul/N5_M4", &BM_Static_MV_Mul< 5, 4 > );
    reg( "Dynamic/MV/Mul/N5_M4", &BM_Dynamic_MV_Mul< 5, 4 > );

    reg( "Static/MV/Composite/N5_M2", &BM_Static_MV_Composite< 5, 2 > );
    reg( "Dynamic/MV/Composite/N5_M2", &BM_Dynamic_MV_Composite< 5, 2 > );
    reg( "Static/MV/Composite/N5_M4", &BM_Static_MV_Composite< 5, 4 > );
    reg( "Dynamic/MV/Composite/N5_M4", &BM_Dynamic_MV_Composite< 5, 4 > );
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
