#include <array>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

#include <benchmark/benchmark.h>
#include <tax/tax.hpp>

#ifdef TAX_BENCH_HAVE_DACE
#include <dace/dace.h>
#endif

namespace
{

template < int N, class Op >
void runTaxBenchmark( benchmark::State& state, double x0, Op&& op )
{
    const auto x = tax::TE< N >::variable( x0 );

    for ( auto _ : state )
    {
        tax::TE< N > y = op( x );
        benchmark::DoNotOptimize( y );
        benchmark::ClobberMemory();
    }
}

// =============================================================================
// Raw univariate Cauchy kernels — three variants for direct A/B comparison.
//
// All operate on `std::array<double, N+1>` (matching `tax::TE<N>::Coeffs`)
// and produce identical results modulo floating-point ordering.  Only the
// `loop` variant is wired into the production `cauchyProduct` today; the
// other two are candidates for the M=1 path optimisation.
// =============================================================================

template < int N >
using Buf = std::array< double, std::size_t( N + 1 ) >;

// --- Variant 1: current production path. Triangular loop. -------------------
template < int N >
inline void cauchyUni_Loop( Buf< N >& out, const Buf< N >& f, const Buf< N >& g ) noexcept
{
    out = {};
    for ( int d = 0; d <= N; ++d )
        for ( int k = 0; k <= d; ++k ) out[d] += f[k] * g[d - k];
}

template < int N >
inline void selfUni_Loop( Buf< N >& out, const Buf< N >& f ) noexcept
{
    out = {};
    for ( int d = 0; d <= N; ++d )
    {
        for ( int k = 0; k + k < d; ++k ) out[d] += 2.0 * f[k] * f[d - k];
        if ( d % 2 == 0 ) out[d] += f[d / 2] * f[d / 2];
    }
}

// --- Variant 2: compile-time full unroll via pack expansion. ---------------
//
// Hoisted into named helper templates to side-step a GCC 13 ICE on
// `nested-pack-expansion-inside-generic-lambda`.
template < int N, std::size_t D, std::size_t... Ks >
inline double cauchyUni_UnrollRow( const Buf< N >& f, const Buf< N >& g,
                                   std::index_sequence< Ks... > ) noexcept
{
    return ( ( f[Ks] * g[D - Ks] ) + ... + 0.0 );
}

template < int N, std::size_t... Ds >
inline void cauchyUni_UnrollImpl( Buf< N >& out, const Buf< N >& f, const Buf< N >& g,
                                  std::index_sequence< Ds... > ) noexcept
{
    ( ( out[Ds] = cauchyUni_UnrollRow< N, Ds >(
            f, g, std::make_index_sequence< Ds + 1 >{} ) ),
      ... );
}

template < int N >
inline void cauchyUni_Unroll( Buf< N >& out, const Buf< N >& f, const Buf< N >& g ) noexcept
{
    cauchyUni_UnrollImpl< N >( out, f, g, std::make_index_sequence< N + 1 >{} );
}

template < int N, std::size_t D, std::size_t... Ks >
inline double selfUni_UnrollRow( const Buf< N >& f,
                                 std::index_sequence< Ks... > ) noexcept
{
    // Unordered enumeration: 2·f[k]·f[D-k] for k < D-k, plus f[D/2]² when D is even.
    return ( ( ( 2 * Ks < D ? 2.0 : ( 2 * Ks == D ? 1.0 : 0.0 ) ) * f[Ks] * f[D - Ks] )
             + ... + 0.0 );
}

template < int N, std::size_t... Ds >
inline void selfUni_UnrollImpl( Buf< N >& out, const Buf< N >& f,
                                std::index_sequence< Ds... > ) noexcept
{
    ( ( out[Ds] = selfUni_UnrollRow< N, Ds >(
            f, std::make_index_sequence< Ds / 2 + 1 >{} ) ),
      ... );
}

template < int N >
inline void selfUni_Unroll( Buf< N >& out, const Buf< N >& f ) noexcept
{
    selfUni_UnrollImpl< N >( out, f, std::make_index_sequence< N + 1 >{} );
}

// --- Variant 3: reversed-buffer dot products (forward-only memory access). --
template < int N >
inline void cauchyUni_Reverse( Buf< N >& out, const Buf< N >& f, const Buf< N >& g ) noexcept
{
    Buf< N > g_rev;
    for ( int k = 0; k <= N; ++k ) g_rev[std::size_t( k )] = g[std::size_t( N - k )];

    // g[d-k] = g_rev[N-d+k], so both reads are now strictly forward.
    for ( int d = 0; d <= N; ++d )
    {
        double acc = 0.0;
        const std::size_t off = std::size_t( N - d );
        for ( int k = 0; k <= d; ++k ) acc += f[std::size_t( k )] * g_rev[off + std::size_t( k )];
        out[std::size_t( d )] = acc;
    }
}

template < int N >
inline void selfUni_Reverse( Buf< N >& out, const Buf< N >& f ) noexcept
{
    Buf< N > f_rev;
    for ( int k = 0; k <= N; ++k ) f_rev[std::size_t( k )] = f[std::size_t( N - k )];

    for ( int d = 0; d <= N; ++d )
    {
        double acc = 0.0;
        const std::size_t off = std::size_t( N - d );
        for ( int k = 0; k + k < d; ++k )
            acc += 2.0 * f[std::size_t( k )] * f_rev[off + std::size_t( k )];
        if ( d % 2 == 0 ) acc += f[std::size_t( d / 2 )] * f[std::size_t( d / 2 )];
        out[std::size_t( d )] = acc;
    }
}

template < int N, class Fn >
void runUniKernelBench( benchmark::State& s, Fn&& kernel )
{
    Buf< N > f{}, g{}, out{};
    for ( int i = 0; i <= N; ++i )
    {
        f[std::size_t( i )] = 0.1 + 0.01 * i;
        g[std::size_t( i )] = 0.2 - 0.005 * i;
    }
    for ( auto _ : s )
    {
        kernel( out, f, g );
        benchmark::DoNotOptimize( out );
        benchmark::ClobberMemory();
    }
}

template < int N, class Fn >
void runUniSelfKernelBench( benchmark::State& s, Fn&& kernel )
{
    Buf< N > f{}, out{};
    for ( int i = 0; i <= N; ++i ) f[std::size_t( i )] = 0.1 + 0.01 * i;
    for ( auto _ : s )
    {
        kernel( out, f );
        benchmark::DoNotOptimize( out );
        benchmark::ClobberMemory();
    }
}

#ifdef TAX_BENCH_HAVE_DACE
template < int N, class Op >
void runDaceBenchmark( benchmark::State& state, Op&& op )
{
    DACE::DA::init( N, 1 );
    const DACE::DA xr( 1 );

    for ( auto _ : state )
    {
        DACE::DA yr = op( xr );
        benchmark::DoNotOptimize( yr );
        benchmark::ClobberMemory();
    }
}
#endif

template < int N >
void BM_Tax_Sin( benchmark::State& state )
{
    runTaxBenchmark< N >( state, 0.0, []( const auto& x ) { return tax::sin( x ); } );
}

template < int N >
void BM_Tax_Exp( benchmark::State& state )
{
    runTaxBenchmark< N >( state, 0.0, []( const auto& x ) { return tax::exp( x ); } );
}

template < int N >
void BM_Tax_Log( benchmark::State& state )
{
    runTaxBenchmark< N >( state, 1.0, []( const auto& x ) { return tax::log( x ); } );
}

template < int N >
void BM_Tax_Sqrt( benchmark::State& state )
{
    runTaxBenchmark< N >( state, 2.0, []( const auto& x ) { return tax::sqrt( x ); } );
}

template < int N >
void BM_Tax_IPow( benchmark::State& state )
{
    runTaxBenchmark< N >( state, 2.0, []( const auto& x ) { return tax::pow( x, 5 ); } );
}

template < int N >
void BM_Tax_Pow( benchmark::State& state )
{
    runTaxBenchmark< N >( state, 2.0, []( const auto& x ) { return tax::pow( x, 0.5 ); } );
}

template < int N >
void BM_Tax_Mul( benchmark::State& state )
{
    auto x = tax::TE< N >::variable( 0.4 );
    auto y = tax::TE< N >::variable( 0.6 );
    for ( auto _ : state )
    {
        benchmark::DoNotOptimize( x );
        benchmark::DoNotOptimize( y );
        tax::TE< N > z = x * y;
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

template < int N >
void BM_Tax_Square( benchmark::State& state )
{
    auto x = tax::TE< N >::variable( 0.4 );
    for ( auto _ : state )
    {
        benchmark::DoNotOptimize( x );
        tax::TE< N > z = tax::square( x );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

#ifdef TAX_BENCH_HAVE_DACE
template < int N >
void BM_Dace_Sin( benchmark::State& state )
{
    runDaceBenchmark< N >( state, []( const DACE::DA& xr ) { return xr.sin(); } );
}

template < int N >
void BM_Dace_Exp( benchmark::State& state )
{
    runDaceBenchmark< N >( state, []( const DACE::DA& xr ) { return xr.exp(); } );
}

template < int N >
void BM_Dace_Log( benchmark::State& state )
{
    runDaceBenchmark< N >( state, []( const DACE::DA& xr ) { return ( 1.0 + xr ).log(); } );
}

template < int N >
void BM_Dace_Sqrt( benchmark::State& state )
{
    runDaceBenchmark< N >( state, []( const DACE::DA& xr ) { return ( 2.0 + xr ).sqrt(); } );
}

template < int N >
void BM_Dace_IPow( benchmark::State& state )
{
    runDaceBenchmark< N >( state, []( const DACE::DA& xr ) { return ( 2.0 + xr ).pow( 5 ); } );
}

template < int N >
void BM_Dace_Pow( benchmark::State& state )
{
    runDaceBenchmark< N >( state, []( const DACE::DA& xr ) { return ( 2.0 + xr ).pow( 0.5 ); } );
}
#endif

void registerBenchmarks()
{
    auto reg = []( const char* name, auto fn ) {
        benchmark::RegisterBenchmark( name, fn )->Unit( benchmark::kMicrosecond );
    };

    reg( "Tax/Sin/N10", &BM_Tax_Sin< 10 > );
    reg( "Tax/Sin/N20", &BM_Tax_Sin< 20 > );
    reg( "Tax/Sin/N40", &BM_Tax_Sin< 40 > );

    reg( "Tax/Exp/N10", &BM_Tax_Exp< 10 > );
    reg( "Tax/Exp/N20", &BM_Tax_Exp< 20 > );
    reg( "Tax/Exp/N40", &BM_Tax_Exp< 40 > );

    reg( "Tax/Log/N10", &BM_Tax_Log< 10 > );
    reg( "Tax/Log/N20", &BM_Tax_Log< 20 > );
    reg( "Tax/Log/N40", &BM_Tax_Log< 40 > );

    reg( "Tax/Sqrt/N10", &BM_Tax_Sqrt< 10 > );
    reg( "Tax/Sqrt/N20", &BM_Tax_Sqrt< 20 > );
    reg( "Tax/Sqrt/N40", &BM_Tax_Sqrt< 40 > );

    reg( "Tax/IPow/N10", &BM_Tax_IPow< 10 > );
    reg( "Tax/IPow/N20", &BM_Tax_IPow< 20 > );
    reg( "Tax/IPow/N40", &BM_Tax_IPow< 40 > );

    reg( "Tax/Pow/N10", &BM_Tax_Pow< 10 > );
    reg( "Tax/Pow/N20", &BM_Tax_Pow< 20 > );
    reg( "Tax/Pow/N40", &BM_Tax_Pow< 40 > );

    reg( "Tax/Mul/N5",  &BM_Tax_Mul< 5 > );
    reg( "Tax/Mul/N10", &BM_Tax_Mul< 10 > );
    reg( "Tax/Mul/N20", &BM_Tax_Mul< 20 > );
    reg( "Tax/Mul/N40", &BM_Tax_Mul< 40 > );

    reg( "Tax/Square/N5",  &BM_Tax_Square< 5 > );
    reg( "Tax/Square/N10", &BM_Tax_Square< 10 > );
    reg( "Tax/Square/N20", &BM_Tax_Square< 20 > );
    reg( "Tax/Square/N40", &BM_Tax_Square< 40 > );

    // --- Raw univariate Cauchy kernels (Loop vs Unroll vs Reverse). ---------
    auto regKernel = [&]( const char* name, auto fn ) {
        benchmark::RegisterBenchmark( name, fn )->Unit( benchmark::kNanosecond );
    };
    auto regMul = [&]< int N >( std::integral_constant< int, N > ) {
        regKernel( ( "Tax/UniMul/Loop/N"    + std::to_string( N ) ).c_str(),
                   []( benchmark::State& s ) { runUniKernelBench< N >( s, &cauchyUni_Loop< N > ); } );
        regKernel( ( "Tax/UniMul/Unroll/N"  + std::to_string( N ) ).c_str(),
                   []( benchmark::State& s ) { runUniKernelBench< N >( s, &cauchyUni_Unroll< N > ); } );
        regKernel( ( "Tax/UniMul/Reverse/N" + std::to_string( N ) ).c_str(),
                   []( benchmark::State& s ) { runUniKernelBench< N >( s, &cauchyUni_Reverse< N > ); } );
    };
    auto regSelf = [&]< int N >( std::integral_constant< int, N > ) {
        regKernel( ( "Tax/UniSquare/Loop/N"    + std::to_string( N ) ).c_str(),
                   []( benchmark::State& s ) { runUniSelfKernelBench< N >( s, &selfUni_Loop< N > ); } );
        regKernel( ( "Tax/UniSquare/Unroll/N"  + std::to_string( N ) ).c_str(),
                   []( benchmark::State& s ) { runUniSelfKernelBench< N >( s, &selfUni_Unroll< N > ); } );
        regKernel( ( "Tax/UniSquare/Reverse/N" + std::to_string( N ) ).c_str(),
                   []( benchmark::State& s ) { runUniSelfKernelBench< N >( s, &selfUni_Reverse< N > ); } );
    };
    regMul( std::integral_constant< int, 5 >{} );
    regMul( std::integral_constant< int, 10 >{} );
    regMul( std::integral_constant< int, 20 >{} );
    regMul( std::integral_constant< int, 40 >{} );
    regSelf( std::integral_constant< int, 5 >{} );
    regSelf( std::integral_constant< int, 10 >{} );
    regSelf( std::integral_constant< int, 20 >{} );
    regSelf( std::integral_constant< int, 40 >{} );

#ifdef TAX_BENCH_HAVE_DACE
    reg( "Dace/Sin/N10", &BM_Dace_Sin< 10 > );
    reg( "Dace/Sin/N20", &BM_Dace_Sin< 20 > );
    reg( "Dace/Sin/N40", &BM_Dace_Sin< 40 > );

    reg( "Dace/Exp/N10", &BM_Dace_Exp< 10 > );
    reg( "Dace/Exp/N20", &BM_Dace_Exp< 20 > );
    reg( "Dace/Exp/N40", &BM_Dace_Exp< 40 > );

    reg( "Dace/Log/N10", &BM_Dace_Log< 10 > );
    reg( "Dace/Log/N20", &BM_Dace_Log< 20 > );
    reg( "Dace/Log/N40", &BM_Dace_Log< 40 > );

    reg( "Dace/Sqrt/N10", &BM_Dace_Sqrt< 10 > );
    reg( "Dace/Sqrt/N20", &BM_Dace_Sqrt< 20 > );
    reg( "Dace/Sqrt/N40", &BM_Dace_Sqrt< 40 > );

    reg( "Dace/IPow/N10", &BM_Dace_IPow< 10 > );
    reg( "Dace/IPow/N20", &BM_Dace_IPow< 20 > );
    reg( "Dace/IPow/N40", &BM_Dace_IPow< 40 > );

    reg( "Dace/Pow/N10", &BM_Dace_Pow< 10 > );
    reg( "Dace/Pow/N20", &BM_Dace_Pow< 20 > );
    reg( "Dace/Pow/N40", &BM_Dace_Pow< 40 > );
#endif
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
