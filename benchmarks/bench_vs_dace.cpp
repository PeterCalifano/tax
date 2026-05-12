// SPDX-License-Identifier: BSD-3-Clause
//
// Three-backend bench: tax static (`TE<N>` / `TEn<N,M>`), tax dynamic
// (`DynTE<>`), and DACE — same operations, same expansion points, same
// orders/sizes.  Drives the comparison table in
// `benchmarks/results/tax_vs_dace.md`.
//
// Univariate grid: order N ∈ {5, 10, 20, 40}.
// Multivariate grid: M = 6, N ∈ {2, 4, 6, 8}.

#include <array>
#include <cstddef>
#include <span>
#include <string>

#include <benchmark/benchmark.h>
#include <tax/tax.hpp>

#ifdef TAX_BENCH_HAVE_DACE
#include <dace/dace.h>
#endif

namespace
{

// =============================================================================
// Univariate — static `TE<N>`
// =============================================================================

template < int N >
void BM_Static_Uni_Mul( benchmark::State& s )
{
    auto x = tax::TE< N >::variable( 0.4 );
    auto y = tax::TE< N >::variable( 0.6 );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        benchmark::DoNotOptimize( y );
        tax::TE< N > z = x * y;
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

template < int N >
void BM_Static_Uni_Reciprocal( benchmark::State& s )
{
    auto x = tax::TE< N >::variable( 2.0 );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        tax::TE< N > z = 1.0 / x;
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

template < int N >
void BM_Static_Uni_Sqrt( benchmark::State& s )
{
    auto x = tax::TE< N >::variable( 2.0 );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        tax::TE< N > z = tax::sqrt( x );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

template < int N >
void BM_Static_Uni_Exp( benchmark::State& s )
{
    auto x = tax::TE< N >::variable( 0.4 );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        tax::TE< N > z = tax::exp( x );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

template < int N >
void BM_Static_Uni_Log( benchmark::State& s )
{
    auto x = tax::TE< N >::variable( 2.0 );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        tax::TE< N > z = tax::log( x );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

template < int N >
void BM_Static_Uni_Sin( benchmark::State& s )
{
    auto x = tax::TE< N >::variable( 0.4 );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        tax::TE< N > z = tax::sin( x );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

template < int N >
void BM_Static_Uni_Pow( benchmark::State& s )
{
    auto x = tax::TE< N >::variable( 2.0 );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        tax::TE< N > z = tax::pow( x, 0.5 );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

template < int N >
void BM_Static_Uni_IPow( benchmark::State& s )
{
    auto x = tax::TE< N >::variable( 2.0 );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        tax::TE< N > z = tax::pow( x, 5 );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

// =============================================================================
// Univariate — dynamic `DynTE<>`
// =============================================================================

void BM_Dynamic_Uni_Mul( benchmark::State& s, int N )
{
    auto x = tax::DynTE<>::variable( 0.4, 0, N, 1 );
    auto y = tax::DynTE<>::variable( 0.6, 0, N, 1 );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        benchmark::DoNotOptimize( y );
        auto z = x * y;
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dynamic_Uni_Reciprocal( benchmark::State& s, int N )
{
    auto x = tax::DynTE<>::variable( 2.0, 0, N, 1 );
    auto one = tax::DynTE<>::constant( 1.0, N, 1 );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        auto z = one / x;
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dynamic_Uni_Sqrt( benchmark::State& s, int N )
{
    auto x = tax::DynTE<>::variable( 2.0, 0, N, 1 );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        auto z = tax::sqrt( x );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dynamic_Uni_Exp( benchmark::State& s, int N )
{
    auto x = tax::DynTE<>::variable( 0.4, 0, N, 1 );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        auto z = tax::exp( x );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dynamic_Uni_Log( benchmark::State& s, int N )
{
    auto x = tax::DynTE<>::variable( 2.0, 0, N, 1 );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        auto z = tax::log( x );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dynamic_Uni_Sin( benchmark::State& s, int N )
{
    auto x = tax::DynTE<>::variable( 0.4, 0, N, 1 );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        auto z = tax::sin( x );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dynamic_Uni_Pow( benchmark::State& s, int N )
{
    auto x = tax::DynTE<>::variable( 2.0, 0, N, 1 );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        auto z = tax::pow( x, 0.5 );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dynamic_Uni_IPow( benchmark::State& s, int N )
{
    auto x = tax::DynTE<>::variable( 2.0, 0, N, 1 );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        auto z = tax::pow( x, 5 );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

// =============================================================================
// Univariate — DACE
// =============================================================================

#ifdef TAX_BENCH_HAVE_DACE

void BM_Dace_Uni_Mul( benchmark::State& s, int N )
{
    DACE::DA::init( unsigned( N ), 1 );
    DACE::DA x = DACE::DA( 1 ) + 0.4;
    DACE::DA y = DACE::DA( 1 ) + 0.6;  // DACE has one univariate slot
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        benchmark::DoNotOptimize( y );
        DACE::DA z = x * y;
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dace_Uni_Reciprocal( benchmark::State& s, int N )
{
    DACE::DA::init( unsigned( N ), 1 );
    DACE::DA x = DACE::DA( 1 ) + 2.0;
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        DACE::DA z = 1.0 / x;
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dace_Uni_Sqrt( benchmark::State& s, int N )
{
    DACE::DA::init( unsigned( N ), 1 );
    DACE::DA x = DACE::DA( 1 ) + 2.0;
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        DACE::DA z = x.sqrt();
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dace_Uni_Exp( benchmark::State& s, int N )
{
    DACE::DA::init( unsigned( N ), 1 );
    DACE::DA x = DACE::DA( 1 ) + 0.4;
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        DACE::DA z = x.exp();
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dace_Uni_Log( benchmark::State& s, int N )
{
    DACE::DA::init( unsigned( N ), 1 );
    DACE::DA x = DACE::DA( 1 ) + 2.0;
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        DACE::DA z = x.log();
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dace_Uni_Sin( benchmark::State& s, int N )
{
    DACE::DA::init( unsigned( N ), 1 );
    DACE::DA x = DACE::DA( 1 ) + 0.4;
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        DACE::DA z = x.sin();
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dace_Uni_Pow( benchmark::State& s, int N )
{
    DACE::DA::init( unsigned( N ), 1 );
    DACE::DA x = DACE::DA( 1 ) + 2.0;
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        DACE::DA z = x.pow( 0.5 );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dace_Uni_IPow( benchmark::State& s, int N )
{
    DACE::DA::init( unsigned( N ), 1 );
    DACE::DA x = DACE::DA( 1 ) + 2.0;
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        DACE::DA z = x.pow( 5 );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

#endif  // TAX_BENCH_HAVE_DACE

// =============================================================================
// Multivariate — static `TEn<N, M>`. M is fixed at 6 here.
//
// Operands are *dense* linear combinations of all M variables so each
// backend's recurrence does real work on every coordinate axis (rather than
// dropping out into a near-univariate sub-problem hidden behind a fixed M).
// =============================================================================

constexpr int kM = 6;

// Two distinct dense-in-all-M polynomials so binary ops aren't symmetric.
// Pattern A: c=1.1, alphas={0.10, 0.05, 0.03, 0.02, 0.01, 0.005}
// Pattern B: c=1.2, alphas={0.005, 0.01, 0.02, 0.03, 0.05, 0.10}
constexpr std::array< double, kM > kAlphaA{ 0.10, 0.05, 0.03, 0.02, 0.01, 0.005 };
constexpr std::array< double, kM > kAlphaB{ 0.005, 0.01, 0.02, 0.03, 0.05, 0.10 };

template < int N >
tax::TEn< N, kM > denseStaticOperand( double c, const std::array< double, kM >& alphas )
{
    // Expand around the origin; each `vars[i]` is the variable x_i with a[0]=0.
    typename tax::TEn< N, kM >::Input x0{};
    auto vars = tax::TEn< N, kM >::variables( x0 );
    // out = c + sum_i alphas[i] * x_i  — every linear monomial is nonzero.
    return c + alphas[0] * std::get< 0 >( vars ) + alphas[1] * std::get< 1 >( vars )
           + alphas[2] * std::get< 2 >( vars ) + alphas[3] * std::get< 3 >( vars )
           + alphas[4] * std::get< 4 >( vars ) + alphas[5] * std::get< 5 >( vars );
}

template < int N >
void BM_Static_MV_Mul( benchmark::State& s )
{
    auto x = denseStaticOperand< N >( 1.1, kAlphaA );
    auto y = denseStaticOperand< N >( 1.2, kAlphaB );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        benchmark::DoNotOptimize( y );
        tax::TEn< N, kM > z = x * y;
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

template < int N >
void BM_Static_MV_Reciprocal( benchmark::State& s )
{
    auto x = denseStaticOperand< N >( 1.1, kAlphaA );  // a[0]=1.1 > 0
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        tax::TEn< N, kM > z = 1.0 / x;
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

template < int N >
void BM_Static_MV_Sqrt( benchmark::State& s )
{
    auto x = denseStaticOperand< N >( 1.1, kAlphaA );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        tax::TEn< N, kM > z = tax::sqrt( x );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

template < int N >
void BM_Static_MV_Exp( benchmark::State& s )
{
    auto x = denseStaticOperand< N >( 0.1, kAlphaA );  // small constant so exp stays bounded
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        tax::TEn< N, kM > z = tax::exp( x );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

template < int N >
void BM_Static_MV_Log( benchmark::State& s )
{
    auto x = denseStaticOperand< N >( 1.1, kAlphaA );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        tax::TEn< N, kM > z = tax::log( x );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

template < int N >
void BM_Static_MV_Sin( benchmark::State& s )
{
    auto x = denseStaticOperand< N >( 0.1, kAlphaA );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        tax::TEn< N, kM > z = tax::sin( x );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

template < int N >
void BM_Static_MV_Pow( benchmark::State& s )
{
    auto x = denseStaticOperand< N >( 1.1, kAlphaA );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        tax::TEn< N, kM > z = tax::pow( x, 0.5 );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

template < int N >
void BM_Static_MV_IPow( benchmark::State& s )
{
    auto x = denseStaticOperand< N >( 1.1, kAlphaA );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        tax::TEn< N, kM > z = tax::pow( x, 5 );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

// =============================================================================
// Multivariate — dynamic `DynTE<>`. Same dense operand as static.
// =============================================================================

tax::DynTE<> denseDynamicOperand( int N, double c, const std::array< double, kM >& alphas )
{
    std::vector< double > x0( std::size_t( kM ), 0.0 );
    auto vars = tax::DynTE<>::variables(
        std::span< const double >( x0.data(), x0.size() ), N );
    auto x = tax::DynTE<>::constant( c, N, kM );
    for ( int i = 0; i < int( kM ); ++i ) x = x + alphas[std::size_t( i )] * vars[std::size_t( i )];
    return x;
}

void BM_Dynamic_MV_Mul( benchmark::State& s, int N )
{
    auto x = denseDynamicOperand( N, 1.1, kAlphaA );
    auto y = denseDynamicOperand( N, 1.2, kAlphaB );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        benchmark::DoNotOptimize( y );
        auto z = x * y;
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dynamic_MV_Reciprocal( benchmark::State& s, int N )
{
    auto x = denseDynamicOperand( N, 1.1, kAlphaA );
    auto one = tax::DynTE<>::constant( 1.0, N, kM );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        auto z = one / x;
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dynamic_MV_Sqrt( benchmark::State& s, int N )
{
    auto x = denseDynamicOperand( N, 1.1, kAlphaA );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        auto z = tax::sqrt( x );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dynamic_MV_Exp( benchmark::State& s, int N )
{
    auto x = denseDynamicOperand( N, 0.1, kAlphaA );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        auto z = tax::exp( x );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dynamic_MV_Log( benchmark::State& s, int N )
{
    auto x = denseDynamicOperand( N, 1.1, kAlphaA );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        auto z = tax::log( x );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dynamic_MV_Sin( benchmark::State& s, int N )
{
    auto x = denseDynamicOperand( N, 0.1, kAlphaA );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        auto z = tax::sin( x );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dynamic_MV_Pow( benchmark::State& s, int N )
{
    auto x = denseDynamicOperand( N, 1.1, kAlphaA );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        auto z = tax::pow( x, 0.5 );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dynamic_MV_IPow( benchmark::State& s, int N )
{
    auto x = denseDynamicOperand( N, 1.1, kAlphaA );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        auto z = tax::pow( x, 5 );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

// =============================================================================
// Multivariate — DACE
// =============================================================================

#ifdef TAX_BENCH_HAVE_DACE

// Build the same dense operand on the DACE side:
//   c + sum_i alphas[i] * DACE::DA(i + 1)
// (DACE indexes variables starting at 1.)
DACE::DA denseDaceOperand( double c, const std::array< double, kM >& alphas )
{
    DACE::DA x( c );
    for ( int i = 0; i < int( kM ); ++i )
        x += alphas[std::size_t( i )] * DACE::DA( unsigned( i + 1 ) );
    return x;
}

void BM_Dace_MV_Mul( benchmark::State& s, int N )
{
    DACE::DA::init( unsigned( N ), unsigned( kM ) );
    DACE::DA x = denseDaceOperand( 1.1, kAlphaA );
    DACE::DA y = denseDaceOperand( 1.2, kAlphaB );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        benchmark::DoNotOptimize( y );
        DACE::DA z = x * y;
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dace_MV_Reciprocal( benchmark::State& s, int N )
{
    DACE::DA::init( unsigned( N ), unsigned( kM ) );
    DACE::DA x = denseDaceOperand( 1.1, kAlphaA );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        DACE::DA z = 1.0 / x;
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dace_MV_Sqrt( benchmark::State& s, int N )
{
    DACE::DA::init( unsigned( N ), unsigned( kM ) );
    DACE::DA x = denseDaceOperand( 1.1, kAlphaA );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        DACE::DA z = x.sqrt();
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dace_MV_Exp( benchmark::State& s, int N )
{
    DACE::DA::init( unsigned( N ), unsigned( kM ) );
    DACE::DA x = denseDaceOperand( 0.1, kAlphaA );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        DACE::DA z = x.exp();
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dace_MV_Log( benchmark::State& s, int N )
{
    DACE::DA::init( unsigned( N ), unsigned( kM ) );
    DACE::DA x = denseDaceOperand( 1.1, kAlphaA );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        DACE::DA z = x.log();
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dace_MV_Sin( benchmark::State& s, int N )
{
    DACE::DA::init( unsigned( N ), unsigned( kM ) );
    DACE::DA x = denseDaceOperand( 0.1, kAlphaA );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        DACE::DA z = x.sin();
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dace_MV_Pow( benchmark::State& s, int N )
{
    DACE::DA::init( unsigned( N ), unsigned( kM ) );
    DACE::DA x = denseDaceOperand( 1.1, kAlphaA );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        DACE::DA z = x.pow( 0.5 );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

void BM_Dace_MV_IPow( benchmark::State& s, int N )
{
    DACE::DA::init( unsigned( N ), unsigned( kM ) );
    DACE::DA x = denseDaceOperand( 1.1, kAlphaA );
    for ( auto _ : s )
    {
        benchmark::DoNotOptimize( x );
        DACE::DA z = x.pow( 5 );
        benchmark::DoNotOptimize( z );
        benchmark::ClobberMemory();
    }
}

#endif  // TAX_BENCH_HAVE_DACE

// =============================================================================
// Registration
// =============================================================================

#define TAX_REG( name, fn ) benchmark::RegisterBenchmark( ( name ), ( fn ) )->Unit( benchmark::kNanosecond )

void registerBenchmarks()
{
    // ---- Univariate static ----
    TAX_REG( "Static/Uni/Mul/N5",  &BM_Static_Uni_Mul< 5  > );
    TAX_REG( "Static/Uni/Mul/N10", &BM_Static_Uni_Mul< 10 > );
    TAX_REG( "Static/Uni/Mul/N20", &BM_Static_Uni_Mul< 20 > );
    TAX_REG( "Static/Uni/Mul/N40", &BM_Static_Uni_Mul< 40 > );
    TAX_REG( "Static/Uni/Reciprocal/N5",  &BM_Static_Uni_Reciprocal< 5  > );
    TAX_REG( "Static/Uni/Reciprocal/N10", &BM_Static_Uni_Reciprocal< 10 > );
    TAX_REG( "Static/Uni/Reciprocal/N20", &BM_Static_Uni_Reciprocal< 20 > );
    TAX_REG( "Static/Uni/Reciprocal/N40", &BM_Static_Uni_Reciprocal< 40 > );
    TAX_REG( "Static/Uni/Sqrt/N5",  &BM_Static_Uni_Sqrt< 5  > );
    TAX_REG( "Static/Uni/Sqrt/N10", &BM_Static_Uni_Sqrt< 10 > );
    TAX_REG( "Static/Uni/Sqrt/N20", &BM_Static_Uni_Sqrt< 20 > );
    TAX_REG( "Static/Uni/Sqrt/N40", &BM_Static_Uni_Sqrt< 40 > );
    TAX_REG( "Static/Uni/Exp/N5",  &BM_Static_Uni_Exp< 5  > );
    TAX_REG( "Static/Uni/Exp/N10", &BM_Static_Uni_Exp< 10 > );
    TAX_REG( "Static/Uni/Exp/N20", &BM_Static_Uni_Exp< 20 > );
    TAX_REG( "Static/Uni/Exp/N40", &BM_Static_Uni_Exp< 40 > );
    TAX_REG( "Static/Uni/Log/N5",  &BM_Static_Uni_Log< 5  > );
    TAX_REG( "Static/Uni/Log/N10", &BM_Static_Uni_Log< 10 > );
    TAX_REG( "Static/Uni/Log/N20", &BM_Static_Uni_Log< 20 > );
    TAX_REG( "Static/Uni/Log/N40", &BM_Static_Uni_Log< 40 > );
    TAX_REG( "Static/Uni/Sin/N5",  &BM_Static_Uni_Sin< 5  > );
    TAX_REG( "Static/Uni/Sin/N10", &BM_Static_Uni_Sin< 10 > );
    TAX_REG( "Static/Uni/Sin/N20", &BM_Static_Uni_Sin< 20 > );
    TAX_REG( "Static/Uni/Sin/N40", &BM_Static_Uni_Sin< 40 > );
    TAX_REG( "Static/Uni/Pow/N5",  &BM_Static_Uni_Pow< 5  > );
    TAX_REG( "Static/Uni/Pow/N10", &BM_Static_Uni_Pow< 10 > );
    TAX_REG( "Static/Uni/Pow/N20", &BM_Static_Uni_Pow< 20 > );
    TAX_REG( "Static/Uni/Pow/N40", &BM_Static_Uni_Pow< 40 > );
    TAX_REG( "Static/Uni/IPow/N5",  &BM_Static_Uni_IPow< 5  > );
    TAX_REG( "Static/Uni/IPow/N10", &BM_Static_Uni_IPow< 10 > );
    TAX_REG( "Static/Uni/IPow/N20", &BM_Static_Uni_IPow< 20 > );
    TAX_REG( "Static/Uni/IPow/N40", &BM_Static_Uni_IPow< 40 > );

    // ---- Univariate dynamic ----
    auto regDynUni = []( const char* op, auto fn, int N ) {
        const std::string name = std::string( "Dynamic/Uni/" ) + op + "/N" + std::to_string( N );
        benchmark::RegisterBenchmark( name.c_str(), [fn, N]( benchmark::State& s ) { fn( s, N ); } )
            ->Unit( benchmark::kNanosecond );
    };
    for ( int n : { 5, 10, 20, 40 } )
    {
        regDynUni( "Mul",        &BM_Dynamic_Uni_Mul,        n );
        regDynUni( "Reciprocal", &BM_Dynamic_Uni_Reciprocal, n );
        regDynUni( "Sqrt",       &BM_Dynamic_Uni_Sqrt,       n );
        regDynUni( "Exp",        &BM_Dynamic_Uni_Exp,        n );
        regDynUni( "Log",        &BM_Dynamic_Uni_Log,        n );
        regDynUni( "Sin",        &BM_Dynamic_Uni_Sin,        n );
        regDynUni( "Pow",        &BM_Dynamic_Uni_Pow,        n );
        regDynUni( "IPow",       &BM_Dynamic_Uni_IPow,       n );
    }

#ifdef TAX_BENCH_HAVE_DACE
    // ---- Univariate DACE ----
    auto regDaceUni = []( const char* op, auto fn, int N ) {
        const std::string name = std::string( "Dace/Uni/" ) + op + "/N" + std::to_string( N );
        benchmark::RegisterBenchmark( name.c_str(), [fn, N]( benchmark::State& s ) { fn( s, N ); } )
            ->Unit( benchmark::kNanosecond );
    };
    for ( int n : { 5, 10, 20, 40 } )
    {
        regDaceUni( "Mul",        &BM_Dace_Uni_Mul,        n );
        regDaceUni( "Reciprocal", &BM_Dace_Uni_Reciprocal, n );
        regDaceUni( "Sqrt",       &BM_Dace_Uni_Sqrt,       n );
        regDaceUni( "Exp",        &BM_Dace_Uni_Exp,        n );
        regDaceUni( "Log",        &BM_Dace_Uni_Log,        n );
        regDaceUni( "Sin",        &BM_Dace_Uni_Sin,        n );
        regDaceUni( "Pow",        &BM_Dace_Uni_Pow,        n );
        regDaceUni( "IPow",       &BM_Dace_Uni_IPow,       n );
    }
#endif

    // ---- Multivariate static (M = 6) ----
    TAX_REG( "Static/MV/Mul/N2_M6", &BM_Static_MV_Mul< 2 > );
    TAX_REG( "Static/MV/Mul/N4_M6", &BM_Static_MV_Mul< 4 > );
    TAX_REG( "Static/MV/Mul/N6_M6", &BM_Static_MV_Mul< 6 > );
    TAX_REG( "Static/MV/Mul/N8_M6", &BM_Static_MV_Mul< 8 > );
    TAX_REG( "Static/MV/Reciprocal/N2_M6", &BM_Static_MV_Reciprocal< 2 > );
    TAX_REG( "Static/MV/Reciprocal/N4_M6", &BM_Static_MV_Reciprocal< 4 > );
    TAX_REG( "Static/MV/Reciprocal/N6_M6", &BM_Static_MV_Reciprocal< 6 > );
    TAX_REG( "Static/MV/Reciprocal/N8_M6", &BM_Static_MV_Reciprocal< 8 > );
    TAX_REG( "Static/MV/Sqrt/N2_M6", &BM_Static_MV_Sqrt< 2 > );
    TAX_REG( "Static/MV/Sqrt/N4_M6", &BM_Static_MV_Sqrt< 4 > );
    TAX_REG( "Static/MV/Sqrt/N6_M6", &BM_Static_MV_Sqrt< 6 > );
    TAX_REG( "Static/MV/Sqrt/N8_M6", &BM_Static_MV_Sqrt< 8 > );
    TAX_REG( "Static/MV/Exp/N2_M6", &BM_Static_MV_Exp< 2 > );
    TAX_REG( "Static/MV/Exp/N4_M6", &BM_Static_MV_Exp< 4 > );
    TAX_REG( "Static/MV/Exp/N6_M6", &BM_Static_MV_Exp< 6 > );
    TAX_REG( "Static/MV/Exp/N8_M6", &BM_Static_MV_Exp< 8 > );
    TAX_REG( "Static/MV/Log/N2_M6", &BM_Static_MV_Log< 2 > );
    TAX_REG( "Static/MV/Log/N4_M6", &BM_Static_MV_Log< 4 > );
    TAX_REG( "Static/MV/Log/N6_M6", &BM_Static_MV_Log< 6 > );
    TAX_REG( "Static/MV/Log/N8_M6", &BM_Static_MV_Log< 8 > );
    TAX_REG( "Static/MV/Sin/N2_M6", &BM_Static_MV_Sin< 2 > );
    TAX_REG( "Static/MV/Sin/N4_M6", &BM_Static_MV_Sin< 4 > );
    TAX_REG( "Static/MV/Sin/N6_M6", &BM_Static_MV_Sin< 6 > );
    TAX_REG( "Static/MV/Sin/N8_M6", &BM_Static_MV_Sin< 8 > );
    TAX_REG( "Static/MV/Pow/N2_M6", &BM_Static_MV_Pow< 2 > );
    TAX_REG( "Static/MV/Pow/N4_M6", &BM_Static_MV_Pow< 4 > );
    TAX_REG( "Static/MV/Pow/N6_M6", &BM_Static_MV_Pow< 6 > );
    TAX_REG( "Static/MV/Pow/N8_M6", &BM_Static_MV_Pow< 8 > );
    TAX_REG( "Static/MV/IPow/N2_M6", &BM_Static_MV_IPow< 2 > );
    TAX_REG( "Static/MV/IPow/N4_M6", &BM_Static_MV_IPow< 4 > );
    TAX_REG( "Static/MV/IPow/N6_M6", &BM_Static_MV_IPow< 6 > );
    TAX_REG( "Static/MV/IPow/N8_M6", &BM_Static_MV_IPow< 8 > );

    // ---- Multivariate dynamic ----
    auto regDynMv = []( const char* op, auto fn, int N ) {
        const std::string name =
            std::string( "Dynamic/MV/" ) + op + "/N" + std::to_string( N ) + "_M6";
        benchmark::RegisterBenchmark( name.c_str(), [fn, N]( benchmark::State& s ) { fn( s, N ); } )
            ->Unit( benchmark::kNanosecond );
    };
    for ( int n : { 2, 4, 6, 8 } )
    {
        regDynMv( "Mul",        &BM_Dynamic_MV_Mul,        n );
        regDynMv( "Reciprocal", &BM_Dynamic_MV_Reciprocal, n );
        regDynMv( "Sqrt",       &BM_Dynamic_MV_Sqrt,       n );
        regDynMv( "Exp",        &BM_Dynamic_MV_Exp,        n );
        regDynMv( "Log",        &BM_Dynamic_MV_Log,        n );
        regDynMv( "Sin",        &BM_Dynamic_MV_Sin,        n );
        regDynMv( "Pow",        &BM_Dynamic_MV_Pow,        n );
        regDynMv( "IPow",       &BM_Dynamic_MV_IPow,       n );
    }

#ifdef TAX_BENCH_HAVE_DACE
    // ---- Multivariate DACE ----
    auto regDaceMv = []( const char* op, auto fn, int N ) {
        const std::string name =
            std::string( "Dace/MV/" ) + op + "/N" + std::to_string( N ) + "_M6";
        benchmark::RegisterBenchmark( name.c_str(), [fn, N]( benchmark::State& s ) { fn( s, N ); } )
            ->Unit( benchmark::kNanosecond );
    };
    for ( int n : { 2, 4, 6, 8 } )
    {
        regDaceMv( "Mul",        &BM_Dace_MV_Mul,        n );
        regDaceMv( "Reciprocal", &BM_Dace_MV_Reciprocal, n );
        regDaceMv( "Sqrt",       &BM_Dace_MV_Sqrt,       n );
        regDaceMv( "Exp",        &BM_Dace_MV_Exp,        n );
        regDaceMv( "Log",        &BM_Dace_MV_Log,        n );
        regDaceMv( "Sin",        &BM_Dace_MV_Sin,        n );
        regDaceMv( "Pow",        &BM_Dace_MV_Pow,        n );
        regDaceMv( "IPow",       &BM_Dace_MV_IPow,       n );
    }
#endif
}

#undef TAX_REG

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
