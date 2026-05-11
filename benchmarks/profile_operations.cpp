// SPDX-License-Identifier: BSD-3-Clause
//
// Per-operation profiling driver. Walks the full arithmetic + math surface
// against a univariate static TaylorExpansion (`TE<10>`) and a multivariate
// one (`TEn<5, 3>`) and prints one row per op with per-call timing.
//
// Plain `int main()` — no Google Benchmark dependency — so the binary plays
// nicely with `perf`, `callgrind`, `vtune`, etc.:
//
//   perf record -g --call-graph dwarf -- \
//       ./build/benchmarks/profile_operations 200000 20000
//   perf report
//
// Two CLI args:
//   argv[1]   univariate iterations per op   (default 100000)
//   argv[2]   multivariate iterations per op (default 10000)

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <tax/tax.hpp>

using clock_t_ = std::chrono::steady_clock;

namespace
{

constexpr int kN_uni = 10;        ///< Univariate truncation order.
constexpr int kN_mv = 5;          ///< Multivariate truncation order.
constexpr int kM_mv = 3;          ///< Multivariate variable count.

using TeU = tax::TE< kN_uni >;
using TeM = tax::TEn< kN_mv, kM_mv >;

// volatile sink so the compiler can't dead-code-eliminate the result of
// each op.  We touch the constant term so the operation must materialise.
volatile double g_sink = 0.0;

template < typename TTE, typename Fn >
[[gnu::noinline]] double time_op( int iters, const TTE& a, const TTE& b, Fn&& op )
{
    const auto t0 = clock_t_::now();
    for ( int i = 0; i < iters; ++i )
    {
        TTE y = op( a, b );
        g_sink += y[0];
    }
    const auto t1 = clock_t_::now();
    const auto us = std::chrono::duration_cast< std::chrono::nanoseconds >( t1 - t0 ).count();
    return double( us ) / double( iters ) / 1000.0;  // μs per call
}

template < typename TTE, typename Fn >
[[gnu::noinline]] double time_unary( int iters, const TTE& a, Fn&& op )
{
    const auto t0 = clock_t_::now();
    for ( int i = 0; i < iters; ++i )
    {
        TTE y = op( a );
        g_sink += y[0];
    }
    const auto t1 = clock_t_::now();
    const auto us = std::chrono::duration_cast< std::chrono::nanoseconds >( t1 - t0 ).count();
    return double( us ) / double( iters ) / 1000.0;
}

// -----------------------------------------------------------------------------
// Per-row dispatcher.  Takes two callables — one for the univariate operand
// type, one for the multivariate — and runs them through `time_*`.  We do not
// share a single generic lambda across both shapes so each TTE's compile-time
// path is preserved (and we don't accidentally pessimise via type erasure).
// -----------------------------------------------------------------------------

template < typename FU, typename FM >
void row_unary( const char* name, int iters_u, int iters_m, const TeU& xu,
                const TeM& xm, FU fu, FM fm )
{
    const double tu = time_unary( iters_u, xu, fu );
    const double tm = time_unary( iters_m, xm, fm );
    std::printf( "  %-22s  %10.4f   %10.4f\n", name, tu, tm );
}

template < typename FU, typename FM >
void row_binary( const char* name, int iters_u, int iters_m, const TeU& au,
                 const TeU& bu, const TeM& am, const TeM& bm, FU fu, FM fm )
{
    const double tu = time_op( iters_u, au, bu, fu );
    const double tm = time_op( iters_m, am, bm, fm );
    std::printf( "  %-22s  %10.4f   %10.4f\n", name, tu, tm );
}

}  // namespace

int main( int argc, char** argv )
{
    int iters_u = 100000;
    int iters_m = 10000;
    if ( argc >= 2 )
        try
        {
            const int n = std::stoi( argv[1] );
            if ( n > 0 ) iters_u = n;
        }
        catch ( ... )
        {
        }
    if ( argc >= 3 )
        try
        {
            const int n = std::stoi( argv[2] );
            if ( n > 0 ) iters_m = n;
        }
        catch ( ... )
        {
        }

    // Operands — chosen so every function in the set is well-defined at the
    // expansion point (|x| < 1 for asin/atanh, x > 0 for log/sqrt, etc.).
    const auto xu = TeU::variable( 0.4 );
    const auto yu = TeU::variable( 0.6 );
    auto [xm, ym, zm] = TeM::variables( 0.4, 0.3, 0.2 );

    std::printf( "profile_operations: univariate TE<%d, 1> vs multivariate "
                 "TEn<%d, %d>  iters_u=%d  iters_m=%d  "
                 "(per-call time, microseconds)\n\n",
                 kN_uni, kN_mv, kM_mv, iters_u, iters_m );
    std::printf( "  %-22s  %10s   %10s\n", "operation", "univariate", "multivariate" );
    std::printf( "  %s\n", "------------------------------------------------------" );

    // ---- Arithmetic between two TEs --------------------------------------
    row_binary(
        "TE + TE", iters_u, iters_m, xu, yu, xm, ym,
        []( const TeU& a, const TeU& b ) { return TeU{ a + b }; },
        []( const TeM& a, const TeM& b ) { return TeM{ a + b }; } );
    row_binary(
        "TE - TE", iters_u, iters_m, xu, yu, xm, ym,
        []( const TeU& a, const TeU& b ) { return TeU{ a - b }; },
        []( const TeM& a, const TeM& b ) { return TeM{ a - b }; } );
    row_binary(
        "TE * TE", iters_u, iters_m, xu, yu, xm, ym,
        []( const TeU& a, const TeU& b ) { return TeU{ a * b }; },
        []( const TeM& a, const TeM& b ) { return TeM{ a * b }; } );
    row_binary(
        "TE / TE (xu=0.4+, ym>0)", iters_u, iters_m, xu, yu, xm, ym,
        []( const TeU& a, const TeU& b ) { return TeU{ a / ( b + 1.0 ) }; },
        []( const TeM& a, const TeM& b ) { return TeM{ a / ( b + 1.0 ) }; } );

    // ---- TE ⊕ scalar -----------------------------------------------------
    row_unary(
        "TE + double", iters_u, iters_m, xu, xm,
        []( const TeU& a ) { return TeU{ a + 1.5 }; },
        []( const TeM& a ) { return TeM{ a + 1.5 }; } );
    row_unary(
        "TE * double", iters_u, iters_m, xu, xm,
        []( const TeU& a ) { return TeU{ a * 2.0 }; },
        []( const TeM& a ) { return TeM{ a * 2.0 }; } );
    row_unary(
        "-TE (unary minus)", iters_u, iters_m, xu, xm,
        []( const TeU& a ) { return TeU{ -a }; },
        []( const TeM& a ) { return TeM{ -a }; } );

    // ---- Trig ------------------------------------------------------------
    row_unary(
        "sin", iters_u, iters_m, xu, xm,
        []( const TeU& a ) { return TeU{ tax::sin( a ) }; },
        []( const TeM& a ) { return TeM{ tax::sin( a ) }; } );
    row_unary(
        "cos", iters_u, iters_m, xu, xm,
        []( const TeU& a ) { return TeU{ tax::cos( a ) }; },
        []( const TeM& a ) { return TeM{ tax::cos( a ) }; } );
    row_unary(
        "tan", iters_u, iters_m, xu, xm,
        []( const TeU& a ) { return TeU{ tax::tan( a ) }; },
        []( const TeM& a ) { return TeM{ tax::tan( a ) }; } );

    // ---- Hyperbolic -------------------------------------------------------
    row_unary(
        "sinh", iters_u, iters_m, xu, xm,
        []( const TeU& a ) { return TeU{ tax::sinh( a ) }; },
        []( const TeM& a ) { return TeM{ tax::sinh( a ) }; } );
    row_unary(
        "cosh", iters_u, iters_m, xu, xm,
        []( const TeU& a ) { return TeU{ tax::cosh( a ) }; },
        []( const TeM& a ) { return TeM{ tax::cosh( a ) }; } );
    row_unary(
        "tanh", iters_u, iters_m, xu, xm,
        []( const TeU& a ) { return TeU{ tax::tanh( a ) }; },
        []( const TeM& a ) { return TeM{ tax::tanh( a ) }; } );

    // ---- Inverse trig (operands chosen so |a[0]| < 1) --------------------
    row_unary(
        "asin", iters_u, iters_m, xu, xm,
        []( const TeU& a ) { return TeU{ tax::asin( a ) }; },
        []( const TeM& a ) { return TeM{ tax::asin( a ) }; } );
    row_unary(
        "acos", iters_u, iters_m, xu, xm,
        []( const TeU& a ) { return TeU{ tax::acos( a ) }; },
        []( const TeM& a ) { return TeM{ tax::acos( a ) }; } );
    row_unary(
        "atan", iters_u, iters_m, xu, xm,
        []( const TeU& a ) { return TeU{ tax::atan( a ) }; },
        []( const TeM& a ) { return TeM{ tax::atan( a ) }; } );

    // ---- Inverse hyperbolic ---------------------------------------------
    row_unary(
        "asinh", iters_u, iters_m, xu, xm,
        []( const TeU& a ) { return TeU{ tax::asinh( a ) }; },
        []( const TeM& a ) { return TeM{ tax::asinh( a ) }; } );
    // acosh: requires a[0] > 1; bump operands.
    {
        const auto xu2 = TeU::variable( 2.0 );
        const auto xm2 = std::get< 0 >( TeM::variables( 2.0, 0.3, 0.2 ) );
        row_unary(
            "acosh (a[0]=2)", iters_u, iters_m, xu2, xm2,
            []( const TeU& a ) { return TeU{ tax::acosh( a ) }; },
            []( const TeM& a ) { return TeM{ tax::acosh( a ) }; } );
    }
    row_unary(
        "atanh", iters_u, iters_m, xu, xm,
        []( const TeU& a ) { return TeU{ tax::atanh( a ) }; },
        []( const TeM& a ) { return TeM{ tax::atanh( a ) }; } );

    // ---- exp / log -------------------------------------------------------
    row_unary(
        "exp", iters_u, iters_m, xu, xm,
        []( const TeU& a ) { return TeU{ tax::exp( a ) }; },
        []( const TeM& a ) { return TeM{ tax::exp( a ) }; } );
    {
        const auto xu_pos = TeU::variable( 2.0 );
        const auto xm_pos = std::get< 0 >( TeM::variables( 2.0, 0.3, 0.2 ) );
        row_unary(
            "log (a[0]=2)", iters_u, iters_m, xu_pos, xm_pos,
            []( const TeU& a ) { return TeU{ tax::log( a ) }; },
            []( const TeM& a ) { return TeM{ tax::log( a ) }; } );
        row_unary(
            "log10 (a[0]=2)", iters_u, iters_m, xu_pos, xm_pos,
            []( const TeU& a ) { return TeU{ tax::log10( a ) }; },
            []( const TeM& a ) { return TeM{ tax::log10( a ) }; } );
        row_unary(
            "sqrt (a[0]=2)", iters_u, iters_m, xu_pos, xm_pos,
            []( const TeU& a ) { return TeU{ tax::sqrt( a ) }; },
            []( const TeM& a ) { return TeM{ tax::sqrt( a ) }; } );
        row_unary(
            "cbrt (a[0]=2)", iters_u, iters_m, xu_pos, xm_pos,
            []( const TeU& a ) { return TeU{ tax::cbrt( a ) }; },
            []( const TeM& a ) { return TeM{ tax::cbrt( a ) }; } );

        row_unary(
            "pow(., 0.5)", iters_u, iters_m, xu_pos, xm_pos,
            []( const TeU& a ) { return TeU{ tax::pow( a, 0.5 ) }; },
            []( const TeM& a ) { return TeM{ tax::pow( a, 0.5 ) }; } );
        row_unary(
            "pow(., 5) int", iters_u, iters_m, xu_pos, xm_pos,
            []( const TeU& a ) { return TeU{ tax::pow( a, 5 ) }; },
            []( const TeM& a ) { return TeM{ tax::pow( a, 5 ) }; } );
    }

    // ---- Squares / cubes / abs / erf -----------------------------------
    row_unary(
        "square", iters_u, iters_m, xu, xm,
        []( const TeU& a ) { return TeU{ tax::square( a ) }; },
        []( const TeM& a ) { return TeM{ tax::square( a ) }; } );
    row_unary(
        "cube", iters_u, iters_m, xu, xm,
        []( const TeU& a ) { return TeU{ tax::cube( a ) }; },
        []( const TeM& a ) { return TeM{ tax::cube( a ) }; } );
    {
        // abs requires a[0] != 0.
        const auto xu_pos = TeU::variable( 0.7 );
        const auto xm_pos = std::get< 0 >( TeM::variables( 0.7, 0.3, 0.2 ) );
        row_unary(
            "abs (a[0]=0.7)", iters_u, iters_m, xu_pos, xm_pos,
            []( const TeU& a ) { return TeU{ tax::abs( a ) }; },
            []( const TeM& a ) { return TeM{ tax::abs( a ) }; } );
    }
    row_unary(
        "erf", iters_u, iters_m, xu, xm,
        []( const TeU& a ) { return TeU{ tax::erf( a ) }; },
        []( const TeM& a ) { return TeM{ tax::erf( a ) }; } );

    // ---- Two-argument math ----------------------------------------------
    row_binary(
        "atan2(y, x)", iters_u, iters_m, xu, yu, xm, ym,
        []( const TeU& y, const TeU& x ) { return TeU{ tax::atan2( y, x + 1.0 ) }; },
        []( const TeM& y, const TeM& x ) { return TeM{ tax::atan2( y, x + 1.0 ) }; } );
    row_binary(
        "hypot(x, y)", iters_u, iters_m, xu, yu, xm, ym,
        []( const TeU& a, const TeU& b ) { return TeU{ tax::hypot( a, b ) }; },
        []( const TeM& a, const TeM& b ) { return TeM{ tax::hypot( a, b ) }; } );

    std::printf( "\nsink=%g\n", ( double )g_sink );
    return 0;
}
