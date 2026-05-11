// SPDX-License-Identifier: BSD-3-Clause
//
// Per-operation profiling driver — multivariate.  Walks the full arithmetic
// + math surface against a static multivariate TaylorExpansion `TEn<5, 3>`
// and prints one row per op with per-call timing.
//
// Plain `int main()` — no Google Benchmark dependency — so the binary plays
// nicely with `perf`, `callgrind`, `vtune`, etc.:
//
//   perf record -g --call-graph dwarf -- \
//       ./build/benchmarks/profile_operations_multivariate 20000
//   perf report
//
// CLI arg:
//   argv[1]   iterations per op   (default 10000)

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <tax/tax.hpp>

using clock_t_ = std::chrono::steady_clock;

namespace
{

constexpr int kN = 5;  ///< Multivariate truncation order.
constexpr int kM = 3;  ///< Multivariate variable count.

using Te = tax::TEn< kN, kM >;

// volatile sink so the compiler can't dead-code-eliminate the result of
// each op.  We touch the constant term so the operation must materialise.
volatile double g_sink = 0.0;

template < typename Fn >
[[gnu::noinline]] double time_op( int iters, const Te& a, const Te& b, Fn&& op )
{
    const auto t0 = clock_t_::now();
    for ( int i = 0; i < iters; ++i )
    {
        Te y = op( a, b );
        g_sink += y[0];
    }
    const auto t1 = clock_t_::now();
    const auto ns = std::chrono::duration_cast< std::chrono::nanoseconds >( t1 - t0 ).count();
    return double( ns ) / double( iters ) / 1000.0;  // μs per call
}

template < typename Fn >
[[gnu::noinline]] double time_unary( int iters, const Te& a, Fn&& op )
{
    const auto t0 = clock_t_::now();
    for ( int i = 0; i < iters; ++i )
    {
        Te y = op( a );
        g_sink += y[0];
    }
    const auto t1 = clock_t_::now();
    const auto ns = std::chrono::duration_cast< std::chrono::nanoseconds >( t1 - t0 ).count();
    return double( ns ) / double( iters ) / 1000.0;
}

template < typename F >
void row_unary( const char* name, int iters, const Te& x, F f )
{
    const double t = time_unary( iters, x, f );
    std::printf( "  %-22s  %10.4f\n", name, t );
}

template < typename F >
void row_binary( const char* name, int iters, const Te& a, const Te& b, F f )
{
    const double t = time_op( iters, a, b, f );
    std::printf( "  %-22s  %10.4f\n", name, t );
}

}  // namespace

int main( int argc, char** argv )
{
    int iters = 10000;
    if ( argc >= 2 )
        try
        {
            const int n = std::stoi( argv[1] );
            if ( n > 0 ) iters = n;
        }
        catch ( ... )
        {
        }

    // Operands — chosen so every function in the set is well-defined at the
    // expansion point (|x| < 1 for asin/atanh, x > 0 for log/sqrt, etc.).
    auto [x, y, z] = Te::variables( 0.4, 0.3, 0.2 );

    std::printf( "profile_operations_multivariate: TEn<%d, %d>  iters=%d  "
                 "(per-call time, microseconds)\n\n",
                 kN, kM, iters );
    std::printf( "  %-22s  %10s\n", "operation", "time" );
    std::printf( "  %s\n", "----------------------------------" );

    // ---- Arithmetic between two TEs --------------------------------------
    row_binary( "TE + TE", iters, x, y,
                []( const Te& a, const Te& b ) { return Te{ a + b }; } );
    row_binary( "TE - TE", iters, x, y,
                []( const Te& a, const Te& b ) { return Te{ a - b }; } );
    row_binary( "TE * TE", iters, x, y,
                []( const Te& a, const Te& b ) { return Te{ a * b }; } );
    row_binary( "TE / TE", iters, x, y,
                []( const Te& a, const Te& b ) { return Te{ a / ( b + 1.0 ) }; } );

    // ---- TE ⊕ scalar -----------------------------------------------------
    row_unary( "TE + double", iters, x,
               []( const Te& a ) { return Te{ a + 1.5 }; } );
    row_unary( "TE * double", iters, x,
               []( const Te& a ) { return Te{ a * 2.0 }; } );
    row_unary( "-TE (unary minus)", iters, x,
               []( const Te& a ) { return Te{ -a }; } );

    // ---- Trig ------------------------------------------------------------
    row_unary( "sin", iters, x,
               []( const Te& a ) { return Te{ tax::sin( a ) }; } );
    row_unary( "cos", iters, x,
               []( const Te& a ) { return Te{ tax::cos( a ) }; } );
    row_unary( "tan", iters, x,
               []( const Te& a ) { return Te{ tax::tan( a ) }; } );

    // ---- Hyperbolic ------------------------------------------------------
    row_unary( "sinh", iters, x,
               []( const Te& a ) { return Te{ tax::sinh( a ) }; } );
    row_unary( "cosh", iters, x,
               []( const Te& a ) { return Te{ tax::cosh( a ) }; } );
    row_unary( "tanh", iters, x,
               []( const Te& a ) { return Te{ tax::tanh( a ) }; } );

    // ---- Inverse trig (operands chosen so |a[0]| < 1) --------------------
    row_unary( "asin", iters, x,
               []( const Te& a ) { return Te{ tax::asin( a ) }; } );
    row_unary( "acos", iters, x,
               []( const Te& a ) { return Te{ tax::acos( a ) }; } );
    row_unary( "atan", iters, x,
               []( const Te& a ) { return Te{ tax::atan( a ) }; } );

    // ---- Inverse hyperbolic ---------------------------------------------
    row_unary( "asinh", iters, x,
               []( const Te& a ) { return Te{ tax::asinh( a ) }; } );
    {
        // acosh: requires a[0] > 1; bump operand.
        const auto x2 = std::get< 0 >( Te::variables( 2.0, 0.3, 0.2 ) );
        row_unary( "acosh (a[0]=2)", iters, x2,
                   []( const Te& a ) { return Te{ tax::acosh( a ) }; } );
    }
    row_unary( "atanh", iters, x,
               []( const Te& a ) { return Te{ tax::atanh( a ) }; } );

    // ---- exp / log -------------------------------------------------------
    row_unary( "exp", iters, x,
               []( const Te& a ) { return Te{ tax::exp( a ) }; } );
    {
        const auto x_pos = std::get< 0 >( Te::variables( 2.0, 0.3, 0.2 ) );
        row_unary( "log (a[0]=2)", iters, x_pos,
                   []( const Te& a ) { return Te{ tax::log( a ) }; } );
        row_unary( "log10 (a[0]=2)", iters, x_pos,
                   []( const Te& a ) { return Te{ tax::log10( a ) }; } );
        row_unary( "sqrt (a[0]=2)", iters, x_pos,
                   []( const Te& a ) { return Te{ tax::sqrt( a ) }; } );
        row_unary( "cbrt (a[0]=2)", iters, x_pos,
                   []( const Te& a ) { return Te{ tax::cbrt( a ) }; } );

        row_unary( "pow(., 0.5)", iters, x_pos,
                   []( const Te& a ) { return Te{ tax::pow( a, 0.5 ) }; } );
        row_unary( "pow(., 5) int", iters, x_pos,
                   []( const Te& a ) { return Te{ tax::pow( a, 5 ) }; } );
    }

    // ---- Squares / cubes / abs / erf -----------------------------------
    row_unary( "square", iters, x,
               []( const Te& a ) { return Te{ tax::square( a ) }; } );
    row_unary( "cube", iters, x,
               []( const Te& a ) { return Te{ tax::cube( a ) }; } );
    {
        // abs requires a[0] != 0.
        const auto x_pos = std::get< 0 >( Te::variables( 0.7, 0.3, 0.2 ) );
        row_unary( "abs (a[0]=0.7)", iters, x_pos,
                   []( const Te& a ) { return Te{ tax::abs( a ) }; } );
    }
    row_unary( "erf", iters, x,
               []( const Te& a ) { return Te{ tax::erf( a ) }; } );

    // ---- Two-argument math ----------------------------------------------
    row_binary(
        "atan2(y, x)", iters, x, y,
        []( const Te& yy, const Te& xx ) { return Te{ tax::atan2( yy, xx + 1.0 ) }; } );
    row_binary( "hypot(x, y)", iters, x, y,
                []( const Te& a, const Te& b ) { return Te{ tax::hypot( a, b ) }; } );

    std::printf( "\nsink=%g\n", ( double )g_sink );
    return 0;
}
