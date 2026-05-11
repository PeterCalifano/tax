// SPDX-License-Identifier: BSD-3-Clause
//
// Profiling driver — multivariate arithmetic + math surface against a
// static `TEn<5, 3>`.  Intentionally minimal: no timing, no row helpers,
// no Google Benchmark — `perf` / `callgrind` / `vtune` does the measuring.
//
//   perf record -g --call-graph dwarf -- \
//       ./build/benchmarks/profile_operations_multivariate 20000
//   perf report
//
// argv[1]   iterations (default 10000)

#include <cstdio>
#include <cstdlib>
#include <tuple>

#include <tax/tax.hpp>

int main( int argc, char** argv )
{
    int iters = 10000;
    if ( argc >= 2 ) iters = std::atoi( argv[1] );
    if ( iters <= 0 ) iters = 10000;

    using Te = tax::TEn< 5, 3 >;

    auto [x, y, z] = Te::variables( 0.4, 0.3, 0.2 );
    const auto x_pos = std::get< 0 >( Te::variables( 2.0, 0.3, 0.2 ) );
    const auto x_abs = std::get< 0 >( Te::variables( 0.7, 0.3, 0.2 ) );

    volatile double sink = 0.0;

    for ( int i = 0; i < iters; ++i )
    {
        sink += Te{ x + y }.value();
        sink += Te{ x - y }.value();
        sink += Te{ x * y }.value();
        sink += Te{ x / ( y + 1.0 ) }.value();

        sink += Te{ x + 1.5 }.value();
        sink += Te{ x * 2.0 }.value();
        sink += Te{ -x }.value();

        sink += Te{ tax::sin( x ) }.value();
        sink += Te{ tax::cos( x ) }.value();
        sink += Te{ tax::tan( x ) }.value();
        sink += Te{ tax::sinh( x ) }.value();
        sink += Te{ tax::cosh( x ) }.value();
        sink += Te{ tax::tanh( x ) }.value();
        sink += Te{ tax::asin( x ) }.value();
        sink += Te{ tax::acos( x ) }.value();
        sink += Te{ tax::atan( x ) }.value();
        sink += Te{ tax::asinh( x ) }.value();
        sink += Te{ tax::acosh( x_pos ) }.value();
        sink += Te{ tax::atanh( x ) }.value();

        sink += Te{ tax::exp( x ) }.value();
        sink += Te{ tax::log( x_pos ) }.value();
        sink += Te{ tax::log10( x_pos ) }.value();
        sink += Te{ tax::sqrt( x_pos ) }.value();
        sink += Te{ tax::cbrt( x_pos ) }.value();
        sink += Te{ tax::pow( x_pos, 0.5 ) }.value();
        sink += Te{ tax::pow( x_pos, 5 ) }.value();

        sink += Te{ tax::square( x ) }.value();
        sink += Te{ tax::cube( x ) }.value();
        sink += Te{ tax::abs( x_abs ) }.value();
        sink += Te{ tax::erf( x ) }.value();

        sink += Te{ tax::atan2( x, y + 1.0 ) }.value();
        sink += Te{ tax::hypot( x, y ) }.value();
    }

    std::printf( "profile_operations_multivariate: iters=%d sink=%g\n", iters, ( double )sink );
    return 0;
}
