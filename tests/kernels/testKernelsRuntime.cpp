#include "testUtils.hpp"

#include <array>
#include <vector>

#include <tax/kernels.hpp>
#include <tax/utils/combinatorics.hpp>

using namespace tax::detail;

namespace
{

template < int N, int M, typename Fill >
static std::array< double, numMonomials( N, M ) > makeOperand( Fill fill )
{
    std::array< double, numMonomials( N, M ) > a{};
    for ( std::size_t i = 0; i < a.size(); ++i ) a[i] = fill( i );
    return a;
}

template < int N, int M >
static void ExpectRuntimeMatchesStatic_BinKernel( auto static_kernel, auto runtime_kernel )
{
    auto f = makeOperand< N, M >( []( std::size_t i ) { return 0.1 + 0.07 * double( i ); } );
    auto g = makeOperand< N, M >( []( std::size_t i ) { return -0.05 + 0.03 * double( i ); } );

    std::array< double, numMonomials( N, M ) > out_static{};
    static_kernel( out_static, f, g );

    std::vector< double > out_rt( numMonomials( N, M ), 0.0 );
    runtime_kernel( out_rt.data(), f.data(), g.data(), std::size_t( N ), std::size_t( M ) );

    for ( std::size_t i = 0; i < out_static.size(); ++i )
        EXPECT_NEAR( out_rt[i], out_static[i], 1e-12 ) << "i=" << i << " N=" << N << " M=" << M;
}

template < int N, int M >
static void ExpectRuntimeMatchesStatic_UnKernel( auto static_kernel, auto runtime_kernel )
{
    // Use a strictly-positive constant term so sqrt/log/reciprocal are well-defined.
    auto a = makeOperand< N, M >( []( std::size_t i ) { return ( i == 0 ) ? 1.5 : 0.07 * double( i ); } );

    std::array< double, numMonomials( N, M ) > out_static{};
    static_kernel( out_static, a );

    std::vector< double > out_rt( numMonomials( N, M ), 0.0 );
    runtime_kernel( out_rt.data(), a.data(), std::size_t( N ), std::size_t( M ) );

    for ( std::size_t i = 0; i < out_static.size(); ++i )
        EXPECT_NEAR( out_rt[i], out_static[i], 1e-12 ) << "i=" << i << " N=" << N << " M=" << M;
}

}  // namespace

// =============================================================================
// cauchy
// =============================================================================

TEST( KernelsRuntime, CauchyProduct_M1_N5 )
{
    ExpectRuntimeMatchesStatic_BinKernel< 5, 1 >(
        []( auto& o, const auto& f, const auto& g ) { cauchyProduct< double, 5, 1 >( o, f, g ); },
        []( double* o, const double* f, const double* g, std::size_t N, std::size_t M ) {
            cauchyProductRT( o, f, g, N, M );
        } );
}

TEST( KernelsRuntime, CauchyProduct_M3_N4 )
{
    ExpectRuntimeMatchesStatic_BinKernel< 4, 3 >(
        []( auto& o, const auto& f, const auto& g ) { cauchyProduct< double, 4, 3 >( o, f, g ); },
        []( double* o, const double* f, const double* g, std::size_t N, std::size_t M ) {
            cauchyProductRT( o, f, g, N, M );
        } );
}

TEST( KernelsRuntime, CauchySelfProduct_M2_N5 )
{
    auto f = makeOperand< 5, 2 >( []( std::size_t i ) { return 0.1 + 0.07 * double( i ); } );

    std::array< double, numMonomials( 5, 2 ) > out_static{};
    cauchySelfProduct< double, 5, 2 >( out_static, f );

    std::vector< double > out_rt( numMonomials( 5, 2 ), 0.0 );
    cauchySelfProductRT( out_rt.data(), f.data(), std::size_t( 5 ), std::size_t( 2 ) );

    for ( std::size_t i = 0; i < out_static.size(); ++i )
        EXPECT_NEAR( out_rt[i], out_static[i], 1e-12 ) << "i=" << i;
}

// =============================================================================
// algebra
// =============================================================================

TEST( KernelsRuntime, Reciprocal_M1_N6 )
{
    ExpectRuntimeMatchesStatic_UnKernel< 6, 1 >(
        []( auto& o, const auto& a ) { seriesReciprocal< double, 6, 1 >( o, a ); },
        []( double* o, const double* a, std::size_t N, std::size_t M ) {
            seriesReciprocalRT( o, a, N, M );
        } );
}

TEST( KernelsRuntime, Reciprocal_M3_N4 )
{
    ExpectRuntimeMatchesStatic_UnKernel< 4, 3 >(
        []( auto& o, const auto& a ) { seriesReciprocal< double, 4, 3 >( o, a ); },
        []( double* o, const double* a, std::size_t N, std::size_t M ) {
            seriesReciprocalRT( o, a, N, M );
        } );
}

TEST( KernelsRuntime, Sqrt_M1_N6 )
{
    ExpectRuntimeMatchesStatic_UnKernel< 6, 1 >(
        []( auto& o, const auto& a ) { seriesSqrt< double, 6, 1 >( o, a ); },
        []( double* o, const double* a, std::size_t N, std::size_t M ) {
            seriesSqrtRT( o, a, N, M );
        } );
}

TEST( KernelsRuntime, Sqrt_M2_N5 )
{
    ExpectRuntimeMatchesStatic_UnKernel< 5, 2 >(
        []( auto& o, const auto& a ) { seriesSqrt< double, 5, 2 >( o, a ); },
        []( double* o, const double* a, std::size_t N, std::size_t M ) {
            seriesSqrtRT( o, a, N, M );
        } );
}

// =============================================================================
// transcendental
// =============================================================================

TEST( KernelsRuntime, Exp_M1_N6 )
{
    ExpectRuntimeMatchesStatic_UnKernel< 6, 1 >(
        []( auto& o, const auto& a ) { seriesExp< double, 6, 1 >( o, a ); },
        []( double* o, const double* a, std::size_t N, std::size_t M ) {
            seriesExpRT( o, a, N, M );
        } );
}

TEST( KernelsRuntime, Exp_M3_N4 )
{
    ExpectRuntimeMatchesStatic_UnKernel< 4, 3 >(
        []( auto& o, const auto& a ) { seriesExp< double, 4, 3 >( o, a ); },
        []( double* o, const double* a, std::size_t N, std::size_t M ) {
            seriesExpRT( o, a, N, M );
        } );
}

TEST( KernelsRuntime, Log_M1_N6 )
{
    ExpectRuntimeMatchesStatic_UnKernel< 6, 1 >(
        []( auto& o, const auto& a ) { seriesLog< double, 6, 1 >( o, a ); },
        []( double* o, const double* a, std::size_t N, std::size_t M ) {
            seriesLogRT( o, a, N, M );
        } );
}

TEST( KernelsRuntime, Log_M2_N5 )
{
    ExpectRuntimeMatchesStatic_UnKernel< 5, 2 >(
        []( auto& o, const auto& a ) { seriesLog< double, 5, 2 >( o, a ); },
        []( double* o, const double* a, std::size_t N, std::size_t M ) {
            seriesLogRT( o, a, N, M );
        } );
}

TEST( KernelsRuntime, Pow_M1_N5 )
{
    auto a = makeOperand< 5, 1 >( []( std::size_t i ) { return ( i == 0 ) ? 2.0 : 0.05 * double( i ); } );

    std::array< double, numMonomials( 5, 1 ) > out_static{};
    seriesPow< double, 5, 1 >( out_static, a, 0.5 );

    std::vector< double > out_rt( numMonomials( 5, 1 ), 0.0 );
    seriesPowRT( out_rt.data(), a.data(), 0.5, std::size_t( 5 ), std::size_t( 1 ) );

    for ( std::size_t i = 0; i < out_static.size(); ++i )
        EXPECT_NEAR( out_rt[i], out_static[i], 1e-12 ) << "i=" << i;
}

// =============================================================================
// trigonometric
// =============================================================================

TEST( KernelsRuntime, SinCos_M1_N6 )
{
    auto a = makeOperand< 6, 1 >( []( std::size_t i ) { return 0.05 * double( i + 1 ); } );

    std::array< double, numMonomials( 6, 1 ) > s_static{}, c_static{};
    seriesSinCos< double, 6, 1 >( s_static, c_static, a );

    std::vector< double > s_rt( numMonomials( 6, 1 ), 0.0 );
    std::vector< double > c_rt( numMonomials( 6, 1 ), 0.0 );
    seriesSinCosRT( s_rt.data(), c_rt.data(), a.data(), std::size_t( 6 ), std::size_t( 1 ) );

    for ( std::size_t i = 0; i < s_static.size(); ++i )
    {
        EXPECT_NEAR( s_rt[i], s_static[i], 1e-12 );
        EXPECT_NEAR( c_rt[i], c_static[i], 1e-12 );
    }
}

TEST( KernelsRuntime, SinCos_M2_N5 )
{
    auto a = makeOperand< 5, 2 >( []( std::size_t i ) { return 0.04 * double( i + 1 ); } );

    std::array< double, numMonomials( 5, 2 ) > s_static{}, c_static{};
    seriesSinCos< double, 5, 2 >( s_static, c_static, a );

    std::vector< double > s_rt( numMonomials( 5, 2 ), 0.0 );
    std::vector< double > c_rt( numMonomials( 5, 2 ), 0.0 );
    seriesSinCosRT( s_rt.data(), c_rt.data(), a.data(), std::size_t( 5 ), std::size_t( 2 ) );

    for ( std::size_t i = 0; i < s_static.size(); ++i )
    {
        EXPECT_NEAR( s_rt[i], s_static[i], 1e-12 );
        EXPECT_NEAR( c_rt[i], c_static[i], 1e-12 );
    }
}
