#include "../testUtils.hpp"

#include <array>

#include <tax/tax.hpp>

using tax::DynTE;
using tax::TE;
using tax::TEn;

// Helper: dynamic and static TTE of the same shape must produce equal coefficients.
template < int N, int M, typename StaticFn, typename DynFn >
static void ExpectDynStaticAgreement( StaticFn static_op, DynFn dyn_op )
{
    auto static_result = static_op();
    auto dyn_result = dyn_op();
    ASSERT_EQ( static_result.nCoefficients, dyn_result.coeffs().size() );
    for ( std::size_t i = 0; i < dyn_result.coeffs().size(); ++i )
        EXPECT_NEAR( dyn_result.coeffs()[i], static_result[i], 1e-12 ) << "i=" << i;
}

// =============================================================================
// Univariate arithmetic
// =============================================================================

TEST( DynTteArith, UniAdd )
{
    constexpr int N = 6;
    ExpectDynStaticAgreement< N, 1 >(
        []() {
            auto x = TE< N >::variable( 0.5 );
            auto y = TE< N >::variable( 0.3 );
            return TE< N >{ x + y };
        },
        []() {
            auto x = DynTE<>::variable( 0.5, 0, N, 1 );
            auto y = DynTE<>::variable( 0.3, 0, N, 1 );
            return x + y;
        } );
}

TEST( DynTteArith, UniMul )
{
    constexpr int N = 6;
    ExpectDynStaticAgreement< N, 1 >(
        []() {
            auto x = TE< N >::variable( 0.5 );
            auto y = TE< N >::variable( 0.3 );
            return TE< N >{ x * y };
        },
        []() {
            auto x = DynTE<>::variable( 0.5, 0, N, 1 );
            auto y = DynTE<>::variable( 0.3, 0, N, 1 );
            return x * y;
        } );
}

TEST( DynTteArith, UniDiv )
{
    constexpr int N = 5;
    ExpectDynStaticAgreement< N, 1 >(
        []() {
            auto x = TE< N >::variable( 1.0 );
            auto y = TE< N >::variable( 2.0 );
            return TE< N >{ x / y };
        },
        []() {
            auto x = DynTE<>::variable( 1.0, 0, N, 1 );
            auto y = DynTE<>::variable( 2.0, 0, N, 1 );
            return x / y;
        } );
}

TEST( DynTteArith, UniScalarOps )
{
    constexpr int N = 4;
    ExpectDynStaticAgreement< N, 1 >(
        []() {
            auto x = TE< N >::variable( 0.7 );
            return TE< N >{ 2.0 * x + 1.5 - x / 4.0 };
        },
        []() {
            auto x = DynTE<>::variable( 0.7, 0, N, 1 );
            return 2.0 * x + 1.5 - x / 4.0;
        } );
}

// =============================================================================
// Multivariate arithmetic
// =============================================================================

TEST( DynTteArith, MultivariateAdd )
{
    constexpr int N = 4, M = 3;
    ExpectDynStaticAgreement< N, M >(
        []() {
            auto vars = TEn< N, M >::variables( { 0.1, 0.2, 0.3 } );
            return TEn< N, M >{ std::get< 0 >( vars ) + std::get< 1 >( vars ) +
                                std::get< 2 >( vars ) };
        },
        []() {
            std::array< double, M > x0{ 0.1, 0.2, 0.3 };
            auto vars = DynTE<>::variables( std::span< const double >( x0 ), N );
            return vars[0] + vars[1] + vars[2];
        } );
}

TEST( DynTteArith, MultivariateMul )
{
    constexpr int N = 4, M = 2;
    ExpectDynStaticAgreement< N, M >(
        []() {
            auto vars = TEn< N, M >::variables( { 1.0, 2.0 } );
            return TEn< N, M >{ std::get< 0 >( vars ) * std::get< 1 >( vars ) };
        },
        []() {
            std::array< double, M > x0{ 1.0, 2.0 };
            auto vars = DynTE<>::variables( std::span< const double >( x0 ), N );
            return vars[0] * vars[1];
        } );
}

// =============================================================================
// Math functions
// =============================================================================

TEST( DynTteArith, Sin )
{
    constexpr int N = 6;
    ExpectDynStaticAgreement< N, 1 >(
        []() {
            auto x = TE< N >::variable( 0.5 );
            return TE< N >{ tax::sin( x ) };
        },
        []() {
            auto x = DynTE<>::variable( 0.5, 0, N, 1 );
            return tax::sin( x );
        } );
}

TEST( DynTteArith, Exp )
{
    constexpr int N = 5;
    ExpectDynStaticAgreement< N, 1 >(
        []() {
            auto x = TE< N >::variable( 0.2 );
            return TE< N >{ tax::exp( x ) };
        },
        []() {
            auto x = DynTE<>::variable( 0.2, 0, N, 1 );
            return tax::exp( x );
        } );
}

TEST( DynTteArith, Log )
{
    constexpr int N = 5;
    ExpectDynStaticAgreement< N, 1 >(
        []() {
            auto x = TE< N >::variable( 2.0 );
            return TE< N >{ tax::log( x ) };
        },
        []() {
            auto x = DynTE<>::variable( 2.0, 0, N, 1 );
            return tax::log( x );
        } );
}

TEST( DynTteArith, Sqrt )
{
    constexpr int N = 5;
    ExpectDynStaticAgreement< N, 1 >(
        []() {
            auto x = TE< N >::variable( 3.0 );
            return TE< N >{ tax::sqrt( x ) };
        },
        []() {
            auto x = DynTE<>::variable( 3.0, 0, N, 1 );
            return tax::sqrt( x );
        } );
}

TEST( DynTteArith, Pow )
{
    constexpr int N = 5;
    ExpectDynStaticAgreement< N, 1 >(
        []() {
            auto x = TE< N >::variable( 2.0 );
            return TE< N >{ tax::pow( x, 0.5 ) };
        },
        []() {
            auto x = DynTE<>::variable( 2.0, 0, N, 1 );
            return tax::pow( x, 0.5 );
        } );
}

TEST( DynTteArith, Composite_Multivariate )
{
    constexpr int N = 4, M = 2;
    ExpectDynStaticAgreement< N, M >(
        []() {
            auto vars = TEn< N, M >::variables( { 0.5, 0.3 } );
            auto x = std::get< 0 >( vars );
            auto y = std::get< 1 >( vars );
            return TEn< N, M >{ tax::sin( x * y ) + tax::exp( x + y ) };
        },
        []() {
            std::array< double, M > x0{ 0.5, 0.3 };
            auto vars = DynTE<>::variables( std::span< const double >( x0 ), N );
            return tax::sin( vars[0] * vars[1] ) + tax::exp( vars[0] + vars[1] );
        } );
}
