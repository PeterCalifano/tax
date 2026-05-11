#include "../testUtils.hpp"

#include <tax/tax.hpp>

using tax::DynTE;
using tax::TE;

// Helper: dynamic and static TTE of the same shape must produce equal coefficients.
template < int N, typename StaticFn, typename DynFn >
static void ExpectDynStaticAgreement( StaticFn static_op, DynFn dyn_op, double tol = 1e-12 )
{
    auto static_result = static_op();
    auto dyn_result = dyn_op();
    ASSERT_EQ( static_result.nCoefficients, dyn_result.coeffs().size() );
    for ( std::size_t i = 0; i < dyn_result.coeffs().size(); ++i )
        EXPECT_NEAR( dyn_result.coeffs()[i], static_result[i], tol ) << "i=" << i;
}

// =============================================================================
// Trig
// =============================================================================

TEST( DynTteExtMath, Tan )
{
    constexpr int N = 5;
    ExpectDynStaticAgreement< N >(
        []() {
            auto x = TE< N >::variable( 0.2 );
            return TE< N >{ tax::tan( x ) };
        },
        []() {
            auto x = DynTE<>::variable( 0.2, 0, N, 1 );
            return tax::tan( x );
        } );
}

// =============================================================================
// Hyperbolic
// =============================================================================

TEST( DynTteExtMath, Sinh )
{
    constexpr int N = 5;
    ExpectDynStaticAgreement< N >(
        []() {
            auto x = TE< N >::variable( 0.3 );
            return TE< N >{ tax::sinh( x ) };
        },
        []() {
            auto x = DynTE<>::variable( 0.3, 0, N, 1 );
            return tax::sinh( x );
        } );
}

TEST( DynTteExtMath, Cosh )
{
    constexpr int N = 5;
    ExpectDynStaticAgreement< N >(
        []() {
            auto x = TE< N >::variable( 0.3 );
            return TE< N >{ tax::cosh( x ) };
        },
        []() {
            auto x = DynTE<>::variable( 0.3, 0, N, 1 );
            return tax::cosh( x );
        } );
}

TEST( DynTteExtMath, Tanh )
{
    constexpr int N = 5;
    ExpectDynStaticAgreement< N >(
        []() {
            auto x = TE< N >::variable( 0.3 );
            return TE< N >{ tax::tanh( x ) };
        },
        []() {
            auto x = DynTE<>::variable( 0.3, 0, N, 1 );
            return tax::tanh( x );
        } );
}

// =============================================================================
// Inverse trig
// =============================================================================

TEST( DynTteExtMath, Asin )
{
    constexpr int N = 5;
    ExpectDynStaticAgreement< N >(
        []() {
            auto x = TE< N >::variable( 0.4 );
            return TE< N >{ tax::asin( x ) };
        },
        []() {
            auto x = DynTE<>::variable( 0.4, 0, N, 1 );
            return tax::asin( x );
        } );
}

TEST( DynTteExtMath, Acos )
{
    constexpr int N = 5;
    ExpectDynStaticAgreement< N >(
        []() {
            auto x = TE< N >::variable( 0.4 );
            return TE< N >{ tax::acos( x ) };
        },
        []() {
            auto x = DynTE<>::variable( 0.4, 0, N, 1 );
            return tax::acos( x );
        } );
}

TEST( DynTteExtMath, Atan )
{
    constexpr int N = 5;
    ExpectDynStaticAgreement< N >(
        []() {
            auto x = TE< N >::variable( 0.6 );
            return TE< N >{ tax::atan( x ) };
        },
        []() {
            auto x = DynTE<>::variable( 0.6, 0, N, 1 );
            return tax::atan( x );
        } );
}

// =============================================================================
// Algebra extensions
// =============================================================================

TEST( DynTteExtMath, Cube )
{
    constexpr int N = 5;
    ExpectDynStaticAgreement< N >(
        []() {
            auto x = TE< N >::variable( 0.7 );
            return TE< N >{ tax::cube( x ) };
        },
        []() {
            auto x = DynTE<>::variable( 0.7, 0, N, 1 );
            return tax::cube( x );
        } );
}

TEST( DynTteExtMath, Cbrt )
{
    constexpr int N = 5;
    ExpectDynStaticAgreement< N >(
        []() {
            auto x = TE< N >::variable( 2.0 );
            return TE< N >{ tax::cbrt( x ) };
        },
        []() {
            auto x = DynTE<>::variable( 2.0, 0, N, 1 );
            return tax::cbrt( x );
        } );
}

TEST( DynTteExtMath, Abs_PositiveLeading )
{
    auto x = DynTE<>::variable( 1.5, 0, 4, 1 );
    auto y = tax::abs( x );
    // |x| with x[0] = 1.5 > 0 is a no-op: coefficients should match.
    for ( std::size_t i = 0; i < x.coeffs().size(); ++i )
        EXPECT_EQ( y.coeffs()[i], x.coeffs()[i] );
}

TEST( DynTteExtMath, Abs_NegativeLeading )
{
    auto x = DynTE<>::variable( -2.0, 0, 4, 1 );
    auto y = tax::abs( x );
    // |x| with x[0] = -2 negates every coefficient.
    for ( std::size_t i = 0; i < x.coeffs().size(); ++i )
        EXPECT_EQ( y.coeffs()[i], -x.coeffs()[i] );
}

TEST( DynTteExtMath, Log10 )
{
    constexpr int N = 5;
    auto x = DynTE<>::variable( 5.0, 0, N, 1 );
    auto y = tax::log10( x );
    auto y_via_log = tax::log( x );
    y_via_log *= 1.0 / std::log( 10.0 );
    for ( std::size_t i = 0; i < y.coeffs().size(); ++i )
        EXPECT_NEAR( y.coeffs()[i], y_via_log.coeffs()[i], 1e-14 );
    EXPECT_NEAR( y.value(), std::log10( 5.0 ), 1e-14 );
}

// =============================================================================
// Multivariate spot-checks
// =============================================================================

TEST( DynTteExtMath, AtanMultivariate )
{
    constexpr int N = 4, M = 2;
    auto static_eval = [&]() {
        auto vars = tax::TEn< N, M >::variables( { 0.3, 0.4 } );
        return tax::TEn< N, M >{ tax::atan( std::get< 0 >( vars ) * std::get< 1 >( vars ) ) };
    }();
    std::array< double, M > x0{ 0.3, 0.4 };
    auto vars = DynTE<>::variables( std::span< const double >( x0 ), N );
    auto dyn_eval = tax::atan( vars[0] * vars[1] );

    ASSERT_EQ( static_eval.nCoefficients, dyn_eval.coeffs().size() );
    for ( std::size_t i = 0; i < dyn_eval.coeffs().size(); ++i )
        EXPECT_NEAR( dyn_eval.coeffs()[i], static_eval[i], 1e-12 ) << "i=" << i;
}

TEST( DynTteExtMath, CbrtMultivariate )
{
    constexpr int N = 4, M = 2;
    auto static_eval = [&]() {
        auto vars = tax::TEn< N, M >::variables( { 1.2, 0.5 } );
        return tax::TEn< N, M >{ tax::cbrt( std::get< 0 >( vars ) + std::get< 1 >( vars ) ) };
    }();
    std::array< double, M > x0{ 1.2, 0.5 };
    auto vars = DynTE<>::variables( std::span< const double >( x0 ), N );
    auto dyn_eval = tax::cbrt( vars[0] + vars[1] );

    ASSERT_EQ( static_eval.nCoefficients, dyn_eval.coeffs().size() );
    for ( std::size_t i = 0; i < dyn_eval.coeffs().size(); ++i )
        EXPECT_NEAR( dyn_eval.coeffs()[i], static_eval[i], 1e-12 ) << "i=" << i;
}
