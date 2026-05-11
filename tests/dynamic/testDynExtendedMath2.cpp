// Coverage for the second wave of dynamic math: asinh / acosh / atanh / erf /
// atan2 / hypot / pow<int>. Each function is checked on the fully-dynamic
// `DynTE` and (where applicable) on `DynOrderTE<M>`, with the static-shape
// path as the reference.

#include "../testUtils.hpp"

#include <tax/tax.hpp>

using tax::DynOrderTE;
using tax::DynTE;
using tax::TE;
using tax::TEn;

template < int N, typename StaticFn, typename DynFn >
static void ExpectDynAgreesWithStatic( StaticFn static_op, DynFn dyn_op, double tol = 1e-12 )
{
    auto s = static_op();
    auto d = dyn_op();
    ASSERT_EQ( s.nCoefficients, d.coeffs().size() );
    for ( std::size_t i = 0; i < d.coeffs().size(); ++i )
        EXPECT_NEAR( d.coeffs()[i], s[i], tol ) << "i=" << i;
}

// =============================================================================
// asinh / acosh / atanh
// =============================================================================

TEST( DynMath2, Asinh )
{
    constexpr int N = 6;
    ExpectDynAgreesWithStatic< N >(
        []() {
            auto x = TE< N >::variable( 0.4 );
            return TE< N >{ tax::asinh( x ) };
        },
        []() {
            auto x = DynTE<>::variable( 0.4, 0, N, 1 );
            return tax::asinh( x );
        } );
}

TEST( DynMath2, Acosh )
{
    constexpr int N = 5;
    ExpectDynAgreesWithStatic< N >(
        []() {
            auto x = TE< N >::variable( 2.0 );  // a[0] > 1
            return TE< N >{ tax::acosh( x ) };
        },
        []() {
            auto x = DynTE<>::variable( 2.0, 0, N, 1 );
            return tax::acosh( x );
        } );
}

TEST( DynMath2, Atanh )
{
    constexpr int N = 5;
    ExpectDynAgreesWithStatic< N >(
        []() {
            auto x = TE< N >::variable( 0.3 );  // |a[0]| < 1
            return TE< N >{ tax::atanh( x ) };
        },
        []() {
            auto x = DynTE<>::variable( 0.3, 0, N, 1 );
            return tax::atanh( x );
        } );
}

// =============================================================================
// erf
// =============================================================================

TEST( DynMath2, Erf )
{
    constexpr int N = 6;
    ExpectDynAgreesWithStatic< N >(
        []() {
            auto x = TE< N >::variable( 0.3 );
            return TE< N >{ tax::erf( x ) };
        },
        []() {
            auto x = DynTE<>::variable( 0.3, 0, N, 1 );
            return tax::erf( x );
        } );
}

TEST( DynMath2, ErfMultivariate )
{
    constexpr int N = 4, M = 2;
    ExpectDynAgreesWithStatic< N >(
        []() {
            auto vars = TEn< N, M >::variables( { 0.2, 0.3 } );
            return TEn< N, M >{ tax::erf( std::get< 0 >( vars ) + std::get< 1 >( vars ) ) };
        },
        []() {
            std::array< double, M > x0{ 0.2, 0.3 };
            auto vars = DynTE<>::variables( std::span< const double >( x0 ), N );
            return tax::erf( vars[0] + vars[1] );
        } );
}

// =============================================================================
// atan2 / hypot
// =============================================================================

TEST( DynMath2, Atan2 )
{
    constexpr int N = 5;
    ExpectDynAgreesWithStatic< N >(
        []() {
            auto x = TE< N >::variable( 1.0 );
            // For univariate atan2 use the same variable on both arms — fine for
            // verification (just compares Taylor expansion identities).
            return TE< N >{ tax::atan2( x, x + 1.0 ) };
        },
        []() {
            auto x = DynTE<>::variable( 1.0, 0, N, 1 );
            return tax::atan2( x, x + 1.0 );
        } );
}

TEST( DynMath2, Hypot )
{
    constexpr int N = 5, M = 2;
    ExpectDynAgreesWithStatic< N >(
        []() {
            auto vars = TEn< N, M >::variables( { 3.0, 4.0 } );
            return TEn< N, M >{ tax::hypot( std::get< 0 >( vars ), std::get< 1 >( vars ) ) };
        },
        []() {
            std::array< double, M > x0{ 3.0, 4.0 };
            auto vars = DynTE<>::variables( std::span< const double >( x0 ), N );
            return tax::hypot( vars[0], vars[1] );
        } );
}

// =============================================================================
// pow<int>
// =============================================================================

TEST( DynMath2, IntPow_Positive )
{
    constexpr int N = 5;
    ExpectDynAgreesWithStatic< N >(
        []() {
            auto x = TE< N >::variable( 2.0 );
            return TE< N >{ tax::pow( x, 5 ) };
        },
        []() {
            auto x = DynTE<>::variable( 2.0, 0, N, 1 );
            return tax::pow( x, 5 );
        } );
}

TEST( DynMath2, IntPow_Negative )
{
    constexpr int N = 4;
    ExpectDynAgreesWithStatic< N >(
        []() {
            auto x = TE< N >::variable( 2.0 );
            return TE< N >{ tax::pow( x, -3 ) };
        },
        []() {
            auto x = DynTE<>::variable( 2.0, 0, N, 1 );
            return tax::pow( x, -3 );
        } );
}

TEST( DynMath2, IntPow_TemplateAndRuntimeAgree )
{
    constexpr int N = 5;
    auto x = DynTE<>::variable( 2.0, 0, N, 1 );
    auto a = tax::pow( x, 5 );        // runtime int
    auto b = tax::pow< 5 >( x );      // template int
    ASSERT_EQ( a.coeffs().size(), b.coeffs().size() );
    for ( std::size_t i = 0; i < a.coeffs().size(); ++i )
        EXPECT_NEAR( a.coeffs()[i], b.coeffs()[i], 1e-14 );
}

// =============================================================================
// DynOrderTE coverage for the same new functions
// =============================================================================

TEST( DynOrderMath2, Asinh )
{
    constexpr int N = 5, M = 2;
    ExpectDynAgreesWithStatic< N >(
        []() {
            auto vars = TEn< N, M >::variables( { 0.2, 0.3 } );
            return TEn< N, M >{ tax::asinh( std::get< 0 >( vars ) * std::get< 1 >( vars ) ) };
        },
        []() {
            auto vars = DynOrderTE< M >::variables( { 0.2, 0.3 }, N );
            return tax::asinh( vars[0] * vars[1] );
        } );
}

TEST( DynOrderMath2, Atan2 )
{
    constexpr int N = 5, M = 2;
    ExpectDynAgreesWithStatic< N >(
        []() {
            auto vars = TEn< N, M >::variables( { 0.5, 1.0 } );
            return TEn< N, M >{ tax::atan2( std::get< 0 >( vars ), std::get< 1 >( vars ) ) };
        },
        []() {
            auto vars = DynOrderTE< M >::variables( { 0.5, 1.0 }, N );
            return tax::atan2( vars[0], vars[1] );
        } );
}

TEST( DynOrderMath2, Hypot )
{
    constexpr int N = 5, M = 2;
    ExpectDynAgreesWithStatic< N >(
        []() {
            auto vars = TEn< N, M >::variables( { 3.0, 4.0 } );
            return TEn< N, M >{ tax::hypot( std::get< 0 >( vars ), std::get< 1 >( vars ) ) };
        },
        []() {
            auto vars = DynOrderTE< M >::variables( { 3.0, 4.0 }, N );
            return tax::hypot( vars[0], vars[1] );
        } );
}

TEST( DynOrderMath2, IntPow )
{
    constexpr int N = 5, M = 1;
    ExpectDynAgreesWithStatic< N >(
        []() {
            auto x = TE< N >::variable( 1.5 );
            return TE< N >{ tax::pow( x, 7 ) };
        },
        []() {
            auto x = DynOrderTE< 1 >::variable( 1.5, N );
            return tax::pow( x, 7 );
        } );
}
