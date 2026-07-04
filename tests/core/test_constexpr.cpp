#include <gtest/gtest.h>

#include <cmath>

#include "../testUtils.hpp"

// ---------------------------------------------------------------------------
// Compile-time evaluation of the full math surface.
//
// Every dense operation — including the transcendental ones — is constexpr:
// in constant evaluation the constant term goes through tax::detail::cmath
// instead of libm. The static_asserts below run entire expansion pipelines in
// the compiler; the runtime checks then verify that the compile-time
// coefficients agree with the runtime (libm-seeded) ones to ~1 ulp.
// ---------------------------------------------------------------------------

namespace
{

constexpr bool nearAbs( double a, double b, double tol )
{
    const double d = a > b ? a - b : b - a;
    return d <= tol;
}

/// Coefficient-wise |a - b| <= tol * (1 + |a|).
template < typename E >
constexpr bool coeffsNear( const E& a, const E& b, double tol )
{
    for ( std::size_t k = 0; k < E::nCoefficients; ++k )
    {
        const double m = a[k] < 0 ? -a[k] : a[k];
        if ( !nearAbs( a[k], b[k], tol * ( 1.0 + m ) ) ) return false;
    }
    return true;
}

}  // namespace

// ------------------------------- univariate --------------------------------

constexpr auto kX = tax::TE< 8 >::variable( 0.5 );
constexpr auto kF = tax::exp( tax::sin( kX ) ) / tax::sqrt( kX + 1.0 );

static_assert( kF.value() > 0.0, "full transcendental pipeline in constant evaluation" );

// Identities evaluated entirely at compile time.
static_assert( coeffsNear( tax::square( tax::sqrt( kX + 1.0 ) ), kX + 1.0, 1e-14 ) );
static_assert( coeffsNear( tax::exp( tax::log( kX + 2.0 ) ), kX + 2.0, 1e-14 ) );
static_assert( coeffsNear( tax::sin( kX ) * tax::sin( kX ) + tax::cos( kX ) * tax::cos( kX ),
                           tax::TE< 8 >{ 1.0 }, 1e-14 ) );
static_assert( coeffsNear( tax::tan( kX ), tax::sin( kX ) / tax::cos( kX ), 1e-13 ) );
static_assert( coeffsNear( tax::tanh( kX ), tax::sinh( kX ) / tax::cosh( kX ), 1e-13 ) );
static_assert( coeffsNear( tax::asin( tax::sin( kX ) ), kX, 1e-13 ) );
static_assert( coeffsNear( tax::atan( tax::tan( kX ) ), kX, 1e-13 ) );
static_assert( coeffsNear( tax::atanh( tax::tanh( kX ) ), kX, 1e-13 ) );
static_assert( coeffsNear( tax::asinh( tax::sinh( kX ) ), kX, 1e-13 ) );
static_assert( coeffsNear( tax::cbrt( tax::cube( kX + 1.0 ) ), kX + 1.0, 1e-13 ) );
static_assert( coeffsNear( tax::pow( kX + 1.0, 0.5 ), tax::sqrt( kX + 1.0 ), 1e-13 ) );
static_assert( coeffsNear( tax::pow( kX + 1.0, kX ), tax::exp( kX* tax::log( kX + 1.0 ) ),
                           1e-13 ) );
static_assert( coeffsNear( tax::atan2( tax::sin( kX ), tax::cos( kX ) ), kX, 1e-13 ) );

// acos' constant term: asin + acos == pi/2.
static_assert( nearAbs( tax::asin( kX ).value() + tax::acos( kX ).value(), std::numbers::pi / 2,
                        1e-15 ) );

// acosh(cosh(x)) == |x| branch for x > 0.
constexpr auto kXL = tax::TE< 6 >::variable( 1.25 );
static_assert( coeffsNear( tax::acosh( tax::cosh( kXL ) ), kXL, 1e-12 ) );

// deriv / integ round trip at compile time (loses only the constant term).
static_assert( coeffsNear( kF.deriv< 0 >().integ< 0 >() + kF.value(), kF, 1e-13 ) );

// ------------------------------ multivariate -------------------------------

constexpr auto kG = [] {
    using E = tax::TE< 5, 2 >;
    typename E::Input p{ 0.4, 1.2 };
    auto x = E::variable< 0 >( p );
    auto y = E::variable< 1 >( p );
    return tax::atan2( x, y ) * tax::erf( x ) + tax::log( tax::square( x ) + tax::square( y ) );
}();
static_assert( kG.value() != 0.0, "multivariate pipeline in constant evaluation" );

// ----------------------------- named expansions ----------------------------

constexpr auto kNamed = [] {
    auto x = tax::variable< "x", 4 >( 0.3 );
    auto p = tax::variable< "p", 4 >( 2.0 );
    return tax::sin( x ) * tax::sqrt( p ) + tax::pow( p, 3 );
}();
static_assert( kNamed.value() != 0.0, "named expansions in constant evaluation" );

// ---------------------------------------------------------------------------
// Compile-time coefficients agree with runtime (libm-seeded) coefficients.
// ---------------------------------------------------------------------------

namespace
{

template < typename Fn >
void expectCompileTimeMatchesRuntime( Fn fn, double tol = 5e-15 )
{
    constexpr auto ce = Fn{}();  // constant-evaluated: cmath.hpp path
    const auto rt = fn();        // runtime: libm path
    for ( std::size_t k = 0; k < decltype( rt )::nCoefficients; ++k )
        EXPECT_NEAR( ce[k], rt[k], tol * ( 1.0 + std::abs( rt[k] ) ) ) << "k = " << k;
}

}  // namespace

TEST( ConstexprMath, MatchesRuntimeUnivariate )
{
    expectCompileTimeMatchesRuntime(
        [] { return tax::exp( tax::sin( tax::TE< 8 >::variable( 0.5 ) ) ); } );
    expectCompileTimeMatchesRuntime( [] { return tax::log( tax::TE< 8 >::variable( 3.75 ) ); } );
    expectCompileTimeMatchesRuntime( [] { return tax::sqrt( tax::TE< 8 >::variable( 2.0 ) ); } );
    expectCompileTimeMatchesRuntime( [] { return tax::cbrt( tax::TE< 8 >::variable( -8.0 ) ); } );
    expectCompileTimeMatchesRuntime( [] { return tax::tan( tax::TE< 8 >::variable( 1.2 ) ); } );
    expectCompileTimeMatchesRuntime( [] { return tax::asin( tax::TE< 8 >::variable( 0.66 ) ); } );
    expectCompileTimeMatchesRuntime( [] { return tax::acos( tax::TE< 8 >::variable( -0.66 ) ); } );
    expectCompileTimeMatchesRuntime( [] { return tax::atan( tax::TE< 8 >::variable( 42.0 ) ); } );
    expectCompileTimeMatchesRuntime( [] { return tax::erf( tax::TE< 8 >::variable( 1.5 ) ); } );
    expectCompileTimeMatchesRuntime( [] { return tax::sinh( tax::TE< 8 >::variable( 0.01 ) ); } );
    expectCompileTimeMatchesRuntime( [] { return tax::acosh( tax::TE< 8 >::variable( 2.5 ) ); } );
    expectCompileTimeMatchesRuntime( [] { return tax::asinh( tax::TE< 8 >::variable( 0.001 ) ); } );
    expectCompileTimeMatchesRuntime( [] { return tax::atanh( tax::TE< 8 >::variable( 0.98 ) ); } );
    expectCompileTimeMatchesRuntime(
        [] { return tax::pow( tax::TE< 8 >::variable( 1.7 ), -1.5 ); } );
}

TEST( ConstexprMath, MatchesRuntimeAtAwkwardPoints )
{
    // Large exp argument (exercises the 2^k scaling), value near overflow scale.
    expectCompileTimeMatchesRuntime( [] { return tax::exp( tax::TE< 4 >::variable( 250.0 ) ); },
                                     1e-13 );
    // Trig far from zero (argument reduction).
    expectCompileTimeMatchesRuntime( [] { return tax::sin( tax::TE< 6 >::variable( 1000.0 ) ); },
                                     1e-12 );
    expectCompileTimeMatchesRuntime( [] { return tax::cos( tax::TE< 6 >::variable( -273.15 ) ); },
                                     1e-12 );
    // Tiny sqrt argument (exercises the exponent normalisation).
    expectCompileTimeMatchesRuntime( [] { return tax::sqrt( tax::TE< 3 >::variable( 1.0e-10 ) ); },
                                     1e-13 );
}
