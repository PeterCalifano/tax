#include <gtest/gtest.h>

#include "../testUtils.hpp"

// Compile-time evaluation of the pure-polynomial surface. The transcendental
// functions seed their recurrence with a libm call (std::exp/sin/...), so they
// are runtime-only. The polynomial operations — arithmetic,
// square/cube/reciprocal, integer powers, division, and the
// differential/evaluation accessors — are constexpr and evaluate entirely in the
// compiler; the static_asserts below exercise that surface.

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

constexpr auto kX = tax::TE< 8 >::variable( 0.5 );

// A whole polynomial pipeline in constant evaluation.
constexpr auto kF = ( tax::square( kX ) + 2.0 * kX + 1.0 ) / ( kX + 3.0 );
static_assert( kF.value() != 0.0, "polynomial pipeline in constant evaluation" );

// Identities evaluated entirely at compile time.
static_assert( coeffsNear( tax::square( kX ), kX* kX, 1e-14 ) );
static_assert( coeffsNear( tax::cube( kX ), kX* kX* kX, 1e-14 ) );
static_assert( coeffsNear( tax::reciprocal( kX + 1.0 ) * ( kX + 1.0 ), tax::TE< 8 >{ 1.0 },
                           1e-14 ) );
static_assert( coeffsNear( tax::pow( kX + 1.0, 5 ), tax::cube( kX + 1.0 ) * tax::square( kX + 1.0 ),
                           1e-13 ) );
static_assert( coeffsNear( tax::pow( kX + 1.0, -2 ), tax::reciprocal( tax::square( kX + 1.0 ) ),
                           1e-13 ) );

// deriv / integ round trip at compile time (loses only the constant term).
static_assert( coeffsNear( kF.deriv< 0 >().integ< 0 >() + kF.value(), kF, 1e-13 ) );

// eval at a displacement, at compile time.
static_assert( nearAbs( kF.eval( { 0.0 } ), kF.value(), 1e-15 ) );

// truncate at compile time.
static_assert( tax::square( kX ).truncate< 2 >().order_v == 2 );

constexpr auto kG = [] {
    using E = tax::TE< 5, 2 >;
    typename E::Input p{ 0.4, 1.2 };
    auto x = E::variable< 0 >( p );
    auto y = E::variable< 1 >( p );
    return ( tax::square( x ) + tax::square( y ) ) * ( x - y ) + tax::pow( x, 3 );
}();
static_assert( kG.value() != 0.0, "multivariate polynomial pipeline in constant evaluation" );

// Named arithmetic and integer powers are constexpr (the transcendental named
// wrappers are runtime-only, like the dense ones).
constexpr auto kNamed = [] {
    auto x = tax::variable< "x", 4 >( 0.3 );
    auto p = tax::variable< "p", 4 >( 2.0 );
    return x * x * p + tax::pow( p, 3 );
}();
static_assert( kNamed.value() != 0.0, "named polynomial pipeline in constant evaluation" );

// A trivial runtime anchor so the file registers as a test target.
TEST( ConstexprPolynomial, CompileTimeChecksHold ) { SUCCEED(); }
