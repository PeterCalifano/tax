#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <vector>

#include <tax/tax.hpp>
#include <tax/ode/taylor_integrator.hpp>

using namespace tax;

namespace
{

template < int N >
auto makeIntegrator( double abstol )
{
    return ode::Integrator< N >{ ode::IntegratorConfig< double >{ .abstol = abstol } };
}

}  // namespace

// =============================================================================
// Scalar ODE tests
// =============================================================================

// dx/dt = x  →  x(t) = x0 * exp(t)
TEST( TaylorIntegratorScalar, ExponentialGrowth )
{
    constexpr int N      = 25;
    const double  x0     = 1.0;
    const double  t0     = 0.0;
    const double  tmax   = 1.0;
    const double  abstol = 1e-20;

    auto f = []( const auto& x, [[maybe_unused]] const auto& t ) { return x; };

    auto sol = makeIntegrator< N >( abstol ).integrate( f, x0, t0, tmax );

    EXPECT_GT( sol.t.size(), 1u );
    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );
    EXPECT_NEAR( sol.x.back(), std::exp( tmax ), 1e-14 );
}

// dx/dt = -x  →  x(t) = x0 * exp(-t)
TEST( TaylorIntegratorScalar, ExponentialDecay )
{
    constexpr int N      = 25;
    const double  x0     = 3.0;
    const double  t0     = 0.0;
    const double  tmax   = 2.0;
    const double  abstol = 1e-20;

    auto f = []( const auto& x, [[maybe_unused]] const auto& t ) { return -x; };

    auto sol = makeIntegrator< N >( abstol ).integrate( f, x0, t0, tmax );

    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );
    EXPECT_NEAR( sol.x.back(), x0 * std::exp( -tmax ), 1e-14 );
}

// dx/dt = cos(t)  →  x(t) = sin(t)   (x(0) = 0)
TEST( TaylorIntegratorScalar, CosineForcing )
{
    constexpr int N      = 25;
    const double  x0     = 0.0;
    const double  t0     = 0.0;
    const double  tmax   = std::numbers::pi;
    const double  abstol = 1e-20;

    auto f = []( [[maybe_unused]] const auto& x, const auto& t ) {
        using std::cos;
        return cos( t );
    };

    auto sol = makeIntegrator< N >( abstol ).integrate( f, x0, t0, tmax );

    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );
    EXPECT_NEAR( sol.x.back(), std::sin( tmax ), 1e-13 );
}

// dx/dt = 2*t  →  x(t) = t^2   (x(0) = 0)
TEST( TaylorIntegratorScalar, Quadratic )
{
    constexpr int N      = 10;
    const double  x0     = 0.0;
    const double  t0     = 0.0;
    const double  tmax   = 5.0;
    const double  abstol = 1e-20;

    auto f = []( [[maybe_unused]] const auto& x, const auto& t ) { return 2.0 * t; };

    auto sol = makeIntegrator< N >( abstol ).integrate( f, x0, t0, tmax );

    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );
    EXPECT_NEAR( sol.x.back(), tmax * tmax, 1e-12 );
}

// Backward integration: dx/dt = -x from x(0) = e^2 to t = -2.
// Solution: x(t) = e^(2 - t)·... actually for backward we set x0 at t0=0
// and integrate to tmax = -2: solution x(-2) = x0*exp(-(-2)) = x0*e^2 doesn't
// match.  Instead: dx/dt = x, x(0) = 1 → x(-1) = e^{-1}.
TEST( TaylorIntegratorScalar, BackwardIntegration )
{
    constexpr int N      = 25;
    const double  x0     = 1.0;
    const double  t0     = 0.0;
    const double  tmax   = -1.0;
    const double  abstol = 1e-20;

    auto f = []( const auto& x, [[maybe_unused]] const auto& t ) { return x; };

    auto sol = makeIntegrator< N >( abstol ).integrate( f, x0, t0, tmax );

    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );
    EXPECT_NEAR( sol.x.back(), std::exp( -1.0 ), 1e-14 );
}

// =============================================================================
// Dense output (scalar)
// =============================================================================

TEST( TaylorIntegratorScalar, DenseOutput )
{
    constexpr int N      = 25;
    const double  x0     = 1.0;
    const double  t0     = 0.0;
    const double  tmax   = 2.0;
    const double  abstol = 1e-20;

    auto f = []( const auto& x, [[maybe_unused]] const auto& t ) { return x; };

    auto sol = makeIntegrator< N >( abstol ).integrate( f, x0, t0, tmax );

    EXPECT_FALSE( sol.p.empty() );
    EXPECT_EQ( sol.p.size(), sol.t.size() - 1 );

    for ( double t = 0.0; t <= 2.0; t += 0.07 )
    {
        EXPECT_NEAR( sol( t ), std::exp( t ), 1e-13 ) << "  at t=" << t;
    }
}

// =============================================================================
// Vector ODE tests
// =============================================================================

TEST( TaylorIntegratorVector, HarmonicOscillator )
{
    constexpr int N      = 25;
    const double  t0     = 0.0;
    const double  tmax   = 2.0 * std::numbers::pi;
    const double  abstol = 1e-20;

    Eigen::Vector2d x0( 1.0, 0.0 );

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 );
    };

    auto sol = makeIntegrator< N >( abstol ).integrate( f, x0, t0, tmax );

    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );
    EXPECT_NEAR( sol.x.back()( 0 ), 1.0, 1e-12 );
    EXPECT_NEAR( sol.x.back()( 1 ), 0.0, 1e-12 );
}

TEST( TaylorIntegratorVector, DecoupledExponentials )
{
    constexpr int N      = 25;
    const double  t0     = 0.0;
    const double  tmax   = 1.0;
    const double  abstol = 1e-20;

    Eigen::Vector2d x0( 1.0, 1.0 );

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 0 );
        dx( 1 ) = -x( 1 );
    };

    auto sol = makeIntegrator< N >( abstol ).integrate( f, x0, t0, tmax );

    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );
    EXPECT_NEAR( sol.x.back()( 0 ), std::exp( tmax ), 1e-14 );
    EXPECT_NEAR( sol.x.back()( 1 ), std::exp( -tmax ), 1e-14 );
}

TEST( TaylorIntegratorVector, KeplerCircularOrbit )
{
    constexpr int N      = 25;
    const double  t0     = 0.0;
    const double  tmax   = 2.0 * std::numbers::pi;
    const double  abstol = 1e-20;

    Eigen::Vector< double, 4 > x0;
    x0 << 1.0, 0.0, 0.0, 1.0;

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        using std::sqrt;
        auto r2 = x( 0 ) * x( 0 ) + x( 1 ) * x( 1 );
        auto r  = sqrt( r2 );
        auto r3 = r2 * r;
        dx( 0 ) = x( 2 );
        dx( 1 ) = x( 3 );
        dx( 2 ) = -x( 0 ) / r3;
        dx( 3 ) = -x( 1 ) / r3;
    };

    auto sol = makeIntegrator< N >( abstol ).integrate( f, x0, t0, tmax );

    EXPECT_NEAR( sol.x.back()( 0 ), 1.0, 1e-10 );
    EXPECT_NEAR( sol.x.back()( 1 ), 0.0, 1e-10 );
    EXPECT_NEAR( sol.x.back()( 2 ), 0.0, 1e-10 );
    EXPECT_NEAR( sol.x.back()( 3 ), 1.0, 1e-10 );

    const auto& xf = sol.x.back();
    double      r  = std::sqrt( xf( 0 ) * xf( 0 ) + xf( 1 ) * xf( 1 ) );
    double      v2 = xf( 2 ) * xf( 2 ) + xf( 3 ) * xf( 3 );
    EXPECT_NEAR( 0.5 * v2 - 1.0 / r, -0.5, 1e-10 );
}

// =============================================================================
// Dense output (vector)
// =============================================================================

TEST( TaylorIntegratorVector, DenseOutput )
{
    constexpr int N      = 25;
    const double  abstol = 1e-20;

    Eigen::Vector2d x0( 1.0, 0.0 );

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 );
    };

    auto sol = makeIntegrator< N >( abstol ).integrate( f, x0, 0.0, 3.0 );

    EXPECT_FALSE( sol.p.empty() );

    for ( double t = 0.0; t <= 3.0; t += 0.13 )
    {
        auto y = sol( t );
        EXPECT_NEAR( y( 0 ), std::cos( t ), 1e-13 ) << "  x1 at t=" << t;
        EXPECT_NEAR( y( 1 ), -std::sin( t ), 1e-13 ) << "  x2 at t=" << t;
    }
}

// =============================================================================
// Configuration validation
// =============================================================================

TEST( TaylorIntegratorConfig, RejectsNonPositiveTolerance )
{
    EXPECT_THROW(
        ( ode::Integrator< 10 >{ ode::IntegratorConfig< double >{ .abstol = 0.0 } } ),
        std::invalid_argument );
    EXPECT_THROW(
        ( ode::Integrator< 10 >{ ode::IntegratorConfig< double >{ .abstol = -1e-12 } } ),
        std::invalid_argument );
}

TEST( TaylorIntegratorConfig, RejectsNonPositiveMaxSteps )
{
    EXPECT_THROW(
        ( ode::Integrator< 10 >{ ode::IntegratorConfig< double >{ .max_steps = 0 } } ),
        std::invalid_argument );
}

// =============================================================================
// Low-level API: step returns TTE
// =============================================================================

TEST( TaylorIntegratorStep, ScalarReturnsPolynomial )
{
    constexpr int N = 10;

    auto f = []( const auto& x, [[maybe_unused]] const auto& t ) { return x; };

    auto [p, h] = ode::step< N >( f, 1.0, 0.0, 1e-20 );

    double factorial = 1.0;
    for ( int k = 0; k <= N; ++k )
    {
        if ( k > 0 ) factorial *= k;
        EXPECT_NEAR( p[k], 1.0 / factorial, 1e-15 ) << "  coeff k=" << k;
    }

    EXPECT_GT( h, 0.0 );
    EXPECT_LT( h, 100.0 );
}

TEST( TaylorIntegratorStep, ScalarPolynomialEval )
{
    constexpr int N = 25;

    auto f = []( const auto& x, [[maybe_unused]] const auto& t ) { return x; };

    auto [p, h] = ode::step< N >( f, 1.0, 0.0, 1e-20 );

    EXPECT_NEAR( p.eval( h ), std::exp( h ), 1e-14 );
    EXPECT_NEAR( p.eval( 0.5 ), std::exp( 0.5 ), 1e-14 );
}
