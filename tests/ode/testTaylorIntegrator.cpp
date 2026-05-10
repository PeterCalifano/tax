#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

#include <tax/tax.hpp>
#include <tax/ode/taylor_integrator.hpp>

using namespace tax;

namespace
{

template < typename T = double >
auto cfg( T abstol )
{
    return ode::IntegratorConfig< T >{ .abstol = abstol };
}

}  // namespace

// =============================================================================
// Scalar ODE tests
// =============================================================================

TEST( TaylorIntegratorScalar, ExponentialGrowth )
{
    constexpr int N      = 25;
    const double  x0     = 1.0;
    const double  t0     = 0.0;
    const double  tmax   = 1.0;

    auto                          f = []( const auto& x, [[maybe_unused]] const auto& t ) { return x; };
    ode::Integrator< N, double > ig{ f, cfg( 1e-20 ) };

    auto sol = ig.integrate( x0, t0, tmax );

    EXPECT_GT( sol.t.size(), 1u );
    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );
    EXPECT_NEAR( sol.x.back(), std::exp( tmax ), 1e-14 );
}

TEST( TaylorIntegratorScalar, ExponentialDecay )
{
    constexpr int N    = 25;
    const double  x0   = 3.0;
    const double  tmax = 2.0;

    auto                          f = []( const auto& x, [[maybe_unused]] const auto& t ) { return -x; };
    ode::Integrator< N, double > ig{ f, cfg( 1e-20 ) };

    auto sol = ig.integrate( x0, 0.0, tmax );

    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );
    EXPECT_NEAR( sol.x.back(), x0 * std::exp( -tmax ), 1e-14 );
}

TEST( TaylorIntegratorScalar, CosineForcing )
{
    constexpr int N    = 25;
    const double  tmax = std::numbers::pi;

    auto f = []( [[maybe_unused]] const auto& x, const auto& t ) {
        using std::cos;
        return cos( t );
    };
    ode::Integrator< N, double > ig{ f, cfg( 1e-20 ) };

    auto sol = ig.integrate( 0.0, 0.0, tmax );

    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );
    EXPECT_NEAR( sol.x.back(), std::sin( tmax ), 1e-13 );
}

TEST( TaylorIntegratorScalar, Quadratic )
{
    constexpr int N    = 10;
    const double  tmax = 5.0;

    auto                          f = []( [[maybe_unused]] const auto& x, const auto& t ) { return 2.0 * t; };
    ode::Integrator< N, double > ig{ f, cfg( 1e-20 ) };

    auto sol = ig.integrate( 0.0, 0.0, tmax );

    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );
    EXPECT_NEAR( sol.x.back(), tmax * tmax, 1e-12 );
}

TEST( TaylorIntegratorScalar, BackwardIntegration )
{
    constexpr int N    = 25;
    const double  tmax = -1.0;

    auto                          f = []( const auto& x, [[maybe_unused]] const auto& t ) { return x; };
    ode::Integrator< N, double > ig{ f, cfg( 1e-20 ) };

    auto sol = ig.integrate( 1.0, 0.0, tmax );

    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );
    EXPECT_NEAR( sol.x.back(), std::exp( -1.0 ), 1e-14 );
}

TEST( TaylorIntegratorScalar, DenseOutput )
{
    constexpr int N = 25;

    auto                          f = []( const auto& x, [[maybe_unused]] const auto& t ) { return x; };
    ode::Integrator< N, double > ig{ f, cfg( 1e-20 ) };

    auto sol = ig.integrate( 1.0, 0.0, 2.0 );

    EXPECT_FALSE( sol.p.empty() );
    EXPECT_EQ( sol.p.size(), sol.t.size() - 1 );

    for ( double t = 0.0; t <= 2.0; t += 0.07 )
        EXPECT_NEAR( sol( t ), std::exp( t ), 1e-13 ) << "  at t=" << t;
}

// =============================================================================
// Vector ODE tests
// =============================================================================

TEST( TaylorIntegratorVector, HarmonicOscillator )
{
    constexpr int N    = 25;
    const double  tmax = 2.0 * std::numbers::pi;

    Eigen::Vector2d x0( 1.0, 0.0 );

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 );
    };
    ode::Integrator< N, Eigen::Vector2d > ig{ f, cfg( 1e-20 ) };

    auto sol = ig.integrate( x0, 0.0, tmax );

    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );
    EXPECT_NEAR( sol.x.back()( 0 ), 1.0, 1e-12 );
    EXPECT_NEAR( sol.x.back()( 1 ), 0.0, 1e-12 );
}

TEST( TaylorIntegratorVector, DecoupledExponentials )
{
    constexpr int N    = 25;
    const double  tmax = 1.0;

    Eigen::Vector2d x0( 1.0, 1.0 );

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 0 );
        dx( 1 ) = -x( 1 );
    };
    ode::Integrator< N, Eigen::Vector2d > ig{ f, cfg( 1e-20 ) };

    auto sol = ig.integrate( x0, 0.0, tmax );

    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );
    EXPECT_NEAR( sol.x.back()( 0 ), std::exp( tmax ), 1e-14 );
    EXPECT_NEAR( sol.x.back()( 1 ), std::exp( -tmax ), 1e-14 );
}

TEST( TaylorIntegratorVector, KeplerCircularOrbit )
{
    constexpr int N    = 25;
    const double  tmax = 2.0 * std::numbers::pi;

    using Vec = Eigen::Vector< double, 4 >;

    Vec x0;
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
    ode::Integrator< N, Vec > ig{ f, cfg( 1e-20 ) };

    auto sol = ig.integrate( x0, 0.0, tmax );

    EXPECT_NEAR( sol.x.back()( 0 ), 1.0, 1e-10 );
    EXPECT_NEAR( sol.x.back()( 1 ), 0.0, 1e-10 );
    EXPECT_NEAR( sol.x.back()( 2 ), 0.0, 1e-10 );
    EXPECT_NEAR( sol.x.back()( 3 ), 1.0, 1e-10 );

    const auto& xf = sol.x.back();
    double      r  = std::sqrt( xf( 0 ) * xf( 0 ) + xf( 1 ) * xf( 1 ) );
    double      v2 = xf( 2 ) * xf( 2 ) + xf( 3 ) * xf( 3 );
    EXPECT_NEAR( 0.5 * v2 - 1.0 / r, -0.5, 1e-10 );
}

TEST( TaylorIntegratorVector, DenseOutput )
{
    constexpr int N = 25;

    Eigen::Vector2d x0( 1.0, 0.0 );

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 );
    };
    ode::Integrator< N, Eigen::Vector2d > ig{ f, cfg( 1e-20 ) };

    auto sol = ig.integrate( x0, 0.0, 3.0 );

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
    auto f = []( const auto& x, [[maybe_unused]] const auto& t ) { return x; };

    EXPECT_THROW(
        ( ode::Integrator< 10, double >{ f, ode::IntegratorConfig< double >{ .abstol = 0.0 } } ),
        std::invalid_argument );
    EXPECT_THROW(
        ( ode::Integrator< 10, double >{ f, ode::IntegratorConfig< double >{ .abstol = -1e-12 } } ),
        std::invalid_argument );
}

TEST( TaylorIntegratorConfig, RejectsNonPositiveMaxSteps )
{
    auto f = []( const auto& x, [[maybe_unused]] const auto& t ) { return x; };

    EXPECT_THROW(
        ( ode::Integrator< 10, double >{ f, ode::IntegratorConfig< double >{ .max_steps = 0 } } ),
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
