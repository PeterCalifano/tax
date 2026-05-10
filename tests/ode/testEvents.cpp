#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <vector>

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
// Scalar ODE: terminal event with known crossing
// =============================================================================

TEST( OdeEvents, ScalarTerminalCrossing )
{
    constexpr int N = 15;

    auto f = []( [[maybe_unused]] const auto& x, const auto& t ) { return -2.0 * t; };

    std::vector< ode::Event< N, double, double > > events( 1 );
    events[0].g         = []( const auto& x, [[maybe_unused]] const auto& t ) { return x; };
    events[0].direction = ode::EventDirection::Decreasing;
    events[0].terminal  = true;

    ode::Integrator< N, double > ig{ f, cfg( 1e-18 ), events };
    auto                          sol = ig.integrate( 1.0, 0.0, 5.0 );

    ASSERT_EQ( sol.events.size(), 1u );
    EXPECT_EQ( sol.events[0].event_idx, 0u );
    EXPECT_EQ( sol.events[0].direction, ode::EventDirection::Decreasing );
    EXPECT_NEAR( sol.events[0].t, 1.0, 1e-12 );
    EXPECT_NEAR( sol.events[0].x, 0.0, 1e-12 );

    EXPECT_NEAR( sol.t.back(), 1.0, 1e-12 );
    EXPECT_LT( sol.t.back(), 5.0 );
    EXPECT_NEAR( sol.x.back(), 0.0, 1e-12 );
}

TEST( OdeEvents, ScalarDirectionFilter )
{
    constexpr int N = 20;

    auto f = []( [[maybe_unused]] const auto& x, const auto& t ) {
        using std::cos;
        return cos( t );
    };

    std::vector< ode::Event< N, double, double > > events( 1 );
    events[0].g = []( const auto& x, [[maybe_unused]] const auto& t ) { return x - 0.5; };
    events[0].direction = ode::EventDirection::Increasing;
    events[0].terminal  = false;

    ode::Integrator< N, double > ig{ f, cfg( 1e-18 ), events };
    auto                          sol = ig.integrate( 0.0, 0.0, 2.0 * std::numbers::pi );

    ASSERT_EQ( sol.events.size(), 1u );
    EXPECT_EQ( sol.events[0].direction, ode::EventDirection::Increasing );
    EXPECT_NEAR( sol.events[0].t, std::numbers::pi / 6.0, 1e-10 );
    EXPECT_NEAR( sol.events[0].x, 0.5, 1e-10 );

    EXPECT_NEAR( sol.t.back(), 2.0 * std::numbers::pi, 1e-12 );
}

TEST( OdeEvents, ScalarMultipleEvents )
{
    constexpr int N = 10;

    auto f = []( [[maybe_unused]] const auto& x, [[maybe_unused]] const auto& t ) { return 1.0; };

    std::vector< ode::Event< N, double, double > > events( 2 );
    events[0].g        = []( const auto& x, [[maybe_unused]] const auto& t ) { return x - 1.5; };
    events[0].terminal = false;
    events[1].g        = []( const auto& x, [[maybe_unused]] const auto& t ) { return x - 3.0; };
    events[1].terminal = true;

    ode::Integrator< N, double > ig{ f, cfg( 1e-18 ), events };
    auto                          sol = ig.integrate( 0.0, 0.0, 10.0 );

    ASSERT_EQ( sol.events.size(), 2u );
    EXPECT_EQ( sol.events[0].event_idx, 0u );
    EXPECT_NEAR( sol.events[0].t, 1.5, 1e-12 );
    EXPECT_EQ( sol.events[1].event_idx, 1u );
    EXPECT_NEAR( sol.events[1].t, 3.0, 1e-12 );

    EXPECT_NEAR( sol.t.back(), 3.0, 1e-12 );
    EXPECT_LT( sol.t.back(), 10.0 );
}

TEST( OdeEvents, EmptyEventsRunsToTmax )
{
    constexpr int N = 15;

    auto f = []( const auto& x, [[maybe_unused]] const auto& t ) { return -x; };

    ode::Integrator< N, double > ig{ f, cfg( 1e-18 ) };  // no events
    auto                          sol = ig.integrate( 3.0, 0.0, 2.0 );

    EXPECT_TRUE( sol.events.empty() );
    EXPECT_NEAR( sol.t.back(), 2.0, 1e-14 );
    EXPECT_NEAR( sol.x.back(), 3.0 * std::exp( -2.0 ), 1e-14 );
}

TEST( OdeEvents, VectorHarmonicOscillator )
{
    constexpr int N = 20;

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 );
    };

    Eigen::Vector2d x0{ 1.0, 0.0 };

    std::vector< ode::Event< N, Eigen::Vector2d, double > > events( 1 );
    events[0].g = []( const auto& x, [[maybe_unused]] const auto& t ) { return x( 0 ); };
    events[0].direction = ode::EventDirection::Decreasing;
    events[0].terminal  = true;

    ode::Integrator< N, Eigen::Vector2d > ig{ f, cfg( 1e-18 ), events };
    auto                                   sol = ig.integrate( x0, 0.0, 5.0 );

    ASSERT_EQ( sol.events.size(), 1u );
    EXPECT_EQ( sol.events[0].direction, ode::EventDirection::Decreasing );
    EXPECT_NEAR( sol.events[0].t, std::numbers::pi / 2.0, 1e-12 );
    EXPECT_NEAR( sol.events[0].x( 0 ), 0.0, 1e-12 );
    EXPECT_NEAR( sol.events[0].x( 1 ), -1.0, 1e-12 );

    EXPECT_NEAR( sol.t.back(), std::numbers::pi / 2.0, 1e-12 );
}

TEST( OdeEvents, ScalarBackwardIntegration )
{
    constexpr int N = 10;

    auto f = []( [[maybe_unused]] const auto& x, [[maybe_unused]] const auto& t ) { return 1.0; };

    std::vector< ode::Event< N, double, double > > events( 1 );
    events[0].g         = []( const auto& x, [[maybe_unused]] const auto& t ) { return x - 2.0; };
    events[0].direction = ode::EventDirection::Increasing;
    events[0].terminal  = true;

    ode::Integrator< N, double > ig{ f, cfg( 1e-18 ), events };
    auto                          sol = ig.integrate( 5.0, 5.0, 0.0 );

    ASSERT_EQ( sol.events.size(), 1u );
    EXPECT_NEAR( sol.events[0].t, 2.0, 1e-12 );
    EXPECT_NEAR( sol.events[0].x, 2.0, 1e-12 );
    EXPECT_EQ( sol.events[0].direction, ode::EventDirection::Increasing );

    EXPECT_NEAR( sol.t.back(), 2.0, 1e-12 );
}

TEST( OdeEvents, EventRecordSensitivitySurface )
{
    constexpr int N = 15;

    auto f = []( [[maybe_unused]] const auto& x, const auto& t ) { return -2.0 * t; };

    std::vector< ode::Event< N, double, double > > events( 1 );
    events[0].g        = []( const auto& x, [[maybe_unused]] const auto& t ) { return x; };
    events[0].terminal = true;

    ode::Integrator< N, double > ig{ f, cfg( 1e-18 ), events };
    auto                          sol = ig.integrate( 1.0, 0.0, 5.0 );

    ASSERT_EQ( sol.events.size(), 1u );
    const auto& ev = sol.events[0];

    ASSERT_LT( ev.step_idx, sol.p.size() );
    EXPECT_NEAR( sol.p[ev.step_idx].eval( ev.tau ), ev.x, 1e-14 );
    EXPECT_NEAR( ev.g_poly.eval( ev.tau ), 0.0, 1e-12 );
}
