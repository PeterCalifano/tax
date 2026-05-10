#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <vector>

#include <tax/tax.hpp>
#include <tax/ode/taylor_integrator.hpp>

using namespace tax;

// =============================================================================
// Scalar ODE: terminal event with known crossing
// =============================================================================

// dx/dt = -2*t  →  x(t) = 1 - t^2.  Event: x = 0 at t = 1, decreasing in t.
TEST( OdeEvents, ScalarTerminalCrossing )
{
    constexpr int N = 15;
    const double abstol = 1e-18;

    auto f = []( [[maybe_unused]] const auto& x, const auto& t ) { return -2.0 * t; };

    std::vector< ode::Event< N, double, double > > events( 1 );
    events[0].g = []( const auto& x, [[maybe_unused]] const auto& t ) { return x; };
    events[0].direction = ode::EventDirection::Decreasing;
    events[0].terminal = true;

    auto sol = ode::integrate< N >( f, 1.0, 0.0, 5.0, abstol, events );

    ASSERT_EQ( sol.events.size(), 1u );
    EXPECT_EQ( sol.events[0].event_idx, 0u );
    EXPECT_EQ( sol.events[0].direction, ode::EventDirection::Decreasing );
    EXPECT_NEAR( sol.events[0].t, 1.0, 1e-12 );
    EXPECT_NEAR( sol.events[0].x, 0.0, 1e-12 );

    // Integration must have stopped at the event time, not at tmax.
    EXPECT_NEAR( sol.t.back(), 1.0, 1e-12 );
    EXPECT_LT( sol.t.back(), 5.0 );
    EXPECT_NEAR( sol.x.back(), 0.0, 1e-12 );
}

// =============================================================================
// Direction filtering excludes wrong-direction crossings
// =============================================================================

// dx/dt = cos(t) → x(t) = sin(t).  Looking for x = 0 with Increasing filter
// over [0, 2π] should match only t = 0 (skipped: not a strict sign change at
// step start) and t = 2π — but only one crossing has positive slope inside
// the open interval: every increasing zero of sin is at 2kπ.  So the first
// hit with rising direction is t = 2π.  We instead pick a clearer setup:
// look for x = 0.5 with Increasing filter — only matches the rising half.
TEST( OdeEvents, ScalarDirectionFilter )
{
    constexpr int N = 20;
    const double abstol = 1e-18;

    auto f = []( [[maybe_unused]] const auto& x, const auto& t ) {
        using std::cos;
        return cos( t );
    };

    std::vector< ode::Event< N, double, double > > events( 1 );
    events[0].g = []( const auto& x, [[maybe_unused]] const auto& t ) { return x - 0.5; };
    events[0].direction = ode::EventDirection::Increasing;
    events[0].terminal = false;

    auto sol = ode::integrate< N >( f, 0.0, 0.0, 2.0 * std::numbers::pi, abstol, events );

    // Rising zeros of x - 0.5 occur at t = π/6 and t = 2π + π/6 (out of range).
    // The falling zero at t = 5π/6 must be filtered out.
    ASSERT_EQ( sol.events.size(), 1u );
    EXPECT_EQ( sol.events[0].direction, ode::EventDirection::Increasing );
    EXPECT_NEAR( sol.events[0].t, std::numbers::pi / 6.0, 1e-10 );
    EXPECT_NEAR( sol.events[0].x, 0.5, 1e-10 );

    // Integration must have run to the end (non-terminal event).
    EXPECT_NEAR( sol.t.back(), 2.0 * std::numbers::pi, 1e-12 );
}

// =============================================================================
// Multiple events: non-terminal recorded, terminal stops integration
// =============================================================================

// dx/dt = 1 → x(t) = t.
//   Event 0: x = 1.5 (non-terminal)
//   Event 1: x = 3.0 (terminal)
// Both fire over [0, 10]; integration must stop at t = 3.
TEST( OdeEvents, ScalarMultipleEvents )
{
    constexpr int N = 10;
    const double abstol = 1e-18;

    auto f = []( [[maybe_unused]] const auto& x, [[maybe_unused]] const auto& t ) { return 1.0; };

    std::vector< ode::Event< N, double, double > > events( 2 );
    events[0].g = []( const auto& x, [[maybe_unused]] const auto& t ) { return x - 1.5; };
    events[0].terminal = false;
    events[1].g = []( const auto& x, [[maybe_unused]] const auto& t ) { return x - 3.0; };
    events[1].terminal = true;

    auto sol = ode::integrate< N >( f, 0.0, 0.0, 10.0, abstol, events );

    ASSERT_EQ( sol.events.size(), 2u );
    EXPECT_EQ( sol.events[0].event_idx, 0u );
    EXPECT_NEAR( sol.events[0].t, 1.5, 1e-12 );
    EXPECT_EQ( sol.events[1].event_idx, 1u );
    EXPECT_NEAR( sol.events[1].t, 3.0, 1e-12 );

    EXPECT_NEAR( sol.t.back(), 3.0, 1e-12 );
    EXPECT_LT( sol.t.back(), 10.0 );
}

// =============================================================================
// No-event passthrough: empty events list reproduces the no-events overload
// =============================================================================

TEST( OdeEvents, EmptyEventsRunsToTmax )
{
    constexpr int N = 15;
    const double abstol = 1e-18;

    auto f = []( const auto& x, [[maybe_unused]] const auto& t ) { return -x; };

    std::vector< ode::Event< N, double, double > > events;
    auto sol = ode::integrate< N >( f, 3.0, 0.0, 2.0, abstol, events );

    EXPECT_TRUE( sol.events.empty() );
    EXPECT_NEAR( sol.t.back(), 2.0, 1e-14 );
    EXPECT_NEAR( sol.x.back(), 3.0 * std::exp( -2.0 ), 1e-14 );
}

// =============================================================================
// Vector ODE: harmonic oscillator zero crossing
// =============================================================================

// dx/dt = v, dv/dt = -x.  IC (1, 0) → x(t) = cos(t).
// Event: x[0] = 0, terminal.  First crossing at t = π/2, decreasing in t.
TEST( OdeEvents, VectorHarmonicOscillator )
{
    constexpr int N = 20;
    const double abstol = 1e-18;

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 );
    };

    Eigen::Vector2d x0{ 1.0, 0.0 };

    std::vector< ode::Event< N, Eigen::Vector2d, double > > events( 1 );
    events[0].g = []( const auto& x, [[maybe_unused]] const auto& t ) { return x( 0 ); };
    events[0].direction = ode::EventDirection::Decreasing;
    events[0].terminal = true;

    auto sol = ode::integrate< N >( f, x0, 0.0, 5.0, abstol, events );

    ASSERT_EQ( sol.events.size(), 1u );
    EXPECT_EQ( sol.events[0].direction, ode::EventDirection::Decreasing );
    EXPECT_NEAR( sol.events[0].t, std::numbers::pi / 2.0, 1e-12 );
    EXPECT_NEAR( sol.events[0].x( 0 ), 0.0, 1e-12 );
    EXPECT_NEAR( sol.events[0].x( 1 ), -1.0, 1e-12 );  // velocity at the crossing

    EXPECT_NEAR( sol.t.back(), std::numbers::pi / 2.0, 1e-12 );
}

// =============================================================================
// Backward integration: events still detected with t-direction filter
// =============================================================================

// dx/dt = 1 → x(t) = x0 + (t - t0).  Integrate backward from t0 = 5 to tmax = 0
// with x(5) = 5, so x(t) = t.  Event x = 2 with Decreasing-in-t filter (since
// going backward in t still means x decreases as t decreases).
TEST( OdeEvents, ScalarBackwardIntegration )
{
    constexpr int N = 10;
    const double abstol = 1e-18;

    auto f = []( [[maybe_unused]] const auto& x, [[maybe_unused]] const auto& t ) { return 1.0; };

    std::vector< ode::Event< N, double, double > > events( 1 );
    events[0].g = []( const auto& x, [[maybe_unused]] const auto& t ) { return x - 2.0; };
    events[0].direction = ode::EventDirection::Increasing;  // dx/dt = 1 > 0 in t
    events[0].terminal = true;

    auto sol = ode::integrate< N >( f, 5.0, 5.0, 0.0, abstol, events );

    ASSERT_EQ( sol.events.size(), 1u );
    EXPECT_NEAR( sol.events[0].t, 2.0, 1e-12 );
    EXPECT_NEAR( sol.events[0].x, 2.0, 1e-12 );
    EXPECT_EQ( sol.events[0].direction, ode::EventDirection::Increasing );

    EXPECT_NEAR( sol.t.back(), 2.0, 1e-12 );
}

// =============================================================================
// Sensitivity surface: g_poly + step polynomial reproduce x at event time
// =============================================================================

TEST( OdeEvents, EventRecordSensitivitySurface )
{
    constexpr int N = 15;
    const double abstol = 1e-18;

    auto f = []( [[maybe_unused]] const auto& x, const auto& t ) { return -2.0 * t; };

    std::vector< ode::Event< N, double, double > > events( 1 );
    events[0].g = []( const auto& x, [[maybe_unused]] const auto& t ) { return x; };
    events[0].terminal = true;

    auto sol = ode::integrate< N >( f, 1.0, 0.0, 5.0, abstol, events );

    ASSERT_EQ( sol.events.size(), 1u );
    const auto& ev = sol.events[0];

    // Step polynomial evaluated at τ = ev.tau gives the recorded state.
    ASSERT_LT( ev.step_idx, sol.p.size() );
    EXPECT_NEAR( sol.p[ev.step_idx].eval( ev.tau ), ev.x, 1e-14 );

    // The event polynomial g_poly(τ) must be ≈ 0 at τ = ev.tau.
    EXPECT_NEAR( ev.g_poly.eval( ev.tau ), 0.0, 1e-12 );
}
