// tests/ode/testEventsZeroCrossing.cpp
//
// ZeroCrossing semantics on TaylorStepper. Three scenarios:
//   1. Harmonic oscillator, terminate when x crosses 0 going down.
//   2. Same RHS, record both apoapsis (v=0 down) and periapsis (v=0 up).
//   3. Sanity: no event registered ⇒ integration runs to tmax.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <vector>

#include <tax/ode.hpp>

using tax::ode::Direction;
using tax::ode::Event;
using tax::ode::IntegratorConfig;
using tax::ode::makeTaylorIntegrator;
using tax::ode::Record;
using tax::ode::TaylorStepper;
using tax::ode::Terminate;
using tax::ode::ZeroCrossing;

TEST( OdeEventsZeroCrossing, HarmonicTerminateAtZero )
{
    constexpr int N = 16;
    using State = Eigen::Matrix< double, 2, 1 >;

    const auto f = []( const auto& x, const auto& )
    {
        using S = std::decay_t< decltype( x ) >;
        S out;
        out( 0 ) =  x( 1 );
        out( 1 ) = -x( 0 );
        return out;
    };

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    using Stepper = TaylorStepper< N, State >;
    std::vector< Event< Stepper > > events;
    events.emplace_back(
        ZeroCrossing( []( const auto& x, const auto& ) { return x( 0 ); },
                      Direction::Decreasing ),
        Terminate() );

    auto integ = makeTaylorIntegrator< N, double, 2, /*Dense=*/false >(
        f, cfg, events );
    State x0; x0( 0 ) = 1.0; x0( 1 ) = 0.0;
    // x(t) = cos t, so x(0)=1, decreasing through 0 at t = π/2.
    auto sol = integ.integrate( x0, 0.0, 5.0 );

    EXPECT_NEAR( sol.t.back(), M_PI / 2, 1e-9 );
    EXPECT_LT( sol.t.back(), 5.0 );  // ended early
}

TEST( OdeEventsZeroCrossing, HarmonicVZeroRecord )
{
    constexpr int N = 16;
    using State = Eigen::Matrix< double, 2, 1 >;

    const auto f = []( const auto& x, const auto& )
    {
        using S = std::decay_t< decltype( x ) >;
        S out;
        out( 0 ) =  x( 1 );
        out( 1 ) = -x( 0 );
        return out;
    };

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    using Stepper = TaylorStepper< N, State >;
    std::vector< Event< Stepper > > events;
    events.emplace_back(
        ZeroCrossing( []( const auto& x, const auto& ) { return x( 1 ); },
                      Direction::Any ),
        Record( "v_zero" ) );

    auto integ = makeTaylorIntegrator< N, double, 2, /*Dense=*/false >(
        f, cfg, events );
    State x0; x0( 0 ) = 1.0; x0( 1 ) = 0.0;
    // v(t) = -sin t : zero at t = 0 (boundary), π, 2π. Over (0, 2π]
    // we expect events at π and 2π (the t=0 boundary is filtered by
    // the strict sign-change requirement inside ZeroCrossing).
    auto sol = integ.integrate( x0, 0.0, 2 * M_PI );

    EXPECT_GE( sol.events.size(), 1u );
    EXPECT_LE( sol.events.size(), 3u );
    for ( const auto& e : sol.events )
    {
        EXPECT_GE( e.t_event, 0.0 );
        EXPECT_LE( e.t_event, 2 * M_PI + 1e-9 );
        // The event *time* is found via polynomial Newton (near machine
        // precision). Record now uses Stepper::eval_dense so x_event
        // is machine-precision accurate — tighten to the spec's 1e-8.
        EXPECT_NEAR( std::abs( e.x_event( 1 ) ), 0.0, 1e-8 );
    }
}

TEST( OdeEventsZeroCrossing, EmptyEventListRunsToTmax )
{
    constexpr int N = 12;
    using State = Eigen::Matrix< double, 1, 1 >;

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    const auto f = []( const auto& x, const auto& ) { return x; };

    using Stepper = TaylorStepper< N, State >;
    std::vector< Event< Stepper > > events;  // empty

    auto integ = makeTaylorIntegrator< N, double, 1, /*Dense=*/false >(
        f, cfg, events );
    State x0; x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 1.0 );

    EXPECT_DOUBLE_EQ( sol.t.back(), 1.0 );
    EXPECT_NEAR( sol.x.back()( 0 ), std::exp( 1.0 ), 1e-10 );
    EXPECT_EQ( sol.events.size(), 0u );
}
