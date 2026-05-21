// tests/ode/testIntegratorDense.cpp
//
// Dense-mode (Dense=true) integration smoke tests. sol(t_query) for
// any t in [t0, tmax] must agree with the closed-form solution
// within the step's local truncation tolerance.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <stdexcept>

#include <tax/ode.hpp>

using tax::ode::IntegratorConfig;
using tax::ode::makeTaylorIntegrator;

TEST( OdeIntegratorDense, ExpDenseInside )
{
    constexpr int N = 16;
    using State = Eigen::Matrix< double, 1, 1 >;

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    const auto f = []( const auto& x, const auto& /*t*/ ) { return x; };

    auto integ = makeTaylorIntegrator< N, double, 1, /*Dense=*/true >( f, cfg );

    State x0; x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 1.0 );

    // Query at multiple intermediate times.
    for ( const double tq : { 0.07, 0.23, 0.5, 0.83, 0.99 } )
    {
        State x_at_tq = sol( tq );
        EXPECT_NEAR( x_at_tq( 0 ), std::exp( tq ), 1e-10 )
            << "tq=" << tq;
    }

    // Boundaries.
    EXPECT_NEAR( sol( 0.0 )( 0 ), 1.0,              1e-12 );
    EXPECT_NEAR( sol( 1.0 )( 0 ), std::exp( 1.0 ),  1e-11 );
}

TEST( OdeIntegratorDense, OutOfRangeThrows )
{
    constexpr int N = 8;
    using State = Eigen::Matrix< double, 1, 1 >;

    IntegratorConfig< double > cfg;
    const auto f = []( const auto& x, const auto& /*t*/ ) { return x; };

    auto integ = makeTaylorIntegrator< N, double, 1, /*Dense=*/true >( f, cfg );

    State x0; x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 0.5 );

    EXPECT_THROW( sol( -0.1 ), std::out_of_range );
    EXPECT_THROW( sol(  0.6 ), std::out_of_range );
}
