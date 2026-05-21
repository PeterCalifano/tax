// tests/ode/testIntegratorBasic.cpp
//
// Integrator-level smoke tests for the Taylor method (no events).

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>

#include <tax/ode.hpp>

using tax::ode::IntegratorConfig;
using tax::ode::makeTaylorIntegrator;

TEST( OdeIntegrator, ExpEndpointAccurate )
{
    constexpr int N = 16;
    using State = Eigen::Matrix< double, 1, 1 >;

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    const auto f = []( const auto& x, const auto& /*t*/ ) { return x; };

    auto integ = makeTaylorIntegrator< N, double, 1, /*Dense=*/false >( f, cfg );

    State x0; x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, /*t0=*/0.0, /*tmax=*/1.0 );

    EXPECT_GE( sol.size(), 2u );
    EXPECT_DOUBLE_EQ( sol.t.back(), 1.0 );
    EXPECT_NEAR( sol.x.back()( 0 ), std::exp( 1.0 ), 1e-11 );
}

TEST( OdeIntegrator, HarmonicQuarterPeriod )
{
    constexpr int N = 12;
    using State = Eigen::Matrix< double, 2, 1 >;

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    const auto f = []( const auto& x, const auto& /*t*/ )
    {
        using S = std::decay_t< decltype( x ) >;
        S out;
        out( 0 ) =  x( 1 );
        out( 1 ) = -x( 0 );
        return out;
    };

    auto integ = makeTaylorIntegrator< N, double, 2, /*Dense=*/false >( f, cfg );

    State x0; x0( 0 ) = 1.0; x0( 1 ) = 0.0;
    const double T_quarter = M_PI / 2.0;
    auto sol = integ.integrate( x0, 0.0, T_quarter );

    EXPECT_NEAR( sol.x.back()( 0 ),  0.0, 1e-10 );
    EXPECT_NEAR( sol.x.back()( 1 ), -1.0, 1e-10 );
}

TEST( OdeIntegrator, CubicDecayDynamicDim )
{
    constexpr int N = 14;
    using State = Eigen::VectorXd;

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    const auto f = []( const auto& x, const auto& /*t*/ )
    {
        using S = std::decay_t< decltype( x ) >;
        S out{ x.size() };
        out( 0 ) = -x( 0 ) * x( 0 ) * x( 0 );
        return out;
    };

    // Dynamic-D variant.
    auto integ = makeTaylorIntegrator< N >( f, cfg );

    State x0( 1 ); x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 1.0 );

    EXPECT_NEAR( sol.x.back()( 0 ), 1.0 / std::sqrt( 3.0 ), 1e-10 );
}
