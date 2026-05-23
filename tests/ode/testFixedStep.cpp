// tests/ode/testFixedStep.cpp
//
// FixedStep controller: forces every step to use cfg.initial_step
// and to be accepted, regardless of tolerance. One test per stepper.

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <tax/ode.hpp>

using tax::ode::IntegratorConfig;
using tax::ode::controllers::FixedStep;

namespace
{

constexpr double kH = 0.1;

template < class State >
auto identity_rhs()
{
    return []( const State& x, double ) { return x; };
}

template < class Solution >
void check_uniform_grid( const Solution& sol, double h, std::size_t expected_count )
{
    ASSERT_EQ( sol.t.size(), expected_count );
    for ( std::size_t i = 0; i < sol.t.size(); ++i )
        EXPECT_NEAR( sol.t[ i ], h * double( i ), 1e-12 )
            << "step index " << i;
}

}  // namespace

TEST( OdeFixedStep, Verner78AlwaysAcceptedAtTightTol )
{
    using State = Eigen::Matrix< double, 1, 1 >;

    IntegratorConfig< double > cfg;
    cfg.initial_step = kH;
    cfg.abstol = cfg.reltol = 1e-30;        // impossibly tight; must still accept

    auto integ = tax::ode::makeVerner78Integrator< double, 1, false,
                                                    FixedStep< double > >(
        identity_rhs< State >(), cfg );

    State x0; x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 1.0 );

    check_uniform_grid( sol, kH, /*expected_count=*/11u );
}
