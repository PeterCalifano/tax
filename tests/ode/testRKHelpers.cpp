// tests/ode/testRKHelpers.cpp
//
// Direct unit tests for the shared adaptive_rk_step driver and the
// Hermite cubic interpolator. Verified against a 4-stage classical
// RK4 tableau (degenerate "embedded" estimator = b weights so
// err_norm == 0; this just checks the stage propagation).

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <array>
#include <cmath>

#include <tax/ode/detail/adaptive_rk_step.hpp>
#include <tax/ode/detail/hermite_interp.hpp>

namespace
{

// Classical RK4 as a Butcher tableau: 4 stages, FSAL = false.
struct RK4Tab
{
    static constexpr int n_stages = 4;
    static constexpr int order    = 4;
    static constexpr int order_emb = 4;          // degenerate (err = 0)
    static constexpr bool fsal    = false;

    // c_i = column of nodes
    static constexpr std::array< double, 4 > c{ 0.0, 0.5, 0.5, 1.0 };

    // a_ij flattened row-major (lower-triangular, no diagonal):
    // a[0]=a10, a[1]=a20, a[2]=a21, a[3]=a30, a[4]=a31, a[5]=a32
    static constexpr std::array< double, 6 > a{
        0.5,
        0.0, 0.5,
        0.0, 0.0, 1.0
    };

    static constexpr std::array< double, 4 > b{ 1.0 / 6, 1.0 / 3, 1.0 / 3, 1.0 / 6 };
    static constexpr std::array< double, 4 > b_emb = b;  // degenerate
};

}  // namespace

TEST( OdeRKHelpers, RK4OneStepOnExp )
{
    using State = Eigen::Matrix< double, 1, 1 >;
    State x; x( 0 ) = 1.0;

    auto f = []( const State& y, double ) { return y; };

    tax::ode::detail::RKStepData< State, 4 > stages;
    auto out = tax::ode::detail::adaptive_rk_step< RK4Tab >( f, x, 0.0, 0.1, stages );

    // x(0.1) = e^0.1 ≈ 1.10517091808...
    EXPECT_NEAR( out.x_new( 0 ), std::exp( 0.1 ), 1e-7 );
    // RK4's degenerate b_emb yields zero error.
    EXPECT_DOUBLE_EQ( out.err_norm, 0.0 );
}

TEST( OdeRKHelpers, HermiteReproducesBoundaries )
{
    using State = Eigen::Matrix< double, 2, 1 >;
    const double t0 = 1.0, t1 = 2.0;
    State x0; x0( 0 ) = 0.5; x0( 1 ) = -1.0;
    State x1; x1( 0 ) = 0.7; x1( 1 ) = -0.3;
    State f0; f0( 0 ) = 0.2; f0( 1 ) =  0.8;
    State f1; f1( 0 ) = 0.1; f1( 1 ) = -0.4;

    const State at_t0 = tax::ode::detail::hermite_interp( x0, x1, f0, f1, t0, t1, t0 );
    const State at_t1 = tax::ode::detail::hermite_interp( x0, x1, f0, f1, t0, t1, t1 );
    EXPECT_NEAR( ( at_t0 - x0 ).norm(), 0.0, 1e-14 );
    EXPECT_NEAR( ( at_t1 - x1 ).norm(), 0.0, 1e-14 );
}
