// tests/ode/testRKHelpers.cpp
//
// Direct unit tests for the shared adaptive_rk_step driver, plus
// verification that the new concept hierarchy gates dense usage
// correctly.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <functional>

#include <tax/ode.hpp>
#include <tax/ode/detail/adaptive_rk_step.hpp>

namespace
{

// Classical RK4 as a Butcher tableau: 4 stages.
struct RK4Tab
{
    static constexpr int n_stages   = 4;
    static constexpr int order      = 4;
    static constexpr int order_emb  = 4;          // degenerate (err = 0)
    static constexpr bool fsal      = false;

    static constexpr std::array< double, 4 > c{ 0.0, 0.5, 0.5, 1.0 };

    static constexpr std::array< double, 6 > a{
        0.5,
        0.0, 0.5,
        0.0, 0.0, 1.0
    };

    static constexpr std::array< double, 4 > b{ 1.0 / 6, 1.0 / 3, 1.0 / 3, 1.0 / 6 };
    static constexpr std::array< double, 4 > b_emb = b;
};

// Two fake steppers for concept tests:
using FakeState = Eigen::Matrix< double, 1, 1 >;

struct DenseFake
{
    using State = FakeState;
    using T = double;
    using Config = tax::ode::IntegratorConfig< T >;
    using Rhs = std::function< State( const State&, T ) >;
    using DenseData = State;

    static constexpr bool is_adaptive       = true;
    static constexpr bool has_dense_output  = true;

    tax::ode::StepResult< State, DenseFake > step(
        const Rhs&, const State& x, T, T h, const Config& ) const
    {
        tax::ode::StepResult< State, DenseFake > r;
        r.x_new = x; r.h_used = h; r.dense = x; r.accepted = true;
        return r;
    }

    static State eval_dense( const DenseData& d, const T&, const T&, const T& )
    {
        return d;
    }
};

struct PropagationOnlyFake
{
    using State = FakeState;
    using T = double;
    using Config = tax::ode::IntegratorConfig< T >;
    using Rhs = std::function< State( const State&, T ) >;

    static constexpr bool is_adaptive       = true;
    static constexpr bool has_dense_output  = false;

    tax::ode::StepResult< State, PropagationOnlyFake > step(
        const Rhs&, const State& x, T, T h, const Config& ) const
    {
        tax::ode::StepResult< State, PropagationOnlyFake > r;
        r.x_new = x; r.h_used = h; r.accepted = true;
        return r;
    }
};

static_assert(  tax::ode::concepts::Stepper<       DenseFake > );
static_assert(  tax::ode::concepts::DenseStepper<  DenseFake > );
static_assert(  tax::ode::concepts::AdaptiveStepper< DenseFake > );

static_assert(  tax::ode::concepts::Stepper<       PropagationOnlyFake > );
static_assert( !tax::ode::concepts::DenseStepper<  PropagationOnlyFake > );
static_assert(  tax::ode::concepts::AdaptiveStepper< PropagationOnlyFake > );

}  // namespace

TEST( OdeRKHelpers, RK4OneStepOnExp )
{
    using State = Eigen::Matrix< double, 1, 1 >;
    State x; x( 0 ) = 1.0;

    auto f = []( const State& y, double ) { return y; };

    tax::ode::detail::RKStepData< State, 4 > stages;
    auto out = tax::ode::detail::adaptive_rk_step< RK4Tab >( f, x, 0.0, 0.1, stages );

    EXPECT_NEAR( out.x_new( 0 ), std::exp( 0.1 ), 1e-7 );
    EXPECT_DOUBLE_EQ( out.err_norm, 0.0 );
}

TEST( OdeRKHelpers, TaylorStepperSatisfiesDenseStepper )
{
    // Sanity: Taylor (from Plan A) must satisfy DenseStepper after
    // the has_dense_output marker addition.
    using S = tax::ode::TaylorStepper< 8, Eigen::Matrix< double, 1, 1 > >;
    static_assert( tax::ode::concepts::DenseStepper< S > );
    SUCCEED();
}
