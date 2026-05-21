// tests/ode/testConcepts.cpp
//
// Compile-time validation that:
//   1. A trivial fake stepper satisfies concepts::Stepper.
//   2. The same fake stepper extended with adaptive fields
//      satisfies concepts::AdaptiveStepper.
//   3. The non-adaptive fake does NOT satisfy AdaptiveStepper.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <functional>

#include <tax/ode.hpp>

namespace
{

using State = Eigen::Matrix< double, 1, 1 >;

// Minimal stepper that satisfies concepts::Stepper but not
// AdaptiveStepper (no err_norm / h_next / accepted on its StepResult).
struct FakeStepper
{
    using T         = double;
    using State     = ::State;
    using Config    = tax::ode::IntegratorConfig< T >;
    using Rhs       = std::function< State( const State&, T ) >;
    using DenseData = State;

    tax::ode::StepResult< State, FakeStepper >
    step( const Rhs& /*f*/, const State& x, T /*t*/, T h, const Config& /*cfg*/ ) const
    {
        tax::ode::StepResult< State, FakeStepper > r;
        r.x_new = x;
        r.h_used = h;
        r.dense = x;
        return r;
    }

    static State eval_dense( const DenseData& d, const T& /*t0*/, const T& /*t1*/,
                             const T& /*tq*/ )
    {
        return d;
    }
};

static_assert( tax::ode::concepts::Stepper< FakeStepper >,
               "FakeStepper should satisfy concepts::Stepper" );
static_assert( !tax::ode::concepts::AdaptiveStepper< FakeStepper >,
               "FakeStepper should NOT satisfy concepts::AdaptiveStepper" );

TEST( OdeConcepts, FakeStepperCompiles )
{
    FakeStepper s;
    FakeStepper::Rhs f = []( const State& x, double ) { return x; };
    State x = State::Zero();
    auto r = s.step( f, x, 1.0, 0.1, tax::ode::IntegratorConfig< double >{} );
    EXPECT_DOUBLE_EQ( r.h_used, 0.1 );
}

}  // namespace
