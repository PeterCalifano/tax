// include/tax/ode/detail/hermite_interp.hpp
//
// Cubic-Hermite interpolation between two state samples and their
// derivatives. Reproduces (x0, x1) exactly at the boundaries and is
// C^1 across the step. Used by every RK stepper's eval_dense.

#pragma once

#include <Eigen/Core>

namespace tax::ode::detail
{

template < class State, class T >
[[nodiscard]] State hermite_interp(
    const State& x0, const State& x1,
    const State& f0, const State& f1,
    const T& t0, const T& t1, const T& tq )
{
    const T h     = t1 - t0;
    const T theta = ( tq - t0 ) / h;
    const T om    = T{ 1 } - theta;

    // Standard Hermite basis on [0,1]:
    //   H00 =  (1+2θ)(1-θ)^2
    //   H10 =  θ      (1-θ)^2
    //   H01 =  θ^2    (3-2θ)
    //   H11 = -θ^2    (1-θ)
    const T h00 = ( T{ 1 } + T{ 2 } * theta ) * om * om;
    const T h10 = theta * om * om;
    const T h01 = theta * theta * ( T{ 3 } - T{ 2 } * theta );
    const T h11 = -theta * theta * om;

    State out = h00 * x0 + ( h10 * h ) * f0
              + h01 * x1 + ( h11 * h ) * f1;
    return out;
}

}  // namespace tax::ode::detail
