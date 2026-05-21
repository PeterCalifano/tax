// include/tax/ode/concepts.hpp
//
// Stepper concept hierarchy.
//   - Stepper:         minimum — take one step at the supplied h.
//   - AdaptiveStepper: refinement — additionally provides embedded
//                      error estimate and recommended next step.
//                      Keyed off a per-Stepper
//                      `static constexpr bool is_adaptive = true;`
//                      marker so that steppers sharing the same
//                      StepResult struct layout can still be
//                      discriminated at compile time.

#pragma once

#include <concepts>
#include <utility>

#include <tax/ode/step_result.hpp>

namespace tax::ode::concepts
{

template < class S >
concept Stepper = requires(
    S s,
    typename S::Rhs f,
    typename S::State x,
    typename S::T t,
    typename S::T h,
    const typename S::Config& cfg )
{
    typename S::State;
    typename S::T;
    typename S::Config;
    typename S::Rhs;
    typename S::DenseData;

    { s.step( f, x, t, h, cfg ) }
        -> std::same_as< StepResult< typename S::State, S > >;

    { S::eval_dense( std::declval< typename S::DenseData >(), t, t, t ) }
        -> std::same_as< typename S::State >;
};

// AdaptiveStepper: a Stepper that declares
//   static constexpr bool is_adaptive = true;
// Concrete adaptive steppers set this flag; non-adaptive or fixed-step
// steppers omit it (or set it to false) so `if constexpr` dispatch in
// the Integrator core loop can skip rejection/retry logic.
template < class S >
concept AdaptiveStepper = Stepper< S >
    && requires { { S::is_adaptive } -> std::convertible_to< bool >; }
    && S::is_adaptive;

}  // namespace tax::ode::concepts
