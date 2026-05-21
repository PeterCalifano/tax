// include/tax/ode/concepts.hpp
//
// Stepper concept hierarchy.
//   - Stepper:        minimum — take one step at the supplied h.
//   - DenseStepper:   refinement — provides DenseData + eval_dense +
//                     has_dense_output marker. Required for Solution<…,
//                     Dense=true> and for ZeroCrossing events (which
//                     need a continuous extension to locate roots inside
//                     a step).
//   - AdaptiveStepper: refinement — embedded error estimator + retry
//                     loop, keyed off `static constexpr bool is_adaptive`.

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

    { s.step( f, x, t, h, cfg ) }
        -> std::same_as< StepResult< typename S::State, S > >;
};

template < class S >
concept DenseStepper = Stepper< S >
    && requires { { S::has_dense_output } -> std::convertible_to< bool >; }
    && S::has_dense_output
    && requires {
           typename S::DenseData;
           { S::eval_dense( std::declval< typename S::DenseData >(),
                            std::declval< typename S::T >(),
                            std::declval< typename S::T >(),
                            std::declval< typename S::T >() ) }
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
