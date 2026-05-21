// include/tax/ode/step_result.hpp
//
// Result type returned by every Stepper's step() method. The `dense`
// field is the stepper's DenseData payload — for propagation-only
// steppers (has_dense_output == false) this is the trivial empty
// type EmptyDenseData (zero size) so the StepResult layout stays
// uniform.

#pragma once

#include <type_traits>

namespace tax::ode
{

struct EmptyDenseData { };

namespace detail
{
template < class S, class = void >
struct stepper_dense_data { using type = EmptyDenseData; };

template < class S >
struct stepper_dense_data< S, std::void_t< typename S::DenseData > >
{ using type = typename S::DenseData; };

template < class S >
using stepper_dense_data_t = typename stepper_dense_data< S >::type;
}  // namespace detail

template < class State, class Stepper >
struct StepResult
{
    State                                              x_new{};
    typename Stepper::T                                h_used{};
    detail::stepper_dense_data_t< Stepper >            dense{};
    // Adaptive-only — meaningful when Stepper satisfies AdaptiveStepper.
    typename Stepper::T                                h_next{};
    typename Stepper::T                                err_norm{};
    bool                                               accepted = true;
};

}  // namespace tax::ode
