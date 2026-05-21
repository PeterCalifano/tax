// include/tax/ode/event.hpp
//
// Direction / ControlFlow enums and TriggerContext struct. The
// type-erased Event<Stepper> class is defined in this same header
// after the Trigger and Action factories (triggers.hpp / actions.hpp).

#pragma once

namespace tax::ode
{

enum class Direction   { Increasing, Decreasing, Any };
enum class ControlFlow { Continue, Terminate };

template < class State, class T, class DenseData >
struct TriggerContext
{
    const State&     x_old;
    T                t_old;
    const State&     x_new;
    T                h_used;
    const DenseData& dense;
};

}  // namespace tax::ode
