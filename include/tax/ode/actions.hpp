// include/tax/ode/actions.hpp
//
// Action factories. Each factory returns a callable with the signature
//   ControlFlow(const StepperCtx<…>&, T tau_fired, EventSink<State, T>&).
//
// The action is invoked by the Integrator after a Trigger has fired
// with the τ at which the firing took place. Actions decide whether
// the integration continues (ControlFlow::Continue) or terminates
// (ControlFlow::Terminate) and may push EventRecord entries into the
// Solution via the provided sink.

#pragma once

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <tax/ode/event.hpp>
#include <tax/ode/solution.hpp>

namespace tax::ode
{

// Sink that the Integrator hands to each Action so it can push
// EventRecords into the Solution. Defined here (one definition) and
// forward declared in event.hpp.
template < class State, class T >
struct EventSink
{
    std::vector< EventRecord< State, T > >* events;

    void push( EventRecord< State, T > rec )
    {
        if ( events ) events->push_back( std::move( rec ) );
    }
};

// Continue — no-op action that always asks the Integrator to keep going.
inline auto Continue()
{
    return []< class Ctx, class T, class Sink >(
               const Ctx&, T, Sink& ) -> ControlFlow
    {
        return ControlFlow::Continue;
    };
}

// Terminate — always asks the Integrator to halt at the event.
inline auto Terminate()
{
    return []< class Ctx, class T, class Sink >(
               const Ctx&, T, Sink& ) -> ControlFlow
    {
        return ControlFlow::Terminate;
    };
}

// Record(label) — push an EventRecord with a linearly-interpolated
// state at τ into the sink and continue. The linear interpolant is
// Stage 2a's good-enough event coordinate; Stepper-specific Record
// can refine via eval_dense in a later stage.
inline auto Record( std::string label )
{
    return [ lbl = std::move( label ) ]<
               class Ctx, class T, class Sink >(
               const Ctx& ctx, T tau, Sink& sink ) -> ControlFlow
    {
        using SinkState = std::remove_cvref_t< decltype( ctx.x_old ) >;
        const T frac = ( ctx.h_used != T{ 0 } ) ? tau / ctx.h_used : T{ 0 };
        SinkState x_event = ctx.x_old + frac * ( ctx.x_new - ctx.x_old );
        sink.push( { lbl, ctx.t_old + tau, std::move( x_event ) } );
        return ControlFlow::Continue;
    };
}

// Custom(fn) — wrap a user callable as an Action. The wrapped callable
// must accept (Ctx, T, Sink) and return ControlFlow.
template < class Fn >
auto Custom( Fn fn )
{
    return [ fn = std::move( fn ) ]<
               class Ctx, class T, class Sink >(
               const Ctx& ctx, T tau, Sink& sink ) -> ControlFlow
    {
        return fn( ctx, tau, sink );
    };
}

}  // namespace tax::ode
