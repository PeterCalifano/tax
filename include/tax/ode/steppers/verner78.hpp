// include/tax/ode/steppers/verner78.hpp
//
// Verner 8(7) RK stepper. 13 stages, propagates at order 8, uses an
// order-7 embedded estimator for adaptive step-size control. The
// continuous extension is provided by a cubic-Hermite spline between
// step boundaries; has_dense_output = false to signal that
// dense-output accuracy is third-order (cubic) rather than the
// method's eighth-order propagation accuracy.

#pragma once

#include <tax/la/types.hpp>
#include <functional>
#include <type_traits>
#include <utility>

#include <tax/ode/config.hpp>
#include <tax/ode/controllers.hpp>
#include <tax/ode/detail/adaptive_rk_step.hpp>
#include <tax/ode/detail/hermite_interp.hpp>
#include <tax/ode/detail/verner_tableaus.hpp>
#include <tax/ode/step_result.hpp>
#include <tax/ode/vector_ops.hpp>

namespace tax::ode
{

template < class StateT,
           class Controller = controllers::PI< double > >
struct Verner78Stepper
{
    using State  = StateT;
    using T      = double;
    using Config = IntegratorConfig< double >;
    using Rhs    = std::function< State( const State&, T ) >;
    using Tab    = detail::Verner78Tab;

    static constexpr bool is_adaptive      = true;
    static constexpr bool has_dense_output = false;  // Hermite-cubic fallback
    static constexpr int  order_v          = Tab::order;
    static constexpr int  order_emb_v      = Tab::order_emb;

    // DenseData: boundary samples + their derivatives for the
    // cubic-Hermite continuous extension, plus the step length so the
    // interpolation knows the [0, h_step] domain it lives on.
    struct DenseData
    {
        State x0{};
        State x1{};
        State f0{};
        State f1{};
        T     h_step{};
    };

    template < class F >
    [[nodiscard]] StepResult< State, Verner78Stepper > step(
        F&& f, const State& x, T t, T h, const Config& cfg )
    {
        using detail::adaptive_rk_step;
        using detail::RKStepData;

        RKStepData< State, Tab::n_stages > work;
        auto out = adaptive_rk_step< Tab >( f, x, t, h, work );

        const double x_norm = VectorOps< State >::norm( out.x_new );
        const double tol    = cfg.abstol + cfg.reltol * x_norm;

        const auto [ h_next, accepted ] = detail::select_rk_step(
            controller_, h, out.err_norm, out.err_norm, tol, Tab::order_emb );

        DenseData dd;
        dd.x0     = x;
        dd.x1     = out.x_new;
        dd.f0     = work.k[ 0 ];  // f(x, t + c[0]*h) == f(x, t)
        dd.f1     = f( out.x_new, t + h );
        dd.h_step = h;

        StepResult< State, Verner78Stepper > r;
        r.x_new    = std::move( out.x_new );
        r.h_used   = h;
        r.dense    = std::move( dd );
        r.h_next   = h_next;
        r.err_norm = out.err_norm;
        r.accepted = accepted;
        return r;
    }

    [[nodiscard]] static State eval_dense(
        const DenseData& d, const T& t0, const T& tq )
    {
        return detail::hermite_interp< State, T >(
            d.x0, d.x1, d.f0, d.f1, d.h_step, tq - t0 );
    }

private:
    Controller controller_{};
};

}  // namespace tax::ode
