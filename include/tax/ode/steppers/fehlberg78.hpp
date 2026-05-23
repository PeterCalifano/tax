// include/tax/ode/steppers/fehlberg78.hpp
//
// Classical Fehlberg 1968 RK 7(8) pair. 13 stages, propagates at order 7,
// uses an order-8 embedded estimator for adaptive step-size control. Known
// to suffer from the "Fehlberg coincidence" (embedded estimator zero on
// certain steps); see spec Risks. The continuous extension is provided by
// a cubic-Hermite spline between step boundaries; has_dense_output = false
// to signal that dense-output accuracy is third-order (cubic) rather than
// the method's seventh-order propagation accuracy.

#pragma once

#include <Eigen/Core>
#include <functional>
#include <type_traits>
#include <utility>

#include <tax/ode/config.hpp>
#include <tax/ode/controllers.hpp>
#include <tax/ode/detail/adaptive_rk_step.hpp>
#include <tax/ode/detail/fehlberg_tableaus.hpp>
#include <tax/ode/detail/hermite_interp.hpp>
#include <tax/ode/step_result.hpp>

namespace tax::ode
{

template < class StateT,
           class Controller = controllers::PI< typename StateT::Scalar > >
struct Fehlberg78Stepper
{
    using State  = StateT;
    using T      = typename State::Scalar;
    using Config = IntegratorConfig< T >;
    using Rhs    = std::function< State( const State&, T ) >;
    using Tab    = detail::Fehlberg78Tab;

    static constexpr bool is_adaptive      = true;
    static constexpr bool has_dense_output = false;  // Hermite-cubic fallback
    static constexpr int  order_v          = Tab::order;
    static constexpr int  order_emb_v      = Tab::order_emb;

    // DenseData: boundary samples + their derivatives for the
    // cubic-Hermite continuous extension.
    struct DenseData
    {
        State x0{};
        State x1{};
        State f0{};
        State f1{};
    };

    template < class F >
    [[nodiscard]] StepResult< State, Fehlberg78Stepper > step(
        F&& f, const State& x, T t, T h, const Config& cfg )
    {
        using detail::adaptive_rk_step;
        using detail::RKStepData;

        RKStepData< State, Tab::n_stages > work;
        auto out = adaptive_rk_step< Tab >( f, x, t, h, work );

        T x_norm{ 0 };
        for ( Eigen::Index i = 0; i < x.size(); ++i )
        {
            using std::abs;
            const T a = T( abs( out.x_new( i ) ) );
            if ( a > x_norm ) x_norm = a;
        }
        const T tol = cfg.abstol + cfg.reltol * x_norm;

        T    h_next;
        bool accepted;
        if constexpr ( std::is_same_v< Controller, controllers::FixedStep< T > > )
        {
            h_next   = h;
            accepted = true;
        }
        else if constexpr ( std::is_same_v< Controller, controllers::JorbaZou< T > > )
        {
            h_next   = h;  // JorbaZou is Taylor-only; no-op fallback.
            accepted = out.err_norm <= tol;
        }
        else
        {
            h_next   = controller_.next_step( h, out.err_norm, tol, Tab::order_emb );
            accepted = out.err_norm <= tol;
        }

        DenseData dd;
        dd.x0 = x;
        dd.x1 = out.x_new;
        dd.f0 = work.k[ 0 ];  // f(x, t + c[0]*h) == f(x, t)
        dd.f1 = f( out.x_new, t + h );

        StepResult< State, Fehlberg78Stepper > r;
        r.x_new    = std::move( out.x_new );
        r.h_used   = h;
        r.dense    = std::move( dd );
        r.h_next   = h_next;
        r.err_norm = out.err_norm;
        r.accepted = accepted;
        return r;
    }

    [[nodiscard]] static State eval_dense(
        const DenseData& d, const T& t0, const T& t1, const T& tq )
    {
        return detail::hermite_interp< State, T >(
            d.x0, d.x1, d.f0, d.f1, t0, t1, tq );
    }

private:
    Controller controller_{};
};

}  // namespace tax::ode
