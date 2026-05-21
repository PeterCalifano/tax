// include/tax/ode/steppers/taylor.hpp
//
// TaylorStepper<N, State, Controller>.
//
// Propagates the ODE dx/dt = f(x, t) by computing the Taylor
// expansion of x(t) in time (univariate, order N) around the step
// start. Coefficients are obtained iteratively from f's polynomial
// composition: with x_te[i] holding c_0…c_{k-1} of component i, one
// evaluation of f(x_te, t_te) yields f_te[i].coeff(k) which
// determines x_te[i].coeff(k+1) = f_te[i].coeff(k) / (k+1).
// After N evaluations every coefficient is exact (up to truncation).
//
// Slice 2b ships the shell + eval_dense. The real Taylor expansion
// algorithm lands in slice 2c (Task 6).

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <functional>

#include <tax/core/taylor_expansion.hpp>
#include <tax/eigen.hpp>
#include <tax/ode/config.hpp>
#include <tax/ode/controllers.hpp>
#include <tax/ode/step_result.hpp>

namespace tax::ode
{

template < int N,
           class State,
           class Controller = controllers::JorbaZou< typename State::Scalar > >
struct TaylorStepper
{
    static_assert( N >= 2,
                   "TaylorStepper: order N must be at least 2 for meaningful adaptive control" );

    using T               = typename State::Scalar;
    using Config          = IntegratorConfig< T >;
    using Rhs             = std::function< State( const State&, T ) >;

    static constexpr int D = State::RowsAtCompileTime;  // may be Eigen::Dynamic
    static constexpr bool is_adaptive = true;

    // DenseData: per-step Taylor expansion of x(t) in time around the
    // step start. We store one tax::TE<N> per state component.
    using TE        = tax::TE< N, 1 >;
    using DenseData = Eigen::Matrix< TE, D, 1 >;

    StepResult< State, TaylorStepper > step(
        const Rhs& f, const State& x, T t, T h, const Config& cfg );

    [[nodiscard]] static State eval_dense(
        const DenseData& d, const T& t0, const T& /*t1*/, const T& tq );

private:
    Controller controller_{};
};

// -------- eval_dense --------
// x(tq) = sum_k d_i.coeff(k) * (tq - t0)^k
template < int N, class State, class Controller >
State TaylorStepper< N, State, Controller >::eval_dense(
    const DenseData& d, const T& t0, const T& /*t1*/, const T& tq )
{
    const T dt = tq - t0;
    const Eigen::Index dim = d.size();
    State out{ dim };
    for ( Eigen::Index i = 0; i < dim; ++i )
    {
        // Horner-style evaluation of d(i) at dt.
        T acc = d( i )[ static_cast< std::size_t >( N ) ];
        for ( int k = N - 1; k >= 0; --k )
            acc = acc * dt + d( i )[ static_cast< std::size_t >( k ) ];
        out( i ) = acc;
    }
    return out;
}

// -------- step (real implementation lands in Task 6) --------
template < int N, class State, class Controller >
StepResult< State, TaylorStepper< N, State, Controller > >
TaylorStepper< N, State, Controller >::step(
    const Rhs& /*f*/, const State& x, T /*t*/, T h, const Config& /*cfg*/ )
{
    StepResult< State, TaylorStepper > r;
    const Eigen::Index dim = x.size();
    r.dense.resize( dim );
    // Stub: every coefficient is the value (placeholder so the
    // concept is satisfied and tests compile in Task 6).
    for ( Eigen::Index i = 0; i < dim; ++i )
    {
        r.dense( i )[ 0 ] = x( i );
        for ( int k = 1; k <= N; ++k )
            r.dense( i )[ static_cast< std::size_t >( k ) ] = T{ 0 };
    }
    r.x_new   = x;
    r.h_used  = h;
    r.h_next  = h;
    r.err_norm = T{ 0 };
    r.accepted = true;
    return r;
}

}  // namespace tax::ode
