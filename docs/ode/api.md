# ODE API Reference

All names live in `namespace tax::ode` unless noted. The whole surface is
brought in by the umbrella header:

```cpp
#include <tax/ode.hpp>
```

---

## Configuration

```cpp
template <class T = double>
struct IntegratorConfig {
    T   abstol               = T{1e-12};   // absolute local-error tolerance
    T   reltol               = T{1e-12};   // relative local-error tolerance
    T   initial_step         = T{0};       // 0 ⇒ stepper picks an initial guess
    T   min_step             = T{0};       // 0 ⇒ no lower bound (≈ ε·|tmax − t0|)
    T   max_step             = T{0};       // 0 ⇒ tmax − t0
    int max_steps            = 100'000;
    int max_rejects_per_step = 16;
};
```

Validated at `Integrator` construction; invalid values throw
`std::invalid_argument`.

---

## Stepper concepts

```cpp
namespace tax::ode::concepts {

template <class S>
concept Stepper = requires(S s, typename S::Rhs f,
                           typename S::State x, typename S::T t,
                           typename S::T h, const typename S::Config& cfg) {
    typename S::State;
    typename S::T;
    typename S::Config;
    typename S::Rhs;
    typename S::DenseData;

    { s.step(f, x, t, h, cfg) } -> std::same_as<StepResult<typename S::State, S>>;

    { S::eval_dense(std::declval<typename S::DenseData>(),
                    std::declval<typename S::T>(),
                    std::declval<typename S::T>(),
                    std::declval<typename S::T>()) }
      -> std::same_as<typename S::State>;
};

template <class S>
concept AdaptiveStepper = Stepper<S>
    && requires { { S::is_adaptive } -> std::convertible_to<bool>; }
    && S::is_adaptive;

}  // namespace tax::ode::concepts
```

Steppers may also declare `static constexpr bool has_dense_output` to flag
whether `eval_dense` reproduces the method's full propagation order (`true`)
or a cubic-Hermite fallback (`false`).

---

## Step result

```cpp
template <class State, class Stepper>
struct StepResult {
    State                         x_new;
    typename Stepper::T           h_used;
    typename Stepper::DenseData   dense;
    typename Stepper::T           h_next;
    typename Stepper::T           err_norm;
    bool                          accepted;
};
```

The adaptive fields (`h_next`, `err_norm`, `accepted`) are meaningful when
`AdaptiveStepper<Stepper>` holds; the `Integrator` drives the
rejection-and-retry loop from them.

---

## Step-size controllers

`namespace tax::ode::controllers`. Each `next_step` overload returns the
recommended step from the previous step and the latest error norm.

```cpp
template <class T = double>
struct I {
    T safety = 0.9, min_factor = 0.2, max_factor = 5.0;
    [[nodiscard]] T next_step(T h_used, T err_norm, T tol, int p_emb) const noexcept;
};

template <class T = double>
struct PI {
    T safety = 0.9, alpha = 0.7, beta = 0.4, min_factor = 0.2, max_factor = 5.0;
    [[nodiscard]] T next_step(T h_used, T err_norm, T tol, int p_emb) noexcept;
};

template <class T = double>
struct H211b {
    T safety = 0.9, b = 4.0, min_factor = 0.2, max_factor = 5.0;
    [[nodiscard]] T next_step(T h_used, T err_norm, T tol, int p_emb) noexcept;
};

// Taylor-specific (compile-error for non-Taylor steppers)
template <class T = double>
struct JorbaZou {
    T safety = 0.9, min_factor = 0.2, max_factor = 5.0;
    [[nodiscard]] T next_step(T h_used, T c_N_norm, T c_Nm1_norm,
                              T tol, int N_order) const noexcept;
};
```

PI and H211b own a mutable previous-error state; `step()` is therefore
non-`const` on every stepper for uniformity.

---

## Steppers

All steppers share the surface

```cpp
struct Stepper {
    using State, T, Config, Rhs, DenseData;
    static constexpr bool is_adaptive;
    static constexpr bool has_dense_output;
    static constexpr int  order_v;      // method order p
    static constexpr int  order_emb_v;  // embedded estimator order  (RK only)

    StepResult<State, Stepper>
        step(const Rhs& f, const State& x, T t, T h, const Config& cfg);

    static State eval_dense(const DenseData& d, T t0, T t1, T tq);

    static std::optional<T> find_zero(/* g_scalar, g_poly, dense, t0, h_used,
                                          direction, tol */);
};
```

### Taylor method

```cpp
template <int N,
          class StateT,
          class Controller = controllers::JorbaZou<typename StateT::Scalar>>
struct TaylorStepper;
```

`N ≥ 2`. `DenseData = Eigen::Matrix<TE<N>, D, 1>` — the per-component time
expansion that drives both Horner-evaluated dense output and the
polynomial-Newton root finder.

### Runge–Kutta methods

```cpp
template <class StateT, class Controller = controllers::PI<typename StateT::Scalar>>
struct Verner78Stepper;     // 8(7), 13 stages

template <class StateT, class Controller = controllers::PI<typename StateT::Scalar>>
struct Verner89Stepper;     // 9(8), 16 stages

template <class StateT, class Controller = controllers::PI<typename StateT::Scalar>>
struct Fehlberg78Stepper;   // 7(8), 13 stages

template <class StateT, class Controller = controllers::PI<typename StateT::Scalar>>
struct Feagin12Stepper;     // 12(10), 25 stages

template <class StateT, class Controller = controllers::PI<typename StateT::Scalar>>
struct Feagin14Stepper;     // 14(12), 35 stages
```

Each `DenseData` carries the boundary states and derivatives required by the
cubic-Hermite continuous extension.

---

## Solution

Partial-specialised on `Dense`:

```cpp
template <class Stepper, class State, bool Dense>
class Solution;

// Dense = false  — step boundaries + events only, no continuous payload.
template <class Stepper, class State>
class Solution<Stepper, State, false> {
public:
    using T = typename Stepper::T;
    std::vector<T>                       t;       // size = nsteps + 1
    std::vector<State>                   x;       // x[i] at t[i]
    std::vector<EventRecord<State, T>>   events;  // monotonic in t_event

    [[nodiscard]] std::size_t size() const noexcept;
};

// Dense = true   — adds per-step continuous-extension data + sol(t).
template <class Stepper, class State>
class Solution<Stepper, State, true> {
public:
    using T         = typename Stepper::T;
    using DenseData = typename Stepper::DenseData;

    std::vector<T>                       t;       // size = nsteps + 1
    std::vector<State>                   x;       // size = nsteps + 1
    std::vector<DenseData>               dense;   // size = nsteps
    std::vector<EventRecord<State, T>>   events;

    [[nodiscard]] std::size_t size() const noexcept;

    // Binary-search to the enclosing interval, then Stepper::eval_dense.
    [[nodiscard]] State operator()(const T& t_query) const;
};

template <class State, class T>
struct EventRecord {
    std::string label;           // "" if anonymous
    T           t_event;
    State       x_event;
};
```

---

## Integrator

```cpp
template <concepts::Stepper Stepper, class F, bool Dense = false>
class Integrator {
public:
    using State     = typename Stepper::State;
    using T         = typename Stepper::T;
    using Config    = typename Stepper::Config;
    using Solution  = tax::ode::Solution<Stepper, State, Dense>;
    using EventList = std::vector<Event<Stepper>>;

    explicit Integrator(F f, Config cfg = {}, EventList events = {});

    [[nodiscard]] Solution integrate(const State& x0, const T& t0, const T& tmax) const;
};
```

`F` is template-deduced so a generic lambda (`[](const auto& x, const auto& t){…}`)
can be reused across the scalar-state RK steppers and the TE-state Taylor stepper.

---

## Factory functions

All factories deduce the user callable `F` and instantiate the right Stepper.

```cpp
// Taylor method — N is mandatory.
template <int N, class T = double, int D = Eigen::Dynamic,
          bool Dense = false, class F>
[[nodiscard]] auto makeTaylorIntegrator(F f, IntegratorConfig<T> cfg = {});

template <int N, class T = double, int D = Eigen::Dynamic,
          bool Dense = false, class F>
[[nodiscard]] auto makeTaylorIntegrator(F f, IntegratorConfig<T> cfg,
                                        std::vector<Event<TaylorStepper<N, Eigen::Matrix<T, D, 1>>>> events);
```

Each Runge–Kutta family has matching factories:

```cpp
makeVerner78Integrator  <T, D, Dense, Controller, F>(f, cfg[, events])
makeVerner89Integrator  <T, D, Dense, Controller, F>(f, cfg[, events])
makeFehlberg78Integrator<T, D, Dense, Controller, F>(f, cfg[, events])
makeFeagin12Integrator  <T, D, Dense, Controller, F>(f, cfg[, events])
makeFeagin14Integrator  <T, D, Dense, Controller, F>(f, cfg[, events])
```

Default `Controller = controllers::PI<T>`. Power users who want a different
controller (or a non-Eigen state) can construct the raw
`Integrator<Stepper, F, Dense>` directly.

---

## Events

The full event surface — `Direction`, `ControlFlow`, `TriggerContext`,
`StepperCtx`, `Event<Stepper>`, the `EveryStep` / `ZeroCrossing` triggers, and
the `Continue` / `Terminate` / `Record` / `Custom` actions — is covered on its
own page: [Events](events.md).
