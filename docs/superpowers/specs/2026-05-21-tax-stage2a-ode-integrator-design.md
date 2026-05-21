# tax — Stage 2a: flexible ODE integrator (Taylor + Verner 7/8 + Verner 8/9 + Fehlberg 7/8 + Feagin 12/10 + Feagin 14/12) — Design

**Status:** draft for approval
**Date:** 2026-05-21
**Depends on:** Stage 1 (merged on `main`)

## Goal

Reintroduce ODE integration to `tax`, but on a single method-agnostic
`Integrator` surface that swaps between the Taylor method and five explicit
Runge–Kutta pairs (Verner 8(7), Verner 9(8), Fehlberg 7(8), Feagin 12(10),
Feagin 14(12)) by a compile-time policy. Step-size control is itself a
compile-time policy (`controllers::I`, `controllers::PI` *(default)*,
`controllers::H211b`), so users can trade efficiency for robustness without
recompiling the integrator core. Events are expressed as `Trigger + Action`,
factoring zero-crossing detection from the post-detection behaviour so the
same machinery scales from "record an apoapsis" to "subdivide an
initial-condition box" without re-architecting.

Stage 2a is the integrator core only: state is `Eigen::Matrix<T, D, 1>` with
`D` static *or* `Eigen::Dynamic`. DA-state propagation (`State =
Eigen::Matrix<tax::TE<P, M>, D, 1>`) and the ADS subdivision driver are
deferred to Stage 2b, but the seams are deliberately placed so 2b extends
the surface rather than replacing it.

## Out of scope

- **DA-state propagation.** No `Eigen::Matrix<tax::TE<…>, D, 1>` State. The
  Stepper concept does not preclude it; Stage 2b will add concrete steppers
  for DA states and a `DATaylorIntegrator` alias.
- **ADS subdivision driver.** Reintroduction relies on the `EveryStep +
  Custom` event seam plus DA states, so it lands no earlier than Stage 2b.
- **Reverse-time integration** (`tmax < t0`).
- **Fixed-step concrete steppers.** The two-tier concept hierarchy
  (`Stepper` vs `AdaptiveStepper`) keeps fixed-step as a non-breaking
  future addition; no fixed-step instance ships.
- **Stiff solvers, sensitivity analysis, sparse Jacobians, streaming
  output, parameter-aware variants.** Out of `tax`'s 2026 roadmap.

## Decisions log

| Axis | Choice | Rationale |
|---|---|---|
| Scope | Integrator + events only; no DA state, no ADS, but forward-compatible | Smaller surface to spec/plan/ship; ADS naturally plugs into `EveryStep + Custom` once DA states arrive. |
| Method dispatch | Compile-time `concepts::Stepper` policy; one `Integrator` class, six Stepper types | Fits `tax`'s header-only, zero-cost-abstraction style. Generic over method without runtime dispatch or virtual calls. |
| RK methods shipped | Verner 8(7) ("efficient" 13-stage), Verner 9(8) ("efficient" 16-stage), Fehlberg 7(8) (classical Fehlberg 1968, 13-stage), Feagin 12(10) (Feagin 2007, 25-stage), Feagin 14(12) (Feagin 2010, 35-stage) | Verner pairs are the modern default for high-accuracy non-stiff ODEs (efficient stage count, well-behaved embedded estimator). Feagin pairs cover the very-high-order regime for smooth astrodynamics problems where Taylor isn't desired. Fehlberg 7/8 is the classical baseline included for compatibility / cross-validation; its known "Fehlberg coincidence" (embedded estimator zeroes on certain steps) is documented in Risks. |
| Step-size controllers | `controllers::I<T>`, `controllers::PI<T>` (default), `controllers::H211b<T>`; pluggable per-RK-stepper via template parameter | One row of the "what to optimise for" matrix per controller: `I` = robustness baseline (predictable, more rejections); `PI` = Gustafsson PI, modern default (best efficiency on smooth problems); `H211b` = Söderlind digital filter, smoother step-size sequence on bumpy / piecewise-smooth problems. Taylor uses Jorba–Zou intrinsically (not template-parameterised). |
| Aliases | Parameterise on `<N, T, D, Dense>` for Taylor and `<T, D, Dense>` for Verner/Fehlberg | Ergonomic for the common case (`Eigen::Matrix<T, D, 1>`); raw `Integrator<Stepper, Dense>` still available for non-Eigen states later. |
| Stepper hierarchy | Two-tier: `concepts::Stepper` (minimum) + `concepts::AdaptiveStepper` (refinement with `err_norm`, `h_next`, `accepted`) | One Integrator core drives both. Fixed-step methods can plug in later without API break. Stage 2a ships only adaptive instances. |
| Event abstraction | `Trigger + Action` factoring | Decouples *when* an event fires from *what to do*. Built-in triggers (`ZeroCrossing`, `EveryStep`) cover Stage 2a; `Custom` action is the ADS seam. |
| Dense output | Template bool on Integrator/Solution: `Integrator<Stepper, Dense=false>`; two partial specializations of `Solution` | Discrete mode pays nothing for unused storage; dense mode exposes `sol(t)` uniformly across methods. Stepper always produces `DenseData` per step (needed for event bisection); Solution stores it only when `Dense=true`. |
| State type | `Eigen::Matrix<T, D, 1>` with `D` static or `Eigen::Dynamic` | One code path covers scalar (D=1), small static, and dynamic. Scalar specialisation avoided. |
| Direction enum | `Increasing`, `Decreasing`, `Any` | Direct description of the slope of `g(x, t)` at the crossing. |
| Config field names | `initial_step`, `min_step`, `max_step` (not `h0`, `h_min`, `h_max`) | Spelled-out names read better at call sites. |
| Dense evaluation | `static State Stepper::eval_dense(dense, t0, t1, tq)` | Keeps Stepper stateless. `Solution::operator()(t)` doesn't need to hold an instance. |
| Root finding for `ZeroCrossing` | Polynomial-Newton with bisection safeguard for TaylorStepper; Brent's method for all RK steppers | Stepper-specific via `static find_zero` member. TaylorStepper has an analytic polynomial `g_poly` valid on the whole accepted step → safeguarded Newton converges in 3–6 iterations to ULP precision at O(N) flops/iter. RK steppers have only scalar samples via `eval_dense` → Brent's bracketed method (derivative-free, superlinear convergence) is the right scalar default. |
| `ZeroCrossing` user contract | `g` is supplied as a generic lambda `[](const auto& x, const auto& t){…}` so the same `g` instantiates on both scalar state (Verner path) and `tax::TE<N>`-valued state (Taylor polynomial path) | One user-facing signature, two erased forms stored inside `Event`. Non-generic `g`s (e.g. a `std::function<T(const State&, T)>` literal) still work but fall back to Brent on the Taylor path. |
| Source language | C++23, header-only, no new external deps | Same as the rest of `tax`. |

## Public API

All in `namespace tax::ode` unless stated.

### Config

```cpp
template <class T = double>
struct IntegratorConfig {
    T   abstol               = T{1e-12};   // absolute local-error tolerance
    T   reltol               = T{1e-12};   // relative local-error tolerance
    T   initial_step         = T{0};       // 0 ⇒ stepper picks initial guess
    T   min_step             = T{0};       // 0 ⇒ no lower bound (≈ eps × |tmax - t0|)
    T   max_step             = T{0};       // 0 ⇒ tmax - t0
    int max_steps            = 100'000;
    int max_rejects_per_step = 16;
};
```

`IntegratorConfig` is validated at `Integrator` construction; invalid values
throw `std::invalid_argument`.

### Concepts

```cpp
namespace tax::ode::concepts {

// Minimum: take one step at the supplied h. Used by fixed-step methods.
template <class S>
concept Stepper =
    requires(const S s, typename S::Rhs f, typename S::State x,
             typename S::T t, typename S::T h, const typename S::Config& cfg) {
    typename S::State;
    typename S::T;
    typename S::Config;
    typename S::DenseData;
    { s.step(f, x, t, h, cfg) }
        -> std::same_as<StepResult<typename S::State, S>>;
    { S::eval_dense(std::declval<typename S::DenseData>(), t, t, t) }
        -> std::same_as<typename S::State>;
    // ZeroCrossing root-finding hook — Stepper-specific implementation.
    // See "Event root finding" below.
    { S::find_zero(/* g_scalar, g_poly, dense, t0, h_used, direction, tol */) }
        -> std::same_as<std::optional<typename S::T>>;
};

// Refinement: stepper additionally provides embedded error estimate and
// recommended next step. Triggers the rejection-and-retry loop.
template <class S>
concept AdaptiveStepper = Stepper<S> && requires(const StepResult<typename S::State, S>& r) {
    { r.err_norm } -> std::convertible_to<typename S::T>;
    { r.h_next   } -> std::convertible_to<typename S::T>;
    { r.accepted } -> std::convertible_to<bool>;
};

} // namespace tax::ode::concepts
```

### Step result

```cpp
template <class State, class Stepper>
struct StepResult {
    State                          x_new;
    typename Stepper::T            h_used;
    typename Stepper::DenseData    dense;
    // Adaptive-only fields. For non-adaptive steppers these are an
    // [[no_unique_address]] empty type. Settle the exact layout in the plan.
    typename Stepper::T            h_next;
    typename Stepper::T            err_norm;
    bool                           accepted;
};
```

### Step-size controllers

```cpp
namespace tax::ode::controllers {

// I-controller — classic integral. No memory of prior steps.
//   h_new = h_used · safety · (tol / err_norm)^(1/(p+1))
// p is the embedded estimator order, passed to next_step.
template <class T = double>
struct I {
    T safety = T{0.9};
    T min_factor = T{0.2};
    T max_factor = T{5.0};

    T next_step(T h_used, T err_norm, T tol, int p_emb) const noexcept;
};

// PI-controller (Gustafsson) — adds a proportional term from previous error.
//   h_new = h_used · safety · (tol/err)^β · (err_prev/err)^α
// State: err_prev is updated on each call.
template <class T = double>
struct PI {
    T safety = T{0.9};
    T alpha  = T{0.7};   // divided by (p_emb + 1) internally
    T beta   = T{0.4};   // divided by (p_emb + 1) internally
    T min_factor = T{0.2};
    T max_factor = T{5.0};

    T next_step(T h_used, T err_norm, T tol, int p_emb) noexcept;  // mutates err_prev_
private:
    T err_prev_ = T{1};
};

// H211b — Söderlind digital filter; smoother step-size sequence.
// State: err_prev_ and h_prev_ updated each call.
template <class T = double>
struct H211b {
    T safety = T{0.9};
    T b      = T{4};      // smoothing parameter (Söderlind H211b uses b = 4)
    T min_factor = T{0.2};
    T max_factor = T{5.0};

    T next_step(T h_used, T err_norm, T tol, int p_emb) noexcept;
private:
    T err_prev_ = T{1};
    T h_prev_   = T{0};   // 0 ⇒ first call falls back to I-step
};

} // namespace tax::ode::controllers
```

Each RK Stepper owns one controller instance as a member; `step()` is *not*
`const` on RK steppers because the controller's internal state evolves
with each call. TaylorStepper's `step()` remains `const` (Jorba–Zou is
stateless).

### Steppers

```cpp
// Taylor: no Controller template — Jorba–Zou is intrinsic and stateless.
template <int N, class State>
struct TaylorStepper {
    using T         = typename State::Scalar;
    static constexpr int D = State::RowsAtCompileTime;
    using Config    = IntegratorConfig<T>;
    using Rhs       = std::function<State(const State&, T)>;
    using DenseData = /* per-step Taylor expansion in t (one tax::TE<N> per component) */;

    StepResult<State, TaylorStepper> step(
        const Rhs& f, const State& x, T t, T h, const Config& cfg) const;

    static State eval_dense(const DenseData& d, const T& t0, const T& t1, const T& tq);
};

// RK steppers: parameterised on Controller (default PI). step() is non-const.
template <class State, class Controller = controllers::PI<typename State::Scalar>>
struct Verner78Stepper {
    using T = typename State::Scalar;
    using Config = IntegratorConfig<T>;
    using Rhs = std::function<State(const State&, T)>;
    using DenseData = /* 13 stage values + dense-output weights */;

    StepResult<State, Verner78Stepper> step(
        const Rhs& f, const State& x, T t, T h, const Config& cfg);  // non-const

    static State eval_dense(const DenseData&, const T&, const T&, const T&);

private:
    Controller controller_{};
};

template <class State, class Controller = controllers::PI<typename State::Scalar>>
struct Verner89Stepper   { /* same shape, 16 stages */ };

template <class State, class Controller = controllers::PI<typename State::Scalar>>
struct Fehlberg78Stepper { /* same shape, 13 stages */ };

template <class State, class Controller = controllers::PI<typename State::Scalar>>
struct Feagin12Stepper   { /* same shape, 25 stages */ };

template <class State, class Controller = controllers::PI<typename State::Scalar>>
struct Feagin14Stepper   { /* same shape, 35 stages */ };
```

The RHS `f` is `std::function<State(const State&, T)>`. For Stage 2a's
fixed `State = Eigen::Matrix<T, D, 1>`, an ordinary lambda is sufficient.
Writing the lambda *generically*
(`[](const auto& x, const auto& t){ ... }`) is recommended convention but
not required, because Stage 2b will reuse the same user RHS across DA
states (`State = Eigen::Matrix<tax::TE<P, M>, D, 1>`) without source
changes.

### Events

```cpp
enum class Direction  { Increasing, Decreasing, Any };
enum class ControlFlow { Continue, Terminate };

template <class State, class T, class DenseData>
struct TriggerContext {
    const State&     x_old;
    T                t_old;
    const State&     x_new;
    T                h_used;
    const DenseData& dense;
};

// Built-in triggers (Stage 2a):
template <class GFn>      auto ZeroCrossing(GFn g, Direction d = Direction::Any);
                          auto EveryStep();   // fires with tau = h_used

// Built-in actions (Stage 2a):
                          auto Continue();
                          auto Terminate();
                          auto Record(std::string label);
template <class ActionFn> auto Custom(ActionFn);

template <class Stepper>
class Event {
public:
    template <class Trig, class Act>
    Event(Trig trigger, Act action);   // type-erases both
};
```

**Semantics:**

- `ZeroCrossing(g, d)`: at step end, evaluate `g` at the boundaries via
  `g(x_old, t_old)` and `g(x_new, t_old + h_used)`. If the signs differ
  (filtered by `d`), locate the zero of `g` inside `τ ∈ [0, h_used]` by
  calling the Stepper's `find_zero` static (see *Event root finding*
  below). Root finding is method-specific because the available
  information differs: TaylorStepper has the per-component step
  polynomial; Verner steppers only have a dense-output interpolator.
  Both report `std::optional<τ>`; on `nullopt` the event silently
  doesn't fire even though the boundary sign-change check passed (e.g.
  the safeguard refused to converge — diagnosed in `Risks` below).
- `EveryStep()`: fires unconditionally at the step boundary with
  `τ = h_used`. This is the seam through which ADS plugs in later.
- `Record(label)`: appends a `EventRecord{label, t_event, x_event}` to
  `Solution::events` and returns `ControlFlow::Continue`.
- `Custom(fn)`: invokes `fn(ctx, τ_fired, sink) → ControlFlow`. The sink
  exposes a `push(EventRecord)` so custom actions can write to the
  solution's event channel.

**Multiple events per step.** If more than one event's trigger fires during
the same step, the integrator orders detections by `τ_fired` ascending so
the recorded time stream is monotonic. If any action returns
`Terminate`, the loop exits cleanly after the current step.

### Event root finding

Each Stepper provides a static `find_zero` that locates a single root of
`g(x(τ), t0 + τ)` in `τ ∈ [0, h_used]` given a boundary sign change. The
two shipped implementations differ in what information they exploit:

**TaylorStepper — polynomial-Newton with bisection safeguard.** When
`Event::g_poly` is available (i.e., the user wrote `g` generically), the
stepper composes `g_poly_step(τ) = g(eval_state_as_TE(dense, τ), t0 + τ)`
to obtain a `tax::TE<N>` whose coefficients describe `g` on the entire
accepted step. The polynomial is by construction valid on `[0, h_used]`,
so the boundary sign change brackets the root. Algorithm:

1. Compute `s0 = g_poly_step(0)`, `s1 = g_poly_step(h_used)`; apply
   `Direction` filter.
2. Initial bracket `[τ_lo, τ_hi] = [0, h_used]`.
3. Newton step `τ_{k+1} = τ_k - g_poly_step(τ_k) / g_poly_step.deriv()(τ_k)`.
4. If `τ_{k+1} ∉ [τ_lo, τ_hi]` or `|Δτ|` is not at least halving each
   iteration, take one bisection step on the current bracket instead.
5. Tighten the bracket using `τ_{k+1}` and the sign of
   `g_poly_step(τ_{k+1})`.
6. Stop when `|τ_hi - τ_lo| ≤ 16 · ε · (1 + |τ_mid|)`; return the bracket
   midpoint.

Convergence: 3–6 iterations to ULP precision in typical cases;
worst-case bounded by bisection (one bracket halving per safeguard
fallback). Cost: O(N) per iteration. Fallback when `g_poly` is
unavailable (non-generic `g`): treat as the Verner case and use Brent
on scalar samples.

**All RK steppers (Verner 8(7), Verner 9(8), Fehlberg 7(8), Feagin 12(10),
Feagin 14(12)) — Brent's method on scalar samples.** No polynomial
available; sample `g(eval_dense(dense, t0, t0+h_used, t0+τ), t0+τ)`
and use Brent's algorithm (Dekker–Brent: inverse quadratic
interpolation with bisection fallback) on the bracketed sign change.
Same termination criterion as above. Derivative-free, superlinear
convergence, robust on non-monotonic `g`. Implemented once as a shared
helper and reused by every RK stepper's `find_zero`.

Both methods return `std::optional<τ>`; `nullopt` indicates the
safeguard couldn't converge (e.g. interval became degenerate due to
flat-but-not-zero `g`). The integrator logs and skips silently.

Both methods are **bracketed-single-root**: they find the *earliest*
zero between the step boundaries when the boundary signs disagree, and
report nothing when they agree. If the user's `g` has multiple
zeros inside a single accepted step (large `max_step` vs. fast-varying
`g`), the inner zeros are silently missed. Multi-root detection (e.g.
companion-matrix all-roots extraction on `g_poly` for TaylorStepper) is
a future opt-in; for Stage 2a, the documented mitigation is to set
`Config::max_step` consistently with the expected frequency of `g`.

### Solution

`Solution` is partial-specialised on `Dense` so the discrete mode pays
nothing for unused storage:

```cpp
template <class State, class T>
struct EventRecord {
    std::string label;       // "" if anonymous (Custom action without label)
    T           t_event;
    State       x_event;
};

template <class Stepper, class State, bool Dense>
class Solution;

// Discrete (Dense = false): step boundaries + events. No continuous extension.
template <class Stepper, class State>
class Solution<Stepper, State, false> {
public:
    using T = typename Stepper::T;
    std::vector<T>                       t;        // step times; t.front() == t0
    std::vector<State>                   x;        // x[i] at t[i]
    std::vector<EventRecord<State, T>>   events;   // monotonic in t_event

    std::size_t size() const noexcept { return t.size(); }
};

// Dense (Dense = true): adds per-step continuous-extension payload + sol(t).
template <class Stepper, class State>
class Solution<Stepper, State, true> {
public:
    using T = typename Stepper::T;
    using DenseData = typename Stepper::DenseData;

    std::vector<T>                       t;        // size = nsteps + 1
    std::vector<State>                   x;        // size = nsteps + 1
    std::vector<DenseData>               dense;    // size = nsteps; dense[i] covers (t[i], t[i+1])
    std::vector<EventRecord<State, T>>   events;

    [[nodiscard]] State operator()(const T& t_query) const;  // binary search → eval_dense
};
```

### Integrator

```cpp
template <concepts::Stepper Stepper, bool Dense = false>
class Integrator {
public:
    using State    = typename Stepper::State;
    using T        = typename Stepper::T;
    using Rhs      = typename Stepper::Rhs;
    using Config   = typename Stepper::Config;
    using Sol      = Solution<Stepper, State, Dense>;
    using EvList   = std::vector<Event<Stepper>>;

    explicit Integrator(Rhs f, Config cfg = {}, EvList events = {});

    [[nodiscard]] Sol integrate(const State& x0, const T& t0, const T& tmax) const;
};

// Convenience aliases (assume State = Eigen::Matrix<T, D, 1>; default
// Controller = PI for RK steppers — power users who want I or H211b
// instantiate the raw Integrator<Stepper, Dense> form directly).
template <int N, class T = double, int D = Eigen::Dynamic, bool Dense = false>
using TaylorIntegrator     = Integrator<TaylorStepper<N, Eigen::Matrix<T, D, 1>>, Dense>;
template <class T = double, int D = Eigen::Dynamic, bool Dense = false>
using Verner78Integrator   = Integrator<Verner78Stepper<Eigen::Matrix<T, D, 1>>, Dense>;
template <class T = double, int D = Eigen::Dynamic, bool Dense = false>
using Verner89Integrator   = Integrator<Verner89Stepper<Eigen::Matrix<T, D, 1>>, Dense>;
template <class T = double, int D = Eigen::Dynamic, bool Dense = false>
using Fehlberg78Integrator = Integrator<Fehlberg78Stepper<Eigen::Matrix<T, D, 1>>, Dense>;
template <class T = double, int D = Eigen::Dynamic, bool Dense = false>
using Feagin12Integrator   = Integrator<Feagin12Stepper<Eigen::Matrix<T, D, 1>>, Dense>;
template <class T = double, int D = Eigen::Dynamic, bool Dense = false>
using Feagin14Integrator   = Integrator<Feagin14Stepper<Eigen::Matrix<T, D, 1>>, Dense>;
```

### Integrator core loop

```cpp
Sol integ.integrate(x0, t0, tmax) const {
    // ... validate; initial h from cfg.initial_step or stepper guess
    Sol sol{};
    State x = x0; T t = t0; T h = /* initial */;

    while (t < tmax) {
        auto r = stepper_.step(f_, x, t, h, cfg_);

        if constexpr (concepts::AdaptiveStepper<Stepper>) {
            if (!r.accepted) {
                h = std::max(r.h_next, cfg_.min_step_or_eps());
                if (++rejects > cfg_.max_rejects_per_step) throw;
                continue;
            }
        }

        run_events_(r, x, t, sol);          // may push EventRecord; may set terminate flag
        sol.push_step(t + r.h_used, r.x_new, r.dense /* if Dense */);
        x = r.x_new; t += r.h_used;
        if constexpr (concepts::AdaptiveStepper<Stepper>) h = r.h_next;
        if (terminate_) break;
    }
    return sol;
}
```

`run_events_` evaluates each event's Trigger; collects `(τ_fired, action)`
pairs that report a hit; sorts by `τ_fired`; runs each action in order
through the sink. Cost: one std::function call per event per accepted
step (and per rejected step for adaptive steppers — but rejected steps
don't fire events; only accepted steps are evaluated).

## File layout

```
include/tax/
├── ode.hpp                          # umbrella; users include only this
└── ode/
    ├── config.hpp
    ├── concepts.hpp
    ├── step_result.hpp
    ├── solution.hpp                 # Solution<…, false>, Solution<…, true>, EventRecord
    ├── event.hpp                    # Event, TriggerContext, ControlFlow, Direction
    ├── triggers.hpp                 # ZeroCrossing, EveryStep + bisection helper
    ├── actions.hpp                  # Continue, Terminate, Record, Custom
    ├── integrator.hpp               # Integrator + 6 aliases
    ├── controllers.hpp              # controllers::I, controllers::PI, controllers::H211b
    ├── taylor_stepper.hpp
    ├── verner78_stepper.hpp
    ├── verner89_stepper.hpp
    ├── fehlberg78_stepper.hpp
    ├── feagin12_stepper.hpp
    ├── feagin14_stepper.hpp
    └── detail/
        ├── verner_tableaus.hpp      # Verner 8(7) + 9(8) Butcher tables
        ├── fehlberg_tableaus.hpp    # Fehlberg 7(8) Butcher table
        ├── feagin_tableaus.hpp      # Feagin 12(10) + 14(12) Butcher tables
        └── brent_root.hpp           # shared Brent root-finder for RK steppers' find_zero
```

`tax/tax.hpp` does **not** include `tax/ode.hpp` — ODE is opt-in via an
explicit `#include <tax/ode.hpp>`. No new CMake flag; the ODE headers are
part of the `tax` INTERFACE target. Tests are added under
`TAX_BUILD_UNITTESTS`.

## Test surface

```
tests/ode/
├── testConfig.cpp                       # IntegratorConfig validation + defaults
├── testControllers.cpp                  # I, PI, H211b: smoke + state-evolution invariants
├── testTaylorStepper.cpp                # step() + eval_dense agreement on exp, harmonic
├── testVerner78Stepper.cpp              # same RHS set; sweep all 3 controllers
├── testVerner89Stepper.cpp              # same
├── testFehlberg78Stepper.cpp            # same
├── testFeagin12Stepper.cpp              # same
├── testFeagin14Stepper.cpp              # same
├── testIntegratorBasic.cpp              # 6 methods × {D=1, static D=3, dynamic D} × 3 RHS
├── testIntegratorDense.cpp              # Dense=true: sol(t) accuracy within abstol
├── testEventsZeroCrossing.cpp           # 6 methods × Direction × terminal/record
├── testEventsEveryStep.cpp              # EveryStep + Custom; terminate-from-Custom
├── testIntegratorStatic.cpp             # Static D vs Eigen::Dynamic parity (1e-12)
├── testTwoBodyKepler.cpp                # planar Kepler, e=0.5 — energy / L / closure
├── testCR3BPPropagation.cpp             # CR3BP correctness: Jacobi-C drift, no NaN, expected qualitative trajectory
└── testCR3BPEvents.cpp                  # CR3BP with L1 / Moon-periapsis / L2 events
```

**Stepper-level tests** each assert that `eval_dense` evaluated at the
step endpoints reproduces `x_old` (at `tq = t0`) and `x_new` (at
`tq = t1`) exactly; that's a minimal self-consistency check independent
of analytic truth. Each RK stepper test runs the same RHS set under all
three controllers and asserts equivalent endpoint accuracy
(controllers should only affect step-count and rejection behaviour, not
correctness).

**Integrator-level tests** compare endpoint to closed-form solutions
(exp, harmonic) and cross-method (Lotka–Volterra: all six methods
within method-order-scaled tolerances over a moderate horizon —
Fehlberg gets a slightly looser tolerance because of the documented
embedded-estimator behaviour).

**Physical-dynamics tests** exercise the integrator on real conservative
systems:

- `testTwoBodyKepler.cpp` — Planar Kepler, eccentricity `e = 0.5`.
  State `Eigen::Vector4d{x, y, vx, vy}`; canonical units (`GM = 1`,
  `a = 1`); ICs at periapsis (`r_p = 0.5`, `v_p = √3`). Propagate 10
  periods. Assertions per (method, controller): specific energy
  `E = ½‖v‖² − 1/‖r‖` conserved within method-scaled tolerance; specific
  angular momentum `L = x·vy − y·vx` conserved; closure
  `‖r(10T) − r_periapsis‖` within tolerance. Exercises adaptive
  step-size control near periapsis (where steps must shrink dramatically).

- `testCR3BPPropagation.cpp` — Planar CR3BP, Earth–Moon mass ratio
  `μ = 0.01215`. Propagation-only correctness — no events. ICs chosen
  to produce an Earth → Moon-region transit (specific values pinned in
  the implementation plan). Propagate for `T_final ≈ 7` non-dimensional
  units. Assertions per (method, controller): Jacobi constant
  `C(x, y, vx, vy)` preserved within method-scaled tolerance; final
  state matches a high-precision reference trajectory (produced by
  `Feagin14Integrator` + `controllers::H211b` at `abstol=reltol=1e-14`)
  within a tolerance loosened by method order; integration completes
  without rejection-cap exhaustion.

- `testCR3BPEvents.cpp` — Same RHS and ICs as the propagation test,
  with three `ZeroCrossing` events attached:
    * `g = x − x_L1` (Increasing) → `Record("L1")`
    * `g = (x − 1 + μ)·vx + y·vy` (Increasing) → `Record("moon_periapsis")`
      (sign change of radial velocity in the Moon frame ≡ closest approach)
    * `g = x − x_L2` (Increasing) → `Record("L2")`
  Assertions per method: at least one `L1` event before the lunar loop;
  at least one `moon_periapsis` event with `‖r_moon‖ < 0.1`; at least
  one `L2` event before `T_final`. The Jacobi-constant assertion from
  the propagation test is re-checked to confirm the event machinery
  doesn't perturb the trajectory.

## Benchmarks

The propagation-only CR3BP test doubles as a benchmark fixture under
`benchmarks/` (Google Benchmark, gated by `TAX_BUILD_BENCHMARK=ON`):

```
benchmarks/
└── bench_ode_cr3bp.cpp           # all six methods × three controllers + Taylor;
                                  # reports wall time, accepted-step count, rejection
                                  # count, and end-state error vs reference at fixed
                                  # abstol = reltol = 1e-12.
```

The benchmark uses the same ICs and propagation horizon as
`testCR3BPPropagation.cpp` so the correctness test and the performance
profile measure the same problem. Reported metrics let users pick the
(method, controller) pair appropriate for their application —
typically Verner 9(8) + PI for general use, Feagin 14(12) + H211b for
ultra-high precision, Verner 8(7) + I for the most robust baseline.

## Slice ordering

Each slice is one PR (or one merged commit). Workflow per slice: write
failing tests → implement → tests green → commit.

| # | Slice | Result |
|---|---|---|
| 1 | Config + concepts + step result + solution skeleton | `IntegratorConfig`, `concepts::Stepper` / `AdaptiveStepper`, `StepResult`, both `Solution` specialisations as empty shells; `testConfig.cpp` + a stub stepper to exercise the concept compile-time. |
| 2 | TaylorStepper | `TaylorStepper<N, State>::step` + `eval_dense` + Jorba–Zou step-size control; `testTaylorStepper.cpp` exercises stepper-level behaviour directly (no Integrator yet). |
| 3 | Integrator + 3 RHS smoke | `Integrator<Stepper, Dense>` with the `if constexpr (AdaptiveStepper)` retry loop, both `Dense` modes, no event machinery yet; `testIntegratorBasic.cpp` (Taylor only) + `testIntegratorDense.cpp` (Taylor only); aliases. |
| 4 | Events: triggers + actions + integrator wiring | `Direction`, `ControlFlow`, `TriggerContext`, `Event` (storing both scalar `g` and `g_poly` erased forms), `ZeroCrossing`, `EveryStep`, `Continue`, `Terminate`, `Record`, `Custom`; integrator's `run_events_`; `TaylorStepper::find_zero` (polynomial-Newton with bisection safeguard); shared `detail/brent_root.hpp`; `testEventsZeroCrossing.cpp` + `testEventsEveryStep.cpp` (Taylor only). |
| 5 | Step-size controllers + Verner pairs | `controllers.hpp` (`I`, `PI`, `H211b`); `testControllers.cpp`; `verner_tableaus.hpp`, `Verner78Stepper`, `Verner89Stepper` with the controller as a template parameter and built-in dense output; per-stepper tests sweep all three controllers; expand `testIntegratorBasic.cpp` and the events tests to Taylor + Verner. |
| 6 | Fehlberg 7/8 | `fehlberg_tableaus.hpp`, `Fehlberg78Stepper` with controller template parameter and Hermite-based dense output; per-stepper test; widen `testIntegratorBasic.cpp` and event tests. |
| 7 | Feagin 12(10) + Feagin 14(12) | `feagin_tableaus.hpp`, `Feagin12Stepper`, `Feagin14Stepper`; per-stepper tests; widen `testIntegratorBasic.cpp` and event tests to cover all six methods. |
| 8 | Physical-dynamics correctness | `testTwoBodyKepler.cpp` and `testCR3BPPropagation.cpp` (propagation-only); these sweep all six methods × three controllers (Taylor uses Jorba–Zou). The Feagin14 + H211b @ 1e-14 reference trajectory used by the CR3BP test is generated during the test setup. |
| 9 | CR3BP events + benchmark | `testCR3BPEvents.cpp` reuses the slice-8 RHS and ICs and adds the three `ZeroCrossing` events; `benchmarks/bench_ode_cr3bp.cpp` reports timing + step counts for each (method, controller) on the same problem. |
| 10 | Static-vs-dynamic D parity | `testIntegratorStatic.cpp` confirms `Eigen::Matrix<T, D_static, 1>` vs `Eigen::VectorXd` agreement; tighten any remaining loose edges. |

Roughly one PR per slice; final slice ends Stage 2a.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Taylor step-size control (Jorba–Zou) underperforms vs Verner on stiff-ish smooth problems | Surface `abstol` / `reltol` symmetrically; document the per-method behaviour. Cross-method tests catch regressions. |
| Fehlberg 7/8 "coincidence" — embedded estimator returns zero error on certain steps even when true error is nonzero, causing the controller to over-extend the step | Documented limitation of the method itself, not a `tax` bug. The controller's `min_step` floor and the cross-method endpoint tests catch gross failures. Users who need maximum reliability should prefer `Verner78Integrator` or `Feagin12Integrator` (modern replacements); `Fehlberg78Integrator` is shipped for compatibility and cross-validation. |
| Feagin 14(12)'s 35-stage cost makes it slow on dynamic-dimension state where each stage incurs heap allocation | Documented in the BoM: prefer compile-time `D` for Feagin14 hot paths. The benchmark suite reports per-step wall time so users see the trade-off directly. |
| PI / H211b controllers oscillate or diverge when the embedded error estimator is unreliable (rare but possible on Fehlberg-coincidence-prone steps or near a discontinuity) | All three controllers clamp `h_new / h_used` to `[min_factor, max_factor]` (defaults 0.2, 5.0). `Config::max_rejects_per_step` caps the inner retry budget. Users on bumpy problems should pick H211b explicitly. |
| std::function-erased Triggers/Actions become a measurable cost when events are evaluated thousands of times | Each event check happens once per accepted step. Pre-Stage-1 measurements showed it negligible against a Verner 13-stage RHS sweep. Re-measure in slice 4 if doubts emerge; the type-erased shape can be replaced with a variadic template-parameter pack later without changing user code. |
| Step rejection loop hides infinite oscillation | `Config::max_rejects_per_step` (default 16) hard-caps the retries; throw with diagnostic if exceeded. |
| `eval_dense` outside `[t0, t1]` returns nonsense without warning | Document precondition; Stage 2a tests assert in-range usage. `Solution::operator()` clamps to the solution's `[t0, tf]` and throws on out-of-range. |
| Polynomial-Newton diverges on flat/pathological `g_poly` (e.g. multiple very close roots, near-zero derivative) | Bisection safeguard caps fallback at one bracket-halving per safeguarded iteration; tolerance criterion uses both interval width and ULP scale; `find_zero` returns `std::nullopt` on safeguard exhaustion rather than wrong answers. |
| Multiple zeros of `g` inside a single step go undetected by bracketed root finding | Documented limitation. Mitigation in 2a: set `Config::max_step` consistent with the expected frequency of `g`. Future opt-in: companion-matrix all-roots extraction on `g_poly` for TaylorStepper (~O(N³) per detection — fine for rare event evaluation). |
| Non-generic `g` (e.g. raw `std::function<T(const State&, T)>`) silently falls back to Brent even on TaylorStepper, losing the polynomial advantage | Documented user contract: write `g` as a generic lambda. `Event` constructor detects whether `g` is invocable with TE-valued state at compile time and stores only the supported erased form(s); a `[[nodiscard]] bool uses_polynomial_root_finding() const` accessor lets users audit. |
| Forward-compat with DA states relies on the Stepper concept staying generic | Slice 5 explicitly cross-instantiates `TaylorStepper<N, Eigen::Matrix<double, D, 1>>` to confirm the State trait extraction works for both static and dynamic D. Stage 2b will add a `static_assert`-driven test that the same `TaylorStepper` template also instantiates for a TE-valued State. |

## Open questions deferred to implementation

- Exact `DenseData` payload for `TaylorStepper<N>`. Either (a) a flat
  `std::array<T, (N+1) × D>` of step-local Taylor coefficients in row-major
  layout, or (b) a `std::vector<tax::TE<N>>` of size D. Decide in slice 2
  on a perf-vs-clarity basis.
- Whether `StepResult` carries the adaptive fields unconditionally (with
  trivial values for fixed-step) or via `[[no_unique_address]]` empty-base
  for a non-adaptive variant. Settle in slice 1 once the concepts compile.
- Initial `h` heuristic when `Config::initial_step == 0`. Pre-Stage-1
  Taylor used `h0 = (abstol / |f(x0, t0)|)^(1/N)` clamped to
  `(tmax - t0) / 100`. Verner used a Shampine-style estimator. Mirror those
  in slice 2 (Taylor) and slice 5 (Verner).
- Whether the `Custom` action's sink offers richer typed channels beyond
  `EventRecord`. Stage 2a ships string-labeled records only; revisit in
  Stage 2b when ADS needs to push subdomain-tree updates.

---

## Implementation plan

To be produced by the `writing-plans` skill after this spec is approved.
