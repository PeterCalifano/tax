# tax — Stage 2a: flexible ODE integrator (Taylor + Verner 7/8 + Verner 8/9) — Design

**Status:** draft for approval
**Date:** 2026-05-21
**Depends on:** Stage 1 (merged on `main`)

## Goal

Reintroduce ODE integration to `tax`, but on a single method-agnostic
`Integrator` surface that swaps between the Taylor method and the two Verner
Runge–Kutta pairs (7/8 and 8/9) by a compile-time policy. Events are
expressed as `Trigger + Action`, factoring zero-crossing detection from the
post-detection behaviour so the same machinery scales from "record an
apoapsis" to "subdivide an initial-condition box" without re-architecting.

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
| Method dispatch | Compile-time `concepts::Stepper` policy; one `Integrator` class, three Stepper types | Fits `tax`'s header-only, zero-cost-abstraction style. Generic over method without runtime dispatch or virtual calls. |
| Stepper hierarchy | Two-tier: `concepts::Stepper` (minimum) + `concepts::AdaptiveStepper` (refinement with `err_norm`, `h_next`, `accepted`) | One Integrator core drives both. Fixed-step methods can plug in later without API break. Stage 2a ships only adaptive instances. |
| Event abstraction | `Trigger + Action` factoring | Decouples *when* an event fires from *what to do*. Built-in triggers (`ZeroCrossing`, `EveryStep`) cover Stage 2a; `Custom` action is the ADS seam. |
| Dense output | Template bool on Integrator/Solution: `Integrator<Stepper, Dense=false>`; two partial specializations of `Solution` | Discrete mode pays nothing for unused storage; dense mode exposes `sol(t)` uniformly across methods. Stepper always produces `DenseData` per step (needed for event bisection); Solution stores it only when `Dense=true`. |
| State type | `Eigen::Matrix<T, D, 1>` with `D` static or `Eigen::Dynamic` | One code path covers scalar (D=1), small static, and dynamic. Scalar specialisation avoided. |
| Aliases | Parameterise on `<N, T, D, Dense>` for Taylor and `<T, D, Dense>` for Verner | Ergonomic for the common case (`Eigen::Matrix<T, D, 1>`); raw `Integrator<Stepper, Dense>` still available for non-Eigen states later. |
| Direction enum | `Increasing`, `Decreasing`, `Any` | Direct description of the slope of `g(x, t)` at the crossing. |
| Config field names | `initial_step`, `min_step`, `max_step` (not `h0`, `h_min`, `h_max`) | Spelled-out names read better at call sites. |
| Dense evaluation | `static State Stepper::eval_dense(dense, t0, t1, tq)` | Keeps Stepper stateless. `Solution::operator()(t)` doesn't need to hold an instance. |
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

### Steppers

```cpp
template <int N, class State>
struct TaylorStepper {
    using T         = typename State::Scalar;
    static constexpr int D = State::RowsAtCompileTime;     // may be Eigen::Dynamic
    using Config    = IntegratorConfig<T>;
    using Rhs       = std::function<State(const State&, T)>;
    using DenseData = /* per-step Taylor expansion in t (one tax::TE<N> per component) */;

    StepResult<State, TaylorStepper> step(
        const Rhs& f, const State& x, T t, T h, const Config& cfg) const;

    static State eval_dense(const DenseData& d, const T& t0, const T& t1, const T& tq);
};

template <class State> struct Verner78Stepper { /* same shape, no N */ };
template <class State> struct Verner89Stepper { /* same shape, no N */ };
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
  (filtered by `d`), bisect on `τ ∈ [0, h_used]` using
  `Stepper::eval_dense(dense, t_old, t_old + h_used, t_old + τ)` to obtain
  intermediate state samples. Bisection terminates when the τ-bracket is
  within `16 × eps × (1 + |τ_mid|)`. No method-specific event code; the
  same bisection works for Taylor and Verner.
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

// Convenience aliases (assume State = Eigen::Matrix<T, D, 1>)
template <int N, class T = double, int D = Eigen::Dynamic, bool Dense = false>
using TaylorIntegrator   = Integrator<TaylorStepper<N, Eigen::Matrix<T, D, 1>>, Dense>;
template <class T = double, int D = Eigen::Dynamic, bool Dense = false>
using Verner78Integrator = Integrator<Verner78Stepper<Eigen::Matrix<T, D, 1>>, Dense>;
template <class T = double, int D = Eigen::Dynamic, bool Dense = false>
using Verner89Integrator = Integrator<Verner89Stepper<Eigen::Matrix<T, D, 1>>, Dense>;
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
    ├── integrator.hpp               # Integrator + 3 aliases
    ├── taylor_stepper.hpp
    ├── verner78_stepper.hpp
    ├── verner89_stepper.hpp
    └── detail/
        └── verner_tableaus.hpp      # internal Butcher tables
```

`tax/tax.hpp` does **not** include `tax/ode.hpp` — ODE is opt-in via an
explicit `#include <tax/ode.hpp>`. No new CMake flag; the ODE headers are
part of the `tax` INTERFACE target. Tests are added under
`TAX_BUILD_UNITTESTS`.

## Test surface

```
tests/ode/
├── testConfig.cpp                   # IntegratorConfig validation + defaults
├── testTaylorStepper.cpp            # step() + eval_dense agreement on exp, harmonic
├── testVerner78Stepper.cpp          # same RHS set
├── testVerner89Stepper.cpp          # same
├── testIntegratorBasic.cpp          # 3 methods × {D=1, static D=3, dynamic D} × 3 RHS
├── testIntegratorDense.cpp          # Dense=true: sol(t) accuracy within abstol
├── testEventsZeroCrossing.cpp       # 3 methods × Direction × terminal/record
├── testEventsEveryStep.cpp          # EveryStep + Custom; terminate-from-Custom
└── testIntegratorStatic.cpp         # Static D vs Eigen::Dynamic parity (1e-12)
```

Each `Stepper`-level test asserts that `eval_dense` evaluated at the step
endpoints reproduces `x_old` (at `tq = t0`) and `x_new` (at `tq = t1`)
exactly; that's a minimal self-consistency check independent of analytic
truth. Integrator-level tests compare endpoint to closed-form solutions
(exp, harmonic) and cross-method (Lotka–Volterra: Taylor vs Verner78 vs
Verner89 within 1e-10 over a moderate horizon).

## Slice ordering

Each slice is one PR (or one merged commit). Workflow per slice: write
failing tests → implement → tests green → commit.

| # | Slice | Result |
|---|---|---|
| 1 | Config + concepts + step result + solution skeleton | `IntegratorConfig`, `concepts::Stepper` / `AdaptiveStepper`, `StepResult`, both `Solution` specialisations as empty shells; `testConfig.cpp` + a stub stepper to exercise the concept compile-time. |
| 2 | TaylorStepper | `TaylorStepper<N, State>::step` + `eval_dense` + Jorba–Zou step-size control; `testTaylorStepper.cpp` exercises stepper-level behaviour directly (no Integrator yet). |
| 3 | Integrator + 3 RHS smoke | `Integrator<Stepper, Dense>` with the `if constexpr (AdaptiveStepper)` retry loop, both `Dense` modes, but without the event machinery (event list defaults to empty); `testIntegratorBasic.cpp` (Taylor only) + `testIntegratorDense.cpp` (Taylor only); aliases. |
| 4 | Events: triggers + actions + integrator wiring | `Direction`, `ControlFlow`, `TriggerContext`, `Event`, `ZeroCrossing`, `EveryStep`, `Continue`, `Terminate`, `Record`, `Custom`; integrator's `run_events_`; `testEventsZeroCrossing.cpp` + `testEventsEveryStep.cpp` (Taylor only). |
| 5 | Verner 7/8 + Verner 8/9 | `verner_tableaus.hpp`, `Verner78Stepper`, `Verner89Stepper` with PI controller and built-in dense output; per-stepper tests; expand `testIntegratorBasic.cpp` and the events tests to cover all three methods. |
| 6 | Static-vs-dynamic D parity | `testIntegratorStatic.cpp` confirms `Eigen::Matrix<T, 3, 1>` vs `Eigen::VectorXd` agreement; tighten any remaining loose edges. |

Roughly one PR per slice; final slice ends Stage 2a.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Taylor step-size control (Jorba–Zou) underperforms vs Verner on stiff-ish smooth problems | Surface `abstol` / `reltol` symmetrically; document the per-method behaviour. Cross-method tests catch regressions. |
| std::function-erased Triggers/Actions become a measurable cost when events are evaluated thousands of times | Each event check happens once per accepted step. Pre-Stage-1 measurements showed it negligible against a Verner 13-stage RHS sweep. Re-measure in slice 4 if doubts emerge; the type-erased shape can be replaced with a variadic template-parameter pack later without changing user code. |
| Step rejection loop hides infinite oscillation | `Config::max_rejects_per_step` (default 16) hard-caps the retries; throw with diagnostic if exceeded. |
| `eval_dense` outside `[t0, t1]` returns nonsense without warning | Document precondition; Stage 2a tests assert in-range usage. `Solution::operator()` clamps to the solution's `[t0, tf]` and throws on out-of-range. |
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
