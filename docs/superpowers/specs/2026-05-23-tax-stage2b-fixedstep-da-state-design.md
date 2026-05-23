# Stage 2b — FixedStep controller + DA-vector state for RK steppers

**Date:** 2026-05-23
**Status:** approved (pending written-spec review)
**Author:** Claude (Opus 4.7, 1M ctx) with Andrea Pasquale

## Summary

Two independent ODE-module additions, both scoped to the five Runge–Kutta
steppers shipped in Stage 2a (Verner 7(8), Verner 8(9), Fehlberg 7(8),
Feagin 12(10), Feagin 14(12)).

**Slice A — FixedStep controller.** A no-op step-size controller that always
returns `h_used` and forces every step to be accepted. Lets users propagate
on a user-prescribed grid without adaptive shrinking.

**Slice B — DA-vector state for RK steppers.** A `VectorOps<S>` trait that
abstracts norm / axpy / scale-assign on the state type, plus an
`adaptive_rk_step` refactor and per-stepper changes that pin all step-size
control quantities to `double`. Together they let the five RK steppers
propagate states of type `Eigen::Matrix<tax::TEn<P,M>, D, 1>` — a vector of
multivariate Taylor polynomials in the initial-condition deviations — using
the same code path that propagates `Eigen::Matrix<double, D, 1>`.

Slice B also collapses the ten `make…Integrator` factories into one type
alias per method (`Verner78<State>`, …) with a defaulted RHS template
parameter.

## Background

Stage 2a (commits `78ad23e..`) shipped a method-agnostic adaptive
`Integrator` driven by a compile-time Stepper policy. Public surface:

- `Integrator<Stepper, F, Dense>` — generic over stepper, RHS callable, dense-output flag.
- `IntegratorConfig<T>` — `abstol`, `reltol`, `initial_step`, `min_step`, `max_step`, `max_steps`, `max_rejects_per_step`.
- Controllers: `I`, `PI` (Gustafsson), `H211b` (Söderlind), `JorbaZou` (Taylor-only).
- Steppers: `TaylorStepper<N, State, Controller>`, `Verner78Stepper`, `Verner89Stepper`, `Fehlberg78Stepper`, `Feagin12Stepper`, `Feagin14Stepper`.
- Each RK stepper funnels through `detail::adaptive_rk_step<Tab>` which computes both the propagation `x_new` and the embedded `y_emb`, plus an `err_norm` reduction.
- Ten convenience factories (`makeVerner78Integrator` + events overload, etc.).

**Key constraints inherited from Stage 2a:**

- The Stepper concept exposes `using T = typename State::Scalar`. For `Eigen::Matrix<double, D, 1>` state, `T = double`; for `Eigen::Matrix<TEn<P,M>, D, 1>` state, `T = TEn<P,M>` — which is wrong for tolerances and step sizes.
- `adaptive_rk_step` accumulates `err_norm` as `T`, requiring `T` to model `operator>`. `TaylorExpansionT` does not.
- The current controllers' `next_step(T err_norm, T tol, …)` signature is fine when `T == double`; under TE state it would break the same way.

A prototype on branch `claude/add-verner-integrators-vEgRF` solved the same
problem with a customization-point trio (`verner_norm`, `verner_axpy`,
`verner_scale_assign`) overloaded by ADL for scalars, `TaylorExpansionT`, and
`Eigen::Matrix`. The same pattern is adopted here, restructured as a trait
(`VectorOps<S>`) so future state types (ADS / LOADS subdomains) can extend
by specialization.

## Goals & non-goals

### Goals

- Add `FixedStep<T>` controller; the five RK steppers must honour it via a compile-time `if constexpr` branch alongside the existing `JorbaZou` special case.
- Introduce `VectorOps<S>` trait with specializations for `floating_point`, `TaylorExpansionT<T,N,M>`, and `Eigen::Matrix<T,D,1>` (recursive).
- Refactor `detail::adaptive_rk_step` to use `VectorOps<State>` and return `double err_norm`.
- Refactor the five RK steppers to pin `Stepper::T = double` and use `VectorOps` for the residual norm and tolerance scaling.
- Replace the ten `make…Integrator` factories with one type alias per method, plus `Taylor<N, State>` for symmetry. Defaulted RHS template parameter keeps the simple form a one-liner while preserving a zero-overhead path.
- Maintain numerical parity for existing `Eigen::Matrix<double, D, 1>` tests.
- New tests covering FixedStep on each RK family and DA-vector propagation against a finite-difference STM reference.

### Non-goals

- TaylorStepper on TE-vector state (would require nested `TaylorExpansionT<TEn<P,M>, N, 1>`, blocked by the `Scalar = std::floating_point` constraint in `tax/core/concepts.hpp`).
- Relaxing the `Scalar` concept.
- Transcendental-kernel audit for nested-TE composition.
- A DA-aware Taylor integrator (deferred — see *Future work*).
- DA-aware event detection (sign-change bracketing on a DA-valued `g`).
- A general norm-trait API beyond `VectorOps<S>::norm`.
- Float-state support in the RK steppers (no existing usage; explicitly pinned to `double` to simplify).

## Design

### Slice A — `FixedStep` controller

#### A.1 Controller type

New addition in `include/tax/ode/controllers.hpp`:

```cpp
template <class T = double>
struct FixedStep
{
    // Adaptive controllers' signature: (h_used, err_norm, tol, p_emb)
    [[nodiscard]] T next_step(T h_used, T, T, int) const noexcept { return h_used; }

    // JorbaZou's signature: (h_used, c_N_norm, c_Nm1_norm, tol, N_order)
    [[nodiscard]] T next_step(T h_used, T, T, T, int) const noexcept { return h_used; }
};
```

Both overloads make `FixedStep` plug into any stepper that selects its
controller by `is_same_v<Controller, …>`. No per-instance state, no
parameters.

#### A.2 Per-stepper compile-time bypass

Each RK stepper currently sets `accepted = err_norm <= tol`. Adjacent to the
existing JorbaZou branch, add:

```cpp
T    h_next;
bool accepted;
if constexpr (std::is_same_v<Controller, controllers::FixedStep<T>>)
{
    h_next   = h;
    accepted = true;        // always accept; embedded estimator still computed
}
else if constexpr (std::is_same_v<Controller, controllers::JorbaZou<T>>)
{
    h_next   = h;           // RK steppers: JorbaZou no-op fallback (existing behaviour)
    accepted = out.err_norm <= tol;
}
else
{
    h_next   = controller_.next_step(h, out.err_norm, tol, Tab::order_emb);
    accepted = out.err_norm <= tol;
}
```

Same pattern in `TaylorStepper::step` (which has its own JorbaZou and
err_norm paths; FixedStep slots in as one more compile-time case).

#### A.3 Tests

`tests/ode/testFixedStep.cpp` — one test per stepper (6 total: Taylor +
5 RK):

- With `Controller = FixedStep<double>` and `cfg.initial_step = h0`, every reported `r.h_used` equals `h0` and every step is accepted regardless of `abstol`/`reltol`.
- Solution agrees with the equivalent adaptive run to within the stepper's order × `h0^p` (a sanity check that we are actually integrating, not just looping).
- Validation: `cfg.initial_step = 0` with `FixedStep` is allowed (the integrator's existing heuristic kicks in); document this.

### Slice B — `VectorOps<S>` + RK refactor + DA-vector state

#### B.1 `VectorOps<S>` trait

New header `include/tax/ode/vector_ops.hpp`. The primary template is
undefined so unsupported state types produce a clear error.

```cpp
namespace tax::ode {

template <class S> struct VectorOps;   // primary: undefined

// floating-point scalar
template <class T> requires std::is_floating_point_v<T>
struct VectorOps<T> {
    static double norm(T x)                          { return std::abs(double(x)); }
    static void   axpy(T& y, double a, T x)          { y += T(a) * x; }
    static void   scale_assign(T& y, double a, T x)  { y = T(a) * x; }
};

// scalar TaylorExpansionT — sup-norm over coefficients
template <class T, int N, int M>
struct VectorOps<TaylorExpansionT<T,N,M>> {
    using S = TaylorExpansionT<T,N,M>;
    static double norm(const S& x) {
        double n = 0;
        for (std::size_t i = 0; i < S::nCoefficients; ++i)
            n = std::max(n, std::abs(double(x[i])));
        return n;
    }
    static void axpy(S& y, double a, const S& x)         { y = y + T(a) * x; }
    static void scale_assign(S& y, double a, const S& x) { y = T(a) * x; }
};

// Eigen column vector of anything supported above — recurses
template <class T, int D>
struct VectorOps<Eigen::Matrix<T,D,1>> {
    using V     = Eigen::Matrix<T,D,1>;
    using Inner = VectorOps<T>;
    static double norm(const V& x) {
        double n = 0;
        for (Eigen::Index i = 0; i < x.size(); ++i)
            n = std::max(n, Inner::norm(x(i)));
        return n;
    }
    static void axpy(V& y, double a, const V& x) {
        for (Eigen::Index i = 0; i < x.size(); ++i) Inner::axpy(y(i), a, x(i));
    }
    static void scale_assign(V& y, double a, const V& x) {
        if (y.size() != x.size()) y.resize(x.size());
        for (Eigen::Index i = 0; i < x.size(); ++i) Inner::scale_assign(y(i), a, x(i));
    }
};

}  // namespace tax::ode
```

**Norm convention.** Sup over all coefficients — matches the prototype's
`verner_norm` and `infNorm`, and the natural extension of the Hairer-style
Eigen sup-norm.

**Recursion.** The `Eigen::Matrix<T,D,1>` specialization delegates to
`VectorOps<T>`, so `Eigen::Matrix<TaylorExpansionT<double,2,2>, 6, 1>` works
without a dedicated specialization — element-wise calls into the TE
specialization.

**Extending to a new state type.** Add one `VectorOps<MyState>`
specialization providing the three static functions. No stepper changes.

#### B.2 `adaptive_rk_step` refactor

`include/tax/ode/detail/adaptive_rk_step.hpp`. `T` template parameter
dropped from `RKStepOut`; the function body uses `VectorOps` exclusively.

```cpp
template <class State>
struct RKStepOut {
    State  x_new;
    State  y_emb;
    double err_norm;
};

template <class Tab, class F, class State, int NStages>
[[nodiscard]] RKStepOut<State> adaptive_rk_step(
    F&& f, const State& x, double t, double h,
    RKStepData<State, NStages>& work)
{
    using Ops = VectorOps<State>;

    work.k[0] = f(x, t + Tab::c[0] * h);

    std::size_t a_off = 0;
    for (int i = 1; i < NStages; ++i) {
        State y;
        Ops::scale_assign(y, 1.0, x);
        for (int j = 0; j < i; ++j)
            Ops::axpy(y, h * Tab::a[a_off + j], work.k[j]);
        work.k[i] = f(y, t + Tab::c[i] * h);
        a_off += std::size_t(i);
    }

    State x_new, y_emb;
    Ops::scale_assign(x_new, 1.0, x);
    Ops::scale_assign(y_emb, 1.0, x);
    for (int i = 0; i < NStages; ++i) {
        Ops::axpy(x_new, h * Tab::b    [i], work.k[i]);
        Ops::axpy(y_emb, h * Tab::b_emb[i], work.k[i]);
    }

    State diff;
    Ops::scale_assign(diff,  1.0, x_new);
    Ops::axpy        (diff, -1.0, y_emb);
    return { std::move(x_new), std::move(y_emb), Ops::norm(diff) };
}
```

Same control flow as today; the only behavioural change is that scalar
arithmetic on `State` goes through `Ops` instead of direct Eigen
expressions, and `err_norm` is always `double`.

#### B.3 RK stepper changes (×5)

Affected: `verner78.hpp`, `verner89.hpp`, `fehlberg78.hpp`, `feagin12.hpp`,
`feagin14.hpp`. Each receives the same edits:

```cpp
// Before:
using T      = typename State::Scalar;
using Config = IntegratorConfig<T>;

// After:
using T      = double;                                    // step-control scalar
using Config = IntegratorConfig<double>;
```

The dense-output `DenseData` payload (boundary samples `x0, x1` plus
derivatives `f0, f1`) stays `State`-valued — Hermite-cubic interpolation
already broadcasts scalar `dt` onto whatever State is, and `VectorOps` is
not needed there.

Inside `step()`:

```cpp
// Before:
T x_norm{0};
for (Eigen::Index i = 0; i < x.size(); ++i) {
    using std::abs;
    const T a = T(abs(out.x_new(i)));
    if (a > x_norm) x_norm = a;
}
const T tol = cfg.abstol + cfg.reltol * x_norm;

// After:
const double x_norm = VectorOps<State>::norm(out.x_new);
const double tol    = cfg.abstol + cfg.reltol * x_norm;
```

The `accepted` / `h_next` logic gains the FixedStep arm described in Slice
A but is otherwise unchanged.

#### B.4 Type-alias API

Replaces the ten `make…Integrator` factories in `integrator.hpp`. One alias
per method, defaulted RHS template parameter:

```cpp
template <class State,
          class Controller = controllers::PI<double>,
          bool Dense       = false,
          class F          = typename Verner78Stepper<State, Controller>::Rhs>
using Verner78 = Integrator<Verner78Stepper<State, Controller>, F, Dense>;

template <class State,
          class Controller = controllers::PI<double>,
          bool Dense       = false,
          class F          = typename Verner89Stepper<State, Controller>::Rhs>
using Verner89 = Integrator<Verner89Stepper<State, Controller>, F, Dense>;

// Fehlberg78, Feagin12, Feagin14: same shape.

template <int N, class State,
          class Controller = controllers::JorbaZou<double>,
          bool Dense       = false,
          class F          = typename TaylorStepper<N, State, Controller>::Rhs>
using Taylor = Integrator<TaylorStepper<N, State, Controller>, F, Dense>;
```

`Stepper::Rhs` resolves to `std::function<State(const State&, double)>`,
which is one vtable indirection per RHS call. Users who care can pin `F`
to a concrete lambda type and avoid the indirection.

**Use sites:**

```cpp
// Default — std::function indirection, simplest spelling
Verner78<Eigen::Matrix<double, 6, 1>>        integ{f, cfg};
Verner78<Eigen::Matrix<TEn<2,2>, 6, 1>>      integ_da{f, cfg};
Taylor<25, Eigen::Matrix<double, 6, 1>>      integ_taylor{f, cfg};

// FixedStep via Controller slot
Verner78<Eigen::Matrix<double, 6, 1>,
         controllers::FixedStep<double>>     integ_fs{f, cfg};

// With events
Verner78<...> integ{f, cfg, events};

// Zero-overhead — spell F explicitly
auto rhs = [](const auto& x, double t) { /*...*/ return x; };
Verner78<Eigen::Matrix<double, 6, 1>,
         controllers::PI<double>,
         /*Dense=*/false,
         decltype(rhs)>                       integ_fast{rhs, cfg};
```

**Why not full CTAD.** Alias templates cannot carry deduction guides, and
`State` is not deducible from a generic lambda's signature. `<State>` must
be explicit; that is the irreducible bit of typing.

#### B.5 Tests

- `tests/ode/testVectorOps.cpp` — round-trip the trio (`norm`, `axpy`, `scale_assign`) on scalar, `TEn<2,2>`, `Eigen::Matrix<double, 3, 1>`, and `Eigen::Matrix<TEn<2,2>, 3, 1>`. Verifies recursion.
- `tests/ode/testRKWithDaState.cpp` — propagate planar Kepler with `State = Eigen::Matrix<TEn<2,2>, 4, 1>`, IC = `box.center + halfWidth ⊙ δ`:
  - Constant term (`δ = 0`) matches the existing double-state Kepler propagation to within 1e-10 relative error after one orbit.
  - Linear DA coefficients (∂x(T)/∂x(0)) agree with a finite-difference STM (eight one-sided perturbed runs at `eps = 1e-6`) to within 1e-6 absolute error.
  - Parameterized over the five RK families with a single GoogleTest typed-test fixture.
- Existing `testVerner78Stepper.cpp`, `testVerner89Stepper.cpp`, `testFehlberg78Stepper.cpp`, `testFeagin12Stepper.cpp`, `testFeagin14Stepper.cpp` keep passing unchanged — same path with `Eigen::Matrix<double, D, 1>`.
- All test files using `makeVerner78Integrator(...)` etc. update to `Verner78<State>{f, cfg}`. Estimated ~20–30 call-site changes total.

#### B.6 Migration impact

Public API churn:

- **Removed:** twelve factory overloads in `integrator.hpp` — five RK methods × two overloads each (`cfg`-only and `cfg + events`), plus the two `makeTaylorIntegrator` overloads.
- **Added:** six type aliases (`Verner78`, `Verner89`, `Fehlberg78`, `Feagin12`, `Feagin14`, `Taylor`). The underlying `Integrator<Stepper, F, Dense>` class template stays as the escape hatch for users who want to specify `F` explicitly without writing the alias's template-parameter list in full.
- **Behaviour change:** `Stepper::T` is now `double` for every RK stepper (was `typename State::Scalar`, which equalled `double` in all existing uses). No current test relies on `Stepper::T` being anything other than `double`.

Doc churn:

- `docs/ode/api.md` table of factories → table of aliases.
- New section in `docs/ode/methods.md` covering FixedStep semantics and the DA-vector use case.
- New page `docs/ode/da-vector-state.md` (or sub-section) with the planar-Kepler DA example.

#### B.7 What we are not doing

- Nested TE: `TaylorExpansionT<TEn<P,M>, N, 1>`. Blocked by `requires Scalar<T>` on `TaylorExpansionT` (`tax/core/taylor_expansion.hpp:35`) and `Scalar = std::floating_point` (`tax/core/concepts.hpp:11`). Required for a DA-Taylor stepper; see *Future work*.
- TaylorStepper with TE state.
- DA-aware events. Stage 2a's event API takes a real-valued `g(x, t)`; for DA state we would need to either project (`g.value()`) or generalise to a polynomial-valued `g` with sign-change bracketing on a representative coefficient. Defer until a concrete need emerges.
- A `RealOf<T>` trait (we hardcode `double` everywhere in the step-control path; float-state is not supported and was not in Stage 2a either).

## Risks & open questions

- **TE-coefficient allocation.** `VectorOps<TaylorExpansionT<...>>::axpy` is `y = y + T(a) * x` — for static-storage TE this allocates a stack `std::array`, no heap; for `DynTE` it heap-allocates per call. The prototype shipped the same shape; if benchmarks reveal hot-path overhead, switch to in-place coefficient loops in a follow-up.
- **Eigen NumTraits for TE.** `include/tax/eigen.hpp` already specializes `Eigen::NumTraits<TaylorExpansion<T,N,M,Storage>>`, so `Eigen::Matrix<TEn<P,M>, D, 1>` is valid infrastructure. The Slice B tests will catch any operator-resolution surprise.
- **Type-alias diagnostic clarity.** Errors instantiating `Verner78<UnsupportedState>` will surface deep inside `VectorOps`. The undefined primary template gives a clear `incomplete type` error; documenting "to support a custom state, specialize `VectorOps<MyState>`" in the header comment is sufficient.
- **TaylorStepper FixedStep.** TaylorStepper has its own JorbaZou-special branch; FixedStep slots in as an additional `if constexpr` arm. No structural changes.

## Future work

A separate spec covers porting the full DA Taylor integrator from
`origin/claude/add-verner-integrators-vEgRF` (file
`include/tax/ode/da_integrator.hpp`) into Stage 2a-style architecture. That
work depends on:

1. Relaxing `Scalar = std::floating_point` to a broader concept that admits
   `TaylorPolynomial`s as coefficients.
2. Auditing the transcendental kernels (`sin`, `exp`, `sqrt`, …) for
   generic-coefficient composition.
3. Porting `DaIntegrator<N, P, D, Q>` with its `FlowMap`, `makeDaState`,
   `makeDaParams`, `stepDa` machinery.
4. Optionally porting `VernerAdsIntegrator` for subdivision-style ADS on
   top of the DA path.

None of these are Stage 2b scope.

## Implementation slice ordering

Two independent slices, either order is safe:

1. **Slice A — FixedStep** (estimated 1 short PR): controller type + 6 stepper edits + 6 tests.
2. **Slice B — VectorOps + DA-vector state + type-alias API** (estimated 1 medium PR): new `vector_ops.hpp`, `adaptive_rk_step.hpp` refactor, 5 stepper edits, 6 alias declarations, deletion of 10+ factory overloads, 2 new test files, ~25 call-site updates in existing tests, doc updates.

Slice A can ship before B; B does not depend on A but does benefit from
sharing the `if constexpr (FixedStep…)` arm with the existing pattern.
