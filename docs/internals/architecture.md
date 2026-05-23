# Architecture

The library is layered so each concern stays focused on one thing: storage
manages bytes, kernels manage math, operators manage syntax, the Eigen layer
manages linear-algebra interop, and the ODE module assembles the rest into
adaptive solvers.

```
                ┌─────────────────────────────────────────┐
ode             │ Integrator<Stepper, F, Dense>           │  policy-based driver
                │ Stepper × Controller × Event<Stepper>   │
                └────────────────┬────────────────────────┘
                                 │ (consumes TE for the Taylor stepper)
                ┌────────────────▼────────────────────────┐
eigen           │ NumTraits + variables/value/eval/       │  Eigen interop
                │ derivative/gradient/jacobian/invert     │
                └────────────────┬────────────────────────┘
                                 │
                ┌────────────────▼────────────────────────┐
operators       │ +, −, ·, /, sin, exp, log, sqrt, …      │  user-facing surface
                └────────────────┬────────────────────────┘
                                 │
                ┌────────────────▼────────────────────────┐
kernels         │ degree-by-degree recurrences            │  computational core
                │ cauchy, algebra, trig, transcendental,  │
                │ sparse_cauchy, sparse_subs              │
                └────────────────┬────────────────────────┘
                                 │
                ┌────────────────▼────────────────────────┐
core            │ TaylorExpansion<T,N,M,Storage>          │  data type
                │ MultiIndex, enumeration, concepts       │
                │ storage::Dense, storage::Sparse         │
                └─────────────────────────────────────────┘
```

---

## Core data type

`tax::TaylorExpansion<T, N, M, Storage>` is partial-specialised on the storage
tag:

- `storage::Dense` keeps a `std::array<T, C(N+M, M)>` — stack-resident, no heap,
  `constexpr`-friendly accessors.
- `storage::Sparse` keeps two parallel sorted vectors of flat-index / value
  pairs. `nnz()` returns the current support size.

Both expose the same public API (`value`, `coeff`, `derivative`, `eval`,
`deriv`, `integ`). `MultiIndex<M>` and the graded-lexicographic flat indexing
in `tax/core/enumeration.hpp` are storage-agnostic.

---

## Kernels

Every mathematical recurrence lives in `tax/kernels/`. A kernel takes raw
coefficient buffers — `T*` for dense, sorted index/value pairs for sparse —
plus the compile-time shape $(N, M)$ and writes directly into the result.
The kernel layer is the one place where the math of
[Mathematical Foundations](../core/math.md) lives in code.

Univariate vs multivariate is dispatched by `if constexpr (M == 1)` — the
univariate path runs scalar loops over flat indices, the multivariate path
routes through `forEachSubIndex<M>(alpha, lo, hi, callback)`. Sparse uses its
own `sparse_cauchy` and `sparse_subs` kernels that exploit the sorted-index
representation for two-pointer merges.

See [Kernels & Recurrences](kernels.md) for the file-by-file map.

### Build-time toggles

| CMake option | What it changes |
|---|---|
| `TAX_USE_UNROLL`  | Switches the Dense `M == 1` Cauchy kernel to a compile-time-unrolled variant — faster for small $N$. |
| `TAX_USE_STENCIL` | For Dense `M ≥ 2`, precomputes the sub-multi-index stencil at compile time and reuses it across every Cauchy call. |

Both default to `ON`. The non-stencil and non-unroll paths remain in the tree
for cross-validation in `tests/kernels/`.

---

## Operators

`tax/operators/` is a thin facade that wraps each kernel in a free function
returning a fresh `TaylorExpansion`:

```cpp
template <typename T, int N, int M>
[[nodiscard]] constexpr
TaylorExpansion<T, N, M> square(const TaylorExpansion<T, N, M>& x) noexcept {
    TaylorExpansion<T, N, M> r;
    detail::kernels::seriesSquare<T, N, M>(r.coefficients(), x.coefficients());
    return r;
}
```

No lazy expression-template layer — the return is materialised at every
operator boundary. RVO and the named-return optimisation keep this as cheap as
the in-place form when the compiler can see the kernel.

Operator overloads cover three cases: TE × TE, TE × scalar, and scalar × TE,
for `+`, `-`, `*`, `/`. Comparison operators compare the constant terms only
and are useful when threading TE values through Eigen factorisations or
control-flow predicates that branch on a representative value.

---

## Eigen integration

`tax/eigen.hpp` does two things:

1. **NumTraits specialisation** so `Eigen::Matrix<TE, R, C>` is a first-class
   Eigen type. Add/mul cost is set to the monomial count so Eigen's
   cache-aware algorithms pick reasonable strategies.
2. **Vocabulary helpers** — `variables`, `value`, `eval`, `derivative`,
   `gradient`, `hessian`, `jacobian`, `invert`. These wrap the underlying
   `TaylorExpansion` accessors and route through Eigen shape templates so they
   work uniformly with static-size, dynamic-size, and mixed expressions.

The integration is symmetric: any Eigen routine that doesn't require sparse
matrix traits accepts `TE` as a scalar; any user-written generic lambda on
Eigen vectors can be re-instantiated on `TE`-valued state without source
changes. The ODE module exploits this when the Taylor stepper composes the
user RHS on TE-valued state to obtain time-Taylor coefficients via
[automatic differentiation](../ode/math.md#picard-iteration-via-automatic-differentiation).

---

## ODE module

`tax/ode/` adds an integrator on top of the core type. The split:

| Header | Concern |
|---|---|
| `config.hpp`, `step_result.hpp`, `solution.hpp` | data types |
| `concepts.hpp` | `Stepper` / `AdaptiveStepper` concepts the driver uses |
| `controllers.hpp` | step-size policies as mutable structs |
| `steppers/*.hpp` | one stepper per file, each parameterised on `<State, Controller>` |
| `event.hpp`, `triggers.hpp`, `actions.hpp` | the Trigger + Action factoring |
| `integrator.hpp` | the one `Integrator<Stepper, F, Dense>` class plus `make*Integrator` factories |
| `detail/*` | shared helpers — Verner/Feagin/Fehlberg tableaus, adaptive_rk_step, cubic-Hermite interp, Brent root finder |

The driver `Integrator<Stepper, F, Dense>::integrate` is templated on `F` so
the user RHS can be a generic lambda that compiles on both
`Eigen::Matrix<T, D, 1>` (RK path) and `Eigen::Matrix<TE<N>, D, 1>` (Taylor
path). The compile-time `Stepper` policy chooses the algorithm; the
compile-time `Dense` bool chooses whether to keep the per-step
continuous-extension payload alive on the `Solution`.

### Event flow

```
Integrator::integrate(...)
   ↓
   Stepper::step → StepResult{x_new, dense, ...}
   ↓
   for each Event<Stepper> e:
        ε = e.test(ctx)              // Trigger → optional<τ>
        if ε.has_value():
             record (ε, e)            // sort multiple triggers by τ
   sort detected events by τ ascending
   for each detected event:
        flow = e.run(ctx, τ, sink)    // Action returns ControlFlow
        if flow == Terminate: break the outer loop
```

`ZeroCrossing` is the only built-in trigger that needs root finding. The
machinery is method-specific — polynomial-Newton on the Taylor path via
`g_poly`, Brent on the RK path via `Stepper::eval_dense` — and both return
`std::optional<T>`.

---

## Why not expression templates?

The library deliberately *doesn't* use lazy expression templates today. The
fixed-shape `std::array` payload and the kernels (which write
coefficient-by-coefficient into a destination buffer) make the eager path
already free of intermediate `TaylorExpansion` allocations under
RVO. Expression templates would add compile-time complexity without a measured
runtime win on the workloads driving the design (ODE integration, polynomial
maps for ADS in Stage 2b). The door is open if a profile justifies it.
