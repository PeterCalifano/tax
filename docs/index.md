---
hide:
  - navigation
---

# tax

**Truncated Algebraic eXpansions** — a header-only C++23 library for computing
truncated multivariate Taylor polynomials as first-class numerical objects.

Write a natural mathematical expression and tax propagates the full Taylor
series through it, yielding the function value **and** every partial derivative
up to order \(N\) in a single evaluation pass.

```cpp
#include <tax/tax.hpp>

auto x = tax::TE<9>::variable(0.0);    // 9-th order univariate variable at x₀ = 0
tax::TE<9> f = tax::sin(x);            // expression-template tree, evaluated on assign

f.value();          // sin(0)     = 0
f.derivative<1>();  // cos(0)     = 1
f.derivative<3>();  // −cos(0)    = −1
f.eval(0.3);        // sin(0.3) within machine precision
```

!!! warning "Active development"
    APIs and behavior may change between minor versions until 1.0. The Stage 2a
    ODE module landed recently; see the [ODE Integrator](ode/index.md) section.

---

## Why tax?

<div class="grid cards" markdown>

- :material-function-variant:{ .lg .middle } **Fixed-shape, allocation-free**

    Coefficient storage is `std::array<T, C(N+M, M)>` on the stack. Both `N` and
    `M` are compile-time integers, so the optimizer sees through every loop.

- :material-lightning-bolt:{ .lg .middle } **Lazy expression templates**

    `sin(x*y + z)` builds a tree of expressions; the polynomial is materialized
    only on assignment. Sums and products are flattened to N-ary nodes for a
    single pass.

- :material-tune-vertical:{ .lg .middle } **Dense or sparse storage**

    `TE<N, M>` (dense, `std::array`) for the hot path; `STE<N, M>` (sparse,
    sorted-index map) when only a handful of monomials are non-zero. The two
    share the kernel layer and agree numerically.

- :material-orbit:{ .lg .middle } **Eigen-native**

    A `NumTraits` specialisation lets `TaylorExpansion` live inside Eigen
    vectors and matrices; helpers extract values, gradients, Jacobians, and
    Hessians.

- :material-rocket:{ .lg .middle } **Adaptive ODE integration**

    `tax::ode::Integrator<Stepper>` swaps between a high-order Taylor method
    and five Runge–Kutta pairs (Verner 8(7), Verner 9(8), Fehlberg 7(8),
    Feagin 12(10), Feagin 14(12)) by compile-time policy.

- :material-bell-ring:{ .lg .middle } **Event detection**

    Zero-crossing events with direction filters, polynomial-Newton root
    finding on the Taylor path, Brent on the RK path. Custom user actions for
    integration into larger workflows.

</div>

---

## At a glance

| What you write | What you get |
|---|---|
| `tax::TE<N>::variable(x0)` | univariate TE at \(x_0\), order \(N\) |
| `tax::TE<N, M>::variable<I>(x0)` | \(I\)-th coordinate variable, others as parameters |
| `tax::variables<TE<N,M>>(x0)` | Eigen column vector of all \(M\) coordinate variables |
| `tax::sin(x) * tax::exp(y)` | lazy expression, materialized on assignment |
| `f.derivative<2, 1>()` | \(\partial^3 f / \partial x^2 \partial y\) at \(x_0\) |
| `f.eval(dx)` | Horner evaluation of the polynomial at \(x_0 + \delta x\) |
| `tax::jacobian(F, M)` | Eigen Jacobian of a vector function |
| `tax::ode::makeTaylorIntegrator<25>(f)` | adaptive Taylor IVP integrator |
| `tax::ode::makeVerner89Integrator(f)` | adaptive Verner 9(8) integrator |

---

## Navigating the docs

<div class="grid cards" markdown>

- [:material-rocket-launch: __Getting Started__](getting_started.md)

    Install, build, and write your first Taylor expansion.

- [:material-function: __Core__](core/index.md)

    The `TaylorExpansion` type, its math, its API, and worked examples.

- [:material-matrix: __Eigen Integration__](eigen/index.md)

    Use `TaylorExpansion` inside Eigen vectors and matrices. Gradients,
    Jacobians, Hessians.

- [:material-chart-line: __ODE Integrator__](ode/index.md)

    Adaptive Taylor + Runge–Kutta IVP solvers with events and dense output.

- [:material-cog: __Internals__](internals/index.md)

    Expression templates, kernels, recurrence relations.

</div>

---

## Requirements

- C++23 compiler — GCC 13+, Clang 17+, Apple Clang 16+
- CMake 3.28+
- Eigen 3.4+ (linked via `find_package(Eigen3)`)

## License

[BSD 3-Clause](https://github.com/andreapasquale94/tax/blob/main/LICENSE).
