# tax

**tax** is a header-only C++23 library for **Truncated Algebraic eXpansions** — a framework for computing truncated multivariate Taylor polynomials as first-class objects.

Write natural mathematical expressions and tax automatically propagates the full Taylor series, giving you the function value **and** all partial derivatives up to order \(N\) in a single evaluation pass.

> **Stage 1 (current):** static dense + sparse `TaylorExpansion`, Eigen-vocabulary API,
> and the full standard math operator set. Dynamic-shape types, the Taylor ODE integrator,
> Automatic Domain Splitting, Python bindings, and DACE comparison are deferred to later
> stages.

## Features

- **Compile-time shape** — `TaylorExpansion<T, N, M, Storage>` with `Storage = Dense`
  (stack `std::array`) or `Storage = Sparse` (sorted-index map); order \(N\) and variable
  count \(M\) are compile-time integers
- **Convenience aliases** — `TE<N, M=1>` for dense, `STE<N, M=1>` for sparse
- **Lazy expression templates** with automatic sum/product flattening and leaf fast-paths
- **Comprehensive math**: arithmetic, trigonometric, hyperbolic, transcendental, power, and special functions
- **Direct derivative access**: coefficients, partial derivatives, gradient, Jacobian, and Hessian
- **Eigen integration**: `NumTraits` specialisation plus helpers for variables, value extraction, eval, gradient, Jacobian, and Hessian

## Quick Example

```cpp
#include <tax/tax.hpp>
#include <iostream>

int main() {
    // sin(x) expanded at x₀ = 0, up to order 9
    auto x = tax::TE<9>::variable(0.0);
    tax::TE<9> f = tax::sin(x);

    std::cout << f.value()         << "\n";   // sin(0) = 0
    std::cout << f.derivative({1}) << "\n";   // cos(0) = 1
    std::cout << f.eval(0.3)       << "\n";   // ≈ sin(0.3)
}
```

## Modules

| Module | Description |
|--------|-------------|
| [Getting Started](getting_started.md) | Installation and first examples |
| [Core](core/index.md) | Truncated Taylor polynomials, expression templates, and mathematical functions |

## Requirements

- C++23 compiler (GCC 13+, Clang 17+, Apple Clang 16+)
- CMake 3.28+
- Eigen 3.4+

## License

BSD 3-Clause. See [LICENSE](https://github.com/andreapasquale94/tax/blob/main/LICENSE) for details.
