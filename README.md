# tax

[![Tests](https://github.com/andreapasquale94/tax/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/andreapasquale94/tax/actions/workflows/tests.yml)
[![Sanitizers](https://github.com/andreapasquale94/tax/actions/workflows/sanitizers.yml/badge.svg?branch=main)](https://github.com/andreapasquale94/tax/actions/workflows/sanitizers.yml)
[![codecov](https://codecov.io/gh/andreapasquale94/tax/graph/badge.svg?token=XwO5JOoaz6)](https://codecov.io/gh/andreapasquale94/tax)

**tax** is a header-only C++23 library for **Truncated Algebraic eXpansions** — a framework for computing truncated multivariate Taylor polynomials as first-class objects.

Write natural mathematical expressions and tax automatically propagates the full Taylor series, giving you the function value and all partial derivatives up to order N in a single evaluation pass.

    DISCLAIMER: this repository is under active development. APIs and behavior may change; use with care.

> **Stage 1 (current):** static dense + sparse `TaylorExpansion`, Eigen-first API, and the
> full standard math operator set. Dynamic-shape types, the Taylor ODE integrator,
> Automatic Domain Splitting, Python bindings, and DACE comparison are deferred to later
> stages. See the spec and implementation plan in `docs/superpowers/specs/` and
> `docs/superpowers/plans/`.

## Features

- **Compile-time fixed shape** — `TaylorExpansion<T, N, M, Storage>` where `N` and `M` are
  compile-time integers and `Storage` is `Dense` (stack `std::array`) or `Sparse`
  (sorted-index map); the two configurations share the same kernel layer and agree
  numerically
- **Convenience aliases** — `TE<N, M=1>` for dense, `STE<N, M=1>` for sparse
- **Lazy expression templates** with automatic sum/product flattening and leaf fast-paths
- **Comprehensive math**: arithmetic, trigonometric, hyperbolic, transcendental, power, and special functions
- **Direct derivative access**: coefficients, partial derivatives, gradient, Jacobian, and Hessian
- **Eigen integration**: `NumTraits` specialisation plus helpers for variables, value extraction, eval, gradient, Jacobian, and Hessian

## Requirements

- C++23 compiler (GCC 13+, Clang 17+, Apple Clang 16+)
- CMake 3.28+
- Eigen 3.4+

## Quick Start

### Univariate

```cpp
#include <tax/tax.hpp>
#include <iostream>

int main() {
    using tax::TE;

    // sin(x) expanded at x₀ = 0, up to order 9
    auto x = TE<9>::variable(0.0);
    TE<9> f = tax::sin(x);

    std::cout << f.value()          << "\n";   // sin(0) = 0
    std::cout << f.derivative({1})  << "\n";   // cos(0) = 1
    std::cout << f.derivative({2})  << "\n";   // -sin(0) = 0
    std::cout << f.eval(0.3)        << "\n";   // ≈ sin(0.3)
}
```

### Multivariate

```cpp
#include <tax/tax.hpp>
#include <iostream>

int main() {
    using tax::TE;

    // f(x, y) = sin(x + y) expanded at (1, 2)
    auto [x, y] = TE<3, 2>::variables({1.0, 2.0});
    TE<3, 2> f = tax::sin(x + y);

    std::cout << f.value()              << "\n";   // sin(3)
    std::cout << f.derivative({1, 0})   << "\n";   // ∂f/∂x = cos(3)
    std::cout << f.derivative({1, 1})   << "\n";   // ∂²f/∂x∂y = -sin(3)
}
```

### Sparse Storage

```cpp
#include <tax/tax.hpp>

int main() {
    using tax::STE;

    // Same API, sparse coefficient storage — efficient when only few monomials are non-zero
    auto x = STE<5>::variable(1.0);
    STE<5> f = tax::exp(x);
}
```

### Eigen Integration

```cpp
#include <tax/tax.hpp>

int main() {
    using tax::TE;

    auto [x, y] = TE<3, 2>::variables({1.0, 2.0});
    Eigen::Vector2<TE<3, 2>> F = {tax::sin(x), tax::cos(y)};

    auto vals = tax::value(F);           // Eigen::Vector2d of constant terms
    auto J    = tax::jacobian(F, 2);     // 2×2 Jacobian at expansion point
}
```

## Build and Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

| Option | Default | Description |
|--------|---------|-------------|
| `TAX_BUILD_TEST` | `ON` | Build the test suite |
| `TAX_BUILD_BENCHMARK` | `OFF` | Build Google Benchmark suite |
| `TAX_USE_UNROLL` | `ON` | Compile-time-unrolled M=1 Cauchy kernels |
| `TAX_USE_STENCIL` | `ON` | Precomputed Cauchy stencils for M≥2 |

## Install

```bash
cmake --install build --prefix /your/install/prefix
```

From another CMake project:

```cmake
find_package(tax CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE tax::tax)
```

If installed to a non-standard prefix:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/your/install/prefix
```

## API at a Glance

```cpp
#include <tax/tax.hpp>
```

### Types

| Type | Description |
|------|-------------|
| `TE<N>` | `TaylorExpansion<double, N, 1, Dense>` |
| `TE<N, M>` | `TaylorExpansion<double, N, M, Dense>` |
| `STE<N>` | `TaylorExpansion<double, N, 1, Sparse>` |
| `STE<N, M>` | `TaylorExpansion<double, N, M, Sparse>` |

### Factories

```cpp
TE<N>::variable(x0)              // univariate variable at x₀
TE<N,M>::variable<I>(x0)        // I-th variable of a multivariate expansion
TE<N,M>::variables(x0)          // all variables (structured bindings)
TaylorExpansion::constant(v) / zero() / one()
```

### Accessors

```cpp
f.value()            // f(x₀)
f.coeff({2, 1})      // coefficient of δx²·δy
f.derivative({2, 1}) // ∂³f/∂x²∂y at x₀
f.derivatives()      // all partial derivatives
f.coeffsNormInf()    // max |coefficient| (L-infinity norm)
f.eval(dx)           // polynomial evaluated at x₀ + δx
```

### Operations

**Arithmetic**: `+`, `-`, `*`, `/` between DA expressions and scalars

**Unary math**: `abs`, `square`, `cube`, `sqrt`, `cbrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `exp`, `log`, `log10`, `erf`

**Binary math**: `pow` (integer, real, DA exponents), `atan2`, `hypot` (2- and 3-argument)

### Eigen Helpers

```cpp
tax::value(container)        // extract constant terms into Eigen vector/matrix
tax::eval(container, dx)     // evaluate at displacement dx
tax::gradient(f, M)          // gradient vector (Eigen::VectorXd)
tax::jacobian(F, M)          // Jacobian matrix (Eigen::MatrixXd)
tax::hessian(f, M)           // Hessian matrix (Eigen::MatrixXd)
```

## Documentation

Full documentation is available at [andreapasquale94.github.io/tax](https://andreapasquale94.github.io/tax).

| Section | Description |
|---------|-------------|
| [Getting Started](docs/getting_started.md) | Installation and first examples |
| [Core](docs/core/index.md) | Taylor expansion type, expression templates, mathematical functions |

## License

See [LICENSE](LICENSE) for details.
