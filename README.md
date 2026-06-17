# tax

[![Tests](https://github.com/andreapasquale94/tax/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/andreapasquale94/tax/actions/workflows/tests.yml)
[![Sanitizers](https://github.com/andreapasquale94/tax/actions/workflows/sanitizers.yml/badge.svg?branch=main)](https://github.com/andreapasquale94/tax/actions/workflows/sanitizers.yml)
[![Docs](https://github.com/andreapasquale94/tax/actions/workflows/docs.yml/badge.svg?branch=main)](https://andreapasquale94.github.io/tax/)
[![codecov](https://codecov.io/gh/andreapasquale94/tax/graph/badge.svg?token=XwO5JOoaz6)](https://codecov.io/gh/andreapasquale94/tax)

**tax** is a header-only C++23 library for **Truncated Algebraic eXpansions** —
truncated multivariate Taylor polynomials as first-class numerical objects.
Write a natural mathematical expression and tax propagates the full Taylor
series through it, yielding the function value **and** every partial derivative
up to order \(N\) in a single evaluation pass.

> :books: **Full documentation:** <https://andreapasquale94.github.io/tax/>

> :warning: **Active development.** APIs and behavior may change between minor
> versions until 1.0.

---

## Features

- **Compile-time fixed shape** — `TaylorExpansion<T, N, M, Storage>` with `N`
  and `M` as compile-time integers and `Storage` as `Dense` (stack
  `std::array`) or `Sparse` (sorted-index map); both share the kernel layer
  and agree numerically.
- **Convenience aliases** — `TE<N, M=1>` for dense, `STE<N, M=1>` for sparse.
- **Comprehensive math** — arithmetic, trigonometric, hyperbolic,
  transcendental, square/cubic root, reciprocal, integer & real powers,
  `atan2`, `erf`.
- **Direct derivative access** — coefficients, partial derivatives at the
  expansion point, full gradient / Hessian / Jacobian.
- **Eigen integration** — `NumTraits` specialisation plus helpers for
  variables, value extraction, evaluation, gradient, Jacobian, Hessian, and
  formal map inversion.
- **Named expansions** — `NamedTaylorExpansion<T, N, Axes...>` attaches
  compile-time *named axes* to an expansion; values over different axis sets
  compose in their union, and `slice`/`deriv`/`integ` are addressed by name.
  The whole API is re-exported under `tax` (`tax::NE`, `tax::variable(s)`).

> :electric_plug: **Plugin:** adaptive ODE integration and Automatic Domain
> Splitting are available as the optional
> [**tax-flow**](https://github.com/andreapasquale94/tax-flow) project, built on
> top of `tax`.

---

## Requirements

- C++23 compiler — GCC 13+, Clang 17+, Apple Clang 16+
- CMake 3.28+
- Eigen 3.4+

---

## Quick start

### Univariate

```cpp
#include <tax/tax.hpp>
#include <iostream>

int main() {
    auto x = tax::TE<9>::variable(0.0);     // x at x₀ = 0, order 9
    tax::TE<9> f = tax::sin(x);

    std::cout << f.value()           << "\n";   // sin(0)  = 0
    std::cout << f.derivative<1>()   << "\n";   // cos(0)  = 1
    std::cout << f.derivative<3>()   << "\n";   // -cos(0) = -1
    std::cout << f.eval({0.3})       << "\n";   // ≈ sin(0.3)
}
```

### Multivariate

```cpp
#include <tax/tax.hpp>

int main() {
    using TE2 = tax::TE<3, 2>;
    auto x = TE2::variable<0>({1.0, 2.0});
    auto y = TE2::variable<1>({1.0, 2.0});
    TE2 f = tax::sin(x + y);

    f.value();              // sin(3)
    f.derivative<1, 0>();   // ∂f/∂x   = cos(3)
    f.derivative<1, 1>();   // ∂²f/∂x∂y = -sin(3)
}
```

### Sparse storage

```cpp
auto x = tax::STE<5>::variable(1.0);     // same API, sparse storage
tax::STE<5> f = tax::exp(x);             // 6 nonzeros, sorted by flat index
```

### Eigen integration

```cpp
using TE2 = tax::TE<3, 2>;
auto x = TE2::variable<0>({1.0, 2.0});
auto y = TE2::variable<1>({1.0, 2.0});
Eigen::Vector2<TE2> F = { tax::sin(x), tax::cos(y) };

auto vals = tax::value(F);              // Eigen::Vector2d of constant terms
auto J    = tax::jacobian(F);           // 2×2 Jacobian at expansion point
```

---

## Build, test, install

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### CMake options

| Option | Default | Description |
|---|:-:|---|
| `TAX_BUILD_UNITTESTS` | `ON`  | Build the unit-test suite |
| `TAX_BUILD_REGRESSIONS` | `OFF` | Build DACE-based regression tests |

The fast Cauchy kernel paths (`TAX_USE_UNROLL` for M=1, `TAX_USE_STENCIL`
for M≥2) are enabled by default in `<tax/kernels/cauchy.hpp>` itself — no
build-system configuration needed. To opt out, pre-define the macro to `0`
identically in **every** translation unit (differing values are an ODR
violation).

### Consume from another project

```bash
cmake --install build --prefix /your/install/prefix
```

```cmake
find_package(tax CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE tax::tax)
```

---

## Documentation

Hosted at <https://andreapasquale94.github.io/tax/>.

| Section | Topic |
|---|---|
| [Getting Started](https://andreapasquale94.github.io/tax/getting_started/) | Install, build, write your first Taylor expansion |
| [Core](https://andreapasquale94.github.io/tax/core/) | `TaylorExpansion` math, API, examples, Dense vs Sparse storage |
| [Eigen Integration](https://andreapasquale94.github.io/tax/eigen/) | `NumTraits`, helpers, map inversion |
| [Internals](https://andreapasquale94.github.io/tax/internals/) | Architecture, kernels, recurrences |

Source for the docs lives in `docs/` and is built with MkDocs Material:

```bash
pip install mkdocs-material pymdown-extensions
mkdocs serve --strict
```

---

## License

[BSD 3-Clause](LICENSE).
