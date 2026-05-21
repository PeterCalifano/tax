# Getting Started

## Installation

### Requirements

- C++23 compiler (GCC 13+, Clang 17+, Apple Clang 16+)
- CMake 3.28+
- Eigen 3.4+

### Building from Source

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `TAX_BUILD_UNITTESTS` | `ON` | Build Google Test unit-test suite |
| `TAX_BUILD_BENCHMARK` | `OFF` | Build Google Benchmark suite |
| `TAX_USE_UNROLL` | `ON` | Compile-time-unrolled M=1 Cauchy kernels |
| `TAX_USE_STENCIL` | `ON` | Precomputed Cauchy stencils for M≥2 |

### Using tax in Your Project

```bash
cmake --install build --prefix /your/install/prefix
```

```cmake
find_package(tax CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE tax::tax)
```

## Include

A single umbrella header pulls in everything:

```cpp
#include <tax/tax.hpp>
```

## Core Type

The central type is `TaylorExpansion<T, N, M, Storage>`:

| Parameter | Meaning |
|-----------|---------|
| `T` | Scalar coefficient type (e.g. `double`) |
| `N` | Maximum total polynomial order (compile-time integer) |
| `M` | Number of independent variables (compile-time integer, default `1`) |
| `Storage` | `tax::Dense` (default) or `tax::Sparse` |

### Convenience Aliases

```cpp
tax::TE<N>         // dense univariate  — TaylorExpansion<double, N, 1, Dense>
tax::TE<N, M>      // dense multivariate
tax::STE<N>        // sparse univariate — TaylorExpansion<double, N, 1, Sparse>
tax::STE<N, M>     // sparse multivariate
```

## Creating Variables

=== "Univariate (dense)"

    ```cpp
    auto x = tax::TE<5>::variable(1.0);   // x = 1 + δx
    ```

=== "Multivariate (dense)"

    ```cpp
    auto [x, y] = tax::TE<3, 2>::variables({1.0, 2.0});
    ```

=== "Single indexed"

    ```cpp
    auto z = tax::TE<2, 3>::variable<2>({1.0, 2.0, 3.0});
    ```

=== "Sparse"

    ```cpp
    auto x = tax::STE<5>::variable(1.0);  // same factories, sparse storage
    ```

## Building Expressions

Arithmetic and math functions work naturally on both dense and sparse types:

```cpp
auto x = tax::TE<6>::variable(0.0);
tax::TE<6> f = tax::sin(x) + tax::square(x) / 2.0;
```

The right-hand side builds a lazy expression tree. Evaluation happens only on
assignment to a `TaylorExpansion` object.

## Extracting Results

```cpp
f.value();              // f(x₀) — the constant term
f.coeff({k});           // coefficient of δxᵏ
f.derivative({k});      // k-th derivative (= k! · coeff)
f.eval(0.3);            // evaluate polynomial at x₀ + 0.3
```

For multivariate objects:

```cpp
auto [x, y] = tax::TE<3, 2>::variables({0.0, 0.0});
tax::TE<3, 2> g = x*x + x*y + y*y;

g.derivative({2, 0});   // ∂²g/∂x²   = 2
g.derivative({1, 1});   // ∂²g/∂x∂y  = 1
g.coeff<1, 1>();         // compile-time index access
```

## Coefficients vs Derivatives

A Taylor expansion stores **coefficients** of the monomial basis:

\[
f(\mathbf{x}_0 + \delta\mathbf{x}) = \sum_{|\alpha| \le N} f_\alpha \, \delta\mathbf{x}^\alpha
\]

The relationship to partial derivatives is:

\[
f_\alpha = \frac{1}{\alpha!} \frac{\partial^{|\alpha|} f}{\partial x_1^{\alpha_1} \cdots \partial x_M^{\alpha_M}} \bigg|_{\mathbf{x}_0}
\]

The `derivative()` method returns the partial derivative (multiplies the coefficient by
\(\alpha!\)), while `coeff()` returns the raw coefficient.

## Eigen Integration

tax ships with `NumTraits` and vocabulary helpers so `TaylorExpansion` objects can live
inside Eigen vectors and matrices:

```cpp
#include <tax/tax.hpp>

Eigen::Vector2<tax::TE<3, 2>> f = {tax::sin(x), tax::cos(y)};

auto vals = tax::value(f);           // Eigen::Vector2d of constant terms
auto J    = tax::jacobian(f, 2);     // 2×2 Jacobian at expansion point
auto H    = tax::hessian(f[0], 2);   // 2×2 Hessian of f[0]
```

## Next Steps

- [Core Module](core/index.md) — full treatment of the Taylor expansion type and expression templates
