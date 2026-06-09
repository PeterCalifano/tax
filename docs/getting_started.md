# Getting Started

This page walks you from a fresh checkout to writing your first Taylor expansion
and propagating an ODE.

---

## Requirements

| Tool | Minimum version |
|---|---|
| C++ compiler | GCC 13, Clang 17, Apple Clang 16 (C++23 with concepts) |
| CMake | 3.28 |
| Eigen | 3.4 |
| GoogleTest | fetched automatically by CMake if missing |
| Google Benchmark | fetched automatically when `TAX_BUILD_BENCHMARK=ON` |

## Build & test from source

```bash
git clone https://github.com/andreapasquale94/tax.git
cd tax
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### CMake options

| Option | Default | Description |
|---|:-:|---|
| `TAX_BUILD_UNITTESTS` | `ON`  | Build the GoogleTest unit-test suite |
| `TAX_BUILD_BENCHMARK` | `OFF` | Build the Google Benchmark suite |

The fast Cauchy kernel paths (`TAX_USE_UNROLL` for $M=1$, `TAX_USE_STENCIL`
for $M \ge 2$) default to on in `<tax/kernels/cauchy.hpp>`; pre-define the
macro to `0` (identically in every translation unit) to fall back to the
loop kernel.

### Install and consume from another CMake project

```bash
cmake --install build --prefix /your/install/prefix
```

```cmake
find_package(tax CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE tax::tax)
```

If installed to a non-standard prefix:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/your/install/prefix
```

---

## Single umbrella header

```cpp
#include <tax/tax.hpp>          // core + Eigen integration
```

The ODE module is opt-in via a separate header:

```cpp
#include <tax/ode.hpp>          // adaptive Taylor + RK steppers
```

---

## The core type

```cpp
namespace tax {
    template <typename T, int N, int M = 1, typename Storage = storage::Dense>
    class TaylorExpansion;
}
```

| Parameter | Meaning |
|---|---|
| `T`       | Scalar coefficient type (`double`, `float`) |
| `N`       | Maximum total polynomial order, $N \ge 0$ |
| `M`       | Number of independent variables, $M \ge 1$ (default `1`) |
| `Storage` | `tax::storage::Dense` (default) or `tax::storage::Sparse` |

### Convenience aliases

```cpp
tax::TE<N>          // dense univariate  — TaylorExpansion<double, N, 1, Dense>
tax::TE<N, M>       // dense multivariate
tax::STE<N>         // sparse univariate — TaylorExpansion<double, N, 1, Sparse>
tax::STE<N, M>      // sparse multivariate
```

See [Dense vs Sparse Storage](core/storage.md) for the trade-offs.

---

## Creating variables

=== "Univariate (dense)"

    ```cpp
    auto x = tax::TE<5>::variable(1.0);   // x = 1 + δx
    ```

=== "Multivariate (dense)"

    ```cpp
    using TE2 = tax::TE<3, 2>;
    auto x = TE2::variable<0>({1.0, 2.0});   // coordinate 0
    auto y = TE2::variable<1>({1.0, 2.0});   // coordinate 1
    ```

=== "Eigen vector form"

    ```cpp
    // Same as above, returned as Eigen::Vector<TE2, 2>
    auto v = tax::variables<tax::TE<3, 2>>(Eigen::Vector2d{1.0, 2.0});
    auto& x = v(0); auto& y = v(1);
    ```

=== "Sparse"

    ```cpp
    auto x = tax::STE<5>::variable(1.0);   // identical factories, sparse storage
    ```

---

## Building expressions

Arithmetic and math functions work naturally; the right-hand side builds a lazy
expression tree that is materialised only on assignment.

```cpp
auto x = tax::TE<6>::variable(0.0);
tax::TE<6> f = tax::sin(x) + tax::square(x) / 2.0;
```

A complete list of supported operations is in the [API Reference](core/api.md).

---

## Extracting results

```cpp
f.value();              // f(x₀) — the constant term
f.coeff({k});           // coefficient of δxᵏ
f.derivative({k});      // k-th derivative (= k! · coeff)
f.eval(0.3);            // evaluate Taylor polynomial at x₀ + 0.3
```

For multivariate objects:

```cpp
using TE2 = tax::TE<3, 2>;
auto x = TE2::variable<0>({0.0, 0.0});
auto y = TE2::variable<1>({0.0, 0.0});
TE2 g = x*x + x*y + y*y;

g.derivative({2, 0});   // ∂²g/∂x²   = 2
g.derivative({1, 1});   // ∂²g/∂x∂y  = 1
g.coeff<1, 1>();        // compile-time index access
```

### Coefficients vs derivatives

A Taylor expansion stores **coefficients** of the monomial basis:

$$
f(\mathbf{x}_0 + \delta\mathbf{x})
  = \sum_{|\alpha| \le N} f_\alpha \, \delta\mathbf{x}^\alpha
$$

The relationship to partial derivatives is

$$
f_\alpha
  = \frac{1}{\alpha!} \,
    \frac{\partial^{|\alpha|} f}
         {\partial x_1^{\alpha_1} \cdots \partial x_M^{\alpha_M}}
    \bigg|_{\mathbf{x}_0}
$$

so `derivative()` returns `coeff()` multiplied by $\alpha!$.

---

## Eigen integration

`tax::TaylorExpansion` ships with a `NumTraits` specialisation so it can live
inside Eigen vectors and matrices unchanged.

```cpp
#include <tax/tax.hpp>

using TE2 = tax::TE<3, 2>;
auto x = TE2::variable<0>({1.0, 2.0});
auto y = TE2::variable<1>({1.0, 2.0});
Eigen::Vector2<TE2> F = {tax::sin(x), tax::cos(y)};

auto vals = tax::value(F);           // Eigen::Vector2d of constant terms
auto J    = tax::jacobian(F, 2);     // 2×2 Jacobian at expansion point
auto H    = tax::hessian(F[0], 2);   // 2×2 Hessian of F[0]
```

Reference is in [Eigen Integration](eigen/index.md).

---

## A first ODE

```cpp
#include <tax/ode.hpp>
#include <Eigen/Core>

// Harmonic oscillator: dx/dt = v, dv/dt = -x
auto f = [](const auto& x, const auto& /*t*/) {
    using S = std::decay_t<decltype(x)>;
    S out;
    out(0) =  x(1);
    out(1) = -x(0);
    return out;
};

tax::ode::IntegratorConfig<double> cfg;
cfg.abstol = cfg.reltol = 1e-12;

auto integ = tax::ode::makeTaylorIntegrator<25, double, 2>(f, cfg);

Eigen::Vector2d x0{1.0, 0.0};
auto sol = integ.integrate(x0, /*t0=*/0.0, /*tmax=*/2.0 * M_PI);

sol.x.back();  // ≈ (1, 0): one full period
```

More examples — RK methods, events, dense output — in
[ODE Examples](ode/examples.md).

---

## Next steps

- [Core / Mathematical Foundations](core/math.md) — the math behind the type.
- [Core / API Reference](core/api.md) — every public method, listed.
- [ODE / Methods & Benchmarks](ode/methods.md) — pick the right integrator.
- [Internals](internals/index.md) — how expression templates and recurrences fit together.
