# Dynamic-Shape Taylor Expansions

`tax::TaylorExpansionT<T, N, M>` follows Eigen's `Matrix<T, Rows, Cols>`
design: the order `N` and the number of variables `M` are both signed `int`
template parameters that accept either a compile-time non-negative integer or
the sentinel `tax::Dynamic` (= -1).

## Two configurations

| Configuration | Storage | Evaluation | Use case |
|---|---|---|---|
| Both `N >= 0` and `M >= 1` | stack `std::array<T, S>` | expression-template fusion | hot paths (ODE, ADS, performance code) |
| `N == Dynamic` and `M == Dynamic` | heap `std::vector<T>` | eager (operator by operator) | Python bindings, REPL, runtime composition |

The two configurations share the kernel layer:
- Static kernels: `cauchyProduct<T, N, M>(std::array<...>&, ...)` etc.,
  marked `constexpr`, inlined into the calling expression-template node.
- Runtime kernels: `cauchyProductRT(T*, const T*, const T*, std::size_t N,
  std::size_t M)` etc., consumed by the fully-dynamic specialisation.

Both implement the same recurrence relations and agree numerically to
within `1e-12` across every test we run.

## Quick reference (dynamic API)

```cpp
#include <tax/tax.hpp>

// Default alias — `TaylorExpansionT<double, Dynamic, Dynamic>`
using tax::DynTE;

// 1. Build a single coordinate variable.
auto x = DynTE<>::variable(/*x0=*/2.0,
                           /*var_idx=*/0,
                           /*order=*/5,
                           /*size=*/3);

// 2. Build a vector of `M` independent coordinate variables.
std::array<double, 3> x0{1.0, 2.0, 3.0};
auto vars = DynTE<>::variables(std::span<const double>(x0), /*order=*/5);

// 3. Arithmetic — eager, returns a fresh DynTE.
auto sum  = vars[0] + vars[1] + vars[2];
auto prod = vars[0] * vars[1];
auto div  = vars[0] / vars[1];
auto neg  = -vars[2];
auto sa   = 2.0 * vars[0] + 1.5;

// 4. Math functions — eager, return a fresh DynTE.
auto f = tax::sin(vars[0] * vars[1]) + tax::exp(vars[0] + vars[2]);
auto g = tax::log(vars[0]) - tax::sqrt(vars[1]);
auto h = tax::pow(vars[2], 0.5);

// 5. Access coefficients.
double v = f.value();              // constant term
double c = f.coeff({0, 1, 0});     // coefficient at exponent (0, 1, 0)
const auto& all = f.coeffs();      // const ref to the std::vector buffer
```

## Compatibility

- **Cross-shape operations runtime-assert** that both operands have the same
  `(order, size)`. Mixing two `DynTE`s of different shapes triggers an
  `assert` (defined as a contract; build with `-DNDEBUG` to skip).
- **Mixing static and dynamic operands directly is not supported** in this
  release — the dynamic specialisation only operates on other dynamic
  operands. Use the static `TaylorExpansionT<T, N, M>` API for static-shape
  arithmetic.
- **ODE and ADS modules require static shape.** Passing
  `tax::Dynamic` through `tax::ode::integrate<...>`,
  `tax::ode::AdsIntegrator<...>`, etc. produces a clear compile error from
  the `static_assert(N >= 0)` inside the static `TaylorExpansionT` primary
  template.

## When to use which

- **Static `TE<N>` / `TEn<N, M>`**: maximum performance. ODE/ADS hot paths.
  Anywhere you know the shape at compile time.
- **Dynamic `DynTE<T>`**: any time the shape is not known until runtime.
  Most relevant for nanobind / Python bindings, where you want a single
  `tax.TaylorExpansion(order, size)` constructor.
