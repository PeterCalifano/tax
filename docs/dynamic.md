# Dynamic-Shape Taylor Expansions

`tax::TaylorExpansionT<T, N, M>` follows Eigen's `Matrix<T, Rows, Cols>`
design: the order `N` and the number of variables `M` are both signed `int`
template parameters that accept either a compile-time non-negative integer or
the sentinel `tax::Dynamic` (= -1).

## Three configurations

| Configuration | Storage | Evaluation | Use case |
|---|---|---|---|
| Both `N >= 0` and `M >= 1` | stack `std::array<T, S>` | expression-template fusion | hot paths (ODE, ADS, performance code) |
| `N == Dynamic`, `M >= 1` | heap `std::vector<T>` | eager (operator by operator) | **M fixed by the problem (e.g. 6-D state), order chosen at runtime** |
| `N == Dynamic` and `M == Dynamic` | heap `std::vector<T>` | eager (operator by operator) | Python bindings, REPL, runtime composition |

The mixed `<T, N, Dynamic>` case (static order, dynamic size) is not currently
specialised — open a follow-up if you need it.

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

// Mixed-dynamism alias — runtime order, compile-time size.
// `DynOrderTE<M, T>` = `TaylorExpansionT<T, Dynamic, M>`. Storage is still
// `std::vector<T>` (size depends on runtime order), but the variable index
// remains a compile-time concept and `variables()` returns a static array.
using tax::DynOrderTE;

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

## Quick reference (dynamic order, static size)

```cpp
#include <tax/tax.hpp>

// 3-D state, truncation order chosen at runtime.
using TE3 = tax::DynOrderTE<3>;  // = TaylorExpansionT<double, Dynamic, 3>

// All `M` coordinate variables — structured binding works because the
// returned std::array has compile-time size M.
std::array<double, 3> x0{1.0, 2.0, 3.0};
auto [x, y, z] = TE3::variables(x0, /*order=*/5);

// Same arithmetic + math API as the fully-dynamic case.
auto f = tax::sin(x * y) + tax::exp(z);
auto g = tax::pow(x + y, 0.5);

// `size()` is a constexpr method returning M (3); `order()` returns 5
// (the runtime-chosen truncation).
static_assert(decltype(f)::size_ct == 3);
static_assert(decltype(f)::order_ct == tax::Dynamic);

// Compile-time variable index also works since M is static.
auto x_other = TE3::variable<0>(x0, /*order=*/7);
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

## Performance comparison

Run `bench_dynamic_vs_static` (in `benchmarks/`) on the same workload through
both APIs. g++ 13, `-O3 -DNDEBUG`, μs/op:

| Workload | Static | Dynamic | Slowdown |
|---|---|---|---|
| `sin(x)`, N=10, M=1 | 0.060 | 0.118 | 1.97× |
| `sin(x)`, N=40, M=1 | 0.890 | 0.898 | 1.01× |
| `sin(x)·exp(x) + log(x+1)`, N=10, M=1 | 0.255 | 0.415 | 1.62× |
| `sin(x)·exp(x) + log(x+1)`, N=40, M=1 | 2.89 | 3.04 | 1.05× |
| `x·y`, N=5, M=2 | 0.467 | 3.10 | 6.62× |
| `x·y`, N=5, M=4 | 16.75 | 61.42 | 3.67× |
| `sin(x·y) + exp(x+y)`, N=5, M=2 | 1.75 | 10.01 | 5.72× |
| `sin(x·y) + exp(x+y)`, N=5, M=4 | 140.12 | 281.11 | 2.01× |

Dynamic mode pays a fixed per-operation cost (heap allocation for intermediates,
runtime dispatch, no ET fusion) that **amortises away as the per-call work
grows**. The slowdown ranges from ~1× for large univariate composites to ~7×
for very small multivariate operations.

Rule of thumb: use static for inner loops that run many iterations on small
shapes; use dynamic when the shape is unknown until runtime or when you call
the operation a small number of times.
