# CLAUDE.md — AI Assistant Guide for `tax`

## Project Overview

**tax** is a header-only C++23 library for **Truncated Algebraic eXpansions (TAX)** — truncated multivariate Taylor polynomials that propagate complete Taylor series through arbitrary expressions. In a single evaluation pass, it yields function values and all partial derivatives up to order N.

- **Version:** 0.1.0
- **License:** BSD 3-Clause
- **C++ Standard:** C++23 (required)
- **Build system:** CMake

---

## Repository Structure

```
tax/
├── include/tax/          # Header-only library (the entire library lives here)
│   ├── tax.hpp           # Umbrella header — users include only this
│   ├── ads.hpp           # Facade: includes all ads/
│   ├── kernels.hpp       # Facade: includes all kernels/
│   ├── operators.hpp     # Facade: includes all operators/
│   ├── utils.hpp         # Facade: includes all utils/
│   ├── ads/              # Automatic Domain Splitting (ADS)
│   ├── expr/             # Expression template nodes (lazy evaluation)
│   ├── kernels/          # Series computation kernels (recurrence relations,
│   │                       static + runtime-shape overloads)
│   ├── ode/              # Taylor ODE integrator
│   ├── operators/        # Free-function operators and math functions
│   ├── storage/          # ShapeBase EBO + TaylorExpansionT static and dynamic
│   │                       specialisations:
│   │                         shape.hpp                 — EBO holders
│   │                         tte_static.hpp            — primary template (static N, M)
│   │                         tte_dynamic.hpp           — <T, Dynamic, Dynamic>
│   │                         tte_dynamic_order.hpp     — <T, Dynamic, M>
│   ├── utils/            # Type traits, combinatorics, enumeration
│   └── eigen/            # Eigen3 integration helpers
├── tests/                # Google Test suite (~33 test executables, 523 tests)
│   ├── ads/              # ADS tree and runner tests
│   ├── core/             # Basic TTE construction, nesting, composition, deriv/integ
│   ├── dynamic/          # Dynamic-shape TaylorExpansionT tests
│   ├── expr/             # Expression template correctness
│   ├── kernels/          # Kernel algorithm verification (incl. runtime-shape)
│   ├── foundation/       # Combinatorics, enumeration, ShapeBase EBO
│   ├── eigen/            # Eigen integration tests
│   ├── ode/              # Taylor integrator and ADS-integrated ODE tests
│   ├── dace/             # Optional DACE comparative tests
│   ├── testUtils.hpp     # Shared test helpers and macros
│   └── CMakeLists.txt
├── benchmarks/           # Google Benchmark suite
├── python/               # nanobind Python bindings (build with TAX_BUILD_PYTHON=ON)
│   ├── CMakeLists.txt
│   ├── src/tax_module.cpp   — nanobind bindings for DynTE
│   ├── tax/__init__.py      — `import tax` wrapper
│   └── tests/               — pytest suite
├── pyproject.toml        # scikit-build-core wheel config (pip install .)
├── docs/                 # Markdown documentation
├── cmake/                # CMake package config template
├── tools/                # install_eigen.sh helper script
├── .github/workflows/    # CI: tests.yml, sanitizers.yml
├── .clang-format         # Code style configuration
├── CMakeLists.txt        # Root CMake configuration
└── README.md
```

---

## Building

```bash
# Configure
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build

# Test
ctest --test-dir build --output-on-failure
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `TAX_BUILD_UNITTESTS` | `ON` | Build Google Test unit-test suite |
| `TAX_BUILD_BENCHMARK` | `OFF` | Build Google Benchmark suite |
| `TAX_BUILD_PYTHON` | `OFF` | Build the nanobind Python bindings (target `_tax`) |
| `TAX_USE_DACE` | `OFF` | Enable DACE comparison (fetched automatically) |

### With Python bindings

```bash
pip install nanobind pytest
cmake -S . -B build -DTAX_BUILD_PYTHON=ON
cmake --build build -j
PYTHONPATH=$PWD/build/python python3 -c "
    import tax
    x, y = tax.variables([0.5, 0.3], order=4)
    print(tax.sin(x * y) + tax.exp(x + y))
"
```

Or install as a wheel via `pip install .` (uses `scikit-build-core` from
`pyproject.toml`). The pytest suite is wired into ctest as the
`python_bindings` target when both `TAX_BUILD_TEST` and `TAX_BUILD_PYTHON`
are `ON`.

The Python API mirrors `tax::DynTE<double>` — runtime order + runtime
size, eager evaluation. Construction goes through module-level factories
(`zero`, `one`, `constant`, `variable`, `variables`).

### With Benchmarks

```bash
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release \
  -DTAX_BUILD_BENCHMARK=ON -DTAX_USE_DACE=ON
cmake --build build-bench --target bench_univariate -j
./build-bench/benchmarks/bench_univariate
```

### With Coverage

```bash
cmake -S . -B build-cov -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS=--coverage
cmake --build build-cov
ctest --test-dir build-cov --output-on-failure
gcovr --root . build-cov --filter "^include/tax/"
```

### Dependencies

- **Required:** Eigen3 ≥ 3.4 (must be installed or pointed to via `CMAKE_PREFIX_PATH`)
- **Optional:** DACE v2.1.0 — fetched automatically via CMake FetchContent if `TAX_USE_DACE=ON`
- **Test framework:** Google Test v1.17 — fetched automatically if not found
- **Benchmark framework:** Google Benchmark v1.9 — fetched automatically if not found

---

## Core Concepts

### The Main Type

```cpp
tax::TaylorExpansionT<T, N, M>
// T = scalar type (double, float)
// N = truncation order (compile-time integer, or `tax::Dynamic`)
// M = number of variables (compile-time integer, or `tax::Dynamic`)
```

Convenient aliases:
```cpp
tax::TE<N>        // univariate: TaylorExpansionT<double, N, 1>
tax::TEn<N, M>    // multivariate: TaylorExpansionT<double, N, M>
tax::DynTE<T = double>  // fully dynamic: TaylorExpansionT<T, Dynamic, Dynamic>
```

`tax::TruncatedTaylorExpansionT` survives as a deprecated alias for one
transition cycle and emits a `[[deprecated]]` warning at use.

### Static vs. Dynamic Shape

The library follows Eigen's `Matrix<T, Rows, Cols>` design:
- **Both N and M are non-negative integers (static config)**: stack-resident
  `std::array<T, numMonomials(N, M)>` storage, full constexpr surface,
  expression-template fusion. This is the hot path used by ODE and ADS.
- **N and/or M are `tax::Dynamic` (-1)**: shape resolved at construction;
  storage is `std::vector<T>`; operators are eager (no ET fusion). Intended
  for Python bindings, REPL use, and runtime composition.

The two configurations share the kernel layer: static-mode kernels operate
on `std::array`, dynamic-mode `*RT` overloads operate on raw `T*` plus
runtime `(N, M)` sizes; both implement the same recurrence relations and
agree numerically to within 1e-12.

ODE and ADS modules require static shape — they `static_assert(N >= 0)`
via the static `TaylorExpansionT` template itself. Trying to pass `Dynamic`
through `ode::integrate<...>` or `AdsIntegrator<...>` produces a clear
compile error.

### Creating Variables

```cpp
// Static, univariate
auto x = tax::TE<3>::variable(x0);            // x = x0 + 1*dx

// Static, multivariate (structured bindings)
auto [x, y] = tax::TEn<3, 2>::variables(x0, y0);

// Dynamic, runtime order/size
auto z = tax::DynTE<>::variable(/*x0=*/2.0,
                                /*var_idx=*/0,
                                /*order=*/5,
                                /*size=*/3);

// Vector of dynamic coordinate variables
std::array<double, 3> x0{1.0, 2.0, 3.0};
auto vars = tax::DynTE<>::variables(std::span<const double>(x0), /*order=*/5);
auto sum = vars[0] + vars[1] + vars[2];        // eager evaluation
auto f   = tax::sin(vars[0] * vars[1]) + tax::exp(vars[2]);
```

### Using the Library

```cpp
#include <tax/tax.hpp>

auto x = tax::TE<5>::variable(1.0);
auto f = sin(x) * exp(x);

double val  = f.value();           // constant term
double df   = f.derivative(1);     // first derivative
double d2f  = f.derivative(2);     // second derivative
auto   p    = f.eval(dx);          // evaluate Taylor polynomial at x0+dx
```

### Differentiation and Integration

TTE objects support symbolic partial differentiation and integration:

```cpp
auto x = tax::TE<5>::variable(1.0);
auto f = sin(x);

// Compile-time variable index
auto df = f.deriv<0>();       // d/dx sin(x) = cos(x)
auto F  = f.integ<0>();       // integral of sin(x) dx

// Runtime variable index
auto df2 = f.deriv(0);       // equivalent to deriv<0>()
auto F2  = f.integ(0);       // equivalent to integ<0>()

// Multivariate
auto [x, y] = tax::TEn<3, 2>::variables(1.0, 2.0);
auto g = x * x * y;
auto dg_dx = g.deriv<0>();   // partial derivative w.r.t. x
auto dg_dy = g.deriv<1>();   // partial derivative w.r.t. y
auto Gx    = g.integ<0>();   // integral w.r.t. x
```

### Coefficient Storage

- Coefficients stored in `std::array<T, nCoefficients>` (stack-allocated, no heap)
- Graded-lexicographic ordering: all degree-0 first, then degree-1, etc.
- Size: `nCoefficients = C(N+M, M)` (binomial coefficient)
- `coeff(k)` retrieves raw Taylor coefficient; `derivative(k)` applies `k!` scaling

---

## Architecture: Expression Templates

The library uses lazy evaluation via expression templates to avoid materializing intermediate TTE objects:

```
User writes:  sin(x * y + z)
Builds tree:  UnaryExpr<sin, ProductExpr<x, y>, z>
Evaluated:    Only when assigned to TruncatedTaylorExpansionT
```

Key design choices:
- **Sum flattening:** `a + b + c + d` → single `SumExpr<a,b,c,d>` (one pass)
- **Product flattening:** `a * b * c` → single `ProductExpr<a,b,c>` (rolling Cauchy product)
- **Leaf fast-paths:** Binary ops on materialized TTE objects take shortcuts
- **CRTP base:** `Expr<Derived, T, N, M>` unifies all nodes

### Key Files in `expr/`

| File | Purpose |
|------|---------|
| `base.hpp` | CRTP `Expr<>` base with evaluation interface |
| `arithmetic_ops.hpp` | Op tags: `OpAdd`, `OpSub`, `OpMul`, `OpDiv` |
| `bin_expr.hpp` | Binary expression nodes |
| `sum_expr.hpp` | Flattened N-ary sum |
| `product_expr.hpp` | Flattened N-ary product |
| `unary_expr.hpp` | Unary function applications |
| `func_expr.hpp` | Generic function call nodes |
| `math_ops.hpp` | High-level math wrappers |

---

## Kernels

All mathematical operations are implemented as degree-by-degree recurrence relations in `kernels/`:

| File | Operations |
|------|-----------|
| `algebra.hpp` | reciprocal, sqrt, cbrt, square, cube |
| `cauchy.hpp` | Cauchy product, self-product, accumulate |
| `trigonometric.hpp` | sin, cos, tan, asin, acos, atan |
| `transcendental.hpp` | exp, log, sinh, cosh, tanh, and inverses |
| `ops.hpp` | Utility helpers |

Kernel optimisations:
- **Symmetry exploitation:** `cauchySelfProduct` enumerates unordered pairs for ~2x fewer multiplications
- **Univariate fast-paths:** `if constexpr (M == 1)` branches avoid multi-index overhead
- **Incremental `sq` tracking in `seriesCbrt`:** Maintains `out^2` degree-by-degree for O(N^2) instead of O(N^3)

When adding a new mathematical function, implement the recurrence relation in the appropriate kernel file, then expose it via `operators/math_unary.hpp` or `operators/math_binary.hpp`.

---

## Eigen Integration

Located in `include/tax/eigen/`. Enables using TTE types inside Eigen matrices/vectors.

```cpp
#include <tax/tax.hpp>

Eigen::Vector2d x0 = {1.0, 2.0};
auto [x, y] = tax::variables<tax::TEn<3,2>>(x0);

Eigen::Vector2<tax::TEn<3,2>> f = {sin(x), cos(y)};

auto vals = tax::value(f);          // Eigen::Vector2d of constant terms
auto grad = tax::gradient(f[0], 2); // Gradient of f[0] w.r.t. 2 variables
auto J    = tax::jacobian(f, 2);    // 2x2 Jacobian matrix
```

Key helpers in `eigen/`:
- `tax::vector<TTE>(x0)` — element-wise Eigen vector → TTE vector
- `tax::variables<TTE>(x0)` — structured-binding access
- `tax::value(container)` — extract constant terms
- `tax::eval(container, dx)` — evaluate at displacement
- `tax::gradient(f, M)` — gradient vector
- `tax::jacobian(F, M)` — Jacobian matrix

---

## ODE Integration

Located in `include/tax/ode/`. Adaptive Runge–Kutta and Taylor-method integration for vector ODEs, with an event system (triggers/actions), dense output, and an ADS / LOADS pipeline built on top.

### Quick start — propagate

```cpp
#include <tax/ode.hpp>

using namespace tax::ode::methods;

auto rhs = [](const auto& x, double) {
    auto out = x;  // copy-shape
    out(0) =  x(1);
    out(1) = -x(0);
    return out;
};

tax::la::VecNT<2, double> x0{1.0, 0.0};

// Default: Dense=false (queryable only at step boundaries).
auto sol = tax::ode::propagate(Verner89{}, rhs, x0, 0.0, 2 * M_PI);

// Dense=true (sol(t) interpolates).
auto sol_d = tax::ode::propagate</*Dense=*/true>(
    Taylor<16>{}, rhs, x0, 0.0, 2 * M_PI);
auto x_at = sol_d(0.42);

// With events
std::vector<tax::ode::Event<tax::ode::Verner89Stepper<decltype(x0)>>> events;
events.emplace_back(tax::ode::EveryStep(), tax::ode::Record("step"));
auto sol_e = tax::ode::propagate(Verner89{}, rhs, x0, 0.0, 1.0, cfg, events);
```

Methods: `methods::Taylor<N>`, `Verner78`, `Verner89`, `Fehlberg78`, `Feagin12`, `Feagin14`.

### Lower-level API

Construct an `Integrator<Stepper, F, Dense>` explicitly when you need a non-default Controller (e.g. `FixedStep`) or want to reuse an integrator across calls.

### Snapshot CSV helpers (`<tax/ode/io.hpp>`, opt-in)

```cpp
auto times = tax::ode::linspace(0.0, t_final, 200);
tax::ode::writeCsv(sol_d, times, "traj.csv");   // columns: t, x0..x{D-1}
```

### Key Files in `ode/`

| File | Purpose |
|------|---------|
| `propagate.hpp`          | `tax::ode::propagate<Dense>(method, rhs, x0, t0, t1, cfg, events)` + `methods::` tag namespace |
| `integrator.hpp`         | `Integrator<Stepper, F, Dense>` driver + method type aliases |
| `config.hpp`             | `IntegratorConfig<T>` (abstol/reltol/initial_step/min_step/max_step/max_steps/max_rejects_per_step) |
| `controllers.hpp`        | I, PI (Gustafsson), H211b (Söderlind), JorbaZou (Taylor-only), FixedStep |
| `steppers/`              | One stepper per file: `taylor.hpp`, `verner78.hpp`, `verner89.hpp`, `fehlberg78.hpp`, `feagin12.hpp`, `feagin14.hpp` |
| `solution.hpp`           | `Solution<Stepper, State, Dense>` (continuous-extension dense output via `operator()`) |
| `event.hpp`              | `Event<Stepper>`, `TriggerContext`, `StepperCtx`, `Direction`, `ControlFlow` |
| `triggers.hpp`           | `EveryStep`, `ZeroCrossing` (poly-Newton or Brent) |
| `actions.hpp`            | `Continue`, `Terminate`, `Record`, `Custom`, `EventStorage` |
| `vector_ops.hpp`         | `VectorOps<S>` trait — norm/axpy/scale for scalar / TaylorExpansion / Eigen states |
| `io.hpp` (opt-in)        | `linspace`, `writeCsv(sol, times, path)` |

---

## Automatic Domain Splitting (ADS)

Located in `include/tax/ads/`. Implements Wittig 2015's ADS and the LOADS
variant (Losacco/Fossà/Armellin 2024) by composing with the existing
`tax::ode` event infrastructure — `tax::ode` itself is not modified.

### ODE Propagation with ADS

```cpp
#include <tax/ads.hpp>
#include <tax/ode.hpp>

using namespace tax::ode::methods;

auto f = [](const auto& x, double) {
    using S = std::decay_t<decltype(x)>;
    S out{x.size()};
    out(0) =  x(1);
    out(1) = -x(0) - 0.1 * x(0) * x(0) * x(0);
    return out;
};

tax::ads::Box<double, 2> ic_box{
    tax::la::VecNT<2, double>{1.0, 0.0},
    tax::la::VecNT<2, double>{0.5, 0.5},
};
tax::la::VecNT<2, double> center{1.0, 0.0};

tax::ode::IntegratorConfig<double> cfg;
cfg.abstol = cfg.reltol = 1e-12;

auto tree = tax::ads::propagate</*P=*/6>(
    Verner89{},
    tax::ads::TruncationCriterion{/*tol=*/1e-4, /*maxDepth=*/8},
    f, ic_box, center, 0.0, 2 * M_PI, cfg);

for (int i : tree.done()) {
    const auto& l = tree.leaf(i);
    // l.payload — DA-valued flow map at t = t1 on l.box
}
```

### LOADS — Nonlinearity-Index Criterion

Same setup, swap criterion:

```cpp
auto tree = tax::ads::propagate<6>(
    Verner89{}, tax::ads::NliCriterion{0.1, 8},
    f, ic_box, center, 0.0, 2 * M_PI, cfg);
```

### Post-pass Merger

```cpp
auto stats = tax::ads::merge(tree, tax::ads::TruncationCriterion{1e-4});
// stats.passes / stats.merges / stats.rejected
```

### Architecture

- **Leaf-only arena tree** (`AdsTree<Payload, M, T>`): single `std::vector<Leaf>` with `parentIdx`/`siblingIdx` on each leaf. Splits retire the parent in place; merges revive it.
- **Event interop**: `(SplitTrigger, SplitAction)` is appended to the user's event list and passed to `tax::ode::Integrator`. The trigger fires at accepted-step boundaries when the criterion says split; the action records `{dim, t}` into a `SplitRequest` and returns `ControlFlow::Terminate`. The driver consumes the request to decide split vs. mark-done.
- **Templated on Stepper**: any `tax::ode::Stepper` works as long as the state type is `tax::la::VecNT<D, tax::TaylorExpansion<T, N, M, Storage>>` with `N >= 1`, `M >= 1`.
- **Boundary-only splits**: triggers fire only at accepted-step boundaries (matches Wittig's original ADS).
- **Post-pass merger**: `tax::ads::merge(tree, criterion)` walks done sibling pairs bottom-up and collapses any pair whose reconstructed parent payload satisfies the criterion within `tol`.

### Key files in `ads/`

| File                       | Purpose |
|----------------------------|---------|
| `box.hpp`                  | `Box<T, M>` axis-aligned subdomain (Eigen-backed `tax::la::VecNT` storage) |
| `leaf.hpp`                 | `Leaf<Payload, M, T>` POD record |
| `tree.hpp`                 | `AdsTree<Payload, M, T>` arena + BFS queue |
| `criteria.hpp`             | `SplitCriterion` concept + `TruncationCriterion` + `NliCriterion` |
| `nonlinearity_index.hpp`   | LOADS NLI math (`tax::ads::detail`) |
| `split_event.hpp`          | `SplitRequest`, `SplitTrigger`, `SplitAction` |
| `da_state.hpp`             | `create(box, x0)`, `split(state, parent_box, dim)` |
| `driver.hpp`               | `AdsDriver<Stepper, Criterion>` BFS driver |
| `propagate.hpp`            | `tax::ads::propagate<P>(method, criterion, rhs, ic_box, ic_center, t0, t1, cfg)` convenience wrapper |
| `merge.hpp`                | `merge(tree, criterion)` + `MergeStats` |
| `io.hpp` (opt-in)          | `writeTreeCsv`, `writeBoxCountCsv` |

---

## Code Conventions

### Naming

| Category | Convention | Examples |
|----------|-----------|---------|
| Types/Classes | `PascalCase` | `TruncatedTaylorExpansionT`, `MultiIndex`, `AdsTree`, `AdsRunner`, `AdsNode`, `FlowMap`, `TaylorSolution` |
| Template params | `UPPERCASE` or short | `T`, `N`, `M`, `P`, `D`, `Derived` |
| Free functions & methods | `camelCase` | `variable()`, `flatIndex()`, `seriesReciprocal()`, `deriv()`, `integ()`, `findLeaf()`, `addLeaf()`, `markDone()`, `integrateAds()`, `makeAdsRunner()` |
| Local variables | `snake_case` | `n_coeff`, `dx`, `half_width` |
| Namespaces | `lowercase` | `tax`, `tax::detail`, `tax::ode` |
| Op tags | `PascalCase` with prefix | `OpAdd`, `OpSub`, `OpMul` |
| Type aliases | Short uppercase | `TE<N>`, `TEn<N,M>` |

### C++ Patterns

- **`constexpr` everywhere:** All size calculations, index mappings, and coefficient operations must be `constexpr`
- **`noexcept` on all operations:** For zero-overhead guarantees (exception: methods that `throw`, e.g. runtime-index `deriv(int)`)
- **No heap allocation in core library:** Use `std::array` for fixed-size storage; `std::vector` is acceptable only in ODE/ADS modules for variable-length solutions and dynamic tree growth
- **Concepts:** Use `tax::Scalar` concept (wraps `std::floating_point`) for scalar template parameters
- **`if constexpr`:** Used for compile-time branching between univariate (M=1) and multivariate cases
- **`[[nodiscard]]`:** Applied to accessor methods, computation results, and expensive operations
- **Internal details in `tax::detail`:** Do not expose implementation internals in `tax::`; ODE internals use `tax::ode::detail`

### Formatting

Enforced by `.clang-format` (Google style, customized):
- Indent: **4 spaces** (no tabs)
- Column limit: **100 characters**
- Brace wrapping: new line after class/struct/function/namespace/control statements
- Spaces inside parentheses and angle brackets

Run clang-format before committing:
```bash
clang-format -i include/tax/**/*.hpp
```

---

## Testing

### Structure

Tests are organized by feature, one `.cpp` per concern. Each produces a standalone test executable:

```
tests/core/        — TTE constructors, variable factories, composition, deriv/integ
tests/expr/        — One file per math function (sin, exp, log, pow, etc.)
tests/kernels/     — Direct kernel algorithm verification
tests/foundation/  — Combinatorics and enumeration
tests/eigen/       — Eigen integration
tests/ads/         — ADS tree structure and runner (Gaussian approximation)
tests/ode/         — Taylor integrator (scalar, vector, ADS-integrated)
tests/dace/        — Comparative tests against DACE (optional)
```

### Writing Tests

```cpp
#include <gtest/gtest.h>
#include <tax/tax.hpp>
#include "testUtils.hpp"   // ExpectCoeffsNear, kTol

TEST(ExprSin, UnivariateOrder3) {
    auto x = tax::TE<3>::variable(0.0);
    auto f = sin(x);
    // Expected coefficients for sin(x) at x=0: [0, 1, 0, -1/6]
    tax::TE<3> expected{0.0, 1.0, 0.0, -1.0/6.0};
    ExpectCoeffsNear<tax::TE<3>>(f, expected, kTol);
}
```

- Use `ExpectCoeffsNear<TTE_type>(actual, expected, tol)` for coefficient-wise comparison
- Default tolerance: `kTol = 1e-10`
- Each new math function or kernel needs a corresponding test in `tests/expr/` or `tests/kernels/`

### Running Tests

```bash
ctest --test-dir build --output-on-failure
# Or run a single test executable:
./build/tests/testExprTrig
```

---

## CI/CD

### Workflows

**`tests.yml`** — Triggered on push, pull_request, and manual dispatch:
- Matrix: Ubuntu + macOS × GCC + Clang × Eigen 3.4.0 + Eigen 5.0.0 (8 combinations)
- Build type: Release
- Includes DACE for comparison tests
- Coverage job: GCC/Ubuntu/Debug, reports to Codecov (filter: `^include/tax/`)

**`sanitizers.yml`** — Manual dispatch only:
- Precheck: same matrix as tests.yml
- Sanitizer jobs: ASAN, UBSAN, TSAN
- Build type: RelWithDebInfo with `-O1 -fno-omit-frame-pointer -g`

### Before Submitting a PR

1. All 29 test executables pass locally (440 individual tests)
2. Code is formatted with `clang-format`
3. No new dynamic allocations introduced in core library
4. New math operations have kernel tests AND expression tests
5. Eigen helpers have tests in `tests/eigen/`
6. ODE/ADS changes have tests in `tests/ode/` or `tests/ads/`

---

## Adding a New Mathematical Function

1. **Kernel:** Implement the degree-by-degree recurrence in the appropriate `kernels/` file
2. **Operator:** Expose via a free function in `operators/math_unary.hpp` (or `math_binary.hpp`)
3. **Expression node:** If the function has special structure, add a node in `expr/`; otherwise use `UnaryExpr`
4. **Tests:** Add `tests/expr/testExpr<FunctionName>.cpp` with univariate and multivariate cases
5. **CMakeLists:** Register the new test file in `tests/CMakeLists.txt`
6. **Docs:** Update `docs/math_operations.md` with the recurrence relation

---

## Common Pitfalls

- **Do not use `std::vector` or `new` in core library:** The core TTE type must remain allocation-free; `std::vector` is only acceptable in ODE/ADS modules
- **Do not break `constexpr`:** All index arithmetic must stay compile-time
- **graded-lex ordering is sacred:** The coefficient order (`flatIndex`) is used everywhere — never change it
- **M=0 is invalid:** Always assert or static_assert that M ≥ 1
- **Concepts vs. SFINAE:** Prefer C++20 concepts (`requires`, `Scalar` concept) over SFINAE
- **Include the umbrella header in tests:** Use `#include <tax/tax.hpp>`, not individual sub-headers
- **Expression templates store references:** When using ADS or ODE with expression-returning callables, arguments must be taken by `const&` — by-value copies dangle once the function returns its lazy expression

---

## Documentation

- `docs/getting_started.md` — Installation, basic usage, key concepts
- `docs/api_reference.md` — Complete API reference
- `docs/math_operations.md` — Mathematical recurrence relations for all operations
- `docs/eigen_integration.md` — Eigen helper reference
- `README.md` — Project overview with quick-start examples
- Doxygen: `doxygen Doxyfile` generates HTML docs from header comments
