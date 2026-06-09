# CLAUDE.md — AI Assistant Guide for `tax`

## Project Overview

**tax** is a header-only C++23 library for **Truncated Algebraic eXpansions (TAX)** — truncated multivariate Taylor polynomials that propagate complete Taylor series through arbitrary expressions. In a single evaluation pass, it yields function values and all partial derivatives up to order N. On top of the core type it ships an adaptive ODE integration module (`tax::ode`) and Automatic Domain Splitting (`tax::ads`).

- **Version:** 0.1.0
- **License:** BSD 3-Clause
- **C++ Standard:** C++23 (required)
- **Build system:** CMake

---

## Repository Structure

```
tax/
├── include/tax/              # Header-only library (the entire library lives here)
│   ├── tax.hpp               # Umbrella header — users include only this
│   ├── la.hpp                # Facade: linear-algebra / Eigen helpers (tax::la)
│   ├── ode.hpp               # Facade: ODE integration module (tax::ode, opt-in)
│   ├── ads.hpp               # Facade: Automatic Domain Splitting (tax::ads, opt-in)
│   ├── core/                 # The TaylorExpansion type and its foundations
│   │   ├── concepts.hpp      #   Scalar, TaylorPolynomial, DensePolynomial concepts
│   │   ├── multi_index.hpp   #   MultiIndex<M>, flatIndex/unflatIndex, numMonomials,
│   │   │                     #   DegreeOf<N,M> lookup table
│   │   ├── enumeration.hpp   #   forEachMonomial / forEachSubIndex
│   │   ├── storage/          #   Dense (std::array) and Sparse (sorted idx/val
│   │   │                     #   vectors) storage policies
│   │   └── taylor_expansion.hpp  # TaylorExpansion<T, N, M, Storage>:
│   │                             #   Dense + Sparse specialisations
│   ├── kernels/              # Series recurrence kernels (tax::detail::kernels)
│   │   ├── cauchy.hpp        #   cauchyProduct dispatch (+ in-header config macros)
│   │   ├── cauchy_unroll.hpp #   fully unrolled univariate (M == 1) product
│   │   ├── cauchy_stencil.hpp#   precomputed stencil table product (M >= 2)
│   │   ├── algebra.hpp       #   self-product, square/cube, reciprocal, sqrt,
│   │   │                     #   cbrt, pow (real + integer)
│   │   ├── trigonometric.hpp #   sin, cos, tan, asin, acos, atan
│   │   ├── transcendental.hpp#   exp, log, sinh, cosh, tanh + inverses, erf
│   │   ├── sparse_cauchy.hpp #   sparse Cauchy product / self-product
│   │   └── sparse_subs.hpp   #   sparse substitution helpers
│   ├── operators/            # Free-function operator surface over the kernels
│   │   ├── arithmetic.hpp    #   +, -, *, /, compound assignment (dense + sparse)
│   │   ├── math_unary.hpp    #   sin, exp, sqrt, square, …
│   │   └── math_binary.hpp   #   pow, atan2, …
│   ├── la/                   # Eigen integration (namespace tax::la)
│   │   ├── types.hpp         #   Vec, Mat, VecNT<N,T>, MatNT, MatNMT
│   │   ├── num_traits.hpp    #   Eigen::NumTraits<TaylorExpansion>
│   │   ├── values.hpp        #   variables(x0), value(), eval()
│   │   ├── derivatives.hpp   #   derivative, gradient, hessian, jacobian
│   │   └── invert.hpp        #   formal polynomial-map inversion (Picard)
│   ├── ode/                  # Adaptive ODE integration (namespace tax::ode)
│   └── ads/                  # Automatic Domain Splitting (namespace tax::ads)
├── tests/                    # Google Test suite (61 ctest targets)
│   ├── core/                 #   ctor/accessors, multi-index, enumeration, deriv/integ
│   ├── kernels/              #   dense/unroll/stencil/sparse Cauchy verification
│   ├── operators/            #   one file per math-function family
│   ├── sparse/               #   sparse ctor/arith/conversion/substitution
│   ├── eigen/                #   tax::la helpers (gradient, jacobian, invert, …)
│   ├── ode/                  #   steppers/, integrator/, events/, problems/ (CR3BP, Kepler)
│   ├── ads/                  #   box, tree, criteria, driver, merge, parallel
│   ├── regression/           #   DACE comparison suite (opt-in, TAX_BUILD_REGRESSIONS)
│   └── testUtils.hpp         #   shared helpers/macros
├── benchmarks/               # Google Benchmark suite (bench_ode_cr3bp)
├── examples/                 # two_body/, three_body/, wsb/ — Taylor, ADS, LOADS
├── docs/                     # MkDocs documentation (core/, ode/, eigen/, internals/)
├── cmake/                    # CMake package config template
├── .github/workflows/        # CI: tests.yml, sanitizers.yml, regressions.yml, docs.yml
├── .clang-format             # Code style configuration
├── pyproject.toml            # scikit-build-core wheel config (Python bindings are
│                             #   planned; no python/ sources are in the tree yet)
├── CMakeLists.txt            # Root CMake configuration
└── README.md
```

---

## Building

```bash
# Configure
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j

# Test
ctest --test-dir build --output-on-failure
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `TAX_BUILD_UNITTESTS`   | `ON`  | Build Google Test unit-test suite |
| `TAX_BUILD_REGRESSIONS` | `OFF` | Build DACE-based regression tests (fetches DACE v2.1.0) |
| `TAX_BUILD_BENCHMARK`   | `OFF` | Build Google Benchmark suite |
| `TAX_BUILD_EXAMPLES`    | `OFF` | Build example programs under `examples/` |

Kernel dispatch (`TAX_USE_UNROLL` / `TAX_USE_STENCIL`) is configured **in-header**
in `<tax/kernels/cauchy.hpp>` and defaults to ON for every consumer. It is
deliberately *not* injected from the build system: differing values across
translation units would change inline function definitions (ODR). A project may
pre-define either macro to `0`, but the value must be identical project-wide.

### Dependencies

- **Required:** Eigen3 (`find_package(Eigen3 REQUIRED)`), Threads
- **Optional:** DACE v2.1.0 — fetched via FetchContent when `TAX_BUILD_REGRESSIONS=ON`
- **Test framework:** Google Test v1.17 — fetched automatically if not found

---

## Core Concepts

### The Main Type

```cpp
tax::TaylorExpansion<T, N, M = 1, Storage = tax::storage::Dense>
// T       = scalar type (double, float)
// N       = truncation order (compile-time integer)
// M       = number of variables
// Storage = storage::Dense (std::array) or storage::Sparse (sorted idx/val vectors)
```

Convenient aliases (all `double`-valued):
```cpp
tax::TE<N, M = 1>   // dense
tax::TEn<N, M>      // dense, explicit multivariate spelling
tax::STE<N, M = 1>  // sparse
```

- **Dense:** `std::array<T, numMonomials(N, M)>` coefficients in graded-lex
  order, full `constexpr` surface, no heap. This is the hot path used by
  the ODE and ADS modules.
- **Sparse:** two parallel sorted vectors of (flat-index, value) pairs.
  Element access is O(log nnz); arithmetic is a sorted merge walk. Both
  share the same recurrence relations via the kernel layer.

### Creating Variables

```cpp
// Univariate
auto x = tax::TE<5>::variable(1.0);            // x = 1.0 + 1*dx

// Multivariate (static index)
typename tax::TE<3, 2>::Input p{1.0, 2.0};
auto x = tax::TE<3, 2>::variable<0>(p);
auto y = tax::TE<3, 2>::variable<1>(p);

// Multivariate (runtime index)
auto z = tax::TE<3, 2>::variable(1.0, /*var_idx=*/0);

// Eigen vector of coordinate variables (tax::la)
Eigen::Vector2d x0{1.0, 2.0};
auto vars = tax::la::variables<tax::TE<3, 2>>(x0);
```

### Using the Library

```cpp
#include <tax/tax.hpp>

auto x = tax::TE<5>::variable(1.0);
auto f = sin(x) * exp(x);

double val = f.value();           // constant term
double c2  = f.coeff<2>();        // raw Taylor coefficient
double d2  = f.derivative<2>();   // k!-scaled derivative
double y   = f.eval(dx);          // evaluate Taylor polynomial at x0+dx
auto   df  = f.deriv<0>();        // symbolic partial derivative
auto   F   = f.integ<0>();        // symbolic integral
```

`coeff` / `derivative` / `deriv` / `integ` all exist in compile-time
(`<...>`), `MultiIndex<M>`, and runtime-`int` forms.

### Coefficient Storage

- Graded-lexicographic ordering: all degree-0 first, then degree-1, etc.
- Size: `nCoefficients = numMonomials(N, M) = C(N+M, M)`
- `coeff` retrieves the raw Taylor coefficient; `derivative` applies `k!` scaling
- The degree-d monomials occupy the contiguous flat-index block
  `[numMonomials(d-1, M), numMonomials(d, M))` — several hot paths
  (truncation criteria, kernels) rely on this

---

## Kernels

All math operations are degree-by-degree recurrence relations in
`include/tax/kernels/` (`tax::detail::kernels`), operating directly on the
coefficient arrays. The Cauchy product has three dense variants behind one
dispatch (`cauchyProduct`):

- `cauchyProductLoop` — generic, `constexpr`-safe (used in constant evaluation)
- `cauchyProductUnroll` — pack-expansion unrolled, M == 1
- `cauchyProductStencil` — precomputed (out, a, b) index table, M >= 2;
  exactly `numMonomials(N, 2M)` entries, built once at first use
  (runtime static — not usable in constant evaluation)

All other M >= 2 recurrences (exp, log, sin/cos, tan, pow, reciprocal,
sqrt, ...) are driven by one shared decomposition table:
`forEachRecurrenceRow<N, M>(fn)` in `recurrence_stencil.hpp` hands each
kernel the precomputed (flatIndex(beta), flatIndex(gamma), |beta|) rows
per output monomial; only the recurrence weight stays in the kernel.
Constant evaluation (and `TAX_USE_STENCIL=0`) enumerates the same rows
on the fly — bit-identical results. Sparse variants live in
`sparse_cauchy.hpp` and reuse a `thread_local` scratch accumulator (no
per-call heap allocation).

When adding a new math function: implement the recurrence in the right
kernel file, expose it via `operators/math_unary.hpp` or `math_binary.hpp`,
and add tests under `tests/operators/` (plus `tests/kernels/` if it has its
own kernel).

---

## Eigen Integration (`tax::la`)

```cpp
#include <tax/tax.hpp>   // la.hpp is included by the umbrella

Eigen::Vector2d x0{1.0, 2.0};
auto v = tax::la::variables<tax::TE<3, 2>>(x0);   // Eigen vector of TE variables

auto vals = tax::la::value(f);          // constant terms
auto grad = tax::la::gradient(f);       // gradient
auto J    = tax::la::jacobian(F);       // Jacobian
auto H    = tax::la::hessian(f);        // Hessian
auto Finv = tax::la::invert(F);         // formal map inversion (Picard)
```

`num_traits.hpp` specialises `Eigen::NumTraits` so TE types work as Eigen
scalars (`tax::la::VecNT<D, TE>` is the ODE/ADS state type).

---

## ODE Integration (`tax::ode`)

Adaptive Runge–Kutta and Taylor-method integration with an event system
(triggers/actions), dense output, and step-size controllers.

```cpp
#include <tax/ode.hpp>
using namespace tax::ode::methods;

auto rhs = [](const auto& x, double) { /* return state-shaped value */ };
tax::la::VecNT<2, double> x0{1.0, 0.0};

auto sol   = tax::ode::propagate(Verner89{}, rhs, x0, 0.0, 2 * M_PI);
auto sol_d = tax::ode::propagate</*Dense=*/true>(Taylor<16>{}, rhs, x0, 0.0, 2 * M_PI);
auto x_at  = sol_d(0.42);   // dense output
```

Methods: `methods::Taylor<N>`, `Verner78`, `Verner89`, `Fehlberg78`,
`Feagin12`, `Feagin14`.

Key files:

| File | Purpose |
|------|---------|
| `propagate.hpp`   | `propagate<Dense>(method, rhs, x0, t0, t1, cfg, events)` + `methods::` tags |
| `integrator.hpp`  | `Integrator<Stepper, F, Dense>` driver + per-method `Integrator` aliases |
| `config.hpp`      | `IntegratorConfig<T>` (abstol/reltol/initial/min/max step, max_steps, max_rejects_per_step) |
| `controllers.hpp` | `I`, `PI` (Gustafsson), `H211b` (Söderlind), `JorbaZou` (Taylor-only), `FixedStep` |
| `steppers/`       | `taylor.hpp` + five thin per-method headers; all embedded RK methods are aliases of `detail::EmbeddedRKStepper<Tab, State, Controller>` (`detail/embedded_rk_stepper.hpp`) over their Butcher tableaus (`detail/*_tableaus.hpp`) |
| `solution.hpp`    | `Solution<Stepper, State, Dense>` (dense output via `operator()`) |
| `event.hpp`       | `Event<Stepper>`, `TriggerContext`, `StepperCtx`, `ControlFlow` |
| `triggers.hpp`    | `EveryStep`, `ZeroCrossing` (poly-Newton or Brent) |
| `actions.hpp`     | `Continue`, `Terminate`, `Record`, `Custom`, `EventStorage` |
| `vector_ops.hpp`  | `VectorOps<S>` trait — norm/axpy/scale_assign (+ optional `diff_norm`) for scalar / TE / Eigen states |
| `io.hpp` (opt-in) | `linspace`, `writeCsv(sol, times, path)` |

---

## Automatic Domain Splitting (`tax::ads`)

Implements Wittig 2015 ADS and the LOADS variant (Losacco/Fossà/Armellin
2024) by composing with the `tax::ode` event infrastructure.

```cpp
#include <tax/ads.hpp>
#include <tax/ode.hpp>
using namespace tax::ode::methods;

tax::ads::Box<double, 2> ic_box{center_vec, half_width_vec};

auto tree = tax::ads::propagate</*P=*/6>(
    Verner89{},
    tax::ads::TruncationCriterion{/*tol=*/1e-4, /*maxDepth=*/8},   // or NliCriterion (LOADS)
    rhs, ic_box, ic_center, 0.0, 2 * M_PI, cfg);

for (int i : tree.done()) { const auto& l = tree.leaf(i); /* l.payload */ }

auto stats = tax::ads::merge(tree, tax::ads::TruncationCriterion{1e-4});  // post-pass merger
```

Architecture:
- **Leaf-only arena tree** (`AdsTree<Payload, M, T>`): `std::vector<Leaf>` with
  `parentIdx`/`siblingIdx` links; splits retire the parent in place, merges revive it.
- **Event interop**: `(SplitTrigger, SplitAction)` appended to the user's event
  list; the action records `{dim, t}` and terminates; the driver splits or marks done.
- **Splits at accepted-step boundaries only** (matches Wittig's original ADS).
- **Parallel driver**: `AdsDriver` runs `num_threads` jthread workers; integration
  is lock-free on moved-out leaf inputs, the mutex guards only queue/tree mutation.
- **Axis substitution** (`da_state.hpp::detail::substituteAxis`): one routine
  handles both split (`scale = 0.5`) and merge-inverse (`scale = 2`).

Key files: `box.hpp`, `leaf.hpp`, `tree.hpp`, `criteria.hpp` (`SplitCriterion`
concept, `TruncationCriterion`, `NliCriterion`), `nonlinearity_index.hpp`,
`split_event.hpp`, `da_state.hpp`, `driver.hpp`, `propagate.hpp`, `merge.hpp`,
`io.hpp` (opt-in CSV writers).

---

## Code Conventions

### Naming

| Category | Convention | Examples |
|----------|-----------|---------|
| Types/Classes | `PascalCase` | `TaylorExpansion`, `MultiIndex`, `AdsTree`, `Integrator` |
| Template params | `UPPERCASE` or short | `T`, `N`, `M`, `D`, `Tab`, `Derived` |
| Free functions & methods | `camelCase` | `variable()`, `flatIndex()`, `seriesReciprocal()`, `deriv()`, `popFront()` |
| Local variables | `snake_case` | `n_coeff`, `dx`, `half_width` |
| Namespaces | `lowercase` | `tax`, `tax::detail`, `tax::ode`, `tax::ads`, `tax::la` |
| Type aliases | Short uppercase | `TE<N, M>`, `TEn<N, M>`, `STE<N, M>` |

### C++ Patterns

- **`constexpr` everywhere in core:** size calculations, index mappings, and
  coefficient operations must stay `constexpr`; kernels that use runtime
  statics (stencil) must keep a `constexpr`-safe fallback behind `if !consteval`
- **`noexcept` on all operations** (exception: methods that `throw`, e.g.
  runtime-index `deriv(int)` and the `Integrator` config validation)
- **No heap allocation in core:** `std::array` for dense storage;
  `std::vector` is acceptable in Sparse storage and the ODE/ADS modules
- **Concepts over SFINAE:** `tax::Scalar`, `TaylorPolynomial`,
  `DensePolynomial`, `tax::ode::concepts::Stepper`, `tax::ads::SplitCriterion`
- **`if constexpr`** for univariate (M == 1) vs multivariate branches
- **`[[nodiscard]]`** on accessors, computation results, expensive operations
- **Internal details in `detail` namespaces:** `tax::detail::kernels`,
  `tax::ode::detail`, `tax::ads::detail`

### Formatting

Enforced by `.clang-format` (Google style, customized):
- Indent: **4 spaces** (no tabs)
- Column limit: **100 characters**
- Brace wrapping: new line after class/struct/function/namespace/control statements
- Spaces inside parentheses and angle brackets: `TaylorExpansion< T, N, M >`

```bash
clang-format -i $(git ls-files 'include/**/*.hpp')
```

---

## Testing

Tests are organized by module, one `.cpp` per concern, each registered via
`tax_add_test(name SOURCES path.cpp)` in `tests/CMakeLists.txt` (61 ctest
targets).

```cpp
#include <gtest/gtest.h>
#include <tax/tax.hpp>

TEST(Trig, SinUnivariateOrder3)
{
    auto x = tax::TE<3>::variable(0.0);
    auto f = sin(x);
    EXPECT_NEAR(f[1], 1.0, 1e-12);
    EXPECT_NEAR(f[3], -1.0 / 6.0, 1e-12);
}
```

```bash
ctest --test-dir build --output-on-failure
./build/tests/test_trig          # single executable
```

DACE-comparison regression tests live in `tests/regression/` and only build
with `-DTAX_BUILD_REGRESSIONS=ON`.

---

## CI/CD

- **`tests.yml`** — push/PR: build + unit tests across the support matrix
- **`regressions.yml`** — DACE comparison suite
- **`sanitizers.yml`** — ASAN/UBSAN/TSAN jobs
- **`docs.yml`** — MkDocs documentation build/deploy

### Before Submitting a PR

1. All ctest targets pass locally
2. Code is formatted with `clang-format`
3. No new dynamic allocations introduced in the core library
4. New math operations have kernel tests AND operator tests
5. ODE/ADS changes have tests in `tests/ode/` or `tests/ads/`

---

## Common Pitfalls

- **Do not heap-allocate in core:** dense `TaylorExpansion` must remain
  allocation-free; `std::vector` belongs to Sparse storage and ODE/ADS only
- **Do not break `constexpr`:** all index arithmetic stays compile-time; if a
  fast path needs runtime statics, guard it with `if !consteval` and keep the
  loop kernel as the constant-evaluation fallback
- **Graded-lex ordering is sacred:** the coefficient order (`flatIndex`) is
  relied on everywhere — including contiguous-degree-block tricks — never change it
- **Kernel config macros are in-header:** never re-introduce
  `TAX_USE_UNROLL`/`TAX_USE_STENCIL` as build-system definitions (ODR hazard)
- **Sparse invariants:** the idx/val vectors are sorted and deduplicated;
  kernels and operators that append directly (`rawIndices()/rawValues()`)
  must emit in ascending flat-index order with no zeros
- **M = 0 is invalid:** always assert or `static_assert` M >= 1
- **Include the umbrella headers:** `<tax/tax.hpp>` (core + la),
  `<tax/ode.hpp>`, `<tax/ads.hpp>` — not individual sub-headers
- **`pyproject.toml` is forward-looking:** there are no Python binding sources
  in the tree yet
