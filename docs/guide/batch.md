# Batch (SIMD) coefficients

A `TaylorExpansion` is generic in its coefficient type. Besides `double` /
`float`, the library ships a **batched** coefficient type that packs `K`
independent problem instances into every coefficient slot:

```cpp
tax::Batch<T, K>   // K lanes of floating-point scalar T (Eigen-backed)
```

Substituting it for the scalar coefficient makes a single expansion carry `K`
independent expansions that share the same monomial structure. The unified `TE`
alias selects it through a trailing lane-count parameter:

```cpp
tax::TE<N, M, K>   // = TaylorExpansion<Batch<double, K>, N, M>; K defaults to 1 (plain double)
```

All recurrence kernels run **once** and produce all `K` results, with the inner
element-wise work vectorised by Eigen / the compiler. This is ideal for
ensemble / Monte-Carlo propagation, parameter sweeps, and ADS sub-boxes that
share an expansion shape.

## Aliases

| Alias | Meaning |
|-------|---------|
| `tax::Batch<T, K>`   | `K`-lane coefficient with lane scalar `T` |
| `tax::Batchd<K>`     | `Batch<double, K>` |
| `tax::Batchf<K>`     | `Batch<float, K>` |
| `tax::TE<N, M, K>`   | `TaylorExpansion<Batch<double, K>, N, M>` (`M`, `K` default to 1) |

## Usage

```cpp
#include <tax/tax.hpp>

constexpr int N = 6, M = 1, K = 4;
using TE = tax::TE<N, M, K>;

// Per-lane expansion centre: lanes evaluate four different problems at once.
typename TE::Input p{};
tax::Batch<double, K> x0;
for (int k = 0; k < K; ++k) x0[k] = 0.2 + 0.05 * k;
p[0] = x0;

auto x = TE::variable<0>(p);
auto f = sin(x) * exp(x);          // whole math surface works

double lane2_value = f.value()[2]; // read a single lane
```

The full unary/binary math surface is supported: `+ - * /`, `sqrt`, `cbrt`,
`exp`, `log`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, the hyperbolics and
their inverses, `erf`, `pow` (integer and real exponent), and `atan2`. Every
lane is bit-for-bit identical to the equivalent scalar `TaylorExpansion`
computation.

`Eigen::NumTraits<Batch<...>>` is specialised, so batched expansions work as
Eigen scalars: `tax::la::value`, `gradient`, `jacobian`, and
`tax::la::VecNT<D, tax::TE<N, M, K>>` states all behave as expected (per lane).

## Notes & limits

- **Dense storage only.** Sparse storage keys off exact-zero coefficients,
  which is not well defined per lane.
- **Real-exponent `pow`.** `pow(x, 2.5)` selects the real-exponent kernel and
  `pow(x, 3)` the integer one, exactly as for scalar coefficients.
- **`Batch` is a runtime SIMD type** (Eigen-backed), so a batched expansion is
  not usable in `constexpr` evaluation, unlike a scalar one. The compile-time
  accessors that fold in a `k!` factor — `derivative<Alpha...>()`, `coeff<...>()`
  evaluated in a constant context — are therefore the scalar-only surface; use
  the **runtime** `derivative(MultiIndex)` / `coeff(...)` overloads with a batched
  expansion (they work per lane).
- **Not streamable.** `operator<<` / `tax::series` are not defined for batched
  expansions (a `Batch` has no single textual value). Read individual lanes via
  `f.value()[k]`, `f.coeff(...)[k]`, or `f[i][k]`.
- The two enabling core hooks are the `is_tax_scalar` / `real_scalar` traits in
  `core/concepts.hpp`; any user type that presents the same element-wise math
  surface can opt in the same way.
