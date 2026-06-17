# Core Module

The core module provides truncated multivariate Taylor polynomials as
first-class C++ objects. A single evaluation pass through any supported
mathematical expression yields the function value and all partial derivatives up
to a compile-time truncation order $N$.

The library uses **expression templates** for lazy evaluation: arithmetic and
transcendental operations build a lightweight expression tree that is
materialised only on assignment, eliminating intermediate temporary objects and
enabling automatic sum/product flattening.

The central type, `TaylorExpansion<T, N, M, Storage>`, stores
$\binom{N+M}{M}$ Taylor coefficients in **graded-lexicographic order**. With
`Storage = Dense` the coefficients live in a fixed-size `std::array` with zero
heap allocation; with `Storage = Sparse` only nonzero monomials are stored as
two parallel sorted vectors.

A comprehensive set of mathematical functions — trigonometric, hyperbolic,
transcendental, algebraic, and the error function — is implemented via
degree-by-degree recurrence relations, supporting both univariate and
multivariate expansions.

## Pages in this section

| Page | Topic |
|---|---|
| [Mathematical Foundations](math.md) | Coefficient storage, recurrence relations, univariate vs multivariate |
| [API Reference](api.md) | Complete signature reference for `TaylorExpansion` |
| [Examples](examples.md) | Worked examples — variables, expressions, derivatives |
| [Dense vs Sparse Storage](storage.md) | When to use `TE` vs `STE`, performance trade-offs |
| [Named Expansions](named.md) | `NamedTaylorExpansion` — axes addressed by name, compose/slice/derive |

## Key headers

| Header | Contents |
|---|---|
| `tax/tax.hpp` | Umbrella header — the only include users need |
| `tax/core/taylor_expansion.hpp` | Primary `TaylorExpansion` class template + Dense/Sparse specialisations |
| `tax/core/named.hpp` | `NamedTaylorExpansion` — Taylor expansions with named, type-level axes |
| `tax/core/multi_index.hpp` | `MultiIndex<M>`, flat-index ↔ multi-index conversion |
| `tax/core/enumeration.hpp` | Compile-time monomial enumeration utilities |
| `tax/core/concepts.hpp` | `Scalar` concept and related traits |
| `tax/operators/arithmetic.hpp` | `+`, `-`, `*`, `/` between TE and scalars |
| `tax/operators/math_unary.hpp` | `sin`, `cos`, `exp`, `log`, `sqrt`, … |
| `tax/operators/math_binary.hpp` | `pow`, `atan2`, … |
| `tax/kernels/cauchy.hpp` | Cauchy product (multiplication) |
| `tax/kernels/algebra.hpp` | reciprocal, sqrt, cbrt, square, cube |
| `tax/kernels/trigonometric.hpp` | trigonometric recurrences |
| `tax/kernels/transcendental.hpp` | exp, log, hyperbolic, inverse, erf |
| `tax/eigen.hpp` | `NumTraits` + Eigen-vocabulary helpers |
