# Kernels & Recurrences

The kernel layer is where the math lives in code. Each function in
`tax/kernels/` implements one of the degree-by-degree recurrence relations
from [Mathematical Foundations](../core/math.md), operating on raw coefficient
buffers rather than the user-facing `TaylorExpansion` type. The operator layer
in `tax/operators/` is a thin wrapper that calls a kernel and returns a fresh
`TaylorExpansion`.

---

## File map

| Header | Kernels | Operations |
|---|---|---|
| `tax/kernels/cauchy.hpp` | `seriesCauchy`, `seriesCauchyAccumulate`, `seriesSquare`, `seriesCube` | multiplication, integer powers |
| `tax/kernels/cauchy_unroll.hpp` | unrolled `M==1` Dense Cauchy | tight univariate hot path (`TAX_USE_UNROLL`) |
| `tax/kernels/cauchy_stencil.hpp` | precomputed stencil-driven `M≥2` Dense Cauchy | multivariate hot path (`TAX_USE_STENCIL`) |
| `tax/kernels/algebra.hpp` | `seriesReciprocal`, `seriesSqrt`, `seriesCbrt`, `seriesSquare`, `seriesCube` | algebraic recurrences |
| `tax/kernels/trigonometric.hpp` | coupled `seriesSinCos`, `seriesTan`, inverse trig | trig and inverse trig |
| `tax/kernels/transcendental.hpp` | `seriesExp`, `seriesLog`, `seriesSinhCosh`, `seriesTanh`, `seriesErf`, `seriesPow` | exp/log/hyperbolic/erf/power |
| `tax/kernels/sparse_cauchy.hpp` | `seriesCauchy` for the sparse storage | multiplication on sparse polynomials |
| `tax/kernels/sparse_subs.hpp` | shared sparse subroutines (add/sub merges, etc.) | sparse arithmetic helpers |

---

## Univariate vs multivariate dispatch

Every recurrence picks one of two paths via `if constexpr (M == 1)`:

- **Univariate** (`M == 1`) — scalar loops over flat indices. Simple,
  branchless, easy to vectorise. The unrolled Dense Cauchy variant
  (`TAX_USE_UNROLL`) is what runs here.
- **Multivariate** (`M ≥ 2`) — routes through
  `forEachSubIndex<M>(alpha, lo, hi, callback)`, which enumerates all
  sub-multi-indices \(\beta \le \alpha\) with \(\text{lo} \le |\beta| \le \text{hi}\)
  and calls `callback(flatIndex(beta), flatIndex(alpha - beta), |beta|)`. The
  callback shape is exactly what every recurrence in
  [Mathematical Foundations](../core/math.md) needs.

For Dense `M ≥ 2`, `TAX_USE_STENCIL=ON` (default) precomputes the
sub-multi-index enumeration once at compile time and reuses the stencil across
every Cauchy invocation. The non-stencil path is still in the tree and cross-
validated by `tests/kernels/testCauchyStencilDiff`.

---

## Recurrence shape — anatomy of `seriesExp`

For \(g = \exp(f)\) the closed-form recurrence is

\[
g_\alpha \;=\; \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| \le |\alpha|}} |\beta| \, f_\beta \, g_{\alpha - \beta},
\qquad g_0 = \exp(f_0)
\]

which the kernel implements (in pseudocode) as

```cpp
g[0] = std::exp(f[0]);
for (int d = 1; d <= N; ++d) {
    for_each_alpha_of_degree<M>(d, [&](MultiIndex<M> alpha) {
        T sum{};
        forEachSubIndex<M>(alpha, /*lo=*/1, /*hi=*/d, [&](
            flat_index_t fb, flat_index_t fab, int beta_deg)
        {
            sum += T(beta_deg) * f[fb] * g[fab];
        });
        g[flat_index(alpha)] = sum / T(d);
    });
}
```

Every other recurrence has the same skeleton: initialise the
constant-term value with the scalar math, then walk degrees \(d = 1, \ldots, N\)
applying the appropriate sub-multi-index sum.

For *coupled* recurrences (sin/cos, sinh/cosh) the kernel maintains both
arrays in lockstep — see `seriesSinCos`. For *helper-driven* recurrences
(asin via \(h = \sqrt{1 - f^2}\), tan via \(\cos \cdot t = \sin\)) the helper is
materialised first by an inner kernel call, then the outer recurrence runs.

---

## Symmetric exploitation

A few recurrences appear with explicit factor-of-two savings:

- `seriesSquare` enumerates only unordered pairs \((\beta, \alpha - \beta)\)
  with \(\beta \le \alpha - \beta\) in flat-index order, doubling pairs and
  counting diagonals once.
- `seriesCbrt` maintains \(q = g^2\) incrementally, dropping the worst-case
  cost from \(\mathcal{O}(N^3)\) to \(\mathcal{O}(N^2)\).
- `seriesSqrt` uses the same symmetric self-product enumeration as
  `seriesSquare` inside its outer recurrence.

The symmetries don't change the math — they're computational identities that
follow directly from the closed-form recurrence and the graded-lexicographic
ordering of the flat index.

---

## Sparse Cauchy

`sparse_cauchy` exploits the parallel-sorted-vector representation of
`storage::Sparse`. For inputs with `nnz_a` and `nnz_b` nonzeros, the kernel
enumerates pairs in roughly \(\mathcal{O}(\text{nnz}_a \cdot \text{nnz}_b)\)
operations (rather than the dense \(\mathcal{O}(\binom{N+M}{M}^2)\)) and
inserts each contribution into a result buffer keyed by flat index. The result
is materialised into a sorted-index representation at the end.

Sparse add/sub uses the standard two-pointer merge over the sorted index
arrays — \(\mathcal{O}(\text{nnz}_a + \text{nnz}_b)\).

---

## Tests as the canonical executable spec

Every kernel has a paired test in `tests/kernels/`:

| Test | What it pins |
|---|---|
| `test_cauchy_dense.cpp`         | Dense `cauchy` against a direct convolution baseline |
| `test_cauchy_unroll_diff.cpp`   | Unrolled vs non-unrolled univariate Cauchy agree |
| `test_cauchy_stencil_diff.cpp`  | Stencil vs non-stencil multivariate Cauchy agree |
| `test_sparse_cauchy.cpp`        | Sparse Cauchy against the Dense Cauchy reference |

Treat these tests as the canonical executable specification: when in doubt
about a recurrence's expected behavior, they show inputs and the exact
expected outputs.
