# Kernels & Recurrences

The kernel layer is where the math lives in code. Each function in
`tax/kernels/` implements one of the degree-by-degree recurrence relations
from [Recurrence Relations](recurrences.md), operating on raw coefficient
buffers rather than the user-facing `TaylorExpansion` type. The operator layer
in `tax/operators/` is a thin wrapper that calls a kernel and returns a fresh
`TaylorExpansion`.

---

## File map

| Header | Kernels | Operations |
|---|---|---|
| `tax/core/cmath.hpp` | `ctExp`, `ctSin`, `ctSqrt`, … (`tax::detail::cmath`) | constexpr constant-term seeds — see [below](#constexpr-constant-term-seeding) |
| `tax/kernels/cauchy.hpp` | `cauchyProduct` dispatch (loop / unroll / stencil), `cauchySelfProduct` | multiplication, squares |
| `tax/kernels/cauchy_unroll.hpp` | unrolled `M==1` Dense Cauchy | tight univariate hot path (`TAX_USE_UNROLL`) |
| `tax/kernels/cauchy_stencil.hpp` | precomputed stencil-driven `M≥2` Dense Cauchy | multivariate hot path (`TAX_USE_STENCIL`) |
| `tax/kernels/recurrence_stencil.hpp` | `RecurrenceStencil`, `forEachRecurrenceRow` | shared decomposition table driving every `M≥2` recurrence |
| `tax/kernels/algebra.hpp` | shared drivers `seriesDerivQuotient` / `seriesDerivProduct`; `seriesSquare`, `seriesCube`, `seriesReciprocal`, `seriesDivide`, `seriesSqrt`, `seriesCbrt`, `seriesPow`, `seriesPowInt` | recurrence drivers + algebraic recurrences and powers |
| `tax/kernels/trigonometric.hpp` | coupled `seriesSinCos`, `seriesTan`, `seriesAsin`/`seriesAcos`/`seriesAtan`/`seriesAtan2` | trig and inverse trig |
| `tax/kernels/transcendental.hpp` | `seriesExp`, `seriesLog`, `seriesSinhCosh` (+ single-output `seriesSinh`/`seriesCosh`), `seriesTanh`, inverse hyperbolics, `seriesErf` | exp/log/hyperbolic/erf |
| `tax/kernels/fused.hpp` | `seriesExpSinCos` (+ `seriesExpSin`/`seriesExpCos`), `seriesSqrtInvSqrt` | pair-fused exp·trig and sqrt/invsqrt |
| `tax/kernels/mixed_stencils.hpp` | Cauchy / recurrence stencils for `MixedScheme` | mixed-order (anisotropic) layouts |
| `tax/kernels/sparse_cauchy.hpp` | sparse Cauchy product / self-product | multiplication on sparse polynomials |
| `tax/kernels/sparse_subs.hpp` | shared sparse subroutines (add/sub merges, etc.) | sparse arithmetic helpers |

---

## Univariate vs multivariate dispatch

Every recurrence picks one of two paths via `if constexpr (M == 1)`:

- **Univariate** (`M == 1`) — scalar loops over flat indices. Simple,
  branchless, easy to vectorise. The unrolled Dense Cauchy variant
  (`TAX_USE_UNROLL`) is what runs here.
- **Multivariate** (`M ≥ 2`) — routes through
  `forEachRecurrenceRow<N, M>(fn)`, which calls
  `fn(ai, d, row)` for every output flat index `ai` of degree `d ≥ 1` in
  graded-lex order, where `row` is the span of all decompositions
  $(\beta, \gamma)$ with $\beta + \gamma = \alpha$ and $|\beta| \ge 1$ —
  each entry carrying precomputed `flatIndex(beta)`, `flatIndex(gamma)`
  and $|\beta|$. That row shape is exactly what every recurrence in
  [Recurrence Relations](recurrences.md) needs; the weight
  ($1$, $|\beta|$, $|\gamma|$, $c|\beta| - |\gamma|$, …) stays in the kernel.

With `TAX_USE_STENCIL=1` (the in-header default) the rows come from a
`RecurrenceStencil<N, M>` table built once at first use — no multi-index
arithmetic remains in the inner loops. The general Cauchy product uses its
own flat `CauchyStencil` table. With the macro off — and always in constant
evaluation — the rows are enumerated on the fly in the same order, so the
two paths are bit-identical; the agreement is pinned by
`tests/kernels/test_cauchy_stencil_diff.cpp`.

---

## Recurrence shape — anatomy of `seriesExp`

For $g = \exp(f)$ the closed-form recurrence is

$$
g_\alpha \;=\; \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| \le |\alpha|}} |\beta| \, f_\beta \, g_{\alpha - \beta},
\qquad g_0 = \exp(f_0)
$$

which the shared driver implements (in pseudocode) as

```cpp
g[0] = cmath::ctExp(f[0]);      // constexpr-safe constant-term seed
forEachRecurrenceRow<N, M>([&](std::size_t ai, int d,
                               std::span<const RecurrenceEntry> row) {
    T sum{};
    for (const RecurrenceEntry& e : row)
        sum += T(e.db) * f[e.b_idx] * g[e.g_idx];
    g[ai] = sum / T(d);
});
```

### Two shared drivers

That skeleton is not repeated per kernel. Two drivers in
`tax/kernels/algebra.hpp` implement the common recurrence shapes once
(univariate loop + multivariate row walk):

- **`seriesDerivProduct`** solves $\text{out}' = \text{src}' \cdot h$ —
  the exp shape. `seriesExp` passes $h = \text{out}$ itself (rows read $h$
  only at strictly lower, already-final degree); `seriesErf` passes
  $h = \tfrac{2}{\sqrt\pi} e^{-f^2}$.
- **`seriesDerivQuotient`** solves $h \cdot \text{out}' = \pm\text{src}'$ —
  the log / inverse-trig / inverse-hyperbolic shape (`log`, `asin`, `acos`,
  `atan`, `atan2`, `asinh`, `acosh`, `atanh`).

Most transcendental kernels therefore reduce to three lines: *compute $h$,
seed the constant term, call the driver*. The math for both shapes, with
the per-function helper table, is in
[Recurrence Relations](recurrences.md#shared-recurrence-drivers).

For *coupled* recurrences (sin/cos, the fused exp·trig pass) the kernel
maintains both arrays in lockstep — see `seriesSinCos` and
`seriesExpSinCos`. For *helper-driven* recurrences
(asin via $h = \sqrt{1 - f^2}$, tan via $\cos \cdot t = \sin$) the helper is
materialised first by an inner kernel call, then the driver runs.

---

## Fused pair kernels

`tax/kernels/fused.hpp` holds the two kernels ported from the
expression-template prototype branch. The ET prototype's own benchmarks
showed its lazy-evaluation layer was a wash against eager evaluation, but
that two *fusions* were robust wins — those kernels, and only those, were
ported; the ET layer itself was deliberately not:

- **`seriesExpSinCos`** — $h = e^v\cos u$ and $q = e^v\sin u$ in one
  coupled pass ($h' = v'h - u'q$, $q' = v'q + u'h$) instead of three
  recurrences plus a Cauchy product. Exposed as `expSin`, `expCos`,
  `expSinCos`; the fused pass is roughly 1.4–1.8x faster than composing
  `exp(v) * cos(u)`.
- **`seriesSqrtInvSqrt`** — $s = \sqrt{u}$ and $r = 1/\sqrt{u}$ interleaved
  per degree; $r$ costs one forward substitution on top of $s$, with scalar
  divisions by $s_0$ only. A single-output caller should use `seriesSqrt` /
  `seriesPow` instead: computing the unused companion is a measured net
  loss.

The public pair-returning surface (`sinCos`, `sinhCosh`, `sqrtInvSqrt`,
`expSinCos` — dense, named, and mixed) lives in
`tax/operators/math_fused.hpp`; see
[Guide / Fused Operations](../guide/fused.md).

---

## Constexpr constant-term seeding

Every series kernel evaluates exactly one scalar transcendental — the
constant-term seed (`out[0] = exp(a[0])`, …). `<cmath>` is not constexpr in
C++23, so the seeds go through the `ct*` dispatchers of
`tax::detail::cmath` (`tax/core/cmath.hpp`): at runtime each dispatcher
forwards to `std::`/ADL exactly as before (custom scalar-like coefficient
types behave unchanged), and inside `if consteval` it switches to a
constexpr implementation computed in `long double`.

The accuracy contract, restated from the header:

- The wide intermediate precision absorbs the truncation error of the
  internal series, so **compile-time results agree with the runtime libm to
  within a few ulp of `double`** — usually the last ulp or exactly.
- They are **not guaranteed bit-identical**: an expansion built in a
  constant expression may differ from the same expansion built at runtime
  in the trailing ulp of each coefficient.
- Trigonometric argument reduction is plain extended precision (no
  Payne–Hanek): constant terms of huge magnitude ($|x| \gtrsim 10^{15}$)
  lose accuracy, and $|x| \ge 2^{62}$ returns NaN.

This is what makes the whole dense surface constexpr end-to-end
(`tests/core/test_constexpr.cpp` runs entire transcendental pipelines in
`static_assert`). When adding a kernel, keep it constexpr: seed through
`cmath::ctExp`-style dispatchers, never a bare `std::exp`.

---

## Symmetric exploitation

A few recurrences appear with explicit factor-of-two savings:

- `cauchySelfProduct` (univariate, and multivariate in constant evaluation)
  enumerates only unordered pairs $(\beta, \alpha - \beta)$, doubling pairs
  and counting diagonals once. At runtime for `M ≥ 2` it routes through the
  stencil-driven general product instead — the table walk's elimination of
  per-pair index arithmetic outweighs the halved multiplication count.
- `seriesCbrt` maintains $q = g^2$ incrementally, dropping the worst-case
  cost from $\mathcal{O}(N^3)$ to $\mathcal{O}(N^2)$.
- `seriesSqrt`'s ordered row walk needs no $|\beta| < d$ filter: the
  $\beta = \alpha$ entry reads `out[ai]`, which is still zero when the row
  is processed.

The symmetries don't change the math — they're computational identities that
follow directly from the closed-form recurrence and the graded-lexicographic
ordering of the flat index.

---

## Sparse Cauchy

`sparse_cauchy` exploits the parallel-sorted-vector representation of
`storage::Sparse`. For inputs with `nnz_a` and `nnz_b` nonzeros, the kernel
enumerates pairs in roughly $\mathcal{O}(\text{nnz}_a \cdot \text{nnz}_b)$
operations (rather than the dense $\mathcal{O}(\binom{N+M}{M}^2)$) and
inserts each contribution into a result buffer keyed by flat index. The result
is materialised into a sorted-index representation at the end.

Sparse add/sub uses the standard two-pointer merge over the sorted index
arrays — $\mathcal{O}(\text{nnz}_a + \text{nnz}_b)$.

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
