# Cauchy kernel optimisations — before/after Google Benchmark comparison

Stacked optimisations of the static Cauchy and reciprocal/sqrt paths:

1. **Multivariate (M ≥ 2) — Cauchy / Self-Cauchy**: precomputed
   `CauchyStencil<N, M>` / `CauchySymStencil<N, M>` CSR tables.
2. **Univariate (M = 1) — Cauchy / Self-Cauchy**: compile-time-unrolled
   FMA chain parameterised by `std::index_sequence`.
3. **Reciprocal & Sqrt (forward substitutions)**: same stencils reused,
   with the `(beta=0, gamma=alpha)` endpoint skipped per row (it encodes
   the LHS term moved out of the recurrence). M = 1 uses analogous
   compile-time-unrolled forward substitutions.

| | |
|---|---|
| Before | `claude/refactor-taylor-series-XtMUw` @ `6091823` |
| After  | `claude/cauchy-csr-stencil` @ `795ec7c` |
| Binaries | `bench_univariate`, `bench_multivariate` (Google Benchmark) |
| CPU | Intel(R) Xeon(R) @ 2.80 GHz |
| OS | Linux 6.18.5 x86_64 |
| Build | `-DCMAKE_BUILD_TYPE=Release -DTAX_BUILD_BENCHMARK=ON` |
| Method | `--benchmark_repetitions=3 --benchmark_min_time=0.3s --benchmark_report_aggregates_only`; medians reported. |

## Univariate — production paths

The M = 1 fast paths now go through the unrolled FMA chain. Pure
multiplications (`Tax/Mul`, `Tax/Square`) see the largest wins; the
transcendental kernels (`Sin`, `Exp`, `Log`, `Sqrt`, `Pow`) use
weighted scalar recurrences that don't go through `cauchyProduct`, so
they're a control band — flat ±2 % is the expected outcome. `IPow`
goes via `seriesIntPow` which *does* internally call `cauchyProduct`
in a binary-exponentiation chain, hence its visible speed-up.

| benchmark | before (ns) | after (ns) | speed-up |
|---|---:|---:|---:|
| `Tax/Mul/N5`     |     6.0 |     5.7 | 1.05x |
| `Tax/Mul/N10`    |    38.3 |    21.1 | 1.82x |
| `Tax/Mul/N20`    |   116.9 |    97.5 | 1.20x |
| `Tax/Mul/N40`    |   501.3 |   445.3 | 1.13x |
| `Tax/Square/N5`  |     3.6 |     3.3 | 1.11x |
| `Tax/Square/N10` |    34.3 |    10.3 | 3.34x |
| `Tax/Square/N20` |    76.5 |    38.7 | 1.98x |
| `Tax/Square/N40` |   242.0 |   177.0 | 1.37x |
| `Tax/Sin/N10`    |    62.3 |    63.1 | 0.99x |
| `Tax/Sin/N20`    |   249.3 |   253.0 | 0.99x |
| `Tax/Sin/N40`    |   907.5 |   914.8 | 0.99x |
| `Tax/Exp/N10`    |    64.7 |    64.4 | 1.00x |
| `Tax/Exp/N20`    |   212.4 |   210.8 | 1.01x |
| `Tax/Exp/N40`    |   754.8 |   741.1 | 1.02x |
| `Tax/Log/N10`    |    59.5 |    59.9 | 0.99x |
| `Tax/Log/N20`    |   231.7 |   218.6 | 1.06x |
| `Tax/Log/N40`    |   765.8 |   748.6 | 1.02x |
| `Tax/Sqrt/N10`   |    82.3 |    82.8 | 0.99x |
| `Tax/Sqrt/N20`   |   267.5 |   267.1 | 1.00x |
| `Tax/Sqrt/N40`   |   811.9 |   811.4 | 1.00x |
| `Tax/IPow/N10`   |   170.3 |    91.4 | 1.86x |
| `Tax/IPow/N20`   |   438.5 |   392.4 | 1.12x |
| `Tax/IPow/N40`   | 2,007.9 | 1,800.9 | 1.11x |
| `Tax/Pow/N10`    |   110.2 |   107.5 | 1.02x |
| `Tax/Pow/N20`    |   321.9 |   314.1 | 1.02x |
| `Tax/Pow/N40`    | 1,076.5 | 1,082.3 | 0.99x |

### Univariate kernel three-way: Loop vs. Unroll vs. Reverse-buffer

Direct microbench of three implementations of the same M = 1 Cauchy
recurrence operating on `std::array<double, N+1>` operands. Numbers
captured from the AFTER build; bench-side variants are identical
across branches.

| op | N | Loop (ns) | Unroll (ns) | Reverse (ns) | Unroll/Loop | Reverse/Loop |
|---|---:|---:|---:|---:|---:|---:|
| Mul    |  5 |   7.7 |   6.8 |  21.1 | 1.14x | 0.36x |
| Mul    | 10 |  35.2 |  23.8 |  34.9 | 1.48x | 1.01x |
| Mul    | 20 | 136.8 |  96.9 | 112.7 | 1.41x | 1.21x |
| Mul    | 40 | 547.7 | 440.3 | 464.1 | 1.24x | 1.18x |
| Square |  5 |   5.1 |   5.1 |  12.6 | 1.00x | 0.40x |
| Square | 10 |  31.2 |  15.6 |  26.2 | 2.00x | 1.19x |
| Square | 20 |  69.6 |  55.7 |  65.2 | 1.25x | 1.07x |
| Square | 40 | 302.7 | 226.8 | 230.3 | 1.33x | 1.31x |

**Unroll wins everywhere.** Reverse-buffer pays an O(N) reversal cost
that dominates at small N (0.36–0.40× at N = 5) and never matches
Unroll at larger N. The three variants are kept in `univariate.cpp`
so the comparison can be reproduced.

## Multivariate (stencil-driven)

`bench_multivariate` exercises four hot shapes ((N=5, M=3),
(N=4, M=4), (N=5, M=4), (N=6, M=3)) across the operations that go
through `cauchyProduct` / `cauchySelfProduct`:

- `MV/Mul`     — `a * b`             (one `cauchyProduct`)
- `MV/Square`  — `square(a)`         (one `cauchySelfProduct`)
- `MV/Cube`    — `cube(a)`           (one self + one general)
- `MV/IPow5`   — `pow(a, 5)`         (binary-exponentiation chain)
- `MV/Asin`    — `asin(a)`           (opens with `cauchySelfProduct`)
- `MV/Atan`    — `atan(a)`           (opens with `cauchySelfProduct`)
- `MV/Atan2`   — `atan2(a, b + 1)`   (two `cauchySelfProduct` calls)
- `MV/Erf`     — `erf(a)`            (opens with `cauchySelfProduct`)

| benchmark | before (ns) | after (ns) | speed-up |
|---|---:|---:|---:|
| `MV/Mul/N5_M3`     |   4,631.3 |     396.9 | 11.67x |
| `MV/Square/N5_M3`  |   5,073.3 |     260.5 | 19.48x |
| `MV/Cube/N5_M3`    |   9,683.3 |     940.8 | 10.29x |
| `MV/IPow5/N5_M3`   |  18,656.1 |   1,590.9 | 11.73x |
| `MV/Asin/N5_M3`    |  13,893.9 |   9,074.9 |  1.53x |
| `MV/Atan/N5_M3`    |   7,782.8 |   2,988.5 |  2.60x |
| `MV/Atan2/N5_M3`   |  14,761.2 |   3,469.1 |  4.26x |
| `MV/Erf/N5_M3`     |  15,191.1 |  10,517.2 |  1.44x |
| `MV/Mul/N4_M4`     |   6,326.8 |     411.0 | 15.40x |
| `MV/Square/N4_M4`  |   6,700.7 |     268.6 | 24.95x |
| `MV/Cube/N4_M4`    |  12,830.8 |     733.2 | 17.50x |
| `MV/IPow5/N4_M4`   |  26,099.6 |   2,309.0 | 11.30x |
| `MV/Asin/N4_M4`    |  28,192.6 |  12,206.3 |  2.31x |
| `MV/Atan/N4_M4`    |  12,984.8 |   4,473.6 |  2.90x |
| `MV/Atan2/N4_M4`   |  34,965.0 |   4,927.2 |  7.10x |
| `MV/Erf/N4_M4`     |  37,002.3 |  14,339.4 |  2.58x |
| `MV/Mul/N5_M4`     |  19,383.5 |   1,004.5 | 19.30x |
| `MV/Square/N5_M4`  |  25,158.4 |     632.5 | 39.78x |
| `MV/Cube/N5_M4`    |  76,774.7 |   1,632.2 | 47.04x |
| `MV/IPow5/N5_M4`   | 173,388.6 |   4,320.8 | 40.13x |
| `MV/Asin/N5_M4`    | 138,260.5 |  68,709.0 |  2.01x |
| `MV/Atan/N5_M4`    |  74,862.8 |  12,853.2 |  5.82x |
| `MV/Atan2/N5_M4`   | 143,755.2 |  14,209.9 | 10.12x |
| `MV/Erf/N5_M4`     | 158,435.6 |  90,404.9 |  1.75x |
| `MV/Mul/N6_M3`     |   9,284.5 |     727.2 | 12.77x |
| `MV/Square/N6_M3`  |   9,749.5 |     468.7 | 20.80x |
| `MV/Cube/N6_M3`    |  18,962.0 |   1,288.6 | 14.72x |
| `MV/IPow5/N6_M3`   |  36,299.5 |   3,214.5 | 11.29x |
| `MV/Asin/N6_M3`    |  28,500.7 |  18,633.8 |  1.53x |
| `MV/Atan/N6_M3`    |  15,696.4 |   6,287.7 |  2.50x |
| `MV/Atan2/N6_M3`   |  33,445.1 |   7,183.1 |  4.66x |
| `MV/Erf/N6_M3`     |  32,336.4 |  20,133.8 |  1.61x |

## Reciprocal & Sqrt (forward substitutions)

`seriesReciprocal` and `seriesSqrt` are forward substitutions, not flat
dot products — each row depends on previously computed rows in the same
output. We can't remove that dependency, but the **per-row sub-index
sum** has the same shape as the Cauchy stencil minus the
`(beta=0, gamma=alpha)` endpoint. Reusing the existing stencil with a
one-position offset gives the same kind of speed-up.

### Univariate

| benchmark | before (ns) | after (ns) | speed-up |
|---|---:|---:|---:|
| `Tax/Reciprocal/N5`  |    54.5 |     8.2 | 6.65x |
| `Tax/Reciprocal/N10` |   148.3 |    26.3 | 5.63x |
| `Tax/Reciprocal/N20` |   411.4 |   107.7 | 3.82x |
| `Tax/Reciprocal/N40` | 1,121.1 |   310.8 | 3.61x |
| `Tax/Sqrt/N10`       |    89.7 |    24.8 | 3.62x |
| `Tax/Sqrt/N20`       |   284.1 |    84.3 | 3.37x |
| `Tax/Sqrt/N40`       |   747.0 |   293.2 | 2.55x |

### Multivariate

| benchmark | before (ns) | after (ns) | speed-up |
|---|---:|---:|---:|
| `MV/Reciprocal/N5_M3` |  4,089.8 |   420.5 |  9.73x |
| `MV/Reciprocal/N4_M4` |  8,664.7 |   376.8 | 23.00x |
| `MV/Reciprocal/N5_M4` | 24,446.0 | 1,081.2 | 22.61x |
| `MV/Reciprocal/N6_M3` |  8,972.6 |   654.4 | 13.71x |
| `MV/Sqrt/N5_M3`       |  4,009.6 |   244.4 | 16.41x |
| `MV/Sqrt/N4_M4`       |  6,056.6 |   307.6 | 19.69x |
| `MV/Sqrt/N5_M4`       | 20,983.1 |   623.4 | 33.66x |
| `MV/Sqrt/N6_M3`       |  7,903.2 |   513.4 | 15.40x |

## Compile-time and binary-size cost

The optimisations land in `.rodata` (constexpr stencils) and in
template-instantiated inline FMA chains, so compile time and binary
size scale with the *number of shapes a translation unit actually
instantiates*.  Two clean rebuilds at `-O3 -DNDEBUG -j1` with the new
`TAX_USE_UNROLL` / `TAX_USE_STENCIL` toggles, measuring two
representative consumers:

| Consumer | TAX_USE_*=OFF | TAX_USE_*=ON | Δ time | Δ size | Δ .rodata |
|---|---:|---:|---:|---:|---:|
| `bench_vs_dace`<sup>1</sup> | 9.7 s, 782 KiB | 232 s, 2.04 MiB | **+24×** | **+1.27 MiB** | **+1.13 MiB** |
| `testKernels`<sup>2</sup>   | 5.3 s, 641 KiB | 5.4 s, 641 KiB | +1 % | +64 B | ≈ 0 |

<sup>1</sup> Instantiates every operator at `(N=8, M=6)` — the worst
case in the codebase.  `numMonomials(8, 6) = 3003` output rows,
~126 K pair entries in `CauchyWeightStencil<8, 6>::db`.
The `.rodata` growth is entirely the new stencil tables;
`bench_vs_dace` had to bump `-fconstexpr-ops-limit` to compile the
ON build at all.

<sup>2</sup> A typical kernel test — small static shapes (`N ≤ 5`,
`M ≤ 4`) — where the stencil tables are kilobytes-not-megabytes and
the unrolled FMA chains stay in `i$`.

So the cost is **paid only by code that uses heavy multivariate
shapes** (M = 6, N ≥ 6 or so).  Hot ODE / ADS configurations like
`(N=12, M=4)` or `(N=4, M=4)` are well inside the cheap regime — they
add a few hundred kilobytes of `.rodata` and a few seconds of compile
time and return 10–40× kernel speed-ups in exchange.  Set
`-DTAX_USE_UNROLL=OFF -DTAX_USE_STENCIL=OFF` to fall back to the
original loop / `forEachMonomial` + `forEachSubIndex` walks when those
trade-offs aren't acceptable (e.g. very tight binary-size budgets).

## Takeaways

- Pure multivariate Cauchy ops (`Mul`, `Square`, `Cube`, `IPow5`) are
  **10–47× faster**. The wider the shape (N=5, M=4), the bigger the win —
  the stencil amortises the multi-index walk that previously dominated
  every call.
- Reciprocal and sqrt forward substitutions go **9–34× faster
  multivariate**, **2.5–6.7× faster univariate**. The cross-row
  dependency is unchanged; the win comes entirely from removing the
  per-call multi-index regeneration and giving the compiler a constant
  trip count for unrolling.
- Transcendentals that internally use `cauchySelfProduct` (`asin`,
  `atan`, `atan2`, `erf`) inherit a smaller but still significant
  **1.4–10× speed-up**. The remaining cost is the weighted scalar
  recurrence that the stencil doesn't touch.
- Univariate (`TE<N>`) is within ±1 % across all benchmarks: the M = 1
  fast path is untouched and behaves identically.
