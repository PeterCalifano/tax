# Cauchy kernel optimisations — before/after Google Benchmark comparison

Comparison of the static Cauchy path before and after two stacked
optimisations:

1. **Multivariate (M ≥ 2)**: precomputed `CauchyStencil<N, M>` /
   `CauchySymStencil<N, M>` CSR tables.
2. **Univariate (M = 1)**: compile-time-unrolled FMA chain parameterised
   by `std::index_sequence`.

| | |
|---|---|
| Before | `claude/refactor-taylor-series-XtMUw` @ `6091823` |
| After  | `claude/cauchy-csr-stencil` @ `02fbcb2` |
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

## Takeaways

- Pure multivariate Cauchy ops (`Mul`, `Square`, `Cube`, `IPow5`) are
  **10–47× faster**. The wider the shape (N=5, M=4), the bigger the win —
  the stencil amortises the multi-index walk that previously dominated
  every call.
- Transcendentals that internally use `cauchySelfProduct` (`asin`,
  `atan`, `atan2`, `erf`) inherit a smaller but still significant
  **1.4–10× speed-up**. The remaining cost is the weighted scalar
  recurrence that the stencil doesn't touch.
- Univariate (`TE<N>`) is within ±1 % across all benchmarks: the M = 1
  fast path is untouched and behaves identically.
