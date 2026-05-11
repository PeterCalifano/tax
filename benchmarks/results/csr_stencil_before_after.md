# CauchyStencil — before/after Google Benchmark comparison

Comparison of the static multivariate Cauchy path before and after the
precomputed `CauchyStencil<N, M>` / `CauchySymStencil<N, M>` swap.

| | |
|---|---|
| Before | `claude/refactor-taylor-series-XtMUw` @ `6091823` |
| After  | `claude/cauchy-csr-stencil` @ `3de4993` |
| Binaries | `bench_univariate`, `bench_multivariate` (Google Benchmark) |
| CPU | Intel(R) Xeon(R) @ 2.80 GHz |
| OS | Linux 6.18.5 x86_64 |
| Build | `-DCMAKE_BUILD_TYPE=Release -DTAX_BUILD_BENCHMARK=ON` |
| Method | `--benchmark_repetitions=3 --benchmark_min_time=0.3s --benchmark_report_aggregates_only`; medians reported. |

## Univariate (control — no stencil involvement)

The M = 1 fast paths in `cauchyProduct` / `cauchyAccumulate` /
`cauchySelfProduct` are untouched by this commit, so the univariate
suite is a regression check: numbers should land within run-to-run
noise of the baseline.

| benchmark | before (ns) | after (ns) | speed-up |
|---|---:|---:|---:|
| `Tax/Sin/N10`  |    62.4 |    62.3 | 1.00x |
| `Tax/Sin/N20`  |   245.5 |   243.3 | 1.01x |
| `Tax/Sin/N40`  |   906.3 |   898.5 | 1.01x |
| `Tax/Exp/N10`  |    63.4 |    64.3 | 0.99x |
| `Tax/Exp/N20`  |   207.5 |   209.4 | 0.99x |
| `Tax/Exp/N40`  |   744.6 |   743.5 | 1.00x |
| `Tax/Log/N10`  |    60.4 |    61.0 | 0.99x |
| `Tax/Log/N20`  |   232.2 |   232.3 | 1.00x |
| `Tax/Log/N40`  |   766.1 |   768.7 | 1.00x |
| `Tax/Sqrt/N10` |    82.6 |    82.5 | 1.00x |
| `Tax/Sqrt/N20` |   267.6 |   267.9 | 1.00x |
| `Tax/Sqrt/N40` |   811.9 |   810.9 | 1.00x |
| `Tax/IPow/N10` |   161.8 |   161.7 | 1.00x |
| `Tax/IPow/N20` |   455.8 |   460.1 | 0.99x |
| `Tax/IPow/N40` | 2,013.3 | 2,027.7 | 0.99x |
| `Tax/Pow/N10`  |   109.1 |   108.3 | 1.01x |
| `Tax/Pow/N20`  |   321.9 |   318.1 | 1.01x |
| `Tax/Pow/N40`  | 1,074.5 | 1,063.6 | 1.01x |

All entries are within ±1 % — flat, as expected.

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
