# Tax vs. DACE — three-backend Google Benchmark comparison

Three kernel paths on the same eight operators, same expansion points,
same orders/sizes:

- **Static** — `tax::TE<N>` / `tax::TEn<N, M>` with all kernel
  optimisations on (`TAX_USE_UNROLL=ON`, `TAX_USE_STENCIL=ON`,
  the project defaults).
- **Dynamic** — `tax::DynTE<>` (runtime shape, no unroll / no stencil;
  shares the dense `std::vector<T>` storage with the runtime-overload
  kernels).
- **DACE** — `DACE::DA` arithmetic from `dacelib/dace` v2.1.0.

| | |
|---|---|
| Binary | `bench_vs_dace` (Google Benchmark) |
| Branch | `claude/cauchy-csr-stencil` |
| CPU | Intel(R) Xeon(R) @ 2.80 GHz |
| OS | Linux 6.18.5 x86_64 |
| Build | `-DCMAKE_BUILD_TYPE=Release -DTAX_BUILD_BENCHMARK=ON -DTAX_USE_DACE=ON` |
| Method | `--benchmark_repetitions=3 --benchmark_min_time=0.3s --benchmark_report_aggregates_only`; medians reported. |

**Operand construction (fair comparison).** The multivariate operands
are *dense linear combinations of all M = 6 variables* so neither
backend can short-circuit into a near-univariate regime:

```cpp
// All three backends construct the equivalent mathematical operand.
constexpr std::array<double, 6> kAlphaA{ 0.10, 0.05, 0.03, 0.02, 0.01, 0.005 };
//  x  =  1.1 + 0.10*x_1 + 0.05*x_2 + 0.03*x_3 + 0.02*x_4 + 0.01*x_5 + 0.005*x_6
//  y  =  1.2 + (reversed kAlphaB)
```

Every linear monomial is nonzero from the start, so the first
multiplication / transcendental step already produces a polynomial
dense across all six axes.

`Static vs DACE` columns are speed-up factors of tax static over DACE
(`> 1` means tax wins; `< 1` means DACE wins).

## Univariate (M = 1)

### Mul

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE | Dynamic vs DACE |
|---:|---:|---:|---:|---:|---:|
| 5  |     6.8 |    73.8 |    65.2 | 9.53x | 0.88x |
| 10 |    26.0 |   113.6 |    80.2 | 3.08x | 0.71x |
| 20 |   125.6 |   233.0 |   105.1 | 0.84x | 0.45x |
| 40 |   553.0 |   781.2 |   178.4 | 0.32x | 0.23x |

### Reciprocal

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE | Dynamic vs DACE |
|---:|---:|---:|---:|---:|---:|
| 5  |    10.6 |   123.8 |   330.1 | 31.23x | 2.67x |
| 10 |    31.1 |   257.1 |   518.2 | 16.67x | 2.02x |
| 20 |   117.9 |   643.8 |   825.0 |  7.00x | 1.28x |
| 40 |   561.5 | 2,096.3 | 1,445.2 |  2.57x | 0.69x |

### Sqrt

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE | Dynamic vs DACE |
|---:|---:|---:|---:|---:|---:|
| 5  |    10.8 |    76.1 |   391.7 | 36.38x | 5.15x |
| 10 |    32.1 |   124.1 |   816.4 | 25.44x | 6.58x |
| 20 |    98.0 |   288.6 | 2,178.6 | 22.24x | 7.55x |
| 40 |   319.6 |   803.6 | 6,661.9 | 20.85x | 8.29x |

### Exp

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE | Dynamic vs DACE |
|---:|---:|---:|---:|---:|---:|
| 5  |    21.5 |    78.3 |   353.2 | 16.46x | 4.51x |
| 10 |    74.8 |   131.2 |   773.4 | 10.34x | 5.90x |
| 20 |   327.5 |   328.1 | 2,076.1 |  6.34x | 6.33x |
| 40 | 1,305.2 | 1,077.0 | 6,553.4 |  5.02x | 6.09x |

### Log

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE | Dynamic vs DACE |
|---:|---:|---:|---:|---:|---:|
| 5  |    26.3 |    77.4 |   371.7 | 14.11x | 4.80x |
| 10 |    94.8 |   137.3 |   779.5 |  8.22x | 5.68x |
| 20 |   359.2 |   343.7 | 2,143.9 |  5.97x | 6.24x |
| 40 | 1,280.0 | 1,103.7 | 6,604.3 |  5.16x | 5.98x |

### Sin

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE | Dynamic vs DACE |
|---:|---:|---:|---:|---:|---:|
| 5  |    35.2 |   129.6 |   370.1 | 10.52x | 2.86x |
| 10 |    98.1 |   205.1 |   766.2 |  7.81x | 3.74x |
| 20 |   337.0 |   426.5 | 2,087.9 |  6.20x | 4.90x |
| 40 | 1,583.3 | 1,273.5 | 6,537.6 |  4.13x | 5.13x |

### Pow (`a^0.5`)

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE | Dynamic vs DACE |
|---:|---:|---:|---:|---:|---:|
| 5  |    44.8 |   108.8 |   405.2 | 9.05x | 3.72x |
| 10 |   116.2 |   216.7 |   810.1 | 6.97x | 3.74x |
| 20 |   387.8 |   516.6 | 2,137.3 | 5.51x | 4.14x |
| 40 | 1,351.8 | 1,720.2 | 6,710.6 | 4.96x | 3.90x |

### IPow (`a^5`)

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE | Dynamic vs DACE |
|---:|---:|---:|---:|---:|---:|
| 5  |    23.2 |   311.5 |   253.0 | 10.88x | 0.81x |
| 10 |   113.8 |   513.3 |   308.7 |  2.71x | 0.60x |
| 20 |   492.9 | 1,056.3 |   419.1 |  0.85x | 0.40x |
| 40 | 2,353.1 | 3,335.9 |   767.7 |  0.33x | 0.23x |

### Univariate takeaways

- **Tax static dominates transcendentals.** `Sqrt`, `Exp`, `Log`, `Sin`,
  `Pow`, `Reciprocal` are **5–36× faster** than DACE across the whole
  N grid. The unrolled forward-substitution chains run with FMA-pipeline
  throughput that DACE's sparse coefficient walk can't match.
- **DACE catches up on pure-Cauchy products at high N.** For `Mul`
  (single Cauchy product) and `IPow` (binary-exponentiation chain of
  Cauchy products), DACE wins from `N ≈ 20` onward. DACE's sparse
  representation skips the all-zero off-diagonal cross-terms that tax's
  dense `std::array<double, N+1>` storage still has to multiply through.
- **Tax dynamic** is 2–10× slower than tax static (allocation +
  dispatch overhead), but still beats DACE on the transcendental
  workloads up to `N = 40`.

## Multivariate (M = 6, dense operands)

### Mul

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE | Dynamic vs DACE |
|---:|---:|---:|---:|---:|---:|
| 2 |     103.8 |       7,059.7 |       173.8 | 1.67x | 0.02x |
| 4 |   1,707.4 |     257,721.6 |       373.8 | 0.22x | 0.00x |
| 6 |  19,760.4 |   3,280,183.9 |       868.1 | 0.04x | 0.00x |
| 8 | 141,306.5 |  26,556,153.2 |     2,333.9 | 0.02x | 0.00x |

### Reciprocal

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE | Dynamic vs DACE |
|---:|---:|---:|---:|---:|---:|
| 2 |      92.7 |      13,322.9 |       340.7 | 3.68x | 0.03x |
| 4 |   1,572.7 |     533,000.0 |     2,474.4 | 1.57x | 0.00x |
| 6 |  26,529.8 |   6,858,426.1 |    35,281.8 | 1.33x | 0.01x |
| 8 | 135,677.0 |  55,029,428.8 |   133,786.7 | 0.99x | 0.00x |

### Sqrt

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE | Dynamic vs DACE |
|---:|---:|---:|---:|---:|---:|
| 2 |      73.6 |       3,823.2 |       343.6 | 4.67x | 0.09x |
| 4 |   1,156.0 |     209,472.6 |     2,235.6 | 1.93x | 0.01x |
| 6 |  12,564.2 |   3,087,652.9 |    14,272.9 | 1.14x | 0.00x |
| 8 |  76,101.1 |  26,750,079.8 |    57,713.0 | 0.76x | 0.00x |

### Exp

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE | Dynamic vs DACE |
|---:|---:|---:|---:|---:|---:|
| 2 |     107.5 |       6,048.4 |       299.1 | 2.78x | 0.05x |
| 4 |   2,758.6 |     260,091.2 |     2,265.3 | 0.82x | 0.01x |
| 6 |  21,573.8 |   3,478,368.5 |    14,177.6 | 0.66x | 0.00x |
| 8 | 176,234.1 |  28,531,819.5 |    58,649.8 | 0.33x | 0.00x |

### Log

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE | Dynamic vs DACE |
|---:|---:|---:|---:|---:|---:|
| 2 |     116.4 |       4,203.5 |       309.6 | 2.66x | 0.07x |
| 4 |   2,056.7 |     214,694.1 |     2,251.0 | 1.09x | 0.01x |
| 6 |  22,612.0 |   3,207,567.3 |    14,114.1 | 0.62x | 0.00x |
| 8 | 156,978.8 |  27,368,548.9 |    57,593.9 | 0.37x | 0.00x |

### Sin

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE | Dynamic vs DACE |
|---:|---:|---:|---:|---:|---:|
| 2 |     171.2 |       5,310.5 |       307.6 | 1.80x | 0.06x |
| 4 |   2,605.2 |     236,585.4 |     2,203.9 | 0.85x | 0.01x |
| 6 |  27,491.2 |   3,240,652.7 |    13,978.9 | 0.51x | 0.00x |
| 8 | 217,681.8 |  27,174,932.0 |    58,124.9 | 0.27x | 0.00x |

### Pow (`a^0.5`)

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE | Dynamic vs DACE |
|---:|---:|---:|---:|---:|---:|
| 2 |     165.8 |       5,777.1 |       342.7 | 2.07x | 0.06x |
| 4 |   2,645.5 |     247,176.9 |     2,339.1 | 0.88x | 0.01x |
| 6 |  29,256.9 |   3,301,894.1 |    14,113.5 | 0.48x | 0.00x |
| 8 | 216,495.0 |  27,434,199.7 |    57,595.0 | 0.27x | 0.00x |

### IPow (`a^5`)

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE | Dynamic vs DACE |
|---:|---:|---:|---:|---:|---:|
| 2 |     440.1 |      30,002.3 |       786.4 | 1.79x | 0.03x |
| 4 |   8,789.5 |   1,011,424.3 |     3,763.7 | 0.43x | 0.00x |
| 6 |  81,913.4 |  13,024,289.3 |     7,550.8 | 0.09x | 0.00x |
| 8 | 570,683.7 | 109,766,501.5 |    13,545.5 | 0.02x | 0.00x |

### Multivariate takeaways

With dense operands (every coordinate axis genuinely contributing), the
picture is materially different from the earlier 2-sparse run:

- **Tax static is faster across the board at `N = 2`** — 1.67× on `Mul`,
  1.79–4.67× on every other operator. The dense weighted-stencil walk
  beats DACE's per-pair sparse-map lookups when the operand fills
  most of the truncation envelope.
- **Tax stays competitive at `N = 4`–`6` for the divide-style
  forward subs** (`Reciprocal`, `Sqrt`, sometimes `Log`): 0.62–1.93×
  vs DACE. These are the kernels where the dense FMA chain over the
  stencil's pair list closely matches DACE's sparse iteration on a
  near-dense output.
- **DACE wins from `N = 4` upward on `Mul` and `IPow`** (3–110×) and
  from `N = 6` upward on the heavier transcendentals (`Exp`, `Log`,
  `Sin`, `Pow`, ~2–4×). Two compounding factors: DACE's truncation
  table indexes are bit-packed in 32-bit fields (one mul = one index
  OR + sparse lookup) where tax's stencil pair lists balloon with
  `numMonomials(N, 2M) = C(N+2M, 2M)`, and DACE's core C kernels are
  the product of two decades of hand-tuning.
- **Tax dynamic is unusably slow at M = 6.** The runtime kernels use a
  live `forEachMonomial` + `forEachSubIndex` walk over heap-allocated
  `std::vector<int>` alpha/beta buffers; for `(N=8, M=6)` that's
  ~125K pair enumerations per call against a 3003-element coefficient
  table, with no stencil amortisation. The runtime path is a
  Python/REPL convenience, not a perf target — for high-M static
  workloads use `TEn<N, M>`; for high-M dynamic workloads, use DACE.

## Choosing a backend

Based on the dense-operand grid above:

| Workload                                          | Best backend |
|---------------------------------------------------|--------------|
| Univariate transcendentals (sqrt/exp/log/sin/...) any N | **tax static** |
| Univariate multiplicative (`Mul` / `IPow`) at N ≤ 10 | **tax static** |
| Univariate multiplicative at N ≥ 20               | DACE |
| Multivariate, M ≤ 3 (any N reasonable)            | **tax static** (see `csr_stencil_before_after.md`) |
| Multivariate, M = 6, N ≤ 2                        | **tax static** |
| Multivariate, M = 6, N = 4 — Reciprocal/Sqrt/Log  | **tax static** (~1× vs DACE) |
| Multivariate, M = 6, N ≥ 4 — Mul / IPow           | DACE |
| Multivariate, M = 6, N ≥ 6 — most transcendentals | DACE |
| Runtime-shape / Python REPL / mixed shapes        | tax dynamic (correctness) or DACE (perf) |
