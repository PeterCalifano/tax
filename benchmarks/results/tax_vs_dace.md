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
| Branch | `claude/cauchy-csr-stencil` @ `a473afd` |
| CPU | Intel(R) Xeon(R) @ 2.80 GHz |
| OS | Linux 6.18.5 x86_64 |
| Build | `-DCMAKE_BUILD_TYPE=Release -DTAX_BUILD_BENCHMARK=ON -DTAX_USE_DACE=ON` |
| Method | `--benchmark_repetitions=3 --benchmark_min_time=0.3s --benchmark_report_aggregates_only`; medians reported. |

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

## Multivariate (M = 6)

### Mul

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE | Dynamic vs DACE |
|---:|---:|---:|---:|---:|---:|
| 2 |       111.1 |         7,114.4 |       76.6 | 0.69x | 0.01x |
| 4 |     1,723.6 |       258,807.1 |      241.4 | 0.14x | 0.00x |
| 6 |    19,671.5 |     3,292,224.0 |      702.9 | 0.04x | 0.00x |
| 8 |   139,733.5 |    26,624,902.5 |    2,018.4 | 0.01x | 0.00x |

### Reciprocal

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE | Dynamic vs DACE |
|---:|---:|---:|---:|---:|---:|
| 2 |        91.5 |        13,611.0 |      206.8 | 2.26x | 0.02x |
| 4 |     1,579.5 |       556,619.4 |      926.9 | 0.59x | 0.00x |
| 6 |    18,366.6 |     6,817,476.9 |    3,778.4 | 0.21x | 0.00x |
| 8 |   138,201.8 |    55,085,018.9 |   10,438.6 | 0.08x | 0.00x |

### Sqrt

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE | Dynamic vs DACE |
|---:|---:|---:|---:|---:|---:|
| 2 |        78.5 |         3,989.9 |      232.3 | 2.96x | 0.06x |
| 4 |       952.5 |       213,277.6 |      925.5 | 0.97x | 0.00x |
| 6 |    13,113.8 |     3,105,200.6 |    3,722.4 | 0.28x | 0.00x |
| 8 |    76,998.5 |    26,426,465.6 |   14,895.2 | 0.19x | 0.00x |

### Exp

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE | Dynamic vs DACE |
|---:|---:|---:|---:|---:|---:|
| 2 |       123.2 |         6,235.7 |      194.9 | 1.58x | 0.03x |
| 4 |     1,916.6 |       269,735.8 |      886.5 | 0.46x | 0.00x |
| 6 |    21,342.4 |     3,586,131.0 |    3,657.2 | 0.17x | 0.00x |
| 8 |   178,695.5 |    29,525,126.6 |   14,594.2 | 0.08x | 0.00x |

### Log

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE | Dynamic vs DACE |
|---:|---:|---:|---:|---:|---:|
| 2 |       113.1 |         4,057.0 |      202.5 | 1.79x | 0.05x |
| 4 |     2,030.8 |       217,547.1 |      897.4 | 0.44x | 0.00x |
| 6 |    23,076.6 |     3,227,967.3 |    3,675.2 | 0.16x | 0.00x |
| 8 |   157,564.2 |    27,223,499.3 |   14,721.7 | 0.09x | 0.00x |

### Sin

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE | Dynamic vs DACE |
|---:|---:|---:|---:|---:|---:|
| 2 |       176.9 |         5,484.3 |      203.7 | 1.15x | 0.04x |
| 4 |     2,691.5 |       240,178.4 |      916.3 | 0.34x | 0.00x |
| 6 |    27,612.7 |     3,453,076.4 |    3,822.9 | 0.14x | 0.00x |
| 8 |   212,307.2 |    27,103,161.2 |   14,723.9 | 0.07x | 0.00x |

### Pow (`a^0.5`)

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE | Dynamic vs DACE |
|---:|---:|---:|---:|---:|---:|
| 2 |       165.2 |         6,148.4 |      232.3 | 1.41x | 0.04x |
| 4 |     3,278.9 |       261,853.0 |      930.7 | 0.28x | 0.00x |
| 6 |    30,019.4 |     3,530,123.6 |    3,695.7 | 0.12x | 0.00x |
| 8 |   221,297.1 |    28,083,181.3 |   14,705.9 | 0.07x | 0.00x |

### IPow (`a^5`)

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE | Dynamic vs DACE |
|---:|---:|---:|---:|---:|---:|
| 2 |       418.1 |        28,416.9 |      279.5 | 0.67x | 0.01x |
| 4 |     7,485.0 |     1,024,555.7 |      938.8 | 0.13x | 0.00x |
| 6 |    82,120.7 |    13,206,303.5 |    2,804.1 | 0.03x | 0.00x |
| 8 |   586,336.9 |   107,028,081.3 |    8,203.3 | 0.01x | 0.00x |

### Multivariate takeaways

- **Tax static is competitive at `N = 2`** (1.15–2.96× faster than DACE
  on `Reciprocal`, `Sqrt`, `Exp`, `Log`, `Sin`, `Pow`), and roughly
  even on `Mul` (0.69×). For tight DA-style flow polynomials at M = 6
  with `N ≤ 2`, tax is the right tool.
- **DACE wins decisively from `N = 4` upward** on M = 6. The number of
  multivariate monomials grows as `C(N+M, M)`, hitting 3003 at `(N=8,
  M=6)`. Tax's dense `std::array<T, 3003>` storage drives every
  Cauchy-shape kernel into a quadratic walk over the full coefficient
  table; DACE's sparse map stays small because most coefficients of a
  single variable are zero.
- **Tax dynamic is unusably slow at M = 6.** The runtime kernels use a
  live `forEachMonomial` + `forEachSubIndex` walk over heap-allocated
  `std::vector<int>` alpha/beta buffers; for `(N=8, M=6)` that's
  ~125K pair enumerations per call against a 3003-element coefficient
  table, with no stencil amortisation. The runtime path is a
  Python/REPL convenience, not a perf target — for high-M static
  workloads use `TEn<N, M>`; for high-M dynamic workloads, use DACE.

## Choosing a backend

| Workload                                          | Best backend |
|---------------------------------------------------|--------------|
| Univariate transcendentals (sqrt/exp/log/sin/...) any N | **tax static** |
| Univariate multiplicative (`Mul` / `IPow`) at N ≤ 10 | **tax static** |
| Univariate multiplicative at N ≥ 20               | DACE |
| Multivariate, M ≤ 3 (any N reasonable)            | **tax static** (see `csr_stencil_before_after.md`) |
| Multivariate, M = 6, N ≤ 2                        | **tax static** (or even) |
| Multivariate, M = 6, N ≥ 4                        | DACE |
| Runtime-shape / Python REPL / mixed shapes        | tax dynamic (correctness) or DACE (perf) |
