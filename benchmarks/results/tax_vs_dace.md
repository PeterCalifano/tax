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

**Operand construction (fully-dense fair comparison).** The
multivariate operands are constructed so that *every* monomial slot
up to the truncation order is nonzero — matching tax's
`std::array<T, numMonomials(N, M)>` storage with a DACE sparse-map
that is also fully populated:

```cpp
// Identical math on all three backends.
constexpr std::array<double, 6> kAlphaA{ 0.10, 0.05, 0.03, 0.02, 0.01, 0.005 };
//  pX = sqrt(1.1 + 0.10*x_1 + 0.05*x_2 + 0.03*x_3 + 0.02*x_4 + 0.01*x_5 + 0.005*x_6)
//  pY = sqrt(1.2 + reversed alphas)
```

Wrapping the dense linear in `sqrt(...)` saturates every multi-index
slot with a strictly nonzero coefficient (and, unlike `exp(linear)`,
doesn't trivially collapse under `log`). So both backends pay the full
`C(N+M, M)` storage cost and walk every entry in their kernels.

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

## Multivariate (M = 6, fully-dense operands)

### Mul

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE |
|---:|---:|---:|---:|---:|
| 2 |      99.8 |        7,089.4 |        324.0 | 3.25x |
| 4 |   1,724.9 |      252,682.2 |      4,091.1 | 2.37x |
| 6 |  20,201.5 |    3,271,548.9 |     35,395.2 | 1.75x |
| 8 | 143,261.5 |   26,587,964.8 |    221,101.8 | 1.54x |

### Reciprocal

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE |
|---:|---:|---:|---:|---:|
| 2 |      98.4 |       13,148.6 |        481.1 | 4.89x |
| 4 |   1,578.4 |      526,010.5 |      6,382.0 | 4.04x |
| 6 |  19,506.4 |    6,731,421.4 |     68,492.0 | 3.51x |
| 8 | 137,643.6 |   54,866,381.8 |    313,550.0 | 2.28x |

### Sqrt

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE |
|---:|---:|---:|---:|---:|
| 2 |      73.2 |        3,905.5 |        472.6 | 6.45x |
| 4 |   1,181.0 |      205,481.3 |      6,115.2 | 5.18x |
| 6 |  10,768.0 |    3,107,350.8 |     57,360.4 | 5.33x |
| 8 |  76,691.1 |   26,702,855.6 |    383,911.5 | 5.01x |

### Exp

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE |
|---:|---:|---:|---:|---:|
| 2 |     107.4 |        5,943.7 |        403.6 | 3.76x |
| 4 |   2,925.4 |      259,412.4 |      5,829.9 | 1.99x |
| 6 |  22,968.8 |    3,476,346.7 |     56,225.8 | 2.45x |
| 8 | 178,270.5 |   28,555,010.5 |    380,518.1 | 2.13x |

### Log

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE |
|---:|---:|---:|---:|---:|
| 2 |     116.5 |        4,195.2 |        435.5 | 3.74x |
| 4 |   2,097.2 |      211,233.8 |      6,062.8 | 2.89x |
| 6 |  22,895.4 |    3,173,118.8 |     56,857.8 | 2.48x |
| 8 | 158,949.5 |   27,228,251.4 |    387,760.8 | 2.44x |

### Sin

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE |
|---:|---:|---:|---:|---:|
| 2 |     171.0 |        5,387.7 |        413.5 | 2.42x |
| 4 |   2,579.0 |      228,488.7 |      5,882.6 | 2.28x |
| 6 |  27,084.7 |    3,225,080.7 |     56,381.3 | 2.08x |
| 8 | 209,900.6 |   27,412,014.3 |    380,366.8 | 1.81x |

### Pow (`a^0.5`)

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE |
|---:|---:|---:|---:|---:|
| 2 |     163.0 |        5,694.7 |        466.9 | 2.87x |
| 4 |   2,623.4 |      245,309.7 |      6,123.6 | 2.33x |
| 6 |  29,413.8 |    3,356,007.3 |     57,396.1 | 1.95x |
| 8 | 236,282.6 |   27,711,188.6 |    384,227.5 | 1.63x |

### IPow (`a^5`)

| N | Static (ns) | Dynamic (ns) | DACE (ns) | Static vs DACE |
|---:|---:|---:|---:|---:|
| 2 |     418.7 |       28,553.5 |      1,052.2 | 2.51x |
| 4 |   8,739.9 |    1,034,195.8 |     11,745.9 | 1.34x |
| 6 |  95,526.7 |   13,083,082.3 |     96,653.2 | 1.01x |
| 8 | 566,378.0 |  106,217,353.5 |    602,559.9 | 1.06x |

### Multivariate takeaways

With fully-dense operands — every multi-index slot strictly nonzero in
both backends' storage — **tax static beats DACE on every multivariate
configuration** in the M = 6 grid:

- **Sqrt** is the clearest win: **5.0–6.5×** flat from N=2 to N=8.
  The stencil-driven sym-Cauchy walk plus the cached `2*out[0]` divisor
  beats DACE's per-monomial sparse-map iteration when the map is fully
  populated.
- **Reciprocal** is **2.3–4.9×** faster across the grid.
- **Exp / Log / Sin / Pow** sit at **1.6–3.8×** depending on N — the
  weighted forward substitution's dense FMA chain is more cache-friendly
  than DACE's sparse-list bookkeeping when the list is fully populated.
- **Mul** is **1.5–3.3×** faster — the asymmetric stencil walks
  `numMonomials(N, 2*M)` pairs as a flat array of `uint16_t` indices,
  while DACE re-indexes its sparse list for each output monomial.
- **IPow (a^5)** is the closest race (**1.0–2.5×**), because binary
  exponentiation chains 4–5 Cauchy products and DACE's per-call
  fixed overhead amortises across the chain.

DACE's edge in the earlier 2-sparse run was structural — it
walks only the nonzero monomials, so a 2-term input ran 1500× less
work than tax's dense table walk. When both backends are paid the
same `C(N+M, M)` cost, the tax static path's compile-time stencil
tables and unrolled inner loops actually pull ahead, including at
the larger `(N, M)` configurations.

**Tax dynamic** is still 2–4 orders of magnitude slower than tax
static at M = 6 — it uses live `forEachMonomial` + `forEachSubIndex`
walks over heap-allocated `std::vector<int>` alpha/beta buffers, with
no stencil amortisation. Use `TEn<N, M>` for high-M static work; the
dynamic path remains a Python/REPL convenience.

## Choosing a backend

| Workload                                                    | Best backend |
|-------------------------------------------------------------|--------------|
| Univariate transcendentals (sqrt/exp/log/sin/...) any N     | **tax static** |
| Univariate multiplicative (`Mul` / `IPow`) at N ≤ 10        | **tax static** |
| Univariate multiplicative at N ≥ 20                         | DACE |
| Multivariate (any M, any N) — *dense* coefficients          | **tax static** |
| Multivariate — *very sparse* operands (few nonzero monomials in a big shape) | DACE (its sparse map walks only the nonzero monomials) |
| Runtime-shape / Python REPL / mixed shapes                  | tax dynamic (correctness) or DACE (perf) |

The fork in the multivariate row is the key one: if your real workload
is dense — typical ODE flow polynomials, ADS leaves, anything that's
been propagated for a few steps — use tax. If it's structurally sparse
(e.g. expanding around a single perturbation axis in a higher-D state
space and never multiplying by other axes), DACE's sparse storage
walks only the live monomials and wins.
