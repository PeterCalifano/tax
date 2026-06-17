# Foundations & Ordering

This page covers the mathematical objects **tax** propagates: truncated Taylor polynomials, the graded-lexicographic ordering used to store their coefficients, and how the univariate picture generalises to many variables.

For the per-operation recurrence relations (arithmetic, algebraic, trigonometric, hyperbolic, transcendental, power, and special functions), see [Recurrence Relations](../internals/recurrences.md). For truncation-error bounds and convergence diagnostics, see [Convergence & Truncation](convergence.md).

---

## Truncated Taylor Polynomials

A truncated Taylor expansion of a function $f$ in $M$ variables around a point $\mathbf{x}_0$ up to order $N$ is:

$$
f(\mathbf{x}_0 + \delta\mathbf{x}) = \sum_{|\alpha| \le N} f_\alpha \, \delta\mathbf{x}^\alpha + \mathcal{O}(|\delta\mathbf{x}|^{N+1})
$$

where $\alpha = (\alpha_1, \ldots, \alpha_M)$ is a **multi-index** with non-negative integer entries, $|\alpha| = \alpha_1 + \cdots + \alpha_M$ is its total degree, and

$$
\delta\mathbf{x}^\alpha = \delta x_1^{\alpha_1} \cdots \delta x_M^{\alpha_M}
$$

The Taylor coefficients $f_\alpha$ are related to partial derivatives by:

$$
f_\alpha = \frac{1}{\alpha!} \partial^\alpha f(\mathbf{x}_0), \qquad \alpha! = \alpha_1! \cdots \alpha_M!
$$

In the **univariate** case ($M = 1$), the multi-index reduces to a single integer $d$, and the expansion simplifies to:

$$
f(x_0 + \delta x) = \sum_{d=0}^{N} f_d \, \delta x^d, \qquad f_d = \frac{f^{(d)}(x_0)}{d!}
$$

---

## Graded Lexicographic Ordering

Coefficients are stored using **graded lexicographic (grlex) ordering**:
monomials are grouped by total degree $|\alpha|$, and within each degree group
they are sorted lexicographically by the exponent vector $\alpha$. For dense
storage the container is a `std::array<T, S>`; for sparse storage only the
nonzero coefficients are stored alongside their flat indices (see
[Dense vs Sparse Storage](../guide/storage.md)).

**Example for $M = 2$, $N = 2$:**

| Flat index | Multi-index $(\alpha_1, \alpha_2)$ | Monomial | Degree |
|:----------:|:------------------------------------:|:--------:|:------:|
| 0 | (0, 0) | 1 | 0 |
| 1 | (0, 1) | $\delta x_2$ | 1 |
| 2 | (1, 0) | $\delta x_1$ | 1 |
| 3 | (0, 2) | $\delta x_2^2$ | 2 |
| 4 | (1, 1) | $\delta x_1 \delta x_2$ | 2 |
| 5 | (2, 0) | $\delta x_1^2$ | 2 |

The total number of coefficients for $M$ variables at order $N$ is:

$$
S = \binom{N + M}{M}
$$

**Coefficient counts for common $(N, M)$ pairs:**

| $N \backslash M$ | 1 | 2 | 3 | 4 | 5 | 6 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| 2 | 3 | 6 | 10 | 15 | 21 | 28 |
| 3 | 4 | 10 | 20 | 35 | 56 | 84 |
| 4 | 5 | 15 | 35 | 70 | 126 | 210 |
| 5 | 6 | 21 | 56 | 126 | 252 | 462 |
| 6 | 7 | 28 | 84 | 210 | 462 | 924 |
| 8 | 9 | 45 | 165 | 495 | 1287 | 3003 |
| 10 | 11 | 66 | 286 | 1001 | 3003 | 8008 |

---

## Multivariate Generalisation

All univariate recurrences in the [Recurrence Relations](../internals/recurrences.md) reference generalize naturally to the multivariate case. The key substitutions are:

1. **Scalar index $d$** is replaced by **multi-index $\alpha$** with $|\alpha| = d$.
2. **Inner sums** $\sum_{k=0}^{d}$ become **sub-multi-index sums** $\sum_{\beta \le \alpha}$, iterated over all $\beta$ with $\beta_i \le \alpha_i$ for each component.
3. **Degree constraints** like $1 \le k \le d-1$ become $1 \le |\beta| \le |\alpha|-1$ (or the appropriate range).
4. The **weight factor** $d - k$ generalises to $|\alpha| - |\beta|$, and $k$ to $|\beta|$.

In the implementation, the function `forEachSubIndex<M>(alpha, lo, hi, callback)` enumerates all sub-multi-indices $\beta \le \alpha$ with $\text{lo} \le |\beta| \le \text{hi}$, calling `callback(flatIndex(beta), flatIndex(alpha - beta), |beta|)`. This provides a uniform interface for both univariate and multivariate kernels, with the univariate path using simple scalar loops as a fast path via `if constexpr (M == 1)`.
