# Convergence & Truncation

This page describes how well a truncated expansion approximates the true function, and the practical diagnostics **tax** exposes for estimating that error. For the definitions of truncated Taylor polynomials and coefficient ordering it relies on, see [Foundations & Ordering](foundations.md).

---

## Truncation Error and Convergence

The truncated polynomial $\tilde{f}_N(\mathbf{x}_0 + \delta\mathbf{x})$ of
order $N$ approximates the true function with error

$$
\bigl| f(\mathbf{x}_0 + \delta\mathbf{x}) - \tilde{f}_N(\mathbf{x}_0 + \delta\mathbf{x}) \bigr|
  \le C \, |\delta\mathbf{x}|^{N+1}
$$

within the **radius of convergence** of the underlying Taylor series, where
$C$ bounds the magnitude of the $(N+1)$-th derivatives. The library does
not compute $C$ itself, but two practical proxies are exposed on
`TaylorExpansion`:

- **Coefficient $\ell^\infty$ norm** — `coeffs_norm_inf()` returns
  $\max_{|\alpha| \le N} |f_\alpha|$. A small value at the boundary
  $|\alpha| = N$ suggests the truncation is harmless on the displacement
  scale of interest.
- **Per-degree extraction** — `f.coeff({k})` (or `f.coeff<k,...>()`) gives the
  individual $f_\alpha$; the geometric decay rate of the largest-magnitude
  coefficient at each total degree estimates the convergence radius.

This is the same idea a Jorba–Zou step-size controller applies inside a Taylor
ODE integrator to choose the step $h$ from the last two coefficient norms of
the per-step expansion.

---

## References

- W. Rudin, *Principles of Mathematical Analysis*, 3rd ed., McGraw-Hill, 1976 —
  Taylor's theorem and the Lagrange form of the remainder.
- K. Makino and M. Berz, *Taylor models and other validated functional
  inclusion methods*, International Journal of Pure and Applied Mathematics
  4(4), 379–456, 2003 — rigorous remainder bounds for truncated Taylor models.
- À. Jorba and M. Zou, *A Software Package for the Numerical Integration of ODEs
  by Means of High-Order Taylor Methods*, Experimental Mathematics 14(1),
  99–117, 2005 — using coefficient decay to estimate the convergence radius.
