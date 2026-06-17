# Map Inversion

`tax::invert` (in `tax/la/invert.hpp`) formally inverts a square polynomial map
$\mathbf{P} : \mathbb{R}^M \to \mathbb{R}^M$ by Picard iteration on its
non-constant part. This page documents the math; for usage see the
[Eigen Integration guide](../guide/eigen.md#map-inversion).

## Setup

The constant term of every component is dropped first, so inversion operates on
an **origin-centered perturbation map** $\mathbf{P}$ with
$\mathbf{P}(\mathbf 0)=\mathbf 0$. Split it into its linear and nonlinear parts,

$$
\mathbf{P}(\boldsymbol{\delta x}) = \underbrace{J\,\boldsymbol{\delta x}}_{\text{linear}}
  \; + \; \underbrace{\mathbf{N}(\boldsymbol{\delta x})}_{\text{order} \ge 2},
\qquad
J = \left.\frac{\partial \mathbf{P}}{\partial \boldsymbol{\delta x}}\right|_{\mathbf 0},
$$

where $J$ is the Jacobian of the linear part and $\mathbf{N}$ collects every
monomial of degree $\ge 2$.

## Fixed-point form

We seek the inverse map $\mathbf{G}$ with
$\mathbf{P}\big(\mathbf{G}(\mathbf y)\big) = \mathbf y$. Substituting the split
and isolating the linear occurrence of $\mathbf{G}$ gives

$$
\mathbf{G}(\mathbf y) = J^{-1}\Big(\mathbf y - \mathbf{N}\big(\mathbf{G}(\mathbf y)\big)\Big).
$$

## Picard iteration

The fixed point is reached by iterating from the linear inverse:

$$
\mathbf{G}_0(\mathbf y) = J^{-1}\mathbf y,
\qquad
\mathbf{G}_{k+1}(\mathbf y) = J^{-1}\Big(\mathbf y - \mathbf{N}\big(\mathbf{G}_k(\mathbf y)\big)\Big).
$$

Because $\mathbf{N}$ has no constant or linear part, composing it with
$\mathbf{G}_k$ raises by at least one the lowest order at which
$\mathbf{G}_{k+1}$ can still be wrong. The degree-1 coefficients are already
exact in $\mathbf{G}_0$, so after $N-1$ iterations $\mathbf{G}$ is exact through
order $N$ — and the truncation discards everything beyond, so the iteration
**terminates** rather than merely converging. Formal left and right inverses of
a truncated series coincide, hence
$\mathbf{G}\circ\mathbf{P} = \mathbf{P}\circ\mathbf{G} = \mathrm{id}$ to order $N$.

## Implementation notes

- The linear part must be invertible. `invert` factors $J$ with a full-pivot LU
  (`Eigen::FullPivLU`) and throws `std::invalid_argument` if it is singular.
- Each Picard step composes the truncated series $\mathbf{N}\circ\mathbf{G}_k$
  using the same monomial substitution machinery as series composition
  (`detail::composeMap`), so the cost is dominated by $N-1$ map compositions.
- Inversion is of the **perturbation** part only. For a flow map
  $\mathbf F(\mathbf x_0 + \boldsymbol{\delta x}) = \mathbf y_0 + \mathbf{P}(\boldsymbol{\delta x})$,
  recover the displacement-to-displacement inverse by evaluating the returned
  map at $\mathbf y - \mathbf y_0$ and adding $\mathbf x_0$.

## References

- M. Berz, *Modern Map Methods in Particle Beam Physics*, Advances in Imaging
  and Electron Physics, Vol. 108, Academic Press, 1999 — differential algebra
  and the fixed-point inversion of polynomial maps.
- A. Griewank and A. Walther, *Evaluating Derivatives: Principles and Techniques
  of Algorithmic Differentiation*, 2nd ed., SIAM, 2008 — Taylor-coefficient
  arithmetic and series composition underlying the iteration.
- P. Henrici, *Applied and Computational Complex Analysis, Vol. 1*, Wiley,
  1974 — reversion (inversion) of formal power series.
