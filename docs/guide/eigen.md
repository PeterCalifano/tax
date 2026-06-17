# Eigen Integration

`tax::TaylorExpansion` ships with a `NumTraits` specialisation so it behaves as
a first-class Eigen scalar. Anywhere you can write
`Eigen::Matrix<double, R, C>`, you can write
`Eigen::Matrix<tax::TE<N, M>, R, C>` and Eigen's linear-algebra machinery just
works — at the cost of one Cauchy product per scalar multiplication.

```cpp
#include <tax/tax.hpp>

using TE2 = tax::TE<3, 2>;
auto x = TE2::variable<0>({1.0, 2.0});
auto y = TE2::variable<1>({1.0, 2.0});

Eigen::Vector2<TE2> F = { tax::sin(x) * tax::cos(y),
                          tax::exp(x + y) };

auto val = tax::value(F);             // Eigen::Vector2d of constant terms
auto J   = tax::jacobian(F);          // 2×2 Jacobian
auto vp  = tax::eval(F, Eigen::Vector2d{0.05, -0.1});  // displace + evaluate
```

The complete signature list lives in the [Eigen API reference](../reference/eigen.md);
this page is the how-to.

---

## What's provided

| Capability | Helper |
|---|---|
| Build coordinate variables as an Eigen vector | `tax::variables<TE>(x0)` |
| Strip constant terms | `tax::value(F)` |
| Evaluate at a displacement | `tax::eval(F, dx)` |
| Element-wise compile-time partial | `tax::derivative<Alpha...>(F)` |
| Gradient of a scalar TE | `tax::gradient(f)` |
| Hessian of a scalar TE | `tax::hessian(f)` |
| Jacobian of a vector TE | `tax::jacobian(F)` |
| Local map inverse | `tax::invert(F)` |

All helpers are free functions in `namespace tax` and work uniformly with
Dense and Sparse TE scalars.

---

## NumTraits

The specialisation in `tax/eigen.hpp` reports `IsComplex = 0`, `IsInteger = 0`,
`IsSigned = 1`, `RequireInitialization = 1`, and cost estimates proportional to
the monomial count $\binom{N+M}{M}$. This makes Eigen prefer cache-friendly
algorithms when working with TE-valued matrices.

You can use any Eigen routine that does not require additional traits
(determinant via `FullPivLU`, dense matrix products, `.norm()` on a vector,
fixed-size linear solves). Sparse-matrix routines are not (yet) wired up.

---

## Vector-valued functions

A natural use is propagating a Taylor expansion of a vector function — for
example, the right-hand side of an ODE:

```cpp
// f : R^2 → R^2,  f(x, y) = (x² − y, x · y)
auto f = [](const auto& v) {
    using S = std::decay_t<decltype(v)>;
    S out;
    out(0) = v(0) * v(0) - v(1);
    out(1) = v(0) * v(1);
    return out;
};

auto v = tax::variables<tax::TE<3, 2>>(Eigen::Vector2d{1.0, 0.5});
auto F = f(v);                        // Eigen::Vector2<TE<3, 2>>

auto J = tax::jacobian(F);            // 2×2 Jacobian at (1, 0.5)
auto H0 = tax::hessian(F(0));         // 2×2 Hessian of F₀ at (1, 0.5)
```

The same idea drives generic ODE integrators built on `tax`: a user-written
generic RHS lambda can be instantiated on plain `Eigen::Matrix<T, D, 1>` for
the RK steppers and on `Eigen::Matrix<tax::TE<N, M>, D, 1>` for the Taylor
stepper without source changes.

---

## Map inversion

`tax::invert(F)` formally inverts a square polynomial map
$\mathbf{P} : \mathbb{R}^M \to \mathbb{R}^M$ by Picard iteration on its
non-constant part:

```cpp
auto v = tax::variables<tax::TE<5, 2>>(Eigen::Vector2d::Zero());
Eigen::Vector2<tax::TE<5, 2>> F = { v(0) + v(0)*v(1),
                                    v(1) + 0.5 * v(0)*v(0) };

auto Finv = tax::invert(F);
// Composition Finv ∘ F is the identity up to order N
```

The linear part of the map must be invertible; otherwise `invert` throws
`std::invalid_argument`. The inverse is returned in the same Eigen shape as
the input.

### The math

The constant term of every component is dropped first, so `invert` works on an
**origin-centered perturbation map** $\mathbf{P}$ with $\mathbf{P}(\mathbf 0)=\mathbf 0$.
Split it into its linear and nonlinear parts,

$$
\mathbf{P}(\boldsymbol{\delta x}) = \underbrace{J\,\boldsymbol{\delta x}}_{\text{linear}}
  \; + \; \underbrace{\mathbf{N}(\boldsymbol{\delta x})}_{\text{order} \ge 2},
\qquad
J = \left.\frac{\partial \mathbf{P}}{\partial \boldsymbol{\delta x}}\right|_{\mathbf 0},
$$

where $J$ is the Jacobian of the linear part and $\mathbf{N}$ collects all
monomials of degree $\ge 2$. We seek the inverse map $\mathbf{G}$ with
$\mathbf{P}\big(\mathbf{G}(\mathbf y)\big) = \mathbf y$. Substituting the split
and solving for the linear occurrence of $\mathbf{G}$ gives the fixed-point form

$$
\mathbf{G}(\mathbf y) = J^{-1}\Big(\mathbf y - \mathbf{N}\big(\mathbf{G}(\mathbf y)\big)\Big).
$$

This is solved by **Picard iteration**, starting from the linear inverse:

$$
\mathbf{G}_0(\mathbf y) = J^{-1}\mathbf y,
\qquad
\mathbf{G}_{k+1}(\mathbf y) = J^{-1}\Big(\mathbf y - \mathbf{N}\big(\mathbf{G}_k(\mathbf y)\big)\Big).
$$

Because $\mathbf{N}$ has no constant or linear part, composing it with
$\mathbf{G}_k$ raises the lowest order at which $\mathbf{G}_{k+1}$ can still be
wrong by at least one. The degree-1 coefficients are already exact in
$\mathbf{G}_0$, so iterating $N-1$ times makes $\mathbf{G}$ exact through order
$N$ — and the truncation discards everything beyond, so the iteration
**terminates** rather than merely converging. Formal left and right inverses of
a truncated series coincide, so $\mathbf{G}\circ\mathbf{P} = \mathbf{P}\circ\mathbf{G} = \mathrm{id}$
to order $N$.

`invert` requires $J$ to be invertible (checked with a full-pivot LU; a singular
linear part throws). It inverts the **perturbation** part only: for a flow map
$\mathbf F(\mathbf x_0 + \boldsymbol{\delta x}) = \mathbf y_0 + \mathbf{P}(\boldsymbol{\delta x})$,
recover the displacement-to-displacement inverse by evaluating the returned map
at $\mathbf y - \mathbf y_0$ and adding $\mathbf x_0$ yourself.
