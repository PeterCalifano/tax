# Extracting Results

Once you have an expression, a `TaylorExpansion` exposes everything the single
evaluation pass computed: the function value, the raw coefficients, the scaled
derivatives, symbolic derivatives and integrals, and a polynomial evaluator.
Every snippet assumes `#include <tax/tax.hpp>`.

---

## The value

`value()` returns the constant term — the function evaluated at the expansion
point.

```cpp
auto x = tax::TE<5>::variable(0.0);
tax::TE<5> f = tax::sin(x);

double v = f.value();   // sin(0) = 0
```

---

## Coefficients vs derivatives

A Taylor expansion stores **coefficients** of the monomial basis:

$$
f(\mathbf{x}_0 + \delta\mathbf{x})
  = \sum_{|\alpha| \le N} f_\alpha \, \delta\mathbf{x}^\alpha
$$

The relationship to partial derivatives is

$$
f_\alpha
  = \frac{1}{\alpha!} \,
    \frac{\partial^{|\alpha|} f}
         {\partial x_1^{\alpha_1} \cdots \partial x_M^{\alpha_M}}
    \bigg|_{\mathbf{x}_0}
$$

so `derivative()` returns `coeff()` multiplied by $\alpha!$. Reach for
`coeff(...)` when you want the raw Taylor coefficient (e.g. to feed another
series), and `derivative(...)` when you want an actual partial derivative.

=== "Univariate"

    ```cpp
    auto x = tax::TE<5>::variable(0.0);
    tax::TE<5> f = tax::sin(x);

    double c1 = f.coeff({1});        // 1                (1st Taylor coefficient)
    double c3 = f.coeff({3});        // -1/6
    double d1 = f.derivative({1});   // cos(0)  = 1
    double d3 = f.derivative({3});   // -cos(0) = -1     (= 3! · c3)
    ```

=== "Multivariate"

    ```cpp
    using TE2 = tax::TE<3, 2>;
    const std::array<double, 2> p{1.0, 2.0};
    auto x = TE2::variable<0>(p);
    auto y = TE2::variable<1>(p);

    TE2 g = x*x*y;

    double c_200    = g.coeff({2, 0});      // coefficient of δx²
    double c_110    = g.coeff({1, 1});      // coefficient of δx·δy
    double c_110_ct = g.coeff<1, 1>();      // compile-time index access
    double d_110    = g.derivative<1, 1>(); // ∂²g/∂x∂y at (1, 2)
    ```

`coeff`, `derivative`, `deriv`, and `integ` all come in compile-time
(`<...>`), `MultiIndex<M>`, and runtime-`int` forms; the multi-index `{...}`
spelling is shown above.

---

## Symbolic differentiation and integration

`deriv<I>()` and `integ<I>()` return a *new* expansion that is the symbolic
partial derivative or integral with respect to coordinate `I`.

```cpp
using TE2 = tax::TE<4, 2>;
const std::array<double, 2> p{1.0, 2.0};
auto x = TE2::variable<0>(p);
auto y = TE2::variable<1>(p);

TE2 f = x*x*y + y*y;

auto df_dx = f.deriv<0>();    // ∂f/∂x  = 2xy
auto df_dy = f.deriv<1>();    // ∂f/∂y  = x² + 2y

auto F_x   = f.integ<0>();    // ∫f dx
auto F_y   = f.integ<1>();    // ∫f dy
```

The coordinate index can also be supplied at runtime:

```cpp
auto x = tax::TE<5>::variable(1.0);
tax::TE<5> f = tax::exp(x);

auto df = f.deriv(0);          // d/dx exp(x)
auto F  = f.integ(0);          // ∫ exp(x) dx
```

!!! tip "Verifying an identity"
    Symbolic differentiation recovers the analytic derivative term by term:

    ```cpp
    auto x = tax::TE<6>::variable(0.5);
    tax::TE<6> f = tax::sin(x);

    tax::TE<6> df       = f.deriv<0>();
    tax::TE<6> expected = tax::cos(tax::TE<6>::variable(0.5));
    // The coefficient arrays of df and expected agree to machine precision.
    ```

---

## Polynomial evaluation

`eval()` Horner-evaluates the truncated Taylor polynomial at a displacement
$\delta x$ from the expansion point.

```cpp
auto x = tax::TE<15>::variable(0.0);
tax::TE<15> f = tax::sin(x);

double approx = f.eval({0.3});   // sin(0.3) within machine precision
```

Multivariate evaluation takes one displacement per coordinate:

```cpp
using TE2 = tax::TE<5, 2>;
const std::array<double, 2> p{0.0, 0.0};
auto x = TE2::variable<0>(p);
auto y = TE2::variable<1>(p);

TE2 f = tax::sin(x) * tax::cos(y);
double approx = f.eval({0.3, 0.5});   // ≈ sin(0.3) * cos(0.5)
```

---

For vector- and matrix-valued results — gradients, Hessians, and Jacobians of
Eigen-shaped expansions — see [Eigen Integration](eigen.md). The graded-lex
coefficient ordering and the theory behind the $f_\alpha$ relationship are
covered in [Foundations](../concepts/foundations.md); every method signature is
listed in the [Core API Reference](../reference/core.md).
