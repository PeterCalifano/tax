# Recurrence Relations

This page is the **recurrence reference** for **tax**: for every supported operation it gives the degree-by-degree recurrence relation used to propagate truncated Taylor polynomials. Each entry lists the univariate ($M = 1$) form and its multivariate generalisation, matching the kernels in `include/tax/kernels/`. See [Kernels & Recurrences](kernels.md) for how these are dispatched and implemented.

For the underlying theory — what truncated Taylor polynomials are, the graded-lexicographic coefficient ordering, and how univariate recurrences extend to many variables — see [Foundations & Ordering](../concepts/foundations.md). For truncation-error bounds and convergence diagnostics, see [Convergence & Truncation](../concepts/convergence.md).

---

## Shared recurrence drivers

Nearly every transcendental recurrence below is an instance of one of **two**
degree-by-degree shapes, implemented once in `tax/kernels/algebra.hpp` as the
drivers `seriesDerivProduct` and `seriesDerivQuotient`. Each individual kernel
reduces to "compute the helper series $h$, seed the constant term, call the
driver". These are the classic Taylor-coefficient recurrences of forward-mode
algorithmic differentiation (Griewank & Walther [2], Chapter 13; the
asymptotically fast alternatives for very high orders are due to Brent & Kung
[5], which tax does not need at the moderate orders it targets).

Both drivers come from the same idea: for $f = g(u)$ with $u$ a truncated
series, differentiate the defining identity once to obtain a first-order ODE
in the series coefficients, then match coefficients degree by degree.

### `seriesDerivProduct` — solve $f' = u' \cdot h$

Used when $g'(u)$ is a series $h$ that is *known* (or, for `exp`, is the
output itself):

| Function | $h$ |
|---|---|
| $\exp(u)$ | $h = f$ itself (the output, read only at lower degree) |
| $\operatorname{erf}(u)$ | $h = \tfrac{2}{\sqrt{\pi}}\, e^{-u^2}$ |

Matching coefficients of $f' = u'h$ at degree $d$ gives the **univariate** form

$$
f_d = \frac{1}{d} \sum_{k=1}^{d} k \, u_k \, h_{d-k}, \qquad d \ge 1,
$$

with the seed $f_0 = g(u_0)$. The **multivariate** form replaces the
convolution by the graded decomposition rows $\beta + \gamma = \alpha$ with
$|\beta| \ge 1$, weighted by $|\beta|$:

$$
f_\alpha = \frac{1}{|\alpha|} \sum_{\substack{\beta + \gamma = \alpha \\ |\beta| \ge 1}} |\beta| \, u_\beta \, h_\gamma .
$$

For `exp`, $h$ *is* $f$: the rows only ever read $h$ at strictly lower total
degree, which is already final, so no copy is needed.

### `seriesDerivQuotient` — solve $h \cdot f' = \pm u'$

Used when $1/g'(u)$ is the cheap side — the whole
"integral of a quotient" family. The helper $h$ and sign per function:

| Function | $h$ | Sign |
|---|---|:-:|
| $\log(u)$ | $h = u$ | $+$ |
| $\arcsin(u)$ | $h = \sqrt{1 - u^2}$ | $+$ |
| $\arccos(u)$ | $h = \sqrt{1 - u^2}$ | $-$ |
| $\arctan(u)$ | $h = 1 + u^2$ | $+$ |
| $\operatorname{atan2}(y, x)$ | $h = 1 + r^2$ with $r = y/x$ | $+$ |
| $\operatorname{asinh}(u)$ | $h = \sqrt{1 + u^2}$ | $+$ |
| $\operatorname{acosh}(u)$ | $h = \sqrt{u^2 - 1}$ | $+$ |
| $\operatorname{atanh}(u)$ | $h = 1 - u^2$ | $+$ |

Matching coefficients of $h f' = \pm u'$ at degree $d$ and solving for the
top coefficient gives the **univariate** form

$$
f_d = \frac{1}{h_0} \left( \pm u_d - \frac{1}{d} \sum_{k=1}^{d-1} k \, h_{d-k} \, f_k \right), \qquad d \ge 1,
$$

with $f_0 = g(u_0)$ and the requirement $h_0 \ne 0$. The **multivariate**
form walks the same decomposition rows, weighting each $(\beta, \gamma)$
pair by $|\gamma| = |\alpha| - |\beta|$ (entries with $|\beta| = |\alpha|$
carry weight zero and drop out):

$$
f_\alpha = \frac{1}{h_0} \left( \pm u_\alpha - \frac{1}{|\alpha|} \sum_{\substack{\beta + \gamma = \alpha \\ |\beta| \ge 1}} (|\alpha| - |\beta|) \, h_\beta \, f_\gamma \right).
$$

The per-function sections below list every recurrence in this closed form;
where a function is implemented through a driver, the entry says so.

---

## Arithmetic Operations

### Addition and Subtraction

Addition is coefficient-wise:

$$
(f + g)_\alpha = f_\alpha + g_\alpha
$$

Subtraction is analogous. Scalar addition modifies only the constant term: $(f + c)_\alpha = f_\alpha + c \cdot \delta_{\alpha,0}$.

### Cauchy Product (Multiplication)

**Univariate.** The product of two truncated series is the discrete convolution truncated at order $N$:

$$
(f \cdot g)_d = \sum_{k=0}^{d} f_k \, g_{d-k}, \qquad d = 0, \ldots, N
$$

**Multivariate.** The Cauchy product generalizes to a sum over sub-multi-indices:

$$
(f \cdot g)_\alpha = \sum_{\beta \le \alpha} f_\beta \, g_{\alpha - \beta}
$$

where $\beta \le \alpha$ means $\beta_i \le \alpha_i$ for all $i$.

The library exploits **symmetry** in the self-product $f \cdot f$: only unordered pairs $(\beta, \alpha - \beta)$ with $\beta \le \alpha - \beta$ (in flat index) are enumerated, roughly halving the number of multiplications.

### Scalar Multiplication and Division

Scalar multiplication scales all coefficients: $(c \cdot f)_\alpha = c \cdot f_\alpha$. Division by a scalar is multiplication by $1/c$. Division by a polynomial uses the reciprocal recurrence (see below).

---

## Algebraic Operations

### Reciprocal

Given $f$ with $f_0 \ne 0$, compute $g = 1/f$ by solving $f \cdot g = 1$ degree by degree.

**Univariate:**

$$
g_0 = \frac{1}{f_0}, \qquad g_d = -\frac{1}{f_0} \sum_{k=1}^{d} f_k \, g_{d-k}, \quad d \ge 1
$$

**Multivariate:**

$$
g_\alpha = \frac{1}{f_0} \left( \delta_{\alpha,0} - \sum_{\substack{\beta \le \alpha \\ 0 < |\beta| \le |\alpha|}} f_\beta \, g_{\alpha-\beta} \right)
$$

### Square Root

Given $f$ with $f_0 > 0$, compute $g = \sqrt{f}$ by solving $g^2 = f$.

**Univariate:**

$$
g_0 = \sqrt{f_0}, \qquad g_d = \frac{1}{2g_0} \left( f_d - \sum_{k=1}^{d-1} g_k \, g_{d-k} \right), \quad d \ge 1
$$

The inner sum exploits symmetry: for even $d$, the middle term $g_{d/2}^2$ is counted once; other pairs $(k, d-k)$ are counted twice.

**Multivariate:**

$$
g_\alpha = \frac{1}{2g_0} \left( f_\alpha - \sum_{\substack{\beta \le \alpha \\ 0 < |\beta| < |\alpha|}} g_\beta \, g_{\alpha - \beta} \right)
$$

with symmetric enumeration: pairs $(\beta, \alpha - \beta)$ with flat index $\beta < \alpha - \beta$ are counted twice; diagonal pairs ($\beta = \alpha - \beta$) are counted once.

### Cubic Root

Given $f$ with $f_0 \ne 0$, compute $g = \sqrt[3]{f}$ by solving $g^3 = f$.

**Univariate:**

$$
g_0 = \sqrt[3]{f_0}, \qquad g_d = \frac{1}{3g_0^2} \left( f_d - g_0 \cdot q_d^* - \sum_{j=1}^{d-1} g_j \, q_{d-j} \right), \quad d \ge 1
$$

where $q = g^2$ is maintained incrementally: $q_d^* = \sum_{k=1}^{d-1} g_k \, g_{d-k}$ is the partial self-product (excluding the unknown $g_d$), then finalized as $q_d = 2 g_0 g_d + q_d^*$. This yields $\mathcal{O}(N^2)$ total work instead of $\mathcal{O}(N^3)$.

**Multivariate:**

$$
g_\alpha = \frac{1}{3g_0^2} \left( f_\alpha - \sum_{\substack{\beta \le \alpha \\ 0 < |\beta| < |\alpha|}} g_\beta \bigl( g_0 \, g_{\alpha-\beta} + q_{\alpha-\beta} \bigr) \right)
$$

with $q = g^2$ updated degree by degree using symmetric enumeration.

---

## Trigonometric Functions

### Sine and Cosine

The sine and cosine of a series $f$ are computed simultaneously via the coupled recurrence. Let $s = \sin(f)$ and $c = \cos(f)$.

**Univariate:**

$$
s_0 = \sin(f_0), \quad c_0 = \cos(f_0)
$$

$$
s_d = \frac{1}{d} \sum_{k=0}^{d-1} (d - k) \, f_{d-k} \, c_k, \qquad d \ge 1
$$

$$
c_d = -\frac{1}{d} \sum_{k=0}^{d-1} (d - k) \, f_{d-k} \, s_k, \qquad d \ge 1
$$

**Multivariate:**

$$
s_\alpha = \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 0 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, f_{\alpha-\beta} \, c_\beta
$$

$$
c_\alpha = -\frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 0 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, f_{\alpha-\beta} \, s_\beta
$$

### Tangent

Tangent is computed by solving $c \cdot t = s$ degree by degree, where $s = \sin(f)$ and $c = \cos(f)$ are obtained from the coupled recurrence above.

**Univariate:**

$$
t_d = \frac{1}{c_0} \left( s_d - \sum_{k=1}^{d} c_k \, t_{d-k} \right), \qquad d \ge 0
$$

**Multivariate:**

$$
t_\alpha = \frac{1}{c_0} \left( s_\alpha - \sum_{\substack{\beta \le \alpha \\ 0 < |\beta| \le |\alpha|}} c_\beta \, t_{\alpha-\beta} \right)
$$

### Arcsine

Compute $g = \arcsin(f)$ using the helper $h = \sqrt{1 - f^2}$. This reduces to solving $h \cdot g' = f'$ degree by degree — a direct `seriesDerivQuotient` instance (see [Shared recurrence drivers](#shared-recurrence-drivers)).

**Univariate:**

$$
g_0 = \arcsin(f_0), \qquad g_d = \frac{1}{h_0} \left( f_d - \frac{1}{d} \sum_{k=1}^{d-1} k \, h_{d-k} \, g_k \right), \quad d \ge 1
$$

**Multivariate:**

$$
g_\alpha = \frac{1}{h_0} \left( f_\alpha - \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, h_\beta \, g_{\alpha-\beta} \right)
$$

### Arccosine

Since $\arccos'(f) = -\arcsin'(f)$, arccosine is the *same* driver call as
arcsine with the sign flipped: solve $h \cdot g' = -f'$ with
$h = \sqrt{1 - f^2}$ (that is, `seriesDerivQuotient` with `Sign = -1`) and
seed the constant term with $g_0 = \arccos(f_0)$. The nonconstant
coefficients are exactly the negated arcsine coefficients, consistent with
$\arccos(f) = \pi/2 - \arcsin(f)$.

### Arctangent

Compute $g = \arctan(f)$ using the helper $h = 1 + f^2$. Solves $h \cdot g' = f'$ degree by degree (`seriesDerivQuotient`).

**Univariate:**

$$
g_0 = \arctan(f_0), \qquad g_d = \frac{1}{h_0} \left( f_d - \frac{1}{d} \sum_{k=1}^{d-1} k \, h_{d-k} \, g_k \right), \quad d \ge 1
$$

**Multivariate:**

$$
g_\alpha = \frac{1}{h_0} \left( f_\alpha - \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, h_\beta \, g_{\alpha-\beta} \right)
$$

### Arctangent (Two-Argument)

Compute $g = \text{atan2}(y, x)$ by first forming the ratio series
$r = y / x$ in a single forward-substitution pass (requires $x_0 \ne 0$),
then running the arctangent driver on $r$: solve $h \cdot g' = r'$ with
$h = 1 + r^2$, seeded with the correct-quadrant constant term
$g_0 = \text{atan2}(y_0, x_0)$.

Only the seed differs from plain $\arctan(y/x)$ — the nonconstant
coefficients are identical, since $\text{atan2}$ and $\arctan \circ (y/x)$
differ by a locally constant multiple of $\pi$.

**Univariate:**

$$
g_0 = \text{atan2}(y_0, x_0), \qquad g_d = \frac{1}{h_0} \left( r_d - \frac{1}{d} \sum_{k=1}^{d-1} k \, h_{d-k} \, g_k \right), \quad d \ge 1
$$

**Multivariate:**

$$
g_\alpha = \frac{1}{h_0} \left( r_\alpha - \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, h_\beta \, g_{\alpha-\beta} \right)
$$

---

## Hyperbolic Functions

### Hyperbolic Sine and Cosine

$\sinh(f)$ and $\cosh(f)$ are computed from one shared exponential pair.
Two `seriesDerivProduct` passes (one negation apart) produce
$e^{f}$ and $e^{-f}$, and the results are the half sum / half difference:

$$
p = \exp(f), \qquad m = \exp(-f)
$$

$$
\text{sh}_\alpha = \tfrac{1}{2} (p_\alpha - m_\alpha), \qquad
\text{ch}_\alpha = \tfrac{1}{2} (p_\alpha + m_\alpha), \qquad |\alpha| \ge 1,
$$

with the constant terms seeded directly as $\text{sh}_0 = \sinh(f_0)$,
$\text{ch}_0 = \cosh(f_0)$ (rather than through the half sums, to keep the
constant term at full scalar accuracy).

The fused `sinhCosh(x)` returns both series from the single shared pair;
the single-output `sinh`/`cosh` kernels run the same pair but write only the
requested combination — writing a discarded companion measurably costs at
small $N$.

### Hyperbolic Tangent

Computed by solving $\text{ch} \cdot t = \text{sh}$ degree by degree, identical in structure to the tangent recurrence.

**Univariate:**

$$
t_d = \frac{1}{\text{ch}_0} \left( \text{sh}_d - \sum_{k=1}^{d} \text{ch}_k \, t_{d-k} \right), \qquad d \ge 0
$$

**Multivariate:**

$$
t_\alpha = \frac{1}{\text{ch}_0} \left( \text{sh}_\alpha - \sum_{\substack{\beta \le \alpha \\ 0 < |\beta| \le |\alpha|}} \text{ch}_\beta \, t_{\alpha-\beta} \right)
$$

### Inverse Hyperbolic Sine

Compute $g = \text{asinh}(f)$ using $h = \sqrt{1 + f^2}$. Solves $h \cdot g' = f'$ (`seriesDerivQuotient`).

**Univariate:**

$$
g_0 = \text{asinh}(f_0), \qquad g_d = \frac{1}{h_0} \left( f_d - \frac{1}{d} \sum_{k=1}^{d-1} k \, h_{d-k} \, g_k \right), \quad d \ge 1
$$

**Multivariate:**

$$
g_\alpha = \frac{1}{h_0} \left( f_\alpha - \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, h_\beta \, g_{\alpha-\beta} \right)
$$

### Inverse Hyperbolic Cosine

Compute $g = \text{acosh}(f)$ using $h = \sqrt{f^2 - 1}$. Requires $f_0 > 1$. Same driver call as asinh.

**Univariate:**

$$
g_0 = \text{acosh}(f_0), \qquad g_d = \frac{1}{h_0} \left( f_d - \frac{1}{d} \sum_{k=1}^{d-1} k \, h_{d-k} \, g_k \right), \quad d \ge 1
$$

**Multivariate:**

$$
g_\alpha = \frac{1}{h_0} \left( f_\alpha - \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, h_\beta \, g_{\alpha-\beta} \right)
$$

### Inverse Hyperbolic Tangent

Compute $g = \text{atanh}(f)$ using $h = 1 - f^2$. Requires $|f_0| < 1$. Same driver call.

**Univariate:**

$$
g_0 = \text{atanh}(f_0), \qquad g_d = \frac{1}{h_0} \left( f_d - \frac{1}{d} \sum_{k=1}^{d-1} k \, h_{d-k} \, g_k \right), \quad d \ge 1
$$

**Multivariate:**

$$
g_\alpha = \frac{1}{h_0} \left( f_\alpha - \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, h_\beta \, g_{\alpha-\beta} \right)
$$

---

## Transcendental Functions

### Exponential

Compute $g = \exp(f)$.

**Univariate:**

$$
g_0 = \exp(f_0), \qquad g_d = \frac{1}{d} \sum_{k=0}^{d-1} (d - k) \, f_{d-k} \, g_k, \quad d \ge 1
$$

**Multivariate:**

$$
g_\alpha = \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| \le |\alpha|}} |\beta| \, f_\beta \, g_{\alpha-\beta}
$$

This recurrence follows from differentiating $g = \exp(f)$ to get $g' = f' \cdot g$, then matching coefficients degree by degree — it is `seriesDerivProduct` with $h = g$ itself (the driver only ever reads $h$ at strictly lower total degree, which is already final).

### Logarithm

Compute $g = \ln(f)$ with $f_0 > 0$.

**Univariate:**

$$
g_0 = \ln(f_0), \qquad g_d = \frac{1}{f_0} \left( f_d - \frac{1}{d} \sum_{k=1}^{d-1} k \, f_{d-k} \, g_k \right), \quad d \ge 1
$$

**Multivariate:**

$$
g_\alpha = \frac{1}{f_0} \left( f_\alpha - \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| < |\alpha|}} (|\alpha| - |\beta|) \, f_\beta \, g_{\alpha-\beta} \right)
$$

This is derived from $f \cdot g' = f'$, matching coefficients — `seriesDerivQuotient` with $h = f$.

---

## Power Functions

### Integer Power

For integer exponent $n$, $f^n$ is computed via **binary exponentiation** using the Cauchy product. Special cases: $n = 0$ returns 1, $n = 1$ returns $f$, $n = -1$ uses the reciprocal recurrence, and negative $n$ computes the reciprocal first, then raises to $|n|$.

### Real Power

Compute $g = f^c$ for real exponent $c$ with $f_0 \ne 0$.

**Univariate:**

$$
g_0 = f_0^c, \qquad g_d = \frac{1}{d \cdot f_0} \sum_{k=0}^{d-1} \bigl( c(d-k) - k \bigr) \, f_{d-k} \, g_k, \quad d \ge 1
$$

**Multivariate:**

$$
g_\alpha = \frac{1}{|\alpha| \cdot f_0} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| \le |\alpha|}} \bigl( c \cdot |\beta| - (|\alpha| - |\beta|) \bigr) \, f_\beta \, g_{\alpha-\beta}
$$

This recurrence is derived from the identity $f \cdot g' = c \cdot f' \cdot g$.

### Half-Integer Powers

`halfPow<K>(f)` computes $f^{K/2}$ by a compile-time dispatch: even $K$
routes to the integer-power chain above (valid for $f_0 < 0$, and requiring
$f_0 \ne 0$ only when $K < 0$); odd $K$ runs a single real-power recurrence
with $c = K/2$ (requiring $f_0 > 0$). `invSqrtPow<K>(f)` is
`halfPow<-K>(f)` with $K \ge 1$, i.e. $f^{-K/2}$. One `seriesPow` pass is
the fastest single-output spelling of a half-integer power; the joint
[`sqrtInvSqrt`](#joint-square-root-and-inverse-square-root) pass below wins
only when both $\sqrt{f}$ and $1/\sqrt{f}$ are consumed.

---

## Fused Pair Recurrences

Some operation pairs share so much recurrence structure that computing both
in one pass costs barely more than computing one. Sine/cosine and
sinh/cosh are already coupled internally (see above); the two genuinely
fused kernels are the exp·trig product and the joint square root / inverse
square root (`tax/kernels/fused.hpp`).

### Fused Exponential-Trigonometric Product

For $h = e^{v} \cos u$ and $q = e^{v} \sin u$, differentiating gives the
coupled linear system

$$
h' = v' h - u' q, \qquad q' = v' q + u' h .
$$

Matching coefficients degree by degree yields the joint recurrence
(univariate; multivariate uses the graded decomposition rows exactly as in
the [drivers](#shared-recurrence-drivers)):

$$
h_0 = e^{v_0} \cos u_0, \qquad q_0 = e^{v_0} \sin u_0
$$

$$
h_d = \frac{1}{d} \sum_{k=0}^{d-1} \bigl( (d-k) \, v_{d-k} \, h_k - (d-k) \, u_{d-k} \, q_k \bigr), \qquad d \ge 1
$$

$$
q_d = \frac{1}{d} \sum_{k=0}^{d-1} \bigl( (d-k) \, v_{d-k} \, q_k + (d-k) \, u_{d-k} \, h_k \bigr), \qquad d \ge 1
$$

One coupled pass replaces three recurrences plus a Cauchy product
(`exp(v)`, the coupled `sin(u)`/`cos(u)`, and the multiply). The public
surface is `expSin(v, u)`, `expCos(v, u)`, and the pair-returning
`expSinCos(v, u)`; the single-output forms run the same coupled pass and
keep the companion kernel-internal.

### Joint Square Root and Inverse Square Root

`sqrtInvSqrt(f)` interleaves two forward substitutions per degree
(requires $f_0 > 0$): first the square-root coefficient $s_\alpha$ from
$s^2 = f$ (exactly the [square-root recurrence](#square-root)), then the
inverse-square-root coefficient $r_\alpha$ from $r \cdot s = 1$ using the
just-finalised $s_\alpha$:

$$
s_0 = \sqrt{f_0}, \qquad r_0 = \frac{1}{s_0}
$$

$$
r_\alpha = -\frac{1}{s_0} \sum_{\substack{\beta \le \alpha \\ 0 \le |\beta| < |\alpha|}} r_\beta \, s_{\alpha-\beta}, \qquad |\alpha| \ge 1
$$

The extra output costs one forward substitution on top of the square root,
with scalar divisions by $s_0$ only — no second nonlinear recurrence and no
division series. It is a measured win only when **both** outputs are
consumed; a single-output caller should use `sqrt` or `halfPow`/`pow`.

---

## Special Functions

### Error Function

Compute $g = \text{erf}(f)$ using the helper:

$$
h = \frac{2}{\sqrt{\pi}} \exp(-f^2)
$$

which is the derivative of $\text{erf}$. Then the recurrence is `seriesDerivProduct` with this $h$ — the same shape as the exponential.

**Univariate:**

$$
g_0 = \text{erf}(f_0), \qquad g_d = \frac{1}{d} \sum_{k=0}^{d-1} (d-k) \, f_{d-k} \, h_k, \quad d \ge 1
$$

**Multivariate:**

$$
g_\alpha = \frac{1}{|\alpha|} \sum_{\substack{\beta \le \alpha \\ 1 \le |\beta| \le |\alpha|}} |\beta| \, f_\beta \, h_{\alpha-\beta}
$$

---

## Constant-term seeding

Every recurrence above evaluates exactly one scalar transcendental — the
constant-term seed $g_0 = g(f_0)$. At runtime this goes through
`std::`/ADL as usual; in constant evaluation it switches to the constexpr
implementations in `tax::detail::cmath`, computed in `long double`. The
results agree with libm to within a few ulp of `double` but are **not**
guaranteed bit-identical — see
[Kernels & Recurrences](kernels.md#constexpr-constant-term-seeding) for the
full accuracy contract.

---

## References

The degree-by-degree propagation of truncated Taylor series goes back to the
early days of validated numerics (Moore [1]); the driver recurrences of this
page are the classic forward-mode AD recurrences tabulated in Chapter 13 of
Griewank & Walther [2]. The multivariate, differential-algebra view of
truncated polynomials — the tradition tax belongs to — is due to Berz [3];
DACE [4] is the reference implementation of that approach, and the one tax's
regression suite compares against.

1. R. E. Moore, *Interval Analysis*, Prentice-Hall, 1966.
2. A. Griewank and A. Walther, *Evaluating Derivatives: Principles and
   Techniques of Algorithmic Differentiation*, 2nd ed., SIAM, 2008 — the
   series-recurrence tables for the elementary functions are Chapter 13.
3. M. Berz, *Modern Map Methods in Particle Beam Physics*, Academic Press,
   1999 — differential algebra (DA) of truncated multivariate Taylor
   polynomials.
4. M. Rasotto et al., *Differential Algebra Space Toolbox for Nonlinear
   Uncertainty Propagation in Space Dynamics* (the DACE library), 6th
   International Conference on Astrodynamics Tools and Techniques (ICATT),
   2016.
5. R. P. Brent and H. T. Kung, *Fast Algorithms for Manipulating Formal
   Power Series*, Journal of the ACM 25(4), 1978 — asymptotically fast
   power-series composition and inversion.
6. R. D. Neidinger, *Introduction to Automatic Differentiation and MATLAB
   Object-Oriented Programming*, SIAM Review 52(3), 545–563, 2010 — explicit
   degree-by-degree recurrences for the standard transcendental functions.
