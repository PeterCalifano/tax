# Fused Operations

Several common expression shapes — `sin` next to `cos`, `exp(v) * cos(u)`,
`sqrt(x)` next to `1/sqrt(x)` — share almost all of their recurrence work.
The fused surface computes such pairs in a **single coupled recurrence pass**
instead of two (or three) separate kernel calls plus a Cauchy product.

Every function on this page works for dense `TE`, named `NE`, and mixed-order
`MTE` expansions alike, and lives in `namespace tax` (reachable from
`<tax/tax.hpp>`). They are **runtime-only** (each seeds its recurrence with a
libm call at the constant term), not `constexpr`. Pair-returning functions
order the `std::pair` **as spelled in the name**: `sinCos → {sin, cos}`,
`sqrtInvSqrt → {sqrt, 1/sqrt}`, `expSinCos → {exp·sin, exp·cos}`.

---

## The fused surface at a glance

| Function | Computes | Typical speedup vs the composed spelling |
|---|---|---|
| `sinCos(x)`       | `{sin(x), cos(x)}`             | ~2x vs `sin(x)` + `cos(x)` |
| `sinhCosh(x)`     | `{sinh(x), cosh(x)}`           | ~1.8x vs `sinh(x)` + `cosh(x)` |
| `sqrtInvSqrt(x)`  | `{sqrt(x), 1/sqrt(x)}`         | ~1.3–1.5x univariate; parity at large multivariate sizes |
| `expSin(v, u)`    | `exp(v) * sin(u)`              | ~1.4–1.8x vs `exp(v) * sin(u)` |
| `expCos(v, u)`    | `exp(v) * cos(u)`              | ~1.4–1.8x vs `exp(v) * cos(u)` |
| `expSinCos(v, u)` | `{exp(v)*sin(u), exp(v)*cos(u)}` | both results for the price of one fused pass |
| `halfPow<K>(x)`   | `x^(K/2)`                      | one `pow` recurrence instead of `sqrt` + power chain |
| `invSqrtPow<K>(x)`| `x^(-K/2)`                     | likewise; `invSqrtPow<3>(r2)` is the 1/r³ gravity kernel |
| `pow<N>(x)`       | `x^N` (compile-time integer)  | integer Cauchy chain, `constexpr`; ~1.2x vs real `pow` |
| `pow<Num,Den>(x)` | `x^(Num/Den)`                 | reduces to the cheapest kernel; ~1.3x when it lands on the integer chain |
| `norm<P, Q>(v)`   | `‖v‖_P^Q` (Q defaults to 1)   | P-norm of a vector as one series; raises once to Q/P (~1.6x vs re-raising on 1/r³) |

The speedups are measured on the library's own benchmarks; the exp·trig
fusion replaces **three** recurrences plus a Cauchy product (`exp`, the
coupled `sin`/`cos`, and the multiply) with one coupled pass.

---

## Pair-returning functions and structured bindings

`sinCos`, `sinhCosh`, `sqrtInvSqrt`, and `expSinCos` return a
`std::pair` of expansions; structured bindings are the natural spelling:

```cpp
#include <tax/tax.hpp>

auto x = tax::TE<8>::variable(0.7);

auto [s, c]   = tax::sinCos(x);        // {sin(x), cos(x)} in one pass
auto [sh, ch] = tax::sinhCosh(x);      // {sinh(x), cosh(x)} — one shared exp pair

auto r2 = x * x + 1.0;
auto [r, ir]  = tax::sqrtInvSqrt(r2);  // {sqrt(r2), 1/sqrt(r2)}, requires r2.value() > 0
```

`sin`/`cos` were *already* computed by one coupled recurrence internally —
calling both `tax::sin(x)` and `tax::cos(x)` simply ran that recurrence
twice and discarded the companion each time. `sinCos` hands you both
results of the single pass, hence the ~2x.

!!! warning "`sqrtInvSqrt` only pays when BOTH outputs are consumed"
    The inverse square root costs one extra forward substitution on top of
    the `sqrt` pass — cheap, but not free. If you need only one of the two,
    call `tax::sqrt(x)` or `invSqrtPow<K>(x)` instead: computing the unused
    companion is a measured net loss. The classic profitable case is a
    radius where both `r` and `1/r`-powers appear in the same formula.

## Fused exp·trig

`expSin`, `expCos`, and `expSinCos` compute `exp(v)` times a trig function
of a *different* argument `u` — the damped-oscillation shape
$e^{v}\sin u$ / $e^{v}\cos u$ — in one coupled recurrence:

```cpp
auto t = tax::TE<10>::variable(0.0);
auto v = -0.5 * t;          // decay exponent
auto u = 3.0 * t;           // phase

tax::TE<10> d = tax::expCos(v, u);       // exp(v)*cos(u), one pass
auto [qs, qc] = tax::expSinCos(v, u);    // both quadratures at once
```

Use `expSin`/`expCos` when a single output is needed, `expSinCos` when both
are — the coupled pass computes both internally either way.

---

## Half-integer powers: `halfPow<K>` and `invSqrtPow<K>`

`halfPow<K>(x)` computes $x^{K/2}$ for a compile-time integer `K`, picking
the cheapest correct path at compile time:

- **Even `K`** dispatches to the integer-power chain (binary
  exponentiation) — valid for **negative constant terms** too, and requires
  `x.value() != 0` only when `K` is negative.
- **Odd `K`** runs the single real-exponent `pow` recurrence — requires
  `x.value() > 0`.

`invSqrtPow<K>(x)` is the spelling for $x^{-K/2} = 1/\sqrt{x}^{\,K}$ with
`K >= 1` (a `static_assert` enforces it); it requires `x.value() > 0`.

```cpp
// Point-mass gravity: a = -mu * r / |r|^3, with r2 = x² + y² + z².
auto ir3 = tax::invSqrtPow<3>(r2);   // r2^(-3/2) — one recurrence pass
auto ax  = -mu * x * ir3;

auto v3  = tax::halfPow<3>(r2);      // r2^(3/2) = |r|³
auto s   = tax::halfPow<-4>(r2);     // r2^(-2), integer chain: r2.value() < 0 is fine
```

One `seriesPow` pass is the fastest single-output spelling — it beats the
fused `sqrtInvSqrt` pair plus a power chain whenever only one output is
consumed. A caller that needs `sqrt(x)` *alongside* `x^(-K/2)` should
combine `sqrtInvSqrt` with `pow` instead.

### Compile-time powers: `pow<N>` and `pow<Num, Den>`

`halfPow`/`invSqrtPow` are the `Den == 2` face of a general compile-time power:

- `pow<N>(x)` — integer power with the exponent in the type. Same integer
  Cauchy chain as `pow(x, N)`, `constexpr`, no libm. Prefer it whenever the
  exponent is a constant.
- `pow<Num, Den>(x)` — the rational power $x^{\text{Num}/\text{Den}}$. The
  exponent is reduced by `gcd` at compile time and bound to the cheapest
  kernel: the integer chain when `Den | Num` (so `pow<6,3>` becomes `pow<2>`,
  ~1.3× faster than a real-exponent `pow`), the sqrt/invsqrt chain when the
  reduced denominator is 2 (`pow<3,2>` == `halfPow<3>`, `pow<-3,2>` ==
  `invSqrtPow<3>`), otherwise a single real-exponent recurrence
  (`pow<2,5>` == $x^{2/5}$).

For a genuine fractional exponent there is nothing faster than that single
`seriesPow` pass — it out-measures both a dedicated root (`cbrt`) and
`sqrt`-then-integer-power, because $x^{c}$ obeys the same degree-by-degree
recurrence whatever `c` is; only the scalar constant term needs a libm seed.

---

## Vector norms: `norm<P, Q>`

`tax::norm<P, Q>(v)` is the $Q$-th power of the $P$-norm of a vector of
expansions, as a single series. `Q` defaults to 1 (the plain $P$-norm) and
`P` to the Euclidean 2-norm:

$$
\operatorname{norm}\langle P, Q\rangle(v) = \lVert v\rVert_P^{\,Q}
   = \Big(\textstyle\sum_i v_i^{P}\Big)^{Q/P}.
$$

The input is an Eigen column vector of expansions (as returned by
`tax::la::variables`) or any range of them (e.g. the `std::array` from
`tax::variables`); the elements may be dense `TE`, named `NE`, or mixed `MTE`.

```cpp
Eigen::Vector3d r0{3.0, 4.0, 12.0};
auto r = tax::la::variables<tax::TE<8, 3>>(r0);   // vector of coordinate variables

auto len = tax::norm(r);          // sqrt(x²+y²+z²)  — value 13 at r0   (P=2, Q=1)
auto n3  = tax::norm<3>(r);       // (Σ vᵢ³)^(1/3)   — the 3-norm
auto ir3 = tax::norm<2, -3>(r);   // 1/|r|³          — the gravity kernel
```

The single `norm<P, Q>` is the **fused** spelling: it raises the accumulated
power-sum **once** to `Q/P` (via `pow<Q, P>`), rather than taking the root and
re-raising — one recurrence pass instead of two. On the `1/|r|³` gravity
kernel that is **1.64× faster** than `pow(norm(r), -3)`, and for `P == 2` it
binds straight to `invSqrtPow`, so `norm<2,-3>` is bit-identical to
`invSqrtPow<3>(x² + y² + z²)`.

!!! note "What `norm<P>` assumes"
    It computes $(\sum_i v_i^{P})^{1/P}$, which is the true $P$-norm when the
    summands are non-negative — even `P`, or a vector with positive components
    at the expansion point. It requires $\sum_i v_i^{P} > 0$ at the expansion
    point (`abs` is not smooth, so the odd-`P` signed power-sum is used as-is).

---

## Named and mixed-order overloads

The whole fused surface exists for `NamedTaylorExpansion` and
`MixedTaylorExpansion` as well. Single-operand forms (`sinCos`, `sinhCosh`,
`sqrtInvSqrt`, `halfPow<K>`, `invSqrtPow<K>`) preserve the operand's axis
set; the two-operand forms (`expSin`, `expCos`, `expSinCos`) compose in the
**union of the operands' axis sets**, exactly like `operator*` and `atan2`:

```cpp
auto t = tax::variable<"t", 8>(0.0);     // axes {t}
auto w = tax::variable<"w", 8>(3.0);     // axes {w}

auto [qs, qc] = tax::expSinCos(-0.5 * t, w * t);   // both over axes {t, w}

auto g = tax::invSqrtPow<3>(r2);         // axis set of r2, unchanged
```

Mixed-order operands additionally follow the usual max-order promotion on
shared axes — see [Mixed-Order Expansions](mixed.md).

---

## When *not* to fuse

- **Only one output consumed** → use the single-output function
  (`sin`, `sqrt`, `halfPow`). The single-output kernels deliberately do not
  write a discarded companion.
- **Large multivariate shapes with `sqrtInvSqrt`** → the win shrinks to
  parity as the Cauchy-product work dominates; it never becomes a loss when
  both outputs are used, but don't expect the univariate speedup.

---

**Next:** the exact signatures and return types are tabulated in the
[Core API Reference](../reference/core.md); the coupled recurrences behind
these kernels are derived in
[Internals / Recurrence Relations](../internals/recurrences.md).
