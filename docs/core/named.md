# Named Expansions

`tax::named::NamedTaylorExpansion<T, N, Axes...>` wraps a dense
`TaylorExpansion<T, N, M>` and attaches a compile-time list of **named axes** to
it. Each axis is a contiguous block of the underlying $M$ variables identified
by a compile-time string. The whole public API is re-exported directly under
`tax`, so `tax::NE`, `tax::variable`, `tax::variables`, `tax::Axis`, and
`tax::NamedTaylorExpansion` are all available from `<tax/tax.hpp>`.

The named layer lets you build, combine, and project Taylor expansions by
**variable name** instead of by raw flat index — the dependency set of a result
is tracked in its type and derived automatically from its operands.

```cpp
#include <tax/tax.hpp>

// One variable per named axis, order 4, expanded about the given points.
auto x = tax::variable<"x", 4>(1.0);
auto p = tax::variable<"p", 4>(0.5);

// x and p carry *different* axis sets; composing them runs in the union.
auto f = sin(x) + x * p;     // type: NamedTaylorExpansion over axes {p, x}
```

---

## Axes and the canonical type

An axis is `tax::Axis<Name, Dim>` — a name (a `tax::FixedString` NTTP) and a
block of `Dim ≥ 1` consecutive variables:

```cpp
using PosX = tax::Axis<"x", 3>;   // a 3-D axis called "x"
using Time = tax::Axis<"t", 1>;   // a scalar axis called "t"
```

The axis list of every `NamedTaylorExpansion` is kept in **canonical order**
(sorted by name, unique), so the type does not depend on the order you wrote the
operands: `x * p` and `p * x` produce the *same* type. The convenience alias

```cpp
template <int N, typename... Axes>
using NE = tax::named::NamedTaylorExpansion<double, N, Axes...>;
```

spells the common `double`-valued case, e.g. `tax::NE<4, Axis<"x", 3>>`.

---

## Creating named variables

```cpp
// Scalar (1-D) axis — returns a single expansion:
auto t = tax::variable<"t", 6>(0.0);                 // NE<6, Axis<"t",1>>

// Multi-dimensional axis — returns std::array of the D coordinate variables:
std::array<double, 3> x0{1.0, 2.0, 3.0};
auto x = tax::variables<"x", 6>(x0);                 // std::array<NE<6,Axis<"x",3>>, 3>
auto& x1 = x[1];

// Eigen expansion point (from <tax/tax.hpp>, tax::named::variables overload):
Eigen::Vector3d v0{1.0, 2.0, 3.0};
auto xv = tax::named::variables<"x", 6>(v0);         // Eigen vector of named variables
```

---

## Composition across axis sets

Arithmetic between expansions over **different** axis sets runs in the union of
the two sets: both operands are first **embedded** into the union, then the
existing dense kernels do the work. The result type carries the union of axes:

```cpp
auto x = tax::variable<"x", 4>(1.0);   // axes {x}
auto y = tax::variable<"y", 4>(2.0);   // axes {y}

auto g = x * x + x * y + y * y;        // axes {x, y}, computed exactly
```

A value that depends on **fewer** axes promotes implicitly into a wider axis set
(value-preserving: the absent axes simply get zero derivatives), so you can pass
a narrow expansion where a wider one is expected.

All the usual math functions (`sin`, `exp`, `sqrt`, `pow`, `atan2`, …) work on
named expansions and preserve the axis set.

---

## Slicing and named derivatives

`slice<Names...>()` projects an expansion back onto a subset of its axes by
keeping the monomials that do not depend on the dropped axes (i.e. restricting
the dropped axes to their expansion point):

```cpp
auto h = f.slice<"x">();        // drop every axis except "x"
```

`deriv<"name">()` and `integ<"name">()` differentiate / integrate symbolically
with respect to a named axis (the axis set is preserved). Composing them with
`slice` gives the "sub-derivative" projection:

```cpp
auto dfx = f.deriv<"p">().slice<"x">();   // ∂f/∂p, then restricted to axis x
```

---

## Eigen helpers

`<tax/tax.hpp>` provides a `NumTraits` specialisation so named expansions live
inside Eigen vectors/matrices, plus name-addressed differential operators:

```cpp
auto gx = tax::named::gradient<"x">(f);   // gradient w.r.t. the coords of axis "x"
auto Hx = tax::named::hessian<"x">(f);    // Hessian  w.r.t. axis "x"
auto Jx = tax::named::jacobian<"x">(F);   // Jacobian of an Eigen vector F w.r.t. "x"
```

---

## Key headers

| Header | Contents |
|---|---|
| `tax/core/named.hpp` | `NamedTaylorExpansion`, `Axis`, `FixedString`, `NE`, `variable`/`variables`, embed/slice/compose |
| `tax/la/named.hpp`   | `NumTraits` for named expansions + `gradient`/`hessian`/`jacobian` by axis name |
