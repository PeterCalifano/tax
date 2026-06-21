# Mixed-order (anisotropic) named expansions — design

**Date:** 2026-06-21
**Status:** Approved for planning
**Topic:** A new, separate named-expansion family whose axes carry **independent truncation orders**, stored on a genuinely **anisotropic** (per-axis-capped "box") monomial set — no dense blow-up — exposed through the full production math surface and `tax::la`.

## Motivation

For an ODE flow map (and similar) one wants the expansion in some variables (e.g. time `t`) to a much higher order than in others (e.g. state `x`): `x@4, t@20`. Today's `NamedTaylorExpansion<T, N, Axes...>` carries a **single global order N** (a joint total-degree simplex over all variables), so representing `x@4, t@20` would require `N = 24` and store every `x⁵…x²⁴` term as forced zeros. Across several axes this explodes (e.g. `x@4` over 3 vars + `t@20` → a box of ~20,000 coefficients vs ~735 actually wanted). Masking a dense box cannot avoid this — it makes it worse. The representation must be **anisotropic from the start**.

The companion prototype (`prototypes/mixed/` on `origin/claude/taylor-expansion-prototypes-hxq111`) demonstrated the structural payoff of a single flat anisotropic expansion (`MixedExpansion<T,Nt,Ns,Ms,Nj>`): at equal coefficient counts ~1.3–1.6× over a nested representation, and **4–6×** once a joint total-degree cap `Nj` drops the high mixed terms a nested/box form is forced to carry. This design productionizes and generalizes that prototype from 2 groups to N named axes, and wires it through the named API and the full kernel surface.

## Scope & decisions (all approved during brainstorming)

1. **Anisotropic storage from the start** — the kept set is the **box**: monomial kept iff for every axis (group) *g*, the partial degree in *g*'s variables `≤ order_g`. No masking-on-a-dense-box.
2. **A separate type** — today's joint-simplex `NamedTaylorExpansion<T, N, Axes...>`, `Axis<Name,Dim>`, `NE`, `tax::variable<"x",N>`, `tax::jacobian<"x">` are **left entirely unchanged**. The mixed-order family is parallel and additive. No existing semantics change.
3. **Order on the axis** — `OrderedAxis<Name, Dim, Order>`; the expansion is `MixedTaylorExpansion<T, Axes...>` with no global `N`. Canonical type (axes sorted-by-name, unique) via the existing `FixedString`/`axisSign`/`Merge` machinery, so `x*p == p*x` as types.
4. **Full production math surface in one spec** — arithmetic + every transcendental/algebraic op + `tax::la`, not a minimal subset.
5. **Optional joint cap** lives in the core scheme, defaulting to `Σ orders` (full box). Reachable via an explicit form for the perf win; the named type defaults to the box.
6. **Promotion = union of axes + max order per shared axis** (the direct generalization of today's automatic axis-set union).

## Architecture

### The index-scheme abstraction (the central move)

Every recurrence kernel today is templated `<T, N, M>`, operates on `Coeffs<T,N,M> = std::array<T, numMonomials(N,M)>`, and calls `forEachRecurrenceRow<N,M>` / a `CauchyStencil<N,M>`. All shared structure is "walk output monomials in graded (ascending total-degree) order; `out[ai] = f(rows, inputs, d, db)`." We introduce an **index scheme** that supplies exactly what the kernels need:

- `static constexpr std::size_t nCoeff` — storage size.
- `forEachRecurrenceRow(fn)` → `fn(ai, d, std::span<const RecurrenceEntry>)`, where each `RecurrenceEntry{b_idx, g_idx, db}` means `β+γ = α(ai)`, `|β| = db ≥ 1`, `b_idx = flat(β)`, `g_idx = flat(γ)`. **Same callback signature as today.**
- The Cauchy product stencil — the `(out_idx, a_idx, b_idx)` triple list.
- `flat ↔ multi-index` maps (the `pos` map) for the bespoke ops.

Two schemes implement this concept:

- **`IsotropicScheme<N, M>`** — wraps exactly today's `numMonomials` / `forEachRecurrenceRow<N,M>` / `CauchyStencil<N,M>`. The existing `TaylorExpansion<T,N,M,Dense>` hot path is rewired onto it and must produce **bit-identical** tables and codegen, including the `M==1` unrolled Cauchy product and the `M≥2` stencil specialization. Zero behavioral or performance change is a hard requirement, gated by the full existing test suite.
- **`MixedScheme<Groups...>`** — each `Group` is `(dim, order)`; an optional joint total-degree cap `Nj` (default `Σ order_g` = full box). Kept set = box (∩ optional joint cap). Ordering **graded by total degree** with per-group sub-structure within each degree (the prototype's layout), which keeps the forward-substitution recurrences causal. `pos` map `(per-group multi-index) → flat`, built once at first use (runtime-static) with a `constexpr`-evaluation fallback that enumerates on the fly (mirrors the existing `if !consteval` pattern, so the type stays usable in constant expressions).

**Pay-off:** once a kernel is scheme-generic, all 20 recurrence kernels work on `MixedScheme` with **no per-kernel math change** — `seriesExp/Log/Sin/Cos/Tan/Sqrt/Cbrt/Pow/Erf/Asin/Acos/Atan/Atan2/Sinh/Cosh/Tanh/Asinh/Acosh/Atanh/Reciprocal/Divide`. Arithmetic `+`/`-` is elementwise on the shared layout; `*` uses the box-filtered Cauchy stencil.

### Bespoke (non-stencil) ops, made scheme-aware

These thread `flatIndex`/`unflatIndex`/degree arithmetic directly and get scheme-aware reimplementations:

- **`eval`** — degree-graded Horner / power-table accumulation over the scheme's monomials.
- **`deriv` / `integ`** — scatter through the `pos` map; `integ`'s `totalDegree ≥ N` guard becomes "is the incremented monomial still in the kept box?".
- **`truncate`** — the graded-prefix-copy assumption no longer holds; per-axis order-lowering via the `pos` map (`truncate<"name", N2>()`).
- **named `embed` / `slice`** — the `unflatIndex → axis-remap → flatIndex` round-trip is replaced by a map through the source and target `pos` tables (`embedMixed<Source, Target>`).

Sparse kernels are **out of scope** — the mixed family is dense-only (consistent with named, which is dense-only).

### The named layer

- **`OrderedAxis<Name, Dim, Order>`** — reuses `FixedString`/`axisSign`/`Merge`; canonical sorted-unique axis lists.
- **`MixedTaylorExpansion<T, Axes...>`** — wraps `Inner = ` the core anisotropic expansion over `MixedScheme<groups-from-axes>`; `vars_v = Σ dim`.
- **Factories** in namespace `tax::mixed` (to avoid colliding with `tax::variable<"x",N>`):
  ```cpp
  auto x = tax::mixed::variable<"x", 4>(1.0);      // axis "x" at order 4
  auto p = tax::mixed::variables<"p", 20>(arr3);   // 3-D axis "p" at order 20
  auto f = sin(x) + x * p[0];                      // "x"@4, "p"@20 — no x^5… stored
  ```
- **Promotion.** Binary ops embed both operands into the **union shape** (union of axes, **max order per shared axis**) via `embedMixed<Source, Target>` (source must be a sub-box of target: axis set ⊆ and every shared order ≤), then the scheme kernel runs and truncates to the union box.
- **Operation surface** (all scheme-aware): `value`, `coeff`/`derivative` (compile-time, runtime, `MultiIndex` forms), `eval`, full arithmetic + math surface, `slice<Names...>` (drop axes, keep monomials with zero degree in dropped axes, preserving kept orders), `deriv<"name",Local>` / `integ<"name",Local>` (axes/orders preserved, matching named's behavior), `truncate<"name", N2>()`.
- **Joint cap placement.** The cap lives in the core `MixedScheme` (default `Σ orders`). The named type defaults to the box; the capped form is reached via an explicit form (a leading cap parameter or a `*_capped` alias — finalized in the plan). In this spec, **cross-operand promotion is defined for box operands and equal-cap operands; mixing different explicit caps is a `static_assert` error** (documented). The single-fixed-cap perf path is fully usable.

### `tax::la::mixed`

`Eigen::NumTraits<MixedTaylorExpansion<...>>` (so Eigen matrices can hold the type), plus name-addressed `gradient<"name">` / `hessian<"name">` / `jacobian<"name">`, analogous to `la/named.hpp`: build a `MultiIndex<vars_v>` and call `inner().derivative(alpha)` — works once the mixed `Inner` supports `derivative(MultiIndex)` through the `pos` map.

## File structure (all additive)

```
include/tax/core/index_scheme.hpp     # IndexScheme concept; IsotropicScheme<N,M>; MixedScheme<Groups...>
include/tax/kernels/mixed_stencils.hpp# box-filtered Cauchy stencil + forEachRecurrenceRow for MixedScheme
include/tax/core/mixed_expansion.hpp  # core names-free anisotropic dense expansion over a scheme
include/tax/core/mixed_named.hpp      # OrderedAxis, MixedTaylorExpansion, factories, embed/slice/compose/deriv/integ/truncate, promotion
include/tax/la/mixed.hpp              # NumTraits + name-addressed gradient/hessian/jacobian
include/tax/tax.hpp                   # umbrella: add the new includes
tests/mixed/…                         # scheme tables, mixed cauchy, core math surface, named, la
docs/guide/mixed.md  (+ mkdocs nav)
```

The existing kernels (`algebra.hpp`, `transcendental.hpp`, `trigonometric.hpp`, `cauchy*.hpp`, `recurrence_stencil.hpp`) are refactored to be scheme-generic; `taylor_expansion.hpp` is rewired onto `IsotropicScheme` with no semantic change.

## Implementation milestones (the plan sequences these as tasks)

1. **`IndexScheme` + `IsotropicScheme` refactor.** Introduce the concept; rewire existing kernels and `TaylorExpansion` onto `IsotropicScheme<N,M>`. **Full existing suite green, bit-for-bit** — the one hot-path touch and the safety gate. No new user-facing behavior.
2. **`MixedScheme` + mixed stencils.** Shape machinery (groups, optional joint cap), graded ordering, `pos` map, `constexpr` fallback; box-filtered Cauchy stencil + recurrence rows. Unit-test the tables directly (counts, ordering graded, round-trip `flat↔multi`, stencil correctness vs brute force).
3. **Core `MixedExpansion`.** Storage (`std::array<T, keptCount>`), factories, `value`/`coeff`/`eval`/`deriv`/`integ`/`truncate`, arithmetic + full math surface via the scheme-generic kernels. Validate against the isotropic oracle (below).
4. **Named layer.** `OrderedAxis`, `MixedTaylorExpansion`, `tax::mixed` factories, promotion/`embedMixed`/`slice`/`deriv`/`integ`/`truncate`.
5. **`la::mixed` + docs.** `NumTraits`, name-addressed gradient/hessian/jacobian; `docs/guide/mixed.md`.

## Testing strategy

- **Isotropic regression gate (M1):** the entire existing test suite must stay green and unchanged after the `IsotropicScheme` refactor — the proof that the hot-path rewire is behavior-preserving.
- **Primary correctness oracle for the full math surface (M3):** every box monomial `α` of a mixed result equals the same coefficient of an **isotropic `TaylorExpansion<T, Σorders, vars>`** evaluating the same expression. Rationale: the box is a subset of the order-`Σorders` simplex, and monomial degrees add under multiplication, so no out-of-box intermediate factor (e.g. `x⁵` when `x@4`) can contribute to an in-box output. Therefore `mixed.coeff(α) == isotropic.coeff(α)` for all kept `α`. This validates the entire surface (arithmetic + every transcendental) by reusing the existing, trusted isotropic library — no new reference implementation.
- **Joint-cap form:** checked against a nested/prototype baseline (the cap drops high mixed terms the box keeps).
- **Named layer:** promotion (max-per-axis, axis-set union), `slice`, `deriv`/`integ`, `truncate<"name",N2>`, canonical-type equality (`x*p` and `p*x` same type), and per-axis correctness.
- **`la`:** `gradient`/`hessian`/`jacobian` by axis name vs analytic values; Eigen-matrix interop.

## Invariants & edge cases

- **No per-operation heap** in the dense core: mixed storage is `std::array`. Scheme tables are compile-time-sized where the count is tractable; otherwise a **one-time runtime-static cache** built at first use (a shared stencil cache, not per-expansion heap) — the exact representation (fixed `std::array` vs built-once flat buffer) is finalized in M2.
- **`static_assert` guards** on `keptCount` blow-up (a too-large shape fails to compile with a clear message).
- **Graded ordering is mandatory** for the mixed layout (causal recurrences) — analogous to the sacredness of graded-lex isotropically.
- Degenerate cases: a single axis (≈ univariate, routed through the general scheme rather than the `M==1` unroll), `order = 0` axes, `dim ≥ 1` per axis (`dim = 0` invalid).
- `constexpr`: the mixed type keeps a constant-evaluation path (like the isotropic stencil fallback), so it is usable in constant expressions where the scalar `T` allows.

## Out of scope (YAGNI / follow-ups)

- Changing or deprecating the existing joint-simplex `NamedTaylorExpansion` — it stays as-is.
- Sparse-storage mixed expansions.
- Cross-operand promotion between **different** explicit joint caps (this spec: box/equal-cap only; mismatch is a compile error).
- Batched (`tax::Batch`) coefficients on the mixed type — the scheme-generic kernels should not preclude it, but it is not a deliverable here.
- The `tax-flow` ODE/ADS consumers that would use mixed-order flow maps (separate repo).
