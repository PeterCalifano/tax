# Mixed-order (anisotropic) expansions — design

**Date:** 2026-06-21
**Status:** Approved for planning (revised for the unified-type design — Option B)
**Topic:** Per-variable-group **independent truncation orders** on a genuinely **anisotropic** (per-axis-capped "box") monomial set — no dense blow-up — delivered by **unifying the existing `TaylorExpansion` over an index scheme** (so the mixed type *is* a `TaylorExpansion` with the full math + `tax::la` surface), plus a named per-axis-order layer on top.

## Motivation

For an ODE flow map (and similar) one wants the expansion in some variables (e.g. time `t`) to a much higher order than in others (e.g. state `x`): `x@4, t@20`. Today's joint-simplex expansions carry a **single global order N**, so representing `x@4, t@20` would require `N = 24` and store every `x⁵…x²⁴` term as forced zeros. Across several axes this explodes (e.g. `x@4` over 3 vars + `t@20` → a box of ~20,000 coefficients vs ~735 actually wanted). Masking a dense box cannot avoid this — it makes it worse. The representation must be **anisotropic from the start**.

The companion prototype (`prototypes/mixed/` on `origin/claude/taylor-expansion-prototypes-hxq111`) demonstrated the structural payoff of a single flat anisotropic expansion (`MixedExpansion<T,Nt,Ns,Ms,Nj>`): ~1.3–1.6× over a nested representation at equal coefficient counts, and **4–6×** once a joint total-degree cap `Nj` drops the high mixed terms. This design productionizes and generalizes that prototype from 2 groups to N groups/axes, threading it through the existing `TaylorExpansion` type and its full kernel/`la` surface.

## Scope & decisions (approved during brainstorming; revised per Option B)

1. **Anisotropic storage from the start** — the kept set is the **box**: monomial kept iff for every group *g*, the partial degree in *g*'s variables `≤ order_g`. No masking-on-a-dense-box.
2. **Unify, don't duplicate (Option B).** Rather than a separate parallel `MixedExpansion`, the existing **`TaylorExpansion` is re-parameterized over an `IndexScheme`** — `TaylorExpansion<T, Scheme, Storage>`. The classic type is `TaylorExpansion<T, IsotropicScheme<N,M>>` (kept user-spelling `TE<N,M>`); the mixed type is `TaylorExpansion<T, MixedScheme<Groups…>>` (`MixedTE<Groups…>`). The mixed type therefore **is a `TaylorExpansion`** — it satisfies the `TaylorPolynomial`/`DensePolynomial` concepts and works with `tax::la` and `io` with no duplicated surface.
3. **Existing joint-simplex *named* layer is untouched.** `NamedTaylorExpansion<T, N, Axes…>`, `Axis<Name,Dim>`, `NE`, `tax::variable<"x",N>`, `tax::jacobian<"x">` keep their joint-simplex semantics. The new per-axis-order **named** layer (decision 4) is parallel and additive.
4. **Order on the axis** (named layer) — `OrderedAxis<Name, Dim, Order>`; the named expansion `MixedTaylorExpansion<T, Axes…>` wraps a `TaylorExpansion<T, MixedScheme<groups-from-axes>>`, no global `N`. Canonical type (axes sorted-by-name, unique) via the existing `FixedString`/`axisSign`/`Merge` machinery, so `x*p == p*x` as types.
5. **Optional joint cap** lives in the core `MixedScheme`, defaulting to `Σ orders` (full box); reachable via an explicit form for the perf win.
6. **Promotion = union of axes + max order per shared axis** (named layer; the direct generalization of today's automatic axis-set union).

## Architecture

### The index-scheme abstraction (the central move) — *done in M1/M2*

Every recurrence kernel was templated `<T, N, M>`, operated on `Coeffs<T,N,M> = std::array<T, numMonomials(N,M)>`, and called `forEachRecurrenceRow<N,M>` / a `CauchyStencil<N,M>`. All shared structure is "walk output monomials in graded (ascending total-degree) order; `out[ai] = f(rows, inputs, d, db)`." An **index scheme** supplies exactly what the kernels need: `nCoeff`; `flatOf`/`multiOf`; `order`/`vars`/`isUnivariate`; `forEachRecurrenceRow(fn)`; `cauchyProduct<T>`/`cauchySelfProduct<T>`.

- **`IsotropicScheme<N, M>`** (M1) — the classic single-order graded-lex layout; delegates to today's `numMonomials`/`forEachRecurrenceRow`/`CauchyStencil`, bit-identical.
- **`MixedScheme<Groups…>`** (M2) — each `Group` is `(dim, order)` (+ optional joint cap, default `Σ order_g` = full box). Kept set = box, graded by total degree (causal recurrences), with `flatOf`/`multiOf` and box-filtered stencils.

The scheme-generic kernels (M1/M2) already run on both schemes: all 20 recurrence kernels (`exp/log/sin/cos/tan/sqrt/cbrt/pow/erf/asin/acos/atan/atan2/sinh/cosh/tanh/asinh/acosh/atanh/reciprocal/divide`) plus the Cauchy product.

### Unify `TaylorExpansion` over the scheme (Option B) — *the core of M3*

Re-parameterize the dense expansion type by its scheme:

```cpp
template < typename T, typename Scheme, typename Storage = storage::Dense >
    requires IndexScheme< Scheme >
class TaylorExpansion;            // dense specialization is generic over Scheme
```

- **Aliases (user-facing spellings preserved):** `template<int N,int M=1> using TE = TaylorExpansion<double, IsotropicScheme<N,M>>;`  `template<typename... Groups> using MixedTE = TaylorExpansion<double, MixedScheme<Groups...>>;`  `template<int N,int M=1> using STE = TaylorExpansion<double, IsotropicScheme<N,M>, storage::Sparse>;`
- **One dense surface, written once over `Scheme`:** `value`/`coeff` (compile-time/runtime/`MultiIndex`)/`operator[]`/`variable`/`derivative`/`eval`/`deriv`/`integ` use `Scheme::flatOf/multiOf/nCoeff/order/vars/isUnivariate`. The bespoke index ops become scheme-generic:
  - **`eval`** — power-table accumulation over `Scheme::multiOf`.
  - **`deriv`/`integ`** — scatter through `flatOf`/`multiOf`; `integ`'s `totalDegree ≥ N` guard becomes "is the incremented monomial still in the box?" (`flatOf != kNotInBox`).
- **Operators + math + `la` migrate from `<T,N,M>` to `<T,Scheme>`** (`operators/arithmetic.hpp`, `math_unary.hpp`, `math_binary.hpp`; `la/num_traits.hpp`, `la/values.hpp`, `la/derivatives.hpp`). Because they call the already-scheme-generic kernels and use `derivative(MultiIndex<vars>)`, the **mixed type gets the full math surface, `NumTraits`, `gradient`/`jacobian`/`hessian`, `value`/`eval`, and `io` for free** — no duplicate `MixedExpansion`, `mixed_ops`, or `la::mixed`.
- **Isotropic behavior is bit-identical** — `TE<N,M>` resolves to `TaylorExpansion<double, IsotropicScheme<N,M>>` and every result/`constexpr`-ness is unchanged; the full existing suite is the gate. The `M==1` fast paths survive via `Scheme::isUnivariate`.
- **Sparse stays isotropic-only:** the Sparse specialization pairs only with `IsotropicScheme` (`static_assert`); mixed is dense-only.

### The named per-axis-order layer — *M4*

A parallel, additive named family (the existing joint-simplex `NamedTaylorExpansion` is untouched):

- **`OrderedAxis<Name, Dim, Order>`** — reuses `FixedString`/`axisSign`/`Merge`; canonical sorted-unique axis lists.
- **`MixedTaylorExpansion<T, Axes…>`** — wraps `Inner = TaylorExpansion<T, MixedScheme<groups-from-axes>>`; `vars_v = Σ dim`.
- **Factories** in `tax::mixed` (avoid colliding with `tax::variable<"x",N>`): `tax::mixed::variable<"x",4>(x0)`, `tax::mixed::variables<"p",20>(arr)`.
- **Promotion.** Binary ops embed both operands into the **union shape** (union of axes, **max order per shared axis**) via `embedMixed<Source,Target>` (source a sub-box of target), then the unified `TaylorExpansion`/kernel surface runs and truncates to the union box.
- **Operation surface:** `value`/`coeff`/`derivative`/`eval`, full arithmetic + math (delegated to the wrapped `TaylorExpansion`), `slice<Names…>`, `deriv<"name",Local>`/`integ<"name",Local>`, `truncate<"name",N2>()` (per-axis order-lowering via `embedMixed` onto a smaller box).
- **Joint cap placement.** In the core `MixedScheme` (default `Σ orders`); the named type defaults to the box; the capped form via an explicit alias. Cross-operand promotion is defined for box/equal-cap operands; mixing different explicit caps is a `static_assert` error (documented).
- **Named `la` helpers** are thin name-addressed wrappers (`gradient<"name">`/`hessian<"name">`/`jacobian<"name">`) over the now-free generic `tax::la` — analogous to `la/named.hpp`.

## File structure

```
include/tax/core/index_scheme.hpp     # [M1] IndexScheme concept; IsotropicScheme<N,M>
include/tax/core/mixed_scheme.hpp     # [M2] Group, MixedScheme<Groups...> (box index core)
include/tax/kernels/mixed_stencils.hpp# [M2] box-filtered Cauchy + recurrence stencils
include/tax/core/taylor_expansion.hpp # [M3] re-parameterized over Scheme; TE/MixedTE/STE aliases
include/tax/operators/*.hpp           # [M3] arithmetic/math_unary/math_binary -> <T,Scheme>
include/tax/la/*.hpp                   # [M3] num_traits/values/derivatives -> <T,Scheme>
include/tax/core/named.hpp            # [M3] existing joint-simplex named: rewire to TaylorExpansion<T,IsotropicScheme<N,vars>>
include/tax/io/series.hpp             # [M3] -> <T,Scheme>
include/tax/core/mixed_named.hpp      # [M4] OrderedAxis, MixedTaylorExpansion, factories, promotion/embed/slice/deriv/integ/truncate
include/tax/la/mixed_named.hpp        # [M4] name-addressed gradient/hessian/jacobian for the named mixed type
docs/guide/mixed.md  (+ mkdocs nav)   # [M5]
tests/mixed/…, tests/core/…           # per milestone
```

No separate `mixed_expansion.hpp` / `mixed_ops.hpp` / `la/mixed.hpp` — the unified `TaylorExpansion` subsumes them.

## Implementation milestones

1. **[done] `IndexScheme` + `IsotropicScheme`** — kernels scheme-generic; isotropic bit-identical.
2. **[done] `MixedScheme` + mixed stencils** — box index core (`flatOf`/`multiOf`, graded), box-filtered Cauchy + recurrence stencils; finished scheme-generic Cauchy-based kernels (dropped `Scheme::vars`).
3. **Unify `TaylorExpansion` over `IndexScheme` (Option B).** Re-parameterize the dense type; migrate accessors/`eval`/`deriv`/`integ`, operators, `la`, `named` (existing), and `io` from `<T,N,M>` to `<T,Scheme>`; keep `TE<N,M>` and add `MixedTE<Groups…>`. **Isotropic behavior bit-identical (existing suite is the gate).** Result: `MixedTE` works through the full math + `la` surface, validated by the isotropic-superset oracle.
4. **Named per-axis-order layer.** `OrderedAxis`, `MixedTaylorExpansion` wrapping `TaylorExpansion<T, MixedScheme<…>>`, `tax::mixed` factories, max-per-axis promotion/`embedMixed`/`slice`/`deriv`/`integ`/`truncate<"name",N2>`; thin named-`la` helpers.
5. **Docs.** `docs/guide/mixed.md` + nav; any remaining polish.

## Testing strategy

- **Isotropic regression gate (M3):** the entire existing suite stays green and bit-identical after the unification — the proof the re-parameterization is behavior-preserving. (M1/M2 already pass it.)
- **Primary correctness oracle (M3):** every box coefficient `α` of a `MixedTE` result equals the same coefficient of an isotropic `TaylorExpansion<T, Σorders, vars>` evaluating the same expression. Rationale: the box ⊆ the order-`Σorders` simplex and monomial degrees add, so no out-of-box factor can reach an in-box output. Validates the entire surface (arithmetic + every transcendental + `eval`/`deriv`/`integ` + `la`) by reusing the trusted isotropic library. (M2 already proved this at the raw-array level; M3 re-checks it through the `TaylorExpansion`/`la` surface.)
- **`la` on the mixed type:** `gradient`/`jacobian`/`hessian`/`NumTraits` on a `MixedTE` vs analytic values and vs the isotropic superset; Eigen-matrix interop.
- **Named layer (M4):** promotion (max-per-axis, axis-set union), `slice`, `deriv`/`integ`, `truncate<"name",N2>`, canonical-type equality (`x*p == p*x`), per-axis correctness.

## Invariants & edge cases

- **No per-operation heap** in the dense core: storage is `std::array`; scheme stencil tables are built-once runtime statics (fixed `std::array`), not per-op allocations.
- **`static_assert` guards** on `keptCount`/table blow-up.
- **Graded ordering is mandatory** for the mixed layout (causal recurrences) — analogous to graded-lex's sacredness.
- **`constexpr`:** the unified dense type keeps the constant-evaluation path (scheme stencils have an `if !consteval` fallback) wherever the isotropic type had it.
- Degenerate cases: a single group/axis (≈ univariate, via the general scheme), `order = 0` groups, `dim ≥ 1` (`dim = 0` invalid). `MixedScheme::multiOf` has a documented `k ∈ [0,nCoeff)` precondition (add a debug `assert` for parity with `IsotropicScheme`).

## Out of scope (YAGNI / follow-ups)

- Changing or deprecating the existing joint-simplex `NamedTaylorExpansion` — it stays as-is (rewired only to spell its backing type via `IsotropicScheme`, no semantic change).
- Sparse-storage mixed expansions (sparse is isotropic-only).
- Cross-operand promotion between **different** explicit joint caps (box/equal-cap only; mismatch is a compile error).
- Batched (`tax::Batch`) coefficients on the mixed type — the scheme-generic surface should not preclude it, but it is not a deliverable here.
- The `tax-flow` ODE/ADS consumers of mixed-order flow maps (separate repo).
