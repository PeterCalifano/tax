# Mixed-order Milestone 4 — named per-axis-order layer (`MixedTaylorExpansion`)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. **After committing + reporting DONE, STOP — do not self-review or amend; the controller runs review.**

**Goal:** A named, sliceable, composable expansion whose axes carry **independent orders** — `MixedTaylorExpansion<T, OrderedAxis<Name,Dim,Order>…>` wrapping a `TaylorExpansion<T, MixedScheme<groups-from-axes>>` — with axis-name `variable`/`slice`/`deriv`/`integ`/`truncate`, automatic **union + max-order-per-axis promotion**, and name-addressed `la`. Parallel and additive to the existing joint-simplex `NamedTaylorExpansion` (untouched).

**Architecture:** Mirror `core/named.hpp` (`NamedTaylorExpansion`), substituting `OrderedAxis<Name,Dim,Order>` for `Axis<Name,Dim>` and `MixedScheme<groups>` for `IsotropicScheme<N,vars>` as the backing. Reuse the existing generic axis helpers (`FixedString`, `axisSign`, `OffsetOf`, `DimOfName`, `TotalDim`, `buildAxisMap`, `IsCanonical`, `IsSubsetOf`) — they are duck-typed on `::name`/`::dim`. The genuinely new machinery is an **order-aware merge** (same-name axis → max order), the **axis→group** mapping, and **`embedMixed`** (box→box reindex via `MixedScheme::flatOf`/`multiOf`).

**Tech Stack:** C++23, header-only, Eigen3, Google Test, CMake. Build in the `tax` mamba/conda env.

**Milestone 4 of 5** from `docs/superpowers/specs/2026-06-21-mixed-order-named-expansions-design.md`. M1–M3 are complete (the unified `TaylorExpansion<T,Scheme>`; `MixedTE<Groups…>` is a first-class type with full math + `la`). M5 is docs.

## Global Constraints

- C++23, header-only — code in `include/tax/`. No per-operation heap (dense `std::array` backing).
- **Existing layers untouched:** `NamedTaylorExpansion<T,N,Axes…>`, `Axis<Name,Dim>`, `tax::variable<"x",N>`, the unified `TaylorExpansion`/`TE`/`MixedTE` — no behavior change. The full existing suite stays green.
- **Canonical type:** `OrderedAxis` lists are sorted-by-name and unique (`IsCanonical`), so `x*p == p*x` as types. Build via factories/composition, never by hand-spelling axes.
- **Promotion:** binary ops over different axis sets / orders embed both operands into the **union axes with max order per shared axis**, then run the unified `TaylorExpansion` op. A shared axis must have the **same dim** (`static_assert`).
- **Box semantics inherited from M2/M3:** the wrapped `MixedScheme` is the full box (per-axis caps, no joint cap in the named layer); `MixedTE`'s oracle (vs isotropic `TE<Σorder,vars>`) holds.
- Factories live in namespace `tax::mixed` (avoid colliding with `tax::variable<"x",N>`).
- clang-format **v21**: format only new/edited regions; revert gratuitous reflow of pre-existing lines; never reorder `tax.hpp` includes.
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Branch `feature/mixed-order-expansions`.

## Interfaces from M1–M3 (on the branch)

- `tax::Group<int Dim,int Order>`; `tax::MixedScheme<Groups…>` (`nCoeff/vars/order/kNotInBox/flatOf/multiOf`); unified `tax::TaylorExpansion<T, Scheme, Storage>`; `tax::MixedTE<Groups…> = TaylorExpansion<double, MixedScheme<Groups…>>`.
- Existing named machinery in `include/tax/core/named.hpp`: `tax::named::FixedString`, `tax::named::axisSign`, `tax::named::fixedCompare`, `tax::named::Axis` (the joint-simplex one); and in `tax::named::detail`: `TypeList`, `Prepend`, `OffsetOf`, `DimOfName`, `TotalDim`, `IsCanonical`, `IsSubsetOf`, `buildAxisMap<Src,Tgt,allowDrop>()`, `Rebind`, `Merge`/`MergeChoose`. **Reuse the generic ones** (`OffsetOf`/`DimOfName`/`TotalDim`/`IsCanonical`/`IsSubsetOf`/`buildAxisMap` read `::name`/`::dim`, so they work for `OrderedAxis`). The new layer's types live in `tax::named` (reusing `FixedString` etc.) and are **re-exported to `tax`** (mirror the `namespace tax { using named::… }` block at `core/named.hpp:667-675`). Name-addressed `la` in `include/tax/la/named.hpp` (mirror it).
- `tax::la` (`gradient`/`hessian`/`jacobian`/`value`/`NumTraits`) is scheme-generic → works on the wrapped `MixedTE` directly.

## File Structure

- `include/tax/core/mixed_named.hpp` — **new**: `OrderedAxis`, axis→group mapping, order-aware merge, `MixedTaylorExpansion<T, Axes…>`, factories (`tax::mixed::variable`/`variables`), `embedMixed`, binary ops, `slice`/`deriv`/`integ`/`truncate`.
- `include/tax/la/mixed_named.hpp` — **new**: name-addressed `gradient<"name">`/`hessian<"name">`/`jacobian<"name">` for `MixedTaylorExpansion`.
- `include/tax/tax.hpp` — **modify**: add the two includes (after `core/named.hpp` / `la.hpp`; do NOT reorder existing lines).
- `tests/mixed/test_mixed_named.cpp`, `tests/mixed/test_mixed_named_la.cpp` — **new**.

## Build & test reference (repo root `/Users/andrea/Documents/Codes/tax`)
Configure (if stale): `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`. Full suite: `cmake --build build -j && ctest --test-dir build --output-on-failure`.

---

### Task 1: `OrderedAxis`, axis→group mapping, the `MixedTaylorExpansion` type + factories

**Files:** Create `include/tax/core/mixed_named.hpp`; Create `tests/mixed/test_mixed_named.cpp`; Modify `tests/CMakeLists.txt`, `include/tax/tax.hpp`.

**Interfaces produced:**
- `template<tax::named::FixedString Name, int Dim, int Order> struct OrderedAxis { static constexpr auto name = Name; static constexpr int dim = Dim; static constexpr int order = Order; static_assert(Dim>=1 && Order>=0); };` — defined in `tax::named`, re-exported to `tax` (so `tax::OrderedAxis`).
- axis→group: a `detail` metafunction turning `OrderedAxis<Name,Dim,Order>…` (already canonical-sorted) into `MixedScheme<Group<Dim,Order>…>` (in axis order = the scheme's group order).
- `template<typename T, typename... Axes> class tax::MixedTaylorExpansion` with: `using axis_list = named::detail::TypeList<Axes...>;` `scalar_type=T`; `IsCanonical` static_assert; `static constexpr int vars_v = TotalDim<axis_list>;` `using Inner = TaylorExpansion<T, MixedScheme<groups-from-axes>, storage::Dense>;` `using Input = typename Inner::Input;` ctors (default/`T`/`explicit(Inner)`); `value()`, `inner()`, `operator[]`, `coeff(MultiIndex<vars_v>)`, `coeff<int...Alpha>()`, `derivative(MultiIndex<vars_v>)`.
- Factories in `tax::mixed`: `template<FixedString Name,int Order> auto variable(double x0)` → 1-D axis `OrderedAxis<Name,1,Order>` expansion; `template<FixedString Name,int Order,std::size_t D> auto variables(const std::array<double,D>&)` → `std::array` of D `MixedTaylorExpansion`s over axis `OrderedAxis<Name,D,Order>` (one per coordinate). (Mirror `tax::variable<"x",N>`/`variables` in `core/named.hpp`, but the template arg is the **order**, and the backing is mixed.)

- [ ] **Step 1: Write failing tests**

`tests/mixed/test_mixed_named.cpp`:
```cpp
#include <gtest/gtest.h>
#include <tax/tax.hpp>

TEST( MixedNamed, ConstructAndType )
{
    auto x = tax::mixed::variable<"x", 4>( 1.0 );   // axis "x" dim 1 order 4
    using X = decltype( x );
    static_assert( X::vars_v == 1 );
    static_assert( X::Inner::nCoefficients == 5 );   // numMonomials(4,1)
    EXPECT_DOUBLE_EQ( x.value(), 1.0 );
}

TEST( MixedNamed, VariablesArrayAndAxisDim )
{
    std::array< double, 3 > p{ 0.1, 0.2, 0.3 };
    auto v = tax::mixed::variables<"p", 6, 3>( p );  // 3-D axis "p" order 6
    static_assert( decltype( v[0] )::vars_v == 3 );
    EXPECT_DOUBLE_EQ( v[0].value(), 0.1 );
    EXPECT_DOUBLE_EQ( v[2].value(), 0.3 );
}
```

- [ ] **Step 2: Register + confirm fail.** Add `tax_add_test(test_mixed_named SOURCES mixed/test_mixed_named.cpp)`; add `#include <tax/core/mixed_named.hpp>` to `tax.hpp` after `core/named.hpp`. Build target → FAIL.

- [ ] **Step 3: Implement `mixed_named.hpp` (this task: type + factories + accessors)**

Mirror the `NamedTaylorExpansion` skeleton (`core/named.hpp:301-468`), substituting `OrderedAxis` and the `MixedScheme` backing. The axis→group metafunction maps the (already canonical) axis pack to `MixedScheme<Group<Axes::dim, Axes::order>...>`. Factories mirror `core/named.hpp`'s `variable`/`variables` but take `Order` as the template arg and build the 1-axis/`D`-axis `MixedTaylorExpansion` via `Inner::variable<I>(...)`. Reuse `named::detail::{FixedString, TypeList, IsCanonical, TotalDim, OffsetOf, DimOfName}`.

- [ ] **Step 4: Build + run green; Step 5: clang-format + commit** (`feat(core): OrderedAxis + MixedTaylorExpansion type & factories`).

---

### Task 2: Order-aware merge, promotion, and `embedMixed` (binary ops)

**Files:** Modify `include/tax/core/mixed_named.hpp`; Modify `tests/mixed/test_mixed_named.cpp`.

**Interfaces produced:**
- `detail::MergeOrdered<ListA,ListB>` — like `named::detail::Merge` but the same-name case requires equal `dim` and takes **`max(order)`** (produce `OrderedAxis<Name, dim, max(orderA,orderB)>`); `MergedMixedTaylorExpansion<T,ListA,ListB>` rebinds the union into a `MixedTaylorExpansion`.
- `template<typename R> R MixedTaylorExpansion::embed() const` — embed into target `R` whose axes are a superset and whose per-axis orders are ≥ this expansion's (a **sub-box** of `R`). Reindex: for each source coeff `k`, `α_src = Inner::scheme::multiOf(k)`; remap variables via `buildAxisMap<axis_list, R::axis_list, false>()`; `R::Inner::scheme::flatOf(α_dst)` (always in-box since target orders ≥ source) → write.
- Free operators in `tax`: `+ - * /` (binary, same-or-different axes) and scalar forms + unary `-`, for `MixedTaylorExpansion`. Each embeds both operands into `MergedMixedTaylorExpansion<…>` then runs the **unified `TaylorExpansion` operator** on the inners (`a.inner() OP b.inner()`), wrapping the result. (Mirror the `TAX_NAMED_BINOP` macro in `core/named.hpp:495-546`.)

- [ ] **Step 1: Write failing tests (composition, promotion, oracle)**

Append:
```cpp
TEST( MixedNamed, ComposeUnionAxesNoBlowup )
{
    auto x = tax::mixed::variable<"x", 4>( 0.3 );
    auto t = tax::mixed::variable<"t", 20>( 0.1 );
    auto f = x * t + x;                 // union {t@20, x@4} (sorted by name)
    using F = decltype( f );
    static_assert( F::vars_v == 2 );
    // box size = numMonomials(4,1) * numMonomials(20,1) = 5 * 21 = 105 (NOT (24+2 choose 2))
    static_assert( F::Inner::nCoefficients == 105 );
    // x*t coefficient present; x^5 not representable (axis x capped at 4)
    EXPECT_NEAR( f.template coeff<"x", 1, "t", 1>(), 1.0, 1e-12 );   // see helper note
}

// canonical type equality: x*t and t*x are the same type
TEST( MixedNamed, CanonicalTypeOrderIndependent )
{
    auto x = tax::mixed::variable<"x", 4>( 0.3 );
    auto t = tax::mixed::variable<"t", 20>( 0.1 );
    static_assert( std::is_same_v< decltype( x * t ), decltype( t * x ) > );
    SUCCEED();
}

// max-order promotion on a shared axis
TEST( MixedNamed, SharedAxisPromotesToMaxOrder )
{
    auto x2 = tax::mixed::variable<"x", 2>( 0.3 );
    auto x5 = tax::mixed::variable<"x", 5>( 0.3 );
    auto p = x2 * x5;                   // shared axis x -> order 5
    static_assert( decltype( p )::Inner::nCoefficients == 6 );  // numMonomials(5,1)
    SUCCEED();
}
```
(For the `coeff<"x",1,"t",1>()` convenience: if a named multi-axis `coeff<Names+exponents>` is not in scope, assert via `f.inner()[f.inner().scheme::flatOf(monomial)]` instead — keep the test pinning the x·t coefficient and the box size, which are the load-bearing checks.)

- [ ] **Step 2: Build, expect fail.** Run target → FAIL (operators/embed undefined).

- [ ] **Step 3: Implement `MergeOrdered` + `embedMixed` + the binary-op macro.** Follow the interfaces above; mirror `core/named.hpp`'s `Merge`/`MergeChoose` (adding the max-order same-name case), `embed()` (using `MixedScheme::multiOf`/`flatOf` instead of `unflatIndex`/`flatIndex`), and `TAX_NAMED_BINOP`.

- [ ] **Step 4: Add an isotropic-superset oracle test**

Compose `f = sin(x*t) + exp(x)` for `MixedTaylorExpansion` (`x@4,t@20` → but use small orders for the oracle, e.g. `x@3,t@4`, Σ=7) and compare every box coefficient of `f.inner()` to the isotropic `tax::TE<7,2>` computing the same — `f.inner()[k] == iso[tax::flatIndex<2>(f.inner().scheme::multiOf(k))]` to 1e-12. (Reuses the M3 oracle idea at the named layer.)

- [ ] **Step 5: Build + run green; Step 6: clang-format + commit** (`feat(core): mixed-named promotion (union + max-order), embedMixed, binary ops`).

---

### Task 3: `slice`, `deriv`, `integ`, `truncate` (axis-name ops)

**Files:** Modify `include/tax/core/mixed_named.hpp`; Modify `tests/mixed/test_mixed_named.cpp`.

**Interfaces produced (members of `MixedTaylorExpansion`):**
- `template<FixedString Name,int Local=0> deriv() const` / `integ() const` — map `(Name,Local)` to the global variable index (`OffsetOf<axis_list, OrderedAxis<Name, DimOfName…, …>>` + Local), call `inner_.deriv<idx>()` / `inner_.integ<idx>()`; result keeps the same axes/orders.
- `template<FixedString... Names> slice() const` — project onto the named subset axes: target axis list = the `Names…` axes (with their dims+orders), reindex via `buildAxisMap<axis_list, Tgt, /*allowDrop=*/true>()`, **drop** any source monomial with nonzero degree in a dropped axis (mirror `NamedTaylorExpansion::slice`, `core/named.hpp:396-439`, but through `MixedScheme::multiOf`/`flatOf`).
- `template<FixedString Name,int N2> truncate() const` — lower axis `Name`'s order to `N2` (`N2 <= current order`): target is `MixedTaylorExpansion` with that one axis's `OrderedAxis` order replaced by `N2` (a sub-box); reindex via `embedMixed`-style mapping, dropping monomials whose `Name`-axis degree `> N2` (they fall out of the smaller box → `flatOf == kNotInBox`).

- [ ] **Step 1: Write failing tests** — `deriv<"x">` / `integ<"x">` vs the isotropic superset (build the same fn on `MixedTaylorExpansion` and `TE<Σ,vars>`, deriv both, compare box coeffs); `slice<"x">` of an `{x,t}` expansion drops `t`-degree>0 monomials and keeps the `x` sub-expansion; `truncate<"t",2>` of `{x@4,t@20}` yields `{x@4,t@2}` with the high-`t` coefficients dropped and the rest preserved (compare to the original's low-`t` coefficients).

- [ ] **Step 2: Build (fail) → Step 3: Implement → Step 4: run green → Step 5: clang-format + commit** (`feat(core): mixed-named slice/deriv/integ/truncate by axis name`).

---

### Task 4: Name-addressed `la` for `MixedTaylorExpansion`

**Files:** Create `include/tax/la/mixed_named.hpp`; Modify `include/tax/tax.hpp` (add include after `la.hpp`); Create `tests/mixed/test_mixed_named_la.cpp`; Modify `tests/CMakeLists.txt`.

**Interfaces produced (in `tax`):** `gradient<"name">(f)`, `hessian<"name">(f)`, `jacobian<"name">(F)` for `MixedTaylorExpansion` — build the `MultiIndex<vars_v>` for the named axis's coordinates and call `f.inner().derivative(alpha)` (mirror `include/tax/la/named.hpp`). `Eigen::NumTraits<MixedTaylorExpansion<…>>` (delegate to the wrapped `Inner`'s `NumTraits`, mirroring how `la/named.hpp` does it for `NamedTaylorExpansion`) so `tax::la::VecNT<D, MixedTaylorExpansion<…>>` works.

- [ ] **Step 1: Write failing tests** — `gradient<"x">(f)` / `jacobian<"x">(F)` for a multi-axis `MixedTaylorExpansion` vs analytic / isotropic-superset values; `tax::la::VecNT` of `MixedTaylorExpansion` interop.
- [ ] **Step 2: Build (fail) → Step 3: Implement (mirror `la/named.hpp`) → Step 4: run green → Step 5: clang-format + commit** (`feat(la): name-addressed gradient/hessian/jacobian for MixedTaylorExpansion`).

---

### Task 5: Umbrella + milestone gate

- [ ] **Step 1:** Confirm `tax.hpp` includes `core/mixed_named.hpp` and `la/mixed_named.hpp` (added in Tasks 1/4); `git diff include/tax/tax.hpp` shows only the added include lines (no reorder).
- [ ] **Step 2:** Clean gate: `rm -rf build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build --output-on-failure` → ALL pass (existing suite + the new mixed-named tests).
- [ ] **Step 3:** Confirm the existing joint-simplex `NamedTaylorExpansion` and `tax::variable<"x",N>` are untouched: `git diff main -- include/tax/core/named.hpp include/tax/la/named.hpp` shows only the M3 backing-spelling change (no M4 edits).
- [ ] **Step 4:** Commit any final touch (`feat: expose mixed-named layer via the umbrella`).

## Self-Review notes

- **Spec coverage (M4):** the named per-axis-order layer (`OrderedAxis`, `MixedTaylorExpansion`, `tax::mixed` factories, max-per-axis promotion/`embedMixed`/`slice`/`deriv`/`integ`/`truncate`, name-addressed `la`) — spec §"The named per-axis-order layer". Docs are M5.
- **Reuse over duplication:** the generic axis helpers (`OffsetOf`/`DimOfName`/`TotalDim`/`buildAxisMap`/`IsCanonical`/`IsSubsetOf`) are reused as-is (duck-typed on `name`/`dim`); only `MergeOrdered` (max-order) and `embedMixed` (box→box) are new. The binary ops delegate to the **unified `TaylorExpansion`** operators on the inners, so the math/`la` surface is inherited, not re-implemented.
- **Picks up the M3 minor:** the loose compile-time `coeff<Alpha...>` `static_assert` (summed vs per-group caps) — if a named compile-time `coeff<"name",exps…>` is added, enforce per-axis caps there.
- **Deferred:** the optional joint cap (named type stays full box); cross-operand promotion between different explicit caps (compile error).

## Execution Handoff

After M4, M5 is docs (`docs/guide/mixed.md` + nav) — the final milestone.
