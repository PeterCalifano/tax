# Mixed-order Milestone 3 — unify `TaylorExpansion` over `IndexScheme` (Option B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **After committing + reporting DONE, STOP — do not self-review or amend; the controller runs review.**

**Goal:** Re-parameterize the dense `TaylorExpansion` by its **index scheme** so that `TaylorExpansion<T, IsotropicScheme<N,M>>` is today's type (alias `TE<N,M>`) and `TaylorExpansion<T, MixedScheme<Groups…>>` is the anisotropic type (alias `MixedTE<Groups…>`) — one type, one math + `tax::la` surface, the mixed type a first-class `TaylorExpansion`. **No separate `MixedExpansion`.**

**Architecture:** The class becomes `TaylorExpansion<T, Scheme, Storage>`. The dense methods (`value`/`coeff`/`variable`/`derivative`/`eval`/`deriv`/`integ`) and the operator/`la`/`io` surface migrate from `<T,N,M>` to `<T,Scheme>`, using `Scheme::flatOf/multiOf/nCoeff/order/vars/isUnivariate` (already provided by both schemes from M1/M2). Isotropic behavior is **bit-identical** (the full existing suite is the gate). Sparse stays isotropic-only. The batch coefficient (`T`) is orthogonal to the scheme, so `TE<N,M,K>` is preserved.

**Tech Stack:** C++23, header-only, Eigen3, Google Test, CMake. Build in the `tax` mamba/conda env (active).

**Milestone 3 of 5** from `docs/superpowers/specs/2026-06-21-mixed-order-named-expansions-design.md`. M1 (`IndexScheme`/`IsotropicScheme`) and M2 (`MixedScheme` + stencils) are **complete on this branch**. M4 (named per-axis-order layer wrapping `TaylorExpansion<T, MixedScheme<…>>`) and M5 (docs) follow. This M3 supersedes the earlier `2026-06-21-mixed-order-m3-mixed-expansion.md` (separate-type plan), which is obsolete.

## Global Constraints

- C++23, header-only — all code in `include/tax/`. No per-operation heap (storage `std::array`; scheme stencil tables are built-once runtime statics).
- **Isotropic behavior is bit-identical.** `TE<N,M>` / `STE<N,M>` results, `constexpr`-ness, and the full existing suite must be unchanged. The suite is the gate — Option B *intentionally* edits `taylor_expansion.hpp`/operators/`la`/`named`/`io`, so "no file diff vs main" does NOT apply; **"suite green + identical numeric results" is the behavior contract.**
- **Preserve batch:** `TE<int N, int M=1, int K=1>` keeps its `K` (batch lane count); `K>1` selects `Batch<double,K>` as the coefficient `T` (orthogonal to `Scheme`).
- **Sparse stays isotropic-only:** the Sparse specialization pairs only with `IsotropicScheme` (`static_assert`); no mixed sparse.
- Binary ops between two dense expansions require the **same `Scheme`** (mixed cross-axis promotion is M4).
- clang-format **v21**: format only edited regions; after `clang-format -i`, `git diff` and revert gratuitous reflow of pre-existing lines you did not logically change (`#    define` in `cauchy.hpp`, `requires`-braces, include order). The diff is large by design — but every hunk should be an intended `<T,N,M>`→`<T,Scheme>` migration, not formatting noise.
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Branch `feature/mixed-order-expansions` (checked out).

## Build & test reference (from repo root `/Users/andrea/Documents/Codes/tax`)

- Configure (once if stale): `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
- Full suite (THE gate): `cmake --build build -j && ctest --test-dir build --output-on-failure`
- One target/test: `cmake --build build -j --target <name>` / `ctest --test-dir build -R <name> --output-on-failure`

## Interfaces from M1/M2 (already on the branch)

- `tax::IndexScheme` concept; `tax::IsotropicScheme<N,M>` (`nCoeff/order/vars/isUnivariate/flatOf/multiOf/forEachRecurrenceRow/cauchyProduct<T>/cauchySelfProduct<T>`); `tax::Group<Dim,Order>`, `tax::MixedScheme<Groups…>` (same surface + `kNotInBox`).
- Scheme-generic kernels `tax::detail::kernels::seriesX<T,Scheme>(…)` for the full surface; free `tax::cauchyProduct<T,Scheme>`/`tax::cauchySelfProduct<T,Scheme>`.

## Migration blast radius (measured) — `TaylorExpansion<…>` spellings to migrate

`operators/arithmetic.hpp` (79), `core/named.hpp` (35), `operators/math_binary.hpp` (19), `core/taylor_expansion.hpp` (17), `operators/math_unary.hpp` (11), `la/named.hpp` (11), `io/series.hpp` (6), `la/num_traits.hpp` (4), `la/values.hpp` (3), `la/derivatives.hpp` (2), `core/batch.hpp` (2). The class-signature change is **atomic** — every header in the umbrella must compile against the new signature before the suite builds, so Task 1 is one coupled change gated by the suite.

## File Structure

- `include/tax/core/index_scheme.hpp` — **modify**: add `static constexpr std::size_t kNotInBox` + make `IsotropicScheme::flatOf` return it when `totalDegree(a) > N` (uniform out-of-set sentinel; the only isotropic behavior touched is previously-out-of-range `coeff(α)` with `|α|>N`, which now returns 0 — untested, safe).
- `include/tax/core/taylor_expansion.hpp` — **modify (core)**: class → `<T, Scheme, Storage>`; Dense spec generic over `Scheme`; Sparse spec `<T, Scheme, Sparse>` (IsotropicScheme-only); aliases `TE`/`TEn`/`STE` rewritten + new `MixedTE`.
- `include/tax/operators/{arithmetic,math_unary,math_binary}.hpp`, `include/tax/la/{num_traits,values,derivatives,named}.hpp`, `include/tax/core/named.hpp`, `include/tax/io/series.hpp`, `include/tax/core/batch.hpp` — **modify**: migrate `<T,N,M>` → `<T,Scheme>` (dense, scheme-generic) / `<T,IsotropicScheme<N,M>>` (sparse + existing named backing).
- `tests/mixed/test_mixed_te.cpp`, `tests/mixed/test_mixed_la.cpp` — **new** (M3 Tasks 2–3).

---

### Task 1: Re-parameterize `TaylorExpansion<T, Scheme, Storage>` and migrate all consumers

This is one atomic, suite-gated change. Work header-by-header guided by build errors; the deliverable is **the full existing suite green and bit-identical**, with `MixedTE<Groups…>` instantiable.

**Files:** all "modify" files above. **Interfaces produced:**
- `template<typename T, typename Scheme, typename Storage=storage::Dense> requires IndexScheme<Scheme> class TaylorExpansion;` with Dense + Sparse specializations.
- Dense exposes: `using scheme = Scheme; using scalar_type = T; using Input = std::array<T, Scheme::vars>; static constexpr int order_v = Scheme::order; static constexpr int vars_v = Scheme::vars; static constexpr std::size_t nCoefficients = Scheme::nCoeff;` plus the existing accessor/factory/diff/eval surface, now scheme-driven.
- Aliases: `template<int N,int M=1,int K=1> using TE = TaylorExpansion<std::conditional_t<K==1,double,Batch<double,K>>, IsotropicScheme<N,M>>;` `template<int N,int M> using TEn = TaylorExpansion<double, IsotropicScheme<N,M>>;` `template<int N,int M=1> using STE = TaylorExpansion<double, IsotropicScheme<N,M>, storage::Sparse>;` `template<typename... Groups> using MixedTE = TaylorExpansion<double, MixedScheme<Groups...>>;`

- [ ] **Step 1: Add `kNotInBox` to the scheme surface**

In `include/tax/core/index_scheme.hpp`, add to `IsotropicScheme`:
```cpp
static constexpr std::size_t kNotInBox = std::size_t( -1 );
```
and make `flatOf` return it for out-of-set indices:
```cpp
[[nodiscard]] static constexpr std::size_t flatOf( const MultiIndex< M >& a ) noexcept
{
    if ( totalDegree( a ) > N ) return kNotInBox;
    return flatIndex< M >( a );
}
```
(`MixedScheme` already has `kNotInBox`/`flatOf`.) Run the full suite — still 38/38 (no existing path queries `|α|>N`).

- [ ] **Step 2: Re-parameterize the class declaration + aliases (`taylor_expansion.hpp`)**

Change the forward declaration to `template<typename T, typename Scheme, typename Storage = storage::Dense> requires IndexScheme<Scheme> class TaylorExpansion;` (include `<tax/core/index_scheme.hpp>`). Change the Dense partial specialization head to `class TaylorExpansion<T, Scheme, storage::Dense>` and the Sparse head to `class TaylorExpansion<T, Scheme, storage::Sparse>` with `static_assert` that `Scheme` is an `IsotropicScheme` instantiation (a trait `is_isotropic_scheme<Scheme>`; add a tiny one in `index_scheme.hpp`). Rewrite the four aliases as above.

- [ ] **Step 3: Migrate the Dense specialization body to `Scheme`**

Inside the Dense class, replace: `N`→`Scheme::order`, `M`→`Scheme::vars`, `numMonomials(N,M)`→`Scheme::nCoeff`, `Coeffs<T,N,M>`→`std::array<T,Scheme::nCoeff>`, `flatIndex<M>(a)`→`Scheme::flatOf(a)`, `unflatIndex<M>(k)`→`Scheme::multiOf(k)`, the `if constexpr (M==1)` fast paths→`if constexpr (Scheme::isUnivariate)`. For `coeff(alpha)`/`deriv`/`integ`: where a built multi-index could be out of set, guard on `Scheme::flatOf(...) == Scheme::kNotInBox` (→ coefficient 0 / drop the scatter) — this makes `integ`'s old `totalDegree ≥ N` drop and the mixed box-drop the same code. `variable<I>` `static_assert(Scheme::flatOf(unit_I) != Scheme::kNotInBox, "variable: coordinate's group has order 0")`. Keep the math identical to the current isotropic code (just re-expressed via the scheme).

- [ ] **Step 4: Migrate the Sparse specialization (`taylor_expansion.hpp`)**

The Sparse class stays isotropic-only. Replace its `N`/`M` uses with `Scheme::order`/`Scheme::vars` (it already keys on N,M; mechanical). Keep its sparse internals (idx/val vectors, `seriesReciprocalSparse` etc.) unchanged in logic. `STE<N,M>` resolves to `TaylorExpansion<double, IsotropicScheme<N,M>, Sparse>`.

- [ ] **Step 5: Migrate operators (`arithmetic.hpp`, `math_unary.hpp`, `math_binary.hpp`)**

Rewrite the dense overloads from `template<typename T,int N,int M> … (TaylorExpansion<T,N,M>)` to `template<typename T, IndexScheme Scheme> … (TaylorExpansion<T,Scheme>)`, calling the scheme-generic kernels (`detail::kernels::seriesX<T,Scheme>` / `tax::cauchyProduct<T,Scheme>`). The unary-math macros (`TAX_UNARY_OP*`) change their generated signature to `<T, IndexScheme Scheme>` over `TaylorExpansion<T,Scheme>` calling `KERNEL<T,Scheme>`. Sparse overloads stay `<typename T,int N,int M>` over `TaylorExpansion<T, IsotropicScheme<N,M>, storage::Sparse>` (or keep them keyed via the sparse spelling). `pow`/`atan2` in `math_binary.hpp` likewise become `<T,Scheme>` (the real-exponent `pow` keeps its separate `std::floating_point P` param). **Worked example (one unary):**
```cpp
// before: template<typename T,int N,int M> TaylorExpansion<T,N,M> exp(const TaylorExpansion<T,N,M>& x)
template < typename T, IndexScheme Scheme >
[[nodiscard]] TaylorExpansion< T, Scheme > exp( const TaylorExpansion< T, Scheme >& x ) noexcept
{
    TaylorExpansion< T, Scheme > r;
    detail::kernels::seriesExp< T, Scheme >( r.coefficients(), x.coefficients() );
    return r;
}
```

- [ ] **Step 6: Migrate `la` (`num_traits.hpp`, `values.hpp`, `derivatives.hpp`)**

Rewrite `Eigen::NumTraits<TaylorExpansion<T,N,M,Storage>>` → `<TaylorExpansion<T,Scheme,Storage>>`; `variables`/`value`/`eval`/`gradient`/`hessian`/`jacobian` from `<T,N,M>` to `<T,Scheme>` (they already derive scalar/vars from traits and call `derivative(MultiIndex<vars>)`/member `gradient()` — which now work for any scheme). `gradient()`/`hessian()` members in the Dense class return `tax::la::VecNT<Scheme::vars, T>` etc.

- [ ] **Step 7: Migrate `core/named.hpp` + `la/named.hpp` (existing joint-simplex named)**

The existing `NamedTaylorExpansion<T,N,Axes...>` is unchanged in semantics; only its backing spelling moves: `using Inner = TaylorExpansion<T, IsotropicScheme<N, vars_v>>;` (was `TaylorExpansion<T,N,vars_v>`). Its `embed`/`slice` use `flatIndex<vars_v>`/`unflatIndex<vars_v>` directly (NOT through the scheme) — leave those as-is (they operate on the isotropic layout). Its binary-op result types and `la/named.hpp` helpers: update any `TaylorExpansion<T,N,M>` spelling to the `IsotropicScheme` form. No behavior change.

- [ ] **Step 8: Migrate `io/series.hpp` and `core/batch.hpp`**

`io/series.hpp`: `operator<<`/`writeSeries` from `<T,N,M>` to `<T,Scheme>` (it uses `numMonomials(N,M)`/`unflatIndex<M>`/`totalDegree` — re-express via `Scheme::nCoeff`/`Scheme::multiOf`). `core/batch.hpp`: its 2 `TaylorExpansion<…>` spellings (if any name the class directly) update to the scheme form; the `Batch`/`NumTraits<Batch>` content is unaffected (coefficient-type concern).

- [ ] **Step 9: Build the FULL suite green (bit-identical gate)**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`
Expected: **38/38 PASS** (all existing isotropic + batch + named + la + io + the M1/M2 mixed-scheme/kernel tests). If anything fails, the migration changed isotropic behavior — fix until bit-identical. Also confirm `MixedTE` instantiates: add a one-line compile check in an existing mixed test or a scratch `static_assert` that `tax::MixedTE<tax::Group<1,4>, tax::Group<1,3>>::nCoefficients == 20` (remove the scratch before commit, or keep as a tiny test).

- [ ] **Step 10: clang-format edited regions + commit**

```bash
cd /Users/andrea/Documents/Codes/tax
clang-format -i $(git diff --name-only)
git diff --stat   # sanity: only the intended headers; no #define/requires churn
git add -A
git commit -m "refactor(core): re-parameterize TaylorExpansion over IndexScheme (unify mixed + isotropic)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

> This is a large, atomic, mechanical-but-coupled migration. If you get genuinely stuck (template deduction/ambiguity you cannot resolve, or an isotropic test that won't go green), STOP and report BLOCKED with the specific compile error / failing test — do NOT weaken any test or change isotropic numeric behavior to force green.

---

### Task 2: `MixedTE` math-surface oracle (through the unified type)

**Files:** Create `tests/mixed/test_mixed_te.cpp`; Modify `tests/CMakeLists.txt`.

**Interfaces:** Consumes `tax::MixedTE<Groups…>` (Task 1) and the unified operator/math surface; oracle `tax::TE<Σorder, vars>`.

- [ ] **Step 1: Write the oracle tests**

For `using ME = tax::MixedTE<tax::Group<1,4>, tax::Group<1,3>>;` (vars=2, Σ=7) and `using ISO = tax::TE<7,2>;`, build the SAME polynomial on both (`variable<0>`/`variable<1>` + a constant), apply each of `+ - * /`, `exp/log/sqrt/sin/cos/...` (full surface) and `pow`/`atan2`, and assert every box coefficient matches: `me_result[k] == iso_result[tax::flatIndex<2>(ME::scheme::multiOf(k))]` to `1e-12`. This re-validates the M2 oracle **through the `TaylorExpansion` type + operators** (not just raw kernels). Use the public surface: `auto x = ME::variable<0>(p); auto f = sin(x)*exp(x);` etc. Domain-restricted fns: seed the constant term appropriately.

- [ ] **Step 2: Register, build, run green**

Add `tax_add_test(test_mixed_te SOURCES mixed/test_mixed_te.cpp)`.
Run: `cmake --build build -j --target test_mixed_te && ctest --test-dir build -R test_mixed_te --output-on-failure` → PASS.

- [ ] **Step 3: clang-format + commit** (`test(mixed): MixedTE math surface matches isotropic superset via the unified type`).

---

### Task 3: `tax::la` on `MixedTE`

**Files:** Create `tests/mixed/test_mixed_la.cpp`; Modify `tests/CMakeLists.txt`. (Source fixes only if a `la` path breaks — Task 1 made `la` scheme-generic, so none expected.)

**Interfaces:** `tax::la::variables`/`value`/`eval`/`gradient`/`hessian`/`jacobian` and `Eigen::NumTraits` on `MixedTE`.

- [ ] **Step 1: Write the `la` tests**

For a `MixedTE<Group<1,4>, Group<1,3>>` (vars=2): verify `f.gradient()` and `tax::la::gradient(f)`/`jacobian(F)`/`value(F)` match the isotropic `TE<7,2>` superset (or analytic values) per component; build a `tax::la::VecNT<2, ME>` and run `jacobian` — exercising `NumTraits<TaylorExpansion<double, MixedScheme<…>>>`. Mirror `tests/eigen/`-style checks.

- [ ] **Step 2: Build + run green**

Run: `cmake --build build -j --target test_mixed_la && ctest --test-dir build -R test_mixed_la --output-on-failure` → PASS. If a `la` helper fails to compile for the mixed scheme, fix it narrowly at the failing site (prefer deriving from `Scheme`/traits over assuming `(N,M)`); report what you changed.

- [ ] **Step 3: clang-format + commit** (`test(eigen): tax::la works on MixedTE (gradient/jacobian/NumTraits)`).

---

### Task 4: Milestone gate + cleanup

**Files:** optional cleanup in kernels; verification only otherwise.

- [ ] **Step 1: Retire the now-unused `<T,N,M>` kernel wrappers (optional but preferred)**

After Task 1, operators call `seriesX<T,Scheme>` directly, so the M1 `<T,N,M>` kernel wrappers in `algebra.hpp`/`transcendental.hpp`/`trigonometric.hpp`/`cauchy.hpp` may be unused. Grep for remaining callers (`grep -rn "seriesExp< [a-zA-Z_]*, [0-9]" include tests` etc.); if none, remove the `<T,N,M>` wrapper overloads (keep the `<T,Scheme>` forms). If any caller remains (e.g. sparse paths), leave the wrappers. Full suite must stay green.

- [ ] **Step 2: Clean build + full suite (bit-identical gate)**

Run: `rm -rf build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build --output-on-failure`
Expected: ALL pass — existing suite (isotropic/batch/named/la/io/sparse) unchanged + `test_mixed_te` + `test_mixed_la` + the M1/M2 mixed tests.

- [ ] **Step 3: Behavior-identity spot check**

Confirm a couple of representative isotropic results are unchanged (e.g. an existing trig/exp test value) — already covered by the suite, but note in the report that the suite is the bit-identity proof (Option B changes files by design, so there is no "empty git diff" gate).

- [ ] **Step 4: commit any cleanup** (`refactor(kernels): drop unused <T,N,M> kernel wrappers after scheme migration`).

## Self-Review notes

- **Spec coverage (M3):** unifies `TaylorExpansion` over `IndexScheme` (spec §"Unify `TaylorExpansion`…"), so `MixedTE` is a first-class `TaylorExpansion` with the full math + `la` + `io` surface; validated by the isotropic regression gate (Task 1/4) and the isotropic-superset oracle through the type (Tasks 2–3). Named per-axis-order layer is M4; docs are M5.
- **Risk:** Task 1 is a large atomic refactor; the safety net is the bit-identical existing suite. The bulk is mechanical `<T,N,M>`→`<T,Scheme>` substitution; the genuinely new logic is the `kNotInBox` guard unifying `integ`/`coeff` and the scheme-driven `eval`/`deriv`/`integ`.
- **Picks up M2 deferrals:** the `kNotInBox` guard generalizes `integ`'s drop; the `multiOf` debug-assert / table-cache (M2 minors) can be folded into Task 1/3 if `eval`/`deriv` make `multiOf` hot — note if skipped.
- **Batch preserved:** `TE<N,M,K>` keeps `K`; `MixedTE` could later take batch coefficients (not a deliverable here).

## Execution Handoff

After M3, M4 (named per-axis-order layer: `OrderedAxis`, `MixedTaylorExpansion<T,Axes…>` wrapping `TaylorExpansion<T, MixedScheme<…>>`, `tax::mixed` factories, max-per-axis promotion/`embedMixed`/`slice`/`deriv`/`integ`/`truncate<"name",N2>`, thin named-`la` helpers) is planned against the unified `TaylorExpansion` interface.
