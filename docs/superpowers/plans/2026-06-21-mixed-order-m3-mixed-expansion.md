# Mixed-order Milestone 3 — core `MixedExpansion<T, Groups…>` type

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. After committing + reporting DONE, STOP — do not self-review or amend; the controller runs review.

**Goal:** A names-free, user-facing dense expansion type `tax::MixedExpansion<T, Groups…>` over a `MixedScheme`, with the full accessor + differential + arithmetic + math surface, validated against the isotropic `TaylorExpansion<T, Σorder, vars>` superset oracle.

**Architecture:** `MixedExpansion` holds a `std::array<T, MixedScheme::nCoeff>` and mirrors `TaylorExpansion`'s public API, but indexes through `MixedScheme::flatOf`/`multiOf` (the anisotropic box). The math surface (sin/exp/sqrt/pow/…) is macro-wrapped over the M1/M2 scheme-generic kernels, so it comes essentially for free; `eval`/`deriv`/`integ` are reimplemented box-aware. Binary ops require operands over the SAME `Groups…` (cross-axis promotion is the M4 named layer).

**Tech Stack:** C++23, header-only, Eigen3, Google Test, CMake. Build in the `tax` mamba/conda env (active).

**This is Milestone 3 of 5** from `docs/superpowers/specs/2026-06-21-mixed-order-named-expansions-design.md`. M1 (`IndexScheme`/`IsotropicScheme`) and M2 (`MixedScheme` + stencils) are **complete on this branch**. M4 (named layer: `OrderedAxis`/`MixedTaylorExpansion` + promotion/embed/slice/truncate-by-name) and M5 (`la::mixed` + docs) follow.

## Global Constraints

- C++23, header-only — all code in `include/tax/`. **No per-operation heap** (storage is `std::array`; the scheme stencil tables are pre-existing runtime statics).
- **Isotropic untouched:** do not modify `TaylorExpansion` or its operators/`tax.hpp` public behavior; the full existing suite must stay green. `MixedExpansion` is a parallel, additive type.
- **Binary ops are same-scheme only** in M3: `MixedExpansion<T, Groups…> ∘ MixedExpansion<T, Groups…>` (identical `Groups…`). Mixed-axis-set promotion is M4.
- **Correctness oracle:** every box coefficient of a `MixedExpansion` result must equal the same coefficient of an isotropic `TaylorExpansion<T, Σorder, vars>` computing the same expression (box ⊆ order-Σorder simplex; degrees add ⇒ no out-of-box factor reaches an in-box output). Use this for the math/arith/deriv/integ surface.
- Keep `constexpr` where `TaylorExpansion` is: storage/ctors/accessors/coeff/`deriv`/`integ`/`eval` and the pure-polynomial ops (`square`/`cube`/`reciprocal`) are `constexpr`; transcendental wrappers are runtime (their kernels call `std::exp` etc.), exactly mirroring `operators/math_unary.hpp`.
- clang-format **v21** locally: format only newly added code; after `clang-format -i`, `git diff` and revert any gratuitous reflow of pre-existing lines (`#    define` in `cauchy.hpp`, `requires`-braces in `concepts.hpp`, `tax.hpp` include order). Commit only intended logical changes.
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Work on branch `feature/mixed-order-expansions` (checked out).

## Build & test reference (from repo root `/Users/andrea/Documents/Codes/tax`)

- Configure (once if stale): `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
- One target: `cmake --build build -j --target <name>`; one test exe: `ctest --test-dir build -R <name> --output-on-failure`
- Full suite: `cmake --build build -j && ctest --test-dir build --output-on-failure`

## Interfaces from M1/M2 (already on the branch)

- `tax::Group<int Dim, int Order>`; `tax::MixedScheme<Groups…>` with: `static constexpr std::size_t nCoeff`; `static constexpr int vars`, `order`; `static constexpr bool isUnivariate`; `static constexpr std::size_t kNotInBox`; `static constexpr std::size_t flatOf(const MultiIndex<vars>&)` (→ kNotInBox if out of box); `static constexpr MultiIndex<vars> multiOf(std::size_t)`; and members `cauchyProduct<T>`, `cauchySelfProduct<T>`, `forEachRecurrenceRow(fn)` (`include/tax/core/mixed_scheme.hpp`, `include/tax/kernels/mixed_stencils.hpp`).
- Scheme-generic kernels `tax::detail::kernels::seriesX<T, Scheme>(std::array<T,Scheme::nCoeff>&, …)` for the full surface; free `tax::cauchyProduct<T,Scheme>` / `tax::cauchySelfProduct<T,Scheme>`.
- Oracle reference: `tax::TE<SUM, V>` (= `TaylorExpansion<double, SUM, V>`); `tax::flatIndex<V>`, `tax::numMonomials`, `tax::MultiIndex<V>`.

## File Structure

- `include/tax/core/mixed_expansion.hpp` — **new**: the `MixedExpansion<T, Groups…>` type (storage, ctors, factories, accessors, `eval`/`deriv`/`integ`, member compound-assign arithmetic).
- `include/tax/operators/mixed_ops.hpp` — **new**: free arithmetic operators (`+ - * /`, scalar forms) and the macro-wrapped unary/binary math surface for `MixedExpansion`.
- `include/tax/tax.hpp` — **modify**: add the two includes (after the existing core/operators includes; do NOT reorder existing ones).
- `tests/mixed/test_mixed_expansion.cpp`, `tests/mixed/test_mixed_ops.cpp` — **new** (register in `tests/CMakeLists.txt`).

Deferred to M4 (do NOT build here): cross-`Groups` promotion/embed, `slice`/`truncate`-by-axis-name, named axes. M3's binary ops are same-`Groups` only.

---

### Task 1: `MixedExpansion` type — storage, ctors, factories, accessors

**Files:**
- Create: `include/tax/core/mixed_expansion.hpp`
- Create: `tests/mixed/test_mixed_expansion.cpp`; Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces `tax::MixedExpansion<T, Groups…>` with: `using scheme = MixedScheme<Groups…>;`, `using scalar_type = T;`, `using Input = std::array<T, scheme::vars>;`, `using Data = std::array<T, scheme::nCoeff>;`, `static constexpr int vars_v = scheme::vars;`, `static constexpr std::size_t nCoefficients = scheme::nCoeff;`; ctors `MixedExpansion()` (zero), `MixedExpansion(T)` (constant), `explicit MixedExpansion(Data)`; factories `zero()`, `constant(T)`, `variable<int I>(const Input&)` (requires the group owning coordinate `I` has order ≥ 1), `variable(T x0, int var_idx)` (runtime, throws on range); accessors `value()`, `operator[](size_t)` (read+write), `coeff(const MultiIndex<vars>&)`, `coeff<int... Alpha>()`, `derivative(const MultiIndex<vars>&)`; `coefficients()` (read+write `Data&`).

- [ ] **Step 1: Write failing tests**

Create `tests/mixed/test_mixed_expansion.cpp`:

```cpp
#include <gtest/gtest.h>

#include <tax/core/mixed_expansion.hpp>
#include <tax/core/multi_index.hpp>

using tax::Group;
using tax::MixedExpansion;

TEST( MixedExpansion, ConstructAndAccess )
{
    using ME = MixedExpansion< double, Group< 1, 4 >, Group< 1, 3 > >;  // vars=2
    static_assert( ME::vars_v == 2 );
    static_assert( ME::nCoefficients == 20 );  // 5 * 4

    ME c{ 2.5 };  // constant
    EXPECT_DOUBLE_EQ( c.value(), 2.5 );
    EXPECT_DOUBLE_EQ( c[0], 2.5 );

    typename ME::Input p{ 0.3, -0.2 };
    auto x = ME::template variable< 0 >( p );   // coord 0 (group 0, order 4)
    auto y = ME::template variable< 1 >( p );   // coord 1 (group 1, order 3)
    EXPECT_DOUBLE_EQ( x.value(), 0.3 );
    EXPECT_DOUBLE_EQ( y.value(), -0.2 );
    // linear coefficient of x is 1 at monomial e0
    tax::MultiIndex< 2 > e0{ 1, 0 }, e1{ 0, 1 };
    EXPECT_DOUBLE_EQ( x.coeff( e0 ), 1.0 );
    EXPECT_DOUBLE_EQ( x.coeff( e1 ), 0.0 );
    EXPECT_DOUBLE_EQ( y.coeff( e1 ), 1.0 );
}

TEST( MixedExpansion, DerivativeScalingMatchesMonomial )
{
    using ME = MixedExpansion< double, Group< 2, 3 > >;  // 2 vars, order 3, vars=2
    ME f{};
    tax::MultiIndex< 2 > a{ 2, 1 };          // monomial x0^2 x1
    f[f.scheme::flatOf( a )] = 1.0;          // raw coeff 1
    // derivative d^3/dx0^2 dx1 of x0^2 x1 = 2! * 1! * (raw coeff) = 2
    EXPECT_DOUBLE_EQ( f.derivative( a ), 2.0 );
}
```

- [ ] **Step 2: Register + confirm fail**

Add `tax_add_test(test_mixed_expansion SOURCES mixed/test_mixed_expansion.cpp)` to `tests/CMakeLists.txt`.
Run: `cmake --build build -j --target test_mixed_expansion 2>&1 | tail -20` → FAIL (`mixed_expansion.hpp` missing).

- [ ] **Step 3: Implement `include/tax/core/mixed_expansion.hpp`**

Mirror `TaylorExpansion<T,N,M,Dense>`'s public surface (see `include/tax/core/taylor_expansion.hpp:30-200`), substituting `MixedScheme<Groups…>` for the `(N,M)` layout: storage `Data c_{}`; `value()=c_[0]`; `variable<I>` sets `c_[0]=p[I]` and `c_[scheme::flatOf(unit_I)]=1` (with `static_assert` that `scheme::flatOf(unit_I) != scheme::kNotInBox`, i.e. the owning group's order ≥ 1); `coeff(alpha)=c_[scheme::flatOf(alpha)]` (return `T{0}` if `kNotInBox`); `derivative(alpha)` multiplies `coeff(alpha)` by `Π_i alpha_i!` (copy the factorial logic from `TaylorExpansion::derivative`). Everything `constexpr` where `TaylorExpansion`'s equivalents are. The tests in Step 1 are the contract.

- [ ] **Step 4: Build + run, expect green**

Run: `cmake --build build -j --target test_mixed_expansion && ctest --test-dir build -R test_mixed_expansion --output-on-failure`
Expected: both cases PASS.

- [ ] **Step 5: clang-format + commit**

```bash
cd /Users/andrea/Documents/Codes/tax
clang-format -i include/tax/core/mixed_expansion.hpp tests/mixed/test_mixed_expansion.cpp
git add include/tax/core/mixed_expansion.hpp tests/mixed/test_mixed_expansion.cpp tests/CMakeLists.txt
git commit -m "feat(core): MixedExpansion type — storage, factories, accessors

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `eval` (evaluate the box polynomial at a displacement)

**Files:**
- Modify: `include/tax/core/mixed_expansion.hpp` (add `eval`)
- Modify: `tests/mixed/test_mixed_expansion.cpp` (append eval test)

**Interfaces:**
- Produces `T MixedExpansion::eval(const Input& dx) const` — returns `Σ_k c_[k] * Π_i dx_i^(multiOf(k)_i)` over all kept monomials.

- [ ] **Step 1: Write the failing test (brute-force oracle)**

Append to `tests/mixed/test_mixed_expansion.cpp`:

```cpp
TEST( MixedExpansion, EvalMatchesBruteForce )
{
    using ME = MixedExpansion< double, Group< 1, 2 >, Group< 2, 3 > >;  // vars=3
    ME f{};
    for ( std::size_t k = 0; k < ME::nCoefficients; ++k ) f[k] = 0.1 + 0.05 * double( k );
    typename ME::Input dx{ 0.2, -0.1, 0.3 };

    double ref = 0.0;  // brute force: sum c_k * prod dx_i^alpha_i
    for ( std::size_t k = 0; k < ME::nCoefficients; ++k )
    {
        auto a = ME::scheme::multiOf( k );
        double term = f[k];
        for ( int v = 0; v < ME::vars_v; ++v )
            for ( int e = 0; e < a[std::size_t( v )]; ++e ) term *= dx[std::size_t( v )];
        ref += term;
    }
    EXPECT_NEAR( f.eval( dx ), ref, 1e-12 );
}
```

- [ ] **Step 2: Build, expect fail** (`eval` undefined).
Run: `cmake --build build -j --target test_mixed_expansion 2>&1 | tail -10`

- [ ] **Step 3: Implement `eval`**

Add to `MixedExpansion`: build a per-variable power table `pw[v][e] = dx[v]^e` for `e` up to the max exponent variable `v` can have (= the order of `v`'s group), then `result = Σ_k c_[k] * Π_v pw[v][multiOf(k)_v]`. Iterate `k = 0..nCoeff-1` using `scheme::multiOf(k)`. `constexpr`-capable (no transcendental).

- [ ] **Step 4: Build + run green; Step 5: clang-format + commit**

Run: `cmake --build build -j --target test_mixed_expansion && ctest --test-dir build -R test_mixed_expansion --output-on-failure` → PASS.
```bash
clang-format -i include/tax/core/mixed_expansion.hpp tests/mixed/test_mixed_expansion.cpp
git add include/tax/core/mixed_expansion.hpp tests/mixed/test_mixed_expansion.cpp
git commit -m "feat(core): MixedExpansion::eval over the box

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `deriv` / `integ` (box-aware, same-scheme)

**Files:**
- Modify: `include/tax/core/mixed_expansion.hpp`
- Modify: `tests/mixed/test_mixed_expansion.cpp`

**Interfaces:**
- Produces `MixedExpansion MixedExpansion::deriv<int I>() const` / `deriv(int var_idx)` and `integ<int I>() const` / `integ(int var_idx)`, returning a `MixedExpansion` over the SAME `Groups…`.
- **deriv semantics:** for each kept `α` with `α_I > 0`, `out[flatOf(α with α_I-1)] += c_[flatOf(α)] * α_I`. (The lowered monomial is always in-box.)
- **integ semantics:** for each kept `α`, let `β = α with α_I+1`; if `flatOf(β) != kNotInBox` (still in box) then `out[flatOf(β)] += c_[flatOf(α)] / (α_I+1)`, else drop (the box-truncation guard — analog of `TaylorExpansion::integ`'s `totalDegree ≥ N` drop).

- [ ] **Step 1: Write failing tests (isotropic-superset oracle)**

Append — verify `deriv`/`integ` of a `MixedExpansion` match the isotropic `TE<Σ, vars>` on every box monomial:

```cpp
#include <tax/tax.hpp>  // for tax::TE oracle

TEST( MixedExpansion, DerivMatchesIsotropicSuperset )
{
    using ME = MixedExpansion< double, Group< 1, 4 >, Group< 1, 3 > >;  // vars=2, Σ=7
    constexpr int V = 2, SUM = 7;
    using ISO = tax::TE< SUM, V >;

    // same polynomial c0 + x0 on both layouts
    ME f{};
    f[0] = 0.3;
    tax::MultiIndex< 2 > e0{ 1, 0 };
    f[ME::scheme::flatOf( e0 )] = 1.0;
    typename ISO::Input p{ 0.3, 0.0 };
    ISO g = ISO::template variable< 0 >( p );

    auto df = f.template deriv< 0 >();
    auto dg = g.template deriv< 0 >();
    for ( std::size_t k = 0; k < ME::nCoefficients; ++k )
    {
        auto a = ME::scheme::multiOf( k );
        EXPECT_NEAR( df[k], dg[tax::flatIndex< V >( a )], 1e-12 );
    }
}
```

(Add a matching `integ<0>()` test the same way; the box-drop guard is exercised when `α_I+1` exceeds the group order.)

- [ ] **Step 2: Build, expect fail.** Run: `cmake --build build -j --target test_mixed_expansion 2>&1 | tail -10`

- [ ] **Step 3: Implement `deriv`/`integ`** per the semantics above, using `scheme::multiOf`/`flatOf`. Mirror `TaylorExpansion::deriv`/`integ` (`taylor_expansion.hpp:255-329`) but scatter through `flatOf` and apply the box guard for `integ`. Provide compile-time `<I>` and runtime `(int)` forms (runtime throws on range, like `TaylorExpansion`).

- [ ] **Step 4: Build + run green; Step 5: clang-format + commit**

```bash
clang-format -i include/tax/core/mixed_expansion.hpp tests/mixed/test_mixed_expansion.cpp
git add include/tax/core/mixed_expansion.hpp tests/mixed/test_mixed_expansion.cpp
git commit -m "feat(core): MixedExpansion deriv/integ (box-aware, same-scheme)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Arithmetic operators (same-scheme + scalar)

**Files:**
- Create: `include/tax/operators/mixed_ops.hpp`
- Create: `tests/mixed/test_mixed_ops.cpp`; Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces free operators in `tax`: `operator+`, `operator-` (binary and unary), `operator*`, `operator/` for `(MixedExpansion<T,Groups…>, MixedExpansion<T,Groups…>)` (same `Groups…`), and scalar forms `(MixedExpansion, T)` / `(T, MixedExpansion)`; plus compound `+= -= *= /=`. `*` uses `tax::cauchyProduct<T, scheme>`; `/` is `a * reciprocal(b)` (or `seriesDivide<T,scheme>` if preferred — use the kernel that exists); `+`/`-` are elementwise; scalar `*`/`+` touch the constant term / scale all coefficients exactly as `TaylorExpansion`'s scalar ops do (see `operators/arithmetic.hpp`).

- [ ] **Step 1: Write failing tests (isotropic-superset oracle for + - * /)**

Create `tests/mixed/test_mixed_ops.cpp` — build two box polynomials, do `+ - * /`, and compare every box coefficient to the isotropic `TE<Σ,vars>` doing the same (seed identical coeffs on both layouts via `flatOf`/`flatIndex`). Example for `*`:

```cpp
#include <gtest/gtest.h>

#include <tax/tax.hpp>
#include <tax/core/mixed_expansion.hpp>
#include <tax/operators/mixed_ops.hpp>

using tax::Group;
using tax::MixedExpansion;

TEST( MixedOps, ProductMatchesIsotropicSuperset )
{
    using ME = MixedExpansion< double, Group< 1, 4 >, Group< 1, 3 > >;  // vars=2, Σ=7
    constexpr int V = 2, SUM = 7;
    using ISO = tax::TE< SUM, V >;

    ME a{}, b{};
    ISO ia{}, ib{};
    for ( std::size_t k = 0; k < ME::nCoefficients; ++k )
    {
        auto al = ME::scheme::multiOf( k );
        double va = 0.1 + 0.03 * double( k ), vb = -0.2 + 0.05 * double( k );
        a[k] = va; b[k] = vb;
        ia[tax::flatIndex< V >( al )] = va;  // same coeffs in the isotropic superset
        ib[tax::flatIndex< V >( al )] = vb;
    }
    auto mc = a * b;
    auto ic = ia * ib;
    for ( std::size_t k = 0; k < ME::nCoefficients; ++k )
    {
        auto al = ME::scheme::multiOf( k );
        EXPECT_NEAR( mc[k], ic[tax::flatIndex< V >( al )], 1e-12 );
    }
}
```

(Add `+`, `-`, `/`, and a scalar-op test the same way. For `/`, seed `b`'s constant term away from 0.)

- [ ] **Step 2: Build, expect fail.** Add `tax_add_test(test_mixed_ops SOURCES mixed/test_mixed_ops.cpp)`. Run target → FAIL.

- [ ] **Step 3: Implement `operators/mixed_ops.hpp`** with the operators above. `+`/`-`/compound are elementwise loops over `nCoeff`; `*` calls `tax::cauchyProduct<T, typename ME::scheme>(r.coefficients(), a.coefficients(), b.coefficients())`; scalar ops mirror `operators/arithmetic.hpp`.

- [ ] **Step 4: Build + run green; Step 5: clang-format + commit**

```bash
clang-format -i include/tax/operators/mixed_ops.hpp tests/mixed/test_mixed_ops.cpp
git add include/tax/operators/mixed_ops.hpp tests/mixed/test_mixed_ops.cpp tests/CMakeLists.txt
git commit -m "feat(operators): MixedExpansion arithmetic (same-scheme + scalar)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Unary + binary math surface; full oracle through the type

**Files:**
- Modify: `include/tax/operators/mixed_ops.hpp` (add the math wrappers)
- Modify: `tests/mixed/test_mixed_ops.cpp` (append the full-surface oracle)

**Interfaces:**
- Produces free unary math `square cube reciprocal sqrt cbrt exp log sinh cosh tanh asinh acosh atanh erf sin cos tan asin acos atan` and binary `pow(MixedExpansion,int)`, `pow(MixedExpansion, std::floating_point)`, `atan2(MixedExpansion,MixedExpansion)` for `MixedExpansion<T,Groups…>`, each calling the scheme-generic kernel.

- [ ] **Step 1: Add the macro-wrapped math surface**

Mirror `operators/math_unary.hpp`'s `TAX_UNARY_OP_CE`/`TAX_UNARY_OP` macros, but for `MixedExpansion<T,Groups…>` calling `detail::kernels::KERNEL<T, typename MixedExpansion<T,Groups…>::scheme>(r.coefficients(), x.coefficients())`. Same `_CE` (constexpr: `square`/`cube`/`reciprocal`) vs runtime split. Add `pow`/`atan2` mirroring `operators/math_binary.hpp` (calling `seriesPowInt`/`seriesPow`/`seriesAtan2` with the scheme; the real-exponent overload takes a separate `std::floating_point P`).

- [ ] **Step 2: Write the full-surface oracle test**

Append to `tests/mixed/test_mixed_ops.cpp` — for `MixedExpansion<double, Group<1,4>, Group<1,3>>`, build a variable, apply each math function, and compare every box coefficient to the isotropic `TE<7,2>` applying the same function (seed the same polynomial; for domain-restricted fns like `sqrt`/`log`/`asin` seed the constant term appropriately, e.g. `1.5 + x`). Cover the whole surface listed above with `EXPECT_NEAR(..., 1e-12)`. (One `TEST` per function or grouped logically; each is a few lines following the Task-4 product example pattern.)

- [ ] **Step 3: Build + run green**

Run: `cmake --build build -j --target test_mixed_ops && ctest --test-dir build -R test_mixed_ops --output-on-failure`
Expected: PASS — the whole math surface on `MixedExpansion` matches the isotropic superset coefficient-for-coefficient.

- [ ] **Step 4: clang-format + commit**

```bash
clang-format -i include/tax/operators/mixed_ops.hpp tests/mixed/test_mixed_ops.cpp
git add include/tax/operators/mixed_ops.hpp tests/mixed/test_mixed_ops.cpp
git commit -m "feat(operators): MixedExpansion full math surface (oracle-verified vs isotropic)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Umbrella include, alias, milestone gate

**Files:**
- Modify: `include/tax/tax.hpp`
- Modify: `include/tax/core/mixed_expansion.hpp` (add a convenience alias)

**Interfaces:**
- Produces: `tax::tax.hpp` includes `mixed_expansion.hpp` + `mixed_ops.hpp`; a convenience alias `template<int N, int M> using MTE` is NOT added (M is per-group) — instead leave the user-facing ergonomic alias to the M4 named layer. (No alias added here unless trivial; keep M3 names-free.)

- [ ] **Step 1: Add umbrella includes**

In `include/tax/tax.hpp`, add (by hand, in dependency order, WITHOUT reordering existing lines) after `core/batch.hpp` (or after `core/taylor_expansion.hpp` if batch is absent) and after the operators block respectively:
```cpp
#include <tax/core/mixed_expansion.hpp>
...
#include <tax/operators/mixed_ops.hpp>
```
Run `git diff include/tax/tax.hpp` and confirm only the new include line(s) were added (no clang-format include reordering).

- [ ] **Step 2: Confirm the umbrella compiles a mixed expression**

Add a tiny smoke check (can live in `test_mixed_ops.cpp`): with only `#include <tax/tax.hpp>`, `auto x = tax::MixedExpansion<double, tax::Group<1,4>, tax::Group<1,3>>::variable<0>({0.3,0.0}); auto f = sin(x) + x; (void)f;` compiles and runs.

- [ ] **Step 3: Milestone gate**

Run: `rm -rf build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build --output-on-failure`
Expected: ALL pass — existing isotropic suite unchanged + the new `test_mixed_expansion` and `test_mixed_ops`.
Run: `git diff main -- include/tax/core/taylor_expansion.hpp include/tax/operators/arithmetic.hpp include/tax/operators/math_unary.hpp include/tax/operators/math_binary.hpp` → EMPTY (M3 does not change the isotropic type or its operators).

- [ ] **Step 4: Commit**

```bash
clang-format -i include/tax/core/mixed_expansion.hpp
git add include/tax/tax.hpp include/tax/core/mixed_expansion.hpp tests/mixed/test_mixed_ops.cpp
git commit -m "feat: expose MixedExpansion via the umbrella header

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

## Self-Review notes

- **Spec coverage (M3):** the core `MixedExpansion` type — value/coeff/eval/deriv/integ + the full arithmetic/math surface — validated by the isotropic-superset oracle. Named axes, promotion/embed/slice, and axis-name `truncate` are M4; `la::mixed` + docs are M5.
- **Math surface is mostly free:** the M1/M2 scheme-generic kernels mean Task 5 is macro wrapping; the real new code is the type (Task 1) and the box-aware `eval`/`deriv`/`integ` (Tasks 2–3), pinned by brute-force + isotropic-superset oracles.
- **Pick up M2 deferrals where cheap:** add the `multiOf` out-of-range debug `assert` (M2 minor) when implementing the type; if `eval`/`deriv`/`integ` make per-call `multiOf` hot, consider caching a `multiOf` table on the scheme (M2 perf minor) — optional, note if skipped.
- **Binary ops same-`Groups` only** — cross-axis promotion is explicitly M4.

## Execution Handoff

After M3 lands, M4 (named layer: `OrderedAxis<Name,Dim,Order>`, `MixedTaylorExpansion<T, Axes…>` wrapping `MixedExpansion`, factories in `tax::mixed`, max-per-axis promotion/`embed`/`slice`/`deriv`/`integ`/axis-name `truncate`) is planned against the concrete `MixedExpansion` interface.
