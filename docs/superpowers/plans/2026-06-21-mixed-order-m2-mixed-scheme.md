# Mixed-order Milestone 2 — `MixedScheme` + mixed stencils

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an anisotropic ("box") index scheme `MixedScheme<Groups…>` that satisfies the `IndexScheme` interface from M1, so the scheme-generic kernels run on a per-axis-capped monomial set with no dense blow-up — and finish making the Cauchy-based kernels scheme-generic so the whole recurrence surface works on it.

**Architecture:** A `MixedScheme` describes a list of variable groups, each `(dim, order)`, with an optional joint total-degree cap. Its kept set is the box (∩ joint cap), laid out graded by total degree (so recurrences stay causal), with a `pos` map between full multi-indices and the compact flat layout. It provides the same members the kernels use — `nCoeff`, `forEachRecurrenceRow`, `cauchyProduct`, `cauchySelfProduct` — built once at first use (runtime-static) with a `constexpr` enumeration fallback. The generalization is the M1 prototype's 2-group `MixedScheme` extended to G groups.

**Tech Stack:** C++23, header-only, Eigen3, Google Test, CMake. Build in the `tax` mamba/conda env (active).

**This is Milestone 2 of 5** from `docs/superpowers/specs/2026-06-21-mixed-order-named-expansions-design.md`. M1 (`IndexScheme` + `IsotropicScheme`, kernels scheme-generic) is **complete and merged on this branch**. M3 (core `MixedExpansion` type), M4 (named layer), M5 (`la::mixed` + docs) follow.

## Global Constraints

- C++23, header-only — all code in `include/tax/`. **No per-operation heap** in the dense core: scheme coefficient arrays are `std::array`; the stencil tables are built-once runtime statics (like `CauchyStencil`/`RecurrenceStencil`), not per-call allocations. Prefer fixed-size `std::array` tables (sizes are compile-time computable, see formulas below).
- **Graded-by-total-degree ordering is mandatory** for `MixedScheme` (forward-substitution recurrences must see all lower-degree dependencies already computed) — the analog of graded-lex's sacredness.
- **`constexpr` discipline:** the index math (`keptCount`, `flatOf`/`multiOf`, the graded enumeration) must be `constexpr`; the stencil-backed paths keep an `if !consteval` enumeration fallback (mirror `forEachRecurrenceRow` / `cauchyProductStencil`).
- **Isotropic behavior preserved:** every change to shared kernel code must keep `IsotropicScheme` results bit-identical — the full existing suite (36/36) is the gate after each task that touches shared code.
- `MixedScheme` must satisfy the M1 `IndexScheme` concept and expose the same member surface the kernels call: `nCoeff`, `order`, `isUnivariate` (= `false`), `flatOf`, `multiOf`, `forEachRecurrenceRow`, `cauchyProduct<T>`, and (added in this milestone) `cauchySelfProduct<T>`.
- Dense only; no sparse, no `Batch`, no named layer in M2.
- clang-format **v21** locally: format only newly added code; after `clang-format -i`, `git diff` and revert any gratuitous reflow of pre-existing lines (notably the `#    define` indentation in `cauchy.hpp`, `requires`-braces in `concepts.hpp`, and `tax.hpp` include order). Commit only intended logical changes.
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Work on branch `feature/mixed-order-expansions` (checked out).

## Build & test reference (from repo root `/Users/andrea/Documents/Codes/tax`)

- Configure (once if stale): `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
- One target: `cmake --build build -j --target <name>`; one test exe: `ctest --test-dir build -R <name> --output-on-failure`
- Full suite (isotropic gate): `cmake --build build -j && ctest --test-dir build --output-on-failure` (must stay 36/36 plus any new mixed tests)

## Interfaces from M1 (already on the branch)

- `tax::IndexScheme` concept; `tax::IsotropicScheme<N,M>` with `nCoeff/order/vars/isUnivariate/flatOf/multiOf/forEachRecurrenceRow/cauchyProduct<T>` (`include/tax/core/index_scheme.hpp`).
- Free `template<typename T, IndexScheme Scheme> void tax::cauchyProduct(std::array<T,Scheme::nCoeff>&, …)` forwarding to `Scheme::cauchyProduct<T>`.
- Scheme-generic recurrence kernels `template<typename T, IndexScheme Scheme> void detail::kernels::seriesX(std::array<T,Scheme::nCoeff>&, …)` for all 20 recurrences, with `<T,N,M>` wrappers preserved.
- **Known M1 carry-over:** 8 recurrence kernels (`seriesAsinh/Acosh/Atanh/Erf`, `seriesAsin/Acos/Atan/Atan2`) currently form a helper via `cauchySelfProduct<T, Scheme::order, Scheme::vars>` — using `Scheme::vars`, which is NOT in the `IndexScheme` concept and is wrong for `MixedScheme`. Task 1 removes this coupling.

## File Structure

- `include/tax/core/mixed_scheme.hpp` — **new**: `tax::Group<int Dim, int Order>`, `tax::MixedScheme<Groups…>` (index core + members), the box `keptCount`/`keptPairCount`, graded enumeration, `pos`/`flatOf`/`multiOf`.
- `include/tax/kernels/mixed_stencils.hpp` — **new**: box-filtered Cauchy stencil + self-product stencil + recurrence-row table for a `MixedScheme` (the runtime-static tables + `constexpr` fallbacks the `MixedScheme` members call).
- `include/tax/kernels/algebra.hpp` — **modify**: add scheme-generic `cauchySelfProduct<T,Scheme>`, `seriesSquare<T,Scheme>`, `seriesCube<T,Scheme>`, `seriesPowInt<T,Scheme>` (keep `<T,N,M>` wrappers); update the 8 recurrence kernels' helper calls to `cauchySelfProduct<T,Scheme>`.
- `include/tax/core/index_scheme.hpp` — **modify**: add `IsotropicScheme::cauchySelfProduct<T>` member + a free `cauchySelfProduct<T,Scheme>` (mirroring the M1 `cauchyProduct<T,Scheme>`).
- `tests/mixed/test_mixed_scheme.cpp`, `tests/mixed/test_mixed_kernels.cpp` — **new** (register in `tests/CMakeLists.txt`).

Reference (read-only): the prototype `prototypes/mixed/mixed_expansion.hpp` on `origin/claude/taylor-expansion-prototypes-hxq111` — the 2-group `MixedScheme` whose construction (graded ordering, `pos`, product stencil) this milestone generalizes to G groups.

---

### Task 1: Finish scheme-generic Cauchy-based kernels; drop the `Scheme::vars` coupling

Do this first: it is isotropic-only (no `MixedScheme` yet), removes the M1 carry-over, and is fully gated by the existing suite.

**Files:**
- Modify: `include/tax/core/index_scheme.hpp` (add `cauchySelfProduct`), `include/tax/kernels/algebra.hpp`
- Test: existing `tests/operators/*`, `tests/kernels/*`, `tests/core/test_index_scheme.cpp` (gate; behavior-preserving).

**Interfaces:**
- Produces:
  - `IsotropicScheme::cauchySelfProduct<T>(std::array<T,nCoeff>& out, const std::array<T,nCoeff>& f)` delegating to `detail::kernels::cauchySelfProduct<T,N,M>`.
  - free `template<typename T, IndexScheme Scheme> void tax::cauchySelfProduct(std::array<T,Scheme::nCoeff>& out, const std::array<T,Scheme::nCoeff>& f)` → `Scheme::template cauchySelfProduct<T>(out,f)`.
  - scheme-generic `detail::kernels::cauchySelfProduct<T,Scheme>`, `seriesSquare<T,Scheme>`, `seriesCube<T,Scheme>`, `seriesPowInt<T,Scheme>` (each with the `<T,N,M>` wrapper preserved).

- [ ] **Step 1: Add `cauchySelfProduct` to the scheme surface (`index_scheme.hpp`)**

In `IsotropicScheme` (after `cauchyProduct`):

```cpp
template < typename T >
static void cauchySelfProduct( std::array< T, nCoeff >& out,
                               const std::array< T, nCoeff >& f ) noexcept
{
    detail::kernels::cauchySelfProduct< T, N, M >( out, f );
}
```

and a free forwarder next to the M1 `cauchyProduct<T,Scheme>`:

```cpp
template < typename T, IndexScheme Scheme >
void cauchySelfProduct( std::array< T, Scheme::nCoeff >& out,
                        const std::array< T, Scheme::nCoeff >& f ) noexcept
{
    Scheme::template cauchySelfProduct< T >( out, f );
}
```

- [ ] **Step 2: Make `cauchySelfProduct`/`seriesSquare`/`seriesCube`/`seriesPowInt` scheme-generic (`algebra.hpp`)**

Apply the M1 transform (see `docs/superpowers/plans/2026-06-21-mixed-order-m1-index-scheme.md` Task 2 recipe): add `template<typename T, IndexScheme Scheme>` overloads operating on `std::array<T,Scheme::nCoeff>`, with `M==1` → `Scheme::isUnivariate`, `N` → `Scheme::order`, internal `cauchyProduct<T,N,M>`/`cauchySelfProduct<T,N,M>` calls → the scheme-generic `cauchyProduct<T,Scheme>`/`cauchySelfProduct<T,Scheme>`, and the `std::array<T,S>` scratch sizes → `Scheme::nCoeff`. Replace each old `<T,N,M>` body with a one-line wrapper delegating to `<T, IsotropicScheme<N,M>>`. For `cauchySelfProduct`'s `M>=2` branch, call `tax::cauchyProduct<T,Scheme>(out, f, f)`.

- [ ] **Step 3: Update the 8 recurrence kernels to drop `Scheme::vars`**

In `transcendental.hpp` (`seriesAsinh/Acosh/Atanh/Erf`) and `trigonometric.hpp` (`seriesAsin/Acos/Atan/Atan2`), change each helper construction `cauchySelfProduct<T, Scheme::order, Scheme::vars>(h, a)` to `cauchySelfProduct<T, Scheme>(h, a)` (the scheme-generic free function). No other change. These kernels no longer reference `Scheme::vars`.

- [ ] **Step 4: Full suite gate (behavior-preserving)**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`
Expected: 36/36 PASS — `IsotropicScheme::cauchySelfProduct` produces identical results, so every dependent kernel (erf, inverse trig, square, cube, pow) is unchanged.

- [ ] **Step 5: clang-format and commit**

```bash
cd /Users/andrea/Documents/Codes/tax
clang-format -i include/tax/core/index_scheme.hpp include/tax/kernels/algebra.hpp include/tax/kernels/transcendental.hpp include/tax/kernels/trigonometric.hpp
git diff --stat   # confirm only intended changes; no #define/requires/tax.hpp churn
git add include/tax/core/index_scheme.hpp include/tax/kernels/algebra.hpp include/tax/kernels/transcendental.hpp include/tax/kernels/trigonometric.hpp
git commit -m "refactor(kernels): scheme-generic cauchySelfProduct/square/cube/powInt; drop Scheme::vars

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `MixedScheme` index core (groups, box layout, `pos`/`flat↔multi`)

**Files:**
- Create: `include/tax/core/mixed_scheme.hpp`
- Create: `tests/mixed/test_mixed_scheme.cpp`; Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces:
  - `template<int Dim, int Order> struct tax::Group { static constexpr int dim = Dim; static constexpr int order = Order; static_assert(Dim>=1 && Order>=0); };`
  - `template<typename... Groups> struct tax::MixedScheme` with (this task): `static constexpr int groupCount`, `static constexpr int vars` (= Σ dim), `static constexpr int order` (= Σ order, the max kept total degree — joint cap deferred; see note), `static constexpr bool isUnivariate = false`, `static constexpr std::size_t nCoeff = keptCount`, and `static constexpr std::size_t flatOf(const MultiIndex<vars>&)` (returns `kNotInBox` if outside the box) / `static constexpr MultiIndex<vars> multiOf(std::size_t)`.
  - `static constexpr std::size_t MixedScheme::kNotInBox = std::size_t(-1);`

**Box & layout definitions (the contract the tests pin):**
- A multi-index `α` (length `vars`) splits into per-group sub-vectors `α_g` (consecutive blocks of `dim_g`). `α` is **in the box** iff `totalDegree(α_g) ≤ order_g` for every group `g`.
- `keptCount = Π_g numMonomials(order_g, dim_g)` (product of per-group simplex sizes).
- **Graded ordering** (generalize the prototype's loop): iterate total degree `d = 0 … Σorder`; within `d`, iterate the per-group degree tuples `(d_0,…,d_{G-1})` with `Σ d_g = d` and `d_g ≤ order_g`, in a fixed canonical order (e.g. lexicographic over `(d_0,…)`); within each tuple, iterate the per-group monomials of exactly degree `d_g` in graded-lex order, as a mixed-radix product across groups (group 0 outermost). Assign flat indices `0,1,2,…` in this visitation order. `flatOf`/`multiOf` are the inverse maps; both `constexpr`.

> **Joint cap note:** M2 ships the full box (`order = Σ order_g`, no joint cap) to keep the index core tractable; the spec's optional joint cap is a later additive parameter (`MixedScheme<…, JointCap>`), out of scope here. Record as a follow-up.

- [ ] **Step 1: Write failing table tests**

Create `tests/mixed/test_mixed_scheme.cpp`:

```cpp
#include <gtest/gtest.h>

#include <set>
#include <tax/core/mixed_scheme.hpp>
#include <tax/core/multi_index.hpp>

using tax::Group;
using tax::MixedScheme;

// Box count = product of per-group simplex sizes; differs from the joint simplex.
TEST( MixedScheme, KeptCountIsBoxProduct )
{
    // x@2 (1 var) box t@2 (1 var): 3 * 3 = 9 (joint simplex numMonomials(2,2)=6).
    static_assert( MixedScheme< Group< 1, 2 >, Group< 1, 2 > >::nCoeff == 9 );
    // x@4 (1 var) box p@20 (1 var): 5 * 21 = 105.
    static_assert( MixedScheme< Group< 1, 4 >, Group< 1, 20 > >::nCoeff == 105 );
    // x@4 over 3 vars (numMonomials(4,3)=35) box t@20: 35 * 21 = 735.
    static_assert( MixedScheme< Group< 3, 4 >, Group< 1, 20 > >::nCoeff == 735 );
    SUCCEED();
}

// flatOf/multiOf are inverse over the whole box, dense in [0,nCoeff), graded.
TEST( MixedScheme, FlatRoundTripDenseAndGraded )
{
    using S = MixedScheme< Group< 1, 2 >, Group< 2, 3 > >;  // x(1 var)@2, y(2 vars)@3
    std::set< std::size_t > seen;
    int prev_degree = -1;
    bool graded = true;
    // Walk all flats; each maps to an in-box multi-index, round-trips, and is graded.
    for ( std::size_t k = 0; k < S::nCoeff; ++k )
    {
        auto a = S::multiOf( k );
        EXPECT_EQ( S::flatOf( a ), k );                 // round trip
        // in-box: per-group degree caps
        EXPECT_LE( a[0], 2 );                            // x degree
        EXPECT_LE( a[1] + a[2], 3 );                     // y total degree
        int d = tax::totalDegree( a );
        if ( d < prev_degree ) graded = false;          // non-decreasing total degree
        prev_degree = d;
        seen.insert( k );
    }
    EXPECT_TRUE( graded );
    EXPECT_EQ( seen.size(), S::nCoeff );                 // dense, no gaps
}

// An out-of-box multi-index is rejected.
TEST( MixedScheme, OutOfBoxRejected )
{
    using S = MixedScheme< Group< 1, 2 >, Group< 1, 2 > >;
    tax::MultiIndex< 2 > over{ 3, 0 };                   // x^3, x order is 2
    EXPECT_EQ( S::flatOf( over ), S::kNotInBox );
}
```

- [ ] **Step 2: Register + confirm fail**

Add to `tests/CMakeLists.txt`: `tax_add_test(test_mixed_scheme SOURCES mixed/test_mixed_scheme.cpp)`.
Run: `cmake --build build -j --target test_mixed_scheme 2>&1 | tail -20` → FAIL (`mixed_scheme.hpp` missing).

- [ ] **Step 3: Implement `include/tax/core/mixed_scheme.hpp`**

Implement `Group`, `MixedScheme<Groups…>` with the box layout above. Use a `constexpr std::array<GroupDesc, groupCount>` of `{dim, order, varOffset, simplexSize}` built from the pack, and `constexpr` `keptCount`, `flatOf`, `multiOf` following the **graded mixed-radix** algorithm (generalize the prototype's nested loop in `prototypes/mixed/mixed_expansion.hpp` from 2 groups to a runtime loop over the group array). Per-group monomial enumeration reuses `flatIndex<dim_g>`/`unflatIndex<dim_g>` restricted to a given degree. Keep everything `constexpr` (no runtime statics in this task — pure index math). The tests in Step 1 are the exact contract.

- [ ] **Step 4: Build + run, expect green**

Run: `cmake --build build -j --target test_mixed_scheme && ctest --test-dir build -R test_mixed_scheme --output-on-failure`
Expected: all three cases PASS.

- [ ] **Step 5: clang-format and commit**

```bash
cd /Users/andrea/Documents/Codes/tax
clang-format -i include/tax/core/mixed_scheme.hpp tests/mixed/test_mixed_scheme.cpp
git add include/tax/core/mixed_scheme.hpp tests/mixed/test_mixed_scheme.cpp tests/CMakeLists.txt
git commit -m "feat(core): MixedScheme box index core (groups, pos map, graded flat<->multi)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `MixedScheme` stencils + member surface (`cauchyProduct`/`cauchySelfProduct`/`forEachRecurrenceRow`)

**Files:**
- Create: `include/tax/kernels/mixed_stencils.hpp`
- Modify: `include/tax/core/mixed_scheme.hpp` (add the member methods that call the stencils)
- Test: `tests/mixed/test_mixed_scheme.cpp` (append stencil-correctness tests)

**Interfaces:**
- Consumes: `MixedScheme` index core (Task 2); `StencilEntry`/`RecurrenceEntry`.
- Produces on `MixedScheme`: `template<typename T> static void cauchyProduct(out,a,b)`, `template<typename T> static void cauchySelfProduct(out,f)`, and `template<class RowFn> static void forEachRecurrenceRow(RowFn&&)` — completing the `IndexScheme` member surface so the scheme-generic kernels instantiate on `MixedScheme`.

**Sizing formulas (compile-time, for fixed `std::array` tables):**
- Cauchy stencil entry count (full box) = `Π_g numMonomials(order_g, 2·dim_g)` — pairs `(β_g, γ_g)` per group with `|β_g|+|γ_g| ≤ order_g`, producted across groups.
- Recurrence stencil entry count = (Cauchy count) − `nCoeff` (drop the `|β|==0` row per output, as `RecurrenceStencil` does).

- [ ] **Step 1: Write failing stencil-correctness tests**

Append to `tests/mixed/test_mixed_scheme.cpp` — verify the box Cauchy product against a brute-force reference (multiply two random-ish box polynomials, compare every coefficient):

```cpp
#include <tax/core/index_scheme.hpp>  // tax::cauchyProduct<T,Scheme>

TEST( MixedScheme, CauchyProductMatchesBruteForce )
{
    using S = MixedScheme< Group< 1, 2 >, Group< 2, 3 > >;
    std::array< double, S::nCoeff > a{}, b{}, out{};
    for ( std::size_t k = 0; k < S::nCoeff; ++k )
    {
        a[k] = 0.1 + 0.03 * double( k );
        b[k] = -0.2 + 0.05 * double( k );
    }
    tax::cauchyProduct< double, S >( out, a, b );

    // Brute force: for every pair of in-box monomials whose product is in-box, accumulate.
    std::array< double, S::nCoeff > ref{};
    for ( std::size_t i = 0; i < S::nCoeff; ++i )
        for ( std::size_t j = 0; j < S::nCoeff; ++j )
        {
            auto ai = S::multiOf( i );
            auto aj = S::multiOf( j );
            tax::MultiIndex< S::vars > sum{};
            for ( int v = 0; v < S::vars; ++v ) sum[std::size_t( v )] = ai[std::size_t( v )] + aj[std::size_t( v )];
            std::size_t o = S::flatOf( sum );
            if ( o != S::kNotInBox ) ref[o] += a[i] * b[j];
        }
    for ( std::size_t k = 0; k < S::nCoeff; ++k ) EXPECT_DOUBLE_EQ( out[k], ref[k] );
}
```

- [ ] **Step 2: Build, expect fail**

Run: `cmake --build build -j --target test_mixed_scheme 2>&1 | tail -20` → FAIL (`MixedScheme::cauchyProduct` / `forEachRecurrenceRow` not defined).

- [ ] **Step 3: Implement `mixed_stencils.hpp` + the `MixedScheme` members**

In `mixed_stencils.hpp`, build (generalizing `CauchyStencil`/`RecurrenceStencil` with box filtering and the `MixedScheme` `pos` map):
- a box Cauchy stencil `std::array<StencilEntry, kCauchyEntries>` (graded by output degree), and a box self-product stencil (or reuse the Cauchy one with `a==b`),
- a box recurrence-row table `std::array<RecurrenceEntry, kRecEntries>` + per-output `row[]` bounds + `degree[]` (exactly `RecurrenceStencil`'s shape), built in graded order.
Tables are `static const` runtime instances (one per `MixedScheme` type), as `CauchyStencil` does. On `MixedScheme` add the three members: `cauchyProduct`/`cauchySelfProduct` run the scatter-add over the box Cauchy stencil; `forEachRecurrenceRow(fn)` walks the box recurrence rows (with a `constexpr` enumeration fallback that builds rows on the fly, mirroring M1's `forEachRecurrenceRow`).

- [ ] **Step 4: Build + run, expect green**

Run: `cmake --build build -j --target test_mixed_scheme && ctest --test-dir build -R test_mixed_scheme --output-on-failure`
Expected: the Cauchy-vs-brute-force case (and Task-2 cases) PASS.

- [ ] **Step 5: clang-format and commit**

```bash
cd /Users/andrea/Documents/Codes/tax
clang-format -i include/tax/kernels/mixed_stencils.hpp include/tax/core/mixed_scheme.hpp tests/mixed/test_mixed_scheme.cpp
git add include/tax/kernels/mixed_stencils.hpp include/tax/core/mixed_scheme.hpp tests/mixed/test_mixed_scheme.cpp
git commit -m "feat(kernels): box-filtered Cauchy + recurrence stencils for MixedScheme

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: End-to-end oracle — scheme-generic kernels on `MixedScheme` vs isotropic `TE<Σorders>`

**Files:**
- Create: `tests/mixed/test_mixed_kernels.cpp`; Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `MixedScheme` (Tasks 2–3); the scheme-generic kernels; `tax::TE<Σorders, vars>` as the oracle.

**Oracle (the spec's primary correctness argument):** for any box monomial `α`, a `MixedScheme` result coefficient equals the same coefficient of an isotropic `TaylorExpansion<double, Σorder, vars>` computing the same expression — because the box ⊆ the order-`Σorder` simplex and monomial degrees add, so no out-of-box factor can contribute to an in-box output. Concretely: `mixed_out[S::flatOf(α)] == iso_result[flatIndex<vars>(α)]` for all `α` with `S::flatOf(α) != kNotInBox`.

- [ ] **Step 1: Write the oracle test**

Create `tests/mixed/test_mixed_kernels.cpp` — build a variable on `MixedScheme`, run several kernels (exp, sqrt, the Cauchy product, a self-product), and compare each box coefficient to the isotropic `TE<Σorder, vars>` computing the same thing. Example for `exp` on `S = MixedScheme<Group<1,4>, Group<1,3>>` (vars=2, Σorder=7):

```cpp
#include <gtest/gtest.h>

#include <tax/tax.hpp>
#include <tax/core/mixed_scheme.hpp>
#include <tax/core/index_scheme.hpp>

using tax::Group;
using tax::MixedScheme;

TEST( MixedKernels, ExpMatchesIsotropicSuperset )
{
    using S = MixedScheme< Group< 1, 4 >, Group< 1, 3 > >;   // vars=2, Sigma order=7
    constexpr int V = S::vars;
    constexpr int SUM = 7;
    using ISO = tax::TE< SUM, V >;                            // order-7 simplex superset

    // Seed the SAME polynomial p(x) = c0 + x_0 on both layouts (coord 0 is the
    // variable; coord 1 stays 0 here). exp(p) then depends only on coord 0, so it
    // is representable in both the box and the order-7 simplex.
    const double c0 = 0.3;

    // mixed input array: constant c0 at flat 0, coefficient 1 on the x_0 monomial.
    std::array< double, S::nCoeff > a{};
    a[0] = c0;
    tax::MultiIndex< V > e0{ 1, 0 };
    a[S::flatOf( e0 )] = 1.0;
    std::array< double, S::nCoeff > mexp{};
    tax::detail::kernels::seriesExp< double, S >( mexp, a );

    // isotropic input: the same polynomial c0 + x_0 in the order-7 simplex,
    // via the public variable factory (constant = c0, linear coeff on coord 0 = 1).
    typename ISO::Input p{ c0, 0.0 };
    ISO ax = ISO::template variable< 0 >( p );
    auto iexp = exp( ax );

    for ( std::size_t k = 0; k < S::nCoeff; ++k )
    {
        auto alpha = S::multiOf( k );
        EXPECT_NEAR( mexp[k], iexp[tax::flatIndex< V >( alpha )], 1e-12 );
    }
}
```

(Repeat the same pattern for `seriesSqrt` on `a` shifted away from zero, and for `tax::cauchyProduct`/`tax::cauchySelfProduct` of two box polynomials vs the isotropic product truncated to the box — each is a short block; write each as its own `TEST`.)

- [ ] **Step 2: Register + run, expect green**

Add `tax_add_test(test_mixed_kernels SOURCES mixed/test_mixed_kernels.cpp)`.
Run: `cmake --build build -j --target test_mixed_kernels && ctest --test-dir build -R test_mixed_kernels --output-on-failure`
Expected: PASS — the box coefficients match the isotropic superset.

- [ ] **Step 3: clang-format and commit**

```bash
cd /Users/andrea/Documents/Codes/tax
clang-format -i tests/mixed/test_mixed_kernels.cpp
git add tests/mixed/test_mixed_kernels.cpp tests/CMakeLists.txt
git commit -m "test(mixed): scheme-generic kernels on MixedScheme match isotropic superset oracle

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Milestone gate

**Files:** none (verification only).

- [ ] **Step 1: Clean build + full suite**

Run: `rm -rf build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build --output-on-failure`
Expected: ALL PASS — the original 36 (isotropic behavior-preserving) plus the new `test_mixed_scheme` and `test_mixed_kernels`.

- [ ] **Step 2: No isotropic drift / no churn**

Run: `git diff main -- include/tax/core/taylor_expansion.hpp include/tax/operators` → EMPTY (M2 does not touch the public isotropic type or operators).
Run: `git diff main --stat -- include/ | cat` and confirm only: `index_scheme.hpp`, `mixed_scheme.hpp`, `mixed_stencils.hpp`, `algebra.hpp`, `transcendental.hpp`, `trigonometric.hpp`. Confirm no `#    define`→`#define` churn in `cauchy.hpp`, no `concepts.hpp`/`tax.hpp` churn.

## Self-Review notes

- **Spec coverage (M2):** the `MixedScheme` machinery (box layout, `pos` map, graded ordering, stencils) and the scheme-generic completion of the Cauchy-based kernels — spec milestone 2. The core `MixedExpansion` type, named layer, and `la` are M3–M5.
- **Tests-as-contract:** Tasks 2–3 implement genuinely new index math; their unit tests (box count, dense/graded round-trip, out-of-box rejection, Cauchy-vs-brute-force) are the precise correctness contract, and Task 4's isotropic-superset oracle validates the kernels end-to-end on the box without needing the `MixedExpansion` type (M3).
- **Deferred:** the optional joint cap (`MixedScheme<…, JointCap>`); making `cauchyProduct`/`cauchySelfProduct` `constexpr` on schemes (M1 minor) so the mixed type gets a constant-evaluation path — fold into M3 where `MixedExpansion`'s `constexpr` surface is defined.

## Execution Handoff

After M2 lands, M3 (core `MixedExpansion<T, Groups…>` type wrapping a `MixedScheme`, with the full value/coeff/eval/deriv/integ/truncate + operator/math surface and the isotropic-superset oracle) is planned against the concrete `MixedScheme` interface.
