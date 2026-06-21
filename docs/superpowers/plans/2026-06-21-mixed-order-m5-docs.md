# Mixed-order Milestone 5 — documentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax. **After committing + reporting DONE, STOP — do not self-review or amend; the controller runs review.**

**Goal:** Document the mixed-order (anisotropic, per-axis-order) feature for users: a `docs/guide/mixed.md` guide page wired into the nav, plus light refresh of any docs made stale by the `TaylorExpansion`-over-`IndexScheme` unification.

**Architecture:** Docs-only milestone. Mirror the structure of the existing `docs/guide/batch.md` / `docs/guide/named.md`. No code changes.

**Tech Stack:** MkDocs (Material), Markdown. (`mkdocs` may not be installed in the env — the build check is optional, not a gate.)

**Milestone 5 of 5** from `docs/superpowers/specs/2026-06-21-mixed-order-named-expansions-design.md`. M1–M4 are complete on the branch (the unified `TaylorExpansion<T,Scheme>`, `MixedTE<Groups…>`, and the named per-axis-order layer `MixedTaylorExpansion` + `tax::mixed` factories + promotion/slice/deriv/integ/truncate + named `la`).

## Global Constraints

- Docs only — no `include/` or `tests/` changes. The full test suite is unaffected.
- Match the existing guide style/voice (see `docs/guide/batch.md`, `docs/guide/named.md`): short intro, code blocks using `#include <tax/tax.hpp>`, an aliases table, a "Notes & limits" section.
- Use the **real, current spellings** (verify against the headers): `tax::TE<N, M>`, `tax::MixedTE<tax::Group<Dim,Order>…>`, `tax::MixedTaylorExpansion<…>`, `tax::mixed::variable<"x", Order>(x0)`, `tax::mixed::variables<"p", Order, D>(arr)`, `slice<"name">()`, `deriv<"name">()`, `integ<"name">()`, `truncate<"name", N2>()`, `tax::gradient<"name">(f)` / `tax::jacobian<"name">(F)`.
- clang-format/code style N/A (Markdown). Don't reorder existing `mkdocs.yml` nav entries — only add the one Guide line.
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Branch `feature/mixed-order-expansions`.

## File Structure

- `docs/guide/mixed.md` — **new**: the mixed-order guide.
- `mkdocs.yml` — **modify**: add `- Mixed-Order Expansions: guide/mixed.md` to the `Guide:` nav, after the `Batch (SIMD) Coefficients` line.
- Possibly `docs/reference/core.md` / `docs/guide/storage.md` — **modify (light)**: refresh any stale `TaylorExpansion<T, N, M, Storage>` class-signature wording to note the scheme parameter (Task 2).

---

### Task 1: Write `docs/guide/mixed.md` + nav entry

**Files:** Create `docs/guide/mixed.md`; Modify `mkdocs.yml`.

- [ ] **Step 1: Write `docs/guide/mixed.md`**

Cover these sections (concrete content; mirror `batch.md`'s tone and length, ~80–130 lines):

1. **Intro / motivation.** Per-axis truncation orders on one expansion; the "box" keeps a monomial iff *each axis's* partial degree ≤ that axis's order, so `x@4, t@20` carries `x⁴·t²⁰` but never `x⁵` — no dense blow-up (contrast: a single-order `TE` would need order 24 and store all the `x⁵…x²⁴` zeros).
2. **The unified type.** `TaylorExpansion` is parameterized by an *index scheme*: `tax::TE<N, M>` is the classic single-order (total-degree) form; `tax::MixedTE<tax::Group<Dim,Order>…>` is the anonymous box form. Stress that **`MixedTE` *is* a `TaylorExpansion`** — it has the full math surface, `tax::la`, and Eigen interop, no separate type. Example:
   ```cpp
   #include <tax/tax.hpp>
   using ME = tax::MixedTE<tax::Group<1,4>, tax::Group<1,3>>;  // 2 axes, orders 4 and 3
   ME x = ME::variable<0>({0.3, -0.2});
   ME y = ME::variable<1>({0.3, -0.2});
   auto f = sin(x * y) + exp(x);   // full math surface
   auto g = f.gradient();          // tax::la works (Eigen vector of doubles)
   ```
3. **Named axes (the ergonomic layer).** `tax::mixed::variable<"x", Order>` / `variables<"p", Order, D>`; composition merges axis sets and takes **max order per shared axis**; canonical type (`x*p == p*x`). Example:
   ```cpp
   auto x = tax::mixed::variable<"x", 4>(1.0);        // axis "x" @ order 4
   auto p = tax::mixed::variables<"p", 20, 3>(arr3);  // 3-D axis "p" @ order 20
   auto f = sin(x) + x * p[0];                        // composes in {p, x}; no x^5… stored
   ```
4. **Operations.** `value`/`coeff`/`eval`/`deriv`/`integ`; named ops `deriv<"x">()`, `integ<"x">()`, `slice<"x">()` (project onto an axis subset), `truncate<"t", 2>()` (lower one axis's order — drops the high terms); named `la`: `tax::gradient<"x">(f)`, `tax::jacobian<"x">(F)`, `tax::hessian<"x">(f)`.
5. **Box vs simplex; the joint cap.** The named/`MixedTE` default is the full box (per-axis caps, no degree-summing across axes). The core `MixedScheme` supports an optional joint total-degree cap (off by default) for the prototype's coefficient-dropping speedup — mention it exists; the box is the default.
6. **Aliases table** (mirror batch.md's table): `tax::Group<Dim,Order>`, `tax::MixedTE<Groups…>`, `tax::OrderedAxis<Name,Dim,Order>`, `tax::MixedTaylorExpansion<T,Axes…>`, `tax::mixed::variable`/`variables`.
7. **Notes & limits.** Dense storage only (no sparse mixed); `Batch` coefficients are compatible in principle but not a shipped combo; a single multi-dimensional axis truncates by **total** degree *within* that axis's variables (per-axis caps are *across* axes). Cross-link `named.md` (joint-simplex named layer) and `batch.md`.

Verify every spelling against the headers (`include/tax/core/mixed_named.hpp`, `core/scheme/mixed.hpp`, `core/taylor_expansion.hpp` aliases, `la/mixed_named.hpp`) before writing — do not invent API.

- [ ] **Step 2: Add the nav entry**

In `mkdocs.yml`, under `Guide:`, immediately after `- Batch (SIMD) Coefficients: guide/batch.md`, add:
```yaml
      - Mixed-Order Expansions: guide/mixed.md
```
Run `git diff mkdocs.yml` — confirm only that one line was added (no reordering).

- [ ] **Step 3: Optional build check**

If `mkdocs` is installed: `mkdocs build --strict 2>&1 | tail -20` — expect no warnings about `guide/mixed.md` (unreferenced/missing). If `mkdocs` is not installed, SKIP (it is a docs-only check, not a code gate) and note that in the report.

- [ ] **Step 4: Commit**

```bash
cd /Users/andrea/Documents/Codes/tax
git add docs/guide/mixed.md mkdocs.yml
git commit -m "docs(guide): mixed-order (anisotropic) expansions guide

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Refresh docs made stale by the unification (light)

**Files:** Modify only the doc files that need it (likely `docs/reference/core.md`, possibly `docs/guide/storage.md` / `docs/concepts/*`).

- [ ] **Step 1: Find stale class-signature wording**

The unification changed the class from `TaylorExpansion<T, N, M, Storage>` to `TaylorExpansion<T, Scheme, Storage>` (with `TE<N,M>` preserved as the user alias). Grep the docs for descriptions that still present the old class signature or claim a single global order is intrinsic to the type:
```bash
grep -rn "TaylorExpansion< *T *, *N *, *M\|TaylorExpansion<T,N,M>\|template.*int N.*int M.*TaylorExpansion\|order N\b" docs/
```
Review hits in `docs/reference/`, `docs/guide/`, `docs/concepts/`, `docs/internals/`.

- [ ] **Step 2: Update only what is now inaccurate**

Where a doc describes the class template signature, note it is now `TaylorExpansion<T, Scheme, Storage>` with `IsotropicScheme<N,M>` the classic scheme (spelled `TE<N,M>`) and `MixedScheme<…>` the anisotropic one — keeping user-facing `TE<N,M>` examples unchanged (they still work). Do NOT rewrite docs that only use the `TE<N,M>`/`TEn`/`STE` aliases (those are still correct). Keep edits minimal and accurate; do not restructure docs unrelated to the change. If nothing is stale, report that and make no change.

- [ ] **Step 3: Commit (if any changes)**

```bash
git add docs/
git commit -m "docs: note TaylorExpansion is parameterized by an index scheme

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
(If Step 2 found nothing stale, skip the commit and say so in the report.)

## Self-Review notes

- **Spec coverage (M5):** the user-facing mixed-order guide + nav (spec milestone 5) and the unification doc-refresh.
- **Docs-only:** no code/test changes; verify spellings against the actual headers so examples compile conceptually.
- This is the **final milestone** of the mixed-order feature. After it lands, the whole feature (M1–M5) is complete and ready to finish the branch (PR/merge via superpowers:finishing-a-development-branch).
