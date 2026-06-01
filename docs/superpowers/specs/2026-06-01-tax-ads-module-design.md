# `tax::ads` — Automatic Domain Splitting module

**Date:** 2026-06-01
**Status:** approved (pending written-spec review)
**Author:** Claude (Opus 4.7, 1M ctx) with Andrea Pasquale

## Summary

A new header-only module `tax::ads` that implements Automatic Domain Splitting
(ADS, Wittig et al. 2015) and its Low-Order variant (LOADS, Losacco/Fossà/Armellin,
J. Guid. Control Dyn. 2024) on top of the existing `tax::ode` integration
infrastructure.

The module ships:

- **Geometry**: `Box<T, M>` axis-aligned subdomain.
- **Tree**: `AdsTree<Payload, M, T>` — leaf-only binary tree with parent/sibling
  links (no internal "split" nodes).
- **Splitting criteria**: `TruncationCriterion` (Wittig) and `NliCriterion`
  (LOADS), satisfying a `SplitCriterion` concept.
- **NLI math**: nonlinearity-index and Jacobian-variation-bound helpers.
- **Event interop**: `SplitTrigger` / `SplitAction` factories that plug a split
  request into `tax::ode::Event<Stepper>` — `tax::ode` itself is not modified.
- **Driver**: `AdsDriver<Stepper, Criterion>` — BFS loop that integrates each
  leaf, splits on event, and writes results back to the tree.
- **Merger**: bottom-up `merge` post-pass + `MergeStats`.

**Out of scope for this stage**: standalone function approximation (the
original prototype branch's `AdsRunner` use case), and any modification of
`tax::ode`. The Step-2 ODE integrators referenced in the user's original
request are *this* module — there is no separate Step-2.

## Background

### Existing infrastructure

The `tax::ode` module already exposes a full event hook:

- `Event<Stepper>` — type-erased `(Trigger, Action)` pair stored in the
  `Integrator`'s `EventList`.
- `Trigger : (StepperCtx) -> std::optional<T>` — returns the `τ ∈ [0, h_used]`
  at which it fires, or `nullopt`.
- `Action : (StepperCtx, T τ, EventStorage&) -> ControlFlow` — runs at the
  fired τ, returns `Continue` or `Terminate`. `Terminate` truncates the
  solution at the event time using `Stepper::eval_dense`.
- Trigger factories: `EveryStep`, `ZeroCrossing(g, dir)` (polynomial-Newton
  on TE-valued g; Brent on scalar samples).
- Action factories: `Continue`, `Terminate`, `Record(label)`, `Custom(fn)`.

This means the ADS split mechanism does *not* need a new hook in
`Integrator`: it can be expressed as `(SplitTrigger, SplitAction)` pair
composed via the existing `Event<Stepper>` interface.

The Stage-2b DA-vector state path also already works: `Integrator<Stepper, …>`
propagates `Eigen::Matrix<TEn<P,M>, D, 1>` through any RK or Taylor stepper
via `VectorOps<S>`.

### Prototype reference

Branch `claude/add-verner-integrators-vEgRF` carries a working ADS / LOADS
prototype. Its file layout (`ads/box.hpp`, `ads/ads_node.hpp`,
`ads/ads_tree.hpp`, `ads/ads_runner.hpp`, `ads/low_order_ads_runner.hpp`,
`ads/nonlinearity_index.hpp`, `ads/ads_merger.hpp`, plus parallel
`ode/ads_integrator.hpp` and `ode/low_order_ads_integrator.hpp`) duplicates
the step loop inside an `AdsRunner` rather than composing with
`tax::ode::Event`. The math (NLI bounds, jacobianVariationBound, merger
predicate) ports directly; the control flow does not.

## Goals & non-goals

### Goals

- Add `tax/ads/` as a self-contained module with no changes to `tax/ode/`.
- Use a single arena `std::vector<Leaf>` to back the tree — no internal
  node variant. Splits append two leaves, retire the parent. Merger walks
  sibling pairs.
- Template the driver on a `Stepper` policy so callers can choose Taylor,
  Verner78, Feagin12, etc. — same flexibility as `tax::ode::Integrator`.
- Provide two `SplitCriterion` types: `TruncationCriterion` (Wittig:
  degree-N coefficient mass) and `NliCriterion` (LOADS: row-bound NLI).
- Support arbitrary user events alongside the ADS split event: the driver
  forwards extra `Event<Stepper>` to the integrator unchanged.
- Detect splits only at *accepted-step boundaries* (Wittig's original
  trade-off).
- Implement a bottom-up `merge` pass with the same criteria.
- Tests under `tests/ads/` mirror the source structure.

### Non-goals

- Standalone function approximation (dropped; the prototype's `AdsRunner`
  use case is not needed for the user's workflow).
- Modifying `tax::ode` — no new hooks, no signature changes.
- Sub-step-resolution splitting (no integration to mid-step τ to find the
  split point — boundary-only).
- Dynamic-shape support: ADS stays static-only via `TaylorExpansionT<T, N, M>`
  with `N >= 0`, `M >= 0`, mirroring the ODE module.
- Parallel execution of independent leaves (single-threaded BFS).
- Visualization / IO helpers.
- Mid-propagation merging (LOADS-style on-the-fly). Merge is a post-pass.

## Design

### File layout

```
include/tax/ads/
├── box.hpp                   Box<T, M>
├── leaf.hpp                  Leaf<Payload, M, T>
├── tree.hpp                  AdsTree<Payload, M, T>
├── criteria.hpp              SplitCriterion concept + TruncationCriterion + NliCriterion
├── nonlinearity_index.hpp    NLI math (detail)
├── split_event.hpp           SplitRequest + SplitTrigger + SplitAction factories
├── da_state.hpp              create + split helpers
├── driver.hpp                AdsDriver<Stepper, Criterion>
└── merge.hpp                 merge() + MergeStats

include/tax/ads.hpp           umbrella
```

### `Box<T, M>`

```cpp
template <class T, int M>
struct Box
{
    static_assert(M >= 1, "Box dimension must be at least 1");
    std::array<T, M> center{};
    std::array<T, M> halfWidth{};

    constexpr Box() noexcept = default;
    constexpr Box(std::array<T, M> c, std::array<T, M> hw) noexcept;

    template <class CenterDerived, class HalfDerived>
    Box(const Eigen::MatrixBase<CenterDerived>& c,
        const Eigen::MatrixBase<HalfDerived>&   hw);

    [[nodiscard]] constexpr bool                contains(const std::array<T, M>& pt) const noexcept;
    [[nodiscard]] constexpr std::pair<Box, Box> split(int dim) const noexcept;
    [[nodiscard]] constexpr std::array<T, M>    denormalize(const std::array<T, M>& d) const noexcept;

    [[nodiscard]] tax::la::VecNT<M, T>          centerEigen()    const noexcept;
    [[nodiscard]] tax::la::VecNT<M, T>          halfWidthEigen() const noexcept;
    template <class Derived>
    [[nodiscard]] bool                          contains(const Eigen::MatrixBase<Derived>& pt) const;
    template <class Derived>
    [[nodiscard]] tax::la::VecNT<M, T>          denormalize(const Eigen::MatrixBase<Derived>& d) const;
};
```

- `contains` is inclusive at the boundary.
- `split(dim)` halves the `halfWidth` along `dim`; left center shifts down,
  right center shifts up.
- `denormalize(d)` maps `d ∈ [-1, 1]^M` to the box: `center + halfWidth ⊙ d`.

### `Leaf<Payload, M, T>`

```cpp
template <class Payload, int M, class T = double>
struct Leaf
{
    Box<T, M> box;
    Payload   payload;
    int       depth      = 0;
    bool      done       = false;
    bool      retired    = false;  // parent of an active pair; not a current leaf
    int       parentIdx  = -1;
    int       siblingIdx = -1;
    int       splitDim   = -1;     // dim that separated us from our sibling
    T         splitValue = 0;      // cut value in parent's coordinates
    T         tEntry     = 0;      // time at which this leaf entered the queue
};
```

A retired leaf is the *parent* of two active or done leaves; it stays in
the arena because the merger may collapse the pair back onto it.
`AdsTree::leaves()` and `leaf` skip retired leaves.

### `AdsTree<Payload, M, T>`

```cpp
template <class Payload, int M, class T = double>
class AdsTree
{
public:
    using LeafT = Leaf<Payload, M, T>;

    int  init(Box<T, M> box, Payload payload, T tEntry = T{0});

    // Work-queue interface.
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] int  front() const;     // peek
    int                popFront();        // pop active queue front

    // Split a leaf in place: parent becomes retired, two children appended.
    // Returns (leftIdx, rightIdx); both are pushed to the work queue.
    std::pair<int, int> split(int idx, int dim, T splitValue,
                              Payload leftPayload, Payload rightPayload,
                              T tEntry);

    // Mark a leaf done; moves it from active to done list.
    void finalize(int idx);

    // Merger: collapse a sibling pair back onto their parent.
    void merge(int leftIdx, int rightIdx, Payload mergedPayload);

    [[nodiscard]] const LeafT& leaf(int idx) const noexcept;
    [[nodiscard]] LeafT&       leaf(int idx)       noexcept;

    [[nodiscard]] std::span<const int> active() const noexcept;
    [[nodiscard]] std::span<const int> done()   const noexcept;
    [[nodiscard]] std::span<const int> roots()  const noexcept;

    // Linear scan over active + done; skips retired.
    [[nodiscard]] std::optional<int> leaf(const std::array<T, M>& pt) const;
    template <class Derived>
    [[nodiscard]] std::optional<int> leaf(const Eigen::MatrixBase<Derived>& pt) const;

private:
    std::vector<LeafT>     leaves_;
    std::vector<int>       activeList_;   // O(1) swap-pop on finalize/split
    std::vector<int>       doneList_;
    std::vector<int>       roots_;
    std::deque<int>        workQueue_;    // BFS order
};
```

**Invariants**

- Every entry in `activeList_` references a leaf with `done=false`,
  `retired=false`.
- `doneList_` entries have `done=true`, `retired=false`.
- Retired leaves are not in `activeList_` or `doneList_`. They are reachable
  only through `parentIdx` of their children.
- `roots_` contains all leaves with `parentIdx == -1`.
- `workQueue_` is a BFS view of the active list; entries are valid leaf
  indices with `done=false`, `retired=false` at enqueue time.

**Complexity**

- `init`, `split`, `finalize`: O(1) amortised (vector pushes + swap-pop).
- `leaf`: O(active+done) linear scan. Typical N = 10–1000; acceptable.
- `merge`: O(1).

### `criteria.hpp` — split decision

```cpp
template <class C, class State>
concept SplitCriterion = requires(C c, const State& x, int depth)
{
    { c.shouldSplit(x, depth) } -> std::convertible_to<bool>;
    { c.splitDim(x)            } -> std::convertible_to<int>;
};

struct TruncationCriterion
{
    double tol;
    int    maxDepth = 30;
    // shouldSplit: sum of degree-N coefficient absolute values > tol
    // splitDim:    coordinate contributing the most degree-N mass
};

struct NliCriterion
{
    double tol;
    int    maxDepth = 30;
    // shouldSplit: nonlinearityIndex(state) > tol
    // splitDim:    argmax over coordinate-wise row bounds
};
```

Both criteria take the current DA-valued state
`Eigen::Matrix<TEn<P,M>, D, 1>` and the depth of the leaf.

### `nonlinearity_index.hpp` — math

In `tax::ads::detail`:

```cpp
template <class T, int N, int M>
std::array<T, M> jacobianVariationBound(const TaylorExpansionT<T, N, M>& f);
// Per-coordinate bound on |∂f/∂x_i(x) - ∂f/∂x_i(0)| over [-1,1]^M.

template <class T, int N, int M>
T linRowBound(const TaylorExpansionT<T, N, M>& f);
// Sum of |linear-term coefficients| of f.

template <class T, int N, int M, int D>
double nonlinearityIndex(const Eigen::Matrix<TaylorExpansionT<T,N,M>, D, 1>& f);
// LOADS NLI: max_i (||∂f_i variation||_1 / ||∂f_i linear||_1).

template <class T, int N, int M, int D>
int nliSplitDim(const Eigen::Matrix<TaylorExpansionT<T,N,M>, D, 1>& f);
// Argmax over j of the total degree-N mass projected onto coordinate j.
```

These port from the prototype's `nonlinearity_index.hpp` with minor
re-spelling to match the `tax::la` aliases.

### `split_event.hpp` — interop with `tax::ode`

```cpp
template <class T>
struct SplitRequest
{
    bool fired = false;
    int  dim   = -1;
    T    t     = T{0};
};

// Trigger: at each accepted step, ask the criterion whether to split.
template <class Criterion>
auto SplitTrigger(Criterion crit, int depth);

// Action: record the split request in *out and Terminate the integration.
template <class Criterion, class T>
auto SplitAction(Criterion crit, SplitRequest<T>* out);
```

`SplitTrigger` returns `ctx.h_used` when the criterion says split, otherwise
`std::nullopt`. `SplitAction` writes `{fired=true, dim, t}` to `*out` and
returns `ControlFlow::Terminate`. The driver is responsible for keeping
`out` alive across the `integrate` call.

### `da_state.hpp` — DA construction & split

```cpp
template <class T, int N, int M, int D>
Eigen::Matrix<TaylorExpansionT<T,N,M>, D, 1>
create(const Box<T, M>& box, const Eigen::Matrix<T, D, 1>& center_state);
// Builds a DA-valued state x = center_state + identity DA over the box.

template <class T, int N, int M, int D>
std::pair<Eigen::Matrix<TaylorExpansionT<T,N,M>, D, 1>,
          Eigen::Matrix<TaylorExpansionT<T,N,M>, D, 1>>
split(const Eigen::Matrix<TaylorExpansionT<T,N,M>, D, 1>& state,
             const Box<T, M>& parent_box, int dim);
// Re-identifies the state's domain on the two halves of parent_box.
// Each half is a re-scaled, re-shifted polynomial in the new local coords.
```

Re-identification uses the standard DA composition: replace the dim-th
identity variable with `0.5 + 0.5 * x_new` (right half) or
`-0.5 + 0.5 * x_new` (left half).

### `driver.hpp` — the BFS driver

```cpp
template <class Stepper, class Criterion>
class AdsDriver
{
public:
    using State    = typename Stepper::State;        // Eigen::Matrix<TEn<P,M>, D, 1>
    using T        = typename Stepper::T;            // double
    using Cfg      = typename Stepper::Config;
    using ExtraEvt = std::vector<Event<Stepper>>;

    static constexpr int M = /* derive from State::Scalar */;
    using Tree    = AdsTree<State, M, T>;            // Payload = DA state
    using BoxT    = Box<T, M>;

    AdsDriver(Criterion crit, Cfg cfg, ExtraEvt extras = {});

    template <class F>
    Tree run(F&& rhs,
             const BoxT& ic_box,
             const Eigen::Matrix<T, /*D=*/Eigen::Dynamic, 1>& ic_center,
             T t0, T t1);

private:
    Criterion crit_;
    Cfg       cfg_;
    ExtraEvt  extras_;
};
```

Internal loop (paraphrased):

```cpp
auto state0 = create<P, M>(ic_box, ic_center);
Tree tree;
int root = tree.init(ic_box, state0, t0);

while (!tree.empty()) {
    int idx = tree.popFront();
    auto& l = tree.leaf(idx);

    SplitRequest<T> req;
    auto events = extras_;
    events.push_back(Event<Stepper>{
        SplitTrigger(crit_, l.depth),
        SplitAction(crit_, &req)
    });

    Integrator<Stepper, F> integ{rhs, cfg_, events};
    auto sol = integ.integrate(l.payload, l.tEntry, t1);

    if (req.fired) {
        auto [boxL, boxR] = l.box.split(req.dim);
        auto [xL,   xR  ] = split(sol.x.back(), l.box, req.dim);
        tree.split(idx, req.dim, l.box.center[req.dim],
                   std::move(xL), std::move(xR), req.t);
    } else {
        // store the final-time DA state in the leaf payload
        l.payload = std::move(sol.x.back());
        tree.finalize(idx);
    }
}
return tree;
```

Notes:

- The driver does *not* store the full `Solution` per leaf (it could in a
  future extension; for now the final-time DA flow map is what users want).
- `extras_` lets callers register their own events (terminal conditions,
  recordings) and they fire alongside the split event. If a user event
  fires *before* the split event, the integration terminates at the user
  event time and the leaf is marked done.
- Depth-limit handling: `Criterion::shouldSplit(x, depth)` returns false
  once `depth >= maxDepth`, so the split event simply never fires and the
  leaf integrates to `t1`.

### `merge.hpp` — bottom-up collapse

```cpp
struct MergeStats
{
    int passes   = 0;
    int merges   = 0;
    int rejected = 0;
};

template <class Stepper, class Criterion>
MergeStats merge(AdsTree<typename Stepper::State,
                         /*M derived*/,
                         typename Stepper::T>& tree,
                  Criterion crit);
```

Algorithm: in each pass, scan `tree.done()` for sibling pairs (both done,
shared `parentIdx`). For each pair, build a candidate merged state on the
parent's box (re-identify the two halves into the parent's coordinates and
take the polynomial that matches the higher-quality of the two). Check
`!crit.shouldSplit(merged, parent_depth)`; if so, `merge`. Repeat
until a pass makes no merges.

Notes:

- Merging is post-propagation only — the driver never merges during the
  forward pass.
- The "candidate merged state" construction is the same operation as
  `split`'s inverse: shift+scale the two half-domain polynomials
  back onto `[-1, 1]^M`, then verify they agree to within criterion
  tolerance.

### Umbrella

```cpp
// include/tax/ads.hpp
#pragma once
#include <tax/ads/box.hpp>
#include <tax/ads/leaf.hpp>
#include <tax/ads/tree.hpp>
#include <tax/ads/criteria.hpp>
#include <tax/ads/nonlinearity_index.hpp>
#include <tax/ads/split_event.hpp>
#include <tax/ads/da_state.hpp>
#include <tax/ads/driver.hpp>
#include <tax/ads/merge.hpp>
```

## Testing

`tests/ads/` mirrors source structure:

| File | Coverage |
|---|---|
| `test_box.cpp`               | constructors, `contains` (incl. boundary), `split` halves only the requested axis, `denormalize` round-trip, Eigen-overload parity |
| `test_leaf_tree.cpp`         | `init`/`split`/`finalize` invariants, `popFront` BFS order, retired-leaf accounting, `leaf` correctness, `merge` |
| `test_nonlinearity_index.cpp`| `jacobianVariationBound` matches analytical bound on small polys; `linRowBound`; `nliSplitDim` picks dominant axis on anisotropic input |
| `test_criteria.cpp`          | `TruncationCriterion` triggers on crafted leaves; respects `maxDepth`; `NliCriterion` triggers above tol; `splitDim` picks max-error axis |
| `test_split_event.cpp`       | `SplitTrigger` returns `h_used` on shouldSplit / nullopt otherwise; `SplitAction` writes correct request and terminates |
| `test_da_state.cpp`          | `create` identity property; `split` round-trip (composing both halves on their child boxes reproduces the original state on the parent box) |
| `test_driver.cpp`            | End-to-end ADS propagation on a stiff-ish 2D ODE; truncation criterion produces leaves whose evaluated flow maps match reference samples to within tol; extra user event coexists with split event |
| `test_merge.cpp`             | Start from a deliberately over-split tree, run `merge`, verify mergeable pairs collapse and others survive; `MergeStats.merges > 0`, `passes ≥ 1` |

Tolerances: `kTol = 1e-10` for coefficient comparisons; per-leaf tolerances
1e-5..1e-3 for end-to-end integration tests (limited by ADS criterion, not
integrator).

### Reference problem for `test_driver.cpp`

Harmonic-ish oscillator with mild nonlinearity:
`dx/dt = v`, `dv/dt = -x - 0.1*x^3`,
IC box centered at `(1, 0)` with half-width `(0.5, 0.5)`,
propagate to `t = 2π` with `P = 6`, `M = 2`, `D = 2`.
Sample 20 points in the IC box, integrate each numerically to `t = 2π` with
high-accuracy Verner89, compare against `tree.leaf(ic).payload` evaluated
at the displacement. Expect agreement to within `1e-4`.

## Dependencies

- `tax/la` — `Vec`, `VecNT`, `Eigen::MatrixBase` overloads.
- `tax/core` — `TaylorExpansionT`, `flatIndex`, `numMonomials`.
- `tax/utils` — multi-index enumeration where NLI needs it.
- `tax/ode` — `Event`, `Trigger`/`Action` infrastructure, `Integrator`,
  `Stepper` concepts, `EveryStep` / `ControlFlow` types.
- Std: `<array>`, `<vector>`, `<deque>`, `<optional>`, `<span>`, `<utility>`,
  `<cstddef>`.

## Migration / documentation

- `CLAUDE.md` currently documents an aspirational `tax/ads/` layout with
  `AdsRunner`/`AdsTree`/`integrateAds`. Update the doc to reflect the
  actual surface (`AdsDriver`, leaf-only `AdsTree`, no standalone runner).
- The prototype branch `claude/add-verner-integrators-vEgRF` is the
  reference for the NLI math and merge predicate. Port the math; do not
  port the control flow (it duplicates the step loop).
- No public `tax::` headers move; `tax/ads.hpp` is purely additive.

## Future work (explicitly deferred)

- **Mid-step split resolution.** Use `Stepper::eval_dense` and a bracketing
  search to locate the τ at which the criterion first exceeds tol within a
  step. Higher accuracy at cost of complexity. Defer until needed.
- **On-the-fly merging during propagation.** Periodically run `merge` at
  checkpoint times during the BFS, not only after completion.
- **Dynamic-shape support.** `TaylorExpansionT<T, Dynamic, Dynamic>` for
  ADS would require eager kernel paths throughout NLI. Static-only for now.
- **Parallel BFS.** Independent leaves can integrate in parallel; needs
  a thread-pool dependency. Single-threaded for now.
- **Function-approximation runner.** If a future workflow needs it, add a
  minimal `FunctionDriver` that reuses `AdsTree` + `Box` + criteria; not
  shipped in this stage.
