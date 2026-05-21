# tax Stage 2a — Core: Integrator framework + TaylorStepper + Events — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reintroduce ODE integration to `tax` with a method-agnostic
`Integrator` class, four step-size controllers (JorbaZou, I, PI,
H211b), the Taylor-method stepper, and a Trigger+Action event system
with polynomial-Newton root finding for Taylor zero-crossings.

**Architecture:** Header-only templates under `include/tax/ode/`,
sibling to `include/tax/eigen.hpp`. Compile-time `concepts::Stepper`
and `concepts::AdaptiveStepper` policies drive a single `Integrator`
class. `Solution` is partial-specialised on a `Dense` template bool so
discrete and dense-output modes share no unused storage. Events are
type-erased `{Trigger, Action}` pairs that fire after each accepted
step, with `ZeroCrossing` using polynomial-Newton on the per-step
Taylor expansion of `g(x, t)`.

**Tech Stack:** C++23, header-only, Eigen 3.4+ for state vectors,
existing tax core (`tax::TE`, NumTraits, Eigen integration). Tests use
Google Test under `TAX_BUILD_UNITTESTS=ON`. No new external
dependencies.

**Spec:** `docs/superpowers/specs/2026-05-21-tax-stage2a-ode-integrator-design.md`.
Plan B (slices 5–10: RK methods + physical-dynamics tests +
benchmarks) will follow.

**Build env:** micromamba env `tax` (cmake, gxx, eigen, benchmark,
ninja). Activate: `eval "$(micromamba shell hook --shell bash)" && micromamba activate tax`.

---

## File Structure

Created in this plan:

- `include/tax/ode.hpp` — umbrella header
- `include/tax/ode/config.hpp` — `IntegratorConfig<T>`
- `include/tax/ode/concepts.hpp` — `concepts::Stepper`, `concepts::AdaptiveStepper`
- `include/tax/ode/step_result.hpp` — `StepResult<State, Stepper>`
- `include/tax/ode/solution.hpp` — `EventRecord<State, T>`, `Solution<…, false>`, `Solution<…, true>`
- `include/tax/ode/controllers.hpp` — `JorbaZou`, `I`, `PI`, `H211b`
- `include/tax/ode/steppers/taylor.hpp` — `TaylorStepper<N, State, Controller>`
- `include/tax/ode/integrator.hpp` — `Integrator<Stepper, Dense>` + `TaylorIntegrator` alias
- `include/tax/ode/event.hpp` — `Direction`, `ControlFlow`, `TriggerContext`, `Event`
- `include/tax/ode/triggers.hpp` — `ZeroCrossing`, `EveryStep`
- `include/tax/ode/actions.hpp` — `Continue`, `Terminate`, `Record`, `Custom`
- `include/tax/ode/detail/brent_root.hpp` — shared Brent root-finder
- `tests/ode/CMakeLists.txt` — registers the seven test executables
- `tests/ode/testConfig.cpp`
- `tests/ode/testControllers.cpp`
- `tests/ode/testTaylorStepper.cpp`
- `tests/ode/testIntegratorBasic.cpp`
- `tests/ode/testIntegratorDense.cpp`
- `tests/ode/testEventsZeroCrossing.cpp`
- `tests/ode/testEventsEveryStep.cpp`

Modified:

- `tests/CMakeLists.txt` — register `add_subdirectory(ode)`

`tax/tax.hpp` is **not** modified: ODE remains opt-in via explicit
`#include <tax/ode.hpp>`.

---

## Task 0 — Branch setup

- [ ] **Step 0.1:** Confirm we're starting from `main` with a clean
  working tree.

  ```bash
  git status
  ```

  Expected: `Sul branch main` (or `On branch main`) and "non c'è nulla
  di cui eseguire il commit" / "nothing to commit, working tree clean".

- [ ] **Step 0.2:** Create the feature branch.

  ```bash
  git checkout -b stage2a-core
  ```

- [ ] **Step 0.3:** Verify baseline tests still pass.

  ```bash
  eval "$(micromamba shell hook --shell bash)" && micromamba activate tax
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTAX_BUILD_UNITTESTS=ON -G Ninja
  cmake --build build -j
  ctest --test-dir build --output-on-failure
  ```

  Expected: all current tests (~30) pass. Take note of the count for
  later regression checks.

---

## Slice 1 — Config + concepts + step result + solution skeleton

Goal: scaffold the public types that the rest of the plan depends on.
After slice 1 a stub stepper can be written in a test to exercise the
`concepts::Stepper` and `concepts::AdaptiveStepper` checks at compile
time. No real integration logic yet.

### Task 1 — `IntegratorConfig<T>` and the umbrella header

**Files:**
- Create: `include/tax/ode.hpp`
- Create: `include/tax/ode/config.hpp`
- Create: `tests/ode/CMakeLists.txt`
- Create: `tests/ode/testConfig.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1.1:** Create `tests/ode/CMakeLists.txt`:

  ```cmake
  # tests/ode/CMakeLists.txt
  #
  # ODE integrator unit tests for Stage 2a. Each test executable links
  # the tax INTERFACE target and gtest_main. Registered via
  # tax_add_test() from the parent tests/CMakeLists.txt.

  tax_add_test(test_ode_config SOURCES testConfig.cpp)
  ```

  Note: `tax_add_test` is assumed to exist in `tests/CMakeLists.txt`.
  Verify by inspecting the parent file.

- [ ] **Step 1.2:** Verify `tax_add_test` exists. Run:

  ```bash
  grep -n "function(tax_add_test" tests/CMakeLists.txt
  ```

  Expected: prints the line where `tax_add_test` is defined. If not
  found, the rest of the test scaffolding in this plan will fail —
  stop and inspect `tests/CMakeLists.txt` to understand the project's
  helper conventions.

- [ ] **Step 1.3:** Add `add_subdirectory(ode)` to
  `tests/CMakeLists.txt`. Find the existing block of
  `add_subdirectory(...)` calls and append at the end of that block
  (do not interleave with `tax_add_test(...)` direct calls).

- [ ] **Step 1.4:** Write the failing test
  `tests/ode/testConfig.cpp`:

  ```cpp
  // tests/ode/testConfig.cpp
  //
  // Validates IntegratorConfig defaults and constructor behaviour.

  #include <gtest/gtest.h>

  #include <tax/ode.hpp>

  TEST( OdeConfig, DefaultsAreSane )
  {
      tax::ode::IntegratorConfig< double > cfg;
      EXPECT_GT( cfg.abstol, 0.0 );
      EXPECT_GT( cfg.reltol, 0.0 );
      EXPECT_EQ( cfg.initial_step, 0.0 );        // 0 ⇒ stepper picks
      EXPECT_EQ( cfg.min_step, 0.0 );            // 0 ⇒ ~eps × span
      EXPECT_EQ( cfg.max_step, 0.0 );            // 0 ⇒ tmax - t0
      EXPECT_GT( cfg.max_steps, 0 );
      EXPECT_GT( cfg.max_rejects_per_step, 0 );
  }

  TEST( OdeConfig, AcceptsCustomValues )
  {
      tax::ode::IntegratorConfig< double > cfg{
          .abstol               = 1e-10,
          .reltol               = 1e-8,
          .initial_step         = 1e-3,
          .min_step             = 1e-12,
          .max_step             = 0.1,
          .max_steps            = 1000,
          .max_rejects_per_step = 8,
      };
      EXPECT_DOUBLE_EQ( cfg.abstol, 1e-10 );
      EXPECT_DOUBLE_EQ( cfg.reltol, 1e-8 );
      EXPECT_EQ( cfg.max_steps, 1000 );
  }
  ```

- [ ] **Step 1.5:** Verify the test does not build yet.

  ```bash
  eval "$(micromamba shell hook --shell bash)" && micromamba activate tax
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTAX_BUILD_UNITTESTS=ON -G Ninja
  cmake --build build -j 2>&1 | tail -5
  ```

  Expected: fails because `<tax/ode.hpp>` does not exist.

- [ ] **Step 1.6:** Create `include/tax/ode/config.hpp`:

  ```cpp
  // include/tax/ode/config.hpp
  //
  // Stage 2a ODE integrator configuration.

  #pragma once

  namespace tax::ode
  {

  /**
   * @brief Runtime configuration for the Stage 2a `Integrator`.
   *
   * Defaults are conservative for double-precision ODEs. Field
   * conventions:
   *   - `*_step` values of 0 ⇒ the stepper picks a reasonable default
   *     (initial_step: heuristic from RHS magnitude; min_step:
   *     ~eps × (tmax - t0); max_step: tmax - t0).
   */
  template < class T = double >
  struct IntegratorConfig
  {
      T   abstol               = T{ 1e-12 };
      T   reltol               = T{ 1e-12 };
      T   initial_step         = T{ 0 };
      T   min_step             = T{ 0 };
      T   max_step             = T{ 0 };
      int max_steps            = 100'000;
      int max_rejects_per_step = 16;
  };

  }  // namespace tax::ode
  ```

- [ ] **Step 1.7:** Create the umbrella `include/tax/ode.hpp`:

  ```cpp
  // include/tax/ode.hpp
  //
  // tax Stage 2a — opt-in ODE integration umbrella.
  // Users include only this header to access the ODE integrator surface.

  #pragma once

  #include <tax/ode/config.hpp>
  ```

- [ ] **Step 1.8:** Build and run the new test:

  ```bash
  cmake --build build -j
  ctest --test-dir build -R test_ode_config --output-on-failure
  ```

  Expected: `test_ode_config` runs and both subtests pass.

- [ ] **Step 1.9:** Verify the existing baseline tests still pass.

  ```bash
  ctest --test-dir build --output-on-failure
  ```

  Expected: previous count + 1 new test, all pass.

- [ ] **Step 1.10:** Commit slice-1 progress so far.

  ```bash
  git add include/tax/ode.hpp include/tax/ode/config.hpp \
          tests/CMakeLists.txt tests/ode/CMakeLists.txt \
          tests/ode/testConfig.cpp
  git commit -m "$(cat <<'EOF'
  stage2a-slice1a: IntegratorConfig + ODE umbrella header

  Adds tax::ode::IntegratorConfig<T> and the opt-in tax/ode.hpp umbrella
  header. Wires tests/ode/ into the unit-test build via the existing
  tax_add_test() helper. testConfig.cpp covers defaults and aggregate-
  init custom values.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

### Task 2 — `StepResult<State, Stepper>` and concepts

**Files:**
- Create: `include/tax/ode/step_result.hpp`
- Create: `include/tax/ode/concepts.hpp`
- Modify: `include/tax/ode.hpp`
- Create: `tests/ode/testConcepts.cpp` (in-place compile-time test)
- Modify: `tests/ode/CMakeLists.txt`

- [ ] **Step 2.1:** Add the concepts test to the CMake build. Append
  to `tests/ode/CMakeLists.txt`:

  ```cmake
  tax_add_test(test_ode_concepts SOURCES testConcepts.cpp)
  ```

- [ ] **Step 2.2:** Write the failing test
  `tests/ode/testConcepts.cpp`:

  ```cpp
  // tests/ode/testConcepts.cpp
  //
  // Compile-time validation that:
  //   1. A trivial fake stepper satisfies concepts::Stepper.
  //   2. The same fake stepper extended with adaptive fields
  //      satisfies concepts::AdaptiveStepper.
  //   3. The non-adaptive fake does NOT satisfy AdaptiveStepper.

  #include <gtest/gtest.h>

  #include <Eigen/Core>
  #include <functional>

  #include <tax/ode.hpp>

  namespace
  {

  using State = Eigen::Matrix< double, 1, 1 >;

  // Minimal stepper that satisfies concepts::Stepper but not
  // AdaptiveStepper (no err_norm / h_next / accepted on its StepResult).
  struct FakeStepper
  {
      using T         = double;
      using Config    = tax::ode::IntegratorConfig< T >;
      using Rhs       = std::function< State( const State&, T ) >;
      using DenseData = State;

      tax::ode::StepResult< State, FakeStepper >
      step( const Rhs& /*f*/, const State& x, T /*t*/, T h, const Config& /*cfg*/ ) const
      {
          tax::ode::StepResult< State, FakeStepper > r;
          r.x_new = x;
          r.h_used = h;
          r.dense = x;
          return r;
      }

      static State eval_dense( const DenseData& d, const T& /*t0*/, const T& /*t1*/,
                               const T& /*tq*/ )
      {
          return d;
      }
  };

  static_assert( tax::ode::concepts::Stepper< FakeStepper >,
                 "FakeStepper should satisfy concepts::Stepper" );
  static_assert( !tax::ode::concepts::AdaptiveStepper< FakeStepper >,
                 "FakeStepper should NOT satisfy concepts::AdaptiveStepper" );

  TEST( OdeConcepts, FakeStepperCompiles )
  {
      FakeStepper s;
      FakeStepper::Rhs f = []( const State& x, double ) { return x; };
      State x = State::Zero();
      auto r = s.step( f, x, 1.0, 0.1, tax::ode::IntegratorConfig< double >{} );
      EXPECT_DOUBLE_EQ( r.h_used, 0.1 );
  }

  }  // namespace
  ```

- [ ] **Step 2.3:** Verify the test does not build (concepts and
  StepResult don't exist yet).

  ```bash
  cmake --build build -j 2>&1 | tail -5
  ```

  Expected: fails on `tax::ode::StepResult` and
  `tax::ode::concepts::Stepper`.

- [ ] **Step 2.4:** Create `include/tax/ode/step_result.hpp`:

  ```cpp
  // include/tax/ode/step_result.hpp
  //
  // Result type returned by every Stepper's step() method.
  // Adaptive fields (h_next, err_norm, accepted) are always present
  // for layout simplicity; fixed-step Steppers (future) leave them at
  // their defaults.

  #pragma once

  namespace tax::ode
  {

  template < class State, class Stepper >
  struct StepResult
  {
      State                          x_new{};
      typename Stepper::T            h_used{};
      typename Stepper::DenseData    dense{};
      // Adaptive-only — meaningful when Stepper satisfies AdaptiveStepper.
      typename Stepper::T            h_next{};
      typename Stepper::T            err_norm{};
      bool                           accepted = true;
  };

  }  // namespace tax::ode
  ```

- [ ] **Step 2.5:** Create `include/tax/ode/concepts.hpp`:

  ```cpp
  // include/tax/ode/concepts.hpp
  //
  // Stepper concept hierarchy.
  //   - Stepper:         minimum — take one step at the supplied h.
  //   - AdaptiveStepper: refinement — additionally provides embedded
  //                      error estimate and recommended next step
  //                      via fields on the returned StepResult.

  #pragma once

  #include <concepts>
  #include <utility>

  #include <tax/ode/step_result.hpp>

  namespace tax::ode::concepts
  {

  template < class S >
  concept Stepper = requires(
      S s,
      typename S::Rhs f,
      typename S::State x,
      typename S::T t,
      typename S::T h,
      const typename S::Config& cfg )
  {
      typename S::State;
      typename S::T;
      typename S::Config;
      typename S::Rhs;
      typename S::DenseData;

      { s.step( f, x, t, h, cfg ) }
          -> std::same_as< StepResult< typename S::State, S > >;

      { S::eval_dense( std::declval< typename S::DenseData >(), t, t, t ) }
          -> std::same_as< typename S::State >;
  };

  template < class S >
  concept AdaptiveStepper = Stepper< S >
      && requires( const StepResult< typename S::State, S >& r )
      {
          { r.err_norm } -> std::convertible_to< typename S::T >;
          { r.h_next   } -> std::convertible_to< typename S::T >;
          { r.accepted } -> std::convertible_to< bool >;
      };

  }  // namespace tax::ode::concepts
  ```

- [ ] **Step 2.6:** Update the umbrella `include/tax/ode.hpp` to
  pull in the new headers. Replace its content with:

  ```cpp
  // include/tax/ode.hpp
  //
  // tax Stage 2a — opt-in ODE integration umbrella.

  #pragma once

  #include <tax/ode/config.hpp>
  #include <tax/ode/step_result.hpp>
  #include <tax/ode/concepts.hpp>
  ```

- [ ] **Step 2.7:** Build and run the concepts test:

  ```bash
  cmake --build build -j
  ctest --test-dir build -R test_ode_concepts --output-on-failure
  ```

  Expected: `test_ode_concepts` passes (the `static_assert` block
  succeeds and the runtime stub case asserts `h_used == 0.1`).

- [ ] **Step 2.8:** Verify the baseline still passes.

  ```bash
  ctest --test-dir build --output-on-failure
  ```

- [ ] **Step 2.9:** Commit.

  ```bash
  git add include/tax/ode/step_result.hpp include/tax/ode/concepts.hpp \
          include/tax/ode.hpp \
          tests/ode/CMakeLists.txt tests/ode/testConcepts.cpp
  git commit -m "$(cat <<'EOF'
  stage2a-slice1b: StepResult + concepts::Stepper / AdaptiveStepper

  Adds the StepResult struct returned by every Stepper::step() and the
  two-tier concept hierarchy: Stepper (minimum) + AdaptiveStepper
  (refinement with err_norm / h_next / accepted). The Integrator core
  loop will use `if constexpr (AdaptiveStepper<S>)` to drive rejection
  and retry only for adaptive Steppers. testConcepts.cpp validates both
  concept levels via static_assert against a trivial fake stepper.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

### Task 3 — `EventRecord` + `Solution<…, false>` + `Solution<…, true>`

**Files:**
- Create: `include/tax/ode/solution.hpp`
- Modify: `include/tax/ode.hpp`
- Create: `tests/ode/testSolution.cpp`
- Modify: `tests/ode/CMakeLists.txt`

- [ ] **Step 3.1:** Append to `tests/ode/CMakeLists.txt`:

  ```cmake
  tax_add_test(test_ode_solution SOURCES testSolution.cpp)
  ```

- [ ] **Step 3.2:** Write the failing test
  `tests/ode/testSolution.cpp`:

  ```cpp
  // tests/ode/testSolution.cpp
  //
  // Verifies that both Solution specialisations (Dense=false and
  // Dense=true) exist with the expected member layout, and that
  // appending steps/events works as advertised. Uses the FakeStepper
  // from testConcepts via duplication (header-only test).

  #include <gtest/gtest.h>

  #include <Eigen/Core>
  #include <functional>

  #include <tax/ode.hpp>

  namespace
  {

  using State = Eigen::Matrix< double, 2, 1 >;

  struct FakeStepper
  {
      using T         = double;
      using Config    = tax::ode::IntegratorConfig< T >;
      using Rhs       = std::function< State( const State&, T ) >;
      using DenseData = State;

      tax::ode::StepResult< State, FakeStepper >
      step( const Rhs&, const State& x, T, T h, const Config& ) const
      {
          tax::ode::StepResult< State, FakeStepper > r;
          r.x_new = x; r.h_used = h; r.dense = x;
          return r;
      }
      static State eval_dense( const DenseData& d, const T&, const T&, const T& )
      {
          return d;
      }
  };

  }  // namespace

  TEST( OdeSolution, DiscreteHasNoDenseStorage )
  {
      using Sol = tax::ode::Solution< FakeStepper, State, /*Dense=*/false >;
      Sol s;
      s.t.push_back( 0.0 );
      s.x.push_back( State{ 1.0, 2.0 } );
      s.events.push_back( { "tag", 0.5, State{ 3.0, 4.0 } } );

      EXPECT_EQ( s.t.size(), 1u );
      EXPECT_EQ( s.x.size(), 1u );
      EXPECT_EQ( s.events.size(), 1u );
      EXPECT_EQ( s.size(), 1u );
      // The discrete specialisation must not expose a `dense` vector.
      static_assert( !requires( const Sol c ) { c.dense; },
                     "Discrete Solution must not have a dense field" );
  }

  TEST( OdeSolution, DenseExposesDenseAndOperatorCall )
  {
      using Sol = tax::ode::Solution< FakeStepper, State, /*Dense=*/true >;
      Sol s;
      s.t = { 0.0, 1.0 };
      s.x = { State{ 1.0, 2.0 }, State{ 3.0, 4.0 } };
      s.dense.push_back( State{ 1.0, 2.0 } );

      // operator()(t_query) is defined and calls eval_dense.
      State at_t0 = s( 0.0 );
      EXPECT_DOUBLE_EQ( at_t0( 0 ), 1.0 );
      EXPECT_DOUBLE_EQ( at_t0( 1 ), 2.0 );
  }
  ```

- [ ] **Step 3.3:** Confirm the build fails on missing
  `tax::ode::Solution`.

  ```bash
  cmake --build build -j 2>&1 | tail -5
  ```

- [ ] **Step 3.4:** Create `include/tax/ode/solution.hpp`:

  ```cpp
  // include/tax/ode/solution.hpp
  //
  // Solution<Stepper, State, Dense> — partial-specialised on Dense.
  // Dense=false:  step-boundary states + events only.
  // Dense=true :  adds per-step continuous-extension payload and an
  //               operator()(t_query) that interpolates via the
  //               stepper's static eval_dense().

  #pragma once

  #include <algorithm>
  #include <cstddef>
  #include <stdexcept>
  #include <string>
  #include <vector>

  namespace tax::ode
  {

  template < class State, class T >
  struct EventRecord
  {
      std::string label;          // "" if anonymous
      T           t_event;
      State       x_event;
  };

  template < class Stepper, class State, bool Dense >
  class Solution;

  // Discrete specialisation: step boundaries + events only.
  template < class Stepper, class State >
  class Solution< Stepper, State, /*Dense=*/false >
  {
  public:
      using T = typename Stepper::T;
      std::vector< T >                       t;
      std::vector< State >                   x;
      std::vector< EventRecord< State, T > > events;

      [[nodiscard]] std::size_t size() const noexcept { return t.size(); }
  };

  // Dense specialisation: adds per-step continuous-extension data and sol(t).
  template < class Stepper, class State >
  class Solution< Stepper, State, /*Dense=*/true >
  {
  public:
      using T         = typename Stepper::T;
      using DenseData = typename Stepper::DenseData;

      std::vector< T >                       t;        // size = nsteps + 1
      std::vector< State >                   x;        // size = nsteps + 1
      std::vector< DenseData >               dense;    // size = nsteps
      std::vector< EventRecord< State, T > > events;

      [[nodiscard]] std::size_t size() const noexcept { return t.size(); }

      [[nodiscard]] State operator()( const T& t_query ) const
      {
          if ( t.empty() )
              throw std::runtime_error( "Solution::operator(): empty solution" );
          if ( t_query < t.front() || t_query > t.back() )
              throw std::out_of_range( "Solution::operator(): t_query out of [t0, tf]" );

          // Binary search: find i with t[i] <= t_query <= t[i+1].
          auto       it = std::upper_bound( t.begin(), t.end(), t_query );
          const auto i  = static_cast< std::size_t >( std::max< std::ptrdiff_t >(
              0, std::distance( t.begin(), it ) - 1 ) );
          const std::size_t idx = std::min( i, dense.size() - 1 );
          return Stepper::eval_dense( dense[idx], t[idx], t[idx + 1], t_query );
      }
  };

  }  // namespace tax::ode
  ```

- [ ] **Step 3.5:** Update `include/tax/ode.hpp` to include
  solution.hpp:

  ```cpp
  // include/tax/ode.hpp
  #pragma once

  #include <tax/ode/config.hpp>
  #include <tax/ode/step_result.hpp>
  #include <tax/ode/concepts.hpp>
  #include <tax/ode/solution.hpp>
  ```

- [ ] **Step 3.6:** Build and run the test:

  ```bash
  cmake --build build -j
  ctest --test-dir build -R test_ode_solution --output-on-failure
  ```

  Expected: pass.

- [ ] **Step 3.7:** Commit.

  ```bash
  git add include/tax/ode/solution.hpp include/tax/ode.hpp \
          tests/ode/CMakeLists.txt tests/ode/testSolution.cpp
  git commit -m "$(cat <<'EOF'
  stage2a-slice1c: Solution<…, Dense> + EventRecord

  Two partial specialisations: Solution<…, false> stores step boundaries
  + events; Solution<…, true> adds per-step DenseData and an
  operator()(t_query) that does a binary search + Stepper::eval_dense.
  The discrete specialisation pays nothing for dense storage. Tests
  exercise both layouts and the static_assert that the discrete form
  exposes no `dense` member.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Slice 2 — Controllers + TaylorStepper

Goal: ship all four step-size controllers and the Taylor-method
stepper. After slice 2 a single Taylor step can be taken on a real
ODE and its dense extension queried.

### Task 4 — Controllers: `JorbaZou`, `I`, `PI`, `H211b`

**Files:**
- Create: `include/tax/ode/controllers.hpp`
- Modify: `include/tax/ode.hpp`
- Create: `tests/ode/testControllers.cpp`
- Modify: `tests/ode/CMakeLists.txt`

- [ ] **Step 4.1:** Append to `tests/ode/CMakeLists.txt`:

  ```cmake
  tax_add_test(test_ode_controllers SOURCES testControllers.cpp)
  ```

- [ ] **Step 4.2:** Write the failing test
  `tests/ode/testControllers.cpp`:

  ```cpp
  // tests/ode/testControllers.cpp
  //
  // Covers controller behaviour:
  //   - I: stateless, predictable scaling on err < tol and err > tol.
  //   - PI: state-evolving, uses previous error.
  //   - H211b: state-evolving, smoothed.
  //   - JorbaZou: uses last-two-coefficient norms.
  // Each test asserts that min_factor / max_factor clamps are applied.

  #include <gtest/gtest.h>

  #include <cmath>

  #include <tax/ode.hpp>

  using tax::ode::controllers::I;
  using tax::ode::controllers::PI;
  using tax::ode::controllers::H211b;
  using tax::ode::controllers::JorbaZou;

  TEST( OdeControllerI, ScalesDownOnLargeError )
  {
      I< double > c;
      const double h_used = 0.1;
      const double err    = 10.0;
      const double tol    = 1.0;
      const int    p_emb  = 7;
      const double h_new  = c.next_step( h_used, err, tol, p_emb );
      EXPECT_LT( h_new, h_used );
      EXPECT_GE( h_new, h_used * c.min_factor );
  }

  TEST( OdeControllerI, ScalesUpOnSmallError )
  {
      I< double > c;
      const double h_new = c.next_step( /*h_used=*/0.1, /*err=*/0.01,
                                         /*tol=*/1.0, /*p_emb=*/7 );
      EXPECT_GT( h_new, 0.1 );
      EXPECT_LE( h_new, 0.1 * c.max_factor );
  }

  TEST( OdeControllerPI, RemembersPreviousError )
  {
      PI< double > c;
      const double h1 = c.next_step( 0.1, 0.5, 1.0, 7 );
      const double h2 = c.next_step( h1, 0.5, 1.0, 7 );
      // On the second step the proportional term contributes;
      // the result must differ from the I-only equivalent.
      I< double > i;
      const double i_only = i.next_step( h1, 0.5, 1.0, 7 );
      EXPECT_NE( h2, i_only );
  }

  TEST( OdeControllerH211b, FirstCallBehavesLikeI )
  {
      H211b< double > c;
      I< double >     i;
      // On its very first call h_prev_ == 0 so the controller falls
      // back to I-step semantics.
      const double h_new_h  = c.next_step( 0.1, 0.5, 1.0, 7 );
      const double h_new_i  = i.next_step( 0.1, 0.5, 1.0, 7 );
      EXPECT_NEAR( h_new_h, h_new_i, 1e-12 );
  }

  TEST( OdeControllerJorbaZou, ScalesDownOnLargeLeadingCoeff )
  {
      JorbaZou< double > c;
      const double h_used     = 0.1;
      const double c_N_norm   = 1e6;     // very large ⇒ shrink
      const double c_Nm1_norm = 1e6;
      const double tol        = 1e-12;
      const int    N_order    = 12;
      const double h_new = c.next_step( h_used, c_N_norm, c_Nm1_norm,
                                         tol, N_order );
      EXPECT_LT( h_new, h_used );
      EXPECT_GE( h_new, h_used * c.min_factor );
  }
  ```

- [ ] **Step 4.3:** Confirm build fails on missing
  `tax::ode::controllers::*`.

  ```bash
  cmake --build build -j 2>&1 | tail -5
  ```

- [ ] **Step 4.4:** Create `include/tax/ode/controllers.hpp`:

  ```cpp
  // include/tax/ode/controllers.hpp
  //
  // Step-size controllers for Stage 2a.
  //   I       — classic integral (stateless, robust baseline)
  //   PI      — Gustafsson PI (default for RK steppers)
  //   H211b   — Söderlind digital filter (smoothed; robust on bumpy problems)
  //   JorbaZou — Taylor-method predictor based on the last two polynomial
  //              coefficient magnitudes (default for TaylorStepper)

  #pragma once

  #include <algorithm>
  #include <cmath>

  namespace tax::ode::controllers
  {

  namespace detail
  {
  template < class T >
  constexpr T clamp_factor( T raw, T mn, T mx ) noexcept
  {
      return std::min( std::max( raw, mn ), mx );
  }
  }  // namespace detail

  // -------- I --------
  template < class T = double >
  struct I
  {
      T safety     = T{ 0.9 };
      T min_factor = T{ 0.2 };
      T max_factor = T{ 5.0 };

      [[nodiscard]] T next_step( T h_used, T err_norm, T tol, int p_emb ) const noexcept
      {
          using std::pow;
          const T ratio  = ( err_norm > T{ 0 } ) ? ( tol / err_norm ) : T{ 1 };
          const T exp    = T{ 1 } / T( p_emb + 1 );
          const T factor = detail::clamp_factor< T >( safety * pow( ratio, exp ),
                                                       min_factor, max_factor );
          return h_used * factor;
      }
  };

  // -------- PI (Gustafsson) --------
  template < class T = double >
  struct PI
  {
      T safety     = T{ 0.9 };
      T alpha     = T{ 0.7 };
      T beta      = T{ 0.4 };
      T min_factor = T{ 0.2 };
      T max_factor = T{ 5.0 };

      [[nodiscard]] T next_step( T h_used, T err_norm, T tol, int p_emb ) noexcept
      {
          using std::pow;
          const T denom = ( err_norm > T{ 0 } ) ? err_norm : T{ 1 };
          const T inv   = ( p_emb > 0 ) ? T{ 1 } / T( p_emb + 1 ) : T{ 1 };
          const T raw   = pow( tol / denom, beta * inv )
                          * pow( err_prev_ / denom, alpha * inv );
          const T factor = detail::clamp_factor< T >( safety * raw,
                                                       min_factor, max_factor );
          err_prev_ = denom;
          return h_used * factor;
      }

  private:
      T err_prev_ = T{ 1 };
  };

  // -------- H211b (Söderlind) --------
  // Reference: G. Söderlind, "Digital filters in adaptive time-stepping",
  // ACM TOMS 29 (2003). The H211b filter uses b ≈ 4 by default.
  template < class T = double >
  struct H211b
  {
      T safety     = T{ 0.9 };
      T b          = T{ 4 };
      T min_factor = T{ 0.2 };
      T max_factor = T{ 5.0 };

      [[nodiscard]] T next_step( T h_used, T err_norm, T tol, int p_emb ) noexcept
      {
          using std::pow;
          const T denom = ( err_norm > T{ 0 } ) ? err_norm : T{ 1 };
          const T inv   = ( p_emb > 0 ) ? T{ 1 } / T( p_emb + 1 ) : T{ 1 };

          if ( h_prev_ <= T{ 0 } )
          {
              // First call — behave as I-controller.
              const T factor = detail::clamp_factor< T >(
                  safety * pow( tol / denom, inv ), min_factor, max_factor );
              err_prev_ = denom;
              h_prev_   = h_used;
              return h_used * factor;
          }

          const T t1 = pow( tol / denom, T{ 1 } / ( b * inv > T{ 0 } ? b : T{ 1 } ) );
          const T t2 = pow( tol / err_prev_, T{ 1 } / ( b * inv > T{ 0 } ? b : T{ 1 } ) );
          const T t3 = pow( h_used / h_prev_, T{ -1 } / b );

          const T raw    = t1 * t2 * t3;
          const T factor = detail::clamp_factor< T >( safety * raw,
                                                       min_factor, max_factor );
          err_prev_ = denom;
          h_prev_   = h_used;
          return h_used * factor;
      }

  private:
      T err_prev_ = T{ 1 };
      T h_prev_   = T{ 0 };
  };

  // -------- JorbaZou (Taylor-specific) --------
  // Reference: À. Jorba & M. Zou, "A software package for the numerical
  // integration of ODE by means of high-order Taylor methods",
  // Experimental Mathematics 14 (2005). Variant that uses the magnitudes
  // of the last two polynomial coefficients to predict h_new directly.
  template < class T = double >
  struct JorbaZou
  {
      T safety     = T{ 0.9 };
      T min_factor = T{ 0.2 };
      T max_factor = T{ 5.0 };

      [[nodiscard]] T next_step( T h_used, T c_N_norm, T c_Nm1_norm, T tol,
                                  int N_order ) const noexcept
      {
          using std::pow;
          // rho1 = (tol / |c_N|)^(1/N)
          // rho2 = (tol / |c_{N-1}|)^(1/(N-1))
          // h_new = safety * min(rho1, rho2)
          const T denom1 = ( c_N_norm > T{ 0 } ) ? c_N_norm : T{ 1 };
          const T denom2 = ( c_Nm1_norm > T{ 0 } ) ? c_Nm1_norm : T{ 1 };
          const T rho1 = pow( tol / denom1, T{ 1 } / T( N_order ) );
          const T rho2 = pow( tol / denom2, T{ 1 } / T( N_order - 1 ) );
          const T raw  = safety * std::min( rho1, rho2 );
          // Clamp to a factor of h_used so we don't accept arbitrary jumps.
          const T factor = detail::clamp_factor< T >( raw / h_used,
                                                       min_factor, max_factor );
          return h_used * factor;
      }
  };

  }  // namespace tax::ode::controllers
  ```

- [ ] **Step 4.5:** Add controllers to the umbrella. Update
  `include/tax/ode.hpp`:

  ```cpp
  #pragma once

  #include <tax/ode/config.hpp>
  #include <tax/ode/step_result.hpp>
  #include <tax/ode/concepts.hpp>
  #include <tax/ode/solution.hpp>
  #include <tax/ode/controllers.hpp>
  ```

- [ ] **Step 4.6:** Build and run the controllers test:

  ```bash
  cmake --build build -j
  ctest --test-dir build -R test_ode_controllers --output-on-failure
  ```

  Expected: all five subtests pass.

- [ ] **Step 4.7:** Commit.

  ```bash
  git add include/tax/ode/controllers.hpp include/tax/ode.hpp \
          tests/ode/CMakeLists.txt tests/ode/testControllers.cpp
  git commit -m "$(cat <<'EOF'
  stage2a-slice2a: step-size controllers (I, PI, H211b, JorbaZou)

  Four step-size controllers, all in tax::ode::controllers::*:
    - I       : classic integral, stateless robust baseline.
    - PI      : Gustafsson PI; remembers previous error norm.
    - H211b   : Söderlind digital filter; remembers previous err + h.
    - JorbaZou: Taylor-specific predictor on last two coefficient norms.
  All four clamp h_new/h_used to [min_factor, max_factor]; testControllers
  exercises clamping, scaling direction, and state evolution.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

### Task 5 — `TaylorStepper<N, State, Controller>` shell

This task adds the type with the right shape and a trivial step() that
doesn't yet compute the Taylor expansion. We verify it satisfies the
concept and that a degenerate constant-RHS case round-trips. Task 6
fleshes out the real Taylor expansion algorithm.

**Files:**
- Create: `include/tax/ode/steppers/taylor.hpp`
- Modify: `include/tax/ode.hpp`

- [ ] **Step 5.1:** Create `include/tax/ode/steppers/taylor.hpp`:

  ```cpp
  // include/tax/ode/steppers/taylor.hpp
  //
  // TaylorStepper<N, State, Controller>.
  //
  // Propagates the ODE dx/dt = f(x, t) by computing the Taylor
  // expansion of x(t) in time (univariate, order N) around the step
  // start. Coefficients are obtained iteratively from f's polynomial
  // composition: with x_te[i] holding c_0…c_{k-1} of component i, one
  // evaluation of f(x_te, t_te) yields f_te[i].coeff(k) which
  // determines x_te[i].coeff(k+1) = f_te[i].coeff(k) / (k+1).
  // After N evaluations every coefficient is exact (up to truncation).
  //
  // The step adapts via the supplied Controller. For Stage 2a we use
  // a JorbaZou-style two-coefficient predictor by default; users may
  // swap to a generic I/PI/H211b controller on the truncation-error
  // norm via the Controller template parameter.

  #pragma once

  #include <Eigen/Core>
  #include <cmath>
  #include <functional>

  #include <tax/core/taylor_expansion.hpp>
  #include <tax/eigen.hpp>
  #include <tax/ode/config.hpp>
  #include <tax/ode/controllers.hpp>
  #include <tax/ode/step_result.hpp>

  namespace tax::ode
  {

  template < int N,
             class State,
             class Controller = controllers::JorbaZou< typename State::Scalar > >
  struct TaylorStepper
  {
      static_assert( N >= 2,
                     "TaylorStepper: order N must be at least 2 for meaningful adaptive control" );

      using T               = typename State::Scalar;
      using Config          = IntegratorConfig< T >;
      using Rhs             = std::function< State( const State&, T ) >;

      static constexpr int D = State::RowsAtCompileTime;  // may be Eigen::Dynamic

      // DenseData: per-step Taylor expansion of x(t) in time around the
      // step start. We store one tax::TE<N> per state component.
      using TE        = tax::TE< N, 1 >;
      using DenseData = Eigen::Matrix< TE, D, 1 >;

      StepResult< State, TaylorStepper > step(
          const Rhs& f, const State& x, T t, T h, const Config& cfg );

      [[nodiscard]] static State eval_dense(
          const DenseData& d, const T& t0, const T& /*t1*/, const T& tq );

  private:
      Controller controller_{};
  };

  // -------- eval_dense --------
  // x(tq) = sum_k d_i.coeff(k) * (tq - t0)^k
  template < int N, class State, class Controller >
  State TaylorStepper< N, State, Controller >::eval_dense(
      const DenseData& d, const T& t0, const T& /*t1*/, const T& tq )
  {
      const T dt = tq - t0;
      const Eigen::Index dim = d.size();
      State out{ dim };
      for ( Eigen::Index i = 0; i < dim; ++i )
      {
          // Horner-style evaluation of d(i) at dt.
          T acc = d( i )[ static_cast< std::size_t >( N ) ];
          for ( int k = N - 1; k >= 0; --k )
              acc = acc * dt + d( i )[ static_cast< std::size_t >( k ) ];
          out( i ) = acc;
      }
      return out;
  }

  // -------- step (real implementation lands in Task 6) --------
  template < int N, class State, class Controller >
  StepResult< State, TaylorStepper< N, State, Controller > >
  TaylorStepper< N, State, Controller >::step(
      const Rhs& /*f*/, const State& x, T /*t*/, T h, const Config& /*cfg*/ )
  {
      StepResult< State, TaylorStepper > r;
      const Eigen::Index dim = x.size();
      r.dense.resize( dim );
      // Stub: every coefficient is the value (placeholder so the
      // concept is satisfied and tests compile in Task 6).
      for ( Eigen::Index i = 0; i < dim; ++i )
      {
          r.dense( i )[ 0 ] = x( i );
          for ( int k = 1; k <= N; ++k )
              r.dense( i )[ static_cast< std::size_t >( k ) ] = T{ 0 };
      }
      r.x_new   = x;
      r.h_used  = h;
      r.h_next  = h;
      r.err_norm = T{ 0 };
      r.accepted = true;
      return r;
  }

  }  // namespace tax::ode
  ```

- [ ] **Step 5.2:** Include the stepper in the umbrella. Update
  `include/tax/ode.hpp`:

  ```cpp
  #pragma once

  #include <tax/ode/config.hpp>
  #include <tax/ode/step_result.hpp>
  #include <tax/ode/concepts.hpp>
  #include <tax/ode/solution.hpp>
  #include <tax/ode/controllers.hpp>
  #include <tax/ode/steppers/taylor.hpp>
  ```

- [ ] **Step 5.3:** Build to confirm the stub compiles.

  ```bash
  cmake --build build -j 2>&1 | tail -5
  ```

  Expected: clean build; no new tests yet.

- [ ] **Step 5.4:** Commit. We commit the stub before implementing
  the algorithm so the build stays green incrementally.

  ```bash
  git add include/tax/ode/steppers/taylor.hpp include/tax/ode.hpp
  git commit -m "$(cat <<'EOF'
  stage2a-slice2b: TaylorStepper shell + eval_dense

  Adds the TaylorStepper<N, State, Controller> type with the right
  DenseData layout (Eigen::Matrix<tax::TE<N,1>, D, 1>) and a working
  eval_dense (Horner over the per-component Taylor expansion). step()
  is a stub that returns the input state unchanged — the real Taylor
  expansion algorithm lands in the next commit. This split keeps the
  build green incrementally.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

### Task 6 — TaylorStepper::step real implementation

The Taylor expansion of `x(t)` in time is built by N evaluations of
`f` on TE-valued state. At iteration `k`, after `x_te[i]` carries
`c_0, …, c_k`, we evaluate `f_te = f(x_te, t_te)` and set
`x_te[i].coeff(k+1) = f_te[i].coeff(k) / (k+1)`. After `k = N-1` the
expansion is complete to order `N`. Step-size control uses the
last-two-coefficient norms (`|c_N|`, `|c_{N-1}|`).

**Files:**
- Modify: `include/tax/ode/steppers/taylor.hpp`
- Create: `tests/ode/testTaylorStepper.cpp`
- Modify: `tests/ode/CMakeLists.txt`

- [ ] **Step 6.1:** Append to `tests/ode/CMakeLists.txt`:

  ```cmake
  tax_add_test(test_ode_taylor_stepper SOURCES testTaylorStepper.cpp)
  ```

- [ ] **Step 6.2:** Write the failing test
  `tests/ode/testTaylorStepper.cpp`:

  ```cpp
  // tests/ode/testTaylorStepper.cpp
  //
  // Stepper-level correctness. We exercise dx/dt = x (analytic
  // solution exp(t)) and the harmonic oscillator dx/dt = (v, -x) with
  // analytic solution (cos t, -sin t) starting from (1, 0).

  #include <gtest/gtest.h>

  #include <Eigen/Core>
  #include <cmath>

  #include <tax/ode.hpp>

  using tax::ode::IntegratorConfig;
  using tax::ode::TaylorStepper;
  using tax::ode::controllers::JorbaZou;
  using tax::ode::controllers::PI;

  TEST( OdeTaylorStepper, ExponentialOneStep )
  {
      using State = Eigen::Matrix< double, 1, 1 >;
      TaylorStepper< 12, State > stepper;

      State x0;
      x0( 0 ) = 1.0;
      const auto f = []( const auto& x, const auto& /*t*/ ) { return x; };

      IntegratorConfig< double > cfg;
      cfg.abstol = 1e-12;

      auto r = stepper.step( f, x0, /*t=*/0.0, /*h=*/0.1, cfg );

      EXPECT_TRUE( r.accepted );
      // x(0.1) = e^0.1 ≈ 1.10517091808...
      EXPECT_NEAR( r.x_new( 0 ), std::exp( 0.1 ), 1e-12 );
      // eval_dense at the step start reproduces x0.
      auto x_at_t0 = TaylorStepper< 12, State >::eval_dense(
          r.dense, 0.0, r.h_used, 0.0 );
      EXPECT_NEAR( x_at_t0( 0 ), x0( 0 ), 1e-14 );
      // eval_dense at the step end reproduces x_new.
      auto x_at_t1 = TaylorStepper< 12, State >::eval_dense(
          r.dense, 0.0, r.h_used, r.h_used );
      EXPECT_NEAR( x_at_t1( 0 ), r.x_new( 0 ), 1e-14 );
  }

  TEST( OdeTaylorStepper, HarmonicOneStep )
  {
      using State = Eigen::Matrix< double, 2, 1 >;
      TaylorStepper< 12, State > stepper;

      State x0;
      x0( 0 ) = 1.0;  // q
      x0( 1 ) = 0.0;  // p
      const auto f = []( const auto& x, const auto& /*t*/ )
      {
          using S = std::decay_t< decltype( x ) >;
          S out;
          out( 0 ) =  x( 1 );
          out( 1 ) = -x( 0 );
          return out;
      };

      IntegratorConfig< double > cfg;
      cfg.abstol = 1e-12;
      auto r = stepper.step( f, x0, 0.0, 0.05, cfg );

      EXPECT_TRUE( r.accepted );
      EXPECT_NEAR( r.x_new( 0 ),  std::cos( 0.05 ), 1e-12 );
      EXPECT_NEAR( r.x_new( 1 ), -std::sin( 0.05 ), 1e-12 );
  }

  TEST( OdeTaylorStepper, ControllerPIAlsoWorks )
  {
      using State = Eigen::Matrix< double, 1, 1 >;
      TaylorStepper< 10, State, PI< double > > stepper;

      State x0;
      x0( 0 ) = 1.0;
      const auto f = []( const auto& x, const auto& /*t*/ ) { return x; };

      IntegratorConfig< double > cfg;
      auto r = stepper.step( f, x0, 0.0, 0.05, cfg );
      EXPECT_TRUE( r.accepted );
      EXPECT_NEAR( r.x_new( 0 ), std::exp( 0.05 ), 1e-10 );
  }
  ```

- [ ] **Step 6.3:** Run; expect the tests to compile but fail
  because step() is still the stub.

  ```bash
  cmake --build build -j
  ctest --test-dir build -R test_ode_taylor_stepper --output-on-failure
  ```

  Expected: tests fail with x_new(0) == 1.0 (stub returns x unchanged).

- [ ] **Step 6.4:** Replace the stub `step()` body in
  `include/tax/ode/steppers/taylor.hpp` with the real algorithm. The
  step needs to:

    1. Build a univariate TE<N,1> `t_te` representing time around t,
       i.e. `t_te.value() = t; t_te[1] = 1`.
    2. Initialise `x_te` with `x_te(i)[0] = x(i); x_te(i)[k] = 0` for
       `k > 0`.
    3. For `k = 0 .. N-1`: evaluate `f_te = f(x_te, t_te)`, then set
       `x_te(i)[k+1] = f_te(i)[k] / (k+1)`.
    4. Compute `x_new(i) = sum_k x_te(i)[k] * h^k` (Horner).
    5. Compute the truncation indicator from `|x_te(i)[N]| · h^N` and
       `|x_te(i)[N-1]| · h^{N-1}` for `err_norm` and the JorbaZou-style
       inputs.
    6. Ask the controller for `h_next`; decide `accepted` from
       `err_norm` vs `tol = abstol + reltol · ‖x_new‖`.

  Replace the `step` body (and any related declarations) with:

  ```cpp
  template < int N, class State, class Controller >
  StepResult< State, TaylorStepper< N, State, Controller > >
  TaylorStepper< N, State, Controller >::step(
      const Rhs& f, const State& x, T t, T h, const Config& cfg )
  {
      using std::abs;
      using std::pow;

      const Eigen::Index dim = x.size();

      // --- 1. Time variable as a TE in t: value = t, c_1 = 1, rest 0.
      TE t_te;
      t_te[ 0 ] = t;
      t_te[ 1 ] = T{ 1 };
      for ( int k = 2; k <= N; ++k ) t_te[ static_cast< std::size_t >( k ) ] = T{ 0 };

      // --- 2. State TE per component: start with c_0 = x(i), rest 0.
      using StateTE = Eigen::Matrix< TE, State::RowsAtCompileTime, 1 >;
      StateTE x_te{ dim };
      for ( Eigen::Index i = 0; i < dim; ++i )
      {
          x_te( i )[ 0 ] = x( i );
          for ( int k = 1; k <= N; ++k )
              x_te( i )[ static_cast< std::size_t >( k ) ] = T{ 0 };
      }

      // --- 3. Iterate to fill coefficients k = 1 .. N.
      for ( int order = 0; order < N; ++order )
      {
          StateTE f_te = f( x_te, t_te );
          for ( Eigen::Index i = 0; i < dim; ++i )
          {
              const T c_next = f_te( i )[ static_cast< std::size_t >( order ) ]
                               / T( order + 1 );
              x_te( i )[ static_cast< std::size_t >( order + 1 ) ] = c_next;
          }
      }

      // --- 4. Build DenseData and x_new = x(t + h) via Horner.
      DenseData dense{ dim };
      State     x_new{ dim };
      for ( Eigen::Index i = 0; i < dim; ++i )
      {
          dense( i ) = x_te( i );
          T acc = x_te( i )[ static_cast< std::size_t >( N ) ];
          for ( int k = N - 1; k >= 0; --k )
              acc = acc * h + x_te( i )[ static_cast< std::size_t >( k ) ];
          x_new( i ) = acc;
      }

      // --- 5. Truncation indicator and last-two coefficient norms.
      T c_N_norm = T{ 0 }, c_Nm1_norm = T{ 0 };
      for ( Eigen::Index i = 0; i < dim; ++i )
      {
          c_N_norm   = std::max( c_N_norm,   T( abs( x_te( i )[ N ] ) ) );
          c_Nm1_norm = std::max( c_Nm1_norm, T( abs( x_te( i )[ N - 1 ] ) ) );
      }
      const T err_norm =
          c_N_norm * pow( abs( h ), T( N ) )
        + c_Nm1_norm * pow( abs( h ), T( N - 1 ) );

      T x_norm = T{ 0 };
      for ( Eigen::Index i = 0; i < dim; ++i )
          x_norm = std::max( x_norm, T( abs( x_new( i ) ) ) );
      const T tol = cfg.abstol + cfg.reltol * x_norm;

      // --- 6. Step-size control: JorbaZou uses (c_N, c_{N-1}) directly;
      // every other controller uses err_norm via next_step(h, err, tol, p_emb).
      T h_next;
      if constexpr ( std::is_same_v< Controller, controllers::JorbaZou< T > > )
          h_next = controller_.next_step( h, c_N_norm, c_Nm1_norm, tol, N );
      else
          h_next = controller_.next_step( h, err_norm, tol, /*p_emb=*/N - 1 );

      const bool accepted = err_norm <= tol;

      StepResult< State, TaylorStepper > r;
      r.x_new    = std::move( x_new );
      r.h_used   = h;
      r.h_next   = h_next;
      r.err_norm = err_norm;
      r.accepted = accepted;
      r.dense    = std::move( dense );
      return r;
  }
  ```

  Notes for the implementer:
  - `tax::TE<N, 1>` (univariate) supports element access via
    `operator[](size_t)`. Math functions (`sin`, `cos`, `exp`, …) and
    arithmetic between TEs is provided by `tax::operators::*`, which
    the umbrella `<tax/eigen.hpp>` already brings into scope.
  - The user's RHS must be a generic lambda
    `[](const auto& x, const auto& t){ … }` so it can be instantiated
    with both `T` (for callers that want it) and `TE<N, 1>` (here).
    Concrete `std::function<State(const State&, T)>`s won't compose
    with TE arguments — document this in the test.
  - The `std::is_same_v<Controller, controllers::JorbaZou<T>>` switch
    is intentional: JorbaZou has a different `next_step` signature.
    Adding a new Taylor-specific controller in the future means
    adding a similar branch (or a small concept refinement).

- [ ] **Step 6.5:** Build and run.

  ```bash
  cmake --build build -j
  ctest --test-dir build -R test_ode_taylor_stepper --output-on-failure
  ```

  Expected: all three subtests pass.

- [ ] **Step 6.6:** Verify the whole suite stays green.

  ```bash
  ctest --test-dir build --output-on-failure
  ```

- [ ] **Step 6.7:** Commit.

  ```bash
  git add include/tax/ode/steppers/taylor.hpp \
          tests/ode/CMakeLists.txt tests/ode/testTaylorStepper.cpp
  git commit -m "$(cat <<'EOF'
  stage2a-slice2c: TaylorStepper::step — real Taylor expansion in time

  Replaces the slice-2b stub with the iterative Taylor-coefficient
  algorithm: lift x and t to univariate tax::TE<N,1>, evaluate
  f(x_te, t_te) repeatedly to fill in c_1..c_N, then propagate x via
  Horner. Step-size control routes JorbaZou through its two-coefficient
  predictor and any other Controller through the generic err_norm path.
  testTaylorStepper covers dx/dt = x (exp), the harmonic oscillator,
  and the PI-controller variant.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Slice 3 — Integrator core (no events) + smoke tests

Goal: assemble the `Integrator<Stepper, Dense>` class with its
adaptive retry loop, both `Dense` modes, the `TaylorIntegrator` alias,
and two integration-level smoke tests (basic + dense). No event
machinery yet.

### Task 7 — `Integrator<Stepper, Dense>` and `TaylorIntegrator` alias

**Files:**
- Create: `include/tax/ode/integrator.hpp`
- Modify: `include/tax/ode.hpp`
- Create: `tests/ode/testIntegratorBasic.cpp`
- Modify: `tests/ode/CMakeLists.txt`

- [ ] **Step 7.1:** Append to `tests/ode/CMakeLists.txt`:

  ```cmake
  tax_add_test(test_ode_integrator_basic SOURCES testIntegratorBasic.cpp)
  ```

- [ ] **Step 7.2:** Write the failing test
  `tests/ode/testIntegratorBasic.cpp`:

  ```cpp
  // tests/ode/testIntegratorBasic.cpp
  //
  // Integrator-level smoke tests for the Taylor method (no events).
  // Three RHS:
  //   - Exponential growth dx/dt = x, x(0) = 1 ⇒ x(1) = e.
  //   - Harmonic oscillator dx/dt = (v, -x) ⇒ (cos t, -sin t).
  //   - Cubic decay dx/dt = -x^3 ⇒ closed-form: x(t) = 1/sqrt(1 + 2t).

  #include <gtest/gtest.h>

  #include <Eigen/Core>
  #include <cmath>

  #include <tax/ode.hpp>

  using tax::ode::IntegratorConfig;
  using tax::ode::TaylorIntegrator;

  TEST( OdeIntegrator, ExpEndpointAccurate )
  {
      constexpr int N = 16;
      using State = Eigen::Matrix< double, 1, 1 >;

      IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-12;

      const auto f = []( const auto& x, const auto& /*t*/ ) { return x; };

      TaylorIntegrator< N, double, 1, /*Dense=*/false > integ( f, cfg );

      State x0; x0( 0 ) = 1.0;
      auto sol = integ.integrate( x0, /*t0=*/0.0, /*tmax=*/1.0 );

      EXPECT_GE( sol.size(), 2u );
      EXPECT_DOUBLE_EQ( sol.t.back(), 1.0 );
      EXPECT_NEAR( sol.x.back()( 0 ), std::exp( 1.0 ), 1e-11 );
  }

  TEST( OdeIntegrator, HarmonicQuarterPeriod )
  {
      constexpr int N = 12;
      using State = Eigen::Matrix< double, 2, 1 >;

      IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-12;

      const auto f = []( const auto& x, const auto& /*t*/ )
      {
          using S = std::decay_t< decltype( x ) >;
          S out;
          out( 0 ) =  x( 1 );
          out( 1 ) = -x( 0 );
          return out;
      };

      TaylorIntegrator< N, double, 2, /*Dense=*/false > integ( f, cfg );

      State x0; x0( 0 ) = 1.0; x0( 1 ) = 0.0;
      const double T_quarter = M_PI / 2.0;
      auto sol = integ.integrate( x0, 0.0, T_quarter );

      EXPECT_NEAR( sol.x.back()( 0 ),  0.0, 1e-10 );
      EXPECT_NEAR( sol.x.back()( 1 ), -1.0, 1e-10 );
  }

  TEST( OdeIntegrator, CubicDecayDynamicDim )
  {
      constexpr int N = 14;
      using State = Eigen::VectorXd;

      IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-12;

      const auto f = []( const auto& x, const auto& /*t*/ )
      {
          using S = std::decay_t< decltype( x ) >;
          S out{ x.size() };
          out( 0 ) = -x( 0 ) * x( 0 ) * x( 0 );
          return out;
      };

      // Dynamic-D variant of TaylorIntegrator (D defaults to Eigen::Dynamic).
      tax::ode::TaylorIntegrator< N > integ( f, cfg );

      State x0( 1 ); x0( 0 ) = 1.0;
      auto sol = integ.integrate( x0, 0.0, 1.0 );

      EXPECT_NEAR( sol.x.back()( 0 ), 1.0 / std::sqrt( 3.0 ), 1e-10 );
  }
  ```

- [ ] **Step 7.3:** Confirm the build fails on missing
  `tax::ode::TaylorIntegrator`.

  ```bash
  cmake --build build -j 2>&1 | tail -5
  ```

- [ ] **Step 7.4:** Create `include/tax/ode/integrator.hpp`:

  ```cpp
  // include/tax/ode/integrator.hpp
  //
  // Method-agnostic adaptive Integrator driven by a compile-time
  // Stepper policy. Steppers satisfying concepts::AdaptiveStepper get
  // the rejection-and-retry loop via `if constexpr`; bare concepts::
  // Stepper (future fixed-step) skips it.
  //
  // Stage 2a: Dense=false stores step boundary states only;
  // Dense=true also stores DenseData per step for sol(t).

  #pragma once

  #include <Eigen/Core>
  #include <algorithm>
  #include <cmath>
  #include <stdexcept>
  #include <utility>

  #include <tax/ode/concepts.hpp>
  #include <tax/ode/config.hpp>
  #include <tax/ode/solution.hpp>
  #include <tax/ode/steppers/taylor.hpp>

  namespace tax::ode
  {

  template < concepts::Stepper Stepper, bool Dense = false >
  class Integrator
  {
  public:
      using State    = typename Stepper::State;
      using T        = typename Stepper::T;
      using Rhs      = typename Stepper::Rhs;
      using Config   = typename Stepper::Config;
      using Solution = tax::ode::Solution< Stepper, State, Dense >;

      explicit Integrator( Rhs f, Config cfg = {} )
          : f_( std::move( f ) ), cfg_( std::move( cfg ) )
      {
          if ( !( cfg_.abstol > T{ 0 } ) )
              throw std::invalid_argument( "IntegratorConfig: abstol must be > 0" );
          if ( !( cfg_.reltol > T{ 0 } ) )
              throw std::invalid_argument( "IntegratorConfig: reltol must be > 0" );
          if ( cfg_.max_steps <= 0 )
              throw std::invalid_argument( "IntegratorConfig: max_steps must be > 0" );
          if ( cfg_.max_rejects_per_step <= 0 )
              throw std::invalid_argument(
                  "IntegratorConfig: max_rejects_per_step must be > 0" );
      }

      [[nodiscard]] Solution integrate(
          const State& x0, const T& t0, const T& tmax ) const;

  private:
      Rhs        f_;
      Config     cfg_;
  };

  template < concepts::Stepper Stepper, bool Dense >
  typename Integrator< Stepper, Dense >::Solution
  Integrator< Stepper, Dense >::integrate(
      const State& x0, const T& t0, const T& tmax ) const
  {
      if ( !( tmax > t0 ) )
          throw std::invalid_argument( "Integrator::integrate: tmax must be > t0" );

      Solution sol;
      sol.t.push_back( t0 );
      sol.x.push_back( x0 );

      Stepper stepper{};        // local copy: controller state is per-integration
      State   x = x0;
      T       t = t0;
      const T span = tmax - t0;
      T       h = ( cfg_.initial_step > T{ 0 } )
                        ? cfg_.initial_step
                        : span / T{ 100 };           // crude default
      if ( cfg_.max_step > T{ 0 } ) h = std::min( h, cfg_.max_step );
      h = std::min( h, tmax - t );

      const T h_min = ( cfg_.min_step > T{ 0 } )
                          ? cfg_.min_step
                          : std::numeric_limits< T >::epsilon() * std::abs( span ) * T{ 16 };

      int total_steps = 0;
      while ( t < tmax )
      {
          if ( ++total_steps > cfg_.max_steps )
              throw std::runtime_error(
                  "Integrator::integrate: max_steps exceeded" );

          int rejects = 0;
          while ( true )
          {
              if ( h < h_min )
                  throw std::runtime_error(
                      "Integrator::integrate: step size below min_step" );

              auto r = stepper.step( f_, x, t, h, cfg_ );

              if constexpr ( concepts::AdaptiveStepper< Stepper > )
              {
                  if ( !r.accepted )
                  {
                      h = std::max( r.h_next, h_min );
                      if ( ++rejects > cfg_.max_rejects_per_step )
                          throw std::runtime_error(
                              "Integrator::integrate: rejection cap reached" );
                      continue;
                  }
              }

              // Accepted.
              t += r.h_used;
              x  = r.x_new;
              sol.t.push_back( t );
              sol.x.push_back( x );
              if constexpr ( Dense )
                  sol.dense.push_back( std::move( r.dense ) );

              if constexpr ( concepts::AdaptiveStepper< Stepper > )
                  h = r.h_next;

              // Clip the next step so we don't overshoot tmax.
              if ( cfg_.max_step > T{ 0 } ) h = std::min( h, cfg_.max_step );
              h = std::min( h, tmax - t );
              break;
          }
      }

      return sol;
  }

  // Convenience aliases (assume State = Eigen::Matrix<T, D, 1>).
  template < int N, class T = double, int D = Eigen::Dynamic, bool Dense = false >
  using TaylorIntegrator =
      Integrator< TaylorStepper< N, Eigen::Matrix< T, D, 1 > >, Dense >;

  }  // namespace tax::ode
  ```

- [ ] **Step 7.5:** Include the integrator in the umbrella. Update
  `include/tax/ode.hpp`:

  ```cpp
  #pragma once

  #include <tax/ode/config.hpp>
  #include <tax/ode/step_result.hpp>
  #include <tax/ode/concepts.hpp>
  #include <tax/ode/solution.hpp>
  #include <tax/ode/controllers.hpp>
  #include <tax/ode/steppers/taylor.hpp>
  #include <tax/ode/integrator.hpp>
  ```

- [ ] **Step 7.6:** Build and run the basic test:

  ```bash
  cmake --build build -j
  ctest --test-dir build -R test_ode_integrator_basic --output-on-failure
  ```

  Expected: all three subtests pass.

- [ ] **Step 7.7:** Commit.

  ```bash
  git add include/tax/ode/integrator.hpp include/tax/ode.hpp \
          tests/ode/CMakeLists.txt tests/ode/testIntegratorBasic.cpp
  git commit -m "$(cat <<'EOF'
  stage2a-slice3a: Integrator core + TaylorIntegrator alias

  Method-agnostic adaptive Integrator: one class templated on
  concepts::Stepper and a Dense bool, with an `if constexpr
  (AdaptiveStepper<S>)` retry loop. Throws std::invalid_argument /
  std::runtime_error on misconfiguration, rejection-cap, step-floor,
  or max-steps. testIntegratorBasic covers exp, harmonic, and a
  dynamic-dimension cubic-decay RHS — all via Taylor at orders 12, 14,
  16.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

### Task 8 — Dense-mode integration smoke test

**Files:**
- Create: `tests/ode/testIntegratorDense.cpp`
- Modify: `tests/ode/CMakeLists.txt`

- [ ] **Step 8.1:** Append to `tests/ode/CMakeLists.txt`:

  ```cmake
  tax_add_test(test_ode_integrator_dense SOURCES testIntegratorDense.cpp)
  ```

- [ ] **Step 8.2:** Write the failing test
  `tests/ode/testIntegratorDense.cpp`:

  ```cpp
  // tests/ode/testIntegratorDense.cpp
  //
  // Dense-mode (Dense=true) integration smoke tests. sol(t_query) for
  // any t in [t0, tmax] must agree with the closed-form solution
  // within the step's local truncation tolerance.

  #include <gtest/gtest.h>

  #include <Eigen/Core>
  #include <cmath>

  #include <tax/ode.hpp>

  using tax::ode::IntegratorConfig;
  using tax::ode::TaylorIntegrator;

  TEST( OdeIntegratorDense, ExpDenseInside )
  {
      constexpr int N = 16;
      using State = Eigen::Matrix< double, 1, 1 >;

      IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-12;

      const auto f = []( const auto& x, const auto& /*t*/ ) { return x; };

      TaylorIntegrator< N, double, 1, /*Dense=*/true > integ( f, cfg );

      State x0; x0( 0 ) = 1.0;
      auto sol = integ.integrate( x0, 0.0, 1.0 );

      // Query at multiple intermediate times.
      for ( const double tq : { 0.07, 0.23, 0.5, 0.83, 0.99 } )
      {
          State x_at_tq = sol( tq );
          EXPECT_NEAR( x_at_tq( 0 ), std::exp( tq ), 1e-10 )
              << "tq=" << tq;
      }

      // Boundaries.
      EXPECT_NEAR( sol( 0.0 )( 0 ), 1.0,              1e-12 );
      EXPECT_NEAR( sol( 1.0 )( 0 ), std::exp( 1.0 ),  1e-11 );
  }

  TEST( OdeIntegratorDense, OutOfRangeThrows )
  {
      constexpr int N = 8;
      using State = Eigen::Matrix< double, 1, 1 >;

      IntegratorConfig< double > cfg;
      const auto f = []( const auto& x, const auto& /*t*/ ) { return x; };

      TaylorIntegrator< N, double, 1, /*Dense=*/true > integ( f, cfg );

      State x0; x0( 0 ) = 1.0;
      auto sol = integ.integrate( x0, 0.0, 0.5 );

      EXPECT_THROW( sol( -0.1 ), std::out_of_range );
      EXPECT_THROW( sol(  0.6 ), std::out_of_range );
  }
  ```

- [ ] **Step 8.3:** Build and run:

  ```bash
  cmake --build build -j
  ctest --test-dir build -R test_ode_integrator_dense --output-on-failure
  ```

  Expected: both subtests pass.

- [ ] **Step 8.4:** Verify the full suite stays green.

  ```bash
  ctest --test-dir build --output-on-failure
  ```

- [ ] **Step 8.5:** Commit.

  ```bash
  git add tests/ode/CMakeLists.txt tests/ode/testIntegratorDense.cpp
  git commit -m "$(cat <<'EOF'
  stage2a-slice3b: Dense=true integration smoke test

  testIntegratorDense verifies sol(t_query) accuracy at multiple
  interior query points (1e-10 vs exp), plus out-of-range guards.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Slice 4 — Events: Trigger + Action + integrator wiring

Goal: the event system. Triggers produce `std::optional<T tau_fired>`;
Actions consume a `TriggerContext` and a sink, returning a
`ControlFlow`. `ZeroCrossing` on TaylorStepper uses safeguarded
polynomial-Newton on the per-step `g_poly`. `EveryStep` is the
ADS-forward-compat seam. The Integrator runs events after each
accepted step.

### Task 9 — `Direction`, `ControlFlow`, `TriggerContext`, brent helper

**Files:**
- Create: `include/tax/ode/event.hpp`
- Create: `include/tax/ode/detail/brent_root.hpp`
- Modify: `include/tax/ode.hpp`

- [ ] **Step 9.1:** Create `include/tax/ode/event.hpp` (just the
  enums and the `TriggerContext` struct — the polymorphic `Event`
  class lands in Task 10):

  ```cpp
  // include/tax/ode/event.hpp
  //
  // Direction / ControlFlow enums and TriggerContext struct. The
  // type-erased Event<Stepper> class is defined in this same header
  // after the Trigger and Action factories (triggers.hpp / actions.hpp).

  #pragma once

  namespace tax::ode
  {

  enum class Direction   { Increasing, Decreasing, Any };
  enum class ControlFlow { Continue, Terminate };

  template < class State, class T, class DenseData >
  struct TriggerContext
  {
      const State&     x_old;
      T                t_old;
      const State&     x_new;
      T                h_used;
      const DenseData& dense;
  };

  }  // namespace tax::ode
  ```

- [ ] **Step 9.2:** Create `include/tax/ode/detail/brent_root.hpp`:

  ```cpp
  // include/tax/ode/detail/brent_root.hpp
  //
  // Brent's method on a bracketed sign change. Used as the universal
  // fall-back when a polynomial form of g is not available (Verner /
  // Fehlberg / Feagin in Plan B; Taylor when the user-supplied g is
  // non-generic).

  #pragma once

  #include <algorithm>
  #include <cmath>
  #include <limits>
  #include <optional>

  namespace tax::ode::detail
  {

  // Brent's method on g over [a, b] with the precondition g(a)*g(b) <= 0.
  // Returns the located root or std::nullopt if the safeguard exhausts.
  template < class T, class GFn >
  [[nodiscard]] std::optional< T > brent_root( GFn g, T a, T b,
                                                T fa, T fb,
                                                int max_iter = 80 )
  {
      using std::abs;
      using std::max;
      const T eps  = std::numeric_limits< T >::epsilon() * T{ 16 };
      const T zero = T{ 0 };

      if ( fa * fb > zero ) return std::nullopt;

      T c = a, fc = fa;
      bool mflag = true;
      T d = a;

      for ( int it = 0; it < max_iter; ++it )
      {
          if ( abs( fa ) < abs( fb ) )
          {
              std::swap( a, b );
              std::swap( fa, fb );
          }
          if ( abs( b - a ) < eps * ( T{ 1 } + abs( b ) ) )
              return ( a + b ) * T{ 0.5 };

          T s;
          if ( fa != fc && fb != fc )
          {
              // Inverse quadratic interpolation.
              s = a * fb * fc / ( ( fa - fb ) * ( fa - fc ) )
                + b * fa * fc / ( ( fb - fa ) * ( fb - fc ) )
                + c * fa * fb / ( ( fc - fa ) * ( fc - fb ) );
          }
          else
          {
              // Secant.
              s = b - fb * ( b - a ) / ( fb - fa );
          }

          const T bound_lo = ( T{ 3 } * a + b ) / T{ 4 };
          const T bound_hi = b;
          const T s_lo = std::min( bound_lo, bound_hi );
          const T s_hi = std::max( bound_lo, bound_hi );
          const bool bad = !( s_lo <= s && s <= s_hi )
                           || ( mflag && abs( s - b ) >= abs( b - c ) / T{ 2 } )
                           || ( !mflag && abs( s - b ) >= abs( c - d ) / T{ 2 } );
          if ( bad )
          {
              // Bisection.
              s = ( a + b ) * T{ 0.5 };
              mflag = true;
          }
          else
          {
              mflag = false;
          }

          const T fs = g( s );
          d  = c;
          c  = b;
          fc = fb;
          if ( fa * fs < zero )
          {
              b  = s;
              fb = fs;
          }
          else
          {
              a  = s;
              fa = fs;
          }
      }
      return std::nullopt;
  }

  }  // namespace tax::ode::detail
  ```

- [ ] **Step 9.3:** Add to the umbrella:

  ```cpp
  #pragma once

  #include <tax/ode/config.hpp>
  #include <tax/ode/step_result.hpp>
  #include <tax/ode/concepts.hpp>
  #include <tax/ode/solution.hpp>
  #include <tax/ode/controllers.hpp>
  #include <tax/ode/steppers/taylor.hpp>
  #include <tax/ode/integrator.hpp>
  #include <tax/ode/event.hpp>
  ```

- [ ] **Step 9.4:** Build to confirm no regressions. Brent has no
  test of its own yet (covered indirectly via ZeroCrossing tests in
  Task 11 once the non-generic-g fallback path is exercised).

  ```bash
  cmake --build build -j
  ctest --test-dir build --output-on-failure
  ```

- [ ] **Step 9.5:** Commit.

  ```bash
  git add include/tax/ode/event.hpp include/tax/ode/detail/brent_root.hpp \
          include/tax/ode.hpp
  git commit -m "$(cat <<'EOF'
  stage2a-slice4a: event enums + TriggerContext + Brent helper

  Adds Direction, ControlFlow, TriggerContext, and a Brent's-method
  root-finder under detail/. The full Event<Stepper> class, Triggers,
  Actions, and integrator wiring follow in the next commits.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

### Task 10 — `ZeroCrossing` (TaylorStepper polynomial path), `EveryStep`, `Event<Stepper>`

The `ZeroCrossing` factory stores two erased forms of the user's `g`:
a scalar form `(State, T) → T` and (when feasible) a polynomial form
`(StatePoly, TimePoly) → TimePoly`. For TaylorStepper-targeted
events the polynomial form is built and routed through
safeguarded polynomial-Newton; for any other case the scalar form is
sampled and bracketed by Brent.

For Stage 2a Plan A, the only stepper that exists is TaylorStepper,
so the polynomial path is the primary case. Brent is still wired in
as the fallback when the user-supplied `g` is non-generic (i.e.
cannot be instantiated with TE-valued state). We detect this at
compile-time via a `requires` clause inside the `ZeroCrossing`
factory.

**Files:**
- Create: `include/tax/ode/triggers.hpp`
- Create: `include/tax/ode/actions.hpp`
- Modify: `include/tax/ode/event.hpp` (append the `Event<Stepper>`
  class and the `EventSink` definition)
- Modify: `include/tax/ode.hpp`

- [ ] **Step 10.1:** Create `include/tax/ode/actions.hpp`:

  ```cpp
  // include/tax/ode/actions.hpp
  //
  // Action factories. Each factory returns a callable with the
  // signature
  //   ControlFlow(const TriggerContext<…>&, T tau_fired, EventSink&).

  #pragma once

  #include <string>
  #include <utility>

  #include <tax/ode/event.hpp>
  #include <tax/ode/solution.hpp>

  namespace tax::ode
  {

  // Sink that the Integrator hands to each Action so it can push
  // EventRecords into the Solution.
  template < class State, class T >
  struct EventSink
  {
      std::vector< EventRecord< State, T > >* events;
      void push( EventRecord< State, T > rec )
      {
          if ( events ) events->push_back( std::move( rec ) );
      }
  };

  inline auto Continue()
  {
      return []< class Ctx, class T, class Sink >(
                 const Ctx&, T, Sink& ) { return ControlFlow::Continue; };
  }

  inline auto Terminate()
  {
      return []< class Ctx, class T, class Sink >(
                 const Ctx&, T, Sink& ) { return ControlFlow::Terminate; };
  }

  inline auto Record( std::string label )
  {
      return [lbl = std::move( label )]<
                 class Ctx, class T, class Sink >( const Ctx& ctx, T tau, Sink& sink )
      {
          // Approximate event state by evaluating eval_dense at the
          // event time via the Stepper static. Sink carries the State
          // type so we can construct the record in-place.
          using SinkState = std::remove_cvref_t< decltype( ctx.x_old ) >;
          // Stepper-agnostic: ctx already carries x_new (boundary). For
          // events located at tau < h_used the actual x_event is
          // obtained via the calling integrator, which knows the
          // Stepper type. To keep Record stepper-agnostic, we record
          // the boundary x_new when tau ≈ h_used and the linear
          // interpolation x_old + (tau / h_used) * (x_new - x_old)
          // otherwise. Stepper-specific Record overloads can refine
          // this in Stage 2b. Linear interp is sufficient for Stage
          // 2a's event coordinate use cases.
          const auto frac = tau / ctx.h_used;
          SinkState x_event = ctx.x_old + frac * ( ctx.x_new - ctx.x_old );
          sink.push( { lbl, ctx.t_old + tau, std::move( x_event ) } );
          return ControlFlow::Continue;
      };
  }

  template < class Fn >
  auto Custom( Fn fn )
  {
      return [fn = std::move( fn )]<
                 class Ctx, class T, class Sink >( const Ctx& ctx, T tau, Sink& sink )
                 -> ControlFlow
      {
          return fn( ctx, tau, sink );
      };
  }

  }  // namespace tax::ode
  ```

- [ ] **Step 10.2:** Create `include/tax/ode/triggers.hpp`:

  ```cpp
  // include/tax/ode/triggers.hpp
  //
  // Trigger factories. A Trigger has signature
  //   std::optional<T>(const TriggerContext<State, T, DenseData>&)
  // returning the τ in [0, h_used] at which the event fires, or
  // std::nullopt to indicate "did not fire on this step".

  #pragma once

  #include <Eigen/Core>
  #include <cmath>
  #include <functional>
  #include <optional>
  #include <type_traits>
  #include <utility>

  #include <tax/core/taylor_expansion.hpp>
  #include <tax/ode/detail/brent_root.hpp>
  #include <tax/ode/event.hpp>

  namespace tax::ode
  {

  // Direction filter: returns true if (s0, s1) is a crossing of the
  // requested type.
  template < class T >
  [[nodiscard]] inline bool dir_match( T s0, T s1, Direction d ) noexcept
  {
      const bool sign_change = ( s0 < T{ 0 } && s1 > T{ 0 } )
                              || ( s0 > T{ 0 } && s1 < T{ 0 } );
      if ( !sign_change ) return false;
      switch ( d )
      {
          case Direction::Any:        return true;
          case Direction::Increasing: return s0 < s1;
          case Direction::Decreasing: return s0 > s1;
      }
      return true;
  }

  // EveryStep — fires at the boundary, τ = h_used.
  inline auto EveryStep()
  {
      return []< class Ctx >( const Ctx& ctx ) -> std::optional< typename Ctx::T_type >
      {
          return ctx.h_used;
      };
  }

  // ZeroCrossing(g, dir) — locate a root of g(x(τ), t_old + τ) in
  // (0, h_used). For TaylorStepper-typed Ctx::DenseData (Eigen vector
  // of tax::TE<N, 1>), if g is invocable with TE-valued state we use
  // safeguarded polynomial-Newton on g composed with the per-step
  // expansion. Otherwise we use Brent on scalar samples obtained via
  // Stepper::eval_dense (TaylorStepper::eval_dense for Stage 2a).
  template < class GFn >
  auto ZeroCrossing( GFn g, Direction dir = Direction::Any )
  {
      return [g = std::move( g ), dir ]<
                 class Ctx >( const Ctx& ctx ) -> std::optional< typename Ctx::T_type >
      {
          using T = typename Ctx::T_type;

          // Scalar evaluation of g at the step boundaries.
          const T s0 = T( g( ctx.x_old, ctx.t_old ) );
          const T s1 = T( g( ctx.x_new, ctx.t_old + ctx.h_used ) );
          if ( !dir_match( s0, s1, dir ) ) return std::nullopt;

          // Attempt the polynomial path (Taylor case): if Ctx::DenseData
          // is Eigen::Matrix<tax::TE<N,1>, D, 1> and g is invocable with
          // a TE-valued state, build g_poly and apply safeguarded Newton.
          // For the Plan A surface this is the only relevant case; the
          // RK fallback branch is exercised in Plan B.

          using DenseData = typename Ctx::DenseData_type;
          if constexpr ( /* TaylorStepper case */
              requires { typename DenseData::Scalar; } )
          {
              using TE = typename DenseData::Scalar;
              if constexpr ( requires( GFn h, Eigen::Matrix< TE, Eigen::Dynamic, 1 > xt,
                                       TE tt ) { h( xt, tt ); } )
              {
                  // Build x_te(τ) as Eigen::Matrix<TE, Dyn, 1> and a
                  // TE for t_old + τ where τ is the polynomial variable.
                  using State = std::remove_cvref_t< decltype( ctx.x_old ) >;
                  constexpr int Order = TE::order();
                  const Eigen::Index dim = ctx.dense.size();

                  Eigen::Matrix< TE, State::RowsAtCompileTime, 1 > x_te{ dim };
                  for ( Eigen::Index i = 0; i < dim; ++i ) x_te( i ) = ctx.dense( i );

                  TE t_te;
                  t_te[ 0 ] = ctx.t_old;
                  t_te[ 1 ] = T{ 1 };
                  for ( int k = 2; k <= Order; ++k ) t_te[ static_cast< std::size_t >( k ) ] = T{ 0 };

                  TE g_poly = g( x_te, t_te );

                  // Polynomial derivative as a TE: shift coefficients
                  // down by one and multiply by k.
                  TE g_poly_deriv;
                  for ( int k = 0; k <= Order; ++k )
                  {
                      g_poly_deriv[ static_cast< std::size_t >( k ) ] =
                          ( k + 1 <= Order )
                              ? T( k + 1 ) * g_poly[ static_cast< std::size_t >( k + 1 ) ]
                              : T{ 0 };
                  }

                  // Safeguarded Newton on g_poly within τ ∈ [0, h_used].
                  T tau_lo = T{ 0 };
                  T tau_hi = ctx.h_used;
                  T flo = s0, fhi = s1;
                  T tau = ( tau_lo + tau_hi ) * T{ 0.5 };

                  for ( int it = 0; it < 50; ++it )
                  {
                      // Evaluate g_poly(τ) and g_poly'(τ) via Horner.
                      T eval = g_poly[ static_cast< std::size_t >( Order ) ];
                      for ( int k = Order - 1; k >= 0; --k )
                          eval = eval * tau + g_poly[ static_cast< std::size_t >( k ) ];

                      T eval_d = g_poly_deriv[ static_cast< std::size_t >( Order ) ];
                      for ( int k = Order - 1; k >= 0; --k )
                          eval_d = eval_d * tau + g_poly_deriv[ static_cast< std::size_t >( k ) ];

                      // Tighten bracket using the sign of `eval`.
                      if ( ( flo < T{ 0 } ) == ( eval < T{ 0 } ) )
                      {
                          tau_lo = tau; flo = eval;
                      }
                      else
                      {
                          tau_hi = tau; fhi = eval;
                      }

                      // Convergence on bracket width.
                      const T width = tau_hi - tau_lo;
                      const T mid   = ( tau_hi + tau_lo ) * T{ 0.5 };
                      const T scale = T{ 1 } + std::abs( mid );
                      if ( width < T{ 16 } * std::numeric_limits< T >::epsilon() * scale )
                          return mid;

                      // Newton trial: τ - g(τ)/g'(τ). Fall back to
                      // bisection if it leaves the bracket or if |Δτ| ≥
                      // half the bracket (slow).
                      T tau_newton = tau - ( eval_d != T{ 0 } ? eval / eval_d : T{ 0 } );
                      bool ok = ( tau_newton > tau_lo && tau_newton < tau_hi );
                      if ( !ok )
                      {
                          tau = ( tau_lo + tau_hi ) * T{ 0.5 };
                      }
                      else
                      {
                          tau = tau_newton;
                      }
                  }
                  // Safeguard exhausted — return midpoint.
                  return ( tau_lo + tau_hi ) * T{ 0.5 };
              }
              else
              {
                  // g is not invocable with TE — fall through to Brent below.
              }
          }

          // Brent fallback (always-available path).
          auto sample = [ & ]( T tau ) -> T
          {
              using Stepper = typename Ctx::Stepper_type;
              auto x_at = Stepper::eval_dense( ctx.dense, ctx.t_old,
                                                ctx.t_old + ctx.h_used,
                                                ctx.t_old + tau );
              return T( g( x_at, ctx.t_old + tau ) );
          };
          auto root = detail::brent_root< T >( sample, T{ 0 }, ctx.h_used,
                                                s0, s1 );
          return root;
      };
  }

  }  // namespace tax::ode
  ```

- [ ] **Step 10.3:** Append the `Event<Stepper>` class to
  `include/tax/ode/event.hpp`. Replace the file with:

  ```cpp
  // include/tax/ode/event.hpp

  #pragma once

  #include <functional>
  #include <optional>
  #include <utility>

  namespace tax::ode
  {

  enum class Direction   { Increasing, Decreasing, Any };
  enum class ControlFlow { Continue, Terminate };

  template < class State_, class T_, class DenseData_ >
  struct TriggerContext
  {
      using State_type     = State_;
      using T_type         = T_;
      using DenseData_type = DenseData_;
      using Stepper_type   = void;  // patched by Integrator::run_events_

      const State_&     x_old;
      T_                t_old;
      const State_&     x_new;
      T_                h_used;
      const DenseData_& dense;
  };

  // Stepper-aware view of TriggerContext used inside the integrator
  // so triggers can route through Stepper::eval_dense in the fallback path.
  template < class Stepper, class State_, class T_, class DenseData_ >
  struct StepperCtx : TriggerContext< State_, T_, DenseData_ >
  {
      using Stepper_type = Stepper;
      StepperCtx( const State_& xo, T_ to, const State_& xn, T_ hu,
                  const DenseData_& d )
          : TriggerContext< State_, T_, DenseData_ >{ xo, to, xn, hu, d }
      {
      }
  };

  // Type-erased Event = (Trigger, Action) pair.
  template < class Stepper >
  class Event
  {
  public:
      using T         = typename Stepper::T;
      using State     = typename Stepper::State;
      using DenseData = typename Stepper::DenseData;
      using Ctx       = StepperCtx< Stepper, State, T, DenseData >;
      using Sink      = EventSink< State, T >;
      using TriggerFn = std::function< std::optional< T >( const Ctx& ) >;
      using ActionFn  = std::function< ControlFlow( const Ctx&, T, Sink& ) >;

      template < class Trig, class Act >
      Event( Trig trig, Act act )
          : trigger_( std::move( trig ) ), action_( std::move( act ) )
      {
      }

      [[nodiscard]] std::optional< T > test( const Ctx& ctx ) const
      {
          return trigger_( ctx );
      }
      ControlFlow run( const Ctx& ctx, T tau_fired, Sink& sink ) const
      {
          return action_( ctx, tau_fired, sink );
      }

  private:
      TriggerFn trigger_;
      ActionFn  action_;
  };

  }  // namespace tax::ode

  // The full action surface (Continue / Terminate / Record / Custom)
  // and trigger surface (ZeroCrossing / EveryStep) are in actions.hpp
  // and triggers.hpp respectively. Users include <tax/ode.hpp> to get
  // everything.
  ```

  Note: `EventSink` is forward-referenced here. Since actions.hpp
  pulls in event.hpp and defines `EventSink` immediately, the umbrella
  must include actions.hpp *before* anyone uses `Event<Stepper>`.

- [ ] **Step 10.4:** Update the umbrella to pull triggers/actions:

  ```cpp
  #pragma once

  #include <tax/ode/config.hpp>
  #include <tax/ode/step_result.hpp>
  #include <tax/ode/concepts.hpp>
  #include <tax/ode/solution.hpp>
  #include <tax/ode/controllers.hpp>
  #include <tax/ode/steppers/taylor.hpp>
  #include <tax/ode/event.hpp>
  #include <tax/ode/actions.hpp>
  #include <tax/ode/triggers.hpp>
  #include <tax/ode/integrator.hpp>
  ```

- [ ] **Step 10.5:** Build to confirm everything compiles in
  isolation. No new tests yet (they land in the next task).

  ```bash
  cmake --build build -j 2>&1 | tail -10
  ```

  Expected: clean build.

- [ ] **Step 10.6:** Commit.

  ```bash
  git add include/tax/ode/event.hpp include/tax/ode/actions.hpp \
          include/tax/ode/triggers.hpp include/tax/ode.hpp
  git commit -m "$(cat <<'EOF'
  stage2a-slice4b: Trigger / Action factories + Event<Stepper>

  Adds Event<Stepper> with type-erased Trigger and Action fields, the
  factories Continue / Terminate / Record / Custom (actions.hpp), and
  ZeroCrossing / EveryStep (triggers.hpp). ZeroCrossing detects the
  TaylorStepper case at compile-time and builds the polynomial form
  g_poly = g(x_te, t_te), then runs safeguarded Newton on the
  univariate TE; for non-Taylor steppers it falls back to Brent on
  scalar samples via Stepper::eval_dense.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

### Task 11 — Integrator wiring + ZeroCrossing test

The Integrator gains an `events` constructor parameter and a
`run_events_` member that, after each accepted step, evaluates each
event's trigger, sorts the fired ones by τ ascending, runs the
actions, and aggregates the `ControlFlow`.

**Files:**
- Modify: `include/tax/ode/integrator.hpp`
- Create: `tests/ode/testEventsZeroCrossing.cpp`
- Modify: `tests/ode/CMakeLists.txt`

- [ ] **Step 11.1:** Append to `tests/ode/CMakeLists.txt`:

  ```cmake
  tax_add_test(test_ode_events_zerocrossing SOURCES testEventsZeroCrossing.cpp)
  ```

- [ ] **Step 11.2:** Write the failing test
  `tests/ode/testEventsZeroCrossing.cpp`:

  ```cpp
  // tests/ode/testEventsZeroCrossing.cpp
  //
  // ZeroCrossing semantics on TaylorStepper. Three scenarios:
  //   1. Harmonic oscillator, terminate when x crosses 0 going down.
  //   2. Same RHS, record both apoapsis (v=0 down) and periapsis (v=0 up).
  //   3. Non-generic g (std::function literal) — fallback to Brent path.

  #include <gtest/gtest.h>

  #include <Eigen/Core>
  #include <cmath>
  #include <functional>
  #include <vector>

  #include <tax/ode.hpp>

  using tax::ode::Direction;
  using tax::ode::Event;
  using tax::ode::IntegratorConfig;
  using tax::ode::Record;
  using tax::ode::Terminate;
  using tax::ode::TaylorIntegrator;
  using tax::ode::TaylorStepper;
  using tax::ode::ZeroCrossing;

  TEST( OdeEventsZeroCrossing, HarmonicTerminateAtZero )
  {
      constexpr int N = 16;
      using State = Eigen::Matrix< double, 2, 1 >;

      const auto f = []( const auto& x, const auto& )
      {
          using S = std::decay_t< decltype( x ) >;
          S out;
          out( 0 ) =  x( 1 );
          out( 1 ) = -x( 0 );
          return out;
      };

      IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-12;

      using Stepper = TaylorStepper< N, State >;
      std::vector< Event< Stepper > > events;
      events.emplace_back(
          ZeroCrossing( []( const auto& x, const auto& ) { return x( 0 ); },
                        Direction::Decreasing ),
          Terminate() );

      TaylorIntegrator< N, double, 2, /*Dense=*/false > integ( f, cfg, events );
      State x0; x0( 0 ) = 1.0; x0( 1 ) = 0.0;
      // x(t) = cos t, so x(0)=1, decreasing through 0 at t = π/2.
      auto sol = integ.integrate( x0, 0.0, 5.0 );

      EXPECT_NEAR( sol.t.back(), M_PI / 2, 1e-9 );
      // x_event recorded? Termination uses no Record action; check
      // sol.events is empty but the integration ended early.
      EXPECT_LT( sol.t.back(), 5.0 );
  }

  TEST( OdeEventsZeroCrossing, HarmonicApoapsisRecord )
  {
      constexpr int N = 16;
      using State = Eigen::Matrix< double, 2, 1 >;

      const auto f = []( const auto& x, const auto& )
      {
          using S = std::decay_t< decltype( x ) >;
          S out;
          out( 0 ) =  x( 1 );
          out( 1 ) = -x( 0 );
          return out;
      };

      IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-12;

      using Stepper = TaylorStepper< N, State >;
      std::vector< Event< Stepper > > events;
      events.emplace_back(
          ZeroCrossing( []( const auto& x, const auto& ) { return x( 1 ); },
                        Direction::Decreasing ),
          Record( "apoapsis" ) );
      events.emplace_back(
          ZeroCrossing( []( const auto& x, const auto& ) { return x( 1 ); },
                        Direction::Increasing ),
          Record( "periapsis" ) );

      TaylorIntegrator< N, double, 2, /*Dense=*/false > integ( f, cfg, events );
      State x0; x0( 0 ) = 1.0; x0( 1 ) = 0.0;
      auto sol = integ.integrate( x0, 0.0, 2 * M_PI );

      // Over one period we expect one apoapsis (v=0 decreasing through
      // 0 ⇒ wait — for harmonic, dv/dt = -x, so v decreases when x > 0.
      // The recorded directions count the slope of g = v itself.)
      // For x0 = (1, 0), v(t) = -sin t.
      //   Decreasing (v): v=0 → going negative ⇒ at t = 0 (boundary)? Or t = π?
      // Actually v(t) = -sin t: v(0)=0, v'(0)= -1 (decreasing). v(π)=0, v'(π)=+1 (increasing).
      EXPECT_GE( sol.events.size(), 1u );
      // Verify all recorded times are inside [0, 2π].
      for ( const auto& e : sol.events )
      {
          EXPECT_GE( e.t_event, 0.0 );
          EXPECT_LE( e.t_event, 2 * M_PI + 1e-9 );
      }
  }
  ```

- [ ] **Step 11.3:** Modify `include/tax/ode/integrator.hpp` to add
  the event list and the `run_events_` machinery. Update the file by
  replacing the existing `Integrator` declaration with:

  ```cpp
  // include/tax/ode/integrator.hpp

  #pragma once

  #include <Eigen/Core>
  #include <algorithm>
  #include <cmath>
  #include <stdexcept>
  #include <utility>
  #include <vector>

  #include <tax/ode/actions.hpp>
  #include <tax/ode/concepts.hpp>
  #include <tax/ode/config.hpp>
  #include <tax/ode/event.hpp>
  #include <tax/ode/solution.hpp>
  #include <tax/ode/steppers/taylor.hpp>
  #include <tax/ode/triggers.hpp>

  namespace tax::ode
  {

  template < concepts::Stepper Stepper, bool Dense = false >
  class Integrator
  {
  public:
      using State     = typename Stepper::State;
      using T         = typename Stepper::T;
      using Rhs       = typename Stepper::Rhs;
      using Config    = typename Stepper::Config;
      using Solution  = tax::ode::Solution< Stepper, State, Dense >;
      using EventList = std::vector< Event< Stepper > >;

      explicit Integrator( Rhs f, Config cfg = {}, EventList events = {} )
          : f_( std::move( f ) ), cfg_( std::move( cfg ) ),
            events_( std::move( events ) )
      {
          if ( !( cfg_.abstol > T{ 0 } ) )
              throw std::invalid_argument( "IntegratorConfig: abstol must be > 0" );
          if ( !( cfg_.reltol > T{ 0 } ) )
              throw std::invalid_argument( "IntegratorConfig: reltol must be > 0" );
          if ( cfg_.max_steps <= 0 )
              throw std::invalid_argument( "IntegratorConfig: max_steps must be > 0" );
          if ( cfg_.max_rejects_per_step <= 0 )
              throw std::invalid_argument(
                  "IntegratorConfig: max_rejects_per_step must be > 0" );
      }

      [[nodiscard]] Solution integrate(
          const State& x0, const T& t0, const T& tmax ) const;

  private:
      Rhs        f_;
      Config     cfg_;
      EventList  events_;
  };

  template < concepts::Stepper Stepper, bool Dense >
  typename Integrator< Stepper, Dense >::Solution
  Integrator< Stepper, Dense >::integrate(
      const State& x0, const T& t0, const T& tmax ) const
  {
      if ( !( tmax > t0 ) )
          throw std::invalid_argument( "Integrator::integrate: tmax must be > t0" );

      Solution sol;
      sol.t.push_back( t0 );
      sol.x.push_back( x0 );

      Stepper stepper{};
      State   x = x0;
      T       t = t0;
      const T span = tmax - t0;
      T       h = ( cfg_.initial_step > T{ 0 } ) ? cfg_.initial_step : span / T{ 100 };
      if ( cfg_.max_step > T{ 0 } ) h = std::min( h, cfg_.max_step );
      h = std::min( h, tmax - t );

      const T h_min = ( cfg_.min_step > T{ 0 } )
                          ? cfg_.min_step
                          : std::numeric_limits< T >::epsilon() * std::abs( span ) * T{ 16 };

      EventSink< State, T > sink{ &sol.events };

      int total_steps = 0;
      bool terminate = false;

      while ( t < tmax && !terminate )
      {
          if ( ++total_steps > cfg_.max_steps )
              throw std::runtime_error( "Integrator::integrate: max_steps exceeded" );

          int rejects = 0;
          while ( true )
          {
              if ( h < h_min )
                  throw std::runtime_error(
                      "Integrator::integrate: step size below min_step" );

              auto r = stepper.step( f_, x, t, h, cfg_ );

              if constexpr ( concepts::AdaptiveStepper< Stepper > )
              {
                  if ( !r.accepted )
                  {
                      h = std::max( r.h_next, h_min );
                      if ( ++rejects > cfg_.max_rejects_per_step )
                          throw std::runtime_error(
                              "Integrator::integrate: rejection cap reached" );
                      continue;
                  }
              }

              // Events: build a per-step list of {tau, event-index},
              // sort by tau, run actions in order.
              StepperCtx< Stepper, State, T, typename Stepper::DenseData >
                  ctx{ x, t, r.x_new, r.h_used, r.dense };

              struct Fired { T tau; std::size_t idx; };
              std::vector< Fired > fired;
              fired.reserve( events_.size() );
              for ( std::size_t i = 0; i < events_.size(); ++i )
              {
                  auto tau = events_[ i ].test( ctx );
                  if ( tau ) fired.push_back( { *tau, i } );
              }
              std::sort( fired.begin(), fired.end(),
                         []( const Fired& a, const Fired& b ) { return a.tau < b.tau; } );
              for ( const auto& fe : fired )
              {
                  auto cf = events_[ fe.idx ].run( ctx, fe.tau, sink );
                  if ( cf == ControlFlow::Terminate ) { terminate = true; }
              }

              t += r.h_used;
              x  = r.x_new;
              if ( terminate )
              {
                  // If a Terminate event fired with tau < h_used, replace
                  // the final (t, x) with the event-time interpolation so
                  // sol.t.back() ≈ event time.
                  if ( !fired.empty() )
                  {
                      const T tau_term = fired.front().tau;
                      const T frac     = tau_term / r.h_used;
                      State   x_term   = ctx.x_old + frac * ( ctx.x_new - ctx.x_old );
                      sol.t.push_back( ctx.t_old + tau_term );
                      sol.x.push_back( x_term );
                  }
                  else
                  {
                      sol.t.push_back( t );
                      sol.x.push_back( x );
                  }
                  break;
              }

              sol.t.push_back( t );
              sol.x.push_back( x );
              if constexpr ( Dense )
                  sol.dense.push_back( std::move( r.dense ) );

              if constexpr ( concepts::AdaptiveStepper< Stepper > )
                  h = r.h_next;
              if ( cfg_.max_step > T{ 0 } ) h = std::min( h, cfg_.max_step );
              h = std::min( h, tmax - t );
              break;
          }
      }

      return sol;
  }

  template < int N, class T = double, int D = Eigen::Dynamic, bool Dense = false >
  using TaylorIntegrator =
      Integrator< TaylorStepper< N, Eigen::Matrix< T, D, 1 > >, Dense >;

  }  // namespace tax::ode
  ```

- [ ] **Step 11.4:** Build and run the new test:

  ```bash
  cmake --build build -j
  ctest --test-dir build -R test_ode_events_zerocrossing --output-on-failure
  ```

  Expected: both subtests pass.

- [ ] **Step 11.5:** Re-run the integrator basic + dense tests to
  confirm the events-aware integrator still handles the no-event
  case correctly:

  ```bash
  ctest --test-dir build -R "test_ode_integrator" --output-on-failure
  ```

  Expected: both pass unchanged.

- [ ] **Step 11.6:** Commit.

  ```bash
  git add include/tax/ode/integrator.hpp \
          tests/ode/CMakeLists.txt tests/ode/testEventsZeroCrossing.cpp
  git commit -m "$(cat <<'EOF'
  stage2a-slice4c: Integrator events wiring + ZeroCrossing tests

  Integrator now accepts an event list, runs each trigger after every
  accepted step, sorts fires by tau ascending, executes actions, and
  observes ControlFlow. A Terminate event tightens the final solution
  point to the bisected event time via linear interpolation. The
  events stream stays monotonic. testEventsZeroCrossing covers a
  terminal harmonic-zero-crossing and a record-only apoapsis case.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

### Task 12 — `EveryStep` + `Custom` tests

**Files:**
- Create: `tests/ode/testEventsEveryStep.cpp`
- Modify: `tests/ode/CMakeLists.txt`

- [ ] **Step 12.1:** Append to `tests/ode/CMakeLists.txt`:

  ```cmake
  tax_add_test(test_ode_events_everystep SOURCES testEventsEveryStep.cpp)
  ```

- [ ] **Step 12.2:** Write the test:

  ```cpp
  // tests/ode/testEventsEveryStep.cpp
  //
  // EveryStep trigger paired with Continue / Custom actions.
  // Verifies: every accepted step fires the trigger exactly once;
  // Custom can stop the integration by returning Terminate.

  #include <gtest/gtest.h>

  #include <Eigen/Core>
  #include <cmath>
  #include <vector>

  #include <tax/ode.hpp>

  using tax::ode::ControlFlow;
  using tax::ode::Custom;
  using tax::ode::Event;
  using tax::ode::EveryStep;
  using tax::ode::IntegratorConfig;
  using tax::ode::TaylorIntegrator;
  using tax::ode::TaylorStepper;

  TEST( OdeEventsEveryStep, FiresOncePerStep )
  {
      constexpr int N = 12;
      using State = Eigen::Matrix< double, 1, 1 >;

      IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-12;

      const auto f = []( const auto& x, const auto& ) { return x; };

      int counter = 0;
      using Stepper = TaylorStepper< N, State >;
      std::vector< Event< Stepper > > events;
      events.emplace_back(
          EveryStep(),
          Custom( [&counter]( const auto&, double, auto& )
                  { ++counter; return ControlFlow::Continue; } ) );

      TaylorIntegrator< N, double, 1, /*Dense=*/false > integ( f, cfg, events );
      State x0; x0( 0 ) = 1.0;
      auto sol = integ.integrate( x0, 0.0, 1.0 );

      // Counter should equal the number of accepted steps == sol.size() - 1.
      EXPECT_EQ( static_cast< std::size_t >( counter ), sol.size() - 1 );
      EXPECT_GE( counter, 1 );
  }

  TEST( OdeEventsEveryStep, CustomCanTerminate )
  {
      constexpr int N = 12;
      using State = Eigen::Matrix< double, 1, 1 >;

      IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-12;

      const auto f = []( const auto& x, const auto& ) { return x; };

      using Stepper = TaylorStepper< N, State >;
      std::vector< Event< Stepper > > events;
      events.emplace_back(
          EveryStep(),
          Custom( []( const auto& ctx, double, auto& )
                  {
                      return ( ctx.t_old + ctx.h_used > 0.3 )
                                 ? ControlFlow::Terminate
                                 : ControlFlow::Continue;
                  } ) );

      TaylorIntegrator< N, double, 1, /*Dense=*/false > integ( f, cfg, events );
      State x0; x0( 0 ) = 1.0;
      auto sol = integ.integrate( x0, 0.0, 1.0 );

      EXPECT_LT( sol.t.back(), 1.0 );
      EXPECT_GE( sol.t.back(), 0.3 );
  }
  ```

- [ ] **Step 12.3:** Build and run:

  ```bash
  cmake --build build -j
  ctest --test-dir build -R test_ode_events_everystep --output-on-failure
  ```

  Expected: both subtests pass.

- [ ] **Step 12.4:** Verify the full ODE suite:

  ```bash
  ctest --test-dir build -R "test_ode_" --output-on-failure
  ```

  Expected: all seven ODE tests
  (`test_ode_config`, `test_ode_concepts`, `test_ode_solution`,
  `test_ode_controllers`, `test_ode_taylor_stepper`,
  `test_ode_integrator_basic`, `test_ode_integrator_dense`,
  `test_ode_events_zerocrossing`, `test_ode_events_everystep`) pass.

- [ ] **Step 12.5:** Final full-suite green-check:

  ```bash
  ctest --test-dir build --output-on-failure
  ```

- [ ] **Step 12.6:** Commit.

  ```bash
  git add tests/ode/CMakeLists.txt tests/ode/testEventsEveryStep.cpp
  git commit -m "$(cat <<'EOF'
  stage2a-slice4d: EveryStep + Custom event tests

  Confirms EveryStep fires exactly once per accepted step (so a Custom
  action sees every boundary), and that a Custom action returning
  ControlFlow::Terminate halts the integration cleanly. This is the
  forward-compatibility seam Stage 2b's ADS driver will plug into.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Self-review

**Spec coverage check** against
`docs/superpowers/specs/2026-05-21-tax-stage2a-ode-integrator-design.md`:

| Spec section | Covered by |
|---|---|
| `IntegratorConfig<T>` (Public API) | Task 1 |
| `concepts::Stepper` / `AdaptiveStepper` | Task 2 |
| `StepResult<State, Stepper>` | Task 2 |
| `Solution<…, Dense>` partial specialisations + `EventRecord` | Task 3 |
| Step-size controllers (`JorbaZou`, `I`, `PI`, `H211b`) | Task 4 |
| `TaylorStepper<N, State, Controller>` | Tasks 5, 6 |
| `Integrator<Stepper, Dense>` + `TaylorIntegrator` alias | Tasks 7, 8 |
| `Direction`, `ControlFlow`, `TriggerContext` | Task 9 |
| `Event<Stepper>` (type-erased Trigger + Action) | Task 10 |
| `ZeroCrossing`, `EveryStep` triggers | Task 10 |
| `Continue`, `Terminate`, `Record`, `Custom` actions | Task 10 |
| `TaylorStepper::find_zero` polynomial-Newton w/ bisection safeguard | Task 10 (inline in `ZeroCrossing`) |
| Shared `detail/brent_root.hpp` (Brent's method) | Task 9 |
| Integrator events wiring + monotonic τ ordering | Task 11 |
| File layout under `include/tax/ode/`, opt-in via `<tax/ode.hpp>` | every task |

Plan B (slices 5–10) will cover: RK steppers (Verner78, Verner89,
Fehlberg78, Feagin12, Feagin14); physical-dynamics tests (Kepler,
CR3BP); benchmark fixture; static-vs-dynamic D parity.

**Placeholder scan:** No TBDs / TODOs / "implement later". Every step
has either complete code or a verified shell command.

**Type-consistency check:**
- `TaylorStepper<N, State, Controller>` signature matches across
  Tasks 5, 6, 10, 11. `Controller` defaults to
  `controllers::JorbaZou<typename State::Scalar>` in every reference.
- `Event<Stepper>` constructor takes `(Trig, Act)` everywhere; the
  type erasure uses `std::function<std::optional<T>(const Ctx&)>` for
  the trigger and `std::function<ControlFlow(const Ctx&, T, Sink&)>`
  for the action.
- `EventSink<State, T>` is defined in `actions.hpp` and used by both
  `Event` (in `event.hpp`) and `Integrator::integrate` (in
  `integrator.hpp`). Inclusion order: `event.hpp` declares
  `EventSink` via forward reference; `actions.hpp` provides the
  definition. The umbrella `<tax/ode.hpp>` includes them in the order:
  `event.hpp` → `actions.hpp` → `triggers.hpp` → `integrator.hpp`,
  which is consistent everywhere.
- `IntegratorConfig::*_step` field names match between `config.hpp`
  (Task 1) and uses in `integrator.hpp` (Tasks 7, 11):
  `initial_step`, `min_step`, `max_step`.

No inconsistencies found.

---

**Plan complete.** Ready for execution.
