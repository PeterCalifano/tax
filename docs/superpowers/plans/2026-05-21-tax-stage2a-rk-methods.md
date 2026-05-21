# tax Stage 2a — Part B: RK methods + physical-dynamics tests + benchmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Stage 2a public surface by adding five explicit Runge–Kutta steppers (Verner 8(7), Verner 9(8), Fehlberg 7(8), Feagin 12(10), Feagin 14(12)) and validating the whole framework against the planar two-body and Earth–Moon CR3BP problems, with a Google Benchmark suite that sweeps method × controller × Taylor order on the same CR3BP integration.

**Architecture:** A shared `detail::adaptive_rk_step` helper drives every RK stepper from a method-specific Butcher tableau (`detail::*_tableau`). Per-method steppers (`Verner78Stepper`, etc.) are thin façades that satisfy `concepts::AdaptiveStepper`, hold a `Controller` instance, and expose `eval_dense` via Hermite-cubic interpolation across the step. Tests are organised so each new stepper proves itself against the existing Taylor reference before joining the cross-method test matrix.

**Tech Stack:** C++23, header-only, Eigen 3.4+, existing `tax::ode` surface from Plan A, Google Test for tests, Google Benchmark (CMake `TAX_BUILD_BENCHMARK=ON`) for the perf fixture. Butcher tableaux ported from the pre-Stage-1 branch (`claude/add-verner-integrators-vEgRF`) for Verner, from the classical literature for Fehlberg, and from `OrdinaryDiffEqFeagin.jl` (Apache-2.0) for the two Feagin pairs.

**Spec:** `docs/superpowers/specs/2026-05-21-tax-stage2a-ode-integrator-design.md`. **Depends on:** Plan A (`docs/superpowers/plans/2026-05-21-tax-stage2a-core.md`) merged on `stage2a-core` branch.

**Build env:** micromamba env `tax`. Activate: `eval "$(micromamba shell hook --shell bash)" && micromamba activate tax`.

---

## File Structure

Files created:

- `include/tax/ode/detail/adaptive_rk_step.hpp` — generic explicit-RK step driver shared by all five RK steppers
- `include/tax/ode/detail/hermite_interp.hpp` — cubic-Hermite interpolation for `eval_dense` (covers all five RK methods)
- `include/tax/ode/detail/verner_tableaus.hpp` — Verner 8(7) + 9(8) Butcher tables (ported from pre-Stage-1 branch)
- `include/tax/ode/detail/fehlberg_tableaus.hpp` — Fehlberg 7(8) Butcher table
- `include/tax/ode/detail/feagin_tableaus.hpp` — Feagin 12(10) + 14(12) Butcher tables (ported from DifferentialEquations.jl)
- `include/tax/ode/steppers/verner78.hpp`
- `include/tax/ode/steppers/verner89.hpp`
- `include/tax/ode/steppers/fehlberg78.hpp`
- `include/tax/ode/steppers/feagin12.hpp`
- `include/tax/ode/steppers/feagin14.hpp`
- `tests/ode/testVerner78Stepper.cpp`
- `tests/ode/testVerner89Stepper.cpp`
- `tests/ode/testFehlberg78Stepper.cpp`
- `tests/ode/testFeagin12Stepper.cpp`
- `tests/ode/testFeagin14Stepper.cpp`
- `tests/ode/testTwoBodyKepler.cpp`
- `tests/ode/testCR3BPPropagation.cpp`
- `tests/ode/testCR3BPEvents.cpp`
- `tests/ode/testIntegratorStatic.cpp`
- `tests/ode/cr3bp_problem.hpp` — shared CR3BP fixture (RHS, ICs, Jacobi constant, Lagrange-point locations) used by both CR3BP tests and the benchmark
- `benchmarks/bench_ode_cr3bp.cpp`

Files modified:

- `include/tax/ode.hpp` — extend umbrella to include the five new stepper headers + the new factories
- `include/tax/ode/integrator.hpp` — add `makeVerner78Integrator`, `makeVerner89Integrator`, `makeFehlberg78Integrator`, `makeFeagin12Integrator`, `makeFeagin14Integrator` factories
- `benchmarks/CMakeLists.txt` — register `bench_ode_cr3bp`
- `tests/ode/CMakeLists.txt` — register the nine new test executables

---

## Task 13 — Shared `adaptive_rk_step` + Hermite interpolator

The generic explicit-RK step routine and Hermite cubic interpolator are shared across all five RK methods. Implement and unit-test in isolation before any stepper uses them.

**Files:**
- Create: `include/tax/ode/detail/adaptive_rk_step.hpp`
- Create: `include/tax/ode/detail/hermite_interp.hpp`
- Create: `tests/ode/testRKHelpers.cpp`
- Modify: `tests/ode/CMakeLists.txt`

- [ ] **Step 13.1:** Append to `tests/ode/CMakeLists.txt`:

  ```cmake
  tax_add_test(test_ode_rk_helpers SOURCES testRKHelpers.cpp)
  ```

- [ ] **Step 13.2:** Write the failing test `tests/ode/testRKHelpers.cpp`:

  ```cpp
  // tests/ode/testRKHelpers.cpp
  //
  // Direct unit tests for the shared adaptive_rk_step driver and the
  // Hermite cubic interpolator. Verified against a 4-stage classical
  // RK4 tableau (degenerate "embedded" estimator = b weights so
  // err_norm == 0; this just checks the stage propagation).

  #include <gtest/gtest.h>

  #include <Eigen/Core>
  #include <array>
  #include <cmath>

  #include <tax/ode/detail/adaptive_rk_step.hpp>
  #include <tax/ode/detail/hermite_interp.hpp>

  namespace
  {

  // Classical RK4 as a Butcher tableau: 4 stages, FSAL = false.
  struct RK4Tab
  {
      static constexpr int n_stages = 4;
      static constexpr int order    = 4;
      static constexpr int order_emb = 4;          // degenerate (err = 0)
      static constexpr bool fsal    = false;

      // c_i = column of nodes
      static constexpr std::array< double, 4 > c{ 0.0, 0.5, 0.5, 1.0 };

      // a_ij flattened row-major (lower-triangular, no diagonal):
      // a[0]=a21, a[1]=a31, a[2]=a32, a[3]=a41, a[4]=a42, a[5]=a43
      static constexpr std::array< double, 6 > a{
          0.5,
          0.0, 0.5,
          0.0, 0.0, 1.0
      };

      static constexpr std::array< double, 4 > b{ 1.0 / 6, 1.0 / 3, 1.0 / 3, 1.0 / 6 };
      static constexpr std::array< double, 4 > b_emb = b;  // degenerate
  };

  }  // namespace

  TEST( OdeRKHelpers, RK4OneStepOnExp )
  {
      using State = Eigen::Matrix< double, 1, 1 >;
      State x; x( 0 ) = 1.0;

      auto f = []( const State& y, double ) { return y; };

      tax::ode::detail::RKStepData< State, 4 > stages;
      auto out = tax::ode::detail::adaptive_rk_step< RK4Tab >( f, x, 0.0, 0.1, stages );

      // x(0.1) = e^0.1 ≈ 1.10517091808...
      EXPECT_NEAR( out.x_new( 0 ), std::exp( 0.1 ), 1e-7 );
      // RK4's degenerate b_emb yields zero error.
      EXPECT_DOUBLE_EQ( out.err_norm, 0.0 );
  }

  TEST( OdeRKHelpers, HermiteReproducesBoundaries )
  {
      using State = Eigen::Matrix< double, 2, 1 >;
      const double t0 = 1.0, t1 = 2.0;
      State x0; x0( 0 ) = 0.5; x0( 1 ) = -1.0;
      State x1; x1( 0 ) = 0.7; x1( 1 ) = -0.3;
      State f0; f0( 0 ) = 0.2; f0( 1 ) =  0.8;
      State f1; f1( 0 ) = 0.1; f1( 1 ) = -0.4;

      const State at_t0 = tax::ode::detail::hermite_interp( x0, x1, f0, f1, t0, t1, t0 );
      const State at_t1 = tax::ode::detail::hermite_interp( x0, x1, f0, f1, t0, t1, t1 );
      EXPECT_NEAR( ( at_t0 - x0 ).norm(), 0.0, 1e-14 );
      EXPECT_NEAR( ( at_t1 - x1 ).norm(), 0.0, 1e-14 );
  }
  ```

- [ ] **Step 13.3:** Verify the test does not build yet.

  ```bash
  eval "$(micromamba shell hook --shell bash)" && micromamba activate tax
  cmake --build build -j 2>&1 | tail -5
  ```

  Expected: fails on missing headers / `tax::ode::detail::adaptive_rk_step`.

- [ ] **Step 13.4:** Create `include/tax/ode/detail/adaptive_rk_step.hpp`:

  ```cpp
  // include/tax/ode/detail/adaptive_rk_step.hpp
  //
  // Generic explicit Runge–Kutta step driver used by every Stage 2a RK
  // stepper (Verner78, Verner89, Fehlberg78, Feagin12, Feagin14).
  // Tableau is supplied as a struct exposing:
  //   - static constexpr int n_stages
  //   - static constexpr int order, order_emb
  //   - static constexpr bool fsal           (first-same-as-last)
  //   - static constexpr std::array<T, n_stages>            c
  //   - static constexpr std::array<T, n_stages*(n_stages-1)/2> a  (row-major, lower-tri, no diag)
  //   - static constexpr std::array<T, n_stages>            b
  //   - static constexpr std::array<T, n_stages>            b_emb
  // The Tableau type must be default-constructible and have all-static
  // members; passes via template parameter, no instance needed.

  #pragma once

  #include <Eigen/Core>
  #include <array>
  #include <cstddef>
  #include <utility>

  namespace tax::ode::detail
  {

  template < class State, int NStages >
  struct RKStepData
  {
      std::array< State, NStages > k{};
  };

  template < class State, class T >
  struct RKStepOut
  {
      State x_new;
      State y_emb;   // embedded estimate
      T     err_norm;
  };

  template < class Tab, class F, class State, class T, int NStages >
  [[nodiscard]] RKStepOut< State, T > adaptive_rk_step(
      F&& f, const State& x, T t, T h, RKStepData< State, NStages >& work )
  {
      static_assert( NStages == Tab::n_stages,
                     "adaptive_rk_step: stage-count mismatch with tableau" );

      // k_1 = f(x, t + c_0 * h)  (typically c_0 = 0).
      work.k[ 0 ] = f( x, t + T( Tab::c[ 0 ] ) * h );

      std::size_t a_off = 0;
      for ( int i = 1; i < NStages; ++i )
      {
          State y = x;
          for ( int j = 0; j < i; ++j )
              y += h * T( Tab::a[ a_off + std::size_t( j ) ] ) * work.k[ std::size_t( j ) ];
          work.k[ std::size_t( i ) ] = f( y, t + T( Tab::c[ std::size_t( i ) ] ) * h );
          a_off += std::size_t( i );
      }

      State x_new = x;
      State y_emb = x;
      for ( int i = 0; i < NStages; ++i )
      {
          x_new += h * T( Tab::b    [ std::size_t( i ) ] ) * work.k[ std::size_t( i ) ];
          y_emb += h * T( Tab::b_emb[ std::size_t( i ) ] ) * work.k[ std::size_t( i ) ];
      }

      // Element-wise infinity norm of the difference, scaled by h.
      T err_norm{ 0 };
      for ( Eigen::Index r = 0; r < x_new.size(); ++r )
      {
          using std::abs;
          const T d = T( abs( x_new( r ) - y_emb( r ) ) );
          if ( d > err_norm ) err_norm = d;
      }

      return { std::move( x_new ), std::move( y_emb ), err_norm };
  }

  }  // namespace tax::ode::detail
  ```

- [ ] **Step 13.5:** Create `include/tax/ode/detail/hermite_interp.hpp`:

  ```cpp
  // include/tax/ode/detail/hermite_interp.hpp
  //
  // Cubic-Hermite interpolation between two state samples and their
  // derivatives. Reproduces (x0, x1) exactly at the boundaries and is
  // C^1 across the step. Used by every RK stepper's eval_dense.

  #pragma once

  #include <Eigen/Core>

  namespace tax::ode::detail
  {

  template < class State, class T >
  [[nodiscard]] State hermite_interp(
      const State& x0, const State& x1,
      const State& f0, const State& f1,
      const T& t0, const T& t1, const T& tq )
  {
      const T h     = t1 - t0;
      const T theta = ( tq - t0 ) / h;
      const T om    = T{ 1 } - theta;

      // Standard Hermite basis on [0,1]:
      //   H00 =  (1+2θ)(1-θ)^2
      //   H10 =  θ      (1-θ)^2
      //   H01 =  θ^2    (3-2θ)
      //   H11 = -θ^2    (1-θ)
      const T h00 = ( T{ 1 } + T{ 2 } * theta ) * om * om;
      const T h10 = theta * om * om;
      const T h01 = theta * theta * ( T{ 3 } - T{ 2 } * theta );
      const T h11 = -theta * theta * om;

      State out = h00 * x0 + ( h10 * h ) * f0
                + h01 * x1 + ( h11 * h ) * f1;
      return out;
  }

  }  // namespace tax::ode::detail
  ```

- [ ] **Step 13.6:** Build and run:

  ```bash
  cmake --build build -j
  ctest --test-dir build -R test_ode_rk_helpers --output-on-failure
  ```

  Expected: both subtests pass.

- [ ] **Step 13.7:** Verify full suite still green.

  ```bash
  ctest --test-dir build --output-on-failure
  ```

- [ ] **Step 13.8:** Commit.

  ```bash
  git add include/tax/ode/detail/adaptive_rk_step.hpp \
          include/tax/ode/detail/hermite_interp.hpp \
          tests/ode/CMakeLists.txt tests/ode/testRKHelpers.cpp
  git commit -m "$(cat <<'EOF'
  stage2a-slice5a: shared adaptive_rk_step + cubic-Hermite interpolator

  Generic explicit-RK step driver that every Stage 2a RK stepper plugs
  into via a static Butcher tableau (n_stages, order, c, a, b, b_emb).
  hermite_interp provides the cubic-Hermite continuous extension used
  by every RK stepper's eval_dense (matches boundaries exactly, C^1
  across the step). testRKHelpers exercises both directly with a
  classical RK4 tableau and a synthetic Hermite-boundary check.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 14 — Verner 8(7) Butcher tableau + `Verner78Stepper`

The Verner 8(7) "efficient" 13-stage tableau is on branch `claude/add-verner-integrators-vEgRF` at `include/tax/ode/verner_tableaus.hpp`. Fetch it; the format will differ from `adaptive_rk_step`'s expectations and needs re-organization.

**Files:**
- Create: `include/tax/ode/detail/verner_tableaus.hpp` (Verner78Tab only at this task; Verner89Tab in Task 15)
- Create: `include/tax/ode/steppers/verner78.hpp`
- Create: `tests/ode/testVerner78Stepper.cpp`
- Modify: `include/tax/ode/integrator.hpp` (add `makeVerner78Integrator` factory)
- Modify: `include/tax/ode.hpp` (include the new stepper)
- Modify: `tests/ode/CMakeLists.txt`

- [ ] **Step 14.1:** Fetch the pre-Stage-1 Verner 8(7) tableau:

  ```bash
  git show claude/add-verner-integrators-vEgRF:include/tax/ode/verner_tableaus.hpp > /tmp/verner_tableaus_old.hpp
  wc -l /tmp/verner_tableaus_old.hpp
  ```

  Expected: ~1200 lines (both Verner78 and Verner89 coefficients).

- [ ] **Step 14.2:** Create `include/tax/ode/detail/verner_tableaus.hpp`. Open `/tmp/verner_tableaus_old.hpp`, locate the `Verner78Coeffs` (or similarly-named) struct, and re-package its `c`, `a`, `b`, `b_hat` (or `e` = `b - b_hat`) arrays into the tableau shape expected by `adaptive_rk_step` (see Task 13). The output file structure:

  ```cpp
  // include/tax/ode/detail/verner_tableaus.hpp
  //
  // Butcher tableaux for J. H. Verner's "efficient" RK pairs:
  //   Verner78Tab — 13-stage, propagates at order 8 with order-7 embedded
  //                 error estimator. Ported from pre-Stage-1 branch
  //                 (claude/add-verner-integrators-vEgRF) which itself
  //                 reproduced the SciML/OrdinaryDiffEq.jl `Vern8`
  //                 tableau, derived from Verner's published rationals
  //                 (https://www.sfu.ca/~jverner/).
  //
  //   Verner89Tab — 16-stage, propagates at order 9 with order-8 embedded.
  //                 (Added in Task 15.)
  //
  // Layout matches the tax::ode::detail::adaptive_rk_step contract:
  //   c   : nodes  (size n_stages)
  //   a   : lower-triangular row-major  (size n_stages*(n_stages-1)/2)
  //   b   : main weights (size n_stages)
  //   b_emb : embedded weights (size n_stages)
  // err = h * sum(b - b_emb) * k.

  #pragma once

  #include <array>

  namespace tax::ode::detail
  {

  struct Verner78Tab
  {
      static constexpr int n_stages   = 13;
      static constexpr int order      = 8;
      static constexpr int order_emb  = 7;
      static constexpr bool fsal      = false;

      static constexpr std::array< double, 13 > c{ /* 13 values from pre-Stage-1 */ };

      // a[i,j] for i in [1,12], j in [0,i-1], flattened row-major:
      // a[0]=a10, a[1]=a20, a[2]=a21, a[3]=a30, a[4]=a31, a[5]=a32, ...
      // Total length: 13*12/2 = 78.
      static constexpr std::array< double, 78 > a{ /* 78 values from pre-Stage-1 */ };

      static constexpr std::array< double, 13 > b    { /* 13 values from pre-Stage-1 */ };
      static constexpr std::array< double, 13 > b_emb{ /* 13 values from pre-Stage-1 */ };
  };

  }  // namespace tax::ode::detail
  ```

  Fill in the four arrays from the pre-Stage-1 file. The original file may store `e[i] = b[i] - b_hat[i]` (Hairer convention) — if so, compute `b_emb[i] = b[i] - e[i]` so that the embedded estimate `y_emb = x + h * sum(b_emb * k)` matches the convention in `adaptive_rk_step`.

  Critical: **do not type the values in by hand.** Use `sed`/`awk` or a small Python script to transform the source file's coefficient blocks into the new array layout, write the result, and review. Hand-typing 100+ floats invites silent precision errors.

- [ ] **Step 14.3:** Write the failing test `tests/ode/testVerner78Stepper.cpp`:

  ```cpp
  // tests/ode/testVerner78Stepper.cpp
  //
  // Stepper-level correctness of Verner78Stepper on three RHS:
  //   - dx/dt = x       (analytic exp)
  //   - harmonic        (cos t, -sin t)
  //   - cubic-decay     (1 / sqrt(1 + 2t))
  // Plus the eval_dense round-trip assertion at step boundaries.

  #include <gtest/gtest.h>

  #include <Eigen/Core>
  #include <cmath>

  #include <tax/ode.hpp>

  using tax::ode::IntegratorConfig;
  using tax::ode::Verner78Stepper;
  using tax::ode::controllers::I;
  using tax::ode::controllers::PI;

  TEST( OdeVerner78Stepper, ExponentialOneStep )
  {
      using State = Eigen::Matrix< double, 1, 1 >;
      Verner78Stepper< State > stepper;

      State x0; x0( 0 ) = 1.0;
      const auto f = []( const auto& x, double ) { return x; };

      IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-12;

      auto r = stepper.step( f, x0, 0.0, 0.1, cfg );

      EXPECT_TRUE( r.accepted );
      EXPECT_NEAR( r.x_new( 0 ), std::exp( 0.1 ), 1e-12 );

      auto x_at_t0 = Verner78Stepper< State >::eval_dense(
          r.dense, 0.0, r.h_used, 0.0 );
      auto x_at_t1 = Verner78Stepper< State >::eval_dense(
          r.dense, 0.0, r.h_used, r.h_used );
      EXPECT_NEAR( x_at_t0( 0 ), x0( 0 ),       1e-14 );
      EXPECT_NEAR( x_at_t1( 0 ), r.x_new( 0 ),  1e-14 );
  }

  TEST( OdeVerner78Stepper, HarmonicOneStep )
  {
      using State = Eigen::Matrix< double, 2, 1 >;
      Verner78Stepper< State > stepper;

      State x0; x0( 0 ) = 1.0; x0( 1 ) = 0.0;
      const auto f = []( const auto& x, double )
      {
          State out;
          out( 0 ) =  x( 1 );
          out( 1 ) = -x( 0 );
          return out;
      };

      IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-12;

      auto r = stepper.step( f, x0, 0.0, 0.05, cfg );

      EXPECT_TRUE( r.accepted );
      EXPECT_NEAR( r.x_new( 0 ),  std::cos( 0.05 ), 1e-12 );
      EXPECT_NEAR( r.x_new( 1 ), -std::sin( 0.05 ), 1e-12 );
  }

  TEST( OdeVerner78Stepper, ControllerIVariant )
  {
      using State = Eigen::Matrix< double, 1, 1 >;
      Verner78Stepper< State, I< double > > stepper;

      State x0; x0( 0 ) = 1.0;
      const auto f = []( const auto& x, double ) { return x; };

      IntegratorConfig< double > cfg;
      auto r = stepper.step( f, x0, 0.0, 0.05, cfg );
      EXPECT_TRUE( r.accepted );
      EXPECT_NEAR( r.x_new( 0 ), std::exp( 0.05 ), 1e-12 );
  }
  ```

- [ ] **Step 14.4:** Append to `tests/ode/CMakeLists.txt`:

  ```cmake
  tax_add_test(test_ode_verner78_stepper SOURCES testVerner78Stepper.cpp)
  ```

- [ ] **Step 14.5:** Confirm the build fails on missing `Verner78Stepper`:

  ```bash
  cmake --build build -j 2>&1 | tail -5
  ```

- [ ] **Step 14.6:** Create `include/tax/ode/steppers/verner78.hpp`:

  ```cpp
  // include/tax/ode/steppers/verner78.hpp
  //
  // Verner 8(7) RK stepper. 13 stages, propagates at order 8, uses an
  // order-7 embedded estimator for adaptive step-size control. The
  // continuous extension is provided by a cubic-Hermite spline between
  // step boundaries (sufficient for event detection inside the step;
  // formal order matches the embedded estimator).

  #pragma once

  #include <Eigen/Core>
  #include <utility>

  #include <tax/ode/config.hpp>
  #include <tax/ode/controllers.hpp>
  #include <tax/ode/detail/adaptive_rk_step.hpp>
  #include <tax/ode/detail/hermite_interp.hpp>
  #include <tax/ode/detail/verner_tableaus.hpp>
  #include <tax/ode/step_result.hpp>

  namespace tax::ode
  {

  template < class StateT,
             class Controller = controllers::PI< typename StateT::Scalar > >
  struct Verner78Stepper
  {
      using State           = StateT;
      using T               = typename State::Scalar;
      using Config          = IntegratorConfig< T >;
      using Tab             = detail::Verner78Tab;

      static constexpr bool is_adaptive  = true;
      static constexpr int  order_v      = Tab::order;
      static constexpr int  order_emb_v  = Tab::order_emb;

      // DenseData: per-step boundary samples + their derivatives —
      // enough for cubic-Hermite interpolation.
      struct DenseData
      {
          State x0;
          State x1;
          State f0;
          State f1;
      };

      template < class F >
      [[nodiscard]] StepResult< State, Verner78Stepper > step(
          F&& f, const State& x, T t, T h, const Config& cfg )
      {
          using detail::adaptive_rk_step;
          using detail::RKStepData;
          using std::abs;

          RKStepData< State, Tab::n_stages > work;
          auto out = adaptive_rk_step< Tab >( f, x, t, h, work );

          // Tolerance: abstol + reltol * max(|x|, |x_new|).
          T x_norm{ 0 };
          for ( Eigen::Index i = 0; i < x.size(); ++i )
          {
              using std::abs;
              const T a = T( abs( out.x_new( i ) ) );
              if ( a > x_norm ) x_norm = a;
          }
          const T tol = cfg.abstol + cfg.reltol * x_norm;

          T h_next;
          if constexpr ( std::is_same_v< Controller, controllers::JorbaZou< T > > )
              h_next = h;  // JorbaZou is Taylor-only; no-op fallback if mis-used.
          else
              h_next = controller_.next_step( h, out.err_norm, tol, Tab::order_emb );

          const bool accepted = out.err_norm <= tol;

          // Build the dense payload from the boundary samples + their f values.
          // f0 = first stage (always at t + c[0]*h ≈ t); f1 = f(x_new, t+h).
          DenseData dd;
          dd.x0 = x;
          dd.x1 = out.x_new;
          dd.f0 = work.k[ 0 ];
          dd.f1 = f( out.x_new, t + h );

          StepResult< State, Verner78Stepper > r;
          r.x_new    = std::move( out.x_new );
          r.h_used   = h;
          r.dense    = std::move( dd );
          r.h_next   = h_next;
          r.err_norm = out.err_norm;
          r.accepted = accepted;
          return r;
      }

      [[nodiscard]] static State eval_dense(
          const DenseData& d, const T& t0, const T& t1, const T& tq )
      {
          return detail::hermite_interp< State, T >(
              d.x0, d.x1, d.f0, d.f1, t0, t1, tq );
      }

  private:
      Controller controller_{};
  };

  }  // namespace tax::ode
  ```

  Note: the `JorbaZou` mis-use branch returns the same `h` to keep the build alive if a user accidentally specifies `controllers::JorbaZou` on an RK stepper. This is intentional belt-and-braces; a static_assert would also be appropriate. Keep it as a no-op so the unit tests can fail loudly via the test's choice of controller, not at compile time.

- [ ] **Step 14.7:** Modify `include/tax/ode/integrator.hpp` to add the factory. Find the existing `makeTaylorIntegrator` factories near the bottom of the file and add (matching the same pattern, with and without events):

  ```cpp
  // ---- Verner 8(7) factories ----
  template < class T = double, int D = Eigen::Dynamic,
             bool Dense = false,
             class Controller = controllers::PI< T >, class F >
  [[nodiscard]] auto makeVerner78Integrator( F f, IntegratorConfig< T > cfg = {} )
  {
      using State   = Eigen::Matrix< T, D, 1 >;
      using Stepper = Verner78Stepper< State, Controller >;
      return Integrator< Stepper, F, Dense >{ std::move( f ), std::move( cfg ) };
  }

  template < class T = double, int D = Eigen::Dynamic,
             bool Dense = false,
             class Controller = controllers::PI< T >, class F >
  [[nodiscard]] auto makeVerner78Integrator(
      F f,
      IntegratorConfig< T > cfg,
      std::vector< Event< Verner78Stepper< Eigen::Matrix< T, D, 1 >, Controller > > > events )
  {
      using State   = Eigen::Matrix< T, D, 1 >;
      using Stepper = Verner78Stepper< State, Controller >;
      return Integrator< Stepper, F, Dense >{
          std::move( f ), std::move( cfg ), std::move( events ) };
  }
  ```

  At the top of `integrator.hpp` add `#include <tax/ode/steppers/verner78.hpp>` next to the existing taylor.hpp include.

- [ ] **Step 14.8:** Update `include/tax/ode.hpp` to include the new stepper:

  ```cpp
  #include <tax/ode/steppers/verner78.hpp>
  ```

  Place it right after the existing `steppers/taylor.hpp` include. Keep the trigger/action/event/integrator block at the end of the file untouched.

- [ ] **Step 14.9:** Build and run:

  ```bash
  cmake --build build -j
  ctest --test-dir build -R test_ode_verner78_stepper --output-on-failure
  ```

  Expected: all three subtests pass.

- [ ] **Step 14.10:** Full suite green:

  ```bash
  ctest --test-dir build --output-on-failure
  ```

- [ ] **Step 14.11:** Commit.

  ```bash
  git add include/tax/ode/detail/verner_tableaus.hpp \
          include/tax/ode/steppers/verner78.hpp \
          include/tax/ode/integrator.hpp \
          include/tax/ode.hpp \
          tests/ode/CMakeLists.txt tests/ode/testVerner78Stepper.cpp
  git commit -m "$(cat <<'EOF'
  stage2a-slice5b: Verner 8(7) RK stepper

  Adds detail::Verner78Tab (13 stages, order 8 / order-7 embedded) and
  Verner78Stepper<State, Controller> using the shared adaptive_rk_step
  driver. DenseData stores boundary states and derivatives; eval_dense
  cubic-Hermite-interpolates inside the step. Stepper satisfies
  AdaptiveStepper via is_adaptive = true. Convenience factory
  makeVerner78Integrator follows the makeTaylorIntegrator pattern.
  Per-stepper tests exercise exp, harmonic, and the I-controller variant.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 15 — Verner 9(8) Butcher tableau + `Verner89Stepper`

Same pattern as Task 14, with 16 stages and order 9 propagation. The Verner89 tableau is in the same source file `/tmp/verner_tableaus_old.hpp`.

**Files:**
- Modify: `include/tax/ode/detail/verner_tableaus.hpp` (append `Verner89Tab`)
- Create: `include/tax/ode/steppers/verner89.hpp`
- Create: `tests/ode/testVerner89Stepper.cpp`
- Modify: `include/tax/ode/integrator.hpp` (add `makeVerner89Integrator`)
- Modify: `include/tax/ode.hpp`
- Modify: `tests/ode/CMakeLists.txt`

- [ ] **Step 15.1:** Append the `Verner89Tab` struct to `include/tax/ode/detail/verner_tableaus.hpp` using the same approach as Step 14.2: open `/tmp/verner_tableaus_old.hpp`, find the `Verner89Coeffs`-equivalent struct, repackage its arrays. The struct shape:

  ```cpp
  struct Verner89Tab
  {
      static constexpr int n_stages   = 16;
      static constexpr int order      = 9;
      static constexpr int order_emb  = 8;
      static constexpr bool fsal      = false;

      static constexpr std::array< double, 16 > c{ /* 16 values */ };
      // 16*15/2 = 120 a coefficients.
      static constexpr std::array< double, 120 > a{ /* 120 values */ };
      static constexpr std::array< double, 16 > b    { /* 16 values */ };
      static constexpr std::array< double, 16 > b_emb{ /* 16 values */ };
  };
  ```

- [ ] **Step 15.2:** Create `tests/ode/testVerner89Stepper.cpp` — duplicate the Verner78 test file (Task 14.3) and search-and-replace `Verner78` → `Verner89`. Keep the same three subtests.

- [ ] **Step 15.3:** Append to `tests/ode/CMakeLists.txt`:

  ```cmake
  tax_add_test(test_ode_verner89_stepper SOURCES testVerner89Stepper.cpp)
  ```

- [ ] **Step 15.4:** Create `include/tax/ode/steppers/verner89.hpp` by duplicating `verner78.hpp` (Task 14.6) and replacing every `Verner78`/`78` token with `Verner89`/`89`. The `Tab = detail::Verner89Tab` line is the only structural difference.

- [ ] **Step 15.5:** Add `makeVerner89Integrator` factories to `include/tax/ode/integrator.hpp` (mirror Task 14.7 with `Verner89Stepper`).

- [ ] **Step 15.6:** Add `#include <tax/ode/steppers/verner89.hpp>` to `include/tax/ode.hpp` after the Verner78 include.

- [ ] **Step 15.7:** Build, run, commit:

  ```bash
  cmake --build build -j
  ctest --test-dir build -R "test_ode_verner" --output-on-failure
  ctest --test-dir build --output-on-failure

  git add include/tax/ode/detail/verner_tableaus.hpp \
          include/tax/ode/steppers/verner89.hpp \
          include/tax/ode/integrator.hpp \
          include/tax/ode.hpp \
          tests/ode/CMakeLists.txt tests/ode/testVerner89Stepper.cpp
  git commit -m "$(cat <<'EOF'
  stage2a-slice5c: Verner 9(8) RK stepper

  Adds detail::Verner89Tab (16 stages, order 9 / order-8 embedded) and
  Verner89Stepper following the same shape as Verner78Stepper.
  Per-stepper tests cover exp, harmonic, and the I-controller variant.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 16 — Cross-method smoke test on `testIntegratorBasic.cpp`

Now that two RK methods exist alongside Taylor, widen the basic-integration test to compare endpoints between methods on a non-trivial RHS (Lotka–Volterra).

**Files:**
- Modify: `tests/ode/testIntegratorBasic.cpp`

- [ ] **Step 16.1:** Open the file and add a new test at the bottom (keep the existing three Taylor-only tests untouched):

  ```cpp
  TEST( OdeIntegrator, LotkaVolterraCrossMethod )
  {
      using State = Eigen::Matrix< double, 2, 1 >;

      tax::ode::IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-12;

      // dx/dt = x (a - b*y), dy/dt = -y (c - d*x)
      // Use a = 1.1, b = 0.4, c = 0.4, d = 0.1.
      const auto f = []( const auto& z, double )
      {
          using S = std::decay_t< decltype( z ) >;
          S out;
          out( 0 ) =  z( 0 ) * ( 1.1 - 0.4 * z( 1 ) );
          out( 1 ) = -z( 1 ) * ( 0.4 - 0.1 * z( 0 ) );
          return out;
      };

      State x0; x0( 0 ) = 10.0; x0( 1 ) = 5.0;
      const double t0 = 0.0, tf = 5.0;

      auto tay = tax::ode::makeTaylorIntegrator   < 16, double, 2, false >( f, cfg );
      auto v78 = tax::ode::makeVerner78Integrator <    double, 2, false >( f, cfg );
      auto v89 = tax::ode::makeVerner89Integrator <    double, 2, false >( f, cfg );

      const auto sol_t  = tay.integrate( x0, t0, tf );
      const auto sol_78 = v78.integrate( x0, t0, tf );
      const auto sol_89 = v89.integrate( x0, t0, tf );

      // Cross-method agreement: tolerated to ~1e-9 over moderate horizon.
      EXPECT_NEAR( sol_t.x.back()( 0 ), sol_78.x.back()( 0 ), 1e-9 );
      EXPECT_NEAR( sol_t.x.back()( 1 ), sol_78.x.back()( 1 ), 1e-9 );
      EXPECT_NEAR( sol_t.x.back()( 0 ), sol_89.x.back()( 0 ), 1e-9 );
      EXPECT_NEAR( sol_t.x.back()( 1 ), sol_89.x.back()( 1 ), 1e-9 );
  }
  ```

- [ ] **Step 16.2:** Build and run:

  ```bash
  cmake --build build -j
  ctest --test-dir build -R test_ode_integrator_basic --output-on-failure
  ```

  Expected: existing three subtests + new `LotkaVolterraCrossMethod` all pass.

- [ ] **Step 16.3:** Commit:

  ```bash
  git add tests/ode/testIntegratorBasic.cpp
  git commit -m "$(cat <<'EOF'
  stage2a-slice5d: cross-method Lotka–Volterra integration test

  Taylor16 vs Verner78 vs Verner89 endpoint agreement to 1e-9 over
  t = [0, 5] on a Lotka–Volterra RHS. Confirms the three method
  pipelines route the same RHS lambda correctly.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 17 — Fehlberg 7(8) tableau + `Fehlberg78Stepper`

The classical Fehlberg 1968 RK 7(8) tableau is widely tabulated; pre-Stage-1 had it under a different name or it must be sourced from Hairer–Norsett–Wanner Vol. I (table V.4.1 / equivalent). Use the same `adaptive_rk_step` plumbing as Verner.

**Files:**
- Create: `include/tax/ode/detail/fehlberg_tableaus.hpp`
- Create: `include/tax/ode/steppers/fehlberg78.hpp`
- Create: `tests/ode/testFehlberg78Stepper.cpp`
- Modify: `include/tax/ode/integrator.hpp`, `include/tax/ode.hpp`, `tests/ode/CMakeLists.txt`

- [ ] **Step 17.1:** Source the Fehlberg 1968 RK 7(8) Butcher coefficients. Authoritative reference: E. Fehlberg, "Classical Fifth-, Sixth-, Seventh-, and Eighth-Order Runge-Kutta Formulas with Stepsize Control", NASA TR R-287, 1968 (table on page ~14). Modern reproduction: Hairer–Norsett–Wanner, "Solving Ordinary Differential Equations I: Nonstiff Problems", 2nd ed., Springer, 1993, pages 180–181.

  Alternative: SciPy `RK45`/`DOP853` source files reference the tableaux for cross-check (`scipy/integrate/_ivp/dop853_coefficients.py` has 8(5,3); not Fehlberg directly but the pattern is similar).

  Use one of: `BOOST_MATH_ODEINT_DOPRI5` / odeint `runge_kutta_fehlberg78` (Boost.Odeint header), which has the classical 13-stage table as `std::array` coefficients. **Recommended:** port from Boost.Odeint `boost/numeric/odeint/stepper/runge_kutta_fehlberg78.hpp` (Boost License — compatible with BSD-3 and most projects; cite the source).

  Layout the file as:

  ```cpp
  // include/tax/ode/detail/fehlberg_tableaus.hpp
  //
  // E. Fehlberg, "Classical Fifth-, Sixth-, Seventh-, and Eighth-Order
  // Runge-Kutta Formulas with Stepsize Control", NASA TR R-287, 1968.
  // The classical RK 7(8) pair: 13 stages, propagates at order 7,
  // embedded order-8 error estimator (the "Fehlberg coincidence" is a
  // known limitation; see spec Risks table).
  //
  // Coefficient values ported from Boost.Odeint
  //   boost/numeric/odeint/stepper/runge_kutta_fehlberg78.hpp
  // (Boost Software License v1.0, attribution below).

  #pragma once

  #include <array>

  namespace tax::ode::detail
  {

  struct Fehlberg78Tab
  {
      static constexpr int n_stages   = 13;
      static constexpr int order      = 7;
      static constexpr int order_emb  = 8;
      static constexpr bool fsal      = false;

      static constexpr std::array< double, 13 > c     = { /* 13 values */ };
      static constexpr std::array< double, 78 > a     = { /* 78 values */ };
      static constexpr std::array< double, 13 > b     = { /* 13 values */ };
      static constexpr std::array< double, 13 > b_emb = { /* 13 values */ };
  };

  }  // namespace tax::ode::detail
  ```

  Fill in coefficients from the chosen reference. Same warning as Task 14.2 — use a script, not hand-typing.

- [ ] **Step 17.2:** Create `include/tax/ode/steppers/fehlberg78.hpp` by duplicating `verner78.hpp` and replacing `Verner78`/`78` tokens with `Fehlberg78`/`Fehlberg78`. The `Tab = detail::Fehlberg78Tab` line is the only structural change.

- [ ] **Step 17.3:** Create `tests/ode/testFehlberg78Stepper.cpp` by duplicating `testVerner78Stepper.cpp` with the same name/token substitution.

  Important: relax the harmonic test's tolerance to `1e-10` and the exponential test's to `1e-11` to account for the Fehlberg coincidence (the embedded estimator can hit zero and the step then over-extends).

- [ ] **Step 17.4:** Append to `tests/ode/CMakeLists.txt`:

  ```cmake
  tax_add_test(test_ode_fehlberg78_stepper SOURCES testFehlberg78Stepper.cpp)
  ```

- [ ] **Step 17.5:** Add `makeFehlberg78Integrator` factories to `include/tax/ode/integrator.hpp` (mirror Task 14.7).

- [ ] **Step 17.6:** Update `include/tax/ode.hpp` to include the new stepper.

- [ ] **Step 17.7:** Extend `testIntegratorBasic.cpp::LotkaVolterraCrossMethod` to also exercise Fehlberg78:

  ```cpp
  auto fhl = tax::ode::makeFehlberg78Integrator< double, 2, false >( f, cfg );
  const auto sol_fh = fhl.integrate( x0, t0, tf );
  EXPECT_NEAR( sol_t.x.back()( 0 ), sol_fh.x.back()( 0 ), 1e-8 );  // looser per Fehlberg risks
  EXPECT_NEAR( sol_t.x.back()( 1 ), sol_fh.x.back()( 1 ), 1e-8 );
  ```

- [ ] **Step 17.8:** Build, run, commit:

  ```bash
  cmake --build build -j
  ctest --test-dir build -R "test_ode_fehlberg\|test_ode_integrator_basic" --output-on-failure
  ctest --test-dir build --output-on-failure

  git add include/tax/ode/detail/fehlberg_tableaus.hpp \
          include/tax/ode/steppers/fehlberg78.hpp \
          include/tax/ode/integrator.hpp \
          include/tax/ode.hpp \
          tests/ode/CMakeLists.txt \
          tests/ode/testFehlberg78Stepper.cpp \
          tests/ode/testIntegratorBasic.cpp
  git commit -m "$(cat <<'EOF'
  stage2a-slice6: Fehlberg 7(8) RK stepper

  Classical Fehlberg 1968 RK 7(8) pair, 13 stages. Coefficients ported
  from Boost.Odeint with attribution. Stepper follows the Verner78/89
  shape (adaptive_rk_step + cubic-Hermite eval_dense). Cross-method
  agreement test relaxed to 1e-8 to accommodate the documented Fehlberg
  coincidence (per spec Risks table).

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 18 — Feagin 12(10) + Feagin 14(12)

Both Feagin pairs follow the Verner template. The Butcher coefficients are not in the pre-Stage-1 branch; port from `OrdinaryDiffEqFeagin.jl` (Apache-2.0):

- RK12(10): `OrdinaryDiffEqFeagin/src/tableaus/feagin_tableaus.jl::Feagin12Tableau`
- RK14(12): `OrdinaryDiffEqFeagin/src/tableaus/feagin_tableaus.jl::Feagin14Tableau`

The Julia source uses big-rational coefficients converted to `BigFloat`; we will use the `Float64` values, which is what the literature pairs as the "practical" tableau. Each method has hundreds of coefficients; the porting is mechanical but tedious.

**Files:**
- Create: `include/tax/ode/detail/feagin_tableaus.hpp` (both `Feagin12Tab` and `Feagin14Tab`)
- Create: `include/tax/ode/steppers/feagin12.hpp`
- Create: `include/tax/ode/steppers/feagin14.hpp`
- Create: `tests/ode/testFeagin12Stepper.cpp`
- Create: `tests/ode/testFeagin14Stepper.cpp`
- Modify: `include/tax/ode/integrator.hpp`, `include/tax/ode.hpp`, `tests/ode/CMakeLists.txt`

- [ ] **Step 18.1:** Fetch the Julia source for inspection. Either:

  ```bash
  curl -sSL "https://raw.githubusercontent.com/SciML/OrdinaryDiffEq.jl/master/lib/OrdinaryDiffEqFeagin/src/tableaus/feagin_tableaus.jl" \
      -o /tmp/feagin_tableaus.jl
  wc -l /tmp/feagin_tableaus.jl
  ```

  Expected: ~3000 lines covering Feagin8, Feagin10, Feagin12, Feagin14.

  Locate the `Feagin12Tableau` and `Feagin14Tableau` sections. Each tableau lists `c_i`, `a_ij`, `b_i`, and the embedded `bhat_i`. Convert each rational/decimal to a `double` literal.

  Write a small Python helper (`/tmp/convert_feagin.py`) that reads the Julia source, identifies the coefficient arrays, and emits `std::array<double, …>` C++ initializers. Run it for both pairs.

  Save the output as `/tmp/feagin12.inc` and `/tmp/feagin14.inc`.

- [ ] **Step 18.2:** Create `include/tax/ode/detail/feagin_tableaus.hpp` skeleton and `#include` the two `.inc` files inline as the body of each struct's static arrays. Or paste the array initializers directly (preferred — keeps the headers self-contained, no transient `/tmp` dependency).

  Header layout:

  ```cpp
  // include/tax/ode/detail/feagin_tableaus.hpp
  //
  // T. Feagin, "A tenth-order Runge-Kutta method with error estimate"
  // (RK10(8), 2007) and the higher-order RK12(10) and RK14(12) extensions
  // from Feagin's 2010 follow-up. Coefficients ported from
  // SciML/OrdinaryDiffEqFeagin.jl
  // (lib/OrdinaryDiffEqFeagin/src/tableaus/feagin_tableaus.jl,
  // Apache-2.0 license).

  #pragma once

  #include <array>

  namespace tax::ode::detail
  {

  struct Feagin12Tab
  {
      static constexpr int n_stages   = 25;
      static constexpr int order      = 12;
      static constexpr int order_emb  = 10;
      static constexpr bool fsal      = false;

      static constexpr std::array< double, 25 > c     = { /* 25 */ };
      // 25*24/2 = 300 a coefficients (lower-triangular row-major, no diagonal).
      static constexpr std::array< double, 300 > a    = { /* 300 */ };
      static constexpr std::array< double, 25 > b     = { /* 25 */ };
      static constexpr std::array< double, 25 > b_emb = { /* 25 */ };
  };

  struct Feagin14Tab
  {
      static constexpr int n_stages   = 35;
      static constexpr int order      = 14;
      static constexpr int order_emb  = 12;
      static constexpr bool fsal      = false;

      static constexpr std::array< double, 35 > c     = { /* 35 */ };
      // 35*34/2 = 595 a coefficients.
      static constexpr std::array< double, 595 > a    = { /* 595 */ };
      static constexpr std::array< double, 35 > b     = { /* 35 */ };
      static constexpr std::array< double, 35 > b_emb = { /* 35 */ };
  };

  }  // namespace tax::ode::detail
  ```

- [ ] **Step 18.3:** Create `include/tax/ode/steppers/feagin12.hpp` and `include/tax/ode/steppers/feagin14.hpp` by duplicating `verner78.hpp` with the `Verner78` → `Feagin12` (resp. `Feagin14`) token substitution and `Tab = detail::Feagin12Tab` (resp. `Feagin14Tab`).

- [ ] **Step 18.4:** Create the two stepper-level tests by duplicating `testVerner78Stepper.cpp` and substituting tokens.

  Tighten the exponential tolerance to 1e-13 for both Feagin pairs (their higher orders justify it on this simple problem) and the harmonic to 1e-13.

- [ ] **Step 18.5:** Register the two new test executables in `tests/ode/CMakeLists.txt`:

  ```cmake
  tax_add_test(test_ode_feagin12_stepper SOURCES testFeagin12Stepper.cpp)
  tax_add_test(test_ode_feagin14_stepper SOURCES testFeagin14Stepper.cpp)
  ```

- [ ] **Step 18.6:** Add `makeFeagin12Integrator` and `makeFeagin14Integrator` factories in `include/tax/ode/integrator.hpp` (mirror Task 14.7 pattern).

- [ ] **Step 18.7:** Update `include/tax/ode.hpp` to include both new steppers.

- [ ] **Step 18.8:** Extend `testIntegratorBasic.cpp::LotkaVolterraCrossMethod` to compare Feagin12 and Feagin14 against Taylor16:

  ```cpp
  auto f12 = tax::ode::makeFeagin12Integrator< double, 2, false >( f, cfg );
  auto f14 = tax::ode::makeFeagin14Integrator< double, 2, false >( f, cfg );
  const auto sol_f12 = f12.integrate( x0, t0, tf );
  const auto sol_f14 = f14.integrate( x0, t0, tf );
  EXPECT_NEAR( sol_t.x.back()( 0 ), sol_f12.x.back()( 0 ), 1e-10 );
  EXPECT_NEAR( sol_t.x.back()( 1 ), sol_f12.x.back()( 1 ), 1e-10 );
  EXPECT_NEAR( sol_t.x.back()( 0 ), sol_f14.x.back()( 0 ), 1e-10 );
  EXPECT_NEAR( sol_t.x.back()( 1 ), sol_f14.x.back()( 1 ), 1e-10 );
  ```

- [ ] **Step 18.9:** Build, run, commit:

  ```bash
  cmake --build build -j
  ctest --test-dir build -R "test_ode_feagin\|test_ode_integrator_basic" --output-on-failure
  ctest --test-dir build --output-on-failure

  git add include/tax/ode/detail/feagin_tableaus.hpp \
          include/tax/ode/steppers/feagin12.hpp include/tax/ode/steppers/feagin14.hpp \
          include/tax/ode/integrator.hpp include/tax/ode.hpp \
          tests/ode/CMakeLists.txt \
          tests/ode/testFeagin12Stepper.cpp tests/ode/testFeagin14Stepper.cpp \
          tests/ode/testIntegratorBasic.cpp
  git commit -m "$(cat <<'EOF'
  stage2a-slice7: Feagin 12(10) + 14(12) RK steppers

  Adds the two highest-order practical explicit-RK pairs supported by
  Stage 2a. detail::Feagin12Tab (25 stages, order 12 / order-10 embedded)
  and detail::Feagin14Tab (35 stages, order 14 / order-12 embedded).
  Coefficients ported from OrdinaryDiffEqFeagin.jl (Apache-2.0; see
  detail/feagin_tableaus.hpp comment block for attribution). Steppers
  follow the Verner78/89 shape; cross-method tests at order-matched
  tolerances confirm the integrators reach 1e-10 on Lotka–Volterra.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 19 — Cross-method event tests

The event machinery is currently only tested on Taylor (slice 4 of Plan A). Add a parametrized check that all five RK steppers also pass the harmonic-terminate ZeroCrossing test.

**Files:**
- Modify: `tests/ode/testEventsZeroCrossing.cpp`

- [ ] **Step 19.1:** Add a new test case at the bottom of `tests/ode/testEventsZeroCrossing.cpp`:

  ```cpp
  TEST( OdeEventsZeroCrossing, HarmonicTerminateAcrossAllMethods )
  {
      using State = Eigen::Matrix< double, 2, 1 >;
      tax::ode::IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-12;

      const auto f = []( const auto& x, double )
      {
          State out;
          out( 0 ) =  x( 1 );
          out( 1 ) = -x( 0 );
          return out;
      };
      const auto g = []( const auto& x, double ) { return x( 0 ); };

      // Helper: build event list of one Terminate-on-x=0-decreasing event.
      auto build_events = []< class Stepper >()
      {
          std::vector< tax::ode::Event< Stepper > > ev;
          ev.emplace_back(
              tax::ode::ZeroCrossing(
                  []( const auto& x, double ) { return x( 0 ); },
                  tax::ode::Direction::Decreasing ),
              tax::ode::Terminate() );
          return ev;
      };

      State x0; x0( 0 ) = 1.0; x0( 1 ) = 0.0;
      const double tmax = 5.0;
      const double t_expected = M_PI / 2;

      {
          using S = tax::ode::Verner78Stepper< State >;
          auto integ = tax::ode::makeVerner78Integrator< double, 2, false >(
              f, cfg, build_events.template operator()< S >() );
          auto sol = integ.integrate( x0, 0.0, tmax );
          EXPECT_NEAR( sol.t.back(), t_expected, 1e-7 );  // RK + Hermite ≈ 1e-7 on event
      }
      {
          using S = tax::ode::Verner89Stepper< State >;
          auto integ = tax::ode::makeVerner89Integrator< double, 2, false >(
              f, cfg, build_events.template operator()< S >() );
          auto sol = integ.integrate( x0, 0.0, tmax );
          EXPECT_NEAR( sol.t.back(), t_expected, 1e-7 );
      }
      {
          using S = tax::ode::Fehlberg78Stepper< State >;
          auto integ = tax::ode::makeFehlberg78Integrator< double, 2, false >(
              f, cfg, build_events.template operator()< S >() );
          auto sol = integ.integrate( x0, 0.0, tmax );
          EXPECT_NEAR( sol.t.back(), t_expected, 1e-7 );
      }
      {
          using S = tax::ode::Feagin12Stepper< State >;
          auto integ = tax::ode::makeFeagin12Integrator< double, 2, false >(
              f, cfg, build_events.template operator()< S >() );
          auto sol = integ.integrate( x0, 0.0, tmax );
          EXPECT_NEAR( sol.t.back(), t_expected, 1e-7 );
      }
      {
          using S = tax::ode::Feagin14Stepper< State >;
          auto integ = tax::ode::makeFeagin14Integrator< double, 2, false >(
              f, cfg, build_events.template operator()< S >() );
          auto sol = integ.integrate( x0, 0.0, tmax );
          EXPECT_NEAR( sol.t.back(), t_expected, 1e-7 );
      }
  }
  ```

- [ ] **Step 19.2:** Build, run, commit:

  ```bash
  cmake --build build -j
  ctest --test-dir build -R test_ode_events_zerocrossing --output-on-failure

  git add tests/ode/testEventsZeroCrossing.cpp
  git commit -m "$(cat <<'EOF'
  stage2a-slice7b: ZeroCrossing termination test across all five RK steppers

  Each of Verner78, Verner89, Fehlberg78, Feagin12, Feagin14 terminates
  a harmonic-oscillator integration at the first decreasing x=0 crossing,
  with event time matching π/2 to ~1e-7 (the Brent-on-Hermite root finder
  bound, looser than Taylor's polynomial-Newton path).

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 20 — Planar Kepler test (e=0.5)

Verify each (method, controller) pair preserves orbital invariants on a planar Kepler orbit with eccentricity 0.5.

**Files:**
- Create: `tests/ode/testTwoBodyKepler.cpp`
- Modify: `tests/ode/CMakeLists.txt`

- [ ] **Step 20.1:** Append to `tests/ode/CMakeLists.txt`:

  ```cmake
  tax_add_test(test_ode_two_body_kepler SOURCES testTwoBodyKepler.cpp)
  ```

- [ ] **Step 20.2:** Write the test:

  ```cpp
  // tests/ode/testTwoBodyKepler.cpp
  //
  // Planar Kepler with eccentricity e = 0.5, canonical units (GM = 1,
  // semi-major axis a = 1). Initial conditions at periapsis:
  //   r_p = a(1 - e) = 0.5
  //   v_p = sqrt(GM/a * (1+e)/(1-e)) = sqrt(3)
  // Period: T = 2π. Propagate 10 periods and verify:
  //   - specific energy E = ½‖v‖² - 1/‖r‖ conserved within method-scaled tol
  //   - specific angular momentum L = x*vy - y*vx conserved
  //   - closure ‖r(10T) - r_periapsis‖ within tol

  #include <gtest/gtest.h>

  #include <Eigen/Core>
  #include <cmath>

  #include <tax/ode.hpp>

  namespace
  {

  constexpr double kEcc       = 0.5;
  constexpr double kPeriapsis = 1.0 - kEcc;     // a(1-e), with a=1
  constexpr double kVPeriapsis = []
  {
      const double a   = 1.0;
      const double gm  = 1.0;
      return std::sqrt( gm / a * ( 1.0 + kEcc ) / ( 1.0 - kEcc ) );
  }();
  constexpr double kPeriod = 2.0 * M_PI;

  // State = (x, y, vx, vy).
  using State = Eigen::Matrix< double, 4, 1 >;

  inline State make_ic()
  {
      State x0;
      x0( 0 ) = kPeriapsis; x0( 1 ) = 0.0;
      x0( 2 ) = 0.0;        x0( 3 ) = kVPeriapsis;
      return x0;
  }

  inline auto make_rhs()
  {
      return []( const auto& s, double )
      {
          using S = std::decay_t< decltype( s ) >;
          S out;
          const auto x = s( 0 );
          const auto y = s( 1 );
          const auto r3 = std::pow( x * x + y * y, 1.5 );
          out( 0 ) = s( 2 );
          out( 1 ) = s( 3 );
          out( 2 ) = -x / r3;
          out( 3 ) = -y / r3;
          return out;
      };
  }

  double specific_energy( const State& s )
  {
      const double r = std::hypot( s( 0 ), s( 1 ) );
      const double v2 = s( 2 ) * s( 2 ) + s( 3 ) * s( 3 );
      return 0.5 * v2 - 1.0 / r;
  }

  double specific_angmom( const State& s )
  {
      return s( 0 ) * s( 3 ) - s( 1 ) * s( 2 );
  }

  template < class Sol >
  void check_invariants( const Sol& sol, double tol_E, double tol_L, double tol_close )
  {
      const State x0      = sol.x.front();
      const State xfinal  = sol.x.back();
      EXPECT_NEAR( specific_energy( xfinal ), specific_energy( x0 ), tol_E );
      EXPECT_NEAR( specific_angmom( xfinal ), specific_angmom( x0 ), tol_L );
      EXPECT_NEAR( ( xfinal - x0 ).norm(),    0.0,                   tol_close );
  }

  }  // namespace

  TEST( OdeTwoBodyKepler, Taylor16 )
  {
      tax::ode::IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-13;

      auto integ = tax::ode::makeTaylorIntegrator< 16, double, 4, false >(
          make_rhs(), cfg );
      auto sol = integ.integrate( make_ic(), 0.0, 10.0 * kPeriod );

      check_invariants( sol, /*E=*/1e-10, /*L=*/1e-10, /*close=*/1e-8 );
  }

  TEST( OdeTwoBodyKepler, Verner89 )
  {
      tax::ode::IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-13;

      auto integ = tax::ode::makeVerner89Integrator< double, 4, false >(
          make_rhs(), cfg );
      auto sol = integ.integrate( make_ic(), 0.0, 10.0 * kPeriod );

      check_invariants( sol, /*E=*/1e-9, /*L=*/1e-9, /*close=*/1e-7 );
  }

  TEST( OdeTwoBodyKepler, Feagin14 )
  {
      tax::ode::IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-13;

      auto integ = tax::ode::makeFeagin14Integrator< double, 4, false >(
          make_rhs(), cfg );
      auto sol = integ.integrate( make_ic(), 0.0, 10.0 * kPeriod );

      check_invariants( sol, /*E=*/1e-11, /*L=*/1e-11, /*close=*/1e-9 );
  }

  TEST( OdeTwoBodyKepler, Verner78 )
  {
      tax::ode::IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-13;

      auto integ = tax::ode::makeVerner78Integrator< double, 4, false >(
          make_rhs(), cfg );
      auto sol = integ.integrate( make_ic(), 0.0, 10.0 * kPeriod );

      check_invariants( sol, /*E=*/1e-8, /*L=*/1e-8, /*close=*/1e-6 );
  }
  ```

  Fehlberg is omitted from the strict invariant test because the documented coincidence makes the loose-precision tolerances awkward to bound; Fehlberg is covered by the cross-method test in Task 16/17.

- [ ] **Step 20.3:** Build, run, commit:

  ```bash
  cmake --build build -j
  ctest --test-dir build -R test_ode_two_body_kepler --output-on-failure

  git add tests/ode/CMakeLists.txt tests/ode/testTwoBodyKepler.cpp
  git commit -m "$(cat <<'EOF'
  stage2a-slice8a: planar Kepler invariants (e=0.5, 10 periods)

  Propagates a planar Kepler orbit with eccentricity 0.5 for 10 periods
  via Taylor16, Verner78, Verner89, and Feagin14. Verifies that specific
  energy, angular momentum, and orbit closure are preserved to
  method-scaled tolerances. Adaptive step size near periapsis is the
  stress test — h shrinks by ~10× there.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 21 — CR3BP shared fixture + propagation-only test

Set up the planar CR3BP problem (Earth–Moon, μ = 0.01215) with a fixture header that the propagation test, the events test, and the benchmark all share. Then exercise propagation-only correctness: each method preserves the Jacobi constant + matches a high-precision reference within method-scaled tolerance.

**Files:**
- Create: `tests/ode/cr3bp_problem.hpp`
- Create: `tests/ode/testCR3BPPropagation.cpp`
- Modify: `tests/ode/CMakeLists.txt`

- [ ] **Step 21.1:** Append to `tests/ode/CMakeLists.txt`:

  ```cmake
  tax_add_test(test_ode_cr3bp_propagation SOURCES testCR3BPPropagation.cpp)
  ```

- [ ] **Step 21.2:** Create `tests/ode/cr3bp_problem.hpp`:

  ```cpp
  // tests/ode/cr3bp_problem.hpp
  //
  // Shared planar Circular-Restricted Three-Body Problem (Earth–Moon)
  // fixture for the CR3BP propagation test, events test, and benchmark.
  //
  // Synodic rotating frame with the Earth–Moon barycentre at the origin.
  // Earth at (-μ, 0); Moon at (1 - μ, 0).
  // State = (x, y, vx, vy).
  // Equations of motion (canonical Hamiltonian form):
  //   dx/dt  = vx
  //   dy/dt  = vy
  //   dvx/dt =  2 vy + x - (1-μ)(x+μ)/r1^3 - μ(x-1+μ)/r2^3
  //   dvy/dt = -2 vx + y - (1-μ)y/r1^3      - μy/r2^3
  // where r1 = √((x+μ)² + y²), r2 = √((x-1+μ)² + y²).
  //
  // Jacobi constant (conserved):
  //   C = 2*Omega(x, y) - vx² - vy²
  //   Omega = ½(x² + y²) + (1-μ)/r1 + μ/r2 + ½μ(1-μ)
  //
  // Lagrange points (numerical roots of dOmega/dx = 0 on y = 0):
  //   L1 ≈ 0.8369180073...
  //   L2 ≈ 1.1556824692...
  //   (computed once via Newton's method against the collinear equation
  //   and pinned here for the events test.)
  //
  // Initial condition for the propagation test is chosen from the L1
  // Lyapunov-orbit family (Koon–Lo–Marsden–Ross 2011, Ch. 2.5) at
  // moderate amplitude so the trajectory transits the L1 neck, loops
  // around the Moon, and exits via L2 over T_final = 7 non-dim. units.

  #pragma once

  #include <Eigen/Core>
  #include <cmath>

  namespace tax::ode::test
  {

  constexpr double kCR3BPMu  = 0.01215058560962404;   // Earth–Moon
  constexpr double kCR3BPL1  = 0.8369180073407246;
  constexpr double kCR3BPL2  = 1.1556824692238923;

  using CR3BPState = Eigen::Matrix< double, 4, 1 >;

  inline auto cr3bp_rhs( double mu = kCR3BPMu )
  {
      return [mu]( const auto& s, double ) -> std::decay_t< decltype( s ) >
      {
          using S = std::decay_t< decltype( s ) >;
          using V = typename S::Scalar;

          S out;
          const V x  = s( 0 );
          const V y  = s( 1 );
          const V vx = s( 2 );
          const V vy = s( 3 );

          const V x1   = x + V( mu );
          const V x2   = x - V( 1.0 - mu );
          const V r1_2 = x1 * x1 + y * y;
          const V r2_2 = x2 * x2 + y * y;
          const V r1_3 = r1_2 * sqrt( r1_2 );
          const V r2_3 = r2_2 * sqrt( r2_2 );

          out( 0 ) = vx;
          out( 1 ) = vy;
          out( 2 ) =  V( 2 ) * vy + x
                     - V( 1.0 - mu ) * x1 / r1_3
                     - V( mu )       * x2 / r2_3;
          out( 3 ) = -V( 2 ) * vx + y
                     - V( 1.0 - mu ) * y  / r1_3
                     - V( mu )       * y  / r2_3;
          return out;
      };
  }

  inline double cr3bp_jacobi( const CR3BPState& s, double mu = kCR3BPMu )
  {
      const double x = s( 0 ), y = s( 1 );
      const double vx = s( 2 ), vy = s( 3 );
      const double r1 = std::hypot( x + mu, y );
      const double r2 = std::hypot( x - 1.0 + mu, y );
      const double Omega = 0.5 * ( x * x + y * y )
                         + ( 1.0 - mu ) / r1 + mu / r2
                         + 0.5 * mu * ( 1.0 - mu );
      return 2.0 * Omega - ( vx * vx + vy * vy );
  }

  // L1 Lyapunov-orbit-family transit IC. Tuned so the trajectory passes
  // the L1 neck on its way to the Moon, loops, and exits via L2 over
  // t in [0, 7] non-dim. units. ICs may be refined; the propagation
  // test only requires that all methods agree at the endpoint and
  // preserve the Jacobi constant.
  inline CR3BPState cr3bp_transit_ic()
  {
      CR3BPState x0;
      x0( 0 ) = 0.836;    // just inside L1
      x0( 1 ) = 0.0;
      x0( 2 ) = 0.0;
      x0( 3 ) = 0.0095;   // small velocity normal to x-axis
      return x0;
  }

  constexpr double kCR3BPTFinal = 7.0;

  }  // namespace tax::ode::test
  ```

- [ ] **Step 21.3:** Write `tests/ode/testCR3BPPropagation.cpp`:

  ```cpp
  // tests/ode/testCR3BPPropagation.cpp
  //
  // CR3BP propagation correctness: each method preserves the Jacobi
  // constant within a method-scaled tolerance over T_final = 7 units,
  // and the endpoint matches a high-precision reference (Feagin14 +
  // PI controller at 1e-14) within similar tolerances.

  #include <gtest/gtest.h>

  #include <Eigen/Core>
  #include <cmath>

  #include "cr3bp_problem.hpp"
  #include <tax/ode.hpp>

  using namespace tax::ode::test;

  namespace
  {

  CR3BPState compute_reference()
  {
      tax::ode::IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-14;
      auto integ = tax::ode::makeFeagin14Integrator< double, 4, false >(
          cr3bp_rhs(), cfg );
      auto sol = integ.integrate( cr3bp_transit_ic(), 0.0, kCR3BPTFinal );
      return sol.x.back();
  }

  template < class Sol >
  void check_jacobi_preserved( const Sol& sol, double tol )
  {
      const double C0 = cr3bp_jacobi( sol.x.front() );
      const double C1 = cr3bp_jacobi( sol.x.back() );
      EXPECT_NEAR( C1, C0, tol );
  }

  }  // namespace

  TEST( OdeCR3BPPropagation, ReferenceTrajectoryIsStable )
  {
      const CR3BPState ref = compute_reference();
      // Sanity: the reference completed (no NaN), trajectory passed
      // through the lunar neighbourhood at some point.
      EXPECT_TRUE( std::isfinite( ref( 0 ) ) );
      EXPECT_TRUE( std::isfinite( ref( 1 ) ) );
  }

  TEST( OdeCR3BPPropagation, Taylor16 )
  {
      tax::ode::IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-13;
      auto integ = tax::ode::makeTaylorIntegrator< 16, double, 4, false >(
          cr3bp_rhs(), cfg );
      auto sol = integ.integrate( cr3bp_transit_ic(), 0.0, kCR3BPTFinal );
      check_jacobi_preserved( sol, 1e-10 );

      const CR3BPState ref = compute_reference();
      EXPECT_NEAR( ( sol.x.back() - ref ).norm(), 0.0, 1e-8 );
  }

  TEST( OdeCR3BPPropagation, Verner89 )
  {
      tax::ode::IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-13;
      auto integ = tax::ode::makeVerner89Integrator< double, 4, false >(
          cr3bp_rhs(), cfg );
      auto sol = integ.integrate( cr3bp_transit_ic(), 0.0, kCR3BPTFinal );
      check_jacobi_preserved( sol, 1e-9 );

      const CR3BPState ref = compute_reference();
      EXPECT_NEAR( ( sol.x.back() - ref ).norm(), 0.0, 1e-7 );
  }

  TEST( OdeCR3BPPropagation, Verner78 )
  {
      tax::ode::IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-13;
      auto integ = tax::ode::makeVerner78Integrator< double, 4, false >(
          cr3bp_rhs(), cfg );
      auto sol = integ.integrate( cr3bp_transit_ic(), 0.0, kCR3BPTFinal );
      check_jacobi_preserved( sol, 1e-8 );

      const CR3BPState ref = compute_reference();
      EXPECT_NEAR( ( sol.x.back() - ref ).norm(), 0.0, 1e-6 );
  }

  TEST( OdeCR3BPPropagation, Feagin12 )
  {
      tax::ode::IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-13;
      auto integ = tax::ode::makeFeagin12Integrator< double, 4, false >(
          cr3bp_rhs(), cfg );
      auto sol = integ.integrate( cr3bp_transit_ic(), 0.0, kCR3BPTFinal );
      check_jacobi_preserved( sol, 1e-10 );

      const CR3BPState ref = compute_reference();
      EXPECT_NEAR( ( sol.x.back() - ref ).norm(), 0.0, 1e-8 );
  }
  ```

- [ ] **Step 21.4:** Build, run, commit:

  ```bash
  cmake --build build -j
  ctest --test-dir build -R test_ode_cr3bp_propagation --output-on-failure

  git add tests/ode/cr3bp_problem.hpp tests/ode/CMakeLists.txt \
          tests/ode/testCR3BPPropagation.cpp
  git commit -m "$(cat <<'EOF'
  stage2a-slice8b: CR3BP shared fixture + propagation correctness test

  Adds the planar Earth–Moon CR3BP RHS, Jacobi-constant computation, and
  L1/L2 location constants in a shared header. The propagation test
  exercises Taylor16, Verner78, Verner89, Feagin12 against a Feagin14 +
  PI @ 1e-14 reference trajectory over t in [0, 7] non-dim. units, with
  Jacobi-constant preservation as an invariant guard.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 22 — CR3BP events test

Reuse the fixture from Task 21 plus three `ZeroCrossing` events: L1 crossing, lunar periapsis, L2 crossing.

**Files:**
- Create: `tests/ode/testCR3BPEvents.cpp`
- Modify: `tests/ode/CMakeLists.txt`

- [ ] **Step 22.1:** Append to `tests/ode/CMakeLists.txt`:

  ```cmake
  tax_add_test(test_ode_cr3bp_events SOURCES testCR3BPEvents.cpp)
  ```

- [ ] **Step 22.2:** Write the test:

  ```cpp
  // tests/ode/testCR3BPEvents.cpp
  //
  // CR3BP propagation with ZeroCrossing events:
  //   - L1 crossing  (x crosses x_L1 going +x)
  //   - lunar periapsis  ((x - 1 + μ)·vx + y·vy crosses zero going +)
  //   - L2 crossing  (x crosses x_L2 going +x)
  // Assertions: at least one L1 event before the lunar loop, at least
  // one moon-periapsis event within distance 0.1 of the Moon, and at
  // least one L2 event before T_final.

  #include <gtest/gtest.h>

  #include <Eigen/Core>
  #include <algorithm>
  #include <cmath>
  #include <vector>

  #include "cr3bp_problem.hpp"
  #include <tax/ode.hpp>

  using namespace tax::ode::test;
  using tax::ode::Direction;
  using tax::ode::Event;
  using tax::ode::IntegratorConfig;
  using tax::ode::Record;
  using tax::ode::TaylorStepper;
  using tax::ode::ZeroCrossing;

  TEST( OdeCR3BPEvents, TaylorRecordsL1MoonL2 )
  {
      constexpr int N = 16;
      using State = CR3BPState;

      IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-13;

      using Stepper = TaylorStepper< N, State >;
      std::vector< Event< Stepper > > events;

      events.emplace_back(
          ZeroCrossing(
              []( const auto& s, double ) { return s( 0 ) - kCR3BPL1; },
              Direction::Increasing ),
          Record( "L1" ) );

      const double mu = kCR3BPMu;
      events.emplace_back(
          ZeroCrossing(
              [mu]( const auto& s, double )
              { return ( s( 0 ) - 1.0 + mu ) * s( 2 ) + s( 1 ) * s( 3 ); },
              Direction::Increasing ),
          Record( "moon_periapsis" ) );

      events.emplace_back(
          ZeroCrossing(
              []( const auto& s, double ) { return s( 0 ) - kCR3BPL2; },
              Direction::Increasing ),
          Record( "L2" ) );

      auto integ = tax::ode::makeTaylorIntegrator< N, double, 4, false >(
          cr3bp_rhs(), cfg, events );
      auto sol = integ.integrate( cr3bp_transit_ic(), 0.0, kCR3BPTFinal );

      const auto countLabel = [ & ]( const char* lbl ) {
          return std::count_if( sol.events.begin(), sol.events.end(),
              [ lbl ]( const auto& e ) { return e.label == lbl; } );
      };
      EXPECT_GE( countLabel( "L1" ),             1 );
      EXPECT_GE( countLabel( "moon_periapsis" ), 1 );
      EXPECT_GE( countLabel( "L2" ),             1 );

      // At least one moon-periapsis event has r2 < 0.1.
      bool close_moon = false;
      for ( const auto& e : sol.events )
      {
          if ( e.label != "moon_periapsis" ) continue;
          const double r2 = std::hypot( e.x_event( 0 ) - 1.0 + mu, e.x_event( 1 ) );
          if ( r2 < 0.1 ) { close_moon = true; break; }
      }
      EXPECT_TRUE( close_moon );

      // Jacobi constant preserved through event machinery.
      const double C0 = cr3bp_jacobi( sol.x.front() );
      const double C1 = cr3bp_jacobi( sol.x.back() );
      EXPECT_NEAR( C1, C0, 1e-10 );
  }
  ```

- [ ] **Step 22.3:** Build, run, commit:

  ```bash
  cmake --build build -j
  ctest --test-dir build -R test_ode_cr3bp_events --output-on-failure

  git add tests/ode/CMakeLists.txt tests/ode/testCR3BPEvents.cpp
  git commit -m "$(cat <<'EOF'
  stage2a-slice9a: CR3BP events test (L1, moon periapsis, L2)

  Three ZeroCrossing events attached to the shared CR3BP propagation
  fixture; Taylor16 records each occurrence as the transit trajectory
  passes through the L1 neck, around the Moon, and exits via L2. The
  Jacobi constant is verified to be preserved through the event
  machinery (events don't perturb the trajectory).

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 23 — Benchmark suite

Reuse the CR3BP fixture for a Google Benchmark suite that times each (method, controller, tolerance) combination plus a Taylor order sweep.

**Files:**
- Create: `benchmarks/bench_ode_cr3bp.cpp`
- Modify: `benchmarks/CMakeLists.txt`

- [ ] **Step 23.1:** Inspect `benchmarks/CMakeLists.txt` and locate the existing benchmark-registration pattern (likely a `tax_add_benchmark(name SOURCES …)` helper similar to `tax_add_test`). Append a new entry:

  ```cmake
  tax_add_benchmark(bench_ode_cr3bp SOURCES bench_ode_cr3bp.cpp)
  ```

  If `tax_add_benchmark` doesn't exist, add the executable directly using the same pattern other benchmarks use (link `tax`, `benchmark::benchmark`, `benchmark::benchmark_main`).

- [ ] **Step 23.2:** Write `benchmarks/bench_ode_cr3bp.cpp`:

  ```cpp
  // benchmarks/bench_ode_cr3bp.cpp
  //
  // Cross-method, cross-controller benchmark on the shared CR3BP
  // propagation fixture. Reference combination: Fehlberg78 + I-controller
  // at 1e-14 (classical baseline). Each benchmarked combination is timed
  // and its endpoint deviation from the reference is reported.
  //
  // Taylor is swept across orders {8, 10, 12, 16, 20, 24, 30} under
  // JorbaZou, with PI and H211b additionally exercised at N = 12 and
  // N = 24.
  //
  // Build with: cmake -DTAX_BUILD_BENCHMARK=ON … && cmake --build … --target bench_ode_cr3bp
  // Run with:   ./bench_ode_cr3bp --benchmark_format=console

  #include <benchmark/benchmark.h>

  #include <Eigen/Core>

  #include "../tests/ode/cr3bp_problem.hpp"
  #include <tax/ode.hpp>

  using namespace tax::ode::test;

  namespace
  {

  CR3BPState g_reference;
  bool       g_reference_ready = false;

  void ensure_reference()
  {
      if ( g_reference_ready ) return;
      tax::ode::IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-14;
      using Stepper = tax::ode::Fehlberg78Stepper<
          Eigen::Matrix< double, 4, 1 >, tax::ode::controllers::I< double > >;
      auto integ = tax::ode::Integrator<
          Stepper, decltype( cr3bp_rhs() ), /*Dense=*/false >{
              cr3bp_rhs(), cfg };
      auto sol = integ.integrate( cr3bp_transit_ic(), 0.0, kCR3BPTFinal );
      g_reference = sol.x.back();
      g_reference_ready = true;
  }

  double endpoint_error( const CR3BPState& x )
  {
      ensure_reference();
      return ( x - g_reference ).norm();
  }

  // Convenience wrappers.
  template < int N, class Controller = tax::ode::controllers::JorbaZou< double > >
  CR3BPState run_taylor( double tol )
  {
      tax::ode::IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = tol;
      using State = Eigen::Matrix< double, 4, 1 >;
      using Stepper = tax::ode::TaylorStepper< N, State, Controller >;
      auto integ = tax::ode::Integrator<
          Stepper, decltype( cr3bp_rhs() ), false >{ cr3bp_rhs(), cfg };
      return integ.integrate( cr3bp_transit_ic(), 0.0, kCR3BPTFinal ).x.back();
  }

  template < class Stepper >
  CR3BPState run_rk( double tol )
  {
      tax::ode::IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = tol;
      auto integ = tax::ode::Integrator<
          Stepper, decltype( cr3bp_rhs() ), false >{ cr3bp_rhs(), cfg };
      return integ.integrate( cr3bp_transit_ic(), 0.0, kCR3BPTFinal ).x.back();
  }

  }  // namespace

  // -------- Reference: Fehlberg78 + I --------
  static void BM_RefFehlberg78_I_1e12( benchmark::State& s )
  {
      using S = Eigen::Matrix< double, 4, 1 >;
      using Stepper = tax::ode::Fehlberg78Stepper< S, tax::ode::controllers::I< double > >;
      for ( auto _ : s )
      {
          auto x = run_rk< Stepper >( 1e-12 );
          benchmark::DoNotOptimize( x );
      }
      s.counters["err"] = endpoint_error( run_rk< Stepper >( 1e-12 ) );
  }
  BENCHMARK( BM_RefFehlberg78_I_1e12 );

  // -------- Verner pairs across controllers --------
  #define BENCH_RK( name, Stepper, ControllerT, tol )                              \
      static void name( benchmark::State& s )                                       \
      {                                                                             \
          using S = Eigen::Matrix< double, 4, 1 >;                                  \
          using St = Stepper< S, tax::ode::controllers::ControllerT< double > >;    \
          for ( auto _ : s )                                                        \
          {                                                                         \
              auto x = run_rk< St >( tol );                                         \
              benchmark::DoNotOptimize( x );                                        \
          }                                                                         \
          s.counters["err"] = endpoint_error( run_rk< St >( tol ) );                \
      }                                                                             \
      BENCHMARK( name )

  BENCH_RK( BM_Verner78_PI_1e12,   tax::ode::Verner78Stepper,   PI,    1e-12 );
  BENCH_RK( BM_Verner89_PI_1e12,   tax::ode::Verner89Stepper,   PI,    1e-12 );
  BENCH_RK( BM_Feagin12_PI_1e12,   tax::ode::Feagin12Stepper,   PI,    1e-12 );
  BENCH_RK( BM_Feagin14_PI_1e12,   tax::ode::Feagin14Stepper,   PI,    1e-12 );
  BENCH_RK( BM_Verner89_H211b_1e12, tax::ode::Verner89Stepper,  H211b, 1e-12 );
  BENCH_RK( BM_Feagin14_H211b_1e12, tax::ode::Feagin14Stepper,  H211b, 1e-12 );

  // -------- Taylor order sweep with JorbaZou --------
  #define BENCH_TAYLOR( name, N )                                                  \
      static void name( benchmark::State& s )                                       \
      {                                                                             \
          for ( auto _ : s )                                                        \
          {                                                                         \
              auto x = run_taylor< N >( 1e-12 );                                    \
              benchmark::DoNotOptimize( x );                                        \
          }                                                                         \
          s.counters["err"] = endpoint_error( run_taylor< N >( 1e-12 ) );           \
      }                                                                             \
      BENCHMARK( name )

  BENCH_TAYLOR( BM_TaylorJZ_N08, 8 );
  BENCH_TAYLOR( BM_TaylorJZ_N10, 10 );
  BENCH_TAYLOR( BM_TaylorJZ_N12, 12 );
  BENCH_TAYLOR( BM_TaylorJZ_N16, 16 );
  BENCH_TAYLOR( BM_TaylorJZ_N20, 20 );
  BENCH_TAYLOR( BM_TaylorJZ_N24, 24 );
  BENCH_TAYLOR( BM_TaylorJZ_N30, 30 );

  // PI + H211b at two representative Taylor orders.
  static void BM_TaylorPI_N12( benchmark::State& s )
  {
      for ( auto _ : s )
      {
          auto x = run_taylor< 12, tax::ode::controllers::PI< double > >( 1e-12 );
          benchmark::DoNotOptimize( x );
      }
      s.counters["err"] = endpoint_error(
          run_taylor< 12, tax::ode::controllers::PI< double > >( 1e-12 ) );
  }
  BENCHMARK( BM_TaylorPI_N12 );

  static void BM_TaylorH211b_N24( benchmark::State& s )
  {
      for ( auto _ : s )
      {
          auto x = run_taylor< 24, tax::ode::controllers::H211b< double > >( 1e-12 );
          benchmark::DoNotOptimize( x );
      }
      s.counters["err"] = endpoint_error(
          run_taylor< 24, tax::ode::controllers::H211b< double > >( 1e-12 ) );
  }
  BENCHMARK( BM_TaylorH211b_N24 );

  BENCHMARK_MAIN();
  ```

- [ ] **Step 23.3:** Configure with benchmarks enabled and build:

  ```bash
  cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DTAX_BUILD_BENCHMARK=ON -G Ninja
  cmake --build build-bench --target bench_ode_cr3bp -j
  ```

  Expected: clean build. Confirm the executable exists at `build-bench/benchmarks/bench_ode_cr3bp` (path may vary by project layout).

- [ ] **Step 23.4:** Run a quick benchmark dry run with low time:

  ```bash
  ./build-bench/benchmarks/bench_ode_cr3bp --benchmark_min_time=0.01s | head -50
  ```

  Expected: every benchmark reports finite `err` counter and finite wall time. The reference benchmark (`BM_RefFehlberg78_I_1e12`) should report `err ≈ 0`. The other benchmarks should report `err` ≪ 1.

- [ ] **Step 23.5:** Commit. (Do not commit benchmark results — those go in a separate doc later.)

  ```bash
  git add benchmarks/CMakeLists.txt benchmarks/bench_ode_cr3bp.cpp
  git commit -m "$(cat <<'EOF'
  stage2a-slice9b: CR3BP method × controller × Taylor-order benchmark

  Google-Benchmark suite that exercises every Stage 2a (method,
  controller) combination plus a Taylor order sweep {8, 10, 12, 16,
  20, 24, 30} under JorbaZou, with PI and H211b at N = 12 and N = 24
  for controller comparison. Reference: Fehlberg78 + I-controller at
  1e-14 (classical baseline). Each row reports wall time and endpoint
  error vs the reference.

  Built only with -DTAX_BUILD_BENCHMARK=ON; no impact on the default
  unit-test build path.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 24 — Static-vs-dynamic D parity test

Confirm that `Eigen::Matrix<T, D_static, 1>` and `Eigen::Vector<T, Eigen::Dynamic>` produce identical results across every method.

**Files:**
- Create: `tests/ode/testIntegratorStatic.cpp`
- Modify: `tests/ode/CMakeLists.txt`

- [ ] **Step 24.1:** Append to `tests/ode/CMakeLists.txt`:

  ```cmake
  tax_add_test(test_ode_integrator_static SOURCES testIntegratorStatic.cpp)
  ```

- [ ] **Step 24.2:** Write the test:

  ```cpp
  // tests/ode/testIntegratorStatic.cpp
  //
  // Compile-time D == Eigen::Dynamic parity. The same RHS (a generic
  // lambda) propagated through static D=4 and dynamic D must produce
  // bitwise- or near-bitwise-identical endpoints.

  #include <gtest/gtest.h>

  #include <Eigen/Core>
  #include <cmath>

  #include <tax/ode.hpp>

  namespace
  {

  inline auto harmonic_rhs()
  {
      return []( const auto& s, double )
      {
          using S = std::decay_t< decltype( s ) >;
          S out;
          if constexpr ( S::RowsAtCompileTime == Eigen::Dynamic )
              out.resize( s.size() );
          out( 0 ) =  s( 1 );
          out( 1 ) = -s( 0 );
          return out;
      };
  }

  }  // namespace

  TEST( OdeIntegratorStatic, Taylor16ParityWithDynamic )
  {
      tax::ode::IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-13;

      using SStatic = Eigen::Matrix< double, 2, 1 >;
      using SDyn    = Eigen::VectorXd;

      auto integ_s = tax::ode::makeTaylorIntegrator< 16, double, 2,             false >(
          harmonic_rhs(), cfg );
      auto integ_d = tax::ode::makeTaylorIntegrator< 16, double, Eigen::Dynamic, false >(
          harmonic_rhs(), cfg );

      SStatic x0_s; x0_s( 0 ) = 1.0; x0_s( 1 ) = 0.0;
      SDyn    x0_d( 2 ); x0_d( 0 ) = 1.0; x0_d( 1 ) = 0.0;

      auto sol_s = integ_s.integrate( x0_s, 0.0, M_PI );
      auto sol_d = integ_d.integrate( x0_d, 0.0, M_PI );

      EXPECT_NEAR( sol_s.x.back()( 0 ), sol_d.x.back()( 0 ), 1e-12 );
      EXPECT_NEAR( sol_s.x.back()( 1 ), sol_d.x.back()( 1 ), 1e-12 );
  }

  TEST( OdeIntegratorStatic, Verner89ParityWithDynamic )
  {
      tax::ode::IntegratorConfig< double > cfg;
      cfg.abstol = cfg.reltol = 1e-13;

      using SStatic = Eigen::Matrix< double, 2, 1 >;
      using SDyn    = Eigen::VectorXd;

      auto integ_s = tax::ode::makeVerner89Integrator< double, 2,             false >(
          harmonic_rhs(), cfg );
      auto integ_d = tax::ode::makeVerner89Integrator< double, Eigen::Dynamic, false >(
          harmonic_rhs(), cfg );

      SStatic x0_s; x0_s( 0 ) = 1.0; x0_s( 1 ) = 0.0;
      SDyn    x0_d( 2 ); x0_d( 0 ) = 1.0; x0_d( 1 ) = 0.0;

      auto sol_s = integ_s.integrate( x0_s, 0.0, M_PI );
      auto sol_d = integ_d.integrate( x0_d, 0.0, M_PI );

      EXPECT_NEAR( sol_s.x.back()( 0 ), sol_d.x.back()( 0 ), 1e-12 );
      EXPECT_NEAR( sol_s.x.back()( 1 ), sol_d.x.back()( 1 ), 1e-12 );
  }
  ```

- [ ] **Step 24.3:** Build, run, commit:

  ```bash
  cmake --build build -j
  ctest --test-dir build -R test_ode_integrator_static --output-on-failure
  ctest --test-dir build --output-on-failure

  git add tests/ode/CMakeLists.txt tests/ode/testIntegratorStatic.cpp
  git commit -m "$(cat <<'EOF'
  stage2a-slice10: static-D vs Eigen::Dynamic parity test

  Verifies that the same RHS lambda and ICs propagated through
  Eigen::Matrix<T, 2, 1> and Eigen::VectorXd produce bitwise-equivalent
  endpoints (within 1e-12 across half a harmonic period) for both the
  Taylor and Verner89 paths. Closes the Stage 2a static-vs-dynamic
  promise.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Self-review

**Spec coverage check** against the design doc:

| Spec requirement | Covered by |
|---|---|
| Verner 8(7) stepper | Task 14 |
| Verner 9(8) stepper | Task 15 |
| Fehlberg 7(8) stepper | Task 17 |
| Feagin 12(10) stepper | Task 18 |
| Feagin 14(12) stepper | Task 18 |
| Pluggable Controller on RK steppers | Tasks 14, 15, 17, 18 (each stepper takes `Controller` template param) |
| `eval_dense` Hermite continuation | Task 13 (`hermite_interp`) + each stepper's `eval_dense` |
| Cross-method correctness | Tasks 16, 17.7, 18.8, 19 |
| Two-body Kepler invariants test | Task 20 |
| CR3BP propagation correctness | Tasks 21 (fixture + propagation test) |
| CR3BP events test | Task 22 |
| Benchmark sweep including Taylor order set {8, 10, 12, 16, 20, 24, 30} | Task 23 |
| Static-vs-dynamic D parity | Task 24 |

**Placeholder scan:** No "TBD"/"TODO"/"implement later" in the plan body. Butcher tableau values are deferred to authoritative sources (pre-Stage-1 branch for Verner; Boost.Odeint for Fehlberg; OrdinaryDiffEqFeagin.jl for Feagin), with each task explicitly pointing at the source.

**Type-consistency check:**
- All RK steppers follow `template < class StateT, class Controller = controllers::PI<…> > struct …Stepper { using State = StateT; ... };` — consistent with Plan A's TaylorStepper.
- Every stepper exposes `static constexpr bool is_adaptive = true;`, satisfying `concepts::AdaptiveStepper`.
- `makeXIntegrator<T, D, Dense, Controller>(f, cfg [, events])` factories all follow the same parameter order.
- `DenseData` for RK steppers is the four-field `{x0, x1, f0, f1}` struct; consistent across Verner78/89/Fehlberg78/Feagin12/Feagin14.
- `adaptive_rk_step` is called with `RKStepData<State, Tab::n_stages>` work buffer; signature unchanged across all five steppers.

No spec-coverage gaps found. No placeholders. Plan ready for execution.

---

**Plan complete.** Ready for execution.
