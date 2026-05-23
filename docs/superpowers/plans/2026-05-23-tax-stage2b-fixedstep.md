# Stage 2b Slice A — FixedStep controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `FixedStep<T>` controller that forces every step to use `cfg.initial_step` and to be accepted, and wire it into every shipped stepper (Taylor + 5 RK).

**Architecture:** New stateless controller type in `tax/ode/controllers.hpp`. Each stepper gains an additional `if constexpr` arm alongside the existing `JorbaZou` special case: when `Controller == FixedStep<T>`, set `h_next = h` and `accepted = true` unconditionally. Embedded estimator is still computed (free, useful for diagnostics).

**Tech Stack:** C++23, header-only, Google Test, CMake.

---

## File map

- Modify: `include/tax/ode/controllers.hpp` — append `FixedStep<T>`.
- Modify: `include/tax/ode/steppers/verner78.hpp`, `verner89.hpp`, `fehlberg78.hpp`, `feagin12.hpp`, `feagin14.hpp`, `taylor.hpp` — each gets one `if constexpr` arm added inside `step()`.
- Create: `tests/ode/testFixedStep.cpp` — six tests, one per stepper.
- Modify: `tests/ode/CMakeLists.txt` — register the new test executable.

The spec for this slice lives in `docs/superpowers/specs/2026-05-23-tax-stage2b-fixedstep-da-state-design.md` (Section "Slice A").

---

### Task 1: Add `FixedStep<T>` controller

**Files:**
- Modify: `include/tax/ode/controllers.hpp` — append at end of `tax::ode::controllers` namespace, before the closing `}`.

- [ ] **Step 1: Add the controller**

Append inside `namespace tax::ode::controllers { ... }`, just before its closing brace:

```cpp
// -------- FixedStep --------
// No-op controller: returns the previously-used step size unchanged,
// regardless of error or tolerance. Used to force a user-prescribed
// step grid; steppers also treat this controller as a signal to mark
// every step accepted (so adaptive retry never kicks in).
template < class T = double >
struct FixedStep
{
    [[nodiscard]] T next_step( T h_used, T /*err_norm*/, T /*tol*/,
                                int /*p_emb*/ ) const noexcept
    { return h_used; }

    // Overload matching JorbaZou's call signature, used by TaylorStepper.
    [[nodiscard]] T next_step( T h_used, T /*c_N_norm*/, T /*c_Nm1_norm*/,
                                T /*tol*/, int /*N_order*/ ) const noexcept
    { return h_used; }
};
```

- [ ] **Step 2: Verify it compiles**

Run:
```
cmake --build build --target tax_headers 2>&1 | tail -5
```

Or, if the project does not configure a `tax_headers` target, build any existing ODE test executable instead:
```
cmake --build build --target test_ode_controllers 2>&1 | tail -5
```
Expected: no errors. (The type is unused — nothing references it yet.)

- [ ] **Step 3: Commit**

```bash
git add include/tax/ode/controllers.hpp
git commit -m "ode: add FixedStep<T> step-size controller"
```

---

### Task 2: Register `testFixedStep` and write the first failing test (Verner78)

**Files:**
- Create: `tests/ode/testFixedStep.cpp`
- Modify: `tests/ode/CMakeLists.txt`

- [ ] **Step 1: Register the test executable**

Append to `tests/ode/CMakeLists.txt`:
```cmake
tax_add_test(test_ode_fixed_step SOURCES testFixedStep.cpp)
```

- [ ] **Step 2: Create the test file with the Verner78 case**

Write `tests/ode/testFixedStep.cpp`:

```cpp
// tests/ode/testFixedStep.cpp
//
// FixedStep controller: forces every step to use cfg.initial_step
// and to be accepted, regardless of tolerance. One test per stepper.

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <tax/ode.hpp>

using tax::ode::IntegratorConfig;
using tax::ode::controllers::FixedStep;

namespace {

constexpr double kH = 0.1;

template < class State >
auto identity_rhs()
{
    return []( const State& x, double ) { return x; };
}

template < class Solution >
void check_uniform_grid( const Solution& sol, double h, std::size_t expected_count )
{
    ASSERT_EQ( sol.t.size(), expected_count );
    for ( std::size_t i = 0; i < sol.t.size(); ++i )
        EXPECT_NEAR( sol.t[ i ], h * double( i ), 1e-12 )
            << "step index " << i;
}

}  // namespace

TEST( OdeFixedStep, Verner78AlwaysAcceptedAtTightTol )
{
    using State = Eigen::Matrix< double, 1, 1 >;

    IntegratorConfig< double > cfg;
    cfg.initial_step = kH;
    cfg.abstol = cfg.reltol = 1e-30;        // impossibly tight; must still accept

    auto integ = tax::ode::makeVerner78Integrator< double, 1, false,
                                                    FixedStep< double > >(
        identity_rhs< State >(), cfg );

    State x0; x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 1.0 );

    check_uniform_grid( sol, kH, /*expected_count=*/11u );
}
```

- [ ] **Step 3: Run the test — it must FAIL (rejection cap exception)**

Run:
```
cmake --build build --target test_ode_fixed_step
ctest --test-dir build -R '^test_ode_fixed_step$' --output-on-failure
```
Expected: the test throws `Integrator::integrate: rejection cap reached` from `include/tax/ode/integrator.hpp`. The test must not pass yet.

- [ ] **Step 4: Commit the failing test**

```bash
git add tests/ode/testFixedStep.cpp tests/ode/CMakeLists.txt
git commit -m "ode: add failing test for FixedStep on Verner78"
```

---

### Task 3: Wire FixedStep into Verner78Stepper

**Files:**
- Modify: `include/tax/ode/steppers/verner78.hpp` — lines 71–77 (`h_next` selection and `accepted` computation).

- [ ] **Step 1: Replace the controller-selection block**

In `include/tax/ode/steppers/verner78.hpp`, find:

```cpp
        T h_next;
        if constexpr ( std::is_same_v< Controller, controllers::JorbaZou< T > > )
            h_next = h;  // JorbaZou is Taylor-only; no-op fallback.
        else
            h_next = controller_.next_step( h, out.err_norm, tol, Tab::order_emb );

        const bool accepted = out.err_norm <= tol;
```

Replace with:

```cpp
        T    h_next;
        bool accepted;
        if constexpr ( std::is_same_v< Controller, controllers::FixedStep< T > > )
        {
            h_next   = h;
            accepted = true;
        }
        else if constexpr ( std::is_same_v< Controller, controllers::JorbaZou< T > > )
        {
            h_next   = h;  // JorbaZou is Taylor-only; no-op fallback.
            accepted = out.err_norm <= tol;
        }
        else
        {
            h_next   = controller_.next_step( h, out.err_norm, tol, Tab::order_emb );
            accepted = out.err_norm <= tol;
        }
```

- [ ] **Step 2: Run the test — it must now PASS**

Run:
```
cmake --build build --target test_ode_fixed_step
ctest --test-dir build -R '^test_ode_fixed_step$' --output-on-failure
```
Expected: `OdeFixedStep.Verner78AlwaysAcceptedAtTightTol` passes; the other (not-yet-written) stepper tests do not exist yet.

- [ ] **Step 3: Run the full Verner78 stepper test suite to confirm no regression**

Run:
```
ctest --test-dir build -R 'test_ode_(verner78|fixed_step)' --output-on-failure
```
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add include/tax/ode/steppers/verner78.hpp
git commit -m "ode/verner78: honour FixedStep controller (always-accepted)"
```

---

### Task 4: Verner89 — failing test + wire-up

**Files:**
- Modify: `tests/ode/testFixedStep.cpp` (append a test).
- Modify: `include/tax/ode/steppers/verner89.hpp` — lines 71–77.

- [ ] **Step 1: Add the Verner89 test**

Append to `tests/ode/testFixedStep.cpp`, after the Verner78 test:

```cpp
TEST( OdeFixedStep, Verner89AlwaysAcceptedAtTightTol )
{
    using State = Eigen::Matrix< double, 1, 1 >;

    IntegratorConfig< double > cfg;
    cfg.initial_step = kH;
    cfg.abstol = cfg.reltol = 1e-30;

    auto integ = tax::ode::makeVerner89Integrator< double, 1, false,
                                                    FixedStep< double > >(
        identity_rhs< State >(), cfg );

    State x0; x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 1.0 );

    check_uniform_grid( sol, kH, /*expected_count=*/11u );
}
```

- [ ] **Step 2: Run — must FAIL**

Run:
```
cmake --build build --target test_ode_fixed_step
ctest --test-dir build -R '^test_ode_fixed_step$' --output-on-failure
```
Expected: `Verner89AlwaysAcceptedAtTightTol` throws the rejection-cap exception; `Verner78AlwaysAcceptedAtTightTol` still passes.

- [ ] **Step 3: Apply the same arm to Verner89**

In `include/tax/ode/steppers/verner89.hpp`, replace:

```cpp
        T h_next;
        if constexpr ( std::is_same_v< Controller, controllers::JorbaZou< T > > )
            h_next = h;  // JorbaZou is Taylor-only; no-op fallback.
        else
            h_next = controller_.next_step( h, out.err_norm, tol, Tab::order_emb );

        const bool accepted = out.err_norm <= tol;
```

with:

```cpp
        T    h_next;
        bool accepted;
        if constexpr ( std::is_same_v< Controller, controllers::FixedStep< T > > )
        {
            h_next   = h;
            accepted = true;
        }
        else if constexpr ( std::is_same_v< Controller, controllers::JorbaZou< T > > )
        {
            h_next   = h;  // JorbaZou is Taylor-only; no-op fallback.
            accepted = out.err_norm <= tol;
        }
        else
        {
            h_next   = controller_.next_step( h, out.err_norm, tol, Tab::order_emb );
            accepted = out.err_norm <= tol;
        }
```

- [ ] **Step 4: Run — must PASS**

Run:
```
ctest --test-dir build -R 'test_ode_(verner89|fixed_step)' --output-on-failure
```
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add tests/ode/testFixedStep.cpp include/tax/ode/steppers/verner89.hpp
git commit -m "ode/verner89: honour FixedStep controller (always-accepted)"
```

---

### Task 5: Fehlberg78 — failing test + wire-up

**Files:**
- Modify: `tests/ode/testFixedStep.cpp` (append).
- Modify: `include/tax/ode/steppers/fehlberg78.hpp` — lines 72–78.

- [ ] **Step 1: Add the test**

Append to `tests/ode/testFixedStep.cpp`:

```cpp
TEST( OdeFixedStep, Fehlberg78AlwaysAcceptedAtTightTol )
{
    using State = Eigen::Matrix< double, 1, 1 >;

    IntegratorConfig< double > cfg;
    cfg.initial_step = kH;
    cfg.abstol = cfg.reltol = 1e-30;

    auto integ = tax::ode::makeFehlberg78Integrator< double, 1, false,
                                                      FixedStep< double > >(
        identity_rhs< State >(), cfg );

    State x0; x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 1.0 );

    check_uniform_grid( sol, kH, /*expected_count=*/11u );
}
```

- [ ] **Step 2: Run — must FAIL**

Run:
```
cmake --build build --target test_ode_fixed_step
ctest --test-dir build -R '^test_ode_fixed_step$' --output-on-failure
```
Expected: only `Fehlberg78AlwaysAcceptedAtTightTol` throws; others pass.

- [ ] **Step 3: Apply the arm to Fehlberg78**

In `include/tax/ode/steppers/fehlberg78.hpp`, replace:

```cpp
        T h_next;
        if constexpr ( std::is_same_v< Controller, controllers::JorbaZou< T > > )
            h_next = h;  // JorbaZou is Taylor-only; no-op fallback.
        else
            h_next = controller_.next_step( h, out.err_norm, tol, Tab::order_emb );

        const bool accepted = out.err_norm <= tol;
```

with:

```cpp
        T    h_next;
        bool accepted;
        if constexpr ( std::is_same_v< Controller, controllers::FixedStep< T > > )
        {
            h_next   = h;
            accepted = true;
        }
        else if constexpr ( std::is_same_v< Controller, controllers::JorbaZou< T > > )
        {
            h_next   = h;  // JorbaZou is Taylor-only; no-op fallback.
            accepted = out.err_norm <= tol;
        }
        else
        {
            h_next   = controller_.next_step( h, out.err_norm, tol, Tab::order_emb );
            accepted = out.err_norm <= tol;
        }
```

- [ ] **Step 4: Run — must PASS**

Run:
```
ctest --test-dir build -R 'test_ode_(fehlberg78|fixed_step)' --output-on-failure
```
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add tests/ode/testFixedStep.cpp include/tax/ode/steppers/fehlberg78.hpp
git commit -m "ode/fehlberg78: honour FixedStep controller (always-accepted)"
```

---

### Task 6: Feagin12 — failing test + wire-up (preserves `err_for_ctrl` floor)

**Files:**
- Modify: `tests/ode/testFixedStep.cpp` (append).
- Modify: `include/tax/ode/steppers/feagin12.hpp` — lines 81–87.

- [ ] **Step 1: Add the test**

Append to `tests/ode/testFixedStep.cpp`:

```cpp
TEST( OdeFixedStep, Feagin12AlwaysAcceptedAtTightTol )
{
    using State = Eigen::Matrix< double, 1, 1 >;

    IntegratorConfig< double > cfg;
    cfg.initial_step = kH;
    cfg.abstol = cfg.reltol = 1e-30;

    auto integ = tax::ode::makeFeagin12Integrator< double, 1, false,
                                                    FixedStep< double > >(
        identity_rhs< State >(), cfg );

    State x0; x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 1.0 );

    check_uniform_grid( sol, kH, /*expected_count=*/11u );
}
```

- [ ] **Step 2: Run — must FAIL**

Run:
```
cmake --build build --target test_ode_fixed_step
ctest --test-dir build -R '^test_ode_fixed_step$' --output-on-failure
```
Expected: `Feagin12AlwaysAcceptedAtTightTol` throws; others pass.

- [ ] **Step 3: Apply the arm to Feagin12 (note: `err_for_ctrl` stays for the non-FixedStep controller path)**

In `include/tax/ode/steppers/feagin12.hpp`, replace:

```cpp
        T h_next;
        if constexpr ( std::is_same_v< Controller, controllers::JorbaZou< T > > )
            h_next = h;  // JorbaZou is Taylor-only; no-op fallback.
        else
            h_next = controller_.next_step( h, err_for_ctrl, tol, Tab::order_emb );

        const bool accepted = out.err_norm <= tol;
```

with:

```cpp
        T    h_next;
        bool accepted;
        if constexpr ( std::is_same_v< Controller, controllers::FixedStep< T > > )
        {
            h_next   = h;
            accepted = true;
        }
        else if constexpr ( std::is_same_v< Controller, controllers::JorbaZou< T > > )
        {
            h_next   = h;  // JorbaZou is Taylor-only; no-op fallback.
            accepted = out.err_norm <= tol;
        }
        else
        {
            h_next   = controller_.next_step( h, err_for_ctrl, tol, Tab::order_emb );
            accepted = out.err_norm <= tol;
        }
```

- [ ] **Step 4: Run — must PASS**

Run:
```
ctest --test-dir build -R 'test_ode_(feagin12|fixed_step)' --output-on-failure
```
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add tests/ode/testFixedStep.cpp include/tax/ode/steppers/feagin12.hpp
git commit -m "ode/feagin12: honour FixedStep controller (always-accepted)"
```

---

### Task 7: Feagin14 — failing test + wire-up

**Files:**
- Modify: `tests/ode/testFixedStep.cpp` (append).
- Modify: `include/tax/ode/steppers/feagin14.hpp` — lines 81–87.

- [ ] **Step 1: Add the test**

Append to `tests/ode/testFixedStep.cpp`:

```cpp
TEST( OdeFixedStep, Feagin14AlwaysAcceptedAtTightTol )
{
    using State = Eigen::Matrix< double, 1, 1 >;

    IntegratorConfig< double > cfg;
    cfg.initial_step = kH;
    cfg.abstol = cfg.reltol = 1e-30;

    auto integ = tax::ode::makeFeagin14Integrator< double, 1, false,
                                                    FixedStep< double > >(
        identity_rhs< State >(), cfg );

    State x0; x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 1.0 );

    check_uniform_grid( sol, kH, /*expected_count=*/11u );
}
```

- [ ] **Step 2: Run — must FAIL**

Run:
```
cmake --build build --target test_ode_fixed_step
ctest --test-dir build -R '^test_ode_fixed_step$' --output-on-failure
```
Expected: `Feagin14AlwaysAcceptedAtTightTol` throws; others pass.

- [ ] **Step 3: Apply the arm to Feagin14**

In `include/tax/ode/steppers/feagin14.hpp`, replace:

```cpp
        T h_next;
        if constexpr ( std::is_same_v< Controller, controllers::JorbaZou< T > > )
            h_next = h;  // JorbaZou is Taylor-only; no-op fallback.
        else
            h_next = controller_.next_step( h, err_for_ctrl, tol, Tab::order_emb );

        const bool accepted = out.err_norm <= tol;
```

with:

```cpp
        T    h_next;
        bool accepted;
        if constexpr ( std::is_same_v< Controller, controllers::FixedStep< T > > )
        {
            h_next   = h;
            accepted = true;
        }
        else if constexpr ( std::is_same_v< Controller, controllers::JorbaZou< T > > )
        {
            h_next   = h;  // JorbaZou is Taylor-only; no-op fallback.
            accepted = out.err_norm <= tol;
        }
        else
        {
            h_next   = controller_.next_step( h, err_for_ctrl, tol, Tab::order_emb );
            accepted = out.err_norm <= tol;
        }
```

- [ ] **Step 4: Run — must PASS**

Run:
```
ctest --test-dir build -R 'test_ode_(feagin14|fixed_step)' --output-on-failure
```
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add tests/ode/testFixedStep.cpp include/tax/ode/steppers/feagin14.hpp
git commit -m "ode/feagin14: honour FixedStep controller (always-accepted)"
```

---

### Task 8: TaylorStepper — failing test + wire-up

TaylorStepper has a different `step()` body (no embedded RK estimator). It currently selects the controller via:

```cpp
if constexpr ( std::is_same_v< Controller, controllers::JorbaZou< T > > )
    h_next = controller_.next_step( h, c_N_norm, c_Nm1_norm, tol, N );
else
    h_next = controller_.next_step( h, err_norm, tol, /*p_emb=*/N - 1 );

const bool accepted = err_norm <= tol;
```

We add the FixedStep arm in the same shape.

**Files:**
- Modify: `tests/ode/testFixedStep.cpp` (append).
- Modify: `include/tax/ode/steppers/taylor.hpp` — lines 155–163.

- [ ] **Step 1: Add the test**

Append to `tests/ode/testFixedStep.cpp`:

```cpp
TEST( OdeFixedStep, TaylorAlwaysAcceptedAtTightTol )
{
    using State = Eigen::Matrix< double, 1, 1 >;

    IntegratorConfig< double > cfg;
    cfg.initial_step = kH;
    cfg.abstol = cfg.reltol = 1e-30;

    auto integ = tax::ode::makeTaylorIntegrator< /*N=*/16, double, 1, false,
                                                  decltype( identity_rhs< State >() ) >(
        identity_rhs< State >(), cfg );

    // Override default JorbaZou with FixedStep — Taylor's makeTaylorIntegrator
    // does not expose a Controller template parameter, so instantiate the
    // Integrator type manually.
    using Stepper = tax::ode::TaylorStepper< 16, State, FixedStep< double > >;
    using Integ   = tax::ode::Integrator<
        Stepper, std::function< State( const State&, double ) >, false >;
    Integ integ_fs{ identity_rhs< State >(), cfg };

    State x0; x0( 0 ) = 1.0;
    auto sol = integ_fs.integrate( x0, 0.0, 1.0 );

    check_uniform_grid( sol, kH, /*expected_count=*/11u );
}
```

(The `makeTaylorIntegrator` line above is harmless: it constructs an integrator that is never used. The fixture under test is `integ_fs`. Keeping the helper-call shape makes the test read uniformly with the RK cases above.)

- [ ] **Step 2: Run — must FAIL**

Run:
```
cmake --build build --target test_ode_fixed_step
ctest --test-dir build -R '^test_ode_fixed_step$' --output-on-failure
```
Expected: `TaylorAlwaysAcceptedAtTightTol` throws; the five RK tests pass.

- [ ] **Step 3: Apply the arm to TaylorStepper**

In `include/tax/ode/steppers/taylor.hpp`, replace:

```cpp
    // --- 6. Step-size control: JorbaZou uses (c_N, c_{N-1}) directly;
    // every other controller uses err_norm via next_step(h, err, tol, p_emb).
    T h_next;
    if constexpr ( std::is_same_v< Controller, controllers::JorbaZou< T > > )
        h_next = controller_.next_step( h, c_N_norm, c_Nm1_norm, tol, N );
    else
        h_next = controller_.next_step( h, err_norm, tol, /*p_emb=*/N - 1 );

    const bool accepted = err_norm <= tol;
```

with:

```cpp
    // --- 6. Step-size control:
    //   FixedStep  — always accept, return h unchanged.
    //   JorbaZou   — uses (c_N, c_{N-1}) directly.
    //   any other  — uses err_norm via next_step(h, err, tol, p_emb).
    T    h_next;
    bool accepted;
    if constexpr ( std::is_same_v< Controller, controllers::FixedStep< T > > )
    {
        h_next   = h;
        accepted = true;
    }
    else if constexpr ( std::is_same_v< Controller, controllers::JorbaZou< T > > )
    {
        h_next   = controller_.next_step( h, c_N_norm, c_Nm1_norm, tol, N );
        accepted = err_norm <= tol;
    }
    else
    {
        h_next   = controller_.next_step( h, err_norm, tol, /*p_emb=*/N - 1 );
        accepted = err_norm <= tol;
    }
```

- [ ] **Step 4: Run — must PASS**

Run:
```
ctest --test-dir build -R 'test_ode_(taylor_stepper|fixed_step)' --output-on-failure
```
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add tests/ode/testFixedStep.cpp include/tax/ode/steppers/taylor.hpp
git commit -m "ode/taylor: honour FixedStep controller (always-accepted)"
```

---

### Task 9: Final regression sweep

- [ ] **Step 1: Run the entire ODE test suite**

```
ctest --test-dir build -R '^test_ode_' --output-on-failure
```
Expected: every existing ODE test still passes, and all six `OdeFixedStep.*` tests pass.

- [ ] **Step 2: Verify the build is clean**

```
cmake --build build 2>&1 | grep -E 'warning|error' || echo 'clean'
```
Expected: `clean` (or only pre-existing warnings unrelated to this slice).

- [ ] **Step 3: Done — no further commit needed**

---

## Self-review checklist

- Spec coverage: FixedStep type ✓ (Task 1); per-stepper arms ✓ (Tasks 3, 4, 5, 6, 7, 8); tests ✓ (Tasks 2–8, one per stepper).
- Placeholder scan: none.
- Type consistency: `FixedStep<T>` used uniformly; `controllers::FixedStep` namespacing matches the existing `controllers::JorbaZou` pattern.
- All factory call sites (`makeVerner78Integrator` etc.) survive — Slice A does not touch the factory API. Slice B's plan removes them and migrates call sites.
