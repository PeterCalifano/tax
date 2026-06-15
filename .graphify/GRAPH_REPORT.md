# Graph Report - .  (2026-06-15)

## Corpus Check
- label mode - file stats not available

## Summary
- 797 nodes · 868 edges · 68 communities detected
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 23 edges (avg confidence: 0.82)
- Token cost: 0 input · 0 output
- Edge kinds: contains: 388 · calls: 235 · references: 162 · implements: 44 · semantically_similar_to: 12 · rationale_for: 11 · conceptually_related_to: 7 · method: 7 · shares_data_with: 2


## Graph Freshness
- Built from Git commit: `ca4c57b`
- Compare this hash to `git rev-parse HEAD` before trusting freshness-sensitive graph output.
## God Nodes (most connected - your core abstractions)
1. `CMake Function: tax_add_test` - 66 edges
2. `ODE API Reference` - 17 edges
3. `TaylorExpansion<T, N, M, Storage>` - 13 edges
4. `CMake Function: tax_add_example` - 13 edges
5. `ODE Mathematical Foundations` - 12 edges
6. `ads::refine (Propagate-Then-Assess)` - 11 edges
7. `ODE Methods & Benchmarks` - 11 edges
8. `Two-Body Problem Tutorial` - 9 edges
9. `CMake Function: tax_add_regression` - 9 edges
10. `center()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `Example: three_body_taylor` --semantically_similar_to--> `Tutorial: Three-Body Problem`  [INFERRED] [semantically similar]
  examples/three_body/taylor.cpp → docs/tutorials/three_body.md
- `Example: two_body_refine` --semantically_similar_to--> `Tutorial: Parallel ADS by Refinement`  [INFERRED] [semantically similar]
  examples/two_body/refine.cpp → docs/tutorials/two_body_refine.md
- `Example: two_body_taylor` --semantically_similar_to--> `Tutorial: Two-Body Problem`  [INFERRED] [semantically similar]
  examples/two_body/taylor.cpp → docs/tutorials/two_body.md
- `CMake Function: tax_add_test` --semantically_similar_to--> `CMake Function: tax_add_example`  [INFERRED] [semantically similar]
  tests/CMakeLists.txt → examples/CMakeLists.txt
- `Test: test_ode_cr3bp_propagation` --semantically_similar_to--> `Example: three_body_taylor`  [INFERRED] [semantically similar]
  tests/ode/problems/test_cr3bp_propagation.cpp → examples/three_body/taylor.cpp

## Hyperedges (group relationships)
- **Cauchy Product Dispatch Variants** — concept_cauchy_product_loop, concept_cauchy_product_unroll, concept_cauchy_product_stencil [EXTRACTED 1.00]
- **ADS In-Flight Split Criteria** — concept_truncation_criterion, concept_nli_criterion, concept_nonlinearity_index [EXTRACTED 1.00]
- **ODE Stepper, Controller, and Event System** — concept_integrator, concept_stepper_concept, concept_event, concept_solution [INFERRED 0.90]
- **ADS Driver, Split Criterion, and Tree** — concept_ads_propagate, concept_ads_truncation_criterion, concept_ads_tree, concept_flow_map [INFERRED 0.85]
- **TaylorExpansion, Kernel, and Operator Layer** — concept_taylor_expansion, concept_cauchy_product, concept_recurrence_stencil, concept_layer_architecture [INFERRED 0.85]
- **Tax CMake Build System: test, example, and regression registration functions** — cmake_tax_add_test, cmake_tax_add_example, cmake_tax_add_regression [INFERRED 0.85]
- **DACE Regression Testing Infrastructure** — tests_regression_cmakelists, dep_dace, cmake_option_tax_build_regressions [EXTRACTED 1.00]
- **ODE End-to-End Problem Test Suite** — test_ode_two_body_kepler, test_ode_cr3bp_propagation, test_ode_cr3bp_events [EXTRACTED 1.00]
- **Two-Body Tutorial Visualization Set** — img_two_body_ads, img_two_body_ads_orbit, img_two_body_flow, img_two_body_leaves [INFERRED 0.90]
- **Three-Body Tutorial Visualization Set** — img_three_body_ads, img_three_body_flow, img_three_body_leaves [INFERRED 0.90]
- **Two-Body ADS Refine Tutorial Visualisation Suite (small domain)** — img_two_body_refine_gif, img_two_body_refine_convergence, img_two_body_refine_regions [INFERRED 0.90]
- **Two-Body ADS Refine Tutorial Visualisation Suite (big domain)** — img_two_body_refine_big_gif, img_two_body_refine_big_convergence, img_two_body_refine_big_regions [INFERRED 0.90]

## Communities

### Community 0 - "ADS Test Suite"
Cohesion: 0.03
Nodes (63): CMake Function: tax_add_test, Test: test_ads_box, Test: test_ads_criteria, Test: test_ads_da_state, Test: test_ads_driver, Test: test_ads_leaf_tree, Test: test_ads_merge, Test: test_ads_nonlinearity_index (+55 more)

### Community 1 - "ODE Methods & Controllers"
Cohesion: 0.10
Nodes (38): CR3BP ODE Benchmark, Brent Root Finder, H211b (Soderlind) Step Controller, I (Integral) Step Controller, JorbaZou Step Controller, PI (Gustafsson) Step Controller, Cubic-Hermite Continuous Extension, Dense Output (+30 more)

### Community 2 - "ADS Module Core"
Cohesion: 0.13
Nodes (30): ads::merge Post-Pass Merger, Automatic Domain Splitting (tax::ads), ads::propagate (Classic In-Flight ADS), ads::refine (Propagate-Then-Assess), AdsTree<Payload, M, T>, TruncationCriterion, CoefficientMatchCriterion, CoefficientMatchCriterion (+22 more)

### Community 3 - "TaylorExpansion Core Type"
Cohesion: 0.12
Nodes (26): STE<N, M> Alias (Sparse), TE<N, M> Alias (Dense), Cauchy Product (Multiplication), Cauchy Product Loop (constexpr-safe), Cauchy Product Stencil (M>=2), Cauchy Product Unrolled (M==1), Degree-by-Degree Recurrence Relations, Dense Storage (std::array) (+18 more)

### Community 4 - "Documentation & Tutorials"
Cohesion: 0.08
Nodes (25): CMake Function: tax_add_example, Doc: ADS Refinement, Doc: Mathematical Foundations (Core), Doc: Dense vs Sparse Storage, Doc: Architecture (Internals), Doc: Kernels and Recurrences (Internals), Doc: ODE Events, Doc: ODE Mathematical Foundations (+17 more)

### Community 6 - "Examples I/O Helpers"
Cohesion: 0.16
Nodes (14): CMake Option: TAX_BUILD_REGRESSIONS, CMake Variable: TAX_DACE_TARGET, Root CMakeLists.txt, CMake Function: tax_add_regression, DACE v2.1.0, Eigen3 Library, tax Library (header-only C++23), Regression Test: test_regression_deriv_integ (+6 more)

### Community 7 - "ADS Concepts & Visuals"
Cohesion: 0.18
Nodes (4): jsonArray(), Stopwatch, writeJsonArray(), writeRunJson()

### Community 8 - "TaylorExpansion Methods"
Cohesion: 0.22
Nodes (13): Automatic Domain Splitting (ADS), LOADS (NLI-based ADS variant), Taylor Tube / Flow Tube Propagation, Three-Body CR3BP Problem, TruncationCriterion (Wittig 2015), Two-Body Kepler Problem, Wittig 2015 ADS Reference, Three-Body ADS Orbit Plot (+5 more)

### Community 9 - "Validation Plot Scripts"
Cohesion: 0.15
Nodes (3): TaylorExpansion, TaylorExpansion< T, N, M, storage::Dense >, TaylorExpansion< T, N, M, storage::Sparse >

### Community 10 - "ADS Refine Benchmarks"
Cohesion: 0.31
Nodes (11): cells_at_P(), cells_at_tol(), figure_envelope(), figure_error(), figure_timing(), load_validation(), main(), _plot_vs_P() (+3 more)

### Community 11 - "Three-Body Plot Scripts"
Cohesion: 0.64
Nodes (10): BM_ClassicAds(), BM_OrderNli(), BM_OrderTruncation(), BM_RefineCoeff(), BM_RefineCoeff4way(), BM_RefineVolume(), center(), cfg() (+2 more)

### Community 12 - "Two-Body Plot Scripts"
Cohesion: 0.27
Nodes (10): draw_frame(), load(), main(), plot_ads(), plot_flow(), plot_leaves(), Single polynomial vs ADS partition at the Moon-approach breakdown., Leaf counts vs time, against the manifold's e^{lambda t} stretching. (+2 more)

### Community 13 - "WSB Search Example"
Cohesion: 0.27
Nodes (10): load(), main(), plot_ads(), plot_ads_orbit(), plot_flow(), plot_leaves(), All ADS snapshots overlaid on the full orbit., Leaf count growth per snapshot, ADS vs LOADS. (+2 more)

### Community 14 - "Transcendental Series Kernels"
Cohesion: 0.40
Nodes (10): earthHillR(), main(), makeIc(), moonOrbitR(), normaliseDeg(), rhs(), runSweep(), scanOne() (+2 more)

### Community 18 - "ADS Refine Driver"
Cohesion: 0.33
Nodes (5): check_constant_term_matches_double(), check_linear_term_matches_analytical_stm(), make_harmonic_ic_da(), make_kepler_ic_da(), make_kepler_ic_double()

### Community 19 - "ODE CR3BP Benchmarks"
Cohesion: 0.43
Nodes (6): assess(), drive(), propagateLeaf(), refine(), RefineDriver, run()

### Community 20 - "DACE Regression Suite"
Cohesion: 0.39
Nodes (5): BM_RefFehlberg78_I_1e12(), BM_TaylorH211b_N24(), BM_TaylorPI_N12(), endpoint_error(), ensure_reference()

### Community 21 - "Refine Convergence Scripts"
Cohesion: 0.43
Nodes (7): cloud_limits(), draw_convergence(), main(), make_convergence_png(), make_gif(), make_regions_png(), Converged final-time region of each method, overlaid on Monte Carlo.

### Community 22 - "ADS Parallel Driver"
Cohesion: 0.48
Nodes (5): AdsDriver, driveParallel(), driveSerial(), run(), stepLeaf()

### Community 23 - "ADS Refine Criteria"
Cohesion: 0.43
Nodes (5): acceptable(), childMismatch(), imageVolume(), splitDim(), topDegreeSplitDim()

### Community 24 - "Multi-Index Utilities"
Cohesion: 0.52
Nodes (6): binom(), DegreeOf(), flatIndex(), numMonomials(), totalDegree(), unflatIndex()

### Community 25 - "Kepler Two-Body Tests"
Cohesion: 0.52
Nodes (6): check_invariants(), make_ic(), make_rhs(), specific_angmom(), specific_energy(), TEST()

### Community 26 - "Orbit Plot Scripts"
Cohesion: 0.48
Nodes (6): draw_panel(), load(), main(), panel_xy_limits(), True anomaly ν(t) for an elliptic orbit, ν ∈ [0, 2π)., true_anomaly()

### Community 27 - "Two-Body Validation"
Cohesion: 0.48
Nodes (4): posErrors(), runAdsLike(), runTaylor(), toLocal()

### Community 28 - "Two-Body Common Helpers"
Cohesion: 0.33
Nodes (2): icBox(), icCenter()

### Community 29 - "ADS Refine Tests"
Cohesion: 0.67
Nodes (5): evalLeaf(), icBox(), rhs(), scalarReference(), TEST()

### Community 31 - "Three-Body Common Helpers"
Cohesion: 0.40
Nodes (2): icBox(), icCenter()

### Community 32 - "Box Region Plot Scripts"
Cohesion: 0.53
Nodes (5): draw_panel(), load(), main(), panel_xy_limits(), Union x/y over reference orbits + polygons. All in km, Earth-centred.

### Community 33 - "DA State Utilities"
Cohesion: 0.60
Nodes (3): binom(), split(), substituteAxis()

### Community 34 - "Nonlinearity Index"
Cohesion: 0.70
Nodes (4): jacobianVariationBound(), linRowBound(), nliSplitDim(), nonlinearityIndex()

### Community 35 - "ADS Parallel Tests"
Cohesion: 0.70
Nodes (4): expectTreesEqual(), rhs(), runWith(), TEST()

### Community 37 - "Eigen Derivative Helpers"
Cohesion: 0.50
Nodes (2): derivative(), jacobian()

### Community 39 - "Sparse Container Core"
Cohesion: 0.40
Nodes (1): SparseContainer

### Community 40 - "WSB Plot Scripts"
Cohesion: 0.70
Nodes (4): draw_panel(), load(), main(), panel_xy_limits()

### Community 41 - "WSB Common Helpers"
Cohesion: 0.50
Nodes (2): icBox(), icCenter()

### Community 42 - "ADS Split Criteria"
Cohesion: 0.67
Nodes (2): shouldSplit(), totalTopDegreeMass()

### Community 43 - "ADS CSV I/O"
Cohesion: 0.83
Nodes (3): collectLeafSpans(), writeBoxCountCsv(), writeTreeCsv()

### Community 44 - "ADS Driver Tests"
Cohesion: 1.00
Nodes (3): rhs(), scalarReference(), TEST()

### Community 45 - "Monomial Enumeration"
Cohesion: 0.50
Nodes (4): CMake FetchContent, Dependency: Google Test v1.17.0, tests/CMakeLists.txt, tests/ode/CMakeLists.txt

### Community 50 - "ODE Solution Type"
Cohesion: 0.50
Nodes (1): Integrator

### Community 51 - "Fixed-Step ODE Tests"
Cohesion: 0.50
Nodes (3): Solution, Solution< Stepper, State, /*Dense=*/false >, Solution< Stepper, State, /*Dense=*/true >

### Community 52 - "ODE Event Triggers"
Cohesion: 0.67
Nodes (2): check_uniform_grid(), TEST()

### Community 53 - "CR3BP Problem Setup"
Cohesion: 0.67
Nodes (2): dir_match(), ZeroCrossing()

### Community 55 - "Eigen Regression Tests"
Cohesion: 0.83
Nodes (3): check_jacobi_preserved(), compute_reference(), TEST()

### Community 56 - "Two-Body Refine Example"
Cohesion: 0.83
Nodes (3): daceCoeff1(), daceCoeff2(), TEST()

### Community 57 - "ADS Box Type"
Cohesion: 0.83
Nodes (3): leafInit(), main(), polygonArea()

### Community 59 - "ADS Split Event"
Cohesion: 1.00
Nodes (2): maxCoeffDiff(), merge()

### Community 62 - "ADS Leaf Tree Tests"
Cohesion: 1.00
Nodes (2): evalTe(), TEST()

### Community 63 - "Nonlinearity Index Tests"
Cohesion: 1.00
Nodes (2): TEST(), unitBox()

### Community 65 - "ADS Tree Structure"
Cohesion: 1.00
Nodes (2): makeQuadraticState(), TEST()

### Community 66 - "Static ODE Tests"
Cohesion: 0.67
Nodes (1): AdsTree

### Community 67 - "Cauchy Stencil Kernel"
Cohesion: 1.00
Nodes (2): harmonic_rhs(), TEST()

### Community 74 - "ODE CSV I/O"
Cohesion: 0.67
Nodes (1): Event

### Community 78 - "Univariate Regression Tests"
Cohesion: 1.00
Nodes (2): scaleToUnit(), TEST()

### Community 79 - "Taylor Stepper"
Cohesion: 1.00
Nodes (2): scaleToUnit(), TEST()

### Community 84 - "Enumeration Tests"
Cohesion: 1.00
Nodes (2): Benchmarks CMakeLists.txt, Google Benchmark

### Community 150 - "DACE Target Variable"
Cohesion: 1.00
Nodes (1): Box<T, M>

### Community 151 - "ADS Box Concept"
Cohesion: 1.00
Nodes (1): Leaf<Payload, M, T>

### Community 152 - "ADS Leaf Concept"
Cohesion: 1.00
Nodes (1): Eigen Integration (tax::la)

### Community 153 - "Eigen Integration Facade"
Cohesion: 1.00
Nodes (1): Flat Index (graded-lex)

### Community 154 - "Flat Index Concept"
Cohesion: 1.00
Nodes (1): MultiIndex<M>

### Community 155 - "MultiIndex Type"
Cohesion: 1.00
Nodes (1): numMonomials(N, M) = C(N+M, M)

### Community 157 - "Core Concepts Header"
Cohesion: 1.00
Nodes (1): Google Test v1.17

### Community 161 - "Feagin Butcher Tableaus"
Cohesion: 1.00
Nodes (1): Two-Body ADS Orbit Leaf Snapshot Plot

### Community 180 - "Eigen LA Facade"
Cohesion: 1.00
Nodes (1): Documentation CI Workflow

### Community 181 - "ODE Facade Header"
Cohesion: 1.00
Nodes (1): Regressions CI Workflow

### Community 182 - "Core Umbrella Header"
Cohesion: 1.00
Nodes (1): Sanitizers CI Workflow

### Community 183 - "Community 183"
Cohesion: 1.00
Nodes (1): Tests CI Workflow

## Knowledge Gaps
- **142 isolated node(s):** `Moon + L1 markers shared by both spatial figures (labelled for legends).`, `IC box image under one flow polynomial, drifting from L1 to the Moon.`, `Single polynomial vs ADS partition at the Moon-approach breakdown.`, `Leaf counts vs time, against the manifold's e^{lambda t} stretching.`, `Reference orbit + the IC-box image under the single flow polynomial.` (+137 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Two-Body Common Helpers`** (2 nodes): `icBox()`, `icCenter()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Three-Body Common Helpers`** (2 nodes): `icBox()`, `icCenter()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Eigen Derivative Helpers`** (2 nodes): `derivative()`, `jacobian()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Sparse Container Core`** (1 nodes): `SparseContainer`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `WSB Common Helpers`** (2 nodes): `icBox()`, `icCenter()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `ADS Split Criteria`** (2 nodes): `shouldSplit()`, `totalTopDegreeMass()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `ODE Solution Type`** (1 nodes): `Integrator`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `ODE Event Triggers`** (2 nodes): `check_uniform_grid()`, `TEST()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `CR3BP Problem Setup`** (2 nodes): `dir_match()`, `ZeroCrossing()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `ADS Split Event`** (2 nodes): `maxCoeffDiff()`, `merge()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `ADS Leaf Tree Tests`** (2 nodes): `evalTe()`, `TEST()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Nonlinearity Index Tests`** (2 nodes): `TEST()`, `unitBox()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `ADS Tree Structure`** (2 nodes): `makeQuadraticState()`, `TEST()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Static ODE Tests`** (1 nodes): `AdsTree`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Cauchy Stencil Kernel`** (2 nodes): `harmonic_rhs()`, `TEST()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `ODE CSV I/O`** (1 nodes): `Event`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Univariate Regression Tests`** (2 nodes): `scaleToUnit()`, `TEST()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Taylor Stepper`** (2 nodes): `scaleToUnit()`, `TEST()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Enumeration Tests`** (2 nodes): `Benchmarks CMakeLists.txt`, `Google Benchmark`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `DACE Target Variable`** (1 nodes): `Box<T, M>`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `ADS Box Concept`** (1 nodes): `Leaf<Payload, M, T>`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `ADS Leaf Concept`** (1 nodes): `Eigen Integration (tax::la)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Eigen Integration Facade`** (1 nodes): `Flat Index (graded-lex)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Flat Index Concept`** (1 nodes): `MultiIndex<M>`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `MultiIndex Type`** (1 nodes): `numMonomials(N, M) = C(N+M, M)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Core Concepts Header`** (1 nodes): `Google Test v1.17`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Feagin Butcher Tableaus`** (1 nodes): `Two-Body ADS Orbit Leaf Snapshot Plot`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Eigen LA Facade`** (1 nodes): `Documentation CI Workflow`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `ODE Facade Header`** (1 nodes): `Regressions CI Workflow`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Core Umbrella Header`** (1 nodes): `Sanitizers CI Workflow`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 183`** (1 nodes): `Tests CI Workflow`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CMake Function: tax_add_test` connect `ADS Test Suite` to `Examples I/O Helpers`, `Documentation & Tutorials`, `Monomial Enumeration`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Why does `CMake Function: tax_add_regression` connect `Examples I/O Helpers` to `ADS Test Suite`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **Why does `tax Library (header-only C++23)` connect `Examples I/O Helpers` to `ADS Module Core`, `TaylorExpansion Core Type`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `CMake Function: tax_add_test` (e.g. with `CMake Function: tax_add_regression` and `CMake Function: tax_add_example`) actually correct?**
  _`CMake Function: tax_add_test` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `TaylorExpansion<T, N, M, Storage>` (e.g. with `Polynomial Flow Map` and `Gradient Helper`) actually correct?**
  _`TaylorExpansion<T, N, M, Storage>` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Moon + L1 markers shared by both spatial figures (labelled for legends).`, `IC box image under one flow polynomial, drifting from L1 to the Moon.`, `Single polynomial vs ADS partition at the Moon-approach breakdown.` to the rest of the system?**
  _142 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `ADS Test Suite` be split into smaller, more focused modules?**
  _Cohesion score 0.031746031746031744 - nodes in this community are weakly interconnected._