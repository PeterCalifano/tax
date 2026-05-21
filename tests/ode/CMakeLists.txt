# tests/ode/CMakeLists.txt
#
# ODE integrator unit tests for Stage 2a. Each test executable links
# the tax INTERFACE target and gtest_main. Registered via
# tax_add_test() from the parent tests/CMakeLists.txt.

tax_add_test(test_ode_config SOURCES testConfig.cpp)
tax_add_test(test_ode_concepts SOURCES testConcepts.cpp)
tax_add_test(test_ode_solution SOURCES testSolution.cpp)
tax_add_test(test_ode_controllers SOURCES testControllers.cpp)
tax_add_test(test_ode_taylor_stepper SOURCES testTaylorStepper.cpp)
tax_add_test(test_ode_integrator_basic SOURCES testIntegratorBasic.cpp)
tax_add_test(test_ode_integrator_dense SOURCES testIntegratorDense.cpp)
tax_add_test(test_ode_events_zerocrossing SOURCES testEventsZeroCrossing.cpp)
tax_add_test(test_ode_events_everystep SOURCES testEventsEveryStep.cpp)
tax_add_test(test_ode_rk_helpers SOURCES testRKHelpers.cpp)
tax_add_test(test_ode_verner78_stepper SOURCES testVerner78Stepper.cpp)
tax_add_test(test_ode_verner89_stepper SOURCES testVerner89Stepper.cpp)
tax_add_test(test_ode_fehlberg78_stepper SOURCES testFehlberg78Stepper.cpp)
tax_add_test(test_ode_feagin12_stepper SOURCES testFeagin12Stepper.cpp)
tax_add_test(test_ode_feagin14_stepper SOURCES testFeagin14Stepper.cpp)
tax_add_test(test_ode_two_body_kepler SOURCES testTwoBodyKepler.cpp)
tax_add_test(test_ode_cr3bp_propagation SOURCES testCR3BPPropagation.cpp)
