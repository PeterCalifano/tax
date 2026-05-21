# tests/ode/CMakeLists.txt
#
# ODE integrator unit tests for Stage 2a. Each test executable links
# the tax INTERFACE target and gtest_main. Registered via
# tax_add_test() from the parent tests/CMakeLists.txt.

tax_add_test(test_ode_config SOURCES testConfig.cpp)
