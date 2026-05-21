# tests/regression/CMakeLists.txt
#
# Built only when TAX_BUILD_REGRESSIONS=ON. Each test links the resolved
# DACE target (set in the root CMakeLists.txt under TAX_DACE_TARGET) and
# compares tax coefficient outputs against DACE references.

include(FetchContent)
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

if(NOT TAX_DACE_TARGET)
    message(FATAL_ERROR
        "tests/regression requires TAX_DACE_TARGET to be resolved by the root CMakeLists.txt")
endif()

# tax_add_regression(name SOURCES files...)
function(tax_add_regression name)
    cmake_parse_arguments(R "" "" "SOURCES" ${ARGN})
    add_executable(${name} ${R_SOURCES})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${name} PRIVATE
        tax
        ${TAX_DACE_TARGET}
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

tax_add_regression(test_regression_univariate   SOURCES testUnivariate.cpp)
tax_add_regression(test_regression_multivariate SOURCES testMultivariate.cpp)
tax_add_regression(test_regression_deriv_integ  SOURCES testDerivInteg.cpp)
tax_add_regression(test_regression_eval         SOURCES testEval.cpp)
tax_add_regression(test_regression_vector_calc  SOURCES testEigenVectorCalc.cpp)
tax_add_regression(test_regression_invert       SOURCES testInvert.cpp)
