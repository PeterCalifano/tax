# tests/regression/CMakeLists.txt
#
# Built only when TAX_BUILD_REGRESSIONS=ON. Slice 1 ships a sentinel test
# that does not depend on DACE so the wiring can be verified in isolation.
# Slice 2 replaces it with the real DACE-backed regression tests.

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

# tax_add_regression(name SOURCES files...)
#
# Slice 1: links tax + GTest only. Slice 2 extends this helper to also
# link the resolved DACE target.
function(tax_add_regression name)
    cmake_parse_arguments(R "" "" "SOURCES" ${ARGN})
    add_executable(${name} ${R_SOURCES})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${name} PRIVATE tax GTest::gtest GTest::gtest_main)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

tax_add_regression(test_regression_sentinel SOURCES testSentinel.cpp)
