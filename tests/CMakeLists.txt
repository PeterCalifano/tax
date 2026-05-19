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
# Tests are added by subsequent slices via tax_add_test(name SOURCES files...).
function(tax_add_test name)
    cmake_parse_arguments(T "" "" "SOURCES" ${ARGN})
    add_executable(${name} ${T_SOURCES})
    target_link_libraries(${name} PRIVATE tax GTest::gtest GTest::gtest_main)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

tax_add_test(test_multi_index SOURCES core/test_multi_index.cpp)
tax_add_test(test_enumeration SOURCES core/test_enumeration.cpp)
tax_add_test(test_taylor_expansion_ctor SOURCES core/test_taylor_expansion_ctor.cpp)
tax_add_test(test_taylor_expansion_accessors SOURCES core/test_taylor_expansion_accessors.cpp)
