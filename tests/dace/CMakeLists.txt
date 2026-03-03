set(DACE_TEST_SOURCES
    testUnivariate.cpp
)

include(FetchContent)

FetchContent_Declare(
    DACE
    GIT_REPOSITORY https://github.com/dacelib/dace.git
    GIT_TAG v2.1.0
)
FetchContent_MakeAvailable(DACE)

# Match upstream recommendation from DACE docs:
#   add_library(dace::dace ALIAS dace)
#   add_library(dace::dace_s ALIAS dace_s)
if(TARGET dace AND NOT TARGET dace::dace)
    add_library(dace::dace ALIAS dace)
endif()
if(TARGET dace_s AND NOT TARGET dace::dace_s)
    add_library(dace::dace_s ALIAS dace_s)
endif()

if(TARGET dace::dace_s)
    set(_dace_target dace::dace_s)
elseif(TARGET dace_s)
    set(_dace_target dace_s)
elseif(TARGET dace::dace)
    set(_dace_target dace::dace)
elseif(TARGET dace)
    set(_dace_target dace)
else()
    message(FATAL_ERROR "TAX_DACE_TEST is ON but no DACE target was found after FetchContent(dacelib/dace@v2.1.0).")
endif()

message(STATUS "DACE target for TAX_DACE_TEST: ${_dace_target}")

include(GoogleTest)

foreach(test_file IN LISTS DACE_TEST_SOURCES)
    get_filename_component(test_name ${test_file} NAME_WE)

    add_executable(${test_name} ${test_file})
    target_link_libraries(${test_name}
        PRIVATE
            tax
            ${_dace_target}
            GTest::gtest_main
    )
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    gtest_discover_tests(${test_name})
endforeach()
