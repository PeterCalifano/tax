# examples/CMakeLists.txt
#
# Each example is a single source file linking against the header-only
# tax library. Run via `./two_body_simple_taylor` etc. — they write CSV
# output into the current working directory.

function(tax_add_example name)
    cmake_parse_arguments(E "" "" "SOURCES" ${ARGN})
    add_executable(${name} ${E_SOURCES})
    target_link_libraries(${name} PRIVATE tax)
endfunction()

tax_add_example(two_body_simple_taylor SOURCES two_body/simple_taylor.cpp)
tax_add_example(two_body_ads           SOURCES two_body/ads.cpp)
tax_add_example(two_body_loads         SOURCES two_body/loads.cpp)
