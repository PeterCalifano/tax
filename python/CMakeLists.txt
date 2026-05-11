cmake_minimum_required(VERSION 3.24)

project(tax_python LANGUAGES CXX)

find_package(
    Python 3
    REQUIRED
    COMPONENTS Interpreter Development.Module
)

include(GNUInstallDirs)

#
# Locate nanobind using the active Python interpreter.
#
execute_process(
    COMMAND
        "${Python_EXECUTABLE}"
        -c
        "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE _nanobind_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _nanobind_rc
)

if(NOT _nanobind_rc EQUAL 0)
    message(FATAL_ERROR
        "TAX_BUILD_PYTHON=ON but importing nanobind failed.\n"
        "Install it with:\n"
        "  pip install nanobind"
    )
endif()

list(APPEND CMAKE_PREFIX_PATH "${_nanobind_dir}")

find_package(nanobind CONFIG REQUIRED)

#
# Build the extension module.
#
nanobind_add_module(
    _tax
    src/tax_module.cpp
    src/bind_te.cpp
    src/bind_vec.cpp
    src/bind_mat.cpp
    src/bind_math.cpp
    src/bind_la.cpp
    src/bind_factories.cpp
)

target_link_libraries(
    _tax
    PRIVATE
    tax::tax
)

target_compile_features(
    _tax
    PRIVATE
    cxx_std_23
)

#
# Stage the extension alongside `__init__.py` in the build tree so the
# package is importable as `tax` directly from `${CMAKE_BINARY_DIR}/python`
# (used by the ctest entry below; the `pip install .` path uses the
# install() rules below).
#
set_target_properties(_tax PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/tax"
)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/tax/__init__.py"
    "${CMAKE_BINARY_DIR}/python/tax/__init__.py"
    COPYONLY
)

#
# Helpful diagnostics.
#
message(STATUS "Python executable : ${Python_EXECUTABLE}")
message(STATUS "Python sitearch   : ${Python_SITEARCH}")
message(STATUS "Python sitelib    : ${Python_SITELIB}")

#
# Install compiled extension into site-packages/tax
#
install(
    TARGETS _tax
    LIBRARY DESTINATION "${Python_SITEARCH}/tax"
)

#
# Install pure Python package files.
#
install(
    FILES
        tax/__init__.py
    DESTINATION
        "${Python_SITELIB}/tax"
)

#
# Python tests
#
if(TAX_BUILD_TEST)

    execute_process(
        COMMAND
            "${Python_EXECUTABLE}"
            -c
            "import pytest"
        RESULT_VARIABLE _has_pytest
        OUTPUT_QUIET
        ERROR_QUIET
    )

    if(_has_pytest EQUAL 0)

        add_test(
            NAME python_bindings
            COMMAND
                "${Python_EXECUTABLE}"
                -m pytest -q
                "${CMAKE_CURRENT_SOURCE_DIR}/tests"
        )

        #
        # Tests import from the staged build tree (which has both
        # `__init__.py` and the compiled `_tax.so` next to each other).
        #
        set_tests_properties(
            python_bindings
            PROPERTIES
            ENVIRONMENT
                "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
        )

    else()

        message(STATUS
            "pytest not available in ${Python_EXECUTABLE}; "
            "skipping python_bindings tests."
        )

    endif()

endif()