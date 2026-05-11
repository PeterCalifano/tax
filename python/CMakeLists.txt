# Python bindings for tax via nanobind.
#
# Built only when `TAX_BUILD_PYTHON=ON`. Uses FindPython + nanobind's CMake
# helpers (`nanobind_add_module`) to produce an extension module `_tax`. The
# Python-level package wrapper lives in `python/tax/__init__.py`.

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

# Locate nanobind via the Python interpreter:
#   `python -c "import nanobind; print(nanobind.cmake_dir())"`
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE _nanobind_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _nanobind_rc)

if(NOT _nanobind_rc EQUAL 0)
    message(FATAL_ERROR
        "TAX_BUILD_PYTHON=ON but `python -c 'import nanobind'` failed.\n"
        "Install with: pip install nanobind")
endif()

list(APPEND CMAKE_PREFIX_PATH "${_nanobind_dir}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_tax src/tax_module.cpp)
target_link_libraries(_tax PRIVATE tax::tax)
target_compile_features(_tax PRIVATE cxx_std_23)
set_target_properties(_tax PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/tax")

# Wheel install path used by scikit-build-core for `pip install .`.
install(TARGETS _tax LIBRARY DESTINATION tax)

# Stage the Python package alongside the compiled module so users can run
#   PYTHONPATH=build/python python -c "import tax"
# from the build directory.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tax/__init__.py"
               "${CMAKE_BINARY_DIR}/python/tax/__init__.py" COPYONLY)

# Hook the Python tests into ctest. Use `${Python_EXECUTABLE} -m pytest` so
# the runner uses the same interpreter the extension was built against (which
# has nanobind and any test deps like numpy installed).
if(TAX_BUILD_TEST)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import pytest"
        RESULT_VARIABLE _has_pytest
        OUTPUT_QUIET ERROR_QUIET)
    if(_has_pytest EQUAL 0)
        add_test(
            NAME python_bindings
            COMMAND "${Python_EXECUTABLE}" -m pytest -q
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests")
        set_tests_properties(python_bindings PROPERTIES
            ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python")
    else()
        message(STATUS "pytest not available in ${Python_EXECUTABLE}; "
                       "skipping python_bindings ctest target.")
    endif()
endif()
