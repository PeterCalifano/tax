# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the Eigen-backed container bindings — Vec and
Mat."""

import math

import numpy as np
import pytest

import tax


# ---------------------------------------------------------------------------
# tax.math submodule structure
# ---------------------------------------------------------------------------

def test_math_submodule_exposes_functions():
    assert hasattr(tax, "math")
    for name in (
        "sin", "cos", "tan", "sinh", "cosh", "tanh",
        "asin", "acos", "atan", "asinh", "acosh", "atanh",
        "exp", "log", "log10", "sqrt", "cbrt", "square", "cube",
        "pow", "atan2", "hypot", "abs", "erf",
    ):
        assert callable(getattr(tax.math, name)), name


def test_math_functions_not_at_top_level():
    """Math functions live under `tax.math`, not at the top level."""
    for name in ("sin", "cos", "exp", "log", "sqrt", "pow"):
        assert not hasattr(tax, name), f"`tax.{name}` should now be under tax.math"


# ---------------------------------------------------------------------------
# Vec
# ---------------------------------------------------------------------------

def test_vector_construction_and_indexing():
    x, y = tax.variables([1.0, 2.0], order=3)
    v = tax.Vec([x, y, x * y])
    assert len(v) == 3
    assert v[0].value() == 1.0
    assert v[1].value() == 2.0
    assert v[2].value() == 2.0

    # __setitem__
    v[2] = tax.math.sin(x)
    assert v[2].value() == pytest.approx(math.sin(1.0))


def test_vector_index_out_of_range_raises():
    x, _ = tax.variables([1.0, 2.0], order=3)
    v = tax.Vec([x, x])
    with pytest.raises(Exception):
        _ = v[5]
    with pytest.raises(Exception):
        v[5] = x


def test_vector_value_returns_numpy():
    x, y = tax.variables([0.5, 1.5], order=3)
    v = tax.Vec([x, y, x * y])
    out = v.value()
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [0.5, 1.5, 0.75])


def test_vector_eval_at_displacement():
    x, y = tax.variables([1.0, 2.0], order=3)
    v = tax.Vec([x, y, x * y])
    out = v.eval([0.1, 0.2])
    assert isinstance(out, np.ndarray)
    # x: 1 + 0.1 = 1.1; y: 2 + 0.2 = 2.2; x*y truncated.
    np.testing.assert_allclose(out[:2], [1.1, 2.2])
    # numpy input also works.
    np.testing.assert_allclose(v.eval(np.array([0.1, 0.2])), out)


def test_vector_derivative_per_component():
    x, y = tax.variables([1.0, 2.0], order=3)
    v = tax.Vec([x, y, x * y])
    # d/dx applied component-wise: [1, 0, y].
    d = v.derivative([1, 0])
    np.testing.assert_allclose(d, [1.0, 0.0, 2.0])
    # d/dy: [0, 1, x].
    np.testing.assert_allclose(v.derivative([0, 1]), [0.0, 1.0, 1.0])


def test_vector_jacobian_matches_module_level():
    x, y = tax.variables([1.0, 2.0], order=3)
    fs = [x, y, x * y]
    v = tax.Vec(fs)
    np.testing.assert_allclose(v.jacobian(), tax.jacobian(fs))


def test_vector_repr_round_trips_basic_info():
    x, _ = tax.variables([1.0, 2.0], order=3)
    v = tax.Vec([x, x])
    assert "Vec" in repr(v)
    assert "len=2" in repr(v)


def test_vector_repr_ravels_polynomial_form():
    x, y = tax.variables([1.0, 2.0], order=3)
    r = repr(tax.Vec([x, y]))
    # Each element appears on its own line with an index label and the
    # polynomial form (dx, truncation remainder, etc.).
    assert "0:" in r and "1:" in r
    assert "dx" in r          # has a variable
    assert "O(" in r          # truncation remainder


# ---------------------------------------------------------------------------
# Mat
# ---------------------------------------------------------------------------

def test_matrix_construction_and_indexing():
    x, y = tax.variables([1.0, 2.0], order=3)
    m = tax.Mat([[x, y], [x * y, x + y]])
    assert m.shape == (2, 2)
    assert m.rows == 2
    assert m.cols == 2
    assert m[0, 0].value() == 1.0
    assert m[1, 0].value() == 2.0
    assert m[1, 1].value() == 3.0

    # __setitem__
    m[0, 0] = tax.math.sin(x)
    assert m[0, 0].value() == pytest.approx(math.sin(1.0))


def test_matrix_empty_or_jagged_raises():
    with pytest.raises(Exception):
        tax.Mat([])
    x, _ = tax.variables([1.0, 2.0], order=2)
    with pytest.raises(Exception):
        tax.Mat([[x], [x, x]])


def test_matrix_value_returns_2d_numpy():
    x, y = tax.variables([1.0, 2.0], order=3)
    m = tax.Mat([[x, y], [x * y, x + y]])
    out = m.value()
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, [[1.0, 2.0], [2.0, 3.0]])


def test_matrix_eval_at_displacement():
    x, y = tax.variables([1.0, 2.0], order=3)
    m = tax.Mat([[x, y], [x * y, x + y]])
    out = m.eval([0.1, 0.2])
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out[0, 0], 1.1)
    np.testing.assert_allclose(out[0, 1], 2.2)
    np.testing.assert_allclose(out[1, 1], 3.3)


def test_matrix_derivative_per_element():
    x, y = tax.variables([1.0, 2.0], order=3)
    m = tax.Mat([[x * y, y], [x, x + y]])
    # d/dx: [[y, 0], [1, 1]] -> at expansion = [[2, 0], [1, 1]].
    np.testing.assert_allclose(m.derivative([1, 0]), [[2.0, 0.0], [1.0, 1.0]])
    # d/dy: [[x, 1], [0, 1]] -> at expansion = [[1, 1], [0, 1]].
    np.testing.assert_allclose(m.derivative([0, 1]), [[1.0, 1.0], [0.0, 1.0]])


def test_matrix_repr_includes_shape():
    x, _ = tax.variables([1.0, 2.0], order=2)
    m = tax.Mat([[x, x], [x, x]])
    r = repr(m)
    assert "Mat" in r
    assert "rows=2" in r and "cols=2" in r


def test_matrix_repr_ravels_polynomial_form():
    x, y = tax.variables([1.0, 2.0], order=3)
    r = repr(tax.Mat([[x, y], [x * y, x + y]]))
    # Every cell is listed as (row,col): <polynomial>.
    for label in ("(0,0):", "(0,1):", "(1,0):", "(1,1):"):
        assert label in r
    assert "dx" in r
    assert "O(" in r
