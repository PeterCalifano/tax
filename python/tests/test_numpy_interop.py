# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the numpy integration on the dynamic TaylorExpansion bindings.

Covered:
  - coeffs() and derivatives() return numpy.ndarray
  - from_coeffs(numpy_array, order, size) round-trips
  - inputs that previously took a list of floats accept numpy arrays too
  - module-level gradient / hessian / jacobian return numpy arrays
"""

import math

import numpy as np
import pytest

import tax


# ---------------------------------------------------------------------------
# Return-type checks
# ---------------------------------------------------------------------------

def test_coeffs_returns_numpy_ndarray():
    x = tax.variable(0.3, 0, 5, 1)
    f = tax.sin(x)
    c = f.coeffs()
    assert isinstance(c, np.ndarray)
    assert c.dtype == np.float64
    assert c.ndim == 1
    assert c.shape == (6,)


def test_derivatives_returns_numpy_ndarray():
    x = tax.variable(0.5, 0, 4, 1)
    f = tax.exp(x)
    d = f.derivatives()
    assert isinstance(d, np.ndarray)
    assert d.dtype == np.float64
    assert d.shape == (5,)
    # f.derivatives[k] = c_k * k! for univariate exp.
    np.testing.assert_allclose(
        d, [math.exp(0.5) / math.factorial(k) * math.factorial(k) for k in range(5)]
    )


# ---------------------------------------------------------------------------
# numpy arrays as inputs
# ---------------------------------------------------------------------------

def test_variables_accepts_numpy_array():
    x0 = np.array([1.0, 2.0, 3.0])
    coords = tax.variables(x0, order=3)
    assert len(coords) == 3
    for i, c in enumerate(coords):
        assert c.value() == float(i + 1)


def test_at_accepts_numpy_array():
    x, y = tax.variables([0.5, 0.3], order=4)
    f = tax.sin(x * y)
    dx = np.array([0.05, -0.02])
    v_np = f.at(dx)
    v_list = f.at([0.05, -0.02])
    assert v_np == pytest.approx(v_list)


def test_derivative_accepts_numpy_array():
    x, y = tax.variables([0.5, 0.3], order=4)
    f = tax.exp(x + y)
    alpha = np.array([1, 1], dtype=np.int32)
    # Both numpy int and list should work.
    assert f.derivative(alpha) == pytest.approx(f.derivative([1, 1]))


def test_coeff_accepts_numpy_array():
    x, y = tax.variables([1.0, 2.0], order=3)
    f = x * y
    alpha = np.array([1, 1], dtype=np.int32)
    assert f.coeff(alpha) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# from_coeffs round-trip
# ---------------------------------------------------------------------------

def test_from_coeffs_round_trip():
    x = tax.variable(1.5, 0, 4, 1)
    f = tax.sin(x)
    cs = f.coeffs()
    g = tax.from_coeffs(cs, order=4, size=1)
    np.testing.assert_allclose(g.coeffs(), cs)
    assert g.value() == pytest.approx(f.value())


def test_from_coeffs_wrong_length_raises():
    bad = np.zeros(5)
    with pytest.raises(Exception):
        tax.from_coeffs(bad, order=4, size=2)  # expects C(6,2) = 15


# ---------------------------------------------------------------------------
# Module-level gradient / hessian / jacobian
# ---------------------------------------------------------------------------

def test_gradient_returns_numpy_vector():
    x, y, z = tax.variables([0.5, 0.3, 0.2], order=4)
    f = tax.sin(x * y) + tax.exp(z)
    g = tax.gradient(f)
    assert isinstance(g, np.ndarray)
    assert g.shape == (3,)
    # df/dz at expansion = exp(0.2).
    assert g[2] == pytest.approx(math.exp(0.2))


def test_hessian_returns_numpy_matrix():
    x, y = tax.variables([0.5, 0.3], order=4)
    f = tax.exp(x + y)
    H = tax.hessian(f)
    assert isinstance(H, np.ndarray)
    assert H.shape == (2, 2)
    # d^2/dxdy exp(x+y) at expansion = exp(0.5 + 0.3).
    ref = math.exp(0.5 + 0.3)
    np.testing.assert_allclose(H, np.full((2, 2), ref), rtol=1e-12)


def test_jacobian_returns_numpy_matrix():
    x, y = tax.variables([1.0, 2.0], order=3)
    fs = [x, y, x * y]
    J = tax.jacobian(fs)
    assert isinstance(J, np.ndarray)
    assert J.shape == (3, 2)
    # Rows: d[x]/[dx, dy], d[y]/[dx, dy], d[x*y]/[dx, dy] at (1, 2).
    np.testing.assert_allclose(J, np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]]))


def test_jacobian_empty_raises():
    with pytest.raises(Exception):
        tax.jacobian([])


# ---------------------------------------------------------------------------
# Workflow: build a polynomial, get the coefficients, slice, rebuild
# ---------------------------------------------------------------------------

def test_workflow_numpy_slice_and_rebuild():
    x, y = tax.variables([0.5, 0.3], order=4)
    f = tax.sin(x * y) + tax.exp(x + y)
    cs = f.coeffs()                      # numpy array

    # Manipulate via numpy and rebuild.
    cs2 = cs.copy()
    cs2[0] = 0.0                         # zero out the constant term
    g = tax.from_coeffs(cs2, order=4, size=2)
    assert g.value() == 0.0
    # Non-constant coefficients are preserved.
    np.testing.assert_allclose(g.coeffs()[1:], cs[1:])
