# SPDX-License-Identifier: BSD-3-Clause
"""Reductions on Vec / Mat (sum, mean, prod, min, max, argmin, argmax, trace,
diagonal, row_sums, col_sums) and the `tax.la` factory functions (zeros,
identity, diag).
"""

import math

import numpy as np
import pytest

import tax


# ---------------------------------------------------------------------------
# Vec reductions
# ---------------------------------------------------------------------------

def _vec():
    x, y, z = tax.variables([1.0, 2.0, 3.0], order=2)
    return tax.la.Vec([x, y, z])


def test_vec_sum_method_and_free_function():
    v = _vec()
    assert v.sum().value() == pytest.approx(6.0)
    assert tax.la.sum(v).value() == pytest.approx(6.0)


def test_vec_mean():
    v = _vec()
    assert v.mean().value() == pytest.approx(2.0)
    assert tax.la.mean(v).value() == pytest.approx(2.0)


def test_vec_prod():
    v = _vec()
    assert v.prod().value() == pytest.approx(1.0 * 2.0 * 3.0)


def test_vec_min_max():
    v = _vec()
    assert v.min().value() == pytest.approx(1.0)
    assert v.max().value() == pytest.approx(3.0)
    assert v.argmin() == 0
    assert v.argmax() == 2
    # Free-function variants return the element, not the index.
    assert isinstance(tax.la.min(v), tax.TaylorExpansion)
    assert tax.la.max(v).value() == pytest.approx(3.0)


def test_vec_min_max_picks_by_value_not_address():
    # Build a vec whose element with smallest .value() is at index 1.
    x, y, z = tax.variables([5.0, -1.0, 3.0], order=2)
    v = tax.la.Vec([x, y, z])
    assert v.argmin() == 1
    assert v.min().value() == pytest.approx(-1.0)
    assert v.argmax() == 0
    assert v.max().value() == pytest.approx(5.0)


def test_vec_reductions_on_empty_raise():
    empty = tax.la.Vec([])
    for fn in (empty.sum, empty.mean, empty.prod, empty.min, empty.max,
               empty.argmin, empty.argmax):
        with pytest.raises(Exception):
            fn()


# ---------------------------------------------------------------------------
# Mat reductions
# ---------------------------------------------------------------------------

def _mat():
    x, y, z = tax.variables([1.0, 2.0, 3.0], order=2)
    # [[1, 2], [2, 3]] in values
    return tax.la.Mat([[x, y], [y, z]])


def test_mat_sum():
    m = _mat()
    assert m.sum().value() == pytest.approx(1 + 2 + 2 + 3)
    assert tax.la.sum(m).value() == pytest.approx(8.0)


def test_mat_trace_method_and_free_function():
    m = _mat()
    assert m.trace().value() == pytest.approx(1.0 + 3.0)  # diagonal: 1, 3
    assert tax.la.trace(m).value() == pytest.approx(4.0)


def test_mat_trace_requires_square():
    x, y = tax.variables([1.0, 2.0], order=2)
    rect = tax.la.Mat([[x, y, x], [y, x, y]])  # 2x3
    with pytest.raises(Exception):
        rect.trace()


def test_mat_diagonal():
    m = _mat()
    d = m.diagonal()
    assert isinstance(d, tax.la.Vec)
    np.testing.assert_allclose(d.value(), [1.0, 3.0])
    # Rectangular case: returns min(rows, cols) entries.
    x, y = tax.variables([1.0, 2.0], order=2)
    rect = tax.la.Mat([[x, y, x], [y, x, y]])  # 2x3, diag length = 2
    assert rect.diagonal().value().shape == (2,)


def test_mat_row_and_col_sums():
    m = _mat()
    np.testing.assert_allclose(m.row_sums().value(), [1 + 2, 2 + 3])
    np.testing.assert_allclose(m.col_sums().value(), [1 + 2, 2 + 3])


def test_mat_is_square():
    assert _mat().is_square is True
    x, y = tax.variables([1.0, 2.0], order=2)
    rect = tax.la.Mat([[x, y, x], [y, x, y]])
    assert rect.is_square is False


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def test_zeros_vec_factory():
    v = tax.la.zeros(4, 3, 2)
    assert isinstance(v, tax.la.Vec)
    assert len(v) == 4
    np.testing.assert_allclose(v.value(), [0.0] * 4)
    for i in range(4):
        assert v[i].order == 3 and v[i].size == 2


def test_zeros_mat_factory():
    m = tax.la.zeros(2, 3, 4, 2)
    assert isinstance(m, tax.la.Mat)
    assert m.shape == (2, 3)
    np.testing.assert_allclose(m.value(), np.zeros((2, 3)))
    assert m[0, 0].order == 4 and m[0, 0].size == 2


def test_identity_factory():
    I = tax.la.identity(3, 2, 2)
    assert I.shape == (3, 3)
    np.testing.assert_allclose(I.value(), np.eye(3))
    # Off-diagonal entries are exact zeros.
    assert I[0, 1].value() == 0.0
    assert I[1, 0].value() == 0.0
    # Identity at the expansion point behaves as identity: I @ v == v.
    x, y, z = tax.variables([5.0, 6.0, 7.0], order=2)
    v = tax.la.Vec([x, y, z])
    np.testing.assert_allclose((I @ v).value(), v.value())


def test_diag_factory_from_vec():
    x, y, z = tax.variables([4.0, 5.0, 6.0], order=2)
    d = tax.la.Vec([x, y, z])
    D = tax.la.diag(d)
    assert D.shape == (3, 3)
    # Diagonal contains the input values; off-diagonal entries are zero.
    np.testing.assert_allclose(D.value(), np.diag([4.0, 5.0, 6.0]))
    # tax.la.diagonal is the inverse extraction.
    np.testing.assert_allclose(tax.la.diagonal(D).value(), d.value())


def test_diag_factory_empty_raises():
    with pytest.raises(Exception):
        tax.la.diag(tax.la.Vec([]))


# ---------------------------------------------------------------------------
# Composed identities
# ---------------------------------------------------------------------------

def test_identity_trace_equals_size():
    I = tax.la.identity(5, 2, 2)
    assert I.trace().value() == pytest.approx(5.0)


def test_zeros_sum_is_zero():
    v = tax.la.zeros(7, 3, 2)
    assert v.sum().value() == 0.0
