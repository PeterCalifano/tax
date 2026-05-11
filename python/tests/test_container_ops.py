# SPDX-License-Identifier: BSD-3-Clause
"""Operator coverage for the Eigen-backed Vec and Mat container bindings:
element-wise arithmetic with another container, a TaylorExpansion scalar,
a plain float, or a numpy float array; matrix multiplication; transpose.
"""

import numpy as np
import pytest

import tax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vec_and_mat(order=3):
    x, y = tax.variables([1.0, 2.0], order=order)
    v = tax.la.Vec([x, y, x * y])
    m = tax.la.Mat([[x, y], [x * y, x + y]])
    return x, y, v, m


# ---------------------------------------------------------------------------
# Vec arithmetic
# ---------------------------------------------------------------------------

def test_vec_add_sub_mul_div_with_vec():
    _, _, v, _ = _make_vec_and_mat()
    w = tax.la.Vec([v[2], v[1], v[0]])  # [x*y, y, x]
    np.testing.assert_allclose((v + w).value(), [1 + 2, 2 + 2, 2 + 1])
    np.testing.assert_allclose((v - w).value(), [1 - 2, 2 - 2, 2 - 1])
    np.testing.assert_allclose((v * w).value(), [1 * 2, 2 * 2, 2 * 1])
    np.testing.assert_allclose((v / w).value(), [1 / 2, 2 / 2, 2 / 1])


def test_vec_unary_negation():
    _, _, v, _ = _make_vec_and_mat()
    np.testing.assert_allclose((-v).value(), [-1.0, -2.0, -2.0])


def test_vec_size_mismatch_raises():
    _, _, v, _ = _make_vec_and_mat()
    short = tax.la.Vec([v[0]])
    with pytest.raises(Exception):
        _ = v + short


def test_vec_broadcast_scalar_te():
    x, _, v, _ = _make_vec_and_mat()
    # value(x) = 1, so v + x adds 1 to every value.
    np.testing.assert_allclose((v + x).value(), [2.0, 3.0, 3.0])
    # And reverse.
    np.testing.assert_allclose((x + v).value(), [2.0, 3.0, 3.0])
    # Subtraction.
    np.testing.assert_allclose((v - x).value(), [0.0, 1.0, 1.0])
    np.testing.assert_allclose((x - v).value(), [0.0, -1.0, -1.0])
    # Multiplication / division.
    np.testing.assert_allclose((v * x).value(), [1.0, 2.0, 2.0])
    np.testing.assert_allclose((x * v).value(), [1.0, 2.0, 2.0])


def test_vec_broadcast_float():
    _, _, v, _ = _make_vec_and_mat()
    np.testing.assert_allclose((v + 3.0).value(), [4.0, 5.0, 5.0])
    np.testing.assert_allclose((3.0 + v).value(), [4.0, 5.0, 5.0])
    np.testing.assert_allclose((v - 0.5).value(), [0.5, 1.5, 1.5])
    np.testing.assert_allclose((0.5 - v).value(), [-0.5, -1.5, -1.5])
    np.testing.assert_allclose((v * 2.0).value(), [2.0, 4.0, 4.0])
    np.testing.assert_allclose((2.0 * v).value(), [2.0, 4.0, 4.0])
    np.testing.assert_allclose((v / 2.0).value(), [0.5, 1.0, 1.0])


def test_vec_in_place_ops():
    _, _, v, _ = _make_vec_and_mat()
    v += tax.la.Vec([v[0], v[1], v[2]])
    np.testing.assert_allclose(v.value(), [2.0, 4.0, 4.0])
    v -= tax.la.Vec([tax.constant(1.0, 3, 2)] * 3)
    np.testing.assert_allclose(v.value(), [1.0, 3.0, 3.0])
    v *= 2.0
    np.testing.assert_allclose(v.value(), [2.0, 6.0, 6.0])
    v /= 2.0
    np.testing.assert_allclose(v.value(), [1.0, 3.0, 3.0])


def test_vec_numpy_array_arithmetic():
    _, _, v, _ = _make_vec_and_mat()
    a = np.array([10.0, 20.0, 30.0])
    np.testing.assert_allclose((v + a).value(), [11.0, 22.0, 32.0])
    np.testing.assert_allclose((v - a).value(), [-9.0, -18.0, -28.0])
    np.testing.assert_allclose((v * a).value(), [10.0, 40.0, 60.0])
    np.testing.assert_allclose((v / a).value(), [0.1, 0.1, 2.0 / 30.0])


def test_vec_dot_product_via_matmul():
    _, _, v, _ = _make_vec_and_mat()
    w = tax.la.Vec([v[2], v[1], v[0]])
    # values: v=[1,2,2], w=[2,2,1] -> dot = 1*2 + 2*2 + 2*1 = 8
    assert (v @ w).value() == pytest.approx(8.0)


def test_vec_at_mat_product():
    _, _, _, m = _make_vec_and_mat()  # 2x2 mat
    x, y = tax.variables([1.0, 2.0], order=3)
    v = tax.la.Vec([x, y])
    # v @ m -> [v0*m00 + v1*m10, v0*m01 + v1*m11]
    # values: [1*1 + 2*2, 1*2 + 2*3] = [5, 8]
    result = (v @ m).value()
    np.testing.assert_allclose(result, [5.0, 8.0])


# ---------------------------------------------------------------------------
# Mat arithmetic
# ---------------------------------------------------------------------------

def test_mat_add_sub_mul_div_with_mat():
    _, _, _, m = _make_vec_and_mat()
    np.testing.assert_allclose((m + m).value(), [[2, 4], [4, 6]])
    np.testing.assert_allclose((m - m).value(), [[0, 0], [0, 0]])
    np.testing.assert_allclose((m * m).value(), [[1, 4], [4, 9]])
    np.testing.assert_allclose((m / m).value(), [[1, 1], [1, 1]])


def test_mat_unary_negation():
    _, _, _, m = _make_vec_and_mat()
    np.testing.assert_allclose((-m).value(), [[-1, -2], [-2, -3]])


def test_mat_shape_mismatch_raises():
    x, _, _, m = _make_vec_and_mat()
    wide = tax.la.Mat([[x, x, x], [x, x, x]])
    with pytest.raises(Exception):
        _ = m + wide


def test_mat_broadcast_scalar_te():
    x, _, _, m = _make_vec_and_mat()
    # value(x) = 1
    np.testing.assert_allclose((m + x).value(), [[2, 3], [3, 4]])
    np.testing.assert_allclose((x + m).value(), [[2, 3], [3, 4]])
    np.testing.assert_allclose((m - x).value(), [[0, 1], [1, 2]])
    np.testing.assert_allclose((x - m).value(), [[0, -1], [-1, -2]])
    np.testing.assert_allclose((m * x).value(), [[1, 2], [2, 3]])


def test_mat_broadcast_float():
    _, _, _, m = _make_vec_and_mat()
    np.testing.assert_allclose((m + 3.0).value(), [[4, 5], [5, 6]])
    np.testing.assert_allclose((3.0 + m).value(), [[4, 5], [5, 6]])
    np.testing.assert_allclose((m * 0.5).value(), [[0.5, 1], [1, 1.5]])
    np.testing.assert_allclose((0.5 * m).value(), [[0.5, 1], [1, 1.5]])
    np.testing.assert_allclose((m / 2.0).value(), [[0.5, 1], [1, 1.5]])


def test_mat_in_place_ops():
    _, _, _, m = _make_vec_and_mat()
    m += m
    np.testing.assert_allclose(m.value(), [[2, 4], [4, 6]])
    m *= 0.5
    np.testing.assert_allclose(m.value(), [[1, 2], [2, 3]])


def test_mat_numpy_array_arithmetic():
    _, _, _, m = _make_vec_and_mat()
    a = np.array([[10.0, 20.0], [30.0, 40.0]])
    np.testing.assert_allclose((m + a).value(), [[11, 22], [32, 43]])
    np.testing.assert_allclose((m - a).value(), [[-9, -18], [-28, -37]])
    np.testing.assert_allclose((m * a).value(), [[10, 40], [60, 120]])


def test_mat_at_mat_product():
    x, y, _, m = _make_vec_and_mat()  # 2x2
    # Build a 2x3 mat.
    m2 = tax.la.Mat([[x, y, x * y], [y, x, x + y]])
    p = m @ m2
    # Rows: m_row_0 = [x, y], m_row_1 = [x*y, x+y]
    # cols of m2: [x, y, x*y]^T = col0, [y, x, ...]
    # p(0,0) = x*x + y*y = 1 + 4 = 5
    # p(0,1) = x*y + y*x = 2 + 2 = 4
    # p(0,2) = x*(x*y) + y*(x+y) = 2 + 2*3 = 8
    # p(1,0) = (x*y)*x + (x+y)*y = 2 + 3*2 = 8
    # p(1,1) = (x*y)*y + (x+y)*x = 4 + 3 = 7
    # p(1,2) = (x*y)*(x*y) + (x+y)*(x+y) = 4 + 9 = 13
    np.testing.assert_allclose(p.value(), [[5, 4, 8], [8, 7, 13]])


def test_mat_at_vec_product():
    x, y, _, m = _make_vec_and_mat()
    v = tax.la.Vec([x, y])
    r = m @ v
    # m=[[1,2],[2,3]], v=[1,2] -> [1*1+2*2, 2*1+3*2] = [5, 8]
    np.testing.assert_allclose(r.value(), [5, 8])


def test_mat_transpose():
    x, y, _, m = _make_vec_and_mat()
    t = m.T
    assert t.shape == (2, 2)
    np.testing.assert_allclose(t.value(), [[1, 2], [2, 3]])
    # m is symmetric in value but not in coeffs: (m[0,1] is y, m[1,0] is x*y).
    # After transpose, t[0,1] should equal original m[1,0] (= x*y, value 2).
    assert t[0, 1].value() == 2.0
    assert t[1, 0].value() == 2.0
    assert m[0, 1].value() == 2.0   # m[0,1] is y (value 2)
    assert m[1, 0].value() == 2.0   # m[1,0] is x*y (value 2)


def test_mat_matmul_inner_dim_mismatch_raises():
    x, y, _, m = _make_vec_and_mat()
    bad = tax.la.Mat([[x, x, x]])  # 1x3
    with pytest.raises(Exception):
        _ = m @ bad  # 2x2 @ 1x3


# ---------------------------------------------------------------------------
# Eigen-style slicing
# ---------------------------------------------------------------------------

def test_vec_segment_returns_fresh_vec():
    _, _, v, _ = _make_vec_and_mat()
    seg = v.segment(0, 2)
    np.testing.assert_allclose(seg.value(), [1.0, 2.0])
    # Returned vector is independent — mutating it doesn't disturb the source.
    seg[0] = tax.constant(99.0, 3, 2)
    assert v[0].value() == 1.0


def test_vec_segment_out_of_range_raises():
    _, _, v, _ = _make_vec_and_mat()
    with pytest.raises(Exception):
        v.segment(0, 100)
    with pytest.raises(Exception):
        v.segment(-1, 1)


def test_mat_block_returns_fresh_mat():
    _, _, _, m = _make_vec_and_mat()
    b = m.block(0, 0, 1, 2)
    assert b.shape == (1, 2)
    np.testing.assert_allclose(b.value(), [[1.0, 2.0]])


def test_mat_block_out_of_range_raises():
    _, _, _, m = _make_vec_and_mat()
    with pytest.raises(Exception):
        m.block(0, 0, 100, 1)
    with pytest.raises(Exception):
        m.block(-1, 0, 1, 1)


def test_mat_row_and_col_return_vec():
    _, _, _, m = _make_vec_and_mat()
    r = m.row(0)
    c = m.col(1)
    np.testing.assert_allclose(r.value(), [1.0, 2.0])
    np.testing.assert_allclose(c.value(), [2.0, 3.0])


def test_mat_row_col_out_of_range_raises():
    _, _, _, m = _make_vec_and_mat()
    with pytest.raises(Exception):
        m.row(99)
    with pytest.raises(Exception):
        m.col(-1)


# ---------------------------------------------------------------------------
# Vec norms
# ---------------------------------------------------------------------------

def test_vec_squared_norm_value_matches_sum_of_squares():
    import math
    x, y = tax.variables([3.0, 4.0], order=3)
    v = tax.la.Vec([x, y, x * y])
    sq = v.squared_norm()
    # Returns a TaylorExpansion; its value should be 3² + 4² + 12² = 169.
    assert isinstance(sq, tax.TaylorExpansion)
    assert sq.value() == pytest.approx(3.0 * 3 + 4.0 * 4 + 12.0 * 12)


def test_vec_norm_value_matches_l2():
    import math
    x, y = tax.variables([3.0, 4.0], order=3)
    v = tax.la.Vec([x, y])
    n = v.norm()
    assert isinstance(n, tax.TaylorExpansion)
    assert n.value() == pytest.approx(5.0)
    # 3D version: sqrt(3² + 4² + 12²) = sqrt(169) = 13.
    v3 = tax.la.Vec([x, y, x * y])
    assert v3.norm().value() == pytest.approx(13.0)


def test_vec_norm_empty_raises():
    # Build an explicitly empty vec via list — construct then bypass via Vec().
    # The C++ ctor with an empty list yields an empty vector, so norm() should
    # raise on emptiness.
    with pytest.raises(Exception):
        tax.la.Vec([]).norm()
    with pytest.raises(Exception):
        tax.la.Vec([]).squared_norm()


# ---------------------------------------------------------------------------
# numpy-on-the-left matmul: routes through __rmatmul__ instead of producing
# a numpy object array.
# ---------------------------------------------------------------------------

def test_numpy_matrix_matmul_vec_returns_vec():
    _, _, v, _ = _make_vec_and_mat()
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    out = R @ v
    assert isinstance(out, tax.la.Vec)
    # values: R @ [1, 2, 2]^T = [-2, 1, 2]
    np.testing.assert_allclose(out.value(), [-2.0, 1.0, 2.0])


def test_numpy_identity_matmul_vec_is_identity():
    _, _, v, _ = _make_vec_and_mat()
    out = np.eye(3) @ v
    assert isinstance(out, tax.la.Vec)
    np.testing.assert_allclose(out.value(), v.value())


def test_vec_matmul_numpy_2d():
    _, _, v, _ = _make_vec_and_mat()
    M = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # 3x2
    out = v @ M  # 1x3 @ 3x2 -> 1x2
    assert isinstance(out, tax.la.Vec)
    # values: [1, 2, 2] @ M = [1*1 + 2*0 + 2*1, 1*0 + 2*1 + 2*1] = [3, 4]
    np.testing.assert_allclose(out.value(), [3.0, 4.0])


def test_vec_dot_numpy_1d():
    _, _, v, _ = _make_vec_and_mat()
    a = np.array([1.0, 2.0, 3.0])
    out = v @ a
    assert isinstance(out, tax.TaylorExpansion)
    assert out.value() == pytest.approx(1.0 * 1 + 2.0 * 2 + 3.0 * 2)
    # And reverse.
    out2 = a @ v
    assert isinstance(out2, tax.TaylorExpansion)
    assert out2.value() == pytest.approx(1.0 * 1 + 2.0 * 2 + 3.0 * 2)


def test_numpy_matmul_mat_returns_mat():
    _, _, _, m = _make_vec_and_mat()
    P = np.array([[1.0, 0.0], [0.0, 2.0]])
    out = P @ m
    assert isinstance(out, tax.la.Mat)
    # P @ m = [[1*1+0*2, 1*2+0*3], [0*1+2*2, 0*2+2*3]] = [[1,2],[4,6]]
    np.testing.assert_allclose(out.value(), [[1.0, 2.0], [4.0, 6.0]])


def test_mat_matmul_numpy_2d():
    _, _, _, m = _make_vec_and_mat()
    P = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    out = m @ P  # 2x2 @ 2x3 -> 2x3
    assert isinstance(out, tax.la.Mat)
    assert out.shape == (2, 3)


def test_numpy_1d_matmul_mat_returns_vec():
    _, _, _, m = _make_vec_and_mat()
    row = np.array([1.0, 2.0])
    out = row @ m
    assert isinstance(out, tax.la.Vec)
    # row=[1,2] @ m=[[1,2],[2,3]] = [1*1+2*2, 1*2+2*3] = [5, 8]
    np.testing.assert_allclose(out.value(), [5.0, 8.0])


def test_mat_matmul_numpy_1d_returns_vec():
    _, _, _, m = _make_vec_and_mat()
    col = np.array([1.0, 2.0])
    out = m @ col
    assert isinstance(out, tax.la.Vec)
    # m @ col = [1*1+2*2, 2*1+3*2] = [5, 8]
    np.testing.assert_allclose(out.value(), [5.0, 8.0])


def test_numpy_matmul_vec_shape_mismatch_raises():
    _, _, v, _ = _make_vec_and_mat()
    with pytest.raises(Exception):
        np.eye(5) @ v


def test_array_ufunc_set_to_none():
    """`__array_ufunc__ = None` makes numpy defer ufuncs to our reflected ops.
    Verify directly on the class objects."""
    assert tax.la.Vec.__array_ufunc__ is None
    assert tax.la.Mat.__array_ufunc__ is None
