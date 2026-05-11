# SPDX-License-Identifier: BSD-3-Clause
"""Coverage for the tax.la submodule + the Vec.dot / Vec.cross / Mat.norm
methods. These all share kernels with the underlying TaylorExpansion arithmetic
so the tests focus on shape correctness, error handling, and the standard
identities (||(3,4,0)|| = 5; e_x × e_y = e_z; etc.).
"""

import math

import numpy as np
import pytest

import tax


def _make_vecs():
    x, y, z = tax.variables([3.0, 4.0, 0.0], order=2)
    v = tax.la.Vec([x, y, z])     # value (3, 4, 0)
    w = tax.la.Vec([y, x, z])     # value (4, 3, 0)
    return x, y, z, v, w


# ---------------------------------------------------------------------------
# Submodule surface
# ---------------------------------------------------------------------------

def test_la_submodule_present():
    assert hasattr(tax, "la")
    import types
    assert isinstance(tax.la, types.ModuleType)


def test_la_exposes_vec_and_mat():
    # The container types live exclusively under `tax.la` — not at the
    # top-level `tax` namespace.
    assert hasattr(tax.la, "Vec")
    assert hasattr(tax.la, "Mat")
    assert not hasattr(tax, "Vec")
    assert not hasattr(tax, "Mat")


def test_la_exposes_free_functions():
    for name in ("norm", "dot", "cross"):
        assert callable(getattr(tax.la, name))


# ---------------------------------------------------------------------------
# tax.la.norm — Vec L2 + Mat Frobenius
# ---------------------------------------------------------------------------

def test_la_norm_vec_returns_taylor_expansion():
    _, _, _, v, _ = _make_vecs()
    n = tax.la.norm(v)
    assert isinstance(n, tax.TaylorExpansion)
    assert n.value() == pytest.approx(5.0)


def test_la_norm_mat_returns_frobenius():
    x, y, z, _, _ = _make_vecs()
    m = tax.la.Mat([[x, y], [y, z]])  # values [[3,4],[4,0]]
    n = tax.la.norm(m)
    assert isinstance(n, tax.TaylorExpansion)
    # sqrt(3^2 + 4^2 + 4^2 + 0^2) = sqrt(41)
    assert n.value() == pytest.approx(math.sqrt(41.0))


def test_mat_norm_method_matches_la_norm():
    x, y, z, _, _ = _make_vecs()
    m = tax.la.Mat([[x, y], [y, z]])
    assert m.norm().value() == pytest.approx(tax.la.norm(m).value())
    # squared_norm should be exactly the sum of squared values, before sqrt.
    assert m.squared_norm().value() == pytest.approx(41.0)


def test_la_norm_empty_raises():
    with pytest.raises(Exception):
        tax.la.norm(tax.la.Vec([]))


# ---------------------------------------------------------------------------
# tax.la.dot — Vec @ Vec
# ---------------------------------------------------------------------------

def test_la_dot_vec_vec():
    _, _, _, v, w = _make_vecs()
    # (3, 4, 0) · (4, 3, 0) = 12 + 12 + 0 = 24
    assert tax.la.dot(v, w).value() == pytest.approx(24.0)


def test_la_dot_vec_numpy():
    _, _, _, v, _ = _make_vecs()
    assert tax.la.dot(v, np.array([1.0, 2.0, 3.0])).value() == pytest.approx(
        3 * 1 + 4 * 2 + 0 * 3
    )


def test_vec_dot_method_matches_la_dot():
    _, _, _, v, w = _make_vecs()
    assert v.dot(w).value() == pytest.approx(tax.la.dot(v, w).value())


def test_la_dot_size_mismatch_raises():
    _, _, _, v, _ = _make_vecs()
    short = tax.la.Vec([v[0]])
    with pytest.raises(Exception):
        tax.la.dot(v, short)


# ---------------------------------------------------------------------------
# tax.la.cross — 3-D cross product
# ---------------------------------------------------------------------------

def test_la_cross_basis_vectors():
    """e_x × e_y = e_z."""
    o0 = tax.constant(0.0, 2, 3)
    o1 = tax.constant(1.0, 2, 3)
    ex = tax.la.Vec([o1, o0, o0])
    ey = tax.la.Vec([o0, o1, o0])
    ez = tax.la.Vec([o0, o0, o1])
    np.testing.assert_allclose(tax.la.cross(ex, ey).value(), ez.value())
    np.testing.assert_allclose(tax.la.cross(ey, ex).value(), (-ez).value())


def test_la_cross_with_numpy():
    o0 = tax.constant(0.0, 2, 3)
    o1 = tax.constant(1.0, 2, 3)
    ex = tax.la.Vec([o1, o0, o0])
    ey = np.array([0.0, 1.0, 0.0])
    out = tax.la.cross(ex, ey)
    assert isinstance(out, tax.la.Vec)
    np.testing.assert_allclose(out.value(), [0.0, 0.0, 1.0])


def test_vec_cross_method_matches_la_cross():
    o0 = tax.constant(0.0, 2, 3)
    o1 = tax.constant(1.0, 2, 3)
    ex = tax.la.Vec([o1, o0, o0])
    ey = tax.la.Vec([o0, o1, o0])
    np.testing.assert_allclose(ex.cross(ey).value(), tax.la.cross(ex, ey).value())


def test_la_cross_requires_size_three():
    x, y = tax.variables([1.0, 2.0], order=2)
    v2 = tax.la.Vec([x, y])
    with pytest.raises(Exception):
        tax.la.cross(v2, v2)


# ---------------------------------------------------------------------------
# Geometric identity: ||a|| ||b|| ≥ |a · b| (Cauchy-Schwarz, value side)
# ---------------------------------------------------------------------------

def test_cauchy_schwarz_holds_at_expansion_point():
    _, _, _, v, w = _make_vecs()
    lhs = tax.la.norm(v).value() * tax.la.norm(w).value()
    rhs = abs(tax.la.dot(v, w).value())
    assert lhs + 1e-12 >= rhs


# ---------------------------------------------------------------------------
# normalize — method and free function
# ---------------------------------------------------------------------------

def test_vec_normalize_returns_unit_vector():
    _, _, _, v, _ = _make_vecs()  # value (3, 4, 0)
    u = v.normalize()
    assert isinstance(u, tax.la.Vec)
    # u.value() should be the unit-direction at the expansion point.
    np.testing.assert_allclose(u.value(), [0.6, 0.8, 0.0])
    # ||u|| = 1 to floating-point tolerance.
    assert u.norm().value() == pytest.approx(1.0, abs=1e-12)


def test_la_normalize_matches_method():
    _, _, _, v, _ = _make_vecs()
    np.testing.assert_allclose(v.normalize().value(), tax.la.normalize(v).value())


def test_vec_normalize_is_idempotent():
    _, _, _, v, _ = _make_vecs()
    u = v.normalize()
    uu = u.normalize()
    np.testing.assert_allclose(uu.value(), u.value(), atol=1e-12)


def test_vec_normalize_empty_raises():
    with pytest.raises(Exception):
        tax.la.Vec([]).normalize()
    with pytest.raises(Exception):
        tax.la.normalize(tax.la.Vec([]))
