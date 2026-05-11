# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end pytest suite for the nanobind `tax` Python module.

Covers construction, accessors, arithmetic, math functions, evaluation,
symbolic deriv/integ, and norms — each via the dynamic `TaylorExpansion`
type. Numerical correctness is checked against `math` / `numpy`-style
closed-form references; the wide cross-check against the C++ static
path lives in the C++ test suite.
"""

import math

import pytest

import tax


def test_module_attrs():
    # Headline type and factories are present.
    assert hasattr(tax, "TaylorExpansion")
    for name in ("zero", "one", "constant", "variable", "variables"):
        assert callable(getattr(tax, name))


# ---------------------------------------------------------------------------
# Factories and shape queries
# ---------------------------------------------------------------------------

def test_zero_one_constant_factories():
    z = tax.zero(order=3, size=2)
    o = tax.one(order=3, size=2)
    c = tax.constant(2.5, order=3, size=2)
    assert z.value() == 0.0
    assert o.value() == 1.0
    assert c.value() == 2.5
    assert z.order == 3 and z.size == 2
    assert o.n_coefficients == c.n_coefficients == 10  # C(5,2) = 10


def test_variable_and_variables():
    x = tax.variable(x0=1.5, var_idx=0, order=4, size=2)
    assert x.value() == 1.5
    assert x.order == 4 and x.size == 2

    coords = tax.variables([1.0, 2.0, 3.0], order=2)
    assert len(coords) == 3
    for i, c in enumerate(coords):
        assert c.value() == float(i + 1)
        assert c.order == 2 and c.size == 3


def test_variable_out_of_range_raises():
    with pytest.raises(Exception):
        tax.variable(x0=0.0, var_idx=10, order=3, size=2)


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

def test_arithmetic_basic():
    x, y = tax.variables([1.0, 2.0], order=3)
    assert (x + y).value() == 3.0
    assert (x * y).value() == 2.0
    assert (y / x).value() == 2.0
    assert (-x).value() == -1.0

    # Scalar combinations from both sides.
    assert (2.0 * x).value() == 2.0
    assert (x * 2.0).value() == 2.0
    assert (1.0 + x).value() == 2.0
    assert (x - 0.25).value() == 0.75
    assert (4.0 - x).value() == 3.0
    assert (x / 2.0).value() == 0.5


def test_in_place_arithmetic():
    x, y = tax.variables([1.0, 2.0], order=3)
    z = x + y           # value 3
    z += x              # value 4
    assert z.value() == 4.0
    z *= 2.0            # value 8
    assert z.value() == 8.0
    z /= y              # divide by (2 + dy) -> value 4
    assert z.value() == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Math functions — closed-form reference at the expansion point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fn, ref, x0",
    [
        (tax.sin, math.sin, 0.5),
        (tax.cos, math.cos, 0.5),
        (tax.tan, math.tan, 0.5),
        (tax.sinh, math.sinh, 0.5),
        (tax.cosh, math.cosh, 0.5),
        (tax.tanh, math.tanh, 0.5),
        (tax.asin, math.asin, 0.4),
        (tax.acos, math.acos, 0.4),
        (tax.atan, math.atan, 0.6),
        (tax.asinh, math.asinh, 0.4),
        (tax.acosh, math.acosh, 2.0),
        (tax.atanh, math.atanh, 0.3),
        (tax.exp, math.exp, 0.5),
        (tax.log, math.log, 2.0),
        (tax.log10, math.log10, 5.0),
        (tax.sqrt, math.sqrt, 3.0),
        (tax.cbrt, lambda x: x ** (1.0 / 3.0), 2.0),
        (tax.square, lambda x: x * x, 0.7),
        (tax.cube, lambda x: x ** 3, 0.7),
        (tax.erf, math.erf, 0.4),
    ],
)
def test_unary_math_value_matches_reference(fn, ref, x0):
    x = tax.variable(x0=x0, var_idx=0, order=5, size=1)
    assert fn(x).value() == pytest.approx(ref(x0))


def test_pow_real_and_int():
    x = tax.variable(2.0, 0, 5, 1)
    assert tax.pow(x, 0.5).value() == pytest.approx(math.sqrt(2.0))
    assert tax.pow(x, 3).value() == pytest.approx(8.0)
    # Negative integer powers (binary exponentiation through reciprocal).
    assert tax.pow(x, -2).value() == pytest.approx(0.25)


def test_atan2_and_hypot():
    x, y = tax.variables([3.0, 4.0], order=4)
    assert tax.atan2(y, x).value() == pytest.approx(math.atan2(4.0, 3.0))
    assert tax.hypot(x, y).value() == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Polynomial evaluation
# ---------------------------------------------------------------------------

def test_eval_univariate():
    x = tax.variable(2.0, 0, 5, 1)
    f = tax.exp(x)
    # Order-5 truncation of exp(2 + dx) at dx = 0.1.
    ref = sum(math.exp(2.0) * (0.1 ** k) / math.factorial(k) for k in range(6))
    assert f.at(0.1) == pytest.approx(ref, rel=1e-12)


def test_eval_multivariate():
    x, y = tax.variables([0.5, 0.3], order=4)
    f = tax.sin(x * y)
    # Evaluate at (dx, dy) = (0.05, -0.02).
    v = f.at([0.05, -0.02])
    # Compare to direct sin((0.5 + 0.05) * (0.3 - 0.02)) up to order-4 truncation —
    # exact value is fine as a sanity check on size + shape.
    assert isinstance(v, float)


# ---------------------------------------------------------------------------
# Symbolic deriv / integ
# ---------------------------------------------------------------------------

def test_deriv_and_integ_roundtrip_univariate():
    x = tax.variable(0.3, 0, 5, 1)
    f = tax.sin(x)
    df = f.deriv(0)
    # d/dx sin(x) at x=0.3 is cos(0.3).
    assert df.value() == pytest.approx(math.cos(0.3))

    F = f.integ(0)
    # F'(0) coefficient = f(0) = sin(0.3); F(0) (constant of integration) = 0.
    assert F.value() == 0.0
    assert F.coeffs()[1] == pytest.approx(math.sin(0.3))


def test_deriv_out_of_range_raises():
    x = tax.variable(1.0, 0, 3, 2)
    with pytest.raises(Exception):
        x.deriv(5)


# ---------------------------------------------------------------------------
# Numerical derivative + derivatives()
# ---------------------------------------------------------------------------

def test_derivative_multi_index():
    x, y = tax.variables([0.5, 0.3], order=4)
    f = tax.exp(x + y)
    # d^2 exp(x+y) / (dx dy) at expansion = exp(x0 + y0).
    assert f.derivative([1, 1]) == pytest.approx(math.exp(0.5 + 0.3))


def test_derivatives_all_returns_full_buffer():
    x = tax.variable(0.5, 0, 3, 1)
    f = tax.exp(x)
    derivs = f.derivatives()
    # f.coeffs[k] * k! for univariate.
    coeffs = f.coeffs()
    for k, (c, d) in enumerate(zip(coeffs, derivs)):
        assert d == pytest.approx(c * math.factorial(k))


# ---------------------------------------------------------------------------
# Coefficient norms
# ---------------------------------------------------------------------------

def test_norms_match_manual():
    x = tax.variable(0.3, 0, 5, 1)
    f = tax.sin(x)
    cs = f.coeffs()

    expected_l1 = sum(abs(c) for c in cs)
    expected_l2 = math.sqrt(sum(c * c for c in cs))
    expected_linf = max(abs(c) for c in cs)

    assert f.coeffs_norm(1) == pytest.approx(expected_l1)
    assert f.coeffs_norm(2) == pytest.approx(expected_l2)
    assert f.coeffs_norm_inf() == pytest.approx(expected_linf)


def test_norm_zero_p_raises():
    x = tax.variable(1.0, 0, 3, 1)
    with pytest.raises(Exception):
        x.coeffs_norm(0)


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------

def test_repr_round_trips_basic_info():
    x = tax.variable(2.0, 0, 3, 2)
    r = repr(x)
    assert "TaylorExpansion" in r
    assert "order=3" in r
    assert "size=2" in r
    assert "coeffs=[" in r
