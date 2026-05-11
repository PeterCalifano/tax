# SPDX-License-Identifier: BSD-3-Clause
"""Truncated multivariate Taylor expansions.

The :class:`TaylorExpansion` type has runtime-fixed order and number of
variables; instances are constructed via the module-level factories
:func:`zero`, :func:`one`, :func:`constant`, :func:`variable`, and
:func:`variables`. The class itself is not directly constructible from
Python.

Arithmetic operators (``+``, ``-``, ``*``, ``/``) and the math functions
listed below evaluate eagerly into a fresh :class:`TaylorExpansion` —
Python expressions cannot meaningfully own lazy expression-template
temporaries across statements, so every call materialises the result.
"""

from ._tax import (  # noqa: F401
    TaylorExpansion,
    # factories
    zero,
    one,
    constant,
    variable,
    variables,
    from_coeffs,
    # numerical derivative helpers (return numpy arrays)
    gradient,
    hessian,
    jacobian,
    # trig + hyperbolic
    sin,
    cos,
    tan,
    sinh,
    cosh,
    tanh,
    # inverse trig + hyperbolic
    asin,
    acos,
    atan,
    asinh,
    acosh,
    atanh,
    atan2,
    # exp / log
    exp,
    log,
    log10,
    # roots, powers
    sqrt,
    cbrt,
    square,
    cube,
    pow,
    hypot,
    abs,
    erf,
)

__all__ = [
    "TaylorExpansion",
    "zero",
    "one",
    "constant",
    "variable",
    "variables",
    "from_coeffs",
    "gradient",
    "hessian",
    "jacobian",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "asin",
    "acos",
    "atan",
    "asinh",
    "acosh",
    "atanh",
    "atan2",
    "exp",
    "log",
    "log10",
    "sqrt",
    "cbrt",
    "square",
    "cube",
    "pow",
    "hypot",
    "abs",
    "erf",
]
