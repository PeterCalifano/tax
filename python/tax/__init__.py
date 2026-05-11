# SPDX-License-Identifier: BSD-3-Clause
"""Truncated multivariate Taylor expansions.

The :class:`TaylorExpansion` type has runtime-fixed order and number of
variables; instances are constructed via the module-level factories
:func:`zero`, :func:`one`, :func:`constant`, :func:`variable`,
:func:`variables`, and :func:`from_coeffs`. The class itself is not
directly constructible from Python.

Math functions live under the :mod:`tax.math` submodule
(``tax.math.sin``, ``tax.math.exp``, ...). Numerical derivative helpers
(:func:`gradient`, :func:`hessian`, :func:`jacobian`) live at the
top level and return numpy arrays directly.

Containers :class:`Vec` and
:class:`Mat` wrap Eigen-backed collections of
:class:`TaylorExpansion` and expose ``value``, ``eval``, ``derivative``,
and (for vectors) ``jacobian`` returning numpy arrays.

The :mod:`tax.la` submodule re-exports :class:`Vec` and :class:`Mat`
together with free linear-algebra functions :func:`tax.la.norm`,
:func:`tax.la.dot`, and :func:`tax.la.cross`.
"""

from ._tax import (  # noqa: F401
    # core types
    TaylorExpansion,
    Vec,
    Mat,
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
    # submodules
    math,
    la,
)

__all__ = [
    "TaylorExpansion",
    "Vec",
    "Mat",
    "zero",
    "one",
    "constant",
    "variable",
    "variables",
    "from_coeffs",
    "gradient",
    "hessian",
    "jacobian",
    "math",
    "la",
]
