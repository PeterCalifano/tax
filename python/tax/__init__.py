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

Linear-algebra containers and helpers live under :mod:`tax.la`:
``tax.la.Vec``, ``tax.la.Mat``, ``tax.la.norm``, ``tax.la.dot``,
``tax.la.cross``. The container types are not re-exposed at the top
level — use ``tax.la.Vec`` / ``tax.la.Mat``.
"""

from ._tax import (  # noqa: F401
    # core type
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
    # submodules
    math,
    la,
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
    "math",
    "la",
]
