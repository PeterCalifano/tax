# Python bindings

`tax` ships nanobind-based Python bindings exposing the dynamic-shape
[`TaylorExpansion`](#class-taylorexpansion) type. Order and number of
variables are both fixed at construction; arithmetic and math functions
evaluate eagerly into fresh objects.

## Build

Two paths:

**1. From a regular CMake build** (when developing alongside C++):

```bash
pip install nanobind pytest
cmake -S . -B build -DTAX_BUILD_PYTHON=ON
cmake --build build -j
PYTHONPATH=$PWD/build/python python3 -c "import tax"
```

**2. As a wheel** (for users):

```bash
pip install .
python3 -c "import tax; print(tax.variable(2.0, 0, 5, 1))"
```

`pip install .` uses `scikit-build-core` via `pyproject.toml`. The wheel
contains the compiled `tax/_tax.so` extension plus the `tax/__init__.py`
wrapper.

## Quick reference

```python
import tax, math

# All M = 3 coordinate variables at the expansion point (1.0, 2.0, 3.0)
# with truncation order N = 5.
x, y, z = tax.variables([1.0, 2.0, 3.0], order=5)

f = tax.sin(x * y) + tax.exp(z) + tax.atan2(y, x)

f.order        # 5
f.size         # 3
f.value()      # f at the expansion point
f.coeffs()     # list of raw Taylor coefficients (graded-lex order)

# Polynomial evaluation
f.at([0.01, -0.02, 0.03])

# Symbolic operations
f.deriv(0)                  # partial derivative w.r.t. x — returns a new TE
f.integ(2)                  # indefinite integral w.r.t. z

# Numerical derivatives at the expansion point
f.derivative([1, 0, 0])     # df/dx
f.derivative([1, 1, 0])     # d^2f/dx/dy
f.derivatives()             # full list (c_i * alpha_i!)

# Norms of the coefficient vector
f.coeffs_norm_inf()
f.coeffs_norm(1)
f.coeffs_norm(2)
```

## Available math functions

Math functions live under **`tax.math`** (not the top level):

```python
y = tax.math.sin(x)
z = tax.math.atan2(y, x)
```

| Family | Functions |
|---|---|
| trig | `tax.math.sin`, `cos`, `tan` |
| hyperbolic | `sinh`, `cosh`, `tanh` |
| inverse trig | `asin`, `acos`, `atan`, `atan2(y, x)` |
| inverse hyperbolic | `asinh`, `acosh`, `atanh` |
| exp / log | `exp`, `log`, `log10` |
| roots, powers | `sqrt`, `cbrt`, `square`, `cube`, `pow(x, c)`, `pow(x, n)`, `hypot(x, y)` |
| other | `abs`, `erf` |

`pow` is overloaded: a real `c` calls the real-exponent kernel, an
`int n` calls the integer-power kernel (binary exponentiation, negative
powers handled via reciprocal).

## Eigen-backed containers

For collections of `TaylorExpansion` objects, use the Eigen-backed
wrappers `tax.Vec` and `tax.Mat`.
They share the same `(order, size)` across all elements and expose
common vectorised operations.

```python
x, y = tax.variables([1.0, 2.0], order=3)

# 1-D collection — vector-valued function.
v = tax.Vec([x, y, x * y])

len(v)              # 3
v[2]                # TaylorExpansion (x * y)
v.value()           # numpy array of constant terms, shape (3,)
v.eval([0.1, 0.2])  # evaluate every component, shape (3,)
v.derivative([1, 0])# d/dx of each component
v.jacobian()        # numpy (3, 2) — same as tax.jacobian([x, y, x*y])

# 2-D collection — matrix of polynomials.
m = tax.Mat([[x,    y    ],
                               [x*y,  x + y]])
m.shape             # (2, 2)
m[0, 1]             # TaylorExpansion (y)
m.value()           # numpy (2, 2)
m.eval([0.1, 0.2])  # numpy (2, 2)
m.derivative([0, 1])# numpy (2, 2)
```

Backed by `Eigen::Matrix<DynTE, Dynamic, 1>` and
`Eigen::Matrix<DynTE, Dynamic, Dynamic>` respectively; all the
operations call the existing C++ Eigen helpers (`tax::value`,
`tax::eval`, `tax::derivative`, `tax::jacobian`).

## Factories

- `tax.zero(order, size)` — zero polynomial.
- `tax.one(order, size)` — constant 1.
- `tax.constant(v, order, size)` — constant `v`.
- `tax.variable(x0, var_idx, order, size)` — coordinate variable
  `x_{var_idx}` expanded around `x0`.
- `tax.variables(x0_list, order)` — all `len(x0_list)` coordinate
  variables.

`TaylorExpansion` itself is not directly constructible from Python; use
the factories. This mirrors the C++ class which has no public default
constructor producing a shape-aware object.

## Arithmetic

All standard operators are bound: `+`, `-`, `*`, `/`, unary `-`, plus
the in-place forms (`+=`, `-=`, `*=`, `/=`). Scalar combinations from
both sides work transparently.

## Limitations

- Mixing two `TaylorExpansion` objects of different `(order, size)`
  triggers a C++ assert (or undefined behaviour with `NDEBUG`). The
  bindings do not currently cross-check; the C++ layer's `assert`s do.
- The static-shape C++ type `TaylorExpansionT<T, N, M>` is intentionally
  not exposed — Python users get one type, not a (N, M) grid.
- The `DynOrderTE<M>` form (compile-time M, runtime order) is also not
  exposed yet; could be added as `TaylorExpansionFixedSize_M` for a few
  common M values if needed.
