// SPDX-License-Identifier: BSD-3-Clause
//
// Python bindings for tax via nanobind.
//
// Entry point: declares the `_tax` module, creates the three public type
// objects (`TaylorExpansion`, `Vec`, `Mat`) and dispatches to the
// per-concern `bind_*` functions defined under `python/src/bind_*.cpp`.
//
// The class objects are constructed here so that `bind_vec` and `bind_mat`
// can reference each other's class (needed by their `__matmul__` /
// `block` / `row` / `col` overloads).

#include "common.hpp"

NB_MODULE( _tax, m )
{
    m.doc() =
        "Truncated multivariate Taylor expansions (runtime order and size). "
        "The class is constructed via `zero`, `one`, `constant`, `variable`, "
        "or `variables`; arithmetic and math functions evaluate eagerly.";

    auto te_cls = nb::class_< DynTE >( m, "TaylorExpansion",
                                       R"doc(Truncated multivariate Taylor expansion.

Order and number of variables are fixed at construction. Use the
module-level factories `zero`, `one`, `constant`, `variable`, or
`variables` to build instances.
)doc" );

    auto vec_cls = nb::class_< TeVec >(
        m, "Vec",
        R"doc(Vector of `TaylorExpansion` objects — a 1-D collection.

Backed by `Eigen::Matrix<DynTE, Dynamic, 1>`. All elements must share the
same shape `(order, size)`; this is asserted by the underlying kernels
when arithmetic / derivative queries fire.
)doc" );

    auto mat_cls = nb::class_< TeMat >(
        m, "Mat",
        R"doc(Matrix of `TaylorExpansion` objects — a 2-D collection.

Backed by `Eigen::Matrix<DynTE, Dynamic, Dynamic>`. All elements must
share the same shape `(order, size)`.
)doc" );

    tax_py::bind_te( m, te_cls );
    tax_py::bind_vec( m, vec_cls, mat_cls );
    tax_py::bind_mat( m, vec_cls, mat_cls );
    tax_py::bind_factories( m );
    tax_py::bind_math( m );
    tax_py::bind_la( m, vec_cls, mat_cls );
}
