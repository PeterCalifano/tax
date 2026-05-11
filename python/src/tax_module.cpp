// SPDX-License-Identifier: BSD-3-Clause
//
// Python bindings for tax via nanobind.
//
// Entry point: declares the `_tax` module, creates the type objects, and
// dispatches to the per-concern `bind_*` files.
//
//   tax.TaylorExpansion           lives at module top-level
//   tax.math (submodule)          all math functions
//   tax.la (submodule)            Vec / Mat / norm / dot / cross
//
// Vec / Mat are *only* exposed under `tax.la` — they aren't reachable from
// the top-level `tax` namespace. This keeps the top-level surface lean.

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

    // The `la` submodule owns the Vec / Mat classes — they are not exposed
    // on the top-level `tax` namespace.
    nb::module_ la_mod = m.def_submodule(
        "la", "Linear-algebra helpers for TaylorExpansion vectors / matrices." );

    auto vec_cls = nb::class_< TeVec >(
        la_mod, "Vec",
        R"doc(Vector of `TaylorExpansion` objects — a 1-D collection.

Backed by `Eigen::Matrix<DynTE, Dynamic, 1>`. All elements must share the
same shape `(order, size)`; this is asserted by the underlying kernels
when arithmetic / derivative queries fire.
)doc" );

    auto mat_cls = nb::class_< TeMat >(
        la_mod, "Mat",
        R"doc(Matrix of `TaylorExpansion` objects — a 2-D collection.

Backed by `Eigen::Matrix<DynTE, Dynamic, Dynamic>`. All elements must
share the same shape `(order, size)`.
)doc" );

    tax_py::bind_te( m, te_cls );
    tax_py::bind_vec( vec_cls, mat_cls );
    tax_py::bind_mat( vec_cls, mat_cls );
    tax_py::bind_factories( m );
    tax_py::bind_math( m );
    tax_py::bind_la( la_mod );
}
