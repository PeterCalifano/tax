// SPDX-License-Identifier: BSD-3-Clause
//
// Shared declarations for the nanobind `_tax` extension. Brings in nanobind
// headers, the Eigen helpers, and `tax/tax.hpp`, plus the four type aliases
// used pervasively across the binding files. Each `bind_*.cpp` includes this
// header and contributes its slice to the module via a `bind_xxx` function.

#pragma once

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <Eigen/Core>
#include <span>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "tax/tax.hpp"

namespace nb = nanobind;

using DynTE = tax::DynTE< double >;
using TeVec = Eigen::Matrix< DynTE, Eigen::Dynamic, 1 >;
using TeMat = Eigen::Matrix< DynTE, Eigen::Dynamic, Eigen::Dynamic >;

namespace tax_py
{

// ---------------------------------------------------------------------------
// Small helpers used by several binding files.
// ---------------------------------------------------------------------------

[[nodiscard]] inline std::span< const int > spanOf( const std::vector< int >& v ) noexcept
{
    return std::span< const int >( v.data(), v.size() );
}

// Build N coordinate variables from an `x0` vector and a runtime `order`.
[[nodiscard]] inline std::vector< DynTE > makeVariables( const std::vector< double >& x0,
                                                          std::size_t order )
{
    return DynTE::variables( std::span< const double >( x0.data(), x0.size() ), order );
}

// Pretty polynomial form via `operator<<` — used by both __str__ and __repr__
// (the latter wraps it with a shape envelope when desired).
[[nodiscard]] inline std::string formatStr( const DynTE& t )
{
    std::ostringstream os;
    os << t;
    return os.str();
}

// ---------------------------------------------------------------------------
// Bind-function entry points. Each is implemented in its own .cpp file and
// called from `tax_module.cpp::NB_MODULE` in the right order.
// ---------------------------------------------------------------------------

void bind_te( nb::module_& m, nb::class_< DynTE >& cls );
void bind_vec( nb::class_< TeVec >& vec_cls, nb::class_< TeMat >& mat_cls );
void bind_mat( nb::class_< TeVec >& vec_cls, nb::class_< TeMat >& mat_cls );
void bind_math( nb::module_& m );
void bind_la( nb::module_& la_mod );
void bind_factories( nb::module_& m );

}  // namespace tax_py
