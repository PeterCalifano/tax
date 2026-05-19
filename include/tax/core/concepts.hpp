#pragma once

#include <concepts>

namespace tax
{

/// @brief Scalar constraint used for DA coefficients and function values.
template < typename T >
concept Scalar = std::floating_point< T >;

// TaylorPolynomial / DensePolynomial / SparsePolynomial concepts are filled
// in slices 2 and 9 once the types exist.

}  // namespace tax
