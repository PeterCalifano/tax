#pragma once

#include <array>
#include <concepts>
#include <cstddef>

namespace tax
{

/// @brief Scalar constraint used for DA coefficients and function values.
template < typename T >
concept Scalar = std::floating_point< T >;

/**
 * @brief Sentinel marking a Taylor-shape dimension as resolved at runtime.
 *
 * Modelled on `Eigen::Dynamic` (= -1). When `TaylorExpansionT`'s order
 * or size template parameter is `tax::Dynamic`, that dimension is carried
 * as a runtime member rather than as a compile-time constant.
 */
inline constexpr int Dynamic = -1;

/**
 * @brief Exponent vector `(a_0, ..., a_{M-1})` for multivariate monomials.
 * @tparam M Number of variables.
 */
template < int M >
using MultiIndex = std::array< int, M >;

}  // namespace tax
