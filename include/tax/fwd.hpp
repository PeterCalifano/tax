#pragma once

#include <array>
#include <concepts>
#include <cstddef>

namespace da {

// =============================================================================
// §1  Scalar concept and multi-index type
// =============================================================================

template <typename T>
concept Scalar = std::floating_point<T>;

/// Multi-index alpha in N^M.
template <int M>
using MultiIndex = std::array<int, M>;

} // namespace da
