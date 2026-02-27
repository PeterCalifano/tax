#pragma once

#include <tax/da_type.hpp>

namespace da {

// =============================================================================
// §10  Convenience aliases
// =============================================================================

template <int N>        using DAd  = DA<double, N, 1>;   // univariate (backward-compat)
template <int N>        using DAf  = DA<float,  N, 1>;
template <int N, int M> using DAMd = DA<double, N, M>;   // multivariate
template <int N, int M> using DAMf = DA<float,  N, M>;

} // namespace da
