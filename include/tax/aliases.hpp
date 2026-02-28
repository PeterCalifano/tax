#pragma once

#include <tax/da.hpp>

namespace da {

template <int N>        using DA  = TDA<double, N, 1>;
template <int N, int M> using DAn = TDA<double, N, M>;

} // namespace da
