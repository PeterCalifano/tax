#pragma once

#include <tax/fwd.hpp>

namespace da::detail {

// =============================================================================
// §2  Compile-time combinatorics and grlex flat-index formula
// =============================================================================

constexpr std::size_t binom(int n, int k) noexcept {
    if (k < 0 || n < 0 || k > n) return 0;
    if (k == 0 || k == n)        return 1;
    if (k > n - k) k = n - k;
    std::size_t r = 1;
    for (int i = 0; i < k; ++i) { r *= std::size_t(n - i); r /= std::size_t(i + 1); }
    return r;
}

/// Total number of monomials of degree <= N in M variables: C(N+M, M).
constexpr std::size_t numMonomials(int N, int M) noexcept { return binom(N + M, M); }

template <int M>
constexpr int totalDegree(const da::MultiIndex<M>& a) noexcept {
    int d = 0;
    for (int i = 0; i < M; ++i) d += a[i];
    return d;
}

/// Flat grlex index of multi-index alpha.
///
///   offset = C(d+M-1, M)   — monomials of total degree < d
///
///   Within degree d:
///     pos = sum_{i=0}^{M-2}  C(rem_i - alpha[i] + M-2-i,  M-1-i)
///   where rem_i = d - alpha[0] - ... - alpha[i-1].
///   Each term counts monomials with the same prefix but strictly larger alpha[i].
template <int M>
constexpr std::size_t flatIndex(const da::MultiIndex<M>& alpha) noexcept {
    static_assert(M >= 1);
    const int d = totalDegree<M>(alpha);
    std::size_t idx = binom(d + M - 1, M);
    int rem = d;
    for (int i = 0; i < M - 1; ++i) {
        idx += binom(rem - alpha[i] + (M - 2 - i), M - 1 - i);
        rem  -= alpha[i];
    }
    return idx;
}

} // namespace da::detail
