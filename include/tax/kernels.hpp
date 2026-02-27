#pragma once

#include <cmath>
#include <tax/combinatorics.hpp>

namespace da::detail {

// =============================================================================
// §3  Low-level coefficient-array kernels  (no heap; all in-place)
// =============================================================================

// ── §3a  Element-wise arithmetic ─────────────────────────────────────────────

template <typename T, std::size_t S>
constexpr void addInPlace(std::array<T, S>& o, const std::array<T, S>& r) noexcept
{ for (std::size_t i = 0; i < S; ++i) o[i] += r[i]; }

template <typename T, std::size_t S>
constexpr void subInPlace(std::array<T, S>& o, const std::array<T, S>& r) noexcept
{ for (std::size_t i = 0; i < S; ++i) o[i] -= r[i]; }

template <typename T, std::size_t S>
constexpr void negateInPlace(std::array<T, S>& o) noexcept
{ for (auto& v : o) v = -v; }

template <typename T, std::size_t S>
constexpr void scaleInPlace(std::array<T, S>& o, T s) noexcept
{ for (auto& v : o) v *= s; }

// ── §3b  Multivariate Cauchy product ─────────────────────────────────────────
//
//   (f·g)[alpha] = sum_{beta <= alpha (componentwise)} f[beta] · g[alpha-beta]
//
// Precondition: out, f, g are pairwise distinct arrays.

template <typename T, int N, int M>
constexpr void cauchyProduct(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& f,
    const std::array<T, numMonomials(N, M)>& g) noexcept
{
    out = {};
    da::MultiIndex<M> alpha{};
    da::MultiIndex<M> beta{};

    auto fillBeta = [&](auto& self, int bvar) -> void {
        if (bvar == M) {
            da::MultiIndex<M> gamma{};
            for (int i = 0; i < M; ++i) gamma[i] = alpha[i] - beta[i];
            out[flatIndex<M>(alpha)] +=
                f[flatIndex<M>(beta)] * g[flatIndex<M>(gamma)];
            return;
        }
        for (int b = 0; b <= alpha[bvar]; ++b) { beta[bvar] = b; self(self, bvar + 1); }
    };

    auto fillAlpha = [&](auto& self, int var, int rem) -> void {
        if (var == M - 1) { alpha[var] = rem; fillBeta(fillBeta, 0); return; }
        for (int k = rem; k >= 0; --k) { alpha[var] = k; self(self, var + 1, rem - k); }
    };

    for (int d = 0; d <= N; ++d) fillAlpha(fillAlpha, 0, d);
}

// ── §3c  Multivariate Cauchy accumulate ──────────────────────────────────────
//
//   out[alpha] += sum_{beta <= alpha} f[beta] · g[alpha-beta]
//
// Like cauchyProduct but ACCUMULATES into out rather than overwriting it.
// This is the key kernel that enables Mul::addTo with zero temporaries when
// both operands are leaves: cauchyAccumulate(out, l.coeffs(), r.coeffs()).
//
// Precondition: out, f, g are pairwise distinct arrays.

template <typename T, int N, int M>
constexpr void cauchyAccumulate(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& f,
    const std::array<T, numMonomials(N, M)>& g) noexcept
{
    da::MultiIndex<M> alpha{};
    da::MultiIndex<M> beta{};

    auto fillBeta = [&](auto& self, int bvar) -> void {
        if (bvar == M) {
            da::MultiIndex<M> gamma{};
            for (int i = 0; i < M; ++i) gamma[i] = alpha[i] - beta[i];
            out[flatIndex<M>(alpha)] +=
                f[flatIndex<M>(beta)] * g[flatIndex<M>(gamma)];
            return;
        }
        for (int b = 0; b <= alpha[bvar]; ++b) { beta[bvar] = b; self(self, bvar + 1); }
    };

    auto fillAlpha = [&](auto& self, int var, int rem) -> void {
        if (var == M - 1) { alpha[var] = rem; fillBeta(fillBeta, 0); return; }
        for (int k = rem; k >= 0; --k) { alpha[var] = k; self(self, var + 1, rem - k); }
    };

    for (int d = 0; d <= N; ++d) fillAlpha(fillAlpha, 0, d);
}

// ── §3d  Series reciprocal ───────────────────────────────────────────────────
//
// Solve:  a · out = 1  (as truncated series)
//
// Recurrence (grlex order, degree by degree):
//   out[0]     = 1 / a[0]
//   out[alpha] = -(sum_{0<|beta|<=|alpha|, beta<=alpha} a[beta]·out[alpha-beta]) / a[0]
//
// Precondition: a[0] != 0, out != a (distinct arrays).

template <typename T, int N, int M>
constexpr void seriesReciprocal(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    out = {};
    const T inv_a0 = T{1} / a[0];
    da::MultiIndex<M> alpha{};
    da::MultiIndex<M> beta{};

    auto fillAlpha = [&](auto& self, int var, int rem, int d) -> void {
        if (var == M - 1) {
            alpha[var] = rem;
            const std::size_t ai = flatIndex<M>(alpha);
            T rhs = (d == 0) ? T{1} : T{0};
            for (int db = 1; db <= d; ++db) {
                auto fillBeta = [&](auto& bself, int bvar, int brem) -> void {
                    if (bvar == M - 1) {
                        beta[bvar] = brem;
                        if (beta[bvar] > alpha[bvar]) return;
                        da::MultiIndex<M> gamma{};
                        for (int i = 0; i < M; ++i) gamma[i] = alpha[i] - beta[i];
                        rhs -= a[flatIndex<M>(beta)] * out[flatIndex<M>(gamma)];
                        return;
                    }
                    for (int b = 0; b <= std::min(brem, alpha[bvar]); ++b) {
                        beta[bvar] = b;
                        bself(bself, bvar + 1, brem - b);
                    }
                };
                fillBeta(fillBeta, 0, db);
            }
            out[ai] = rhs * inv_a0;
            return;
        }
        for (int k = rem; k >= 0; --k) { alpha[var] = k; self(self, var + 1, rem - k, d); }
    };

    for (int d = 0; d <= N; ++d) fillAlpha(fillAlpha, 0, d, d);
}

// ── §3d  Series square: out = a² ─────────────────────────────────────────────
//
// Delegates to cauchyProduct(out, a, a).
// Precondition: out != a.

template <typename T, int N, int M>
constexpr void seriesSquare(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    cauchyProduct<T, N, M>(out, a, a);
}

// ── §3e  Series cube: out = a³ via direct triple convolution ─────────────────
//
// Instead of computing a² first and then a·a² (which needs 2 extra arrays
// beyond the input), we enumerate ordered 3-partitions directly:
//
//   out[alpha] = sum_{beta+gamma+delta=alpha}  a[beta] · a[gamma] · a[delta]
//
// This touches the input array exactly once per (alpha, beta, gamma) triple
// and writes the result straight into out — using only 1 extra array (the
// input snapshot held by the caller).
//
// Precondition: out != a.

template <typename T, int N, int M>
constexpr void seriesCube(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    out = {};
    da::MultiIndex<M> alpha{};
    da::MultiIndex<M> beta{};
    da::MultiIndex<M> gamma{};

    // fillGamma: enumerate gamma with |gamma|=dg, gamma[i] <= (alpha-beta)[i]
    auto fillGamma = [&](auto& gself, int gvar, int grem) -> void {
        if (gvar == M - 1) {
            gamma[gvar] = grem;
            if (gamma[gvar] > alpha[gvar] - beta[gvar]) return;
            da::MultiIndex<M> delta{};
            for (int i = 0; i < M; ++i) delta[i] = alpha[i] - beta[i] - gamma[i];
            out[flatIndex<M>(alpha)] +=
                a[flatIndex<M>(beta)] * a[flatIndex<M>(gamma)] * a[flatIndex<M>(delta)];
            return;
        }
        const int maxg = alpha[gvar] - beta[gvar];
        for (int g = 0; g <= std::min(grem, maxg); ++g) {
            gamma[gvar] = g;
            gself(gself, gvar + 1, grem - g);
        }
    };

    // fillBeta: enumerate beta with |beta|=db, beta[i] <= alpha[i]
    auto fillBeta = [&](auto& bself, int bvar, int brem) -> void {
        if (bvar == M - 1) {
            beta[bvar] = brem;
            if (beta[bvar] > alpha[bvar]) return;
            // |gamma| ranges from 0 to |alpha - beta|
            int ab_total = 0;
            for (int i = 0; i < M; ++i) ab_total += alpha[i] - beta[i];
            for (int dg = 0; dg <= ab_total; ++dg)
                fillGamma(fillGamma, 0, dg);
            return;
        }
        for (int b = 0; b <= std::min(brem, alpha[bvar]); ++b) {
            beta[bvar] = b;
            bself(bself, bvar + 1, brem - b);
        }
    };

    // fillAlpha: enumerate alpha in grlex order (by total degree, then grlex)
    auto fillAlpha = [&](auto& aself, int var, int rem) -> void {
        if (var == M - 1) {
            alpha[var] = rem;
            const int d = totalDegree<M>(alpha);
            for (int db = 0; db <= d; ++db)
                fillBeta(fillBeta, 0, db);
            return;
        }
        for (int k = rem; k >= 0; --k) { alpha[var] = k; aself(aself, var + 1, rem - k); }
    };

    for (int d = 0; d <= N; ++d) fillAlpha(fillAlpha, 0, d);
}

// ── §3f  Series square root: solve g·g = a ───────────────────────────────────
//
// Recurrence (grlex order, degree by degree):
//
//   (g·g)[alpha] = 2·g[0]·g[alpha]
//                + sum_{beta: 0<|beta|<|alpha|, beta<=alpha} g[beta]·g[alpha-beta]
//                = a[alpha]
//
//   => g[alpha] = (a[alpha] - sum) / (2·g[0])
//
// Every g[alpha-beta] with |beta|>=1 is already computed when we reach alpha,
// because we process in increasing degree order.
//
// Precondition: a[0] > 0, out != a.

template <typename T, int N, int M>
constexpr void seriesSqrt(
    std::array<T, numMonomials(N, M)>&       out,
    const std::array<T, numMonomials(N, M)>& a) noexcept
{
    using std::sqrt;
    out    = {};
    out[0] = sqrt(a[0]);
    const T inv2g0 = T{1} / (T{2} * out[0]);

    da::MultiIndex<M> alpha{};
    da::MultiIndex<M> beta{};

    auto fillAlpha = [&](auto& self, int var, int rem, int d) -> void {
        if (var == M - 1) {
            alpha[var] = rem;
            const std::size_t ai = flatIndex<M>(alpha);
            T rhs = a[ai];
            // sum over beta: 0 < |beta| < d, beta[i] <= alpha[i]
            for (int db = 1; db < d; ++db) {
                auto fillBeta = [&](auto& bself, int bvar, int brem) -> void {
                    if (bvar == M - 1) {
                        beta[bvar] = brem;
                        if (beta[bvar] > alpha[bvar]) return;
                        da::MultiIndex<M> gamma{};
                        for (int i = 0; i < M; ++i) gamma[i] = alpha[i] - beta[i];
                        rhs -= out[flatIndex<M>(beta)] * out[flatIndex<M>(gamma)];
                        return;
                    }
                    for (int b = 0; b <= std::min(brem, alpha[bvar]); ++b) {
                        beta[bvar] = b;
                        bself(bself, bvar + 1, brem - b);
                    }
                };
                fillBeta(fillBeta, 0, db);
            }
            out[ai] = rhs * inv2g0;
            return;
        }
        for (int k = rem; k >= 0; --k) { alpha[var] = k; self(self, var + 1, rem - k, d); }
    };

    for (int d = 1; d <= N; ++d) fillAlpha(fillAlpha, 0, d, d);
}

} // namespace da::detail
