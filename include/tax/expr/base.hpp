#pragma once

#include <tax/fwd.hpp>
#include <tax/combinatorics.hpp>
#include <tax/kernels.hpp>
#include <tax/leaf.hpp>

namespace da {

// =============================================================================
// §5  CRTP expression base
// =============================================================================

template <typename Derived, typename T, int N, int M>
struct DAExpr {
    using scalar_type                  = T;
    static constexpr int  order        = N;
    static constexpr int  nvars        = M;
    static constexpr std::size_t ncoef = detail::numMonomials(N, M);
    using coeff_array                  = std::array<T, ncoef>;

    [[nodiscard]] constexpr const Derived& self() const noexcept
    { return static_cast<const Derived&>(*this); }

    // Zero-copy write to caller's buffer.  All composition happens here.
    constexpr void evalTo(coeff_array& out) const noexcept { self().evalTo(out); }

    // ── Accumulation modes (internal ET dispatch only) ────────────────────────
    //
    // addTo / subTo are NOT safe for user-facing operator+= / operator-= when
    // the expression contains a leaf that aliases the destination buffer.
    // They are designed solely for the internal ET dispatch chain inside evalTo,
    // where 'out' is always a freshly-allocated buffer distinct from all leaves.
    //
    // Default implementations materialise via evalTo then add/subtract in-place.
    // DA<T,N,M> overrides both with direct addInPlace / subInPlace (0 temps).
    // BinExpr<Add/Sub> overrides both to recurse without any materialisation.
    // UnaryExpr<OpNeg> overrides both, flipping addTo↔subTo (0 temps).
    // BinExpr<Mul> overrides addTo to use cauchyAccumulate with leaf detection.

    constexpr void addTo(coeff_array& out) const noexcept {
        coeff_array tmp{};
        self().evalTo(tmp);
        detail::addInPlace<T, ncoef>(out, tmp);
    }
    constexpr void subTo(coeff_array& out) const noexcept {
        coeff_array tmp{};
        self().evalTo(tmp);
        detail::subInPlace<T, ncoef>(out, tmp);
    }

    // Out-of-place helper (RVO eliminates the copy at the call site).
    [[nodiscard]] constexpr coeff_array eval() const noexcept {
        coeff_array out{};
        self().evalTo(out);
        return out;
    }

    [[nodiscard]] constexpr T value() const noexcept { return eval()[0]; }

    [[nodiscard]] constexpr T coeff(const MultiIndex<M>& alpha) const noexcept
    { return eval()[detail::flatIndex<M>(alpha)]; }

    /// Mixed partial derivative d^|alpha| f / dx^alpha at x0 = alpha! * c[alpha].
    [[nodiscard]] constexpr T derivative(const MultiIndex<M>& alpha) const noexcept {
        std::size_t fac = 1;
        for (int i = 0; i < M; ++i) for (int j = 1; j <= alpha[i]; ++j) fac *= std::size_t(j);
        return eval()[detail::flatIndex<M>(alpha)] * T(fac);
    }
};

} // namespace da
