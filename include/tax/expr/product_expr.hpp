#pragma once

#include <tax/expr/base.hpp>

namespace da::detail {

// =============================================================================
// §7i  ProductExpr<Es...>
// =============================================================================
//
// Variadic multiplication node.  Rolling Cauchy product across all operands.
//
// evalTo uses exactly two scratch arrays (a, b) regardless of chain length:
//   • a — rolling accumulator (holds partial product after each step)
//   • b — materialised current factor (skipped for leaf factors via coeffs())
//
// Peak simultaneous temporaries (beyond the caller's out):
//   • All leaves  : 1 temp (only a; each leaf's coeffs() used directly for b)
//   • Has non-leaf: 2 temps (a and b; b reused across all non-leaf factors)
//
// Compare with BinExpr left-associative chain over N leaves:
//   BinExpr leaf detection reduces each level to 1 temp, but frames nest:
//   peak = N−1 temps simultaneously (one per stack frame).
//   ProductExpr<Es...> always peaks at 1–2 regardless of N.
//
// addTo: computes the full product into a temp, then accumulates (1+1/2 temps).
// For the common 2-factor case, cauchyAccumulate with leaf detection is used
// (0 extra temps when both operands are leaves).

template <typename... Es>
class ProductExpr
    : public da::DAExpr<ProductExpr<Es...>,
                    typename std::tuple_element_t<0, std::tuple<Es...>>::scalar_type,
                    std::tuple_element_t<0, std::tuple<Es...>>::order,
                    std::tuple_element_t<0, std::tuple<Es...>>::nvars>
{
    static_assert(sizeof...(Es) >= 2, "ProductExpr needs at least 2 operands");
    template <typename...> friend class ProductExpr;

public:
    using T = typename std::tuple_element_t<0, std::tuple<Es...>>::scalar_type;
    static constexpr int N = std::tuple_element_t<0, std::tuple<Es...>>::order;
    static constexpr int M = std::tuple_element_t<0, std::tuple<Es...>>::nvars;
    using coeff_array = std::array<T, numMonomials(N, M)>;

    explicit constexpr ProductExpr(stored_t<Es>... es) noexcept : ops_(es...) {}

    template <typename E>
    [[nodiscard]] constexpr auto append(stored_t<E> e) const noexcept {
        return std::apply([&](auto const&... x) noexcept {
            return ProductExpr<Es..., E>(x..., e);
        }, ops_);
    }

    // ── evalTo — rolling Cauchy with leaf detection ───────────────────────────

    constexpr void evalTo(coeff_array& out) const noexcept {
        // We always need a as the accumulator.
        // We need b only when at least one operand after the first is a non-leaf.
        coeff_array a{};
        seedAccumulator(a);                             // a ← operand[0]
        if constexpr (hasAnyNonLeaf<1>()) {
            coeff_array b{};                             // b for non-leaf materialisation
            rollProduct<1>(out, a, &b);
        } else {
            rollProduct<1>(out, a, nullptr);             // b never used: nullptr is fine
        }
    }

    // ── addTo — specialised for 2 factors, fallback for N>2 ──────────────────

    constexpr void addTo(coeff_array& out) const noexcept {
        if constexpr (sizeof...(Es) == 2) {
            // 2-factor: use cauchyAccumulate with leaf detection (same as BinExpr<Mul>).
            using L = std::tuple_element_t<0, std::tuple<Es...>>;
            using R = std::tuple_element_t<1, std::tuple<Es...>>;
            const auto& lop = std::get<0>(ops_);
            const auto& rop = std::get<1>(ops_);
            if constexpr (is_leaf_v<L> && is_leaf_v<R>)
                cauchyAccumulate<T,N,M>(out, lop.coeffs(), rop.coeffs());   // 0 temps
            else if constexpr (is_leaf_v<R>) {
                coeff_array la{};
                lop.evalTo(la);
                cauchyAccumulate<T,N,M>(out, la, rop.coeffs());             // 1 temp
            } else if constexpr (is_leaf_v<L>) {
                coeff_array rb{};
                rop.evalTo(rb);
                cauchyAccumulate<T,N,M>(out, lop.coeffs(), rb);             // 1 temp
            } else {
                coeff_array la{}, rb{};
                lop.evalTo(la);
                rop.evalTo(rb);
                cauchyAccumulate<T,N,M>(out, la, rb);                       // 2 temps
            }
        } else {
            // N>2: materialise the full product, then add.
            coeff_array tmp{};
            evalTo(tmp);
            addInPlace<T, numMonomials(N,M)>(out, tmp);
        }
    }

    constexpr void subTo(coeff_array& out) const noexcept {
        coeff_array tmp{};
        evalTo(tmp);
        subInPlace<T, numMonomials(N,M)>(out, tmp);
    }

private:
    std::tuple<stored_t<Es>...> ops_;

    // ── Seed: evaluate or reference the first operand into a ──────────────────

    constexpr void seedAccumulator(coeff_array& a) const noexcept {
        using E0 = std::tuple_element_t<0, std::tuple<Es...>>;
        if constexpr (is_leaf_v<E0>) a = std::get<0>(ops_).coeffs();
        else                         std::get<0>(ops_).evalTo(a);
    }

    // ── Compile-time predicate: any operand from index I onward is a non-leaf? ─

    template <std::size_t From>
    static constexpr bool hasAnyNonLeaf() noexcept {
        return []<std::size_t... I>(std::index_sequence<I...>) {
            return ((!is_leaf_v<std::tuple_element_t<I + From, std::tuple<Es...>>>) || ...);
        }(std::make_index_sequence<sizeof...(Es) - From>{});
    }

    // ── Rolling Cauchy product from index Start onward ────────────────────────
    //
    // After each step: out = a * factor[I], then a = out (ready for next step).
    // b is non-null only when hasAnyNonLeaf<Start>() is true.

    template <std::size_t Start>
    constexpr void rollProduct(coeff_array& out, coeff_array& a,
                               coeff_array* b) const noexcept
    {
        [&]<std::size_t... I>(std::index_sequence<I...>) noexcept {
            (productStep<I + Start>(out, a, b), ...);
        }(std::make_index_sequence<sizeof...(Es) - Start>{});
    }

    template <std::size_t I>
    constexpr void productStep(coeff_array& out, coeff_array& a,
                               coeff_array* b) const noexcept
    {
        using Ei = std::tuple_element_t<I, std::tuple<Es...>>;
        if constexpr (is_leaf_v<Ei>) {
            cauchyProduct<T, N, M>(out, a, std::get<I>(ops_).coeffs());
        } else {
            std::get<I>(ops_).evalTo(*b);
            cauchyProduct<T, N, M>(out, a, *b);
        }
        a = out;   // promote result to accumulator for the next step
    }
};

} // namespace da::detail
