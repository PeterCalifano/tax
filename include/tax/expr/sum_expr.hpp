#pragma once

#include <tax/expr/base.hpp>

namespace da::detail {

// =============================================================================
// §7h  SumExpr<Es...>
// =============================================================================
//
// Variadic addition node.  Holds N operands as a flat tuple; no nesting.
//
// evalTo — writes the first operand into out, then for each subsequent:
//   • leaf operand  → addInPlace(out, e.coeffs())          (0 extra temps)
//   • ET operand    → A tmp{}; e.evalTo(tmp); addInPlace   (1 temp, reused)
//
// For an N-operand sum with K non-leaf terms:
//   • BinExpr chain : K temps simultaneously alive (nested frames)
//   • SumExpr       : min(K, 1) temps (the one tmp is reused serially)
//
// addTo / subTo — recurse into each component's addTo/subTo.
//   For all-leaf SumExprs this propagates down to DA::addTo/subTo: 0 temps.
//
// The four operator+ overloads (generic, extend-left, extend-right, concat)
// guarantee that a+b+c+d always produces SumExpr<A,B,C,D> regardless of
// parenthesisation, so evalTo never sees a nested SumExpr as an operand.

template <typename... Es>
class SumExpr
    : public da::DAExpr<SumExpr<Es...>,
                    typename std::tuple_element_t<0, std::tuple<Es...>>::scalar_type,
                    std::tuple_element_t<0, std::tuple<Es...>>::order,
                    std::tuple_element_t<0, std::tuple<Es...>>::nvars>
{
    static_assert(sizeof...(Es) >= 2, "SumExpr needs at least 2 operands");
    template <typename...> friend class SumExpr;   // for concat/prepend

public:
    using T = typename std::tuple_element_t<0, std::tuple<Es...>>::scalar_type;
    static constexpr int N = std::tuple_element_t<0, std::tuple<Es...>>::order;
    static constexpr int M = std::tuple_element_t<0, std::tuple<Es...>>::nvars;
    using coeff_array = std::array<T, numMonomials(N, M)>;

    explicit constexpr SumExpr(stored_t<Es>... es) noexcept : ops_(es...) {}

    // ── append / prepend / concat — used by operator+ overloads ──────────────

    template <typename E>
    [[nodiscard]] constexpr auto append(stored_t<E> e) const noexcept {
        return std::apply([&](auto const&... x) noexcept {
            return SumExpr<Es..., E>(x..., e);
        }, ops_);
    }

    template <typename E>
    [[nodiscard]] constexpr auto prepend(stored_t<E> e) const noexcept {
        return std::apply([&](auto const&... x) noexcept {
            return SumExpr<E, Es...>(e, x...);
        }, ops_);
    }

    template <typename... Rs>
    [[nodiscard]] constexpr auto concat(const SumExpr<Rs...>& r) const noexcept {
        return std::apply([&](auto const&... rx) noexcept {
            return std::apply([&](auto const&... lx) noexcept {
                return SumExpr<Es..., Rs...>(lx..., rx...);
            }, ops_);
        }, r.ops_);
    }

    // ── evalTo ────────────────────────────────────────────────────────────────

    constexpr void evalTo(coeff_array& out) const noexcept {
        std::get<0>(ops_).evalTo(out);
        accumRest(out, std::make_index_sequence<sizeof...(Es) - 1>{});
    }

    // ── addTo / subTo: recurse into each component ────────────────────────────
    // For all-leaf SumExpr, this reaches DA::addTo/subTo: 0 temps total.

    constexpr void addTo(coeff_array& out) const noexcept {
        std::apply([&](auto const&... e) noexcept { (e.addTo(out), ...); }, ops_);
    }

    constexpr void subTo(coeff_array& out) const noexcept {
        std::apply([&](auto const&... e) noexcept { (e.subTo(out), ...); }, ops_);
    }

private:
    std::tuple<stored_t<Es>...> ops_;

    template <std::size_t... I>
    constexpr void accumRest(coeff_array& out,
                             std::index_sequence<I...>) const noexcept
    { (accumOne<I + 1>(out), ...); }

    template <std::size_t I>
    constexpr void accumOne(coeff_array& out) const noexcept {
        using E = std::tuple_element_t<I, std::tuple<Es...>>;
        if constexpr (is_leaf_v<E>) {
            // Leaf: bypass evalTo entirely — direct addInPlace on coeffs().
            addInPlace<T, numMonomials(N, M)>(out, std::get<I>(ops_).coeffs());
        } else {
            // ET node: materialise into one reused temp, then accumulate.
            coeff_array tmp{};
            std::get<I>(ops_).evalTo(tmp);
            addInPlace<T, numMonomials(N, M)>(out, tmp);
        }
    }
};

} // namespace da::detail
