#pragma once

#include <tax/expr/base.hpp>

namespace da::detail {

// =============================================================================
// §7d  ScalarDivLExpr<E>
// =============================================================================
// Separate node for  s / DA  because OpScalarDivL is templated on N and M.
//
// Leaf optimisation: when E is a leaf, its coefficient storage can be passed
// directly to seriesReciprocal — no need to copy into out first and then
// snapshot out into a (the usual aliasing guard).  Saves 1 temp.

template <typename E>
class ScalarDivLExpr
    : public da::DAExpr<ScalarDivLExpr<E>, typename E::scalar_type, E::order, E::nvars>
{
public:
    using T = typename E::scalar_type;
    static constexpr int N = E::order, M = E::nvars;
    using coeff_array = std::array<T, numMonomials(N, M)>;

    constexpr ScalarDivLExpr(const E& e, T s) noexcept : e_(e), s_(s) {}

    constexpr void evalTo(coeff_array& out) const noexcept {
        if constexpr (is_leaf_v<E>) {
            // Leaf: pass e_'s storage directly — out != e_.coeffs(), so no aliasing.
            seriesReciprocal<T, N, M>(out, e_.coeffs());   // 0 temps
            scaleInPlace<T, numMonomials(N,M)>(out, s_);
        } else {
            // Non-leaf: materialise into out, snapshot to guard reciprocal recurrence.
            e_.evalTo(out);
            const auto a = out;                              // snapshot (aliasing guard)
            seriesReciprocal<T, N, M>(out, a);
            scaleInPlace<T, numMonomials(N,M)>(out, s_);
        }
    }
private:
    stored_t<E> e_;
    T s_;
};

} // namespace da::detail
