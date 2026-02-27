#pragma once

#include <tax/expr/base.hpp>
#include <tax/expr/arithmetic_ops.hpp>

namespace da::detail {

// =============================================================================
// §7a  BinExpr<L, R, Op>
// =============================================================================
//
// evalTo — three paths selected entirely at compile time:
//
//   Op::is_additive (Add/Sub):
//     l_.evalTo(out)                           [0 temps if l is leaf]
//     r_.addTo(out) or r_.subTo(out)           [0 temps if r is leaf; recurses for ET nodes]
//     Peak: 0 if both leaves, 1 if r is non-leaf ET node.
//
//   !Op::is_additive (Mul/Div) with leaf detection:
//     Both leaves   : Op::apply(out, l_.coeffs(), r_.coeffs())    — 0 temps
//     R leaf only   : l_.evalTo(la); Op::apply(out, la, r_.c)     — 1 temp
//     L leaf only   : r_.evalTo(rb); Op::apply(out, l_.c, rb)     — 1 temp
//     Neither leaf  : l_.evalTo(la); r_.evalTo(rb); apply         — 2 temps
//
// addTo / subTo — accumulate into an already-populated 'out':
//
//   Add/Sub:   recurse l_.addTo/subTo and r_.addTo/subTo          — 0 temps for leaf chains
//   Mul:       cauchyAccumulate with leaf detection                — same temp counts as evalTo
//   Div:       fall back to evalTo into temp, then add/sub        — 1 extra temp

template <typename L, typename R, typename Op>
class BinExpr
    : public da::DAExpr<BinExpr<L, R, Op>, typename L::scalar_type, L::order, L::nvars>
{
    static_assert(L::order  == R::order  && L::nvars == R::nvars &&
                  std::is_same_v<typename L::scalar_type, typename R::scalar_type>);
public:
    using T = typename L::scalar_type;
    static constexpr int N = L::order, M = L::nvars;
    using coeff_array = std::array<T, numMonomials(N, M)>;

    constexpr BinExpr(const L& l, const R& r) noexcept : l_(l), r_(r) {}

    // ── Mode A: evalTo ────────────────────────────────────────────────────────

    constexpr void evalTo(coeff_array& out) const noexcept {
        if constexpr (Op::is_additive) {
            // L is evaluated in-place; R accumulates via addTo/subTo.
            // If R is a leaf, addTo/subTo = addInPlace/subInPlace: 0 temps.
            // If R is an ET node, its addTo recurses: 0 temps if its tree is all leaves.
            l_.evalTo(out);
            if constexpr (Op::negate_right) r_.subTo(out);
            else                             r_.addTo(out);
        } else {
            // Mul / Div: use leaf detection to avoid materialising leaf operands.
            // l_.coeffs() / r_.coeffs() are const-refs into existing DA storage.
            if constexpr (is_leaf_v<L> && is_leaf_v<R>)
                Op::template apply<T>(out, l_.coeffs(), r_.coeffs());  // 0 temps
            else if constexpr (is_leaf_v<R>) {
                coeff_array la{};
                l_.evalTo(la);
                Op::template apply<T>(out, la, r_.coeffs());           // 1 temp
            } else if constexpr (is_leaf_v<L>) {
                coeff_array rb{};
                r_.evalTo(rb);
                Op::template apply<T>(out, l_.coeffs(), rb);           // 1 temp
            } else {
                coeff_array la{}, rb{};
                l_.evalTo(la);
                r_.evalTo(rb);
                Op::template apply<T>(out, la, rb);                    // 2 temps
            }
        }
    }

    // ── Mode B: addTo — out += this expression ────────────────────────────────

    constexpr void addTo(coeff_array& out) const noexcept {
        if constexpr (Op::is_additive) {
            // Recurse both branches; 0 temps all the way down if all leaves.
            l_.addTo(out);
            if constexpr (Op::negate_right) r_.subTo(out);
            else                             r_.addTo(out);
        } else if constexpr (Op::is_convolution) {
            // Mul: cauchyAccumulate with leaf detection.
            if constexpr (is_leaf_v<L> && is_leaf_v<R>)
                cauchyAccumulate<T,N,M>(out, l_.coeffs(), r_.coeffs());  // 0 temps
            else if constexpr (is_leaf_v<R>) {
                coeff_array la{};
                l_.evalTo(la);
                cauchyAccumulate<T,N,M>(out, la, r_.coeffs());           // 1 temp
            } else if constexpr (is_leaf_v<L>) {
                coeff_array rb{};
                r_.evalTo(rb);
                cauchyAccumulate<T,N,M>(out, l_.coeffs(), rb);           // 1 temp
            } else {
                coeff_array la{}, rb{};
                l_.evalTo(la);
                r_.evalTo(rb);
                cauchyAccumulate<T,N,M>(out, la, rb);                    // 2 temps
            }
        } else {
            // Div and other non-convolution ops: materialise then accumulate.
            coeff_array tmp{};
            evalTo(tmp);
            addInPlace<T, numMonomials(N,M)>(out, tmp);
        }
    }

    // ── Mode C: subTo — out -= this expression ────────────────────────────────

    constexpr void subTo(coeff_array& out) const noexcept {
        if constexpr (Op::is_additive) {
            // -(l OP r) where OP is + or -:
            //   -(l + r) → -l - r
            //   -(l - r) → -l + r
            l_.subTo(out);
            if constexpr (Op::negate_right) r_.addTo(out);  // double negation
            else                             r_.subTo(out);
        } else {
            // Non-additive: materialise then subtract.
            coeff_array tmp{};
            evalTo(tmp);
            subInPlace<T, numMonomials(N,M)>(out, tmp);
        }
    }

private:
    stored_t<L> l_;
    stored_t<R> r_;
};

} // namespace da::detail
