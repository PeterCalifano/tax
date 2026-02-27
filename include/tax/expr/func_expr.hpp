#pragma once

#include <tax/expr/base.hpp>
#include <tax/expr/math_ops.hpp>

namespace da::detail {

// =============================================================================
// §7d  FuncExpr<E, Op>  — generic series-function expression node
// =============================================================================
//
// Evaluates the sub-expression E, then applies a series-level kernel defined
// by Op.  This is the "math" counterpart of UnaryExpr (which applies in-place
// fixups): FuncExpr materialises E into a separate buffer first because the
// kernel reads from one array and writes to another (out != a).
//
// Leaf optimisation: when E is a DA leaf, its coefficient storage is passed
// directly to the kernel — no temporary needed (0 temps vs 1 temp).
//
// Op tags: OpSquare, OpCube, OpSqrt, OpReciprocal (defined in math_ops.hpp).
// Op signature: static void apply<T>(array& out, const array& a)

template <typename E, typename Op>
class FuncExpr
    : public da::DAExpr<FuncExpr<E, Op>, typename E::scalar_type, E::order, E::nvars>
{
public:
    using T = typename E::scalar_type;
    static constexpr int N = E::order, M = E::nvars;
    using coeff_array = std::array<T, numMonomials(N, M)>;

    explicit constexpr FuncExpr(const E& e) noexcept : e_(e) {}

    constexpr void evalTo(coeff_array& out) const noexcept {
        if constexpr (is_leaf_v<E>) {
            // Leaf: pass e_'s storage directly — out != e_.coeffs(), so no aliasing.
            Op::template apply<T>(out, e_.coeffs());   // 0 temps
        } else {
            coeff_array a{};
            e_.evalTo(a);                              // materialise E into a
            Op::template apply<T>(out, a);             // 1 temp
        }
    }

private:
    stored_t<E> e_;
};

} // namespace da::detail
