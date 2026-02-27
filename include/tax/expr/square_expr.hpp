#pragma once

#include <tax/expr/base.hpp>

namespace da::detail {

// =============================================================================
// §7e  SquareExpr<E>
// =============================================================================
//
// ET-optimal squaring: materialise E once into a local temp `a`, then write
// the Cauchy self-product directly into out.
//
//   evalTo(out):  e_.evalTo(a);              // 1 write pass  (E -> a)
//                 cauchyProduct(out, a, a);  // 1 write pass  (a^2 -> out)
//
// vs. FuncNMExpr<E, OpSquare> old approach:
//   evalTo(out):  e_.evalTo(out);            // 1 write pass  (E -> out  [junk])
//                 a = out;                   // 1 write pass  (out -> a  [snapshot])
//                 cauchyProduct(out, a, a);  // 1 write pass  (a^2 -> out)
//
// Saves: one full coefficient-array write pass (ncoef element copies).

template <typename E>
class SquareExpr
    : public da::DAExpr<SquareExpr<E>, typename E::scalar_type, E::order, E::nvars>
{
public:
    using T = typename E::scalar_type;
    static constexpr int N = E::order, M = E::nvars;
    using coeff_array = std::array<T, numMonomials(N, M)>;

    explicit constexpr SquareExpr(const E& e) noexcept : e_(e) {}

    constexpr void evalTo(coeff_array& out) const noexcept {
        coeff_array a{};
        e_.evalTo(a);                        // materialise input into a  (1 temp)
        seriesSquare<T, N, M>(out, a);       // write a^2 directly to out
    }
private:
    stored_t<E> e_;
};

} // namespace da::detail
