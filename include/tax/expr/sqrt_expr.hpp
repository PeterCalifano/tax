#pragma once

#include <tax/expr/base.hpp>

namespace da::detail {

// =============================================================================
// §7g  SqrtExpr<E>
// =============================================================================
//
// ET-optimal sqrt: materialise E once into `a`, then run the g·g=a recurrence
// writing g directly into out.
//
//   evalTo(out):  e_.evalTo(a);            // 1 write pass  (E -> a)
//                 seriesSqrt(out, a);      // 1 write pass  (recurrence -> out)
//
// vs. FuncNMExpr<E, OpSqrt> old approach:
//   evalTo(out):  e_.evalTo(out);          // 1 write pass  (E -> out  [junk])
//                 a = out;                 // 1 write pass  (out -> a  [snapshot])
//                 seriesSqrt(out, a);      // 1 write pass  (recurrence -> out)
//
// Saves: one full coefficient-array write pass (ncoef element copies).
//
// Precondition: e_[0] = E.value() > 0.

template <typename E>
class SqrtExpr
    : public da::DAExpr<SqrtExpr<E>, typename E::scalar_type, E::order, E::nvars>
{
public:
    using T = typename E::scalar_type;
    static constexpr int N = E::order, M = E::nvars;
    using coeff_array = std::array<T, numMonomials(N, M)>;

    explicit constexpr SqrtExpr(const E& e) noexcept : e_(e) {}

    constexpr void evalTo(coeff_array& out) const noexcept {
        coeff_array a{};
        e_.evalTo(a);                        // materialise input into a  (1 temp)
        seriesSqrt<T, N, M>(out, a);         // write sqrt(a) directly to out
    }
private:
    stored_t<E> e_;
};

} // namespace da::detail
