#pragma once

#include <tax/expr/base.hpp>

namespace da::detail {

// =============================================================================
// §7f  CubeExpr<E>
// =============================================================================
//
// ET-optimal cubing: materialise E once into `a`, then use the direct triple
// convolution to write a^3 straight into out — no intermediate a^2 array.
//
//   evalTo(out):  e_.evalTo(a);             // 1 write pass  (E -> a)
//                 seriesCube(out, a);        // 1 write pass  (triple-conv -> out)
//
// vs. naive  a * a^2:
//   evalTo(out):  e_.evalTo(a);             // 1 write pass  (E -> a)
//                 cauchyProduct(a2, a, a);  // 1 write pass  (a^2 -> a2)
//                 cauchyProduct(out, a, a2);// 1 write pass  (a*a2 -> out)
//
// Saves: one full coefficient-array write pass (the a^2 intermediate).

template <typename E>
class CubeExpr
    : public da::DAExpr<CubeExpr<E>, typename E::scalar_type, E::order, E::nvars>
{
public:
    using T = typename E::scalar_type;
    static constexpr int N = E::order, M = E::nvars;
    using coeff_array = std::array<T, numMonomials(N, M)>;

    explicit constexpr CubeExpr(const E& e) noexcept : e_(e) {}

    constexpr void evalTo(coeff_array& out) const noexcept {
        coeff_array a{};
        e_.evalTo(a);                        // materialise input into a  (1 temp)
        seriesCube<T, N, M>(out, a);         // write a^3 directly to out
    }
private:
    stored_t<E> e_;
};

} // namespace da::detail
