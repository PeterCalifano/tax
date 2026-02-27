#pragma once

#include <tax/expr/base.hpp>
#include <tax/kernels.hpp>

namespace da {

// =============================================================================
// §8  DA<T, N, M> — leaf / materialised type
// =============================================================================

template <typename T, int N, int M = 1>
class DA
    : public DAExpr<DA<T, N, M>, T, N, M>
    , public DALeaf
{
public:
    static_assert(N >= 0, "DA order must be non-negative");
    static_assert(M >= 1, "Number of variables must be at least 1");

    static constexpr std::size_t ncoef = detail::numMonomials(N, M);
    using coeff_array = std::array<T, ncoef>;
    using point_type  = std::array<T, M>;

    // ── Constructors ──────────────────────────────────────────────────────────

    constexpr DA() noexcept : c_{} {}
    explicit constexpr DA(coeff_array c) noexcept : c_(std::move(c)) {}
    /*implicit*/ constexpr DA(T val) noexcept : c_{} { c_[0] = val; }

    /// Materialise any expression: a single evalTo into c_.  Zero copies.
    template <typename Derived>
    /*implicit*/ constexpr DA(const DAExpr<Derived, T, N, M>& expr) noexcept
        : c_{} { expr.self().evalTo(c_); }

    // ── Variable factories ────────────────────────────────────────────────────

    /// variable<I>(x0): the I-th DA variable expanded around x0.
    ///   c[0] = x0[I],  c[e_I] = 1,  all other coefficients zero.
    template <int I>
    [[nodiscard]] static constexpr DA variable(const point_type& x0) noexcept {
        static_assert(I >= 0 && I < M, "Variable index out of range");
        coeff_array c{};
        c[0] = x0[I];
        if constexpr (N >= 1) {
            MultiIndex<M> ei{};
            ei[I] = 1;
            c[detail::flatIndex<M>(ei)] = T{1};
        }
        return DA{c};
    }

    /// variables(x0): create all M variables as a tuple.
    ///   auto [x, y] = DA<double,3,2>::variables({1.0, 2.0});
    [[nodiscard]] static constexpr auto variables(const point_type& x0) noexcept {
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::tuple{ variable<int(I)>(x0)... };
        }(std::make_index_sequence<std::size_t(M)>{});
    }

    [[nodiscard]] static constexpr DA constant(T v) noexcept { return DA{v}; }

    // ── evalTo / addTo / subTo ────────────────────────────────────────────────
    //
    // DA is a leaf: every accumulation operation has direct access to c_ and
    // therefore never needs to materialise into an intermediate buffer.
    //
    //   evalTo → array copy                (0 additional temps)
    //   addTo  → addInPlace(out, c_)       (0 additional temps)
    //   subTo  → subInPlace(out, c_)       (0 additional temps)
    //
    // These overrides are what make the recursive BinExpr<Add> addTo chain
    // terminate with 0 temporaries for all-leaf addition expressions.

    constexpr void evalTo(coeff_array& out) const noexcept { out = c_; }

    constexpr void addTo(coeff_array& out) const noexcept
    { detail::addInPlace<T, ncoef>(out, c_); }

    constexpr void subTo(coeff_array& out) const noexcept
    { detail::subInPlace<T, ncoef>(out, c_); }

    // ── Element access ────────────────────────────────────────────────────────

    [[nodiscard]] constexpr T  operator[](std::size_t i) const noexcept { return c_[i]; }
    [[nodiscard]] constexpr T& operator[](std::size_t i)       noexcept { return c_[i]; }
    [[nodiscard]] constexpr const coeff_array& coeffs() const noexcept  { return c_; }
    [[nodiscard]] constexpr T value() const noexcept { return c_[0]; }

    [[nodiscard]] constexpr T coeff(const MultiIndex<M>& alpha) const noexcept
    { return c_[detail::flatIndex<M>(alpha)]; }

    [[nodiscard]] constexpr T derivative(const MultiIndex<M>& alpha) const noexcept {
        std::size_t fac = 1;
        for (int i = 0; i < M; ++i) for (int j = 1; j <= alpha[i]; ++j) fac *= std::size_t(j);
        return c_[detail::flatIndex<M>(alpha)] * T(fac);
    }

    // ── In-place operators ────────────────────────────────────────────────────

    constexpr DA& operator+=(const DA& o) noexcept
    { detail::addInPlace<T, ncoef>(c_, o.c_); return *this; }
    constexpr DA& operator-=(const DA& o) noexcept
    { detail::subInPlace<T, ncoef>(c_, o.c_); return *this; }
    template <typename Derived>
    constexpr DA& operator+=(const DAExpr<Derived, T, N, M>& e) noexcept
    { coeff_array t{}; e.self().evalTo(t); detail::addInPlace<T, ncoef>(c_, t); return *this; }
    template <typename Derived>
    constexpr DA& operator-=(const DAExpr<Derived, T, N, M>& e) noexcept
    { coeff_array t{}; e.self().evalTo(t); detail::subInPlace<T, ncoef>(c_, t); return *this; }
    constexpr DA& operator*=(T s) noexcept
    { detail::scaleInPlace<T, ncoef>(c_, s); return *this; }
    constexpr DA& operator/=(T s) noexcept
    { detail::scaleInPlace<T, ncoef>(c_, T{1} / s); return *this; }

private:
    coeff_array c_;
};

} // namespace da
