#pragma once

#include <tax/kernels.hpp>

namespace da::detail {

// =============================================================================
// §6  Operation tags for BinExpr / ScalarExpr / UnaryExpr
// =============================================================================

// -- Left-in-place binary ops (additive) ──────────────────────────────────────
//
// is_additive   = true  → BinExpr uses addTo/subTo on the right operand
// negate_right  = false → OpAdd adds right;  true → OpSub subtracts right
// is_convolution = false (not used for additive ops)

struct OpAdd {
    static constexpr bool leftInPlace    = true;   // legacy alias
    static constexpr bool is_additive    = true;
    static constexpr bool negate_right   = false;
    static constexpr bool is_convolution = false;
    template <typename T, std::size_t S>
    static constexpr void fuse(std::array<T, S>& o, const std::array<T, S>& r) noexcept
    { addInPlace<T, S>(o, r); }
};

struct OpSub {
    static constexpr bool leftInPlace    = true;
    static constexpr bool is_additive    = true;
    static constexpr bool negate_right   = true;
    static constexpr bool is_convolution = false;
    template <typename T, std::size_t S>
    static constexpr void fuse(std::array<T, S>& o, const std::array<T, S>& r) noexcept
    { subInPlace<T, S>(o, r); }
};

// -- Non-additive binary ops (Cauchy / division) ───────────────────────────────
//
// is_convolution = true  → BinExpr uses cauchyAccumulate in addTo
// is_convolution = false → BinExpr falls back to evalTo+addInPlace in addTo

template <int N, int M>
struct OpMul {
    static constexpr bool leftInPlace    = false;
    static constexpr bool is_additive    = false;
    static constexpr bool is_convolution = true;
    template <typename T>
    static constexpr void apply(
        std::array<T, numMonomials(N, M)>&       o,
        const std::array<T, numMonomials(N, M)>& a,
        const std::array<T, numMonomials(N, M)>& b) noexcept
    { cauchyProduct<T, N, M>(o, a, b); }
};

template <int N, int M>
struct OpDiv {
    static constexpr bool leftInPlace    = false;
    static constexpr bool is_additive    = false;
    static constexpr bool is_convolution = false;
    template <typename T>
    static constexpr void apply(
        std::array<T, numMonomials(N, M)>&       o,
        const std::array<T, numMonomials(N, M)>& a,
        const std::array<T, numMonomials(N, M)>& b) noexcept
    {
        std::array<T, numMonomials(N, M)> rec{};
        seriesReciprocal<T, N, M>(rec, b);
        cauchyProduct<T, N, M>(o, a, rec);
    }
};

// -- Scalar ops (0 temps, all in-place on out) ─────────────────────────────────

struct OpScalarAddR { template <typename T, std::size_t S>
    static constexpr void apply(std::array<T,S>& o, T s) noexcept { o[0] += s; } };
struct OpScalarAddL { template <typename T, std::size_t S>
    static constexpr void apply(std::array<T,S>& o, T s) noexcept { o[0] += s; } };
struct OpScalarSubR { template <typename T, std::size_t S>
    static constexpr void apply(std::array<T,S>& o, T s) noexcept { o[0] -= s; } };
struct OpScalarSubL { template <typename T, std::size_t S>
    static constexpr void apply(std::array<T,S>& o, T s) noexcept
    { negateInPlace<T,S>(o); o[0] += s; } };
struct OpScalarMul  { template <typename T, std::size_t S>
    static constexpr void apply(std::array<T,S>& o, T s) noexcept
    { scaleInPlace<T,S>(o, s); } };
struct OpScalarDivR { template <typename T, std::size_t S>
    static constexpr void apply(std::array<T,S>& o, T s) noexcept
    { scaleInPlace<T,S>(o, T{1} / s); } };

// -- s / DA: 1 temp (aliasing guard for the reciprocal recurrence) ------------

template <int N, int M>
struct OpScalarDivL {
    template <typename T>
    static constexpr void apply(
        std::array<T, numMonomials(N, M)>& out, T s) noexcept
    {
        const auto a = out;                              // snapshot (aliasing guard)
        seriesReciprocal<T, N, M>(out, a);
        scaleInPlace<T, numMonomials(N, M)>(out, s);
    }
};

// -- Unary negation (0 temps) -------------------------------------------------

struct OpNeg { template <typename T, std::size_t S>
    static constexpr void apply(std::array<T,S>& o) noexcept
    { negateInPlace<T,S>(o); } };

} // namespace da::detail
