#pragma once

#include <tax/da_type.hpp>
#include <tax/expr/bin_expr.hpp>
#include <tax/expr/unary_expr.hpp>
#include <tax/expr/scalar_expr.hpp>
#include <tax/expr/scalar_div_l_expr.hpp>
#include <tax/expr/square_expr.hpp>
#include <tax/expr/cube_expr.hpp>
#include <tax/expr/sqrt_expr.hpp>
#include <tax/expr/sum_expr.hpp>
#include <tax/expr/product_expr.hpp>

namespace da {

// =============================================================================
// §9  Operator overloads and power/root free functions
// =============================================================================

template <typename L, typename R>
concept CompatibleDA =
    (L::order == R::order) && (L::nvars == R::nvars) &&
    std::is_same_v<typename L::scalar_type, typename R::scalar_type>;

#define DA_BASE(E) DAExpr<E, typename E::scalar_type, E::order, E::nvars>

// ── DA + DA  — four overloads for complete associativity flattening ──────────
//
// Every combination of SumExpr and other DA expressions is handled so that
// a+b+c+d always produces SumExpr<A,B,C,D> regardless of parenthesisation.
//
// Overload priority (most specific wins via partial ordering):
//   (4) SumExpr<Ls...> + SumExpr<Rs...>  →  SumExpr<Ls...,Rs...>   concat
//   (2) SumExpr<Ls...> + DAExpr<R>       →  SumExpr<Ls...,R>        append
//   (3) DAExpr<L>      + SumExpr<Rs...>  →  SumExpr<L,Rs...>        prepend
//   (1) DAExpr<L>      + DAExpr<R>       →  SumExpr<L,R>            generic

// (1) generic
template <typename L, typename R> requires CompatibleDA<L, R>
[[nodiscard]] constexpr auto operator+(const DA_BASE(L)& l, const DA_BASE(R)& r) noexcept
{ return detail::SumExpr<L, R>{l.self(), r.self()}; }

// (2) left-extend
template <typename... Ls, typename R>
requires CompatibleDA<detail::SumExpr<Ls...>, R>
[[nodiscard]] constexpr auto operator+(const detail::SumExpr<Ls...>& l,
                                       const DA_BASE(R)& r) noexcept
{ return l.template append<R>(r.self()); }

// (3) right-extend
template <typename L, typename... Rs>
requires CompatibleDA<L, detail::SumExpr<Rs...>>
[[nodiscard]] constexpr auto operator+(const DA_BASE(L)& l,
                                       const detail::SumExpr<Rs...>& r) noexcept
{ return r.template prepend<L>(l.self()); }

// (4) concat
template <typename... Ls, typename... Rs>
requires CompatibleDA<detail::SumExpr<Ls...>, detail::SumExpr<Rs...>>
[[nodiscard]] constexpr auto operator+(const detail::SumExpr<Ls...>& l,
                                       const detail::SumExpr<Rs...>& r) noexcept
{ return l.concat(r); }

// ── DA - DA  — subtraction stays as BinExpr<OpSub> ───────────────────────────
// Subtraction doesn't chain as naturally as addition (signs matter) and the
// existing BinExpr<OpSub> with addTo/subTo recursion is already correct.
// Notably: -(BinExpr<Sub>) dispatches via subTo correctly through sign flips.

template <typename L, typename R> requires CompatibleDA<L, R>
[[nodiscard]] constexpr auto operator-(const DA_BASE(L)& l, const DA_BASE(R)& r) noexcept
{ return detail::BinExpr<L, R, detail::OpSub>{l.self(), r.self()}; }

// ── DA * DA  — two overloads: generic + left-extend ──────────────────────────
//
// Right-extension (DA * ProductExpr) and concat (Product * Product) are
// intentionally omitted: multiplication is not commutative in the series sense
// for non-scalar expansions and the left-associative chain a*b*c*d is the
// common idiom.  The two overloads handle all typical usage.
//
// (1) generic: produces ProductExpr<L,R>
// (2) left-extend: ProductExpr<Ls...> * R → ProductExpr<Ls..., R>

// (1) generic
template <typename L, typename R> requires CompatibleDA<L, R>
[[nodiscard]] constexpr auto operator*(const DA_BASE(L)& l, const DA_BASE(R)& r) noexcept
{ return detail::ProductExpr<L, R>{l.self(), r.self()}; }

// (2) left-extend: ProductExpr<Ls...> * DA expr → ProductExpr<Ls..., R>
template <typename... Ls, typename R>
requires CompatibleDA<detail::ProductExpr<Ls...>, R>
[[nodiscard]] constexpr auto operator*(const detail::ProductExpr<Ls...>& l,
                                       const DA_BASE(R)& r) noexcept
{ return l.template append<R>(r.self()); }

template <typename L, typename R> requires CompatibleDA<L, R>
[[nodiscard]] constexpr auto operator/(const DA_BASE(L)& l, const DA_BASE(R)& r) noexcept
{ return detail::BinExpr<L, R, detail::OpDiv<L::order, L::nvars>>{l.self(), r.self()}; }

// ── Unary negation ────────────────────────────────────────────────────────────

template <typename E>
[[nodiscard]] constexpr auto operator-(const DA_BASE(E)& e) noexcept
{ return detail::UnaryExpr<E, detail::OpNeg>{e.self()}; }

// ── DA op scalar ─────────────────────────────────────────────────────────────

template <typename E>
[[nodiscard]] constexpr auto operator+(const DA_BASE(E)& e, typename E::scalar_type s) noexcept
{ return detail::ScalarExpr<E, detail::OpScalarAddR>{e.self(), s}; }
template <typename E>
[[nodiscard]] constexpr auto operator-(const DA_BASE(E)& e, typename E::scalar_type s) noexcept
{ return detail::ScalarExpr<E, detail::OpScalarSubR>{e.self(), s}; }
template <typename E>
[[nodiscard]] constexpr auto operator*(const DA_BASE(E)& e, typename E::scalar_type s) noexcept
{ return detail::ScalarExpr<E, detail::OpScalarMul>{e.self(), s}; }
template <typename E>
[[nodiscard]] constexpr auto operator/(const DA_BASE(E)& e, typename E::scalar_type s) noexcept
{ return detail::ScalarExpr<E, detail::OpScalarDivR>{e.self(), s}; }

// ── scalar op DA ─────────────────────────────────────────────────────────────

template <typename E>
[[nodiscard]] constexpr auto operator+(typename E::scalar_type s, const DA_BASE(E)& e) noexcept
{ return detail::ScalarExpr<E, detail::OpScalarAddL>{e.self(), s}; }
template <typename E>
[[nodiscard]] constexpr auto operator-(typename E::scalar_type s, const DA_BASE(E)& e) noexcept
{ return detail::ScalarExpr<E, detail::OpScalarSubL>{e.self(), s}; }
template <typename E>
[[nodiscard]] constexpr auto operator*(typename E::scalar_type s, const DA_BASE(E)& e) noexcept
{ return detail::ScalarExpr<E, detail::OpScalarMul>{e.self(), s}; }
template <typename E>
[[nodiscard]] constexpr auto operator/(typename E::scalar_type s, const DA_BASE(E)& e) noexcept
{ return detail::ScalarDivLExpr<E>{e.self(), s}; }

// ── Power and root free functions ─────────────────────────────────────────────
//
// Each returns a lazy ET node.  Materialisation (evalTo into the caller's
// buffer) happens only when the expression is assigned to DA<T,N,M> or when
// .eval() / .value() is called.
//
// Memory contract: each node holds its sub-expression by stored_t<E>
// (by const-ref if E is a leaf, by value if E is an ET node).
// The single temp array needed at evaluation time lives on the stack inside
// evalTo and is gone by the time the outer expression continues.

/// f^2 via Cauchy self-convolution.  1 temp: the materialised input.
template <typename E>
[[nodiscard]] constexpr auto square(const DA_BASE(E)& e) noexcept
{ return detail::SquareExpr<E>{e.self()}; }

/// f^3 via direct triple convolution.  1 temp: the materialised input.
/// No intermediate f^2 array is ever allocated.
template <typename E>
[[nodiscard]] constexpr auto cube(const DA_BASE(E)& e) noexcept
{ return detail::CubeExpr<E>{e.self()}; }

/// Taylor series of sqrt(f) via the g·g = f recurrence.  1 temp: the input.
/// Precondition: e.value() > 0.
template <typename E>
[[nodiscard]] constexpr auto sqrt(const DA_BASE(E)& e) noexcept
{ return detail::SqrtExpr<E>{e.self()}; }

#undef DA_BASE

} // namespace da
