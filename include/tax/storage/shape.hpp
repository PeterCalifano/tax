#pragma once

#include <cstddef>

#include <tax/utils/combinatorics.hpp>
#include <tax/utils/fwd.hpp>

namespace tax::detail
{

/**
 * @brief Empty-base helper carrying a compile-time order.
 * @tparam N Compile-time non-negative order (must be `>= 0`).
 *
 * Provides a uniform `order()` accessor whose return value is `constexpr` and
 * cost-free. Empty-base optimisation collapses this to zero bytes inside
 * `ShapeBase`.
 */
template < int N >
struct OrderHolder
{
    static_assert( N == Dynamic || N >= 0, "Order must be >= 0 or tax::Dynamic" );

    /// @brief Whether the order is fixed at compile time.
    static constexpr bool order_static = true;

    /// @brief Compile-time order accessor.
    [[nodiscard]] static constexpr std::size_t order() noexcept { return std::size_t( N ); }

   protected:
    constexpr OrderHolder() noexcept = default;
    /// @brief Runtime-arg ctor — argument is ignored on the static path.
    constexpr explicit OrderHolder( std::size_t /*o*/ ) noexcept {}
};

/// @brief Specialisation carrying the order as a runtime member.
template <>
struct OrderHolder< Dynamic >
{
    static constexpr bool order_static = false;

    [[nodiscard]] constexpr std::size_t order() const noexcept { return order_; }

   protected:
    constexpr OrderHolder() noexcept : order_{ 0 } {}
    constexpr explicit OrderHolder( std::size_t o ) noexcept : order_{ o } {}

    std::size_t order_;
};

/**
 * @brief Empty-base helper carrying a compile-time number of variables.
 * @tparam M Compile-time nvars (must be `>= 1` or `tax::Dynamic`).
 */
template < int M >
struct VarsHolder
{
    static_assert( M == Dynamic || M >= 1, "Number of variables must be >= 1 or tax::Dynamic" );

    static constexpr bool vars_static = true;

    [[nodiscard]] static constexpr std::size_t nvars() noexcept { return std::size_t( M ); }

   protected:
    constexpr VarsHolder() noexcept = default;
    constexpr explicit VarsHolder( std::size_t /*v*/ ) noexcept {}
};

template <>
struct VarsHolder< Dynamic >
{
    static constexpr bool vars_static = false;

    [[nodiscard]] constexpr std::size_t nvars() const noexcept { return nvars_; }

   protected:
    constexpr VarsHolder() noexcept : nvars_{ 0 } {}
    constexpr explicit VarsHolder( std::size_t v ) noexcept : nvars_{ v } {}

    std::size_t nvars_;
};

/**
 * @brief Shape base for `TaylorExpansionT<T, N, M>` combining the order and
 *        nvars holders via private multiple inheritance.
 *
 * Any dimension that is `tax::Dynamic` is stored as a runtime member; any
 * dimension that is a non-negative integer is returned as a `constexpr`
 * value from an empty base, collapsed to zero size by EBO. This means a
 * fully-static `ShapeBase<5, 3>` has `sizeof == 1` (empty base), a half-
 * dynamic `ShapeBase<5, Dynamic>` carries one `std::size_t`, and a fully-
 * dynamic `ShapeBase<Dynamic, Dynamic>` carries two.
 */
template < int N, int M >
struct ShapeBase : private OrderHolder< N >, private VarsHolder< M >
{
    using OrderHolder< N >::order;
    using VarsHolder< M >::nvars;

    /// @brief Both order and nvars are compile-time constants.
    static constexpr bool fully_static = ( N != Dynamic ) && ( M != Dynamic );
    /// @brief Either dimension is `tax::Dynamic`.
    static constexpr bool any_dynamic = !fully_static;
    /// @brief Both dimensions are `tax::Dynamic`.
    static constexpr bool fully_dynamic = ( N == Dynamic ) && ( M == Dynamic );

    /// @brief Default-construct: zero-size on dynamic dimensions.
    constexpr ShapeBase() noexcept = default;

    /// @brief Construct with explicit `(order, nvars)`; ignored on static dimensions.
    constexpr ShapeBase( std::size_t o, std::size_t v ) noexcept
        : OrderHolder< N >( o ), VarsHolder< M >( v )
    {
    }

    /// @brief Number of stored coefficients for the current shape.
    [[nodiscard]] constexpr std::size_t coeffsSize() const noexcept
    {
        return numMonomials( this->order(), this->nvars() );
    }
};

}  // namespace tax::detail
