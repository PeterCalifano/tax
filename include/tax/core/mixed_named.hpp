#pragma once

// Named, per-axis-order Taylor expansions: MixedTaylorExpansion
// wraps a dense TaylorExpansion<T, MixedScheme<Group<Dim,Order>...>>
// and attaches a canonical (sorted, unique) list of named ordered axes
// (OrderedAxis<Name, Dim, Order>).
//
// This is additive to (and does not alter) the existing NamedTaylorExpansion
// which remains the joint-simplex named layer. Embed / compose / slice / deriv
// are later tasks; this file delivers only the type + factories + accessors.

#include <array>
#include <cstddef>
#include <tax/core/concepts.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/named.hpp>
#include <tax/core/scheme/mixed.hpp>
#include <tax/core/taylor_expansion.hpp>

namespace tax::named
{

// ---------------------------------------------------------------------------
// OrderedAxis — a named axis with its own per-axis truncation order
// ---------------------------------------------------------------------------

/// A named axis: the compile-time string `Name` labels a block of `Dim`
/// consecutive variables truncated at order `Order`.
template < FixedString Name, int Dim, int Order >
struct OrderedAxis
{
    static constexpr auto name = Name;
    static constexpr int dim = Dim;
    static constexpr int order = Order;
    static_assert( Dim >= 1, "OrderedAxis dimension must be at least 1" );
    static_assert( Order >= 0, "OrderedAxis order must be non-negative" );
};

// ---------------------------------------------------------------------------
// Axis-list metafunction: OrderedAxis pack -> MixedScheme<Group<Dim,Order>...>
// ---------------------------------------------------------------------------

namespace detail
{

/// Map a pack of OrderedAxis types to MixedScheme<Group<Dim,Order>...>.
/// Axis order == group order in the scheme (group 0 = first axis, etc.).
template < typename... Axes >
struct AxesToMixedScheme
{
    using type = MixedScheme< Group< Axes::dim, Axes::order >... >;
};

template < typename... Axes >
using AxesToMixedScheme_t = typename AxesToMixedScheme< Axes... >::type;

}  // namespace detail

// ---------------------------------------------------------------------------
// MixedTaylorExpansion — the named per-axis-order type
// ---------------------------------------------------------------------------

template < typename T, typename... Axes >
    requires Scalar< T >
class MixedTaylorExpansion
{
   public:
    using axis_list = detail::TypeList< Axes... >;
    using scalar_type = T;

    static_assert( detail::IsCanonical< axis_list >::value,
                   "MixedTaylorExpansion axes must be sorted by name and unique; build via "
                   "tax::mixed::variable()/variables() rather than spelling them by hand" );

    /// Number of underlying variables (sum of axis dimensions).
    static constexpr int vars_v = detail::TotalDim< axis_list >::value;

    /// Underlying anonymous dense expansion type (MixedScheme backing).
    using Inner = TaylorExpansion< T, detail::AxesToMixedScheme_t< Axes... >, storage::Dense >;
    using Input = typename Inner::Input;

    // Mirror the underlying storage traits.
    using container_t = typename Inner::container_t;
    static constexpr std::size_t nCoefficients = Inner::nCoefficients;

    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    constexpr MixedTaylorExpansion() noexcept = default;

    /// Constant expansion (value, all higher-order coefficients zero).
    /*implicit*/ constexpr MixedTaylorExpansion( T v ) noexcept : inner_{ v } {}

    /// Wrap an existing anonymous expansion carrying these axes.
    explicit constexpr MixedTaylorExpansion( const Inner& inner ) noexcept : inner_{ inner } {}

    // ------------------------------------------------------------------
    // Access
    // ------------------------------------------------------------------

    /// Constant (zeroth) coefficient.
    [[nodiscard]] constexpr T value() const noexcept { return inner_.value(); }

    /// The underlying anonymous expansion (const and mutable).
    [[nodiscard]] constexpr const Inner& inner() const noexcept { return inner_; }
    [[nodiscard]] constexpr Inner& inner() noexcept { return inner_; }

    /// Read coefficient at flat index `k`.
    [[nodiscard]] constexpr T operator[]( std::size_t k ) const noexcept { return inner_[k]; }

    /// Write coefficient at flat index `k`.
    [[nodiscard]] constexpr T& operator[]( std::size_t k ) noexcept { return inner_[k]; }

    /// Runtime multi-index coefficient lookup.
    [[nodiscard]] constexpr T coeff( const MultiIndex< vars_v >& alpha ) const noexcept
    {
        return inner_.coeff( alpha );
    }

    /// Compile-time multi-index coefficient lookup.
    template < int... Alpha >
    [[nodiscard]] constexpr T coeff() const noexcept
    {
        return inner_.template coeff< Alpha... >();
    }

    /// Runtime partial derivative value `d^|alpha| f / dx^alpha` at x0.
    [[nodiscard]] constexpr T derivative( const MultiIndex< vars_v >& alpha ) const noexcept
    {
        return inner_.derivative( alpha );
    }

   private:
    Inner inner_{};
};

}  // namespace tax::named

// ---------------------------------------------------------------------------
// Public re-exports: OrderedAxis and MixedTaylorExpansion under `tax`
// ---------------------------------------------------------------------------

namespace tax
{
using named::MixedTaylorExpansion;
using named::OrderedAxis;
}  // namespace tax

// ---------------------------------------------------------------------------
// Factories: `tax::mixed::variable` / `tax::mixed::variables`
// ---------------------------------------------------------------------------

namespace tax::mixed
{

/// Build the single coordinate variable of a 1-D ordered axis `Name` at `x0`,
/// truncated to per-axis order `Order`.
template < tax::named::FixedString Name, int Order >
[[nodiscard]] constexpr auto variable( double x0 ) noexcept
{
    using Ax = tax::named::OrderedAxis< Name, 1, Order >;
    using E = tax::named::MixedTaylorExpansion< double, Ax >;
    typename E::Input p{ x0 };
    return E{ E::Inner::template variable< 0 >( p ) };
}

/// Build the `D` coordinate variables of a `D`-dimensional ordered axis `Name`
/// at `x0`, each truncated to per-axis order `Order`. Returns a plain
/// `std::array` (as `tax::named::variables` does).
template < tax::named::FixedString Name, int Order, std::size_t D >
[[nodiscard]] constexpr auto variables( const std::array< double, D >& x0 ) noexcept
{
    using Ax = tax::named::OrderedAxis< Name, int( D ), Order >;
    using E = tax::named::MixedTaylorExpansion< double, Ax >;
    std::array< E, D > out{};
    [&]< std::size_t... I >( std::index_sequence< I... > ) {
        ( ( out[I] = E{ E::Inner::template variable< int( I ) >( x0 ) } ), ... );
    }( std::make_index_sequence< D >{} );
    return out;
}

}  // namespace tax::mixed
