// include/tax/la/named.hpp
//
// Eigen integration for the named-axis layer (tax::named):
//
//   1. Eigen::NumTraits< tax::named::NamedTaylorExpansion<...> > — lets a named
//      expansion act as a first-class Eigen scalar, so
//      Eigen::Matrix< NE<...>, D, 1 > works.
//
//   2. Per-axis differential helpers in namespace tax::named:
//        gradient<"axis">(f)  — gradient restricted to one named block.
//        hessian <"axis">(f)  — Hessian restricted to one named block.
//        jacobian<"axis">(F)  — Jacobian of a vector of named expansions
//                               with respect to one named block.
//
// These mirror tax::la::{gradient,hessian,jacobian} but slice the result
// down to the variables of a single named axis.

#pragma once

#include <Eigen/Core>
#include <tax/core/multi_index.hpp>
#include <tax/core/named.hpp>

// -----------------------------------------------------------------------------
// NumTraits specialization — namespace Eigen
// -----------------------------------------------------------------------------

namespace Eigen
{

template < typename T, int N, typename... Axes >
struct NumTraits< tax::named::NamedTaylorExpansion< T, N, Axes... > > : NumTraits< T >
{
    using Self = tax::named::NamedTaylorExpansion< T, N, Axes... >;
    using Real = Self;
    using NonInteger = Self;
    using Nested = Self;

    static constexpr int kNc = int( tax::numMonomials( N, Self::vars_v ) );
    enum
    {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = kNc,
        AddCost = kNc,
        // kNc * kNc overflows int for kNc > ~46340; clamp to HugeCost.
        MulCost = kNc < 46341 ? kNc * kNc : HugeCost
    };

    // Base NumTraits<T> returns these as scalar T, but Real is Self; re-expose
    // them as constant named expansions (see tax/la/num_traits.hpp).
    static inline Self epsilon() { return Self( NumTraits< T >::epsilon() ); }
    static inline Self dummy_precision() { return Self( NumTraits< T >::dummy_precision() ); }
    static inline Self highest() { return Self( NumTraits< T >::highest() ); }
    static inline Self lowest() { return Self( NumTraits< T >::lowest() ); }
    static inline Self infinity() { return Self( NumTraits< T >::infinity() ); }
    static inline Self quiet_NaN() { return Self( NumTraits< T >::quiet_NaN() ); }
};

}  // namespace Eigen

namespace tax::named
{

// -----------------------------------------------------------------------------
// variables — Eigen-vector overload of the single-axis coordinate factory
// -----------------------------------------------------------------------------

/// Build the coordinate variables of a single named axis `Name` from an Eigen vector expansion
/// point.
template < FixedString Name, int N, typename Derived >
[[nodiscard]] auto variables( const Eigen::MatrixBase< Derived >& x0 )
{
    constexpr int D = Derived::SizeAtCompileTime;
    static_assert( D != Eigen::Dynamic,
                   "variables(Eigen): expansion point must have a compile-time size" );
    using T = typename Derived::Scalar;
    using E = NamedTaylorExpansion< T, N, Axis< Name, D > >;

    typename E::Input p{};
    for ( int i = 0; i < D; ++i ) p[std::size_t( i )] = T( x0( i ) );

    Eigen::Matrix< E, D, 1 > out;
    [&]< int... I >( std::integer_sequence< int, I... > ) {
        ( ( out( I ) = E::template variable< I >( p ) ), ... );
    }( std::make_integer_sequence< int, D >{} );
    return out;
}

// -----------------------------------------------------------------------------
// Per-axis differential helpers
// -----------------------------------------------------------------------------

namespace detail
{

/// Dimension of axis `Name` within an expansion type `E`, or -1. Shared by the
/// NamedTaylorExpansion helpers below and the MixedTaylorExpansion helpers in
/// la/mixed_named.hpp.
template < typename E, FixedString Name >
inline constexpr int axisDim = DimOfName< typename E::axis_list, Name >::value;

/// Variable offset of axis `Name` within an expansion type `E` (the global
/// index of its first coordinate), or -1 when the name is absent — the
/// callers' friendly "axis name not present" static_assert fires first, so
/// this must not instantiate axisVar for an absent name.
template < typename E, FixedString Name >
inline constexpr int axisOffset = []() constexpr -> int {
    if constexpr ( axisDim< E, Name > < 1 )
        return -1;
    else
        return E::template axisVar< Name, 0 >();
}();

// Shared bodies for the per-axis differential helpers: k!-scaled derivatives
// of the wrapped expansion along the `Dim` consecutive variables starting at
// `Off`. Used by both the NamedTaylorExpansion overloads below and the
// MixedTaylorExpansion overloads in la/mixed_named.hpp.

template < int Dim, int Off, typename E >
[[nodiscard]] auto axisGradient( const E& f ) noexcept
{
    using T = typename E::scalar_type;
    Eigen::Matrix< T, Dim, 1 > g;
    MultiIndex< E::vars_v > alpha{};
    for ( int i = 0; i < Dim; ++i )
    {
        alpha[std::size_t( Off + i )] = 1;
        g( i ) = f.inner().derivative( alpha );
        alpha[std::size_t( Off + i )] = 0;
    }
    return g;
}

template < int Dim, int Off, typename E >
[[nodiscard]] auto axisHessian( const E& f ) noexcept
{
    using T = typename E::scalar_type;
    Eigen::Matrix< T, Dim, Dim > H;
    for ( int i = 0; i < Dim; ++i )
    {
        for ( int j = 0; j < Dim; ++j )
        {
            MultiIndex< E::vars_v > alpha{};
            alpha[std::size_t( Off + i )] += 1;
            alpha[std::size_t( Off + j )] += 1;
            H( i, j ) = f.inner().derivative( alpha );
        }
    }
    return H;
}

template < int Dim, int Off, typename Derived >
[[nodiscard]] auto axisJacobian( const Eigen::MatrixBase< Derived >& F )
{
    using E = typename Derived::Scalar;
    using T = typename E::scalar_type;
    constexpr int K = Derived::SizeAtCompileTime;

    Eigen::Matrix< T, K, Dim > out( F.size(), Dim );
    for ( Eigen::Index r = 0; r < F.size(); ++r )
    {
        MultiIndex< E::vars_v > alpha{};
        for ( int j = 0; j < Dim; ++j )
        {
            alpha[std::size_t( Off + j )] = 1;
            out( r, j ) = F.derived().coeff( r ).inner().derivative( alpha );
            alpha[std::size_t( Off + j )] = 0;
        }
    }
    return out;
}

}  // namespace detail

/// Gradient of a named scalar expansion with respect to one named axis.
template < FixedString Name, typename T, int N, typename... Axes >
[[nodiscard]] auto gradient( const NamedTaylorExpansion< T, N, Axes... >& f ) noexcept
{
    using E = NamedTaylorExpansion< T, N, Axes... >;
    static_assert( detail::axisDim< E, Name > >= 1,
                   "gradient<Name>(): axis name not present in this expansion" );
    return detail::axisGradient< detail::axisDim< E, Name >, detail::axisOffset< E, Name > >( f );
}

/// Hessian of a named scalar expansion restricted to one named axis.
template < FixedString Name, typename T, int N, typename... Axes >
[[nodiscard]] auto hessian( const NamedTaylorExpansion< T, N, Axes... >& f ) noexcept
{
    using E = NamedTaylorExpansion< T, N, Axes... >;
    static_assert( detail::axisDim< E, Name > >= 1,
                   "hessian<Name>(): axis name not present in this expansion" );
    return detail::axisHessian< detail::axisDim< E, Name >, detail::axisOffset< E, Name > >( f );
}

/// Jacobian of a vector of named expansions w.r.t. one named axis.
template < FixedString Name, typename Derived >
[[nodiscard]] auto jacobian( const Eigen::MatrixBase< Derived >& F )
{
    using E = typename Derived::Scalar;
    static_assert( detail::axisDim< E, Name > >= 1,
                   "jacobian<Name>(): axis name not present in the expansion" );
    return detail::axisJacobian< detail::axisDim< E, Name >, detail::axisOffset< E, Name > >( F );
}

// -----------------------------------------------------------------------------
// value / eval — mirror tax::la for named states
// -----------------------------------------------------------------------------

namespace detail
{

template < typename >
struct is_named : std::false_type
{
};
template < typename T, int N, typename... Axes >
struct is_named< NamedTaylorExpansion< T, N, Axes... > > : std::true_type
{
};
template < typename T >
inline constexpr bool is_named_v = is_named< T >::value;

}  // namespace detail

/// Constant term of a single named expansion.
template < typename T, int N, typename... Axes >
[[nodiscard]] T value( const NamedTaylorExpansion< T, N, Axes... >& f ) noexcept
{
    return f.value();
}

/// Constant terms of an Eigen matrix/vector of named expansions.
template < typename Derived >
    requires( detail::is_named_v< typename Derived::Scalar > )
[[nodiscard]] auto value( const Eigen::MatrixBase< Derived >& F )
{
    using E = typename Derived::Scalar;
    using T = typename E::scalar_type;
    Eigen::Matrix< T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime > out( F.rows(),
                                                                                    F.cols() );
    for ( Eigen::Index i = 0; i < F.size(); ++i ) out( i ) = F.derived().coeff( i ).value();
    return out;
}

/// Evaluate a single named expansion at a joint displacement `dx`.
template < typename T, int N, typename... Axes, typename DxDerived >
[[nodiscard]] T eval( const NamedTaylorExpansion< T, N, Axes... >& f,
                      const Eigen::MatrixBase< DxDerived >& dx )
{
    return f.inner().eval( dx );
}

/// Evaluate each element of an Eigen matrix/vector of named expansions at
///        a shared joint displacement `dx` (size == joint variable count).
template < typename Derived, typename DxDerived >
    requires( detail::is_named_v< typename Derived::Scalar > )
[[nodiscard]] auto eval( const Eigen::MatrixBase< Derived >& F,
                         const Eigen::MatrixBase< DxDerived >& dx )
{
    using E = typename Derived::Scalar;
    using T = typename E::scalar_type;
    constexpr int V = E::vars_v;
    static_assert(
        DxDerived::SizeAtCompileTime == V || DxDerived::SizeAtCompileTime == Eigen::Dynamic,
        "eval(named): dx size must match the joint variable count" );
    typename E::Input p{};
    for ( int i = 0; i < V; ++i ) p[std::size_t( i )] = T( dx( i ) );
    Eigen::Matrix< T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime > out( F.rows(),
                                                                                    F.cols() );
    for ( Eigen::Index i = 0; i < F.size(); ++i )
        out( i ) = F.derived().coeff( i ).inner().eval( p );
    return out;
}

}  // namespace tax::named

// The per-axis differential helpers (and the Eigen variables overload) are
// reachable directly under `tax` too. The second `using named::variables`
// folds the Eigen overload into the tax-level overload set introduced by
// <tax/core/named.hpp> (using-declarations do not pick up later additions).
namespace tax
{
using named::eval;
using named::gradient;
using named::hessian;
using named::jacobian;
using named::value;
using named::variables;
}  // namespace tax
