// include/tax/la/named.hpp
//
// Eigen integration for the named-axis layer (tax::named):
//
//   1. Eigen::NumTraits< tax::named::NamedTaylorExpansion<...> > — lets a named
//      expansion act as a first-class Eigen scalar, so
//      Eigen::Matrix< NE<...>, D, 1 > works (and, with the ODE
//      VectorOps specialization, can be integrated as an ODE state).
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
        MulCost = kNc * kNc
    };
};

}  // namespace Eigen

namespace tax::named
{

// -----------------------------------------------------------------------------
// Per-axis differential helpers
// -----------------------------------------------------------------------------

namespace detail
{

/// @brief Variable offset of axis `Name` within an expansion type `E`, or -1.
template < typename E, FixedString Name >
inline constexpr int axisOffset =
    OffsetOf< typename E::axis_list,
              Axis< Name, DimOfName< typename E::axis_list, Name >::value > >::value;

/// @brief Dimension of axis `Name` within an expansion type `E`, or -1.
template < typename E, FixedString Name >
inline constexpr int axisDim = DimOfName< typename E::axis_list, Name >::value;

}  // namespace detail

/**
 * @brief Gradient of a named scalar expansion with respect to one named axis.
 *
 * Returns an `Eigen::Matrix<T, dim, 1>` whose i-th entry is the first-order
 * partial derivative with respect to the i-th coordinate of axis `Name`,
 * evaluated at the expansion point.
 */
template < FixedString Name, typename T, int N, typename... Axes >
[[nodiscard]] auto gradient( const NamedTaylorExpansion< T, N, Axes... >& f ) noexcept
{
    using E = NamedTaylorExpansion< T, N, Axes... >;
    constexpr int dim = detail::axisDim< E, Name >;
    static_assert( dim >= 1, "gradient<Name>(): axis name not present in this expansion" );
    constexpr int off = detail::axisOffset< E, Name >;

    Eigen::Matrix< T, dim, 1 > g;
    MultiIndex< E::vars_v > alpha{};
    for ( int i = 0; i < dim; ++i )
    {
        alpha[std::size_t( off + i )] = 1;
        g( i ) = f.inner().derivative( alpha );
        alpha[std::size_t( off + i )] = 0;
    }
    return g;
}

/**
 * @brief Hessian of a named scalar expansion restricted to one named axis.
 *
 * Returns an `Eigen::Matrix<T, dim, dim>` of second-order mixed partials
 * `d^2 f / (dx_i dx_j)` over the coordinates of axis `Name`.
 */
template < FixedString Name, typename T, int N, typename... Axes >
[[nodiscard]] auto hessian( const NamedTaylorExpansion< T, N, Axes... >& f ) noexcept
{
    using E = NamedTaylorExpansion< T, N, Axes... >;
    constexpr int dim = detail::axisDim< E, Name >;
    static_assert( dim >= 1, "hessian<Name>(): axis name not present in this expansion" );
    constexpr int off = detail::axisOffset< E, Name >;

    Eigen::Matrix< T, dim, dim > H;
    for ( int i = 0; i < dim; ++i )
    {
        for ( int j = 0; j < dim; ++j )
        {
            MultiIndex< E::vars_v > alpha{};
            alpha[std::size_t( off + i )] += 1;
            alpha[std::size_t( off + j )] += 1;
            H( i, j ) = f.inner().derivative( alpha );
        }
    }
    return H;
}

/**
 * @brief Jacobian of a vector of named expansions w.r.t. one named axis.
 *
 * @param F  Eigen column vector of `NamedTaylorExpansion<T, N, Axes...>` with `K` rows.
 * @return   `Eigen::Matrix<T, K, dim>` with `J(i, j) = dF_i / dx_j` over the
 *           coordinates of axis `Name`.
 */
template < FixedString Name, typename Derived >
[[nodiscard]] auto jacobian( const Eigen::MatrixBase< Derived >& F )
{
    using E = typename Derived::Scalar;
    using T = typename E::scalar_type;
    constexpr int dim = detail::axisDim< E, Name >;
    static_assert( dim >= 1, "jacobian<Name>(): axis name not present in the expansion" );
    constexpr int off = detail::axisOffset< E, Name >;
    constexpr int K = Derived::SizeAtCompileTime;

    Eigen::Matrix< T, K, dim > out( F.size(), dim );
    for ( Eigen::Index r = 0; r < F.size(); ++r )
    {
        MultiIndex< E::vars_v > alpha{};
        for ( int j = 0; j < dim; ++j )
        {
            alpha[std::size_t( off + j )] = 1;
            out( r, j ) = F.derived().coeff( r ).inner().derivative( alpha );
            alpha[std::size_t( off + j )] = 0;
        }
    }
    return out;
}

}  // namespace tax::named

// The per-axis differential helpers are reachable directly under `tax` too.
namespace tax
{
using named::gradient;
using named::hessian;
using named::jacobian;
}  // namespace tax
