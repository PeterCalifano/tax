// include/tax/la/values.hpp
//
// Builders and evaluators that move between scalar / Eigen-matrix
// forms of TaylorExpansion objects:
//
//   variables(x0)   — Eigen vector of coordinate TE variables.
//   value(F)        — extract the constant term from each element.
//   eval(f|F, dx)   — evaluate at displacement `dx` (Eigen vector
//                     for multivariate, scalar for univariate).

#pragma once

#include <Eigen/Core>
#include <utility>

#include <tax/la/num_traits.hpp>

namespace tax::la
{

// -----------------------------------------------------------------------------
// variables — build an Eigen column vector of TE coordinate variables
// -----------------------------------------------------------------------------

/**
 * @brief Construct an `Eigen::Matrix<TE, M, 1>` of coordinate Taylor variables from an
 *        Eigen vector expansion point `x0`.
 *
 * @tparam TE  A `TaylorExpansion<T, N, M, S>` type.
 * @param  x0  Eigen vector with compile-time size `M` holding the expansion point.
 * @return Eigen column vector of M TE variables.
 */
template < typename TE, typename Derived >
[[nodiscard]] auto variables( const Eigen::MatrixBase< Derived >& x0 )
{
    using tr = detail::te_traits< TE >;
    using T  = typename tr::scalar_type;
    constexpr int M = tr::vars_v;
    static_assert( Derived::SizeAtCompileTime == M || Derived::SizeAtCompileTime == Eigen::Dynamic,
                   "variables(): Eigen input size must match TE::vars_v" );
    typename TE::Input p{};
    for ( int i = 0; i < M; ++i ) p[std::size_t( i )] = T( x0( i ) );
    Eigen::Matrix< TE, M, 1 > out;
    [&]< std::size_t... I >( std::index_sequence< I... > )
    {
        ( ( out( int( I ) ) = TE::template variable< int( I ) >( p ) ), ... );
    }( std::make_index_sequence< std::size_t( M ) >{} );
    return out;
}

// -----------------------------------------------------------------------------
// value — extract constant terms from an Eigen matrix of TE objects
// -----------------------------------------------------------------------------

/**
 * @brief Extract the constant term (value at expansion point) from each element of an
 *        Eigen matrix/vector of `TaylorExpansion` objects.
 *
 * @param F  Eigen matrix/vector whose `Scalar` type is a `TaylorExpansion`.
 * @return   Eigen matrix/vector of the underlying scalar type, same shape as `F`.
 */
template < typename Derived >
    requires( detail::is_te_v< typename Derived::Scalar > )
[[nodiscard]] auto value( const Eigen::MatrixBase< Derived >& F )
{
    using TE = typename Derived::Scalar;
    using T  = typename detail::te_traits< TE >::scalar_type;
    Eigen::Matrix< T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime >
        out( F.rows(), F.cols() );
    for ( Eigen::Index i = 0; i < F.size(); ++i ) out( i ) = F.derived().coeff( i ).value();
    return out;
}

// -----------------------------------------------------------------------------
// eval — evaluate scalar TE or Eigen matrix of TEs at a displacement
// -----------------------------------------------------------------------------

/**
 * @brief Evaluate a scalar `TaylorExpansion` at displacement `dx`.
 *
 * Wraps `TE::eval(Eigen)`.
 */
template < typename T, int N, int M, typename S, typename DxDerived >
[[nodiscard]] T eval( const TaylorExpansion< T, N, M, S >& f,
                      const Eigen::MatrixBase< DxDerived >& dx ) noexcept
{
    return f.eval( dx );
}

/**
 * @brief Evaluate each element of an Eigen matrix/vector of `TaylorExpansion` objects at
 *        a shared displacement vector `dx`.
 *
 * @param F   Eigen matrix/vector of `TaylorExpansion` objects.
 * @param dx  Displacement Eigen vector with `M` entries (compile-time size).
 * @return    Eigen matrix/vector of the same shape, scalar type `T`.
 */
template < typename Derived, typename DxDerived >
    requires( detail::is_te_v< typename Derived::Scalar > )
[[nodiscard]] auto eval( const Eigen::MatrixBase< Derived >& F,
                         const Eigen::MatrixBase< DxDerived >& dx )
{
    using TE = typename Derived::Scalar;
    using T  = typename detail::te_traits< TE >::scalar_type;
    constexpr int M = detail::te_traits< TE >::vars_v;
    static_assert( DxDerived::SizeAtCompileTime == M || DxDerived::SizeAtCompileTime == Eigen::Dynamic,
                   "eval(): dx size must match TE::vars_v" );
    typename TE::Input p{};
    for ( int i = 0; i < M; ++i ) p[std::size_t( i )] = T( dx( i ) );
    Eigen::Matrix< T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime >
        out( F.rows(), F.cols() );
    for ( Eigen::Index i = 0; i < F.size(); ++i ) out( i ) = F.derived().coeff( i ).eval( p );
    return out;
}

/**
 * @brief Evaluate a scalar univariate `TaylorExpansion` at a scalar displacement.
 *
 * Convenience for the `M == 1` case: spares the user from wrapping `dx` in an
 * `Input` array or a 1-vector. Equivalent to `f.eval(Input{dx})`.
 */
template < typename T, int N, typename S >
[[nodiscard]] T eval( const TaylorExpansion< T, N, 1, S >& f, T dx ) noexcept
{
    typename TaylorExpansion< T, N, 1, S >::Input p{ dx };
    return f.eval( p );
}

/**
 * @brief Evaluate each element of an Eigen matrix/vector of univariate
 *        `TaylorExpansion` objects at a shared scalar displacement.
 *
 * Convenience for the `M == 1` case: same as
 * `eval(F, Eigen::Matrix<T,1,1>{dx})` but without the wrapping.
 *
 * @param F   Eigen matrix/vector of univariate `TaylorExpansion` objects.
 * @param dx  Scalar displacement.
 * @return    Eigen matrix/vector of the same shape, scalar type `T`.
 */
template < typename Derived >
    requires( detail::is_te_v< typename Derived::Scalar >
              && detail::te_traits< typename Derived::Scalar >::vars_v == 1 )
[[nodiscard]] auto eval(
    const Eigen::MatrixBase< Derived >&                                 F,
    typename detail::te_traits< typename Derived::Scalar >::scalar_type dx )
{
    using TE = typename Derived::Scalar;
    using T  = typename detail::te_traits< TE >::scalar_type;
    typename TE::Input p{ dx };
    Eigen::Matrix< T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime >
        out( F.rows(), F.cols() );
    for ( Eigen::Index i = 0; i < F.size(); ++i )
        out( i ) = F.derived().coeff( i ).eval( p );
    return out;
}

}  // namespace tax::la
