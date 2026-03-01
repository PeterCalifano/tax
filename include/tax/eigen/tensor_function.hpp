#pragma once

#include <tax/da.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

namespace tax
{

namespace detail::tensor_function
{

template < typename DA >
struct da_traits;

template < typename T, int N, int M >
struct da_traits< TDA< T, N, M > >
{
    using scalar_type = T;
    static constexpr int order = N;
    static constexpr int vars = M;
};

template < typename DA >
inline constexpr bool is_da_v = false;

template < typename T, int N, int M >
inline constexpr bool is_da_v< TDA< T, N, M > > = true;

template < typename Derived, typename Scalar >
using rebind_matrix_t = Eigen::Matrix< Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime,
                                       Derived::Options, Derived::MaxRowsAtCompileTime,
                                       Derived::MaxColsAtCompileTime >;

}  // namespace detail::tensor_function

/**
 * @brief Extract the scalar value (constant term) from each DA matrix/vector element.
 * @returns Eigen matrix/vector with same shape and scalar type `T`.
 */
template < typename Derived >
[[nodiscard]] auto value( const Eigen::DenseBase< Derived >& t )
    requires( detail::tensor_function::is_da_v< typename Derived::Scalar > )
{
    using DA = typename Derived::Scalar;
    using T = typename detail::tensor_function::da_traits< DA >::scalar_type;
    using Out = detail::tensor_function::rebind_matrix_t< Derived, T >;
    Out out( t.rows(), t.cols() );
    for ( Eigen::Index i = 0; i < t.size(); ++i ) out.coeffRef( i ) = t.derived().coeff( i ).value();
    return out;
}

/**
 * @brief Extract a partial derivative from each DA matrix/vector element (runtime multi-index).
 * @param alpha Multi-index specifying the derivative order per variable.
 * @returns Eigen matrix/vector with same shape and scalar type `T`.
 */
template < typename Derived, std::size_t M >
[[nodiscard]] auto derivative( const Eigen::DenseBase< Derived >& t,
                               const std::array< int, M >& alpha )
    requires( detail::tensor_function::is_da_v< typename Derived::Scalar > )
{
    using DA = typename Derived::Scalar;
    using T = typename detail::tensor_function::da_traits< DA >::scalar_type;
    static_assert( M == std::size_t( detail::tensor_function::da_traits< DA >::vars ),
                   "Derivative multi-index arity must match number of variables" );
    using Out = detail::tensor_function::rebind_matrix_t< Derived, T >;
    Out out( t.rows(), t.cols() );
    for ( Eigen::Index i = 0; i < t.size(); ++i )
        out.coeffRef( i ) = t.derived().coeff( i ).derivative( alpha );
    return out;
}

/**
 * @brief Extract the k-th time derivative from each univariate DA matrix/vector element.
 * @param k Derivative order (0 = value, 1 = first derivative, ...).
 * @returns Eigen matrix/vector with same shape and scalar type `T`.
 */
template < typename Derived >
[[nodiscard]] auto derivative( const Eigen::DenseBase< Derived >& t, int k )
    requires( detail::tensor_function::is_da_v< typename Derived::Scalar > &&
              detail::tensor_function::da_traits< typename Derived::Scalar >::vars == 1 )
{
    return derivative( t, MultiIndex< 1 >{ k } );
}

/**
 * @brief Extract a partial derivative from each DA matrix/vector element (compile-time multi-index).
 * @tparam Alpha Derivative orders for each variable.
 * @returns Eigen matrix/vector with same shape and scalar type `T`.
 */
template < int... Alpha, typename Derived >
[[nodiscard]] auto derivative( const Eigen::DenseBase< Derived >& t )
    requires( detail::tensor_function::is_da_v< typename Derived::Scalar > )
{
    using DA = typename Derived::Scalar;
    using T = typename detail::tensor_function::da_traits< DA >::scalar_type;
    constexpr int N = detail::tensor_function::da_traits< DA >::order;
    constexpr int M = detail::tensor_function::da_traits< DA >::vars;
    static_assert( sizeof...( Alpha ) == std::size_t( M ),
                   "Derivative multi-index arity must match number of variables" );
    static_assert( ( ( Alpha >= 0 ) && ... ), "Derivative orders must be non-negative" );
    constexpr int total_order = ( Alpha + ... + 0 );
    static_assert( total_order <= N, "Derivative total order exceeds DA truncation order" );

    using Out = detail::tensor_function::rebind_matrix_t< Derived, T >;
    Out out( t.rows(), t.cols() );
    for ( Eigen::Index i = 0; i < t.size(); ++i )
        out.coeffRef( i ) = t.derived().coeff( i ).template derivative< Alpha... >();
    return out;
}

/**
 * @brief Extract the scalar value (constant term) from each DA element.
 * @returns Eigen::Tensor<T, Rank> with the same dimensions.
 */
template < typename T, int N, int M, int Rank >
[[nodiscard]] auto value( const Eigen::Tensor< TDA< T, N, M >, Rank >& t )
    requires( Rank > 2 )
{
    Eigen::Tensor< T, Rank > out( t.dimensions() );
    for ( Eigen::Index i = 0; i < t.size(); ++i ) out.data()[i] = t.data()[i].value();
    return out;
}

/**
 * @brief Extract a partial derivative from each DA element (runtime multi-index).
 * @param alpha Multi-index specifying the derivative order per variable.
 * @returns Eigen::Tensor<T, Rank> with the same dimensions.
 */
template < typename T, int N, int M, int Rank >
[[nodiscard]] auto derivative( const Eigen::Tensor< TDA< T, N, M >, Rank >& t,
                               const std::array< int, std::size_t( M ) >& alpha )
    requires( Rank > 2 )
{
    Eigen::Tensor< T, Rank > out( t.dimensions() );
    for ( Eigen::Index i = 0; i < t.size(); ++i ) out.data()[i] = t.data()[i].derivative( alpha );
    return out;
}

/**
 * @brief Extract the k-th time derivative from each univariate DA element.
 * @param k Derivative order (0 = value, 1 = first derivative, ...).
 * @returns Eigen::Tensor<T, Rank> with the same dimensions.
 */
template < typename T, int N, int Rank >
[[nodiscard]] auto derivative( const Eigen::Tensor< TDA< T, N, 1 >, Rank >& t, int k )
    requires( Rank > 2 )
{
    return derivative( t, MultiIndex< 1 >{ k } );
}

/**
 * @brief Extract a partial derivative from each DA element (compile-time multi-index).
 * @tparam Alpha Derivative orders for each variable.
 * @returns Eigen::Tensor<T, Rank> with the same dimensions.
 */
template < int... Alpha, typename T, int N, int M, int Rank >
[[nodiscard]] auto derivative( const Eigen::Tensor< TDA< T, N, M >, Rank >& t )
    requires( Rank > 2 )
{
    static_assert( sizeof...( Alpha ) == M,
                   "Derivative multi-index arity must match number of variables" );
    static_assert( ( ( Alpha >= 0 ) && ... ), "Derivative orders must be non-negative" );
    constexpr int total_order = ( Alpha + ... + 0 );
    static_assert( total_order <= N, "Derivative total order exceeds DA truncation order" );

    Eigen::Tensor< T, Rank > out( t.dimensions() );
    for ( Eigen::Index i = 0; i < t.size(); ++i )
        out.data()[i] = t.data()[i].template derivative< Alpha... >();
    return out;
}

}  // namespace tax
