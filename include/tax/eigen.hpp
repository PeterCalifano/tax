#pragma once

// Eigen integration for tax::TaylorExpansion.
// Provides: NumTraits, variables, value, eval, derivative, gradient, hessian, jacobian, invert.

#include <Eigen/LU>
#include <tax/core/enumeration.hpp>
#include <tax/core/taylor_expansion.hpp>

// =============================================================================
// 1. NumTraits specialization — makes TaylorExpansion a first-class Eigen scalar
// =============================================================================

namespace Eigen
{

template < typename T, int N, int M, typename Storage >
struct NumTraits< tax::TaylorExpansion< T, N, M, Storage > > : NumTraits< T >
{
    using Self       = tax::TaylorExpansion< T, N, M, Storage >;
    using Real       = Self;
    using NonInteger = Self;
    using Nested     = Self;
    enum
    {
        IsComplex              = 0,
        IsInteger              = 0,
        IsSigned               = 1,
        RequireInitialization  = 1,
        ReadCost               = int( tax::numMonomials( N, M ) ),
        AddCost                = int( tax::numMonomials( N, M ) ),
        MulCost                = int( tax::numMonomials( N, M ) ) * int( tax::numMonomials( N, M ) )
    };
};

}  // namespace Eigen

// =============================================================================
// 2. Internal traits helpers
// =============================================================================

namespace tax::detail::eigen
{

template < typename >
struct te_traits;

template < typename T, int N, int M, typename S >
struct te_traits< TaylorExpansion< T, N, M, S > >
{
    using scalar_type = T;
    static constexpr int order_v = N;
    static constexpr int vars_v  = M;
    using storage_t              = S;
};

template < typename T >
struct is_te : std::false_type
{
};

template < typename T, int N, int M, typename S >
struct is_te< TaylorExpansion< T, N, M, S > > : std::true_type
{
};

template < typename T >
inline constexpr bool is_te_v = is_te< T >::value;

/// @brief Rebind the scalar type of an Eigen matrix expression.
template < typename Derived, typename Scalar >
using rebind_matrix_t =
    Eigen::Matrix< Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime,
                   Derived::Options, Derived::MaxRowsAtCompileTime,
                   Derived::MaxColsAtCompileTime >;

}  // namespace tax::detail::eigen

// =============================================================================
// 3. Free functions in namespace tax
// =============================================================================

namespace tax
{

// ---------------------------------------------------------------------------
// variables — build an Eigen column vector of TE coordinate variables
// ---------------------------------------------------------------------------

/**
 * @brief Construct an `Eigen::Matrix<TE, M, 1>` of coordinate Taylor variables from an
 *        Eigen vector expansion point `x0`.
 *
 * @tparam TE    A `TaylorExpansion<T, N, M, S>` type.
 * @param  x0   Eigen vector with compile-time size `M` holding the expansion point.
 * @return Eigen column vector of M TE variables.
 */
template < typename TE, typename Derived >
[[nodiscard]] auto variables( const Eigen::MatrixBase< Derived >& x0 )
{
    using tr = detail::eigen::te_traits< TE >;
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

// ---------------------------------------------------------------------------
// value — extract constant terms from an Eigen matrix of TE objects
// ---------------------------------------------------------------------------

/**
 * @brief Extract the constant term (value at expansion point) from each element of an
 *        Eigen matrix/vector of `TaylorExpansion` objects.
 *
 * @param F  Eigen matrix/vector whose `Scalar` type is a `TaylorExpansion`.
 * @return   Eigen matrix/vector of the underlying scalar type, same shape as `F`.
 */
template < typename Derived >
    requires( detail::eigen::is_te_v< typename Derived::Scalar > )
[[nodiscard]] auto value( const Eigen::MatrixBase< Derived >& F )
{
    using TE = typename Derived::Scalar;
    using T  = typename detail::eigen::te_traits< TE >::scalar_type;
    Eigen::Matrix< T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime >
        out( F.rows(), F.cols() );
    for ( Eigen::Index i = 0; i < F.size(); ++i ) out( i ) = F.derived().coeff( i ).value();
    return out;
}

// ---------------------------------------------------------------------------
// eval — evaluate scalar TE or Eigen matrix of TEs at a displacement
// ---------------------------------------------------------------------------

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
    requires( detail::eigen::is_te_v< typename Derived::Scalar > )
[[nodiscard]] auto eval( const Eigen::MatrixBase< Derived >& F,
                         const Eigen::MatrixBase< DxDerived >& dx )
{
    using TE = typename Derived::Scalar;
    using T  = typename detail::eigen::te_traits< TE >::scalar_type;
    constexpr int M = detail::eigen::te_traits< TE >::vars_v;
    static_assert( DxDerived::SizeAtCompileTime == M || DxDerived::SizeAtCompileTime == Eigen::Dynamic,
                   "eval(): dx size must match TE::vars_v" );
    typename TE::Input p{};
    for ( int i = 0; i < M; ++i ) p[std::size_t( i )] = T( dx( i ) );
    Eigen::Matrix< T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime >
        out( F.rows(), F.cols() );
    for ( Eigen::Index i = 0; i < F.size(); ++i ) out( i ) = F.derived().coeff( i ).eval( p );
    return out;
}

// ---------------------------------------------------------------------------
// derivative — element-wise partial derivative extraction
// ---------------------------------------------------------------------------

/**
 * @brief Extract a compile-time partial derivative from each element of an Eigen
 *        matrix/vector of `TaylorExpansion` objects.
 *
 * Usage: `tax::derivative<1, 0>(F)` extracts `dF/dx_0` from a 2-variable TE matrix.
 *
 * @tparam Alpha  Derivative orders (one per variable, must sum to <= N).
 * @param  F      Eigen matrix/vector of `TaylorExpansion` objects.
 * @return        Eigen matrix/vector of the underlying scalar type.
 */
template < int... Alpha, typename Derived >
    requires( detail::eigen::is_te_v< typename Derived::Scalar > )
[[nodiscard]] auto derivative( const Eigen::MatrixBase< Derived >& F )
{
    using TE = typename Derived::Scalar;
    using T  = typename detail::eigen::te_traits< TE >::scalar_type;
    Eigen::Matrix< T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime >
        out( F.rows(), F.cols() );
    for ( Eigen::Index i = 0; i < F.size(); ++i )
        out( i ) = F.derived().coeff( i ).template derivative< Alpha... >();
    return out;
}

// =============================================================================
// 4. gradient, hessian, jacobian, invert
// =============================================================================

// ---------------------------------------------------------------------------
// gradient — free-function wrapper around TaylorExpansion::gradient()
// ---------------------------------------------------------------------------

/**
 * @brief Compute the gradient of a scalar `TaylorExpansion` at its expansion point.
 * @return `Eigen::Matrix<T, M, 1>` of first-order partial derivatives.
 */
template < typename T, int N, int M, typename S >
[[nodiscard]] Eigen::Matrix< T, M, 1 > gradient( const TaylorExpansion< T, N, M, S >& f ) noexcept
{
    return f.gradient();
}

// ---------------------------------------------------------------------------
// hessian — free-function wrapper around TaylorExpansion::hessian()
// ---------------------------------------------------------------------------

/**
 * @brief Compute the Hessian matrix of a scalar `TaylorExpansion` at its expansion point.
 * @return `Eigen::Matrix<T, M, M>` of second-order mixed partial derivatives.
 */
template < typename T, int N, int M, typename S >
[[nodiscard]] Eigen::Matrix< T, M, M > hessian( const TaylorExpansion< T, N, M, S >& f ) noexcept
{
    return f.hessian();
}

// ---------------------------------------------------------------------------
// jacobian — Jacobian of a vector-valued TE function
// ---------------------------------------------------------------------------

/**
 * @brief Compute the Jacobian matrix of a vector-valued `TaylorExpansion` function.
 *
 * @param F  Eigen matrix/vector of `TaylorExpansion` objects with `K` components.
 * @return   Eigen matrix of shape `(K, M)` where `J(i, j) = dF_i / dx_j`.
 */
template < typename Derived >
    requires( detail::eigen::is_te_v< typename Derived::Scalar > )
[[nodiscard]] auto jacobian( const Eigen::MatrixBase< Derived >& F )
{
    using TE = typename Derived::Scalar;
    using tr = detail::eigen::te_traits< TE >;
    using T  = typename tr::scalar_type;
    constexpr int M = tr::vars_v;
    constexpr int K = Derived::SizeAtCompileTime;
    Eigen::Matrix< T, K, M > out( F.size(), M );
    for ( Eigen::Index r = 0; r < F.size(); ++r )
    {
        MultiIndex< M > alpha{};
        for ( int j = 0; j < M; ++j )
        {
            alpha[std::size_t( j )] = 1;
            out( r, j ) = F.derived().coeff( r ).derivative( alpha );
            alpha[std::size_t( j )] = 0;
        }
    }
    return out;
}

// =============================================================================
// 5. invert — formal inversion of a polynomial map (Newton-style Picard iteration)
// =============================================================================

namespace detail::eigen
{

/// @brief Build the identity map as `Eigen::Matrix<TE, M, 1>` around the zero expansion point.
template < typename TE >
[[nodiscard]] auto identityMap()
{
    constexpr int M = te_traits< TE >::vars_v;
    using Map = Eigen::Matrix< TE, M, 1 >;
    Map out{};
    typename TE::Input x0{};
    [&]< std::size_t... I >( std::index_sequence< I... > )
    {
        ( ( out( Eigen::Index( I ) ) = TE::template variable< int( I ) >( x0 ) ), ... );
    }( std::make_index_sequence< std::size_t( M ) >{} );
    return out;
}

/// @brief Compose a single scalar TE `f` with a map `g` (substitution: x_j -> g_j).
template < typename TE, typename Map >
[[nodiscard]] TE composeOne( const TE& f, const Map& g )
{
    using T = typename te_traits< TE >::scalar_type;
    constexpr int N = te_traits< TE >::order_v;
    constexpr int M = te_traits< TE >::vars_v;

    TE out = TE::zero();
    for ( int d = 0; d <= N; ++d )
    {
        forEachMonomialOfDegree< M >( d, [&]( const MultiIndex< M >& alpha )
        {
            const T coeff = f.coeff( alpha );
            if ( coeff == T{} ) return;
            TE term = TE::constant( coeff );
            for ( int j = 0; j < M; ++j )
                for ( int k = 0; k < alpha[std::size_t( j )]; ++k ) term = term * g( j );
            out = out + term;
        } );
    }
    return out;
}

/// @brief Compose a vector map `a` with map `b` component-wise.
template < typename TE, typename Map >
[[nodiscard]] Map composeMap( const Map& a, const Map& b )
{
    Map out{};
    for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = composeOne< TE >( a( i ), b );
    return out;
}

/// @brief Build the linear map `J * vars` as a TE vector.
template < typename TE, typename Mat >
[[nodiscard]] Eigen::Matrix< TE, Mat::RowsAtCompileTime, 1 >
    linear( const Mat& J, const Eigen::Matrix< TE, Mat::ColsAtCompileTime, 1 >& vars )
{
    Eigen::Matrix< TE, Mat::RowsAtCompileTime, 1 > out{};
    for ( Eigen::Index i = 0; i < J.rows(); ++i )
    {
        out( i ) = TE::zero();
        for ( Eigen::Index j = 0; j < J.cols(); ++j ) out( i ) = out( i ) + J( i, j ) * vars( j );
    }
    return out;
}

}  // namespace detail::eigen

/**
 * @brief Formally invert a square polynomial map represented as an Eigen vector of
 *        `TaylorExpansion` components.
 *
 * The constant terms of the input components are ignored; the inversion operates on the
 * non-constant (perturbation) part.  The linear part must be invertible.  The formal
 * inverse is computed via Picard iterations up to order N.
 *
 * @param map_in  Eigen vector of M `TaylorExpansion` components.
 * @return        Inverse map, same Eigen shape as `map_in`.
 * @throws std::invalid_argument if the linear part is singular.
 */
template < typename Derived >
    requires( detail::eigen::is_te_v< typename Derived::Scalar > )
[[nodiscard]] auto invert( const Eigen::MatrixBase< Derived >& map_in )
{
    using TE = typename Derived::Scalar;
    using T  = typename detail::eigen::te_traits< TE >::scalar_type;
    constexpr int N = detail::eigen::te_traits< TE >::order_v;
    constexpr int M = detail::eigen::te_traits< TE >::vars_v;

    static_assert( Derived::SizeAtCompileTime == M || Derived::SizeAtCompileTime == Eigen::Dynamic,
                   "invert: map size must equal number of variables M" );

    using Map = Eigen::Matrix< TE, M, 1 >;
    using Mat = Eigen::Matrix< T, M, M >;

    // Build the non-constant map (zero out constant terms).
    Map map{};
    for ( Eigen::Index i = 0; i < map_in.size(); ++i )
    {
        map( i ) = map_in.derived().coeff( i );
        map( i )[0] = T{};  // drop constant term
    }

    // Build the identity map and Jacobian of the non-constant map.
    const Map I    = detail::eigen::identityMap< TE >();
    const Mat J    = jacobian( map );
    const Eigen::FullPivLU< Mat > lu( J );
    if ( !lu.isInvertible() )
        throw std::invalid_argument( "invert failed: linear part is singular" );

    const Mat Jinv   = lu.inverse();
    const Map Mlin   = detail::eigen::linear< TE >( J, I );
    const Map Minv   = detail::eigen::linear< TE >( Jinv, I );

    // nonlinear = map - linear_part
    Map nonlinear{};
    for ( int i = 0; i < M; ++i ) nonlinear( i ) = map( i ) - Mlin( i );

    // Picard iteration: out_{k+1} = Jinv * (I - nonlinear ∘ out_k)
    Map out = Minv;
    for ( int k = 1; k < N; ++k )
    {
        const Map composed_nonlin = detail::eigen::composeMap< TE >( nonlinear, out );
        Map correction{};
        for ( int i = 0; i < M; ++i ) correction( i ) = I( i ) - composed_nonlin( i );
        out = detail::eigen::composeMap< TE >( Minv, correction );
    }

    // Copy back into an output with the same Eigen shape as map_in.
    detail::eigen::rebind_matrix_t< Derived, TE > ret( map_in.rows(), map_in.cols() );
    for ( Eigen::Index i = 0; i < map_in.size(); ++i ) ret.coeffRef( i ) = out( i );
    return ret;
}

}  // namespace tax
