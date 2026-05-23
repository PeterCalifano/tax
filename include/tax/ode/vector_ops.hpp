// include/tax/ode/vector_ops.hpp
//
// VectorOps<S>: trait that exposes the three operations every ODE
// step-driver needs to remain agnostic of the state's scalar layout —
// infinity-norm reduced to a real `double`, plus axpy and
// scale-assign that take a `double` coefficient.
//
// Default specializations cover:
//   - floating-point scalars
//   - tax::TaylorExpansion<T, N, M, storage::Dense>  (sup over coefficients)
//   - Eigen::Matrix<T, D, 1>          (recurses element-wise)
//
// To support a new state type, specialize VectorOps<MyState> with the
// three static functions; no stepper changes required.

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include <tax/core/taylor_expansion.hpp>

namespace tax::ode
{

template < class S >
struct VectorOps;  // primary: undefined

// ---- floating-point scalar ----
template < class T >
    requires std::is_floating_point_v< T >
struct VectorOps< T >
{
    [[nodiscard]] static double norm( T x ) noexcept
    {
        return std::abs( static_cast< double >( x ) );
    }

    static void axpy( T& y, double a, T x ) noexcept
    {
        y += static_cast< T >( a ) * x;
    }

    static void scale_assign( T& y, double a, T x ) noexcept
    {
        y = static_cast< T >( a ) * x;
    }
};

// ---- tax::TaylorExpansion<T, N, M, storage::Dense> ----
template < class T, int N, int M >
struct VectorOps< TaylorExpansion< T, N, M, storage::Dense > >
{
    using S = TaylorExpansion< T, N, M, storage::Dense >;

    [[nodiscard]] static double norm( const S& x ) noexcept
    {
        double n = 0.0;
        for ( std::size_t i = 0; i < S::nCoefficients; ++i )
            n = std::max( n, std::abs( static_cast< double >( x[i] ) ) );
        return n;
    }

    static void axpy( S& y, double a, const S& x )
    {
        y = y + static_cast< T >( a ) * x;
    }

    static void scale_assign( S& y, double a, const S& x )
    {
        y = static_cast< T >( a ) * x;
    }
};

// ---- Eigen column vector of anything supported above ----
template < class T, int D >
struct VectorOps< Eigen::Matrix< T, D, 1 > >
{
    using V     = Eigen::Matrix< T, D, 1 >;
    using Inner = VectorOps< T >;

    [[nodiscard]] static double norm( const V& x ) noexcept
    {
        double n = 0.0;
        for ( Eigen::Index i = 0; i < x.size(); ++i )
            n = std::max( n, Inner::norm( x( i ) ) );
        return n;
    }

    static void axpy( V& y, double a, const V& x )
    {
        for ( Eigen::Index i = 0; i < x.size(); ++i )
            Inner::axpy( y( i ), a, x( i ) );
    }

    static void scale_assign( V& y, double a, const V& x )
    {
        if ( y.size() != x.size() )
            y.resize( x.size() );
        for ( Eigen::Index i = 0; i < x.size(); ++i )
            Inner::scale_assign( y( i ), a, x( i ) );
    }
};

}  // namespace tax::ode
