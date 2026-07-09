// Vector algebra over Eigen vectors/matrices of Taylor expansions
// (tax::la, re-exported under tax::):
//
//   dot(a, b)         a · b               (vector · vector -> scalar expansion)
//   dot(A, b)         A · b               (matrix · vector -> vector; A may be a
//                                          constant real matrix — a linear map)
//   cross(a, b)       a × b               (3-vectors)
//   angle(a, b)       angle between a, b  (acos of the normalised dot)
//   unitvec(v)        v / |v|
//   unitcross(a, b)   (a × b) / |a × b|
//   projvec(a, d)     projection of a onto the direction d
//   projplane(a, n)   projection of a onto the plane with normal n
//
// Every element is a dense TE, named NE or mixed MTE expansion. The results are
// full Taylor series, so e.g. `gradient(angle(a, b))` gives the sensitivity of
// the angle to the underlying variables.
//
// Fusion notes: norms enter only as `dot(v, v)`, and where two norms multiply
// (angle, unit vectors) the reciprocal square root is taken once over the
// product — one `seriesPow` pass instead of two. `dot(A, b)` scalar-multiplies
// when A is a constant real matrix, avoiding the Cauchy product entirely.

#pragma once

#include <Eigen/Core>
#include <tax/la/norm.hpp>  // detail::ExpansionElement + the pow / acos surface
#include <tax/operators/arithmetic.hpp>
#include <tax/operators/named_arithmetic.hpp>

namespace tax::la
{

// ---------------------------------------------------------------------------
// dot
// ---------------------------------------------------------------------------

/// `a · b` (vector · vector) as a single scalar expansion.
template < typename LA, typename LB >
    requires( bool( LA::IsVectorAtCompileTime ) && bool( LB::IsVectorAtCompileTime ) &&
              detail::ExpansionElement< typename LA::Scalar > )
[[nodiscard]] auto dot( const Eigen::MatrixBase< LA >& a,
                        const Eigen::MatrixBase< LB >& b ) noexcept
{
    using E = typename LA::Scalar;
    E s{};
    for ( Eigen::Index i = 0; i < a.size(); ++i ) s += E( a( i ) ) * E( b( i ) );
    return s;
}

/// `A · b` (matrix · vector). `A` may be a matrix of expansions or a constant
/// real matrix (a linear map) — the latter scalar-multiplies each column,
/// skipping the Cauchy product.
template < typename MA, typename VB >
    requires( !bool( MA::IsVectorAtCompileTime ) && bool( VB::IsVectorAtCompileTime ) &&
              detail::ExpansionElement< typename VB::Scalar > )
[[nodiscard]] auto dot( const Eigen::MatrixBase< MA >& A,
                        const Eigen::MatrixBase< VB >& b ) noexcept
{
    using E = typename VB::Scalar;
    using AS = typename MA::Scalar;
    Eigen::Matrix< E, MA::RowsAtCompileTime, 1 > r( A.rows() );
    for ( Eigen::Index i = 0; i < A.rows(); ++i )
    {
        E s{};
        for ( Eigen::Index j = 0; j < A.cols(); ++j )
        {
            if constexpr ( detail::ExpansionElement< AS > )
                s += E( A( i, j ) ) * E( b( j ) );  // expansion · expansion (Cauchy)
            else
                s += E( b( j ) ) * AS( A( i, j ) );  // constant scalar · expansion
        }
        r( i ) = s;
    }
    return r;
}

// ---------------------------------------------------------------------------
// cross
// ---------------------------------------------------------------------------

/// `a × b` for 3-vectors.
template < typename LA, typename LB >
    requires( detail::ExpansionElement< typename LA::Scalar > )
[[nodiscard]] auto cross( const Eigen::MatrixBase< LA >& a,
                          const Eigen::MatrixBase< LB >& b ) noexcept
{
    using E = typename LA::Scalar;
    static_assert( LA::SizeAtCompileTime == 3 || LA::SizeAtCompileTime == Eigen::Dynamic,
                   "cross(a, b): a must be a 3-vector" );
    static_assert( LB::SizeAtCompileTime == 3 || LB::SizeAtCompileTime == Eigen::Dynamic,
                   "cross(a, b): b must be a 3-vector" );
    Eigen::Matrix< E, 3, 1 > r;
    r( 0 ) = E( a( 1 ) ) * E( b( 2 ) ) - E( a( 2 ) ) * E( b( 1 ) );
    r( 1 ) = E( a( 2 ) ) * E( b( 0 ) ) - E( a( 0 ) ) * E( b( 2 ) );
    r( 2 ) = E( a( 0 ) ) * E( b( 1 ) ) - E( a( 1 ) ) * E( b( 0 ) );
    return r;
}

// ---------------------------------------------------------------------------
// unit vectors
// ---------------------------------------------------------------------------

/// `v / |v|`. Requires `dot(v, v).value() > 0`. The `1/|v|` factor is one
/// reciprocal-square-root pass shared across all components.
template < typename D >
    requires( bool( D::IsVectorAtCompileTime ) && detail::ExpansionElement< typename D::Scalar > )
[[nodiscard]] auto unitvec( const Eigen::MatrixBase< D >& v ) noexcept
{
    using E = typename D::Scalar;
    const E inv = tax::pow< -1, 2 >( dot( v, v ) );  // (v·v)^(-1/2) = 1/|v|
    Eigen::Matrix< E, D::SizeAtCompileTime, 1 > r( v.size() );
    for ( Eigen::Index i = 0; i < v.size(); ++i ) r( i ) = E( v( i ) ) * inv;
    return r;
}

/// `(a × b) / |a × b|` — the unit normal to the plane of `a` and `b`.
template < typename LA, typename LB >
    requires( detail::ExpansionElement< typename LA::Scalar > )
[[nodiscard]] auto unitcross( const Eigen::MatrixBase< LA >& a,
                              const Eigen::MatrixBase< LB >& b ) noexcept
{
    return unitvec( cross( a, b ) );
}

// ---------------------------------------------------------------------------
// angle
// ---------------------------------------------------------------------------

/// Angle between `a` and `b`, `acos( (a·b) / (|a| |b|) )`. The two norms are
/// combined under one reciprocal square root: `(a·b) · (|a|²|b|²)^(-1/2)`.
/// Requires the cosine's constant term to satisfy `|cos| < 1` (the `acos`
/// domain) — i.e. `a` and `b` not parallel at the expansion point.
template < typename LA, typename LB >
    requires( bool( LA::IsVectorAtCompileTime ) && bool( LB::IsVectorAtCompileTime ) &&
              detail::ExpansionElement< typename LA::Scalar > )
[[nodiscard]] auto angle( const Eigen::MatrixBase< LA >& a,
                          const Eigen::MatrixBase< LB >& b ) noexcept
{
    using E = typename LA::Scalar;
    const E cos = dot( a, b ) * tax::pow< -1, 2 >( dot( a, a ) * dot( b, b ) );
    return tax::acos( cos );
}

// ---------------------------------------------------------------------------
// projections
// ---------------------------------------------------------------------------

/// Projection of `a` onto the direction `d`: `(a·d / d·d) · d`.
/// Requires `dot(d, d).value() != 0`.
template < typename LA, typename LD >
    requires( bool( LA::IsVectorAtCompileTime ) && bool( LD::IsVectorAtCompileTime ) &&
              detail::ExpansionElement< typename LA::Scalar > )
[[nodiscard]] auto projvec( const Eigen::MatrixBase< LA >& a,
                            const Eigen::MatrixBase< LD >& d ) noexcept
{
    using E = typename LA::Scalar;
    const E coef = dot( a, d ) / dot( d, d );
    Eigen::Matrix< E, LD::SizeAtCompileTime, 1 > r( d.size() );
    for ( Eigen::Index i = 0; i < d.size(); ++i ) r( i ) = E( d( i ) ) * coef;
    return r;
}

/// Projection of `a` onto the plane with normal `n`: `a - projvec(a, n)`.
/// Requires `dot(n, n).value() != 0`.
template < typename LA, typename LN >
    requires( bool( LA::IsVectorAtCompileTime ) && bool( LN::IsVectorAtCompileTime ) &&
              detail::ExpansionElement< typename LA::Scalar > )
[[nodiscard]] auto projplane( const Eigen::MatrixBase< LA >& a,
                              const Eigen::MatrixBase< LN >& n ) noexcept
{
    using E = typename LA::Scalar;
    const auto p = projvec( a, n );
    Eigen::Matrix< E, LA::SizeAtCompileTime, 1 > r( a.size() );
    for ( Eigen::Index i = 0; i < a.size(); ++i ) r( i ) = E( a( i ) ) - p( i );
    return r;
}

}  // namespace tax::la

// Reachable directly under `tax`.
namespace tax
{
using la::angle;
using la::cross;
using la::dot;
using la::projplane;
using la::projvec;
using la::unitcross;
using la::unitvec;
}  // namespace tax
