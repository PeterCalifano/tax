// include/tax/la/num_traits.hpp
//
// Two pieces that make `tax::TaylorExpansion` interoperate with
// Eigen:
//
//   1. `Eigen::NumTraits` specialization — declares the TE as a
//      first-class Eigen scalar so an Eigen::Matrix of TEs picks up
//      ABI-correct cost estimates and dispatch.
//
//   2. Internal traits inside `tax::la::detail` used by every other
//      la/ header to introspect TE template parameters
//      (te_traits, is_te) and to rebind an Eigen matrix expression's
//      scalar type (rebind_matrix_t).

#pragma once

#include <Eigen/Core>
#include <type_traits>

#include <tax/core/taylor_expansion.hpp>

// -----------------------------------------------------------------------------
// NumTraits specialization — namespace Eigen
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Internal traits — namespace tax::la::detail
// -----------------------------------------------------------------------------

namespace tax::la::detail
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

}  // namespace tax::la::detail
