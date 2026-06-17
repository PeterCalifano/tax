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
#include <tax/core/taylor_expansion.hpp>
#include <type_traits>

// -----------------------------------------------------------------------------
// NumTraits specialization — namespace Eigen
// -----------------------------------------------------------------------------

namespace Eigen
{

template < typename T, int N, int M, typename Storage >
struct NumTraits< tax::TaylorExpansion< T, N, M, Storage > > : NumTraits< T >
{
    using Self = tax::TaylorExpansion< T, N, M, Storage >;
    using Real = Self;
    using NonInteger = Self;
    using Nested = Self;

    static constexpr int kNc = int( tax::numMonomials( N, M ) );

    enum
    {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = kNc,
        AddCost = kNc,
        // kNc * kNc overflows int for kNc > ~46340; clamp to HugeCost (Eigen's
        // "very expensive scalar" sentinel) so the cost stays a valid int.
        MulCost = kNc < 46341 ? kNc * kNc : HugeCost
    };

    // NumTraits<T> (the base) declares these returning the scalar T, but our
    // Real is Self (the expansion). Re-expose them as constant expansions so
    // Eigen's normwise/fuzzy comparison paths see a Real-typed threshold rather
    // than relying on the implicit scalar -> constant-TE conversion.
    static inline Self epsilon() { return Self( NumTraits< T >::epsilon() ); }
    static inline Self dummy_precision() { return Self( NumTraits< T >::dummy_precision() ); }
    static inline Self highest() { return Self( NumTraits< T >::highest() ); }
    static inline Self lowest() { return Self( NumTraits< T >::lowest() ); }
    static inline Self infinity() { return Self( NumTraits< T >::infinity() ); }
    static inline Self quiet_NaN() { return Self( NumTraits< T >::quiet_NaN() ); }
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
    static constexpr int vars_v = M;
    using storage_t = S;
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
    Eigen::Matrix< Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime, Derived::Options,
                   Derived::MaxRowsAtCompileTime, Derived::MaxColsAtCompileTime >;

}  // namespace tax::la::detail
