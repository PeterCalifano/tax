#pragma once

#include <Eigen/Core>
#include <tax/da.hpp>

namespace Eigen
{

template < typename T, int N, int M >
struct NumTraits< tax::TDA< T, N, M > > : NumTraits< T >
{
    using Real = tax::TDA< T, N, M >;
    using NonInteger = tax::TDA< T, N, M >;
    using Literal = tax::TDA< T, N, M >;
    using Nested = tax::TDA< T, N, M >;

    enum
    {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = NumTraits< T >::ReadCost,
        AddCost = NumTraits< T >::AddCost,
        MulCost = NumTraits< T >::MulCost
    };
};

}  // namespace Eigen
