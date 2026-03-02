#pragma once

#include <vector>

namespace tax::ode
{

/**
 * @brief Time/state samples returned by Taylor integration.
 *
 * @tparam Vec State vector type.
 */
template < typename Vec >
struct Solution
{
    std::vector< double >  t;  ///< Time values.
    std::vector< Vec >     y;  ///< State snapshots.
};

}  // namespace tax::ode
