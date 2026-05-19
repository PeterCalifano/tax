#pragma once

#include <array>
#include <cstddef>
#include <tax/core/multi_index.hpp>

namespace tax::storage
{

/// @brief Tag type selecting the dense (std::array) storage policy.
struct Dense
{
};

/**
 * @brief Dense coefficient container for a TaylorExpansion.
 *
 * Stores `numMonomials(N, M)` coefficients in a stack-allocated `std::array`.
 * Provides element access, mutation helpers, and a `forEachNonzero` traversal
 * that iterates over all entries (dense: every entry is "potentially nonzero").
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order (>= 0).
 * @tparam M  Number of variables (>= 1).
 */
template < typename T, int N, int M >
struct DenseContainer
{
    using value_type                            = T;
    using coeffs_type                           = Coeffs< T, N, M >;
    static constexpr std::size_t nCoefficients = numMonomials( N, M );

    coeffs_type data{};

    [[nodiscard]] constexpr T value() const noexcept { return data[0]; }

    [[nodiscard]] constexpr T operator[]( std::size_t k ) const noexcept { return data[k]; }
    [[nodiscard]] constexpr T& operator[]( std::size_t k ) noexcept { return data[k]; }

    constexpr void set( std::size_t k, T v ) noexcept { data[k] = v; }
    constexpr void accumulate( std::size_t k, T v ) noexcept { data[k] += v; }

    template < typename Fn >
    constexpr void forEachNonzero( Fn&& fn ) const noexcept
    {
        for ( std::size_t k = 0; k < nCoefficients; ++k )
            fn( k, data[k] );
    }
};

}  // namespace tax::storage
