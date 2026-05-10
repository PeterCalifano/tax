#pragma once

#include <tax/utils/enumeration.hpp>

namespace tax::detail
{

template < typename T, int N, int M >
/**
 * @brief Truncated multivariate Cauchy product `out = f * g`.
 * @details Output is truncated to total degree `N`.
 */
constexpr void cauchyProduct( std::array< T, numMonomials( N, M ) >& out,
                              const std::array< T, numMonomials( N, M ) >& f,
                              const std::array< T, numMonomials( N, M ) >& g ) noexcept
{
    out = {};

    if constexpr ( M == 1 )
    {
        for ( int d = 0; d <= N; ++d )
            for ( int k = 0; k <= d; ++k ) out[d] += f[k] * g[d - k];
    } else
    {
        for ( int d = 0; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                forEachSubIndex< M >( alpha,
                                      [&]( auto bi, auto gi ) { out[ai] += f[bi] * g[gi]; } );
            } );
        }
    }
}

template < typename T, int N, int M >
/**
 * @brief Truncated self-product `out = f * f`, exploiting symmetry.
 * @details Enumerates each unordered pair {beta, gamma} with beta+gamma=alpha only once,
 *          doubling the off-diagonal contribution. Yields ~2x fewer multiplications than
 *          a general cauchyProduct call.
 */
constexpr void cauchySelfProduct( std::array< T, numMonomials( N, M ) >& out,
                                  const std::array< T, numMonomials( N, M ) >& f ) noexcept
{
    out = {};

    if constexpr ( M == 1 )
    {
        for ( int d = 0; d <= N; ++d )
        {
            // Enumerate only k <= d-k, i.e. k <= d/2.
            for ( int k = 0; k + k < d; ++k ) out[d] += T{ 2 } * f[k] * f[d - k];
            if ( d % 2 == 0 ) out[d] += f[d / 2] * f[d / 2];
        }
    } else
    {
        for ( int d = 0; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                forEachSubIndex< M >( alpha, [&]( auto bi, auto gi ) {
                    if ( bi < gi )
                        out[ai] += T{ 2 } * f[bi] * f[gi];
                    else if ( bi == gi )
                        out[ai] += f[bi] * f[bi];
                } );
            } );
        }
    }
}

template < typename T, int N, int M >
/**
 * @brief Truncated multivariate Cauchy accumulate `out += f * g`.
 * @details Contribution is truncated to total degree `N`.
 */
constexpr void cauchyAccumulate( std::array< T, numMonomials( N, M ) >& out,
                                 const std::array< T, numMonomials( N, M ) >& f,
                                 const std::array< T, numMonomials( N, M ) >& g ) noexcept
{
    if constexpr ( M == 1 )
    {
        for ( int d = 0; d <= N; ++d )
            for ( int k = 0; k <= d; ++k ) out[d] += f[k] * g[d - k];
    } else
    {
        for ( int d = 0; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                forEachSubIndex< M >( alpha,
                                      [&]( auto bi, auto gi ) { out[ai] += f[bi] * g[gi]; } );
            } );
        }
    }
}

// =============================================================================
// Runtime-shape variants (used by the dynamic-shape `TaylorExpansionT`).
// =============================================================================

/// @brief Runtime overload of `cauchyAccumulate`: `out += f * g` with runtime `(N, M)`.
template < typename T >
inline void cauchyAccumulateRT( T* out, const T* f, const T* g, std::size_t N,
                                std::size_t M ) noexcept
{
    if ( M == 1 )
    {
        for ( std::size_t d = 0; d <= N; ++d )
            for ( std::size_t k = 0; k <= d; ++k ) out[d] += f[k] * g[d - k];
    }
    else
    {
        for ( int d = 0; d <= int( N ); ++d )
        {
            forEachMonomial( int( M ), d, [&]( std::span< const int > alpha, std::size_t ai ) {
                forEachSubIndex( alpha,
                                 [&]( std::size_t bi, std::size_t gi ) { out[ai] += f[bi] * g[gi]; } );
            } );
        }
    }
}

/// @brief Runtime overload of `cauchyProduct`: `out = f * g` with runtime `(N, M)`.
template < typename T >
inline void cauchyProductRT( T* out, const T* f, const T* g, std::size_t N,
                             std::size_t M ) noexcept
{
    const std::size_t S = numMonomials( N, M );
    for ( std::size_t i = 0; i < S; ++i ) out[i] = T{ 0 };
    cauchyAccumulateRT( out, f, g, N, M );
}

/// @brief Runtime overload of `cauchySelfProduct`: `out = f * f` exploiting symmetry.
template < typename T >
inline void cauchySelfProductRT( T* out, const T* f, std::size_t N, std::size_t M ) noexcept
{
    const std::size_t S = numMonomials( N, M );
    for ( std::size_t i = 0; i < S; ++i ) out[i] = T{ 0 };

    if ( M == 1 )
    {
        for ( std::size_t d = 0; d <= N; ++d )
        {
            for ( std::size_t k = 0; k + k < d; ++k ) out[d] += T{ 2 } * f[k] * f[d - k];
            if ( d % 2 == 0 ) out[d] += f[d / 2] * f[d / 2];
        }
    }
    else
    {
        for ( int d = 0; d <= int( N ); ++d )
        {
            forEachMonomial( int( M ), d, [&]( std::span< const int > alpha, std::size_t ai ) {
                forEachSubIndex( alpha, [&]( std::size_t bi, std::size_t gi ) {
                    if ( bi < gi )
                        out[ai] += T{ 2 } * f[bi] * f[gi];
                    else if ( bi == gi )
                        out[ai] += f[bi] * f[bi];
                } );
            } );
        }
    }
}

}  // namespace tax::detail
