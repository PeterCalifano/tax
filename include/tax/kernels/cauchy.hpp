#pragma once

#include <cstddef>
#include <utility>

#include <tax/kernels/cauchy_stencil.hpp>
#include <tax/utils/enumeration.hpp>

namespace tax::detail
{

// -----------------------------------------------------------------------------
// Univariate (M = 1) compile-time-unrolled Cauchy kernels.
//
// Output degree `D` and the dot-product range `Ks...` are template parameters
// so the compiler sees one flat FMA chain per row rather than a triangular
// loop nest with varying inner trip count. That lets it schedule the FMAs as
// a wide SIMD graph and beats the equivalent runtime loop by 1.0x-2.0x for
// the N values we use. Helpers are hoisted out of generic lambdas to side-step
// a GCC 13 ICE on nested-pack-expansion-inside-lambda.
// -----------------------------------------------------------------------------

template < typename T, int N, std::size_t D, std::size_t... Ks >
constexpr T cauchyUniRow( const Coeffs< T, N, 1 >& f, const Coeffs< T, N, 1 >& g,
                          std::index_sequence< Ks... > ) noexcept
{
    return ( ( f[Ks] * g[D - Ks] ) + ... + T{ 0 } );
}

template < typename T, int N, std::size_t... Ds >
constexpr void cauchyUniProductImpl( Coeffs< T, N, 1 >& out,
                                     const Coeffs< T, N, 1 >& f,
                                     const Coeffs< T, N, 1 >& g,
                                     std::index_sequence< Ds... > ) noexcept
{
    ( ( out[Ds] = cauchyUniRow< T, N, Ds >( f, g, std::make_index_sequence< Ds + 1 >{} ) ),
      ... );
}

template < typename T, int N, std::size_t... Ds >
constexpr void cauchyUniAccumulateImpl( Coeffs< T, N, 1 >& out,
                                        const Coeffs< T, N, 1 >& f,
                                        const Coeffs< T, N, 1 >& g,
                                        std::index_sequence< Ds... > ) noexcept
{
    ( ( out[Ds] += cauchyUniRow< T, N, Ds >( f, g, std::make_index_sequence< Ds + 1 >{} ) ),
      ... );
}

template < typename T, int N, std::size_t D, std::size_t... Ks >
constexpr T selfUniRow( const Coeffs< T, N, 1 >& f, std::index_sequence< Ks... > ) noexcept
{
    // Range is Ks in [0, D/2]. Factor 2 for off-diagonal (2*Ks < D),
    // 1 for the diagonal pair (2*Ks == D); both folded at compile time.
    return ( ( ( 2 * Ks < D ? T{ 2 } : ( 2 * Ks == D ? T{ 1 } : T{ 0 } ) )
               * f[Ks] * f[D - Ks] )
             + ... + T{ 0 } );
}

template < typename T, int N, std::size_t... Ds >
constexpr void selfUniImpl( Coeffs< T, N, 1 >& out, const Coeffs< T, N, 1 >& f,
                            std::index_sequence< Ds... > ) noexcept
{
    ( ( out[Ds] = selfUniRow< T, N, Ds >( f, std::make_index_sequence< Ds / 2 + 1 >{} ) ),
      ... );
}

template < typename T, int N, int M >
/**
 * @brief Truncated multivariate Cauchy product `out = f * g`.
 * @details Output is truncated to total degree `N`.
 */
constexpr void cauchyProduct( Coeffs< T, N, M >& out,
                              const Coeffs< T, N, M >& f,
                              const Coeffs< T, N, M >& g ) noexcept
{
    if constexpr ( M == 1 )
    {
        cauchyUniProductImpl< T, N >( out, f, g, std::make_index_sequence< N + 1 >{} );
    } else
    {
        out = {};
        using S = CauchyStencil< N, M >;
        for ( std::size_t k = 0; k < S::NC; ++k )
        {
            T acc{ 0 };
            for ( std::size_t j = S::offsets[k]; j < S::offsets[k + 1]; ++j )
                acc += f[S::col_a[j]] * g[S::col_b[j]];
            out[k] = acc;
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
constexpr void cauchySelfProduct( Coeffs< T, N, M >& out,
                                  const Coeffs< T, N, M >& f ) noexcept
{
    if constexpr ( M == 1 )
    {
        selfUniImpl< T, N >( out, f, std::make_index_sequence< N + 1 >{} );
    } else
    {
        out = {};
        using S = CauchySymStencil< N, M >;
        for ( std::size_t k = 0; k < S::NC; ++k )
        {
            T acc{ 0 };
            for ( std::size_t j = S::offsets[k]; j < S::offsets[k + 1]; ++j )
            {
                const T fab = f[S::col_a[j]] * f[S::col_b[j]];
                acc += S::is_diag[j] ? fab : T{ 2 } * fab;
            }
            out[k] = acc;
        }
    }
}

template < typename T, int N, int M >
/**
 * @brief Truncated multivariate Cauchy accumulate `out += f * g`.
 * @details Contribution is truncated to total degree `N`.
 */
constexpr void cauchyAccumulate( Coeffs< T, N, M >& out,
                                 const Coeffs< T, N, M >& f,
                                 const Coeffs< T, N, M >& g ) noexcept
{
    if constexpr ( M == 1 )
    {
        cauchyUniAccumulateImpl< T, N >( out, f, g, std::make_index_sequence< N + 1 >{} );
    } else
    {
        using S = CauchyStencil< N, M >;
        for ( std::size_t k = 0; k < S::NC; ++k )
        {
            T acc{ 0 };
            for ( std::size_t j = S::offsets[k]; j < S::offsets[k + 1]; ++j )
                acc += f[S::col_a[j]] * g[S::col_b[j]];
            out[k] += acc;
        }
    }
}

// =============================================================================
// Runtime-shape variants (used by the dynamic-shape `TaylorExpansionT`).
// =============================================================================

/// @brief Runtime overload of `cauchyAccumulate`: `out += f * g` with runtime `(N, M)`.
template < typename T >
inline void cauchyAccumulate( T* out, const T* f, const T* g, std::size_t N,
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
inline void cauchyProduct( T* out, const T* f, const T* g, std::size_t N,
                             std::size_t M ) noexcept
{
    const std::size_t S = numMonomials( N, M );
    for ( std::size_t i = 0; i < S; ++i ) out[i] = T{ 0 };
    cauchyAccumulate( out, f, g, N, M );
}

/// @brief Runtime overload of `cauchySelfProduct`: `out = f * f` exploiting symmetry.
template < typename T >
inline void cauchySelfProduct( T* out, const T* f, std::size_t N, std::size_t M ) noexcept
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
