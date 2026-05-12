#pragma once

#include <cmath>
#include <cstddef>
#include <utility>

#include <tax/utils/combinatorics.hpp>

namespace tax::detail
{

// =============================================================================
// Compile-time-unrolled scalar recurrences for the univariate (M = 1) kernel
// paths.
//
// For a static `N`, every triangular loop nest in the Taylor recurrences has a
// fixed iteration count.  Turning the trip count into a template parameter
// (via `std::index_sequence`) lets the compiler see one flat FMA chain per
// output row and schedule / SIMD-vectorise it without the runtime trip-count
// uncertainty of the equivalent for-loop.
//
// All helpers live in `tax::detail`; they're hoisted to namespace scope to
// side-step a GCC 13 ICE on nested-pack-expansion inside a generic lambda.
// =============================================================================

// -----------------------------------------------------------------------------
// Cauchy product / accumulate: out[D] = sum_{k=0..D} f[k] * g[D-k]
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

// -----------------------------------------------------------------------------
// Self-product: out[D] = sum_{k=0..D/2} factor(k,D) * f[k] * f[D-k]
//               with factor = 2 off-diagonal, 1 on the diagonal.
// -----------------------------------------------------------------------------

template < typename T, int N, std::size_t D, std::size_t... Ks >
constexpr T selfUniRow( const Coeffs< T, N, 1 >& f, std::index_sequence< Ks... > ) noexcept
{
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

// -----------------------------------------------------------------------------
// Reciprocal (forward substitution): solve f * out = 1.
//
//   out[0] = 1 / f[0]
//   out[D] = -inv_f0 * sum_{k=1..D} f[k] * out[D-k]   (D >= 1)
//
// Cross-row dependency: row D needs rows 0..D-1.  Unroll exposes the FMA
// chain inside one row; rows are sequenced left-to-right by the comma
// expansion.  `Ks` ranges 0..D-1 and we shift by +1 inside.
// -----------------------------------------------------------------------------

template < typename T, int N, std::size_t D, std::size_t... Ks >
constexpr T reciprocalUniRow( const Coeffs< T, N, 1 >& f, const Coeffs< T, N, 1 >& out,
                              std::index_sequence< Ks... > ) noexcept
{
    // k = Ks + 1 in [1, D];  term = f[k] * out[D - k] = f[Ks+1] * out[D-1-Ks].
    return ( ( f[Ks + 1] * out[D - 1 - Ks] ) + ... + T{ 0 } );
}

template < typename T, int N, std::size_t... Ds >
constexpr void reciprocalUniImpl( Coeffs< T, N, 1 >& out, const Coeffs< T, N, 1 >& f,
                                  T inv_f0, std::index_sequence< Ds... > ) noexcept
{
    // Each Ds in 0..N-1; produce row D = Ds + 1.  Row 0 is set by the caller.
    ( ( out[Ds + 1] =
            -reciprocalUniRow< T, N, Ds + 1 >( f, out, std::make_index_sequence< Ds + 1 >{} )
            * inv_f0 ),
      ... );
}

// -----------------------------------------------------------------------------
// Square root (forward substitution): solve out * out = f.
//
//   out[0] = sqrt(f[0])
//   out[D] = (f[D] - sum_{k=1..D/2} factor(k,D) * out[k] * out[D-k]) * inv2g0
//           where factor(k,D) is 2 (off-diagonal) or 1 (diagonal k == D-k).
//
// Same shape as selfUniRow except the range is 1..D/2 (we move the
// 2*out[0]*out[D] endpoint to the LHS).
// -----------------------------------------------------------------------------

template < typename T, int N, std::size_t D, std::size_t... Ks >
constexpr T sqrtUniRow( const Coeffs< T, N, 1 >& out, std::index_sequence< Ks... > ) noexcept
{
    // k = Ks + 1 in [1, D/2];  factor: 2 if 2k<D, 1 if 2k==D.
    return ( ( ( 2 * ( Ks + 1 ) < D ? T{ 2 } : ( 2 * ( Ks + 1 ) == D ? T{ 1 } : T{ 0 } ) )
               * out[Ks + 1] * out[D - Ks - 1] )
             + ... + T{ 0 } );
}

template < typename T, int N, std::size_t... Ds >
constexpr void sqrtUniImpl( Coeffs< T, N, 1 >& out, const Coeffs< T, N, 1 >& f,
                            T inv2g0, std::index_sequence< Ds... > ) noexcept
{
    // Each Ds in 0..N-1; produce row D = Ds + 1.  Row 0 is set by the caller.
    ( ( out[Ds + 1] =
            ( f[Ds + 1]
              - sqrtUniRow< T, N, Ds + 1 >( out, std::make_index_sequence< ( Ds + 1 ) / 2 >{} ) )
            * inv2g0 ),
      ... );
}

}  // namespace tax::detail
