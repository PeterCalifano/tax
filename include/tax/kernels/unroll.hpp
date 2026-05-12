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

// -----------------------------------------------------------------------------
// Weighted forward-substitution row sums shared by several transcendentals.
// Each kernel pairs one of these row sums with a kernel-specific per-row
// scaling.  All M = 1 only — the multivariate path would need a weighted
// stencil (out of scope for this commit set).
//
// Pattern A: sum_{k=0..D-1} (D - k) * f[D - k] * g[k]
//   used by exp, erf, sincos, sinhcosh.
//
// Pattern B: sum_{k=1..D-1} k * f[D - k] * g[k]
//   used by log, asin, atan, asinh, acosh, atanh.
//
// Pattern P (pow): sum_{k=0..D-1} (c*(D - k) - k) * f[D - k] * g[k]
//   used by `pow(a, c)` for real exponents.
//
// `tan`/`tanh` reuse `reciprocalUniRow` (`f`, `out` -> sum_{k=1..D} f[k] * out[D-k]).
// -----------------------------------------------------------------------------

template < typename T, int N, std::size_t D, std::size_t... Ks >
constexpr T weightedFwdSumA( const Coeffs< T, N, 1 >& f, const Coeffs< T, N, 1 >& g,
                             std::index_sequence< Ks... > ) noexcept
{
    return ( ( T( D - Ks ) * f[D - Ks] * g[Ks] ) + ... + T{ 0 } );
}

template < typename T, int N, std::size_t D, std::size_t... Ks >
constexpr T weightedFwdSumB( const Coeffs< T, N, 1 >& f, const Coeffs< T, N, 1 >& g,
                             std::index_sequence< Ks... > ) noexcept
{
    // Ks in 0..D-2; k = Ks+1 in 1..D-1.
    return ( ( T( Ks + 1 ) * f[D - Ks - 1] * g[Ks + 1] ) + ... + T{ 0 } );
}

template < typename T, int N, std::size_t D, std::size_t... Ks >
constexpr T weightedFwdSumPow( T c, const Coeffs< T, N, 1 >& f, const Coeffs< T, N, 1 >& g,
                               std::index_sequence< Ks... > ) noexcept
{
    return ( ( ( c * T( D - Ks ) - T( Ks ) ) * f[D - Ks] * g[Ks] ) + ... + T{ 0 } );
}

// --- Drivers: produce out[D] for D = 1..N (out[0] is the caller's responsibility) ---

/// @brief `exp`-shape recurrence: `out[D] = (1/D) * sum (D-k)*a[D-k]*out[k]`.
template < typename T, int N, std::size_t... Ds >
constexpr void expUniImpl( Coeffs< T, N, 1 >& out, const Coeffs< T, N, 1 >& a,
                           std::index_sequence< Ds... > ) noexcept
{
    ( ( out[Ds + 1] =
            weightedFwdSumA< T, N, Ds + 1 >( a, out, std::make_index_sequence< Ds + 1 >{} )
            / T( Ds + 1 ) ),
      ... );
}

/// @brief `erf`-shape recurrence (h precomputed, not the running output):
///        `out[D] = (1/D) * sum (D-k)*a[D-k]*h[k]`.
template < typename T, int N, std::size_t... Ds >
constexpr void erfUniImpl( Coeffs< T, N, 1 >& out, const Coeffs< T, N, 1 >& a,
                           const Coeffs< T, N, 1 >& h,
                           std::index_sequence< Ds... > ) noexcept
{
    ( ( out[Ds + 1] =
            weightedFwdSumA< T, N, Ds + 1 >( a, h, std::make_index_sequence< Ds + 1 >{} )
            / T( Ds + 1 ) ),
      ... );
}

template < typename T, int N, std::size_t D >
constexpr void sinCosUniRowOne( Coeffs< T, N, 1 >& s, Coeffs< T, N, 1 >& c,
                                const Coeffs< T, N, 1 >& a ) noexcept
{
    const T sr = weightedFwdSumA< T, N, D >( a, c, std::make_index_sequence< D >{} );
    const T cr = weightedFwdSumA< T, N, D >( a, s, std::make_index_sequence< D >{} );
    const T inv_d = T{ 1 } / T( D );
    s[D] = sr * inv_d;
    c[D] = -cr * inv_d;
}

/// @brief Joint sin/cos recurrence — computes both `s[D]` and `c[D]` per D.
template < typename T, int N, std::size_t... Ds >
constexpr void sinCosUniImpl( Coeffs< T, N, 1 >& s, Coeffs< T, N, 1 >& c,
                              const Coeffs< T, N, 1 >& a,
                              std::index_sequence< Ds... > ) noexcept
{
    ( sinCosUniRowOne< T, N, Ds + 1 >( s, c, a ), ... );
}

template < typename T, int N, std::size_t D >
constexpr void sinhCoshUniRowOne( Coeffs< T, N, 1 >& sh, Coeffs< T, N, 1 >& ch,
                                  const Coeffs< T, N, 1 >& a ) noexcept
{
    const T sr = weightedFwdSumA< T, N, D >( a, ch, std::make_index_sequence< D >{} );
    const T cr = weightedFwdSumA< T, N, D >( a, sh, std::make_index_sequence< D >{} );
    const T inv_d = T{ 1 } / T( D );
    sh[D] = sr * inv_d;
    ch[D] = cr * inv_d;
}

/// @brief Joint sinh/cosh recurrence — same shape as sincos but no sign flip.
template < typename T, int N, std::size_t... Ds >
constexpr void sinhCoshUniImpl( Coeffs< T, N, 1 >& sh, Coeffs< T, N, 1 >& ch,
                                const Coeffs< T, N, 1 >& a,
                                std::index_sequence< Ds... > ) noexcept
{
    ( sinhCoshUniRowOne< T, N, Ds + 1 >( sh, ch, a ), ... );
}

/// @brief `log`-shape recurrence: `out[D] = (a[D] - sum/D) * inv_a0`,
///        `sum = sum_{k=1..D-1} k * a[D-k] * out[k]`.
template < typename T, int N, std::size_t... Ds >
constexpr void logUniImpl( Coeffs< T, N, 1 >& out, const Coeffs< T, N, 1 >& a,
                           T inv_a0, std::index_sequence< Ds... > ) noexcept
{
    ( ( out[Ds + 1] =
            ( a[Ds + 1]
              - weightedFwdSumB< T, N, Ds + 1 >( a, out, std::make_index_sequence< Ds >{} )
                    / T( Ds + 1 ) )
            * inv_a0 ),
      ... );
}

/// @brief `asin/atan/asinh/acosh/atanh`-shape — same as log but with `h` instead of `a`
///        on the sum side, and a per-kernel `inv_h0` instead of `inv_a0`.
template < typename T, int N, std::size_t... Ds >
constexpr void asinLikeUniImpl( Coeffs< T, N, 1 >& out, const Coeffs< T, N, 1 >& a,
                                const Coeffs< T, N, 1 >& h, T inv_h0,
                                std::index_sequence< Ds... > ) noexcept
{
    ( ( out[Ds + 1] =
            ( a[Ds + 1]
              - weightedFwdSumB< T, N, Ds + 1 >( h, out, std::make_index_sequence< Ds >{} )
                    / T( Ds + 1 ) )
            * inv_h0 ),
      ... );
}

/// @brief `pow`-shape recurrence: `out[D] = sum * inv_a0 / D`,
///        `sum = sum_{k=0..D-1} (c*(D-k) - k) * a[D-k] * out[k]`.
template < typename T, int N, std::size_t... Ds >
constexpr void powUniImpl( Coeffs< T, N, 1 >& out, const Coeffs< T, N, 1 >& a, T c,
                           T inv_a0, std::index_sequence< Ds... > ) noexcept
{
    ( ( out[Ds + 1] =
            weightedFwdSumPow< T, N, Ds + 1 >( c, a, out, std::make_index_sequence< Ds + 1 >{} )
            * inv_a0 / T( Ds + 1 ) ),
      ... );
}

/// @brief `tan`/`tanh`-shape recurrence (a recip-style solve with a non-zero rhs):
///        `out[D] = (s[D] - sum) * inv_c0`,
///        `sum = sum_{k=1..D} c[k] * out[D-k]` — same as `reciprocalUniRow`.
template < typename T, int N, std::size_t... Ds >
constexpr void tanLikeUniImpl( Coeffs< T, N, 1 >& out, const Coeffs< T, N, 1 >& s,
                               const Coeffs< T, N, 1 >& c, T inv_c0,
                               std::index_sequence< Ds... > ) noexcept
{
    // Each Ds in 0..N-1; produce row D = Ds + 1.  Row 0 is set by the caller.
    ( ( out[Ds + 1] =
            ( s[Ds + 1]
              - reciprocalUniRow< T, N, Ds + 1 >( c, out, std::make_index_sequence< Ds + 1 >{} ) )
            * inv_c0 ),
      ... );
}

}  // namespace tax::detail
