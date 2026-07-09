#pragma once

#include <array>
#include <cmath>
#include <span>
#include <tax/core/scheme/isotropic.hpp>
#include <tax/kernels/cauchy.hpp>
#include <tax/kernels/recurrence_stencil.hpp>

namespace tax::detail::kernels
{

// ---------------------------------------------------------------------------
// Shared recurrence drivers
//
// Nearly every series kernel is one of two degree-by-degree recurrences over
// the same (beta, gamma) decomposition rows; the drivers below implement the
// univariate and multivariate walks once, and the kernels reduce to "compute
// h, seed the constant term, call the driver".
// ---------------------------------------------------------------------------

/// Forward recurrence `h * out' = Sign * src'` with seed `out[0] = out0`
/// (scheme-generic). Degree by degree:
///   d * h[0] * out_alpha = d * Sign * src_alpha - sum (d - |beta|) h_beta out_gamma.
/// Requires `h[0] != 0`; `src`/`h` must not alias `out`. Covers the whole
/// integral-of-a-quotient family: log, asin/acos/atan/atan2, asinh/acosh/atanh.
template < int Sign = 1, typename T, tax::IndexScheme Scheme >
constexpr void seriesDerivQuotient( std::array< T, Scheme::nCoeff >& out, T out0,
                                    const std::array< T, Scheme::nCoeff >& src,
                                    const std::array< T, Scheme::nCoeff >& h ) noexcept
{
    static_assert( Sign == 1 || Sign == -1 );
    out = {};
    out[0] = out0;
    const T inv_h0 = T{ 1 } / h[0];

    if constexpr ( Scheme::isUnivariate )
    {
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 1; k < d; ++k )
                rhs += T( k ) * h[std::size_t( d - k )] * out[std::size_t( k )];
            const T s = Sign > 0 ? src[std::size_t( d )] : -src[std::size_t( d )];
            out[std::size_t( d )] = ( s - rhs / T( d ) ) * inv_h0;
        }
    } else
    {
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
                T rhs = T{ 0 };
                // |beta| == d entries carry weight (d - db) == 0.
                for ( const RecurrenceEntry& e : row )
                    rhs += T( d - int( e.db ) ) * h[e.b_idx] * out[e.g_idx];
                const T s = Sign > 0 ? src[ai] : -src[ai];
                out[ai] = ( s - rhs / T( d ) ) * inv_h0;
            } );
    }
}

/// Forward recurrence `out' = src' * h` with seed `out[0] = out0`
/// (scheme-generic): out_alpha = (1/d) * sum |beta| src_beta h_gamma.
/// `h` may alias `out` (exp is `out' = a' * out`): rows read `h` only at
/// strictly lower total degree, which is already final. `src` must not
/// alias `out`.
template < typename T, tax::IndexScheme Scheme >
constexpr void seriesDerivProduct( std::array< T, Scheme::nCoeff >& out, T out0,
                                   const std::array< T, Scheme::nCoeff >& src,
                                   const std::array< T, Scheme::nCoeff >& h ) noexcept
{
    out = {};
    out[0] = out0;

    if constexpr ( Scheme::isUnivariate )
    {
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 0; k < d; ++k )
                rhs += T( d - k ) * src[std::size_t( d - k )] * h[std::size_t( k )];
            out[std::size_t( d )] = rhs / T( d );
        }
    } else
    {
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
                T rhs = T{ 0 };
                for ( const RecurrenceEntry& e : row ) rhs += T( e.db ) * src[e.b_idx] * h[e.g_idx];
                out[ai] = rhs / T( d );
            } );
    }
}

/// Self-product `out = f * f` (M == 1 exploits pair symmetry; M >= 2 forwards to cauchyProduct).
template < typename T, int N, int M >
constexpr void cauchySelfProduct( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& f ) noexcept
{
    tax::cauchySelfProduct< T, tax::IsotropicScheme< N, M > >( out, f );
}

/// Square series `out = a^2` via the symmetric self-product (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
constexpr void seriesSquare( std::array< T, Scheme::nCoeff >& out,
                             const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    tax::cauchySelfProduct< T, Scheme >( out, a );
}

/// Cube series `out = a^3` via two Cauchy products (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
constexpr void seriesCube( std::array< T, Scheme::nCoeff >& out,
                           const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    constexpr std::size_t S = Scheme::nCoeff;
    std::array< T, S > tmp{};
    tax::cauchySelfProduct< T, Scheme >( tmp, a );
    tax::cauchyProduct< T, Scheme >( out, tmp, a );
}

/// Reciprocal series `a * out = 1` by forward substitution (scheme-generic). Requires `a[0] != 0`.
template < typename T, tax::IndexScheme Scheme >
constexpr void seriesReciprocal( std::array< T, Scheme::nCoeff >& out,
                                 const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    out = {};
    const T inv_a0 = T{ 1 } / a[0];
    out[0] = inv_a0;

    if constexpr ( Scheme::isUnivariate )
    {
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 1; k <= d; ++k ) rhs -= a[std::size_t( k )] * out[std::size_t( d - k )];
            out[std::size_t( d )] = rhs * inv_a0;
        }
    } else
    {
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int, std::span< const RecurrenceEntry > row ) {
                T rhs = T{ 0 };
                for ( const RecurrenceEntry& e : row ) rhs -= a[e.b_idx] * out[e.g_idx];
                out[ai] = rhs * inv_a0;
            } );
    }
}

/// Reciprocal series `a * out = 1` by forward substitution. Requires `a[0] != 0`.
template < typename T, int N, int M >
constexpr void seriesReciprocal( Coeffs< T, N, M >& out, const Coeffs< T, N, M >& a ) noexcept
{
    seriesReciprocal< T, tax::IsotropicScheme< N, M > >( out, a );
}

/// Quotient series `out = a / b` by forward substitution (scheme-generic). Requires `b[0] != 0`;
/// `out` must not alias `a` or `b`.
template < typename T, tax::IndexScheme Scheme >
constexpr void seriesDivide( std::array< T, Scheme::nCoeff >& out,
                             const std::array< T, Scheme::nCoeff >& a,
                             const std::array< T, Scheme::nCoeff >& b ) noexcept
{
    out = {};
    const T inv_b0 = T{ 1 } / b[0];
    out[0] = a[0] * inv_b0;

    if constexpr ( Scheme::isUnivariate )
    {
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T rhs = a[std::size_t( d )];
            for ( int k = 1; k <= d; ++k ) rhs -= b[std::size_t( k )] * out[std::size_t( d - k )];
            out[std::size_t( d )] = rhs * inv_b0;
        }
    } else
    {
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int, std::span< const RecurrenceEntry > row ) {
                T rhs = a[ai];
                for ( const RecurrenceEntry& e : row ) rhs -= b[e.b_idx] * out[e.g_idx];
                out[ai] = rhs * inv_b0;
            } );
    }
}

/// Square-root series `out * out = a` (principal branch, scheme-generic). Requires `a[0] > 0`.
template < typename T, tax::IndexScheme Scheme >
void seriesSqrt( std::array< T, Scheme::nCoeff >& out,
                 const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::sqrt;
    out = {};
    out[0] = sqrt( a[0] );
    const T inv2g0 = T{ 1 } / ( T{ 2 } * out[0] );

    if constexpr ( Scheme::isUnivariate )
    {
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T rhs = a[std::size_t( d )];
            for ( int k = 1; k + k < d; ++k )
                rhs -= T{ 2 } * out[std::size_t( k )] * out[std::size_t( d - k )];
            if ( d % 2 == 0 ) rhs -= out[std::size_t( d / 2 )] * out[std::size_t( d / 2 )];
            out[std::size_t( d )] = rhs * inv2g0;
        }
    } else
    {
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int, std::span< const RecurrenceEntry > row ) {
                T rhs = a[ai];
                // |beta| == d entries read out[ai], which is still zero here,
                // so the ordered walk needs no |beta| < d filter.
                for ( const RecurrenceEntry& e : row ) rhs -= out[e.b_idx] * out[e.g_idx];
                out[ai] = rhs * inv2g0;
            } );
    }
}

/// Forward recurrence for `out = a^c` given a precomputed seed `out[0] = a[0]^c`
/// (scheme-generic). It matches the real-power ODE `a·out' = c·a'·out` degree by
/// degree: `out_alpha = (1/(d·a0)) sum (c·|beta| - (d - |beta|)) a_beta out_gamma`.
/// Valid for any real `c` and any `a[0] != 0` — including `a[0] < 0`, provided
/// the seed is the intended real branch (e.g. the real cube root). `out[0]` is
/// left untouched; `out[1..]` are overwritten. `a` must not alias `out`.
template < typename T, tax::IndexScheme Scheme >
constexpr void seriesPowFromSeed( std::array< T, Scheme::nCoeff >& out,
                                  const std::array< T, Scheme::nCoeff >& a, T c ) noexcept
{
    const T inv_a0 = T{ 1 } / a[0];

    if constexpr ( Scheme::isUnivariate )
    {
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T rhs = T{ 0 };
            for ( int k = 0; k < d; ++k )
                rhs +=
                    ( c * T( d - k ) - T( k ) ) * a[std::size_t( d - k )] * out[std::size_t( k )];
            out[std::size_t( d )] = rhs * inv_a0 / T( d );
        }
    } else
    {
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
                T rhs = T{ 0 };
                for ( const RecurrenceEntry& e : row )
                    rhs += ( c * T( e.db ) - T( d - int( e.db ) ) ) * a[e.b_idx] * out[e.g_idx];
                out[ai] = rhs * inv_a0 / T( d );
            } );
    }
}

/// Cubic-root series `out^3 = a` (real branch, scheme-generic). Requires `a[0] != 0`.
/// Runs the real-power recurrence at exponent 1/3, seeded with the real cube
/// root of the constant term — so it handles `a[0] < 0` (unlike `a^(1/3)` via a
/// libm `pow` seed), and is faster than a bespoke `out^3 = a` recurrence.
template < typename T, tax::IndexScheme Scheme >
void seriesCbrt( std::array< T, Scheme::nCoeff >& out,
                 const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    using std::cbrt;
    out = {};
    out[0] = cbrt( a[0] );
    seriesPowFromSeed< T, Scheme >( out, a, T{ 1 } / T{ 3 } );
}

/// Real-exponent power series `out = a^c` (scheme-generic). Requires `a[0] != 0`
/// (and `a[0] > 0` for a non-integer `c`); not constexpr.
template < typename T, tax::IndexScheme Scheme >
void seriesPow( std::array< T, Scheme::nCoeff >& out, const std::array< T, Scheme::nCoeff >& a,
                T c ) noexcept
{
    using std::pow;
    out = {};
    out[0] = pow( a[0], c );
    seriesPowFromSeed< T, Scheme >( out, a, c );
}

/// Integer-exponent power series `out = a^n` via binary exponentiation (scheme-generic; negative n
/// via reciprocal).
template < typename T, tax::IndexScheme Scheme >
constexpr void seriesPowInt( std::array< T, Scheme::nCoeff >& out,
                             const std::array< T, Scheme::nCoeff >& a, int n ) noexcept
{
    constexpr std::size_t S = Scheme::nCoeff;

    if ( n == 0 )
    {
        out = {};
        out[0] = T{ 1 };
        return;
    }
    if ( n == 1 )
    {
        out = a;
        return;
    }
    if ( n == -1 )
    {
        seriesReciprocal< T, Scheme >( out, a );
        return;
    }
    if ( n < 0 )
    {
        // a^n = (1/a)^|n|: reciprocate, then raise to the POSITIVE exponent.
        // (Passing n here instead of -n recurses forever — negative-n stack
        // overflow, fixed.)
        std::array< T, S > rec{};
        seriesReciprocal< T, Scheme >( rec, a );
        seriesPowInt< T, Scheme >( out, rec, -n );
        return;
    }
    // n >= 2: binary exponentiation (square-and-multiply). Squarings go
    // through the symmetric self-product kernel; `out` is seeded with the
    // base power of the lowest set bit, skipping the wasted 1 * base
    // multiply of the textbook formulation.
    std::array< T, S > base = a;
    int e = n;
    while ( !( e & 1 ) )
    {
        std::array< T, S > tmp{};
        tax::cauchySelfProduct< T, Scheme >( tmp, base );
        base = tmp;
        e >>= 1;
    }
    out = base;
    e >>= 1;
    while ( e > 0 )
    {
        std::array< T, S > sq{};
        tax::cauchySelfProduct< T, Scheme >( sq, base );
        base = sq;
        if ( e & 1 )
        {
            std::array< T, S > tmp{};
            tax::cauchyProduct< T, Scheme >( tmp, out, base );
            out = tmp;
        }
        e >>= 1;
    }
}

}  // namespace tax::detail::kernels
