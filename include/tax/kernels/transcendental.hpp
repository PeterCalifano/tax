#pragma once

#include <cmath>
#include <numbers>
#include <span>
#include <tax/core/cmath.hpp>
#include <tax/core/concepts.hpp>
#include <tax/core/scheme/isotropic.hpp>
#include <tax/kernels/algebra.hpp>

namespace tax::detail::kernels
{

/// Natural exponential series `out = exp(a)` (scheme-generic): out' = a' * out.
template < typename T, tax::IndexScheme Scheme >
constexpr void seriesExp( std::array< T, Scheme::nCoeff >& out,
                          const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    seriesDerivProduct< T, Scheme >( out, cmath::ctExp( a[0] ), a, out );
}

/// Natural logarithm series `out = log(a)` (scheme-generic). Requires `a[0] > 0`.
template < typename T, tax::IndexScheme Scheme >
constexpr void seriesLog( std::array< T, Scheme::nCoeff >& out,
                          const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    seriesDerivQuotient< 1, T, Scheme >( out, cmath::ctLog( a[0] ), a, a );
}

/// Joint `exp(a)` / `exp(-a)` pair: one negation, two recurrence passes.
template < typename T, tax::IndexScheme Scheme >
constexpr void seriesExpPair( std::array< T, Scheme::nCoeff >& ep,
                              std::array< T, Scheme::nCoeff >& em,
                              const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    std::array< T, Scheme::nCoeff > neg_a = a;
    for ( T& v : neg_a ) v = -v;
    seriesExp< T, Scheme >( ep, a );
    seriesExp< T, Scheme >( em, neg_a );
}

/// Coupled hyperbolic series: jointly compute `sinh(a)` and `cosh(a)` from one
/// exp(a)/exp(-a) pair (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
constexpr void seriesSinhCosh( std::array< T, Scheme::nCoeff >& s,
                               std::array< T, Scheme::nCoeff >& c,
                               const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    constexpr std::size_t S = Scheme::nCoeff;
    std::array< T, S > ep{}, em{};
    seriesExpPair< T, Scheme >( ep, em, a );

    s = {};
    c = {};
    s[0] = cmath::ctSinh( a[0] );
    c[0] = cmath::ctCosh( a[0] );
    for ( std::size_t i = 1; i < S; ++i )
    {
        s[i] = ( ep[i] - em[i] ) * T{ 0.5 };
        c[i] = ( ep[i] + em[i] ) * T{ 0.5 };
    }
}

/// Hyperbolic sine series `out = sinh(a)` (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
constexpr void seriesSinh( std::array< T, Scheme::nCoeff >& out,
                           const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    std::array< T, Scheme::nCoeff > c{};
    seriesSinhCosh< T, Scheme >( out, c, a );
}

/// Hyperbolic cosine series `out = cosh(a)` (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
constexpr void seriesCosh( std::array< T, Scheme::nCoeff >& out,
                           const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    std::array< T, Scheme::nCoeff > s{};
    seriesSinhCosh< T, Scheme >( s, out, a );
}

/// Hyperbolic tangent series `out = tanh(a)` (scheme-generic): sinh/cosh in one
/// substitution pass over a single shared exp(a)/exp(-a) pair.
template < typename T, tax::IndexScheme Scheme >
constexpr void seriesTanh( std::array< T, Scheme::nCoeff >& out,
                           const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    constexpr std::size_t S = Scheme::nCoeff;
    std::array< T, S > s{}, c{};
    seriesSinhCosh< T, Scheme >( s, c, a );
    seriesDivide< T, Scheme >( out, s, c );
}

/// Inverse hyperbolic sine series `out = asinh(a)` (scheme-generic):
/// out' = a' / sqrt(1 + a^2).
template < typename T, tax::IndexScheme Scheme >
constexpr void seriesAsinh( std::array< T, Scheme::nCoeff >& out,
                            const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    constexpr std::size_t S = Scheme::nCoeff;

    // h = sqrt(1 + a^2)
    std::array< T, S > asq{}, h{};
    tax::cauchySelfProduct< T, Scheme >( asq, a );
    asq[0] += T{ 1 };
    seriesSqrt< T, Scheme >( h, asq );

    seriesDerivQuotient< 1, T, Scheme >( out, cmath::ctAsinh( a[0] ), a, h );
}

/// Inverse hyperbolic cosine series `out = acosh(a)` (scheme-generic). Requires `a[0] > 1`:
/// out' = a' / sqrt(a^2 - 1).
template < typename T, tax::IndexScheme Scheme >
constexpr void seriesAcosh( std::array< T, Scheme::nCoeff >& out,
                            const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    constexpr std::size_t S = Scheme::nCoeff;

    // h = sqrt(a^2 - 1)
    std::array< T, S > asq{}, h{};
    tax::cauchySelfProduct< T, Scheme >( asq, a );
    asq[0] -= T{ 1 };
    seriesSqrt< T, Scheme >( h, asq );

    seriesDerivQuotient< 1, T, Scheme >( out, cmath::ctAcosh( a[0] ), a, h );
}

/// Inverse hyperbolic tangent series `out = atanh(a)` (scheme-generic). Requires `|a[0]| < 1`:
/// out' = a' / (1 - a^2).
template < typename T, tax::IndexScheme Scheme >
constexpr void seriesAtanh( std::array< T, Scheme::nCoeff >& out,
                            const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    // h = 1 - a^2
    std::array< T, Scheme::nCoeff > h{};
    tax::cauchySelfProduct< T, Scheme >( h, a );
    for ( T& v : h ) v = -v;
    h[0] += T{ 1 };

    seriesDerivQuotient< 1, T, Scheme >( out, cmath::ctAtanh( a[0] ), a, h );
}

/// Error function series `out = erf(a)` (scheme-generic):
/// out' = a' * (2/sqrt(pi)) exp(-a^2).
template < typename T, tax::IndexScheme Scheme >
constexpr void seriesErf( std::array< T, Scheme::nCoeff >& out,
                          const std::array< T, Scheme::nCoeff >& a ) noexcept
{
    constexpr std::size_t S = Scheme::nCoeff;
    // Name the constant in the underlying real scalar so vectorised coefficient
    // types (whose lanes are floating-point) work too; broadcast into T.
    using R = real_scalar_t< T >;
    const T two_over_sqrtpi = T( R{ 2 } * std::numbers::inv_sqrtpi_v< R > );

    // h = (2/sqrt(pi)) * exp(-a^2)
    std::array< T, S > asq{}, h{};
    tax::cauchySelfProduct< T, Scheme >( asq, a );
    for ( T& v : asq ) v = -v;
    seriesExp< T, Scheme >( h, asq );
    for ( T& v : h ) v *= two_over_sqrtpi;

    seriesDerivProduct< T, Scheme >( out, cmath::ctErf( a[0] ), a, h );
}

}  // namespace tax::detail::kernels
