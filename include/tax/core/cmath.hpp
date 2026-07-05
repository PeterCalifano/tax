#pragma once

// ---------------------------------------------------------------------------
// Constexpr scalar math (tax::detail::cmath).
// ---------------------------------------------------------------------------
//
// Every series kernel evaluates exactly one transcendental at the constant
// term (out[0] = exp(a[0]), ...). <cmath> is not constexpr in C++23, so this
// header supplies the constant-evaluation path: each ct* dispatcher is an
// ordinary runtime forwarder to std:: / ADL (float, double and custom
// scalar-like coefficient types behave exactly as before), but inside
// `if consteval` it switches to a constexpr implementation computed in
// `long double`.
//
// Accuracy: the wide intermediate precision absorbs the truncation error of
// the internal series, so compile-time results agree with the runtime libm
// to within a few ulp of double (usually the last ulp or exactly). They are
// NOT guaranteed bit-identical to the runtime value — an expansion built in
// a constant expression may differ from the same expansion built at runtime
// in the trailing ulp of each coefficient. Trigonometric argument reduction
// is plain extended precision (no Payne–Hanek): constant terms of huge
// magnitude (|x| >~ 1e15) lose accuracy, and |x| >= 2^62 returns NaN.
// ---------------------------------------------------------------------------

#include <cmath>
#include <concepts>
#include <limits>
#include <numbers>

namespace tax::detail::cmath
{

namespace impl
{

using Ld = long double;

inline constexpr Ld kNaN = std::numeric_limits< Ld >::quiet_NaN();
inline constexpr Ld kInf = std::numeric_limits< Ld >::infinity();
inline constexpr Ld kPi = std::numbers::pi_v< Ld >;
inline constexpr Ld kLn2 = std::numbers::ln2_v< Ld >;
/// Relative series cutoff: comfortably below double precision, above Ld noise.
inline constexpr Ld kEps = 1e-25L;

[[nodiscard]] constexpr bool isNan( Ld x ) noexcept { return x != x; }
[[nodiscard]] constexpr Ld fabs( Ld x ) noexcept { return x < 0 ? -x : x; }

/// `x * 2^k` via exact power-of-two multiplies.
[[nodiscard]] constexpr Ld scale2( Ld x, long long k ) noexcept
{
    while ( k >= 60 )
    {
        x *= 0x1p60L;
        k -= 60;
    }
    while ( k <= -60 )
    {
        x *= 0x1p-60L;
        k += 60;
    }
    Ld p = 1;
    for ( long long i = 0; i < ( k < 0 ? -k : k ); ++i ) p *= 2;
    return k < 0 ? x / p : x * p;
}

/// Nearest integer for |x| < 2^62 (ties away from zero).
[[nodiscard]] constexpr long long llnear( Ld x ) noexcept
{
    return static_cast< long long >( x + ( x >= 0 ? Ld{ 0.5 } : Ld{ -0.5 } ) );
}

[[nodiscard]] constexpr Ld sqrt( Ld x ) noexcept
{
    if ( isNan( x ) || x < 0 ) return kNaN;
    if ( x == 0 || x == kInf ) return x;
    // Normalise x = m * 4^e with m in [1, 4): sqrt(x) = sqrt(m) * 2^e.
    long long e = 0;
    Ld m = x;
    while ( m >= 4 )
    {
        m *= 0.25L;
        ++e;
    }
    while ( m < 1 )
    {
        m *= 4;
        --e;
    }
    Ld g = ( 1 + m ) * 0.5L;
    for ( int i = 0; i < 8; ++i ) g = 0.5L * ( g + m / g );  // quadratic convergence
    return scale2( g, e );
}

[[nodiscard]] constexpr Ld cbrt( Ld x ) noexcept
{
    if ( isNan( x ) || x == 0 || x == kInf || x == -kInf ) return x;
    const Ld s = x < 0 ? Ld{ -1 } : Ld{ 1 };
    // Normalise |x| = m * 8^e with m in [1, 8): cbrt = cbrt(m) * 2^e.
    long long e = 0;
    Ld m = fabs( x );
    while ( m >= 8 )
    {
        m *= 0.125L;
        ++e;
    }
    while ( m < 1 )
    {
        m *= 8;
        --e;
    }
    Ld g = ( 2 + m ) / 3;
    for ( int i = 0; i < 9; ++i ) g = ( 2 * g + m / ( g * g ) ) / 3;
    return s * scale2( g, e );
}

[[nodiscard]] constexpr Ld exp( Ld x ) noexcept
{
    if ( isNan( x ) ) return x;
    if ( x > 12000 ) return kInf;  // beyond long double overflow
    if ( x < -12000 ) return 0;
    // x = k*ln2 + r, |r| <= ln2/2; exp(x) = 2^k * exp(r).
    const long long k = llnear( x / kLn2 );
    const Ld r = x - Ld( k ) * kLn2;
    Ld term = 1, sum = 1;
    for ( int i = 1; i < 64; ++i )
    {
        term *= r / i;
        sum += term;
        if ( fabs( term ) < kEps * fabs( sum ) ) break;
    }
    return scale2( sum, k );
}

[[nodiscard]] constexpr Ld log( Ld x ) noexcept
{
    if ( isNan( x ) || x < 0 ) return kNaN;
    if ( x == 0 ) return -kInf;
    if ( x == kInf ) return kInf;
    // Normalise x = m * 2^e with m in [sqrt(1/2), sqrt(2)).
    long long e = 0;
    Ld m = x;
    while ( m >= 2 )
    {
        m *= 0.5L;
        ++e;
    }
    while ( m < 1 )
    {
        m *= 2;
        --e;
    }
    if ( m > std::numbers::sqrt2_v< Ld > )
    {
        m *= 0.5L;
        ++e;
    }
    // log(m) = 2*atanh(u), u = (m-1)/(m+1), |u| <= 0.1716.
    const Ld u = ( m - 1 ) / ( m + 1 );
    const Ld u2 = u * u;
    Ld term = u, sum = u;
    for ( int i = 3; i < 64; i += 2 )
    {
        term *= u2;
        sum += term / i;
        if ( fabs( term ) < kEps * fabs( sum ) ) break;
    }
    return 2 * sum + Ld( e ) * kLn2;
}

struct SinCos
{
    Ld s, c;
};

[[nodiscard]] constexpr SinCos sinCos( Ld x ) noexcept
{
    if ( isNan( x ) || x == kInf || x == -kInf ) return { kNaN, kNaN };
    const Ld half_pi = kPi / 2;
    if ( fabs( x ) >= 0x1p62L * half_pi ) return { kNaN, kNaN };  // reduction hopeless
    // x = k*(pi/2) + r, |r| <= pi/4.
    const long long k = llnear( x / half_pi );
    const Ld r = x - Ld( k ) * half_pi;
    const Ld r2 = r * r;
    Ld st = r, s = r;  // sin(r)
    Ld ct = 1, c = 1;  // cos(r)
    for ( int i = 1; i < 32; ++i )
    {
        ct *= -r2 / ( ( 2 * i - 1 ) * ( 2 * i ) );
        st *= -r2 / ( ( 2 * i ) * ( 2 * i + 1 ) );
        c += ct;
        s += st;
        if ( fabs( st ) < kEps && fabs( ct ) < kEps ) break;
    }
    switch ( ( ( k % 4 ) + 4 ) % 4 )
    {
        case 0:
            return { s, c };
        case 1:
            return { c, -s };
        case 2:
            return { -s, -c };
        default:
            return { -c, s };
    }
}

[[nodiscard]] constexpr Ld atan( Ld x ) noexcept
{
    if ( isNan( x ) ) return x;
    const Ld s = x < 0 ? Ld{ -1 } : Ld{ 1 };
    Ld a = fabs( x );
    if ( a == kInf ) return s * kPi / 2;
    const bool inv = a > 1;
    if ( inv ) a = 1 / a;
    // Two half-angle steps: atan(a) = 2*atan(a / (1 + sqrt(1 + a^2))).
    a = a / ( 1 + sqrt( 1 + a * a ) );
    a = a / ( 1 + sqrt( 1 + a * a ) );  // now a <= tan(pi/16) ~ 0.199
    const Ld a2 = a * a;
    Ld term = a, sum = a;
    for ( int i = 3; i < 64; i += 2 )
    {
        term *= -a2;
        sum += term / i;
        if ( fabs( term ) < kEps * fabs( sum ) ) break;
    }
    Ld r = 4 * sum;
    if ( inv ) r = kPi / 2 - r;
    return s * r;
}

[[nodiscard]] constexpr Ld asin( Ld x ) noexcept
{
    if ( isNan( x ) || x > 1 || x < -1 ) return kNaN;
    if ( x == 1 ) return kPi / 2;
    if ( x == -1 ) return -kPi / 2;
    return atan( x / sqrt( 1 - x * x ) );
}

[[nodiscard]] constexpr Ld acos( Ld x ) noexcept
{
    if ( isNan( x ) || x > 1 || x < -1 ) return kNaN;
    return kPi / 2 - asin( x );
}

[[nodiscard]] constexpr Ld atan2( Ld y, Ld x ) noexcept
{
    if ( isNan( y ) || isNan( x ) ) return kNaN;
    if ( x > 0 ) return atan( y / x );
    if ( x < 0 ) return y < 0 ? atan( y / x ) - kPi : atan( y / x ) + kPi;
    // x == 0 (constant-evaluation convention: zero signs are not observable)
    if ( y > 0 ) return kPi / 2;
    if ( y < 0 ) return -kPi / 2;
    return 0;
}

[[nodiscard]] constexpr Ld sinh( Ld x ) noexcept
{
    if ( isNan( x ) || x == kInf || x == -kInf ) return x;
    if ( fabs( x ) < 0.5L )
    {
        // Direct series: avoids the (e^x - e^-x) cancellation near zero.
        const Ld x2 = x * x;
        Ld term = x, sum = x;
        for ( int i = 1; i < 32; ++i )
        {
            term *= x2 / ( ( 2 * i ) * ( 2 * i + 1 ) );
            sum += term;
            if ( fabs( term ) < kEps * fabs( sum ) ) break;
        }
        return sum;
    }
    const Ld e = exp( x );
    return ( e - 1 / e ) * 0.5L;
}

[[nodiscard]] constexpr Ld cosh( Ld x ) noexcept
{
    if ( isNan( x ) ) return x;
    const Ld e = exp( fabs( x ) );
    return ( e + 1 / e ) * 0.5L;
}

[[nodiscard]] constexpr Ld tanh( Ld x ) noexcept
{
    if ( isNan( x ) ) return x;
    if ( x > 60 ) return 1;
    if ( x < -60 ) return -1;
    if ( fabs( x ) < 0.5L ) return sinh( x ) / cosh( x );
    const Ld e2 = exp( 2 * x );
    return ( e2 - 1 ) / ( e2 + 1 );
}

[[nodiscard]] constexpr Ld asinh( Ld x ) noexcept
{
    if ( isNan( x ) || x == kInf || x == -kInf ) return x;
    const Ld s = x < 0 ? Ld{ -1 } : Ld{ 1 };
    const Ld a = fabs( x );
    if ( a < 0.25L )
    {
        // Direct series: avoids log(1 + tiny) cancellation near zero.
        // asinh(x) = sum (-1)^k (2k)! x^(2k+1) / (4^k (k!)^2 (2k+1)).
        const Ld x2 = a * a;
        Ld term = a, sum = a;
        for ( int k = 1; k < 48; ++k )
        {
            term *= -x2 * ( 2 * k - 1 ) / ( 2 * k );
            sum += term / ( 2 * k + 1 );
            if ( fabs( term ) < kEps * fabs( sum ) ) break;
        }
        return s * sum;
    }
    return s * log( a + sqrt( a * a + 1 ) );
}

[[nodiscard]] constexpr Ld acosh( Ld x ) noexcept
{
    if ( isNan( x ) || x < 1 ) return kNaN;
    return log( x + sqrt( x * x - 1 ) );
}

[[nodiscard]] constexpr Ld atanh( Ld x ) noexcept
{
    if ( isNan( x ) || x > 1 || x < -1 ) return kNaN;
    if ( x == 1 ) return kInf;
    if ( x == -1 ) return -kInf;
    if ( fabs( x ) < 0.25L )
    {
        const Ld x2 = x * x;
        Ld term = x, sum = x;
        for ( int i = 3; i < 64; i += 2 )
        {
            term *= x2;
            sum += term / i;
            if ( fabs( term ) < kEps * fabs( sum ) ) break;
        }
        return sum;
    }
    return 0.5L * log( ( 1 + x ) / ( 1 - x ) );
}

[[nodiscard]] constexpr Ld erf( Ld x ) noexcept
{
    if ( isNan( x ) ) return x;
    const Ld s = x < 0 ? Ld{ -1 } : Ld{ 1 };
    const Ld a = fabs( x );
    if ( a > 9 ) return s;  // 1 - erf(9) < 1e-36
    // All-positive-term form (no cancellation, converges for all x):
    // erf(x) = (2/sqrt(pi)) e^{-x^2} sum_{n>=0} 2^n x^{2n+1} / (2n+1)!!.
    const Ld x2 = 2 * a * a;
    Ld term = a, sum = a;
    for ( int n = 0; n < 512; ++n )
    {
        term *= x2 / ( 2 * n + 3 );
        sum += term;
        if ( term < kEps * sum ) break;
    }
    return s * 2 * std::numbers::inv_sqrtpi_v< Ld > * exp( -a * a ) * sum;
}

[[nodiscard]] constexpr Ld pow( Ld x, Ld y ) noexcept
{
    if ( isNan( x ) || isNan( y ) ) return kNaN;
    if ( y == 0 ) return 1;
    if ( x == 0 ) return y > 0 ? Ld{ 0 } : kInf;
    if ( x > 0 ) return exp( y * log( x ) );
    // Negative base: defined only for integral exponents.
    if ( fabs( y ) < 0x1p62L && Ld( llnear( y ) ) == y )
    {
        const long long n = llnear( y );
        const Ld m = exp( y * log( -x ) );
        return ( n % 2 != 0 ) ? -m : m;
    }
    return kNaN;
}

}  // namespace impl

// ---------------------------------------------------------------------------
// Dispatchers: runtime -> std:: / ADL, constant evaluation -> impl. The
// `if consteval` path exists only for floating-point coefficient types;
// non-arithmetic scalar-like coefficient types (found via ADL) always take
// the runtime call.
// ---------------------------------------------------------------------------

#define TAX_CT_UNARY( CT_NAME, STD_NAME )                                   \
    template < typename T >                                                 \
    [[nodiscard]] constexpr T CT_NAME( const T& x ) noexcept                \
    {                                                                       \
        if constexpr ( std::floating_point< T > )                           \
        {                                                                   \
            if consteval                                                    \
            {                                                               \
                return T( impl::STD_NAME( static_cast< impl::Ld >( x ) ) ); \
            }                                                               \
        }                                                                   \
        using std::STD_NAME;                                                \
        return STD_NAME( x );                                               \
    }

TAX_CT_UNARY( ctSqrt, sqrt )
TAX_CT_UNARY( ctCbrt, cbrt )
TAX_CT_UNARY( ctExp, exp )
TAX_CT_UNARY( ctLog, log )
TAX_CT_UNARY( ctAtan, atan )
TAX_CT_UNARY( ctAsin, asin )
TAX_CT_UNARY( ctAcos, acos )
TAX_CT_UNARY( ctSinh, sinh )
TAX_CT_UNARY( ctCosh, cosh )
TAX_CT_UNARY( ctTanh, tanh )
TAX_CT_UNARY( ctAsinh, asinh )
TAX_CT_UNARY( ctAcosh, acosh )
TAX_CT_UNARY( ctAtanh, atanh )
TAX_CT_UNARY( ctErf, erf )

#undef TAX_CT_UNARY

template < typename T >
[[nodiscard]] constexpr T ctSin( const T& x ) noexcept
{
    if constexpr ( std::floating_point< T > )
    {
        if consteval
        {
            return T( impl::sinCos( static_cast< impl::Ld >( x ) ).s );
        }
    }
    using std::sin;
    return sin( x );
}

template < typename T >
[[nodiscard]] constexpr T ctCos( const T& x ) noexcept
{
    if constexpr ( std::floating_point< T > )
    {
        if consteval
        {
            return T( impl::sinCos( static_cast< impl::Ld >( x ) ).c );
        }
    }
    using std::cos;
    return cos( x );
}

template < typename T >
[[nodiscard]] constexpr T ctAtan2( const T& y, const T& x ) noexcept
{
    if constexpr ( std::floating_point< T > )
    {
        if consteval
        {
            return T( impl::atan2( static_cast< impl::Ld >( y ), static_cast< impl::Ld >( x ) ) );
        }
    }
    using std::atan2;
    return atan2( y, x );
}

template < typename T >
[[nodiscard]] constexpr T ctPow( const T& x, const T& y ) noexcept
{
    if constexpr ( std::floating_point< T > )
    {
        if consteval
        {
            return T( impl::pow( static_cast< impl::Ld >( x ), static_cast< impl::Ld >( y ) ) );
        }
    }
    using std::pow;
    return pow( x, y );
}

}  // namespace tax::detail::cmath
