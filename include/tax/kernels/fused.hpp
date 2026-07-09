#pragma once

// Pair-fused series kernels (ported from the expression-template prototype).
//
// The ET prototype benchmarks showed that its lazy-evaluation layer itself was
// a wash against eager evaluation, but that fusing an `exp * trig` pair into a
// single coupled recurrence is a robust 1.3x-1.5x win, and that a joint
// sqrt / 1/sqrt pass wins whenever BOTH outputs are consumed. Those kernels —
// and only those — live here; the ET layer was deliberately not ported.
//
//   seriesExpSinCos : h = exp(v)*cos(u), q = exp(v)*sin(u) in ONE coupled pass
//                     (h' = v'h - u'q, q' = v'q + u'h), instead of the three
//                     recurrences + Cauchy product of exp(v) * cos(u).
//   seriesSqrtInvSqrt : s = sqrt(u) and r = 1/sqrt(u) interleaved per degree —
//                     r costs one forward substitution on top of s, with scalar
//                     divisions by s0 only. A single-output caller should use
//                     seriesSqrt / seriesPow instead: computing the unused
//                     companion is a measured net loss.

#include <array>
#include <cmath>
#include <span>
#include <tax/core/scheme/isotropic.hpp>
#include <tax/kernels/algebra.hpp>

namespace tax::detail::kernels
{

/// Coupled exp-trig series: jointly compute `q = exp(v)*sin(u)` and
/// `h = exp(v)*cos(u)` in a single recurrence pass (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
void seriesExpSinCos( std::array< T, Scheme::nCoeff >& q, std::array< T, Scheme::nCoeff >& h,
                      const std::array< T, Scheme::nCoeff >& v,
                      const std::array< T, Scheme::nCoeff >& u ) noexcept
{
    using std::cos;
    using std::exp;
    using std::sin;
    q = {};
    h = {};
    const T e0 = exp( v[0] );
    q[0] = e0 * sin( u[0] );
    h[0] = e0 * cos( u[0] );

    if constexpr ( Scheme::isUnivariate )
    {
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T hr = T{ 0 }, qr = T{ 0 };
            for ( int k = 0; k < d; ++k )
            {
                const T wv = T( d - k ) * v[std::size_t( d - k )];
                const T wu = T( d - k ) * u[std::size_t( d - k )];
                hr += wv * h[std::size_t( k )] - wu * q[std::size_t( k )];
                qr += wv * q[std::size_t( k )] + wu * h[std::size_t( k )];
            }
            const T inv_d = T{ 1 } / T( d );
            h[std::size_t( d )] = hr * inv_d;
            q[std::size_t( d )] = qr * inv_d;
        }
    } else
    {
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int d, std::span< const RecurrenceEntry > row ) {
                T hr = T{ 0 }, qr = T{ 0 };
                for ( const RecurrenceEntry& e : row )
                {
                    const T wv = T( e.db ) * v[e.b_idx];
                    const T wu = T( e.db ) * u[e.b_idx];
                    hr += wv * h[e.g_idx] - wu * q[e.g_idx];
                    qr += wv * q[e.g_idx] + wu * h[e.g_idx];
                }
                const T inv_d = T{ 1 } / T( d );
                h[ai] = hr * inv_d;
                q[ai] = qr * inv_d;
            } );
    }
}

/// Fused `out = exp(v) * sin(u)` (the cos companion stays kernel-internal).
template < typename T, tax::IndexScheme Scheme >
void seriesExpSin( std::array< T, Scheme::nCoeff >& out, const std::array< T, Scheme::nCoeff >& v,
                   const std::array< T, Scheme::nCoeff >& u ) noexcept
{
    std::array< T, Scheme::nCoeff > h{};
    seriesExpSinCos< T, Scheme >( out, h, v, u );
}

/// Fused `out = exp(v) * cos(u)` (the sin companion stays kernel-internal).
template < typename T, tax::IndexScheme Scheme >
void seriesExpCos( std::array< T, Scheme::nCoeff >& out, const std::array< T, Scheme::nCoeff >& v,
                   const std::array< T, Scheme::nCoeff >& u ) noexcept
{
    std::array< T, Scheme::nCoeff > q{};
    seriesExpSinCos< T, Scheme >( q, out, v, u );
}

/// Joint `s = sqrt(u)`, `r = 1/sqrt(u)`, interleaved per degree: s from s^2 = u
/// (mirrors seriesSqrt, including the beta = alpha zero-read trick), r from
/// r*s = 1 with the explicit r0*s_alpha term (rows exclude beta = 0). Scalar
/// divisions by s0 only. Requires `u[0] > 0` (scheme-generic).
template < typename T, tax::IndexScheme Scheme >
void seriesSqrtInvSqrt( std::array< T, Scheme::nCoeff >& s, std::array< T, Scheme::nCoeff >& r,
                        const std::array< T, Scheme::nCoeff >& u ) noexcept
{
    using std::sqrt;
    s = {};
    r = {};
    s[0] = sqrt( u[0] );
    const T inv_s0 = T{ 1 } / s[0];
    const T inv_2s0 = T{ 1 } / ( T{ 2 } * s[0] );
    r[0] = inv_s0;

    if constexpr ( Scheme::isUnivariate )
    {
        for ( int d = 1; d <= Scheme::order; ++d )
        {
            T srhs = u[std::size_t( d )];
            for ( int k = 1; k + k < d; ++k )
                srhs -= T{ 2 } * s[std::size_t( k )] * s[std::size_t( d - k )];
            if ( d % 2 == 0 ) srhs -= s[std::size_t( d / 2 )] * s[std::size_t( d / 2 )];
            s[std::size_t( d )] = srhs * inv_2s0;

            T rrhs = T{ 0 };
            for ( int k = 0; k < d; ++k ) rrhs -= r[std::size_t( k )] * s[std::size_t( d - k )];
            r[std::size_t( d )] = rrhs * inv_s0;
        }
    } else
    {
        Scheme::forEachRecurrenceRow(
            [&]( std::size_t ai, int, std::span< const RecurrenceEntry > row ) {
                // s: same zero-read trick as seriesSqrt — the b = alpha entry reads s[ai] == 0.
                T srhs = u[ai];
                for ( const RecurrenceEntry& e : row ) srhs -= s[e.b_idx] * s[e.g_idx];
                s[ai] = srhs * inv_2s0;
                // r: rows cover |beta| >= 1 (the b = alpha entry reads r[ai] == 0); the
                // missing beta = 0 term is r0*s[ai], added explicitly AFTER s[ai] is final.
                T rrhs = -r[0] * s[ai];
                for ( const RecurrenceEntry& e : row ) rrhs -= r[e.b_idx] * s[e.g_idx];
                r[ai] = rrhs * inv_s0;
            } );
    }
}

}  // namespace tax::detail::kernels
