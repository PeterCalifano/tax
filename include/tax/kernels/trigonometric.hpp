#pragma once

#include <cmath>
#include <span>
#include <utility>
#include <vector>

#include <tax/kernels/cauchy_stencil.hpp>
#include <tax/kernels/unroll.hpp>
#include <tax/utils/enumeration.hpp>

namespace tax::detail
{

template < typename T, int N, int M >
/**
 * @brief Coupled trigonometric series expansion of `sin(a)` and `cos(a)`.
 * @details Computes both outputs together to share recurrence work.
 */
constexpr void seriesSinCos( Coeffs< T, N, M >& s,
                             Coeffs< T, N, M >& c,
                             const Coeffs< T, N, M >& a ) noexcept
{
    using std::cos;
    using std::sin;
    s = {};
    c = {};
    s[0] = sin( a[0] );
    c[0] = cos( a[0] );

    if constexpr ( M == 1 )
    {
        sinCosUniImpl< T, N >( s, c, a, std::make_index_sequence< N >{} );
    } else
    {
        using S = CauchyStencil< N, M >;
        using W = CauchyWeightStencil< N, M >;
        using D = DegreeRanges< N, M >;
        for ( int d = 1; d <= N; ++d )
        {
            const T inv_d = T{ 1 } / T( d );
            for ( std::size_t k = D::endByDegree[d]; k < D::endByDegree[d + 1]; ++k )
            {
                T sin_rhs{ 0 }, cos_rhs{ 0 };
                // Skip the last pair (db = d, beta = alpha); db spans [0, d-1].
                for ( std::size_t j = S::offsets[k]; j + 1 < S::offsets[k + 1]; ++j )
                {
                    const T fg = T( d - W::db[j] ) * a[S::col_b[j]];
                    sin_rhs += fg * c[S::col_a[j]];
                    cos_rhs += fg * s[S::col_a[j]];
                }
                s[k] = sin_rhs * inv_d;
                c[k] = -cos_rhs * inv_d;
            }
        }
    }
}

template < typename T, int N, int M >
/// @brief Sine series wrapper around `seriesSinCos`.
constexpr void seriesSin( Coeffs< T, N, M >& out,
                          const Coeffs< T, N, M >& a ) noexcept
{
    Coeffs< T, N, M > c{};
    seriesSinCos< T, N, M >( out, c, a );
}

template < typename T, int N, int M >
/// @brief Cosine series wrapper around `seriesSinCos`.
constexpr void seriesCos( Coeffs< T, N, M >& out,
                          const Coeffs< T, N, M >& a ) noexcept
{
    Coeffs< T, N, M > s{};
    seriesSinCos< T, N, M >( s, out, a );
}

template < typename T, int N, int M >
/**
 * @brief Tangent series solve from `sin(a)` and `cos(a)`.
 * @details Solves `cos(a) * out = sin(a)` degree by degree.
 */
constexpr void seriesTan( Coeffs< T, N, M >& out,
                          const Coeffs< T, N, M >& a ) noexcept
{
    constexpr auto S = numMonomials( N, M );
    std::array< T, S > s{}, c{};
    seriesSinCos< T, N, M >( s, c, a );

    // Solve c · out = s  degree by degree (same structure as seriesReciprocal)
    out = {};
    const T inv_c0 = T{ 1 } / c[0];

    if constexpr ( M == 1 )
    {
        out[0] = s[0] * inv_c0;
        tanLikeUniImpl< T, N >( out, s, c, inv_c0, std::make_index_sequence< N >{} );
    } else
    {
        using S = CauchyStencil< N, M >;
        out[0] = s[0] * inv_c0;
        for ( std::size_t k = 1; k < S::NC; ++k )
        {
            T rhs = s[k];
            // Skip the (beta=0, gamma=alpha) pair (encodes the c[0] * out[k] LHS term).
            for ( std::size_t j = S::offsets[k] + 1; j < S::offsets[k + 1]; ++j )
                rhs -= c[S::col_a[j]] * out[S::col_b[j]];
            out[k] = rhs * inv_c0;
        }
    }
}

template < typename T, int N, int M >
constexpr void seriesSinhCosh( Coeffs< T, N, M >& sh,
                               Coeffs< T, N, M >& ch,
                               const Coeffs< T, N, M >& a ) noexcept
{
    using std::cosh;
    using std::sinh;
    sh = {};
    ch = {};
    sh[0] = sinh( a[0] );
    ch[0] = cosh( a[0] );

    if constexpr ( M == 1 )
    {
        sinhCoshUniImpl< T, N >( sh, ch, a, std::make_index_sequence< N >{} );
    } else
    {
        using S = CauchyStencil< N, M >;
        using W = CauchyWeightStencil< N, M >;
        using D = DegreeRanges< N, M >;
        for ( int d = 1; d <= N; ++d )
        {
            const T inv_d = T{ 1 } / T( d );
            for ( std::size_t k = D::endByDegree[d]; k < D::endByDegree[d + 1]; ++k )
            {
                T sh_rhs{ 0 }, ch_rhs{ 0 };
                for ( std::size_t j = S::offsets[k]; j + 1 < S::offsets[k + 1]; ++j )
                {
                    const T fg = T( d - W::db[j] ) * a[S::col_b[j]];
                    sh_rhs += fg * ch[S::col_a[j]];
                    ch_rhs += fg * sh[S::col_a[j]];
                }
                sh[k] = sh_rhs * inv_d;
                ch[k] = ch_rhs * inv_d;
            }
        }
    }
}

template < typename T, int N, int M >
constexpr void seriesSinh( Coeffs< T, N, M >& out,
                           const Coeffs< T, N, M >& a ) noexcept
{
    Coeffs< T, N, M > ch{};
    seriesSinhCosh< T, N, M >( out, ch, a );
}

template < typename T, int N, int M >
constexpr void seriesCosh( Coeffs< T, N, M >& out,
                           const Coeffs< T, N, M >& a ) noexcept
{
    Coeffs< T, N, M > sh{};
    seriesSinhCosh< T, N, M >( sh, out, a );
}

template < typename T, int N, int M >
constexpr void seriesTanh( Coeffs< T, N, M >& out,
                           const Coeffs< T, N, M >& a ) noexcept
{
    constexpr auto S = numMonomials( N, M );
    std::array< T, S > sh{}, ch{};
    seriesSinhCosh< T, N, M >( sh, ch, a );

    out = {};
    const T inv_ch0 = T{ 1 } / ch[0];

    if constexpr ( M == 1 )
    {
        out[0] = sh[0] * inv_ch0;
        tanLikeUniImpl< T, N >( out, sh, ch, inv_ch0, std::make_index_sequence< N >{} );
    } else
    {
        using S = CauchyStencil< N, M >;
        out[0] = sh[0] * inv_ch0;
        for ( std::size_t k = 1; k < S::NC; ++k )
        {
            T rhs = sh[k];
            for ( std::size_t j = S::offsets[k] + 1; j < S::offsets[k + 1]; ++j )
                rhs -= ch[S::col_a[j]] * out[S::col_b[j]];
            out[k] = rhs * inv_ch0;
        }
    }
}

// =============================================================================
// Runtime-shape variants (used by the dynamic-shape `TaylorExpansionT`).
// =============================================================================

/// @brief Runtime overload of `seriesSinCos`: jointly compute `sin(a)` and `cos(a)`.
template < typename T >
inline void seriesSinCos( T* s, T* c, const T* a, std::size_t N, std::size_t M ) noexcept
{
    using std::cos;
    using std::sin;
    const std::size_t S = numMonomials( N, M );
    for ( std::size_t i = 0; i < S; ++i ) s[i] = T{ 0 };
    for ( std::size_t i = 0; i < S; ++i ) c[i] = T{ 0 };
    s[0] = sin( a[0] );
    c[0] = cos( a[0] );

    if ( M == 1 )
    {
        for ( std::size_t d = 1; d <= N; ++d )
        {
            T sr = T{ 0 }, cr = T{ 0 };
            for ( std::size_t k = 0; k < d; ++k )
            {
                const T w = T( d - k ) * a[d - k];
                sr += w * c[k];
                cr += w * s[k];
            }
            const T inv_d = T{ 1 } / T( d );
            s[d] = sr * inv_d;
            c[d] = -cr * inv_d;
        }
    }
    else
    {
        for ( int d = 1; d <= int( N ); ++d )
        {
            forEachMonomial( int( M ), d, [&]( std::span< const int > alpha, std::size_t ai ) {
                T sin_rhs = T{ 0 };
                T cos_rhs = T{ 0 };
                forEachSubIndex( alpha, 0, d - 1, [&]( std::size_t bi, std::size_t gi, int db ) {
                    const T fg = T( d - db ) * a[gi];
                    sin_rhs += fg * c[bi];
                    cos_rhs += fg * s[bi];
                } );
                const T inv_d = T{ 1 } / T( d );
                s[ai] = sin_rhs * inv_d;
                c[ai] = -cos_rhs * inv_d;
            } );
        }
    }
}

/// @brief Runtime sin: thin wrapper around `seriesSinCos`.
template < typename T >
inline void seriesSin( T* out, const T* a, std::size_t N, std::size_t M )
{
    const std::size_t S = numMonomials( N, M );
    std::vector< T > scratch( S, T{ 0 } );
    seriesSinCos( out, scratch.data(), a, N, M );
}

/// @brief Runtime cos: thin wrapper around `seriesSinCos`.
template < typename T >
inline void seriesCos( T* out, const T* a, std::size_t N, std::size_t M )
{
    const std::size_t S = numMonomials( N, M );
    std::vector< T > scratch( S, T{ 0 } );
    seriesSinCos( scratch.data(), out, a, N, M );
}

/// @brief Runtime overload of `seriesTan`: tan(a) via cos·out = sin.
template < typename T >
inline void seriesTan( T* out, const T* a, std::size_t N, std::size_t M )
{
    const std::size_t S = numMonomials( N, M );
    std::vector< T > s( S, T{ 0 } ), c( S, T{ 0 } );
    seriesSinCos( s.data(), c.data(), a, N, M );

    for ( std::size_t i = 0; i < S; ++i ) out[i] = T{ 0 };
    const T inv_c0 = T{ 1 } / c[0];

    if ( M == 1 )
    {
        for ( std::size_t d = 0; d <= N; ++d )
        {
            T rhs = s[d];
            for ( std::size_t k = 1; k <= d; ++k ) rhs -= c[k] * out[d - k];
            out[d] = rhs * inv_c0;
        }
    }
    else
    {
        for ( int d = 0; d <= int( N ); ++d )
        {
            forEachMonomial( int( M ), d, [&]( std::span< const int > alpha, std::size_t ai ) {
                T rhs = s[ai];
                forEachSubIndex( alpha, 1, d, [&]( std::size_t bi, std::size_t gi, int ) {
                    rhs -= c[bi] * out[gi];
                } );
                out[ai] = rhs * inv_c0;
            } );
        }
    }
}

/// @brief Runtime overload of `seriesSinhCosh`: joint sinh/cosh.
template < typename T >
inline void seriesSinhCosh( T* sh, T* ch, const T* a, std::size_t N, std::size_t M ) noexcept
{
    using std::cosh;
    using std::sinh;
    const std::size_t S = numMonomials( N, M );
    for ( std::size_t i = 0; i < S; ++i ) sh[i] = T{ 0 };
    for ( std::size_t i = 0; i < S; ++i ) ch[i] = T{ 0 };
    sh[0] = sinh( a[0] );
    ch[0] = cosh( a[0] );

    if ( M == 1 )
    {
        for ( std::size_t d = 1; d <= N; ++d )
        {
            T sr = T{ 0 }, cr = T{ 0 };
            for ( std::size_t k = 0; k < d; ++k )
            {
                const T w = T( d - k ) * a[d - k];
                sr += w * ch[k];
                cr += w * sh[k];
            }
            const T inv_d = T{ 1 } / T( d );
            sh[d] = sr * inv_d;
            ch[d] = cr * inv_d;
        }
    }
    else
    {
        for ( int d = 1; d <= int( N ); ++d )
        {
            forEachMonomial( int( M ), d, [&]( std::span< const int > alpha, std::size_t ai ) {
                T sh_rhs = T{ 0 };
                T ch_rhs = T{ 0 };
                forEachSubIndex( alpha, 0, d - 1, [&]( std::size_t bi, std::size_t gi, int db ) {
                    const T fg = T( d - db ) * a[gi];
                    sh_rhs += fg * ch[bi];
                    ch_rhs += fg * sh[bi];
                } );
                const T inv_d = T{ 1 } / T( d );
                sh[ai] = sh_rhs * inv_d;
                ch[ai] = ch_rhs * inv_d;
            } );
        }
    }
}

template < typename T >
inline void seriesSinh( T* out, const T* a, std::size_t N, std::size_t M )
{
    const std::size_t S = numMonomials( N, M );
    std::vector< T > scratch( S, T{ 0 } );
    seriesSinhCosh( out, scratch.data(), a, N, M );
}

template < typename T >
inline void seriesCosh( T* out, const T* a, std::size_t N, std::size_t M )
{
    const std::size_t S = numMonomials( N, M );
    std::vector< T > scratch( S, T{ 0 } );
    seriesSinhCosh( scratch.data(), out, a, N, M );
}

/// @brief Runtime overload of `seriesTanh`: tanh(a) via cosh·out = sinh.
template < typename T >
inline void seriesTanh( T* out, const T* a, std::size_t N, std::size_t M )
{
    const std::size_t S = numMonomials( N, M );
    std::vector< T > sh( S, T{ 0 } ), ch( S, T{ 0 } );
    seriesSinhCosh( sh.data(), ch.data(), a, N, M );

    for ( std::size_t i = 0; i < S; ++i ) out[i] = T{ 0 };
    const T inv_ch0 = T{ 1 } / ch[0];

    if ( M == 1 )
    {
        for ( std::size_t d = 0; d <= N; ++d )
        {
            T rhs = sh[d];
            for ( std::size_t k = 1; k <= d; ++k ) rhs -= ch[k] * out[d - k];
            out[d] = rhs * inv_ch0;
        }
    }
    else
    {
        for ( int d = 0; d <= int( N ); ++d )
        {
            forEachMonomial( int( M ), d, [&]( std::span< const int > alpha, std::size_t ai ) {
                T rhs = sh[ai];
                forEachSubIndex( alpha, 1, d, [&]( std::size_t bi, std::size_t gi, int ) {
                    rhs -= ch[bi] * out[gi];
                } );
                out[ai] = rhs * inv_ch0;
            } );
        }
    }
}

}  // namespace tax::detail
