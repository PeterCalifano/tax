#pragma once

#include <type_traits>

#include <tax/core/taylor_expansion.hpp>
#include <tax/kernels/algebra.hpp>
#include <tax/kernels/cauchy.hpp>
#include <tax/kernels/sparse_cauchy.hpp>
#include <tax/kernels/sparse_subs.hpp>

namespace tax
{

// ---------------------------------------------------------------------------
// Addition
// ---------------------------------------------------------------------------

template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > operator+(
    const TaylorExpansion< T, N, M >& a,
    const TaylorExpansion< T, N, M >& b ) noexcept
{
    TaylorExpansion< T, N, M > r;
    for ( std::size_t k = 0; k < a.nCoefficients; ++k )
        r[k] = a[k] + b[k];
    return r;
}

template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > operator+(
    const TaylorExpansion< T, N, M >& a, std::type_identity_t< T > s ) noexcept
{
    TaylorExpansion< T, N, M > r = a;
    r[0] += s;
    return r;
}

template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > operator+(
    std::type_identity_t< T > s, const TaylorExpansion< T, N, M >& a ) noexcept
{
    return a + s;
}

// ---------------------------------------------------------------------------
// Subtraction
// ---------------------------------------------------------------------------

template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > operator-(
    const TaylorExpansion< T, N, M >& a,
    const TaylorExpansion< T, N, M >& b ) noexcept
{
    TaylorExpansion< T, N, M > r;
    for ( std::size_t k = 0; k < a.nCoefficients; ++k )
        r[k] = a[k] - b[k];
    return r;
}

template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > operator-(
    const TaylorExpansion< T, N, M >& a, std::type_identity_t< T > s ) noexcept
{
    TaylorExpansion< T, N, M > r = a;
    r[0] -= s;
    return r;
}

template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > operator-(
    std::type_identity_t< T > s, const TaylorExpansion< T, N, M >& a ) noexcept
{
    TaylorExpansion< T, N, M > r;
    r[0] = s - a[0];
    for ( std::size_t k = 1; k < a.nCoefficients; ++k ) r[k] = -a[k];
    return r;
}

// ---------------------------------------------------------------------------
// Unary negation
// ---------------------------------------------------------------------------

template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > operator-(
    const TaylorExpansion< T, N, M >& a ) noexcept
{
    TaylorExpansion< T, N, M > r;
    for ( std::size_t k = 0; k < a.nCoefficients; ++k )
        r[k] = -a[k];
    return r;
}

// ---------------------------------------------------------------------------
// Scalar multiplication / division
// ---------------------------------------------------------------------------

template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > operator*(
    const TaylorExpansion< T, N, M >& a, std::type_identity_t< T > s ) noexcept
{
    TaylorExpansion< T, N, M > r;
    for ( std::size_t k = 0; k < a.nCoefficients; ++k )
        r[k] = a[k] * s;
    return r;
}

template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > operator*(
    std::type_identity_t< T > s, const TaylorExpansion< T, N, M >& a ) noexcept
{
    return a * s;
}

template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > operator/(
    const TaylorExpansion< T, N, M >& a, std::type_identity_t< T > s ) noexcept
{
    return a * ( T( 1 ) / s );
}

// ---------------------------------------------------------------------------
// Cauchy (TE x TE) multiplication
// ---------------------------------------------------------------------------

template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > operator*(
    const TaylorExpansion< T, N, M >& a,
    const TaylorExpansion< T, N, M >& b ) noexcept
{
    TaylorExpansion< T, N, M > r;
    detail::kernels::cauchyProduct< T, N, M >(
        r.coefficients(), a.coefficients(), b.coefficients() );
    return r;
}

// ---------------------------------------------------------------------------
// TE / TE division via reciprocal
// ---------------------------------------------------------------------------

template < typename T, int N, int M >
[[nodiscard]] constexpr TaylorExpansion< T, N, M > operator/(
    const TaylorExpansion< T, N, M >& a,
    const TaylorExpansion< T, N, M >& b ) noexcept
{
    TaylorExpansion< T, N, M > inv_b;
    detail::kernels::seriesReciprocal< T, N, M >( inv_b.coefficients(), b.coefficients() );
    TaylorExpansion< T, N, M > r;
    detail::kernels::cauchyProduct< T, N, M >( r.coefficients(), a.coefficients(),
                                               inv_b.coefficients() );
    return r;
}

// ---------------------------------------------------------------------------
// Compound assignment (dense)
//
// In-place updates avoid the temporary + copy-assign of `a = a + b`; they are
// the building blocks of hot loops (e.g. the ODE steppers' axpy updates).
// ---------------------------------------------------------------------------

template < typename T, int N, int M >
constexpr TaylorExpansion< T, N, M >& operator+=(
    TaylorExpansion< T, N, M >& a, const TaylorExpansion< T, N, M >& b ) noexcept
{
    for ( std::size_t k = 0; k < a.nCoefficients; ++k )
        a[k] += b[k];
    return a;
}

template < typename T, int N, int M >
constexpr TaylorExpansion< T, N, M >& operator-=(
    TaylorExpansion< T, N, M >& a, const TaylorExpansion< T, N, M >& b ) noexcept
{
    for ( std::size_t k = 0; k < a.nCoefficients; ++k )
        a[k] -= b[k];
    return a;
}

template < typename T, int N, int M >
constexpr TaylorExpansion< T, N, M >& operator+=(
    TaylorExpansion< T, N, M >& a, std::type_identity_t< T > s ) noexcept
{
    a[0] += s;
    return a;
}

template < typename T, int N, int M >
constexpr TaylorExpansion< T, N, M >& operator-=(
    TaylorExpansion< T, N, M >& a, std::type_identity_t< T > s ) noexcept
{
    a[0] -= s;
    return a;
}

template < typename T, int N, int M >
constexpr TaylorExpansion< T, N, M >& operator*=(
    TaylorExpansion< T, N, M >& a, std::type_identity_t< T > s ) noexcept
{
    for ( std::size_t k = 0; k < a.nCoefficients; ++k )
        a[k] *= s;
    return a;
}

template < typename T, int N, int M >
constexpr TaylorExpansion< T, N, M >& operator/=(
    TaylorExpansion< T, N, M >& a, std::type_identity_t< T > s ) noexcept
{
    return a *= ( T( 1 ) / s );
}

/// @brief In-place Cauchy product. A scratch buffer is unavoidable (the
///        convolution reads earlier coefficients of `a`), but the temporary
///        TaylorExpansion of `a = a * b` is not.
template < typename T, int N, int M >
constexpr TaylorExpansion< T, N, M >& operator*=(
    TaylorExpansion< T, N, M >& a, const TaylorExpansion< T, N, M >& b ) noexcept
{
    Coeffs< T, N, M > tmp{};
    detail::kernels::cauchyProduct< T, N, M >( tmp, a.coefficients(), b.coefficients() );
    a.coefficients() = tmp;
    return a;
}

template < typename T, int N, int M >
constexpr TaylorExpansion< T, N, M >& operator/=(
    TaylorExpansion< T, N, M >& a, const TaylorExpansion< T, N, M >& b ) noexcept
{
    Coeffs< T, N, M > inv_b{};
    detail::kernels::seriesReciprocal< T, N, M >( inv_b, b.coefficients() );
    Coeffs< T, N, M > tmp{};
    detail::kernels::cauchyProduct< T, N, M >( tmp, a.coefficients(), inv_b );
    a.coefficients() = tmp;
    return a;
}

// ===========================================================================
// Sparse arithmetic:  S+S, S-S, -S, S+T, T+S, S-T, T-S, S*T, T*S, S/T
// ===========================================================================

using Sparse = storage::Sparse;

/// @brief Sparse + Sparse: two-pointer merge over sorted flat indices.
/// `forEachPair` visits indices in ascending order, so results are appended
/// directly (O(nnz)) instead of inserted via per-element binary search.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, Sparse > operator+(
    const TaylorExpansion< T, N, M, Sparse >& a,
    const TaylorExpansion< T, N, M, Sparse >& b ) noexcept
{
    TaylorExpansion< T, N, M, Sparse > r;
    auto& ri = r.container().rawIndices();
    auto& rv = r.container().rawValues();
    a.container().forEachPair(
        b.container(), [&ri, &rv]( std::size_t k, T va, T vb )
        {
            const T s = va + vb;
            if ( s != T{ 0 } )
            {
                ri.push_back( storage::flat_index_t( k ) );
                rv.push_back( s );
            }
        } );
    return r;
}

/// @brief Sparse - Sparse: two-pointer merge over sorted flat indices.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, Sparse > operator-(
    const TaylorExpansion< T, N, M, Sparse >& a,
    const TaylorExpansion< T, N, M, Sparse >& b ) noexcept
{
    TaylorExpansion< T, N, M, Sparse > r;
    auto& ri = r.container().rawIndices();
    auto& rv = r.container().rawValues();
    a.container().forEachPair(
        b.container(), [&ri, &rv]( std::size_t k, T va, T vb )
        {
            const T d = va - vb;
            if ( d != T{ 0 } )
            {
                ri.push_back( storage::flat_index_t( k ) );
                rv.push_back( d );
            }
        } );
    return r;
}

/// @brief Unary negation (support is unchanged; values are negated).
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, Sparse > operator-(
    const TaylorExpansion< T, N, M, Sparse >& a ) noexcept
{
    TaylorExpansion< T, N, M, Sparse > r;
    auto& ri = r.container().rawIndices();
    auto& rv = r.container().rawValues();
    a.container().forEachNonzero(
        [&ri, &rv]( std::size_t k, T v )
        {
            ri.push_back( storage::flat_index_t( k ) );
            rv.push_back( -v );
        } );
    return r;
}

/// @brief Sparse * scalar (support is unchanged for s != 0).
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, Sparse > operator*(
    const TaylorExpansion< T, N, M, Sparse >& a, std::type_identity_t< T > s ) noexcept
{
    if ( s == T{ 0 } ) return TaylorExpansion< T, N, M, Sparse >{};
    TaylorExpansion< T, N, M, Sparse > r;
    auto& ri = r.container().rawIndices();
    auto& rv = r.container().rawValues();
    a.container().forEachNonzero(
        [&ri, &rv, s]( std::size_t k, T v )
        {
            ri.push_back( storage::flat_index_t( k ) );
            rv.push_back( v * s );
        } );
    return r;
}

/// @brief Scalar * Sparse.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, Sparse > operator*(
    std::type_identity_t< T > s, const TaylorExpansion< T, N, M, Sparse >& a ) noexcept
{
    return a * s;
}

/// @brief Sparse / scalar.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, Sparse > operator/(
    const TaylorExpansion< T, N, M, Sparse >& a, std::type_identity_t< T > s ) noexcept
{
    return a * ( T{ 1 } / s );
}

/// @brief Sparse + scalar: add to constant term.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, Sparse > operator+(
    const TaylorExpansion< T, N, M, Sparse >& a, std::type_identity_t< T > s ) noexcept
{
    if ( s == T{ 0 } ) return a;
    const auto ai = a.container().support();
    const auto av = a.container().values();

    TaylorExpansion< T, N, M, Sparse > r;
    auto& ri = r.container().rawIndices();
    auto& rv = r.container().rawValues();
    ri.reserve( ai.size() + 1 );
    rv.reserve( av.size() + 1 );

    // Emit the constant term first, then bulk-append the (already sorted)
    // remainder — avoids the O(nnz) front-insert of accumulate(0, s).
    std::size_t b = 0;
    if ( !ai.empty() && ai.front() == 0 )
    {
        const T c = av.front() + s;
        if ( c != T{ 0 } )
        {
            ri.push_back( storage::flat_index_t( 0 ) );
            rv.push_back( c );
        }
        b = 1;
    }
    else
    {
        ri.push_back( storage::flat_index_t( 0 ) );
        rv.push_back( s );
    }
    ri.insert( ri.end(), ai.begin() + std::ptrdiff_t( b ), ai.end() );
    rv.insert( rv.end(), av.begin() + std::ptrdiff_t( b ), av.end() );
    return r;
}

/// @brief Scalar + Sparse.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, Sparse > operator+(
    std::type_identity_t< T > s, const TaylorExpansion< T, N, M, Sparse >& a ) noexcept
{
    return a + s;
}

/// @brief Sparse - scalar.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, Sparse > operator-(
    const TaylorExpansion< T, N, M, Sparse >& a, std::type_identity_t< T > s ) noexcept
{
    return a + ( -s );
}

/// @brief Scalar - Sparse.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, Sparse > operator-(
    std::type_identity_t< T > s, const TaylorExpansion< T, N, M, Sparse >& a ) noexcept
{
    return ( -a ) + s;
}

/// @brief Sparse * Sparse: truncated Cauchy product via the sparse kernel.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, Sparse > operator*(
    const TaylorExpansion< T, N, M, Sparse >& a,
    const TaylorExpansion< T, N, M, Sparse >& b ) noexcept
{
    TaylorExpansion< T, N, M, Sparse > r;
    detail::kernels::sparseCauchyProduct< T, N, M >(
        r.container(), a.container(), b.container() );
    return r;
}

/// @brief Sparse / Sparse: Cauchy product of a and 1/b.
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, Sparse > operator/(
    const TaylorExpansion< T, N, M, Sparse >& a,
    const TaylorExpansion< T, N, M, Sparse >& b )
{
    TaylorExpansion< T, N, M, Sparse > inv_b;
    detail::kernels::seriesReciprocalSparse< T, N, M >( inv_b.container(), b.container() );
    TaylorExpansion< T, N, M, Sparse > r;
    detail::kernels::sparseCauchyProduct< T, N, M >( r.container(), a.container(),
                                                     inv_b.container() );
    return r;
}

}  // namespace tax
