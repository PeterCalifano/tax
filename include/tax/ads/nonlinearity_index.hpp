#pragma once

#include <tax/storage/tte_static.hpp>
#include <tax/utils/combinatorics.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>

namespace tax
{

/**
 * @file
 * @brief Nonlinearity index of a Taylor expansion.
 *
 * Implements the polynomial-bound estimator of the Jacobian variation
 * introduced by Losacco, Fossà, Armellin (J. Guid. Control Dyn. 2024;
 * arXiv:2303.05791) and used to drive the low-order automatic domain
 * splitting algorithm.
 *
 * The Taylor polynomial is assumed to be expressed in *normalised*
 * deviations δ ∈ [-1, 1]^M about a box centre.  This matches the
 * coordinate convention of @ref AdsRunner, @ref LowOrderAdsRunner and
 * @ref AdsIntegrator: each leaf TTE is built from variables
 *   x_k = center[k] + halfWidth[k] · δ_k.
 */

// ---------------------------------------------------------------------------
// Jacobian-variation polynomial bound (per scalar TTE)
// ---------------------------------------------------------------------------

/**
 * @brief Polynomial bound of |∂f/∂δ_k(δ) − ∂f/∂δ_k(0)| over δ ∈ [-1,1]^M.
 *
 * For a single TTE f(δ) = Σ_α c_α δ^α the partial Jacobian is
 *   ∂f/∂δ_k(δ) = Σ_{α: α_k>0} c_α · α_k · δ^(α − e_k).
 * Its central value is the linear coefficient c_{e_k}; subtracting it
 * leaves only contributions from degree-≥-2 monomials.  Bounding the
 * remainder term-by-term using |δ_i| ≤ 1 gives
 *   B_k = Σ_{|α|≥2, α_k≥1} |c_α| · α_k.
 *
 * Entries with N < 2 are returned as all-zero (linear or constant
 * polynomial has no Jacobian variation).
 *
 * @return Per-input bound vector  (B_0, …, B_{M-1}).
 */
template < typename T, int N, int M >
[[nodiscard]] constexpr std::array< T, M >
jacobianVariationBound( const TaylorExpansionT< T, N, M >& f ) noexcept
{
    using std::abs;
    std::array< T, M > bound{};
    if constexpr ( N < 2 ) return bound;

    using TTE = TaylorExpansionT< T, N, M >;
    for ( std::size_t i = 0; i < TTE::nCoefficients; ++i )
    {
        const T c = f[i];
        if ( c == T{} ) continue;
        const auto alpha = detail::unflatIndex< M >( i );
        const int  d     = detail::totalDegree< M >( alpha );
        if ( d < 2 ) continue;
        const T mag = abs( c );
        for ( int k = 0; k < M; ++k )
            if ( alpha[k] > 0 ) bound[k] += mag * T( alpha[k] );
    }
    return bound;
}

/**
 * @brief L_∞ row-sum bound on ||J(δ) − J(0)||_∞ for a scalar TTE.
 *
 * Sums the per-input Jacobian-variation bounds (a single row of J).
 */
template < typename T, int N, int M >
[[nodiscard]] constexpr T
jacobianVariationNorm( const TaylorExpansionT< T, N, M >& f ) noexcept
{
    const auto b = jacobianVariationBound< T, N, M >( f );
    T sum{};
    for ( int k = 0; k < M; ++k ) sum += b[k];
    return sum;
}

/**
 * @brief L_∞ row-sum of the central Jacobian |J(0)| for a scalar TTE.
 *
 * Equal to Σ_k |c_{e_k}|.
 */
template < typename T, int N, int M >
[[nodiscard]] constexpr T
centralJacobianNorm( const TaylorExpansionT< T, N, M >& f ) noexcept
{
    using std::abs;
    T sum{};
    if constexpr ( N >= 1 )
    {
        for ( int k = 0; k < M; ++k )
        {
            MultiIndex< M > ek{};
            ek[k] = 1;
            sum += abs( f.coeff( ek ) );
        }
    }
    return sum;
}

// ---------------------------------------------------------------------------
// Nonlinearity index
// ---------------------------------------------------------------------------

/**
 * @brief Nonlinearity index of a vector-valued Taylor map.
 *
 * For outputs `f = (f_1, …, f_D)` expressed in normalised deviations
 * δ ∈ [-1,1]^M, the index is the polynomial bound of
 *   ||J(δ) − J(0)||_∞ / ||J(0)||_∞
 * over the unit box, where ||·||_∞ is the matrix induced ∞-norm (max
 * row sum).  Concretely:
 *   ν = (max_i Σ_k B_{ik}) / (max_i Σ_k |c_{i,e_k}|)
 * with B_{ik} from @ref jacobianVariationBound.
 *
 * - ν = 0 for affine maps (no degree-≥-2 coefficients).
 * - When the central Jacobian is exactly zero, returns 0 if the
 *   Jacobian variation also vanishes, otherwise `+∞`.
 * - Conservative: replaces δ_i by its absolute upper bound 1, so the
 *   index never under-estimates the true Jacobian variation.
 *
 * The function accepts any contiguous range of TTE outputs.  For an
 * Eigen vector use `std::span(vec.data(), vec.size())`.
 */
template < typename T, int N, int M >
[[nodiscard]] double nonlinearityIndex(
    std::span< const TaylorExpansionT< T, N, M > > outputs ) noexcept
{
    double num = 0.0;
    double den = 0.0;
    for ( const auto& f : outputs )
    {
        const double row_b = double( jacobianVariationNorm< T, N, M >( f ) );
        const double row_j = double( centralJacobianNorm< T, N, M >( f ) );
        if ( row_b > num ) num = row_b;
        if ( row_j > den ) den = row_j;
    }
    if ( den == 0.0 )
        return num > 0.0 ? std::numeric_limits< double >::infinity() : 0.0;
    return num / den;
}

/// @brief Scalar overload: nonlinearity index of a single TTE.
template < typename T, int N, int M >
[[nodiscard]] double nonlinearityIndex(
    const TaylorExpansionT< T, N, M >& f ) noexcept
{
    return nonlinearityIndex< T, N, M >(
        std::span< const TaylorExpansionT< T, N, M > >{ &f, 1 } );
}

// ---------------------------------------------------------------------------
// Per-variable split scoring
// ---------------------------------------------------------------------------

/**
 * @brief Per-variable contribution to the Jacobian-variation bound.
 *
 * For each input direction s ∈ [0, M) returns the total bound mass
 * attributable to monomials whose multi-index contains δ_s:
 *   score[s] = Σ_outputs Σ_{|α|≥2, α_s≥1} |c_α| · |α|
 * Splitting δ_s halves these contributions (each monomial's
 * δ-magnitude is multiplied by (1/2)^{α_s} ≤ 1/2), so the input with
 * the largest score is the natural candidate for bisection.
 *
 * Returned scores are always non-negative.  For N < 2 every entry is 0.
 */
template < typename T, int N, int M >
[[nodiscard]] std::array< double, M > nliPerVariable(
    std::span< const TaylorExpansionT< T, N, M > > outputs ) noexcept
{
    std::array< double, M > scores{};
    if constexpr ( N < 2 ) return scores;

    using std::abs;
    using TTE = TaylorExpansionT< T, N, M >;
    for ( const auto& f : outputs )
    {
        for ( std::size_t i = 0; i < TTE::nCoefficients; ++i )
        {
            const T c = f[i];
            if ( c == T{} ) continue;
            const auto alpha = detail::unflatIndex< M >( i );
            const int  d     = detail::totalDegree< M >( alpha );
            if ( d < 2 ) continue;
            const double mag = double( abs( c ) );
            for ( int s = 0; s < M; ++s )
                if ( alpha[s] > 0 ) scores[s] += mag * double( d );
        }
    }
    return scores;
}

/// @brief Scalar overload of @ref nliPerVariable for a single TTE.
template < typename T, int N, int M >
[[nodiscard]] std::array< double, M > nliPerVariable(
    const TaylorExpansionT< T, N, M >& f ) noexcept
{
    return nliPerVariable< T, N, M >(
        std::span< const TaylorExpansionT< T, N, M > >{ &f, 1 } );
}

/// @brief Argmax of @ref nliPerVariable — the recommended split dimension.
template < typename T, int N, int M >
[[nodiscard]] int nliSplitDim(
    std::span< const TaylorExpansionT< T, N, M > > outputs ) noexcept
{
    const auto s = nliPerVariable< T, N, M >( outputs );
    return int( std::max_element( s.begin(), s.end() ) - s.begin() );
}

template < typename T, int N, int M >
[[nodiscard]] int nliSplitDim(
    const TaylorExpansionT< T, N, M >& f ) noexcept
{
    return nliSplitDim< T, N, M >(
        std::span< const TaylorExpansionT< T, N, M > >{ &f, 1 } );
}

}  // namespace tax
