#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <tax/kernels/sparse_cauchy.hpp>
#include <tax/storage/sparse_tte.hpp>
#include <tax/utils/combinatorics.hpp>
#include <tax/utils/degree_of.hpp>

namespace tax::detail
{

/**
 * @brief Additive closure of seed multi-indices under multi-index addition,
 *        truncated to total degree `N`. Always includes `0` at the front.
 * @param seeds Flat indices of the perturbation (must be `> 0`).
 * @return Sorted (graded-lex) list of every flat index reachable as a
 *         non-negative integer combination of `seeds` with total degree `<= N`.
 *
 * Used by the sparse forward-substitution kernels to enumerate the
 * support of the output without paying for the full
 * `numMonomials(N, M)` shape. For a 2-sparse operand `c + α·x_var`,
 * the closure is `{0, x_var, 2·x_var, ..., N·x_var}` — `N+1` entries
 * regardless of the ambient `M`.
 */
template < int N, int M >
[[nodiscard]] inline std::vector< std::uint16_t >
additiveClosure( const std::vector< std::uint16_t >& seeds )
{
    constexpr std::size_t NC = numMonomials( N, M );
    std::vector< bool > seen( NC, false );
    seen[0] = true;
    std::vector< std::uint16_t > frontier{ 0 };
    while ( !frontier.empty() )
    {
        std::vector< std::uint16_t > next;
        for ( std::uint16_t a : frontier )
        {
            const auto alpha = unflatIndex< M >( a );
            for ( std::uint16_t s : seeds )
            {
                const auto beta = unflatIndex< M >( s );
                MultiIndex< M > sum_idx{};
                int deg = 0;
                for ( int i = 0; i < M; ++i )
                {
                    sum_idx[i] = alpha[i] + beta[i];
                    deg += sum_idx[i];
                }
                if ( deg > N ) continue;
                const std::size_t flat = flatIndex< M >( sum_idx );
                if ( !seen[flat] )
                {
                    seen[flat] = true;
                    next.push_back( std::uint16_t( flat ) );
                }
            }
        }
        frontier = std::move( next );
    }
    std::vector< std::uint16_t > result;
    result.reserve( NC );
    for ( std::size_t k = 0; k < NC; ++k )
        if ( seen[k] ) result.push_back( std::uint16_t( k ) );
    return result;
}

/**
 * @brief Sparse reciprocal forward substitution `out = 1 / f`.
 *
 * Iterates only the support set of `out` (the additive closure of `f`'s
 * nonzero perturbation indices). Per-monomial inner loop walks
 * `f.indices()` directly. Total work is `O(|support| · nnz_f)`. For a
 * 2-sparse operand at `(N=8, M=6)` that's ~9·2 = 18 inner iterations
 * instead of the dense `O(NC²)` ≈ 9 M pair walks.
 */
template < typename T, int N, int M >
[[nodiscard]] SparseTaylorExpansionT< T, N, M >
sparseSeriesReciprocal( const SparseTaylorExpansionT< T, N, M >& f )
{
    constexpr std::size_t NC = numMonomials( N, M );

    const auto fi = f.indices();
    const auto fv = f.values();
    if ( fi.empty() || fi.front() != 0 || fv.front() == T{ 0 } )
        throw std::domain_error(
            "sparseSeriesReciprocal: constant term must be nonzero" );
    const T inv_f0 = T{ 1 } / fv.front();

    // Seeds: f's nonzero perturbation indices (flat > 0).
    std::vector< std::uint16_t > seeds;
    seeds.reserve( fi.size() );
    for ( std::size_t k = 1; k < fi.size(); ++k ) seeds.push_back( fi[k] );

    // Support of out (sorted graded-lex, includes 0 at front).
    const auto support = additiveClosure< N, M >( seeds );

    // Dense out scratch — O(1) lookup of out[γ] in the inner update.
    std::vector< T > out( NC, T{ 0 } );
    out[0] = inv_f0;

    // Walk support[1..end]; for each α apply the recurrence
    //   out[α] = -inv_f0 * sum_{β in f.indices(), β > 0, β <= α} f[β] * out[α-β].
    for ( std::size_t k = 1; k < support.size(); ++k )
    {
        const std::size_t ai = support[k];
        const auto alpha = unflatIndex< M >( ai );
        T acc{ 0 };
        for ( std::size_t j = 1; j < fi.size(); ++j )
        {
            const auto beta = unflatIndex< M >( fi[j] );
            MultiIndex< M > gamma{};
            bool valid = true;
            for ( int i = 0; i < M; ++i )
            {
                if ( beta[i] > alpha[i] )
                {
                    valid = false;
                    break;
                }
                gamma[i] = alpha[i] - beta[i];
            }
            if ( !valid ) continue;
            const std::size_t gi = flatIndex< M >( gamma );
            acc += fv[j] * out[gi];
        }
        out[ai] = -inv_f0 * acc;
    }

    SparseTaylorExpansionT< T, N, M > result;
    result.reserve( support.size() );
    for ( std::uint16_t k : support )
        if ( out[k] != T{ 0 } ) result.emplaceBack( k, out[k] );
    return result;
}

/**
 * @brief Sparse square-root forward substitution `out = sqrt(f)`.
 *
 * Same support-set trick as `sparseSeriesReciprocal`: the output of
 * `sqrt(c + perturbation)` is nonzero exactly on the additive closure
 * of `perturbation`'s nonzero indices, truncated to degree `N`. The
 * inner convolution `out[β] * out[α-β]` walks `support[1..k-1]` for β
 * and uses a dense scratch for `out[γ]` lookup. Total work scales with
 * `|support|²` rather than `numMonomials(N, M)²`.
 */
template < typename T, int N, int M >
[[nodiscard]] SparseTaylorExpansionT< T, N, M >
sparseSeriesSqrt( const SparseTaylorExpansionT< T, N, M >& f )
{
    constexpr std::size_t NC = numMonomials( N, M );

    const auto fi = f.indices();
    const auto fv = f.values();
    const T f0 = ( !fi.empty() && fi.front() == 0 ) ? fv.front() : T{ 0 };
    if ( !( f0 > T{ 0 } ) )
        throw std::domain_error(
            "sparseSeriesSqrt: constant term must be strictly positive" );
    const T sqrt_f0 = std::sqrt( f0 );
    const T inv2sqrt = T{ 1 } / ( T{ 2 } * sqrt_f0 );

    std::vector< std::uint16_t > seeds;
    seeds.reserve( fi.size() );
    for ( std::size_t k = 1; k < fi.size(); ++k ) seeds.push_back( fi[k] );

    const auto support = additiveClosure< N, M >( seeds );

    // Dense scratches for O(1) lookup during the recurrence.
    std::vector< T > f_dense( NC, T{ 0 } );
    for ( std::size_t k = 0; k < fi.size(); ++k ) f_dense[fi[k]] = fv[k];
    std::vector< T > out( NC, T{ 0 } );
    out[0] = sqrt_f0;

    // out[α] = (f[α] - sum_{β+γ=α, β,γ>0, both in support} out[β] * out[γ]) / (2*out[0]).
    // For each α at support[k], walk β over support[1..k-1] and compute γ = α-β.
    for ( std::size_t k = 1; k < support.size(); ++k )
    {
        const std::size_t ai = support[k];
        const auto alpha = unflatIndex< M >( ai );
        T acc = f_dense[ai];
        for ( std::size_t j = 1; j < k; ++j )
        {
            const std::size_t bi = support[j];
            const auto beta = unflatIndex< M >( bi );
            MultiIndex< M > gamma{};
            bool valid = true;
            for ( int i = 0; i < M; ++i )
            {
                if ( beta[i] > alpha[i] )
                {
                    valid = false;
                    break;
                }
                gamma[i] = alpha[i] - beta[i];
            }
            if ( !valid ) continue;
            const std::size_t gi = flatIndex< M >( gamma );
            acc -= out[bi] * out[gi];
        }
        out[ai] = acc * inv2sqrt;
    }

    SparseTaylorExpansionT< T, N, M > result;
    result.reserve( support.size() );
    for ( std::uint16_t k : support )
        if ( out[k] != T{ 0 } ) result.emplaceBack( k, out[k] );
    return result;
}

}  // namespace tax::detail

namespace tax
{

/**
 * @brief Sparse `sqrt(f)` — sparse forward substitution over the
 *        additive-closure support of `f`'s perturbation.
 */
template < typename T, int N, int M >
[[nodiscard]] inline SparseTaylorExpansionT< T, N, M >
sqrt( const SparseTaylorExpansionT< T, N, M >& f )
{
    return detail::sparseSeriesSqrt< T, N, M >( f );
}

/**
 * @brief Sparse `1 / f` — sparse forward substitution over the
 *        additive-closure support of `f`'s perturbation.
 */
template < typename T, int N, int M >
[[nodiscard]] inline SparseTaylorExpansionT< T, N, M >
reciprocal( const SparseTaylorExpansionT< T, N, M >& f )
{
    return detail::sparseSeriesReciprocal< T, N, M >( f );
}

/**
 * @brief Sparse polynomial division `a / b = a * reciprocal(b)`.
 */
template < typename T, int N, int M >
[[nodiscard]] inline SparseTaylorExpansionT< T, N, M >
operator/( const SparseTaylorExpansionT< T, N, M >& a,
           const SparseTaylorExpansionT< T, N, M >& b )
{
    return detail::sparseCauchyProduct< T, N, M >( a, detail::sparseSeriesReciprocal< T, N, M >( b ) );
}

/// @brief Sparse scalar / polynomial = `scalar * reciprocal(b)`.
template < typename T, int N, int M >
[[nodiscard]] inline SparseTaylorExpansionT< T, N, M >
operator/( T s, const SparseTaylorExpansionT< T, N, M >& b )
{
    return s * detail::sparseSeriesReciprocal< T, N, M >( b );
}

}  // namespace tax
