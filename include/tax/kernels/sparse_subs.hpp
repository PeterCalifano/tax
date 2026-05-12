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
#include <tax/utils/enumeration.hpp>

namespace tax::detail
{

/**
 * @brief Sparse reciprocal forward substitution `out = 1 / f`.
 *
 * Inner-loop work is `O(nnz_f)` per output monomial (rather than the
 * dense `O(numMonomials(|α|, M))`): for each output multi-index α we
 * accumulate `f[β] * out[α-β]` only for β values that are actually
 * nonzero in `f`. The truncated graded-lex layout of `f.indices()` lets
 * us break out of the inner loop as soon as `degree(β) > degree(α)`.
 *
 * Output is generally fully dense — the Taylor series of
 * `1 / (c + perturbation)` populates every monomial slot — so the
 * result is stored densely during the recurrence and re-emitted as
 * sparse at the end (dropping exact zeros, if any).
 */
template < typename T, int N, int M >
[[nodiscard]] SparseTaylorExpansionT< T, N, M >
sparseSeriesReciprocal( const SparseTaylorExpansionT< T, N, M >& f )
{
    constexpr std::size_t NC = numMonomials( N, M );

    // Read f[0] (must be nonzero) and the rest of f's nonzero entries.
    const auto fi = f.indices();
    const auto fv = f.values();
    if ( fi.empty() || fi.front() != 0 || fv.front() == T{ 0 } )
        throw std::domain_error(
            "sparseSeriesReciprocal: constant term must be nonzero" );
    const T inv_f0 = T{ 1 } / fv.front();

    std::vector< T > out( NC, T{ 0 } );
    out[0] = inv_f0;

    using Deg = DegreeOf< N, M >;

    for ( int d = 1; d <= N; ++d )
    {
        forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
            T acc{ 0 };
            // Iterate only nonzero entries of f at flat index > 0.
            // f.indices() is sorted by flat index = graded-lex; break once
            // degree(β) > d (β couldn't be a sub-index of α).
            for ( std::size_t k = 1; k < fi.size(); ++k )
            {
                const std::size_t bi = fi[k];
                const int db = int( Deg::value[bi] );
                if ( db > d ) break;
                // β <= α componentwise?
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
                acc += fv[k] * out[gi];
            }
            out[ai] = -inv_f0 * acc;
        } );
    }

    SparseTaylorExpansionT< T, N, M > result;
    result.reserve( NC );
    for ( std::size_t k = 0; k < NC; ++k )
        if ( out[k] != T{ 0 } ) result.emplaceBack( std::uint16_t( k ), out[k] );
    return result;
}

/**
 * @brief Sparse square root forward substitution `out = sqrt(f)`.
 *
 * The dense Cauchy convolution `2*out[0]*out[α] + sum_{β+γ=α, β,γ>0}
 * out[β]*out[γ] = f[α]` has an inner sum over OUTPUT sub-indices. We
 * maintain a sorted `out_idx` of currently-nonzero output entries and
 * iterate β over that list — when the operand only perturbs along a
 * subset of variables (e.g. `sqrt(c + α·x_var)` densifies only along
 * `x_var`), `|out_idx|` stays much smaller than `numMonomials(N, M)`
 * and the work per α scales with the operand's variable footprint
 * instead of the full shape.
 *
 * For a fully-dense operand the output densifies completely and the
 * loop degenerates to dense cost — same recurrence, no win or loss.
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

    // f as a dense scratch (O(1) lookup of f[α] in the inner update).
    std::vector< T > f_dense( NC, T{ 0 } );
    for ( std::size_t k = 0; k < fi.size(); ++k ) f_dense[fi[k]] = fv[k];

    std::vector< T > out( NC, T{ 0 } );
    std::vector< std::uint16_t > out_idx;
    out_idx.reserve( NC );
    out[0] = sqrt_f0;
    out_idx.push_back( 0 );

    using Deg = DegreeOf< N, M >;

    for ( int d = 1; d <= N; ++d )
    {
        forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
            T acc = f_dense[ai];
            // Iterate β over currently nonzero out entries with degree in [1, d-1].
            // out_idx is sorted by flat index = graded-lex, so degree is
            // monotonic and we can break once degree(β) >= d.
            for ( std::size_t k = 1; k < out_idx.size(); ++k )
            {
                const std::size_t bi = out_idx[k];
                const int db = int( Deg::value[bi] );
                if ( db >= d ) break;
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
            const T val = acc * inv2sqrt;
            out[ai] = val;
            if ( val != T{ 0 } ) out_idx.push_back( std::uint16_t( ai ) );
        } );
    }

    SparseTaylorExpansionT< T, N, M > result;
    result.reserve( out_idx.size() );
    for ( std::uint16_t k : out_idx ) result.emplaceBack( k, out[k] );
    return result;
}

}  // namespace tax::detail

namespace tax
{

/**
 * @brief Sparse `sqrt(f)` — sparse forward substitution.
 * @details Inner loop iterates only currently-nonzero output entries.
 *          For operands that perturb along a subset of variables, the
 *          result stays correspondingly sparse and the work per
 *          monomial scales with that subset rather than the full
 *          `numMonomials(N, M)` shape.
 */
template < typename T, int N, int M >
[[nodiscard]] inline SparseTaylorExpansionT< T, N, M >
sqrt( const SparseTaylorExpansionT< T, N, M >& f )
{
    return detail::sparseSeriesSqrt< T, N, M >( f );
}

/**
 * @brief Sparse `1 / f` — sparse forward substitution.
 * @details Inner loop iterates only nonzero entries of `f`; per-monomial
 *          work is `O(nnz_f)` instead of the dense `O(numMonomials)`.
 */
template < typename T, int N, int M >
[[nodiscard]] inline SparseTaylorExpansionT< T, N, M >
reciprocal( const SparseTaylorExpansionT< T, N, M >& f )
{
    return detail::sparseSeriesReciprocal< T, N, M >( f );
}

/**
 * @brief Sparse polynomial division: `a / b = a * reciprocal(b)`.
 * @details Both stages stay on the sparse path. Final multiply uses
 *          `sparseCauchyProduct`, so when the dividend `a` is sparse
 *          and the divisor's reciprocal densifies, the output is
 *          bounded by `nnz_a * NC`.
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
