#pragma once

#include <cstddef>

#include <tax/kernels/algebra.hpp>
#include <tax/kernels/cauchy.hpp>
#include <tax/kernels/sparse_cauchy.hpp>
#include <tax/storage/sparse_tte.hpp>
#include <tax/storage/tte_static.hpp>

namespace tax::detail
{

/**
 * @brief Sparse-storage square root via the dense forward-substitution kernel.
 *
 * The Taylor series of `sqrt(c + perturbation)` is generally fully dense
 * (every monomial slot in the truncated polynomial has a nonzero
 * coefficient), so this kernel densifies internally: it materialises a
 * `Coeffs<T, N, M>` from the sparse input, calls the dense `seriesSqrt`,
 * and re-emits the result as `SparseTaylorExpansionT` (dropping exact
 * zeros if any).
 *
 * The point of having this entry on the sparse side is API convenience:
 * sparse data can pass through `sqrt` without the caller having to write
 * `tax::sqrt(s.toDense())` and then re-wrap. There is no algorithmic
 * speed-up over the dense path; the output is the same dense Taylor
 * polynomial in sparse storage.
 */
template < typename T, int N, int M >
[[nodiscard]] SparseTaylorExpansionT< T, N, M >
sparseSeriesSqrt( const SparseTaylorExpansionT< T, N, M >& f ) noexcept
{
    Coeffs< T, N, M > dense_in{};
    {
        const auto fi = f.indices();
        const auto fv = f.values();
        for ( std::size_t k = 0; k < fi.size(); ++k ) dense_in[fi[k]] = fv[k];
    }
    Coeffs< T, N, M > dense_out{};
    seriesSqrt< T, N, M >( dense_out, dense_in );

    SparseTaylorExpansionT< T, N, M > out;
    constexpr std::size_t NC = numMonomials( N, M );
    out.reserve( NC );
    for ( std::size_t k = 0; k < NC; ++k )
        if ( dense_out[k] != T{ 0 } ) out.emplaceBack( std::uint16_t( k ), dense_out[k] );
    return out;
}

/**
 * @brief Sparse-storage reciprocal via the dense forward-substitution kernel.
 * @details Same densify-compute-sparsify pattern as `sparseSeriesSqrt`.
 *          The Taylor series of `1 / (c + perturbation)` is dense; the
 *          sparse entry exists for API symmetry.
 */
template < typename T, int N, int M >
[[nodiscard]] SparseTaylorExpansionT< T, N, M >
sparseSeriesReciprocal( const SparseTaylorExpansionT< T, N, M >& f ) noexcept
{
    Coeffs< T, N, M > dense_in{};
    {
        const auto fi = f.indices();
        const auto fv = f.values();
        for ( std::size_t k = 0; k < fi.size(); ++k ) dense_in[fi[k]] = fv[k];
    }
    Coeffs< T, N, M > dense_out{};
    seriesReciprocal< T, N, M >( dense_out, dense_in );

    SparseTaylorExpansionT< T, N, M > out;
    constexpr std::size_t NC = numMonomials( N, M );
    out.reserve( NC );
    for ( std::size_t k = 0; k < NC; ++k )
        if ( dense_out[k] != T{ 0 } ) out.emplaceBack( std::uint16_t( k ), dense_out[k] );
    return out;
}

}  // namespace tax::detail

namespace tax
{

/**
 * @brief Sparse `sqrt`: forward-substitution kernel, dense intermediate.
 * @details Public free-function wrapper around `detail::sparseSeriesSqrt`.
 *          Result is in sparse storage but is generally fully dense
 *          (every monomial slot nonzero).
 */
template < typename T, int N, int M >
[[nodiscard]] inline SparseTaylorExpansionT< T, N, M >
sqrt( const SparseTaylorExpansionT< T, N, M >& f ) noexcept
{
    return detail::sparseSeriesSqrt< T, N, M >( f );
}

/**
 * @brief Sparse `1 / f`: forward-substitution kernel, dense intermediate.
 */
template < typename T, int N, int M >
[[nodiscard]] inline SparseTaylorExpansionT< T, N, M >
reciprocal( const SparseTaylorExpansionT< T, N, M >& f ) noexcept
{
    return detail::sparseSeriesReciprocal< T, N, M >( f );
}

/**
 * @brief Sparse polynomial division: `a / b = a * reciprocal(b)`.
 * @details The reciprocal step uses the dense forward-substitution kernel
 *          (intermediate is generally fully dense). The final multiply
 *          uses `sparseCauchyProduct`, so if the dividend `a` is sparse
 *          the output slots are bounded by `nnz(a) * NC`.
 */
template < typename T, int N, int M >
[[nodiscard]] inline SparseTaylorExpansionT< T, N, M >
operator/( const SparseTaylorExpansionT< T, N, M >& a,
           const SparseTaylorExpansionT< T, N, M >& b ) noexcept
{
    return detail::sparseCauchyProduct< T, N, M >( a, detail::sparseSeriesReciprocal< T, N, M >( b ) );
}

/// @brief Sparse scalar / polynomial = `scalar * reciprocal(b)`.
template < typename T, int N, int M >
[[nodiscard]] inline SparseTaylorExpansionT< T, N, M >
operator/( T s, const SparseTaylorExpansionT< T, N, M >& b ) noexcept
{
    return s * detail::sparseSeriesReciprocal< T, N, M >( b );
}

}  // namespace tax
