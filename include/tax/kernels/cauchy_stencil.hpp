#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <tax/utils/combinatorics.hpp>
#include <tax/utils/enumeration.hpp>

namespace tax::detail
{

/**
 * @brief Precomputed CSR stencil for the truncated multivariate Cauchy product.
 * @details For a fully static `(N, M)` configuration the full set of
 *          `(k_flat ← beta_flat × gamma_flat)` triples is known at compile time.
 *          This struct caches them as parallel `uint16_t` index arrays grouped
 *          by output `k`, turning the multivariate Cauchy inner loop into a
 *          flat dot-product that the compiler can readily unroll / vectorise.
 *
 *          Row `k` covers entries `j ∈ [offsets[k], offsets[k+1])`:
 *              `out[k] += f[col_a[j]] * g[col_b[j]]`
 *
 *          Pair-count closed form: `PC = numMonomials(N, 2*M)` —
 *          `(beta, gamma)` pairs with `|beta| + |gamma| <= N` biject with
 *          monomials in `2*M` variables of total degree `<= N`.
 */
template < int N, int M >
struct CauchyStencil
{
    static_assert( N >= 0, "CauchyStencil requires N >= 0" );
    static_assert( M >= 1, "CauchyStencil requires M >= 1" );

    static constexpr std::size_t NC = numMonomials( N, M );
    static constexpr std::size_t PC = numMonomials( N, 2 * M );

    static_assert( NC <= std::numeric_limits< uint16_t >::max(),
                   "CauchyStencil row count exceeds uint16_t — widen the index type." );

  private:
    struct Data
    {
        std::array< uint32_t, NC + 1 > offsets{};
        std::array< uint16_t, PC > col_a{};
        std::array< uint16_t, PC > col_b{};
    };

    static consteval Data build() noexcept
    {
        Data out{};

        // Pass 1: count pairs per output row.
        for ( int d = 0; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                std::size_t cnt = 1;
                for ( int v = 0; v < M; ++v ) cnt *= std::size_t( alpha[v] + 1 );
                out.offsets[ai + 1] = uint32_t( cnt );
            } );
        }
        // Prefix-sum into actual offsets.
        for ( std::size_t k = 0; k < NC; ++k )
            out.offsets[k + 1] += out.offsets[k];

        // Pass 2: fill columns. Per-row write cursor starts at offsets[k].
        std::array< std::size_t, NC > cursor{};
        for ( std::size_t k = 0; k < NC; ++k ) cursor[k] = out.offsets[k];

        for ( int d = 0; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                forEachSubIndex< M >( alpha, [&]( std::size_t bi, std::size_t gi ) {
                    const std::size_t pos = cursor[ai]++;
                    out.col_a[pos] = uint16_t( bi );
                    out.col_b[pos] = uint16_t( gi );
                } );
            } );
        }

        return out;
    }

    static constexpr Data data_ = build();

  public:
    static constexpr auto& offsets = data_.offsets;
    static constexpr auto& col_a = data_.col_a;
    static constexpr auto& col_b = data_.col_b;
};

/**
 * @brief Symmetric CSR stencil for `out = f * f`, enumerating unordered pairs.
 * @details Each unordered `{beta, gamma}` with `beta + gamma = alpha` is
 *          recorded once. `is_diag[j]` tags the diagonal pair `beta == gamma`
 *          (one per row at most, present iff every `alpha[v]` is even), so the
 *          runtime path can emit `2·f[a]·f[b]` for off-diagonal entries and
 *          `f[a]²` for diagonal entries without re-deriving the relation.
 */
template < int N, int M >
struct CauchySymStencil
{
    static_assert( N >= 0, "CauchySymStencil requires N >= 0" );
    static_assert( M >= 1, "CauchySymStencil requires M >= 1" );

    static constexpr std::size_t NC = numMonomials( N, M );

  private:
    /// Unordered-pair count: (full + diag) / 2 summed over rows.
    static consteval std::size_t computePCs() noexcept
    {
        std::size_t total = 0;
        for ( int d = 0; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t /*ai*/ ) {
                std::size_t full = 1;
                for ( int v = 0; v < M; ++v ) full *= std::size_t( alpha[v] + 1 );
                bool any_odd = false;
                for ( int v = 0; v < M; ++v )
                {
                    if ( alpha[v] & 1 )
                    {
                        any_odd = true;
                        break;
                    }
                }
                const std::size_t diag = any_odd ? 0 : 1;
                total += ( full + diag ) / 2;
            } );
        }
        return total;
    }

  public:
    static constexpr std::size_t PCs = computePCs();

    static_assert( NC <= std::numeric_limits< uint16_t >::max(),
                   "CauchySymStencil row count exceeds uint16_t — widen the index type." );

  private:
    struct Data
    {
        std::array< uint32_t, NC + 1 > offsets{};
        std::array< uint16_t, PCs > col_a{};
        std::array< uint16_t, PCs > col_b{};
        std::array< uint8_t, PCs > is_diag{};
    };

    static consteval Data build() noexcept
    {
        Data out{};

        // Pass 1: count unordered pairs per output row.
        for ( int d = 0; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                std::size_t cnt = 0;
                forEachSubIndex< M >( alpha, [&]( std::size_t bi, std::size_t gi ) {
                    if ( bi <= gi ) ++cnt;
                } );
                out.offsets[ai + 1] = uint32_t( cnt );
            } );
        }
        for ( std::size_t k = 0; k < NC; ++k )
            out.offsets[k + 1] += out.offsets[k];

        // Pass 2: fill columns + diagonal flag.
        std::array< std::size_t, NC > cursor{};
        for ( std::size_t k = 0; k < NC; ++k ) cursor[k] = out.offsets[k];

        for ( int d = 0; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t ai ) {
                forEachSubIndex< M >( alpha, [&]( std::size_t bi, std::size_t gi ) {
                    if ( bi > gi ) return;
                    const std::size_t pos = cursor[ai]++;
                    out.col_a[pos] = uint16_t( bi );
                    out.col_b[pos] = uint16_t( gi );
                    out.is_diag[pos] = uint8_t( bi == gi ? 1 : 0 );
                } );
            } );
        }

        return out;
    }

    static constexpr Data data_ = build();

  public:
    static constexpr auto& offsets = data_.offsets;
    static constexpr auto& col_a = data_.col_a;
    static constexpr auto& col_b = data_.col_b;
    static constexpr auto& is_diag = data_.is_diag;
};

/**
 * @brief Per-pair `|beta|` (degree of the first factor) for the asymmetric
 *        `CauchyStencil<N, M>`.
 * @details One byte per pair (we cap N at 254 since `numMonomials(255, 4)`
 *          already overflows uint16_t anyway).  Shared between every
 *          multivariate weighted-recurrence kernel: `seriesExp`, `seriesLog`,
 *          `seriesPow`, `seriesSinCos`, `seriesSinhCosh`, `seriesErf`,
 *          `seriesAsin`, `seriesAtan`, `seriesAsinh`, `seriesAcosh`,
 *          `seriesAtanh`.  Layout matches `CauchyStencil<N, M>` row-for-row.
 */
template < int N, int M >
struct CauchyWeightStencil
{
    static_assert( N >= 0 );
    static_assert( M >= 1 );
    static_assert( N <= 254, "CauchyWeightStencil stores |beta| in a uint8_t" );

    static constexpr std::size_t NC = numMonomials( N, M );
    static constexpr std::size_t PC = numMonomials( N, 2 * M );

  private:
    static consteval std::array< uint8_t, PC > build() noexcept
    {
        // Precompute the degree of every output monomial (one byte per row).
        std::array< uint8_t, NC > deg_of{};
        for ( int d = 0; d <= N; ++d )
            forEachMonomial< M >(
                d, [&]( const auto&, std::size_t ai ) { deg_of[ai] = uint8_t( d ); } );

        // Walk the same enumeration as CauchyStencil so positions match.
        std::array< uint8_t, PC > db{};
        std::size_t pos = 0;
        for ( int d = 0; d <= N; ++d )
        {
            forEachMonomial< M >( d, [&]( const auto& alpha, std::size_t /*ai*/ ) {
                forEachSubIndex< M >( alpha, [&]( std::size_t bi, std::size_t /*gi*/ ) {
                    db[pos++] = deg_of[bi];
                } );
            } );
        }

        return db;
    }

  public:
    static constexpr auto data_ = build();
    static constexpr auto& db = data_;
};

/**
 * @brief Cumulative monomial count up to and including each degree.
 * @details `endByDegree[d]` = `numMonomials(d, M)` so the rows of total
 *          degree `d` occupy flat indices `[endByDegree[d-1], endByDegree[d])`
 *          (with the convention `endByDegree[-1] = 0`).
 */
template < int N, int M >
struct DegreeRanges
{
    static_assert( N >= 0 );
    static_assert( M >= 1 );

  private:
    static consteval std::array< std::size_t, std::size_t( N ) + 2 > build() noexcept
    {
        std::array< std::size_t, std::size_t( N ) + 2 > r{};
        r[0] = 0;  // r[0] = numMonomials(-1, M) by convention.
        for ( int d = 0; d <= N; ++d ) r[std::size_t( d ) + 1] = numMonomials( d, M );
        return r;
    }

  public:
    static constexpr auto endByDegree = build();
};

}  // namespace tax::detail
