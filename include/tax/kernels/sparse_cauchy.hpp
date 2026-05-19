#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <tax/core/multi_index.hpp>
#include <tax/core/storage/sparse.hpp>

namespace tax::detail::kernels
{

/**
 * @brief Compute the flat index of the monomial sum alpha+beta, given the flat
 *        indices of alpha and beta. No truncation check; caller must verify.
 */
template < int M >
[[nodiscard]] constexpr std::size_t flatIndexSum( std::size_t ia, std::size_t ib ) noexcept
{
    const auto a = unflatIndex< M >( ia );
    const auto b = unflatIndex< M >( ib );
    MultiIndex< M > sum{};
    for ( int i = 0; i < M; ++i ) sum[std::size_t( i )] = a[std::size_t( i )] + b[std::size_t( i )];
    return flatIndex< M >( sum );
}

/**
 * @brief Truncated sparse Cauchy product `out += f * g`, accumulating into `out`.
 *
 * Iterates only the nonzero entries of `f` and `g`. Truncates pairs whose degree
 * sum exceeds N. Uses a dense scratch buffer indexed by flat index, then emits
 * nonzero results in ascending order via a bitset-accelerated scan.
 *
 * @tparam T  Scalar type.
 * @tparam N  Truncation order.
 * @tparam M  Number of variables.
 */
template < typename T, int N, int M >
void sparseCauchyProduct( storage::SparseContainer< T, N, M >&       out,
                          const storage::SparseContainer< T, N, M >& f,
                          const storage::SparseContainer< T, N, M >& g ) noexcept
{
    constexpr std::size_t NC     = numMonomials( N, M );
    constexpr DegreeOf< N, M >  deg_table{};

    const auto fi = f.support();
    const auto fv = f.values();
    const auto gi = g.support();
    const auto gv = g.values();

    // Scratch: dense accumulator + touched-word bitset.
    std::vector< T >          acc( NC, T{ 0 } );
    constexpr std::size_t kWords = ( NC + 63 ) / 64;
    std::array< std::uint64_t, kWords > touched{};

    for ( std::size_t a = 0; a < fi.size(); ++a )
    {
        const std::size_t ia = fi[a];
        const int         da = deg_table.value[ia];
        if ( da > N ) continue;
        const T va = fv[a];
        for ( std::size_t b = 0; b < gi.size(); ++b )
        {
            const std::size_t ib = gi[b];
            const int         db = deg_table.value[ib];
            if ( da + db > N ) continue;
            const std::size_t k = flatIndexSum< M >( ia, ib );
            acc[k] += va * gv[b];
            touched[k / 64] |= ( std::uint64_t{ 1 } << ( k & 63 ) );
        }
    }

    // Emit nonzero results in ascending flat-index order.
    auto& ri = out.rawIndices();
    auto& rv = out.rawValues();
    for ( std::size_t w = 0; w < kWords; ++w )
    {
        std::uint64_t bits = touched[w];
        while ( bits )
        {
            const int         b = std::countr_zero( bits );
            const std::size_t k = w * 64 + std::size_t( b );
            if ( acc[k] != T{ 0 } )
            {
                ri.push_back( storage::flat_index_t( k ) );
                rv.push_back( acc[k] );
            }
            bits &= bits - 1;
        }
    }
}

/**
 * @brief Truncated sparse self-product `out = f * f`, exploiting pair symmetry.
 *
 * Enumerates each unordered pair {a, b} once and doubles off-diagonal
 * contributions, mirroring the dense `cauchySelfProduct` pattern.
 */
template < typename T, int N, int M >
void sparseCauchySelfProduct( storage::SparseContainer< T, N, M >&       out,
                              const storage::SparseContainer< T, N, M >& f ) noexcept
{
    constexpr std::size_t NC    = numMonomials( N, M );
    constexpr DegreeOf< N, M >  deg_table{};

    const auto fi = f.support();
    const auto fv = f.values();

    std::vector< T >          acc( NC, T{ 0 } );
    constexpr std::size_t kWords = ( NC + 63 ) / 64;
    std::array< std::uint64_t, kWords > touched{};

    for ( std::size_t a = 0; a < fi.size(); ++a )
    {
        const std::size_t ia = fi[a];
        const int         da = deg_table.value[ia];
        if ( da > N ) continue;
        const T va = fv[a];

        // Diagonal: f[ia]^2 contributes once (if 2*da <= N).
        if ( 2 * da <= N )
        {
            const std::size_t k = flatIndexSum< M >( ia, ia );
            acc[k] += va * va;
            touched[k / 64] |= ( std::uint64_t{ 1 } << ( k & 63 ) );
        }
        // Off-diagonal: pair {ia, ib} with ib > ia contributes 2*va*vb.
        for ( std::size_t b = a + 1; b < fi.size(); ++b )
        {
            const std::size_t ib = fi[b];
            const int         db = deg_table.value[ib];
            if ( da + db > N ) continue;
            const std::size_t k = flatIndexSum< M >( ia, ib );
            acc[k] += T{ 2 } * va * fv[b];
            touched[k / 64] |= ( std::uint64_t{ 1 } << ( k & 63 ) );
        }
    }

    auto& ri = out.rawIndices();
    auto& rv = out.rawValues();
    for ( std::size_t w = 0; w < kWords; ++w )
    {
        std::uint64_t bits = touched[w];
        while ( bits )
        {
            const int         b = std::countr_zero( bits );
            const std::size_t k = w * 64 + std::size_t( b );
            if ( acc[k] != T{ 0 } )
            {
                ri.push_back( storage::flat_index_t( k ) );
                rv.push_back( acc[k] );
            }
            bits &= bits - 1;
        }
    }
}

}  // namespace tax::detail::kernels
