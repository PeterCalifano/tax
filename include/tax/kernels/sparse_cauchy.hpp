#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <tax/core/multi_index.hpp>
#include <tax/core/storage/sparse.hpp>

namespace tax::detail::kernels
{

/**
 * @brief Scratch state shared by the sparse Cauchy kernels.
 *
 * One dense accumulator per (T, N, M) instantiation, reused across calls
 * (thread_local) so the kernels do not heap-allocate per multiplication.
 * Invariant: outside a kernel call every accumulator slot is T{0}; the
 * emit pass restores the slots it touched.
 */
template < typename T, int N, int M >
struct SparseCauchyScratch
{
    static constexpr std::size_t NC     = numMonomials( N, M );
    static constexpr std::size_t kWords = ( NC + 63 ) / 64;

    std::array< T, NC >                 acc{};
    std::array< std::uint64_t, kWords > touched{};
    std::vector< MultiIndex< M > >      alpha;  // decoded support of one operand

    [[nodiscard]] static SparseCauchyScratch& instance() noexcept
    {
        static thread_local SparseCauchyScratch s;
        return s;
    }

    /// Decode a support span into multi-indices (reuses the member buffer).
    void decode( std::span< const storage::flat_index_t > support )
    {
        alpha.clear();
        alpha.reserve( support.size() );
        for ( const storage::flat_index_t k : support )
            alpha.push_back( unflatIndex< M >( std::size_t( k ) ) );
    }

    void mark( std::size_t k, T v ) noexcept
    {
        acc[k] += v;
        touched[k / 64] |= ( std::uint64_t{ 1 } << ( k & 63 ) );
    }

    /// Emit nonzero accumulator slots in ascending flat-index order and
    /// restore the scratch invariant (touched slots reset to zero).
    void emit( storage::SparseContainer< T, N, M >& out ) noexcept
    {
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
                acc[k] = T{ 0 };
                bits &= bits - 1;
            }
            touched[w] = 0;
        }
    }
};

/**
 * @brief Truncated sparse Cauchy product `out += f * g`, accumulating into `out`.
 *
 * Iterates only the nonzero entries of `f` and `g`. Each operand's support is
 * decoded to multi-indices once (O(nnz) unflatten instead of O(nnz_f * nnz_g)).
 * Supports are sorted in graded-lex order, so total degree is non-decreasing
 * along each support: the inner loop terminates (rather than skips) at the
 * first pair whose degree sum exceeds N.
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
    static const DegreeOf< N, M > deg_table{};

    const auto fi = f.support();
    const auto fv = f.values();
    const auto gi = g.support();
    const auto gv = g.values();

    auto& scratch = SparseCauchyScratch< T, N, M >::instance();
    scratch.decode( gi );

    for ( std::size_t a = 0; a < fi.size(); ++a )
    {
        const std::size_t ia      = fi[a];
        const int         da      = deg_table.value[ia];
        const auto        alpha_a = unflatIndex< M >( ia );
        const T           va      = fv[a];
        for ( std::size_t b = 0; b < gi.size(); ++b )
        {
            if ( da + deg_table.value[gi[b]] > N )
                break;  // graded order: all later degrees are >= this one
            MultiIndex< M > sum{};
            for ( int i = 0; i < M; ++i )
                sum[std::size_t( i )] =
                    alpha_a[std::size_t( i )] + scratch.alpha[b][std::size_t( i )];
            scratch.mark( flatIndex< M >( sum ), va * gv[b] );
        }
    }

    scratch.emit( out );
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
    static const DegreeOf< N, M > deg_table{};

    const auto fi = f.support();
    const auto fv = f.values();

    auto& scratch = SparseCauchyScratch< T, N, M >::instance();
    scratch.decode( fi );

    for ( std::size_t a = 0; a < fi.size(); ++a )
    {
        const std::size_t ia      = fi[a];
        const int         da      = deg_table.value[ia];
        const auto&       alpha_a = scratch.alpha[a];
        const T           va      = fv[a];

        // Diagonal: f[ia]^2 contributes once (if 2*da <= N).
        if ( 2 * da <= N )
        {
            MultiIndex< M > sum{};
            for ( int i = 0; i < M; ++i )
                sum[std::size_t( i )] = 2 * alpha_a[std::size_t( i )];
            scratch.mark( flatIndex< M >( sum ), va * va );
        }
        // Off-diagonal: pair {ia, ib} with ib > ia contributes 2*va*vb.
        for ( std::size_t b = a + 1; b < fi.size(); ++b )
        {
            if ( da + deg_table.value[fi[b]] > N )
                break;  // graded order: all later degrees are >= this one
            MultiIndex< M > sum{};
            for ( int i = 0; i < M; ++i )
                sum[std::size_t( i )] =
                    alpha_a[std::size_t( i )] + scratch.alpha[b][std::size_t( i )];
            scratch.mark( flatIndex< M >( sum ), T{ 2 } * va * fv[b] );
        }
    }

    scratch.emit( out );
}

}  // namespace tax::detail::kernels
