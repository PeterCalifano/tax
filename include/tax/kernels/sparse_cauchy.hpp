#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

#include <tax/storage/sparse_tte.hpp>
#include <tax/utils/combinatorics.hpp>
#include <tax/utils/degree_of.hpp>

namespace tax::detail
{

/**
 * @brief Add two flat-indexed multi-indices and return the flat index of
 *        their sum. No truncation check; caller must verify total degree.
 * @details Round-trips through `unflatIndex` / `flatIndex` to keep the
 *          combinatorial layout consistent with the rest of the kernel
 *          library. Hot path inside the sparse Cauchy kernels; M is a
 *          compile-time constant so the loop unrolls.
 */
template < int M >
[[nodiscard]] constexpr std::size_t flatIndexSum( std::size_t ia, std::size_t ib ) noexcept
{
    const auto a = unflatIndex< M >( ia );
    const auto b = unflatIndex< M >( ib );
    MultiIndex< M > sum{};
    for ( int i = 0; i < M; ++i ) sum[i] = a[i] + b[i];
    return flatIndex< M >( sum );
}

// =============================================================================
// Sparse Cauchy kernels
// =============================================================================

/**
 * @brief Truncated sparse Cauchy product `out = f * g`.
 * @details Iterates only the nonzero entries of `f` and `g`. Truncates
 *          pairs whose degrees sum to more than `N` in O(1) per pair via
 *          the `DegreeOf<N, M>` lookup table. Accumulates into a stack
 *          scratch buffer indexed by flat monomial index, then emits the
 *          touched slots in ascending order.
 */
template < typename T, int N, int M >
[[nodiscard]] SparseTaylorExpansionT< T, N, M >
sparseCauchyProduct( const SparseTaylorExpansionT< T, N, M >& f,
                     const SparseTaylorExpansionT< T, N, M >& g ) noexcept
{
    constexpr std::size_t NC = numMonomials( N, M );
    using Deg = DegreeOf< N, M >;

    // Heap-allocated scratch — std::vector<bool> avoids template churn for
    // small NC and avoids large stack frames at high (N, M). Per-call cost
    // is one allocation; the inner loop runs nnz_f * nnz_g times so even
    // a few hundred bytes of allocator overhead is amortised well.
    std::vector< T > acc( NC, T{ 0 } );
    constexpr std::size_t kWords = ( NC + 63 ) / 64;
    std::array< std::uint64_t, kWords > touched{};

    const auto fi = f.indices();
    const auto fv = f.values();
    const auto gi = g.indices();
    const auto gv = g.values();

    for ( std::size_t a = 0; a < fi.size(); ++a )
    {
        const std::size_t ia = fi[a];
        const int da = int( Deg::value[ia] );
        if ( da > N ) continue;
        const T va = fv[a];
        for ( std::size_t b = 0; b < gi.size(); ++b )
        {
            const std::size_t ib = gi[b];
            const int db = int( Deg::value[ib] );
            if ( da + db > N ) continue;
            const std::size_t k = flatIndexSum< M >( ia, ib );
            acc[k] += va * gv[b];
            touched[k / 64] |= ( std::uint64_t{ 1 } << ( k & 63 ) );
        }
    }

    SparseTaylorExpansionT< T, N, M > out;
    out.reserve( NC );
    for ( std::size_t w = 0; w < kWords; ++w )
    {
        std::uint64_t bits = touched[w];
        while ( bits )
        {
            const int b = std::countr_zero( bits );
            const std::size_t k = w * 64 + std::size_t( b );
            if ( acc[k] != T{ 0 } ) out.emplaceBack( std::uint16_t( k ), acc[k] );
            bits &= bits - 1;
        }
    }
    return out;
}

/**
 * @brief Truncated sparse self-product `out = f * f`, exploiting symmetry.
 * @details Enumerates each unordered pair `{a, b}` once and doubles
 *          off-diagonal contributions, mirroring the dense
 *          `cauchySelfProduct` pattern.
 */
template < typename T, int N, int M >
[[nodiscard]] SparseTaylorExpansionT< T, N, M >
sparseCauchySelfProduct( const SparseTaylorExpansionT< T, N, M >& f ) noexcept
{
    constexpr std::size_t NC = numMonomials( N, M );
    using Deg = DegreeOf< N, M >;

    std::vector< T > acc( NC, T{ 0 } );
    constexpr std::size_t kWords = ( NC + 63 ) / 64;
    std::array< std::uint64_t, kWords > touched{};

    const auto fi = f.indices();
    const auto fv = f.values();

    for ( std::size_t a = 0; a < fi.size(); ++a )
    {
        const std::size_t ia = fi[a];
        const int da = int( Deg::value[ia] );
        if ( da > N ) continue;
        const T va = fv[a];
        // Diagonal term: f[ia]^2 contributes once.
        if ( 2 * da <= N )
        {
            const std::size_t k = flatIndexSum< M >( ia, ia );
            acc[k] += va * va;
            touched[k / 64] |= ( std::uint64_t{ 1 } << ( k & 63 ) );
        }
        // Off-diagonal: each unordered pair {ia, ib} with ib > ia contributes 2*f[ia]*f[ib].
        for ( std::size_t b = a + 1; b < fi.size(); ++b )
        {
            const std::size_t ib = fi[b];
            const int db = int( Deg::value[ib] );
            if ( da + db > N ) continue;
            const std::size_t k = flatIndexSum< M >( ia, ib );
            acc[k] += T{ 2 } * va * fv[b];
            touched[k / 64] |= ( std::uint64_t{ 1 } << ( k & 63 ) );
        }
    }

    SparseTaylorExpansionT< T, N, M > out;
    out.reserve( NC );
    for ( std::size_t w = 0; w < kWords; ++w )
    {
        std::uint64_t bits = touched[w];
        while ( bits )
        {
            const int b = std::countr_zero( bits );
            const std::size_t k = w * 64 + std::size_t( b );
            if ( acc[k] != T{ 0 } ) out.emplaceBack( std::uint16_t( k ), acc[k] );
            bits &= bits - 1;
        }
    }
    return out;
}

/**
 * @brief Truncated sparse Cauchy accumulate `out += f * g`.
 * @details Convenience for `out = out + sparseCauchyProduct(f, g)` that
 *          avoids materialising the intermediate when the caller is
 *          building an accumulator (e.g. binary-exponentiation chains).
 */
template < typename T, int N, int M >
void sparseCauchyAccumulate( SparseTaylorExpansionT< T, N, M >& out,
                             const SparseTaylorExpansionT< T, N, M >& f,
                             const SparseTaylorExpansionT< T, N, M >& g )
{
    auto prod = sparseCauchyProduct< T, N, M >( f, g );
    out = out + prod;
}

}  // namespace tax::detail

namespace tax
{

/**
 * @brief Sparse polynomial product (Cauchy convolution truncated to order `N`).
 * @details Public-facing free-function wrapper around
 *          `detail::sparseCauchyProduct`. Eager — returns a fully
 *          materialised `SparseTaylorExpansionT` result.
 */
template < typename T, int N, int M >
[[nodiscard]] inline SparseTaylorExpansionT< T, N, M >
operator*( const SparseTaylorExpansionT< T, N, M >& f,
           const SparseTaylorExpansionT< T, N, M >& g ) noexcept
{
    return detail::sparseCauchyProduct< T, N, M >( f, g );
}

}  // namespace tax
