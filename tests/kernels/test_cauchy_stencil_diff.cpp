#include <gtest/gtest.h>
#include <tax/kernels/cauchy.hpp>
#include <tax/kernels/cauchy_stencil.hpp>

template < int N, int M >
void runDiffMulti( double tol = 1e-12 )
{
    tax::Coeffs< double, N, M > a{}, b{}, out_loop{}, out_sten{};
    for ( std::size_t k = 0; k < a.size(); ++k )
    {
        a[k] = 0.1 * double( k + 1 ) - 0.4;
        b[k] = 0.2 - 0.05 * double( k );
    }
    tax::detail::kernels::cauchyProductLoop< double, N, M >( out_loop, a, b );
    tax::detail::kernels::cauchyProductStencil< double, N, M >( out_sten, a, b );
    for ( std::size_t k = 0; k < a.size(); ++k )
        EXPECT_NEAR( out_sten[k], out_loop[k], tol ) << "N=" << N << " M=" << M << " k=" << k;
}

TEST( CauchyStencil, Diff_N3_M2 ) { runDiffMulti< 3, 2 >(); }
TEST( CauchyStencil, Diff_N4_M3 ) { runDiffMulti< 4, 3 >(); }
TEST( CauchyStencil, Diff_N5_M4 ) { runDiffMulti< 5, 4 >(); }

// Entry count is exact: ordered pairs (beta, gamma) with |beta| + |gamma| <= N
// biject with monomials of degree <= N in 2M variables.
static_assert( tax::detail::kernels::CauchyStencil< 3, 2 >::kEntries
               == tax::numMonomials( 3, 4 ) );
static_assert( tax::detail::kernels::CauchyStencil< 4, 3 >::kEntries
               == tax::numMonomials( 4, 6 ) );

// cauchyProduct must remain usable in constant evaluation for M >= 2
// (the stencil path is runtime-only; constexpr callers take the loop kernel).
constexpr tax::Coeffs< double, 2, 2 > kConstexprProduct = []
{
    tax::Coeffs< double, 2, 2 > a{}, b{}, out{};
    a[0] = 1.0;  // 1 + 2x
    a[1] = 2.0;
    b[0] = 3.0;  // 3 + 4y
    b[2] = 4.0;
    tax::detail::kernels::cauchyProduct< double, 2, 2 >( out, a, b );
    return out;
}();
// (1 + 2x)(3 + 4y) = 3 + 6x + 4y + 8xy
static_assert( kConstexprProduct[0] == 3.0 );
static_assert( kConstexprProduct[1] == 6.0 );
static_assert( kConstexprProduct[2] == 4.0 );
static_assert( kConstexprProduct[3] == 0.0 );
static_assert( kConstexprProduct[4] == 8.0 );
static_assert( kConstexprProduct[5] == 0.0 );

TEST( CauchyStencil, ConstexprFallbackCompiles )
{
    SUCCEED();  // assertions above are compile-time
}
