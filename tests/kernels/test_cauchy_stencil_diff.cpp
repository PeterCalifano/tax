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
