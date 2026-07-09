// Shared helpers for DACE regression tests.
//   - expectCoeffsMatch:  coefficient-wise equality between a tax
//                         TaylorExpansion and a DACE::DA reference.
//   - prepareInput:       shared pre-step that wraps the input variable(s)
//                         in a fixed expression to produce a polynomial
//                         with non-trivial structure. Applied identically
//                         on both sides of every test.

#pragma once

#include <dace/dace.h>
#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <vector>

#include <tax/tax.hpp>

namespace tax_regression
{

// expectCoeffsMatch — univariate
template < int N >
::testing::AssertionResult expectCoeffsMatch( const tax::TE< N >& tested,
                                              const DACE::DA&     ref,
                                              double              tol = 1e-12 )
{
    for ( unsigned int k = 0; k <= unsigned( N ); ++k )
    {
        const double               c_ref = ref.getCoefficient( std::vector< unsigned int >{ k } );
        const tax::MultiIndex< 1 > alpha{ int( k ) };
        const double               c_tax = tested.coeff( alpha );

        if ( !( std::isfinite( c_ref ) && std::isfinite( c_tax ) ) )
        {
            return ::testing::AssertionFailure()
                << "Non-finite coefficient at k=" << k << " (DACE=" << c_ref << ", tax=" << c_tax
                << ")";
        }
        const double diff = std::abs( c_ref - c_tax );
        if ( diff > tol )
        {
            return ::testing::AssertionFailure()
                << "Coefficient mismatch at k=" << k
                << " (DACE=" << std::setprecision( 17 ) << c_ref
                << ", tax=" << std::setprecision( 17 ) << c_tax << ", |diff|=" << diff
                << ", tol=" << tol << ")";
        }
    }
    return ::testing::AssertionSuccess();
}

// expectCoeffsMatch — multivariate
template < int N, int M >
::testing::AssertionResult expectCoeffsMatch( const tax::TE< N, M >& tested,
                                              const DACE::DA&         ref,
                                              double                  tol = 1e-12 )
{
    const std::size_t total = tax::numMonomials( N, M );
    for ( std::size_t k = 0; k < total; ++k )
    {
        const auto alpha = tax::unflatIndex< M >( k );

        std::vector< unsigned int > vindex( static_cast< std::size_t >( M ), 0u );
        for ( int i = 0; i < M; ++i )
            vindex[std::size_t( i )] = static_cast< unsigned int >( alpha[std::size_t( i )] );

        const double c_ref = ref.getCoefficient( vindex );
        const double c_tax = tested.coeff( alpha );

        if ( !( std::isfinite( c_ref ) && std::isfinite( c_tax ) ) )
        {
            return ::testing::AssertionFailure()
                << "Non-finite coefficient at k=" << k << " (DACE=" << c_ref << ", tax=" << c_tax
                << ")";
        }
        const double diff = std::abs( c_ref - c_tax );
        if ( diff > tol )
        {
            return ::testing::AssertionFailure()
                << "Coefficient mismatch at k=" << k
                << " (DACE=" << std::setprecision( 17 ) << c_ref
                << ", tax=" << std::setprecision( 17 ) << c_tax << ", |diff|=" << diff
                << ", tol=" << tol << ")";
        }
    }
    return ::testing::AssertionSuccess();
}

// prepareInput — shared pre-step applied identically on both sides.
//
// Default choice: 1.0 + 0.5 * sin(x)^2 (univariate), which yields a
// polynomial with non-trivial structure (every other low-order coefficient
// is zero by symmetry of sin around 0) and stays in [1, 1.5] so domain-
// restricted ops (log, sqrt, asin) accept it.
//
// If a particular op needs a different range it must be addressed by
// shifting this prep globally, not by special-casing the test.

template < int N >
[[nodiscard]] tax::TE< N > prepareInput( const tax::TE< N >& x ) noexcept
{
    const auto s = tax::sin( x );
    return 1.0 + 0.5 * ( s * s );
}

template < int N, int M >
[[nodiscard]] tax::TE< N, M > prepareInput( const tax::TE< N, M >& x ) noexcept
{
    const auto s = tax::sin( x );
    return 1.0 + 0.5 * ( s * s );
}

[[nodiscard]] inline DACE::DA prepareInput( const DACE::DA& x )
{
    const auto s = x.sin();
    return 1.0 + 0.5 * ( s * s );
}

}  // namespace tax_regression
