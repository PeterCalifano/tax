#pragma once

#include <tax/core/multi_index.hpp>

namespace tax
{

/**
 * @brief Enumerate all multi-indices of total degree exactly `degree` in `M`
 *        variables, calling `func(alpha)` for each in graded-lex order.
 */
template < int M, typename Func >
constexpr void forEachMonomialOfDegree( int degree, Func&& func )
{
    MultiIndex< M > alpha{};
    auto fill = [&]( auto& self, int var, int rem ) constexpr -> void {
        if ( var == M - 1 )
        {
            alpha[std::size_t( var )] = rem;
            func( alpha );
            return;
        }
        for ( int k = rem; k >= 0; --k )
        {
            alpha[std::size_t( var )] = k;
            self( self, var + 1, rem - k );
        }
    };
    fill( fill, 0, degree );
}

/**
 * @brief Enumerate all multi-indices with total degree in `[0, N]` in `M`
 *        variables, calling `func(alpha)` for each in graded-lex order.
 * @tparam M Number of variables (first template parameter).
 * @tparam N Truncation order (second template parameter).
 */
template < int M, int N, typename Func >
constexpr void forEachMonomial( Func&& func )
{
    for ( int deg = 0; deg <= N; ++deg )
    {
        forEachMonomialOfDegree< M >( deg, func );
    }
}

/**
 * @brief Enumerate all (k, alpha-k) sub-index pairs where k is componentwise
 *        `<= alpha`, calling `func(k, alpha-k)` for each pair.
 * @tparam M Number of variables.
 */
template < int M, typename Func >
constexpr void forEachSubIndex( const MultiIndex< M >& alpha, Func&& func )
{
    MultiIndex< M > beta{};
    auto fill = [&]( auto& self, int var ) constexpr -> void {
        if ( var == M )
        {
            MultiIndex< M > gamma{};
            for ( int i = 0; i < M; ++i )
                gamma[std::size_t( i )] = alpha[std::size_t( i )] - beta[std::size_t( i )];
            func( beta, gamma );
            return;
        }
        for ( int b = 0; b <= alpha[std::size_t( var )]; ++b )
        {
            beta[std::size_t( var )] = b;
            self( self, var + 1 );
        }
    };
    fill( fill, 0 );
}

}  // namespace tax
