#pragma once

#include <algorithm>
#include <vector>

#include <tax/utils/combinatorics.hpp>

namespace tax::detail
{

/**
 * @brief Enumerate all multi-indices of total degree `degree` in `M` variables.
 * @details Calls `func(alpha, ai)` where `ai = flatIndex<M>(alpha)`.
 */
template < int M, typename Func >
constexpr void forEachMonomial( int degree, Func&& func )
{
    tax::MultiIndex< M > alpha{};
    auto fill = [&]( auto& self, int var, int rem ) constexpr -> void {
        if ( var == M - 1 )
        {
            alpha[var] = rem;
            func( alpha, flatIndex< M >( alpha ) );
            return;
        }
        for ( int k = rem; k >= 0; --k )
        {
            alpha[var] = k;
            self( self, var + 1, rem - k );
        }
    };
    fill( fill, 0, degree );
}

/**
 * @brief Runtime-M overload of `forEachMonomial`.
 * @details `func(std::span<const int> alpha, std::size_t ai)` is called for each
 *          monomial of total degree `degree` in `M` variables.
 */
template < typename Func >
inline void forEachMonomial( int M, int degree, Func&& func )
{
    std::vector< int > alpha( std::size_t( M ), 0 );
    auto fill = [&]( auto& self, int var, int rem ) -> void {
        if ( var == M - 1 )
        {
            alpha[std::size_t( var )] = rem;
            func( std::span< const int >( alpha.data(), std::size_t( M ) ),
                  flatIndex( std::span< const int >( alpha.data(), std::size_t( M ) ) ) );
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
 * @brief Enumerate all (beta, gamma) sub-index pairs with beta + gamma = alpha.
 * @details No degree constraint. Calls `func(bi, gi)`.
 */
template < int M, typename Func >
constexpr void forEachSubIndex( const tax::MultiIndex< M >& alpha, Func&& func )
{
    tax::MultiIndex< M > beta{};
    auto fill = [&]( auto& self, int var ) constexpr -> void {
        if ( var == M )
        {
            tax::MultiIndex< M > gamma{};
            for ( int i = 0; i < M; ++i ) gamma[i] = alpha[i] - beta[i];
            func( flatIndex< M >( beta ), flatIndex< M >( gamma ) );
            return;
        }
        for ( int b = 0; b <= alpha[var]; ++b )
        {
            beta[var] = b;
            self( self, var + 1 );
        }
    };
    fill( fill, 0 );
}

/**
 * @brief Runtime-M overload of `forEachSubIndex` taking a span of exponents.
 * @details Calls `func(bi, gi)` for every pair of sub-indices summing to `alpha`.
 */
template < typename Func >
inline void forEachSubIndex( std::span< const int > alpha, Func&& func )
{
    const int M = int( alpha.size() );
    std::vector< int > beta( std::size_t( M ), 0 );
    std::vector< int > gamma( std::size_t( M ), 0 );
    auto fill = [&]( auto& self, int var ) -> void {
        if ( var == M )
        {
            for ( int i = 0; i < M; ++i ) gamma[std::size_t( i )] = alpha[i] - beta[std::size_t( i )];
            func( flatIndex( std::span< const int >( beta.data(), std::size_t( M ) ) ),
                  flatIndex( std::span< const int >( gamma.data(), std::size_t( M ) ) ) );
            return;
        }
        for ( int b = 0; b <= alpha[var]; ++b )
        {
            beta[std::size_t( var )] = b;
            self( self, var + 1 );
        }
    };
    fill( fill, 0 );
}

/**
 * @brief Enumerate (beta, gamma) sub-index pairs with beta + gamma = alpha
 *        and |beta| in [db_lo, db_hi].
 * @details Calls `func(bi, gi, db)` where `db = |beta|`.
 */
template < int M, typename Func >
constexpr void forEachSubIndex( const tax::MultiIndex< M >& alpha, int db_lo, int db_hi,
                                Func&& func )
{
    tax::MultiIndex< M > beta{};
    for ( int db = db_lo; db <= db_hi; ++db )
    {
        auto fill = [&]( auto& self, int var, int rem ) constexpr -> void {
            if ( var == M - 1 )
            {
                beta[var] = rem;
                if ( beta[var] > alpha[var] ) return;
                tax::MultiIndex< M > gamma{};
                for ( int i = 0; i < M; ++i ) gamma[i] = alpha[i] - beta[i];
                func( flatIndex< M >( beta ), flatIndex< M >( gamma ), db );
                return;
            }
            for ( int b = 0; b <= std::min( rem, alpha[var] ); ++b )
            {
                beta[var] = b;
                self( self, var + 1, rem - b );
            }
        };
        fill( fill, 0, db );
    }
}

}  // namespace tax::detail
