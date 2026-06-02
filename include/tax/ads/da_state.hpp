// include/tax/ads/da_state.hpp
//
// create        — build a DA-valued state vector from a Box and a
//                 center initial condition. Each component is
//                 x0_i + halfWidth_i · ξ_i, so the state spans the box
//                 as ξ runs over [-1, 1]^M.
//
// split         — re-identify a DA state's domain on the two halves of
//                 its parent box along `dim`. Substitution:
//                   ξ_dim  →  -0.5 + 0.5 · ξ'_dim   (left half)
//                   ξ_dim  →  +0.5 + 0.5 · ξ'_dim   (right half)
//                 so the children carry polynomials in their own local
//                 [-1, 1] coordinates.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <tax/ads/box.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/la/types.hpp>
#include <utility>

namespace tax::ads
{

namespace detail
{
// Binomial coefficient C(n, k).
[[nodiscard]] inline constexpr double binom( int n, int k ) noexcept
{
    if ( k < 0 || k > n ) return 0.0;
    double r = 1.0;
    for ( int i = 0; i < k; ++i ) r = r * double( n - i ) / double( i + 1 );
    return r;
}

// Substitute ξ_dim → shift + 0.5 · ξ_dim in a single TE coefficient by
// coefficient via the binomial expansion of (shift + 0.5·ξ_dim)^a_dim.
template < class T, int N, int M, class Storage >
[[nodiscard]] tax::TaylorExpansion< T, N, M, Storage > substituteAxis(
    const tax::TaylorExpansion< T, N, M, Storage >& f, int dim, T shift ) noexcept
{
    tax::TaylorExpansion< T, N, M, Storage > out{};
    constexpr std::size_t Ncoef = tax::numMonomials( N, M );
    for ( std::size_t k = 0; k < Ncoef; ++k )
    {
        const auto alpha = tax::unflatIndex< M >( k );
        const T cval = f[k];
        if ( cval == T{ 0 } ) continue;
        const int aDim = alpha[static_cast< std::size_t >( dim )];
        // Distribute (shift + 0.5·ξ_dim)^aDim into ξ_dim^j terms.
        for ( int j = 0; j <= aDim; ++j )
        {
            tax::MultiIndex< M > beta = alpha;
            beta[static_cast< std::size_t >( dim )] = j;
            int total = 0;
            for ( int q = 0; q < M; ++q ) total += beta[static_cast< std::size_t >( q )];
            if ( total > N ) continue;
            const T coef = cval * T( detail::binom( aDim, j ) ) * std::pow( shift, T( aDim - j ) ) *
                           std::pow( T( 0.5 ), T( j ) );
            out[tax::flatIndex< M >( beta )] += coef;
        }
    }
    return out;
}
}  // namespace detail

// create<P, M[, Storage]>(box, x0): build the identity DA state on box.
template < int P, int M, class Storage = tax::storage::Dense, class T, int D >
[[nodiscard]] Eigen::Matrix< tax::TaylorExpansion< T, P, M, Storage >, D, 1 > create(
    const Box< T, M >& box, const Eigen::Matrix< T, D, 1 >& x0 )
{
    Eigen::Matrix< tax::TaylorExpansion< T, P, M, Storage >, D, 1 > out;
    if constexpr ( D == Eigen::Dynamic ) out.resize( x0.size() );
    for ( Eigen::Index i = 0; i < x0.size(); ++i )
    {
        tax::TaylorExpansion< T, P, M, Storage > comp{};
        comp[0] = x0( i );
        if ( i < M )
        {
            tax::MultiIndex< M > alpha{};
            alpha[static_cast< std::size_t >( i )] = 1;
            comp[tax::flatIndex< M >( alpha )] = box.halfWidth( i );
        }
        out( i ) = std::move( comp );
    }
    return out;
}

// split(state, parent_box, dim): produce the left/right halves.
// Deduces Storage from the input.
template < class T, int N, int M, class Storage, int D >
[[nodiscard]] std::pair< Eigen::Matrix< tax::TaylorExpansion< T, N, M, Storage >, D, 1 >,
                         Eigen::Matrix< tax::TaylorExpansion< T, N, M, Storage >, D, 1 > >
split( const Eigen::Matrix< tax::TaylorExpansion< T, N, M, Storage >, D, 1 >& state,
       const Box< T, M >& /*parent_box*/,  // substitution is in normalized coords
       int dim )
{
    using State = Eigen::Matrix< tax::TaylorExpansion< T, N, M, Storage >, D, 1 >;
    State L{ state.size() };
    State R{ state.size() };
    for ( Eigen::Index i = 0; i < state.size(); ++i )
    {
        L( i ) = detail::substituteAxis( state( i ), dim, T{ -0.5 } );
        R( i ) = detail::substituteAxis( state( i ), dim, T{ 0.5 } );
    }
    return { std::move( L ), std::move( R ) };
}

}  // namespace tax::ads
