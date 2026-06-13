// include/tax/ads/refine_criteria.hpp
//
// Quality criteria for the "propagate-then-assess" ADS refinement
// (tax::ads::refine, see refine.hpp). Unlike the in-flight SplitCriterion
// of the classic driver — which inspects a single flow map mid-integration
// — a QualityCriterion judges a box by comparing the flow map of the parent
// against the flow maps of its two children once *all three* have been
// propagated all the way to the final time. The verdict answers a single
// question: "does splitting this box change the answer?" If not, the parent
// is accepted as-is; if so, the children are kept and refined further.
//
// Two indices are provided:
//
//   CoefficientMatchCriterion — dimension-free. Re-identify the parent map
//     on each half-domain (the same substitution ADS uses to split) and
//     compare it, coefficient by coefficient, to the independently
//     propagated child map. While the parent is accurate the two agree;
//     once it has diverged the mismatch blows up. Normalised by the child
//     magnitude, so `tol` is a relative tolerance.
//
//   AreaRatioCriterion — geometric, for 2-D boxes. Measure the area of the
//     image of the box under each flow map (the polygon traced by the box
//     boundary in two chosen output components) and compare the parent area
//     to the sum of the two child areas. When the parent polynomial is well
//     shaped the children tile it and the ratio is ~1; when it has folded or
//     ballooned past its radius of convergence the ratio departs from 1.
//
// Both honour a `maxDepth` cap: acceptable() returns true (stop) once
// depth >= maxDepth regardless of the index value.

#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <tax/ads/da_state.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/la/types.hpp>
#include <utility>
#include <vector>

namespace tax::ads
{

// A quality criterion drives tax::ads::refine. splitDim picks the
// coordinate to bisect from a flow map; acceptable compares a parent map
// to its two propagated children and reports whether the parent is good
// enough (true => stop, do not split).
template < class C, class State >
concept QualityCriterion =
    requires( C c, const State& p, const State& l, const State& r, int depth ) {
        { c.acceptable( p, l, r, depth ) } -> std::convertible_to< bool >;
        { c.splitDim( p ) } -> std::convertible_to< int >;
        { c.maxDepth } -> std::convertible_to< int >;
    };

namespace detail
{

// Coordinate j carrying the largest order-N coefficient mass — the same
// split-direction heuristic as TruncationCriterion. Graded-lex layout puts
// the degree-N monomials in the contiguous tail block.
template < class T, int N, int M, class Storage, int D >
[[nodiscard]] int topDegreeSplitDim(
    const Eigen::Matrix< tax::TaylorExpansion< T, N, M, Storage >, D, 1 >& f )
{
    std::array< T, M > totals{};
    constexpr std::size_t kLo = ( N > 0 ) ? tax::numMonomials( N - 1, M ) : 0;
    constexpr std::size_t Ncoef = tax::numMonomials( N, M );
    for ( Eigen::Index i = 0; i < f.size(); ++i )
    {
        const auto& row = f( i );
        for ( std::size_t k = kLo; k < Ncoef; ++k )
        {
            const T mag = std::abs( row[k] );
            if ( mag == T{ 0 } ) continue;
            const auto alpha = tax::unflatIndex< M >( k );
            for ( int j = 0; j < M; ++j )
                totals[static_cast< std::size_t >( j )] +=
                    mag * T( alpha[static_cast< std::size_t >( j )] );
        }
    }
    int best = 0;
    T bestVal = totals[0];
    for ( int j = 1; j < M; ++j )
    {
        if ( totals[static_cast< std::size_t >( j )] > bestVal )
        {
            bestVal = totals[static_cast< std::size_t >( j )];
            best = j;
        }
    }
    return best;
}

// Relative coefficient mismatch between the parent map re-identified on one
// half (ξ_dim → shift + 0.5 ξ'_dim) and the independently propagated child.
// shift = -0.5 for the left half, +0.5 for the right (cf. tax::ads::split).
template < class T, int N, int M, class Storage, int D >
[[nodiscard]] T halfMismatch(
    const Eigen::Matrix< tax::TaylorExpansion< T, N, M, Storage >, D, 1 >& parent,
    const Eigen::Matrix< tax::TaylorExpansion< T, N, M, Storage >, D, 1 >& child, int dim, T shift )
{
    T maxDiff{ 0 };
    T maxMag{ 0 };
    constexpr std::size_t Ncoef = tax::numMonomials( N, M );
    for ( Eigen::Index i = 0; i < parent.size(); ++i )
    {
        const auto restricted =
            tax::ads::detail::substituteAxis( parent( i ), dim, shift, T{ 0.5 } );
        for ( std::size_t k = 0; k < Ncoef; ++k )
        {
            const T diff = std::abs( restricted[k] - child( i )[k] );
            if ( diff > maxDiff ) maxDiff = diff;
            const T mag = std::abs( child( i )[k] );
            if ( mag > maxMag ) maxMag = mag;
        }
    }
    return maxMag > T{ 0 } ? maxDiff / maxMag : maxDiff;
}

}  // namespace detail

// Dimension-free quality index: accept the parent when re-identifying it on
// each half reproduces the independently propagated child to a relative
// tolerance `tol`.
struct CoefficientMatchCriterion
{
    double tol = 1e-3;
    int maxDepth = 8;

    template < class T, int N, int M, class Storage, int D >
    [[nodiscard]] int splitDim(
        const Eigen::Matrix< tax::TaylorExpansion< T, N, M, Storage >, D, 1 >& f ) const
    {
        return detail::topDegreeSplitDim( f );
    }

    template < class T, int N, int M, class Storage, int D >
    [[nodiscard]] bool acceptable(
        const Eigen::Matrix< tax::TaylorExpansion< T, N, M, Storage >, D, 1 >& parent,
        const Eigen::Matrix< tax::TaylorExpansion< T, N, M, Storage >, D, 1 >& left,
        const Eigen::Matrix< tax::TaylorExpansion< T, N, M, Storage >, D, 1 >& right,
        int depth ) const
    {
        if ( depth >= maxDepth ) return true;
        const int dim = detail::topDegreeSplitDim( parent );
        const T mismatch = std::max( detail::halfMismatch( parent, left, dim, T{ -0.5 } ),
                                     detail::halfMismatch( parent, right, dim, T{ 0.5 } ) );
        return mismatch <= T{ tol };
    }
};

// Geometric quality index for 2-D boxes: ratio of the parent image area to
// the summed child image areas. Accept when |ratio - 1| <= tol. `outX` and
// `outY` are the two output components whose plane the area is measured in;
// `nEdge` is the boundary sampling density per box edge.
struct AreaRatioCriterion
{
    double tol = 0.05;
    int maxDepth = 8;
    int outX = 0;
    int outY = 1;
    int nEdge = 24;

    template < class T, int N, int M, class Storage, int D >
    [[nodiscard]] int splitDim(
        const Eigen::Matrix< tax::TaylorExpansion< T, N, M, Storage >, D, 1 >& f ) const
    {
        return detail::topDegreeSplitDim( f );
    }

    template < class T, int N, int M, class Storage, int D >
    [[nodiscard]] bool acceptable(
        const Eigen::Matrix< tax::TaylorExpansion< T, N, M, Storage >, D, 1 >& parent,
        const Eigen::Matrix< tax::TaylorExpansion< T, N, M, Storage >, D, 1 >& left,
        const Eigen::Matrix< tax::TaylorExpansion< T, N, M, Storage >, D, 1 >& right,
        int depth ) const
    {
        if ( depth >= maxDepth ) return true;
        const double ap = imageArea( parent );
        const double denom = imageArea( left ) + imageArea( right );
        if ( !( denom > 0.0 ) ) return true;
        return std::abs( ap / denom - 1.0 ) <= tol;
    }

    // Area of the image of [-1, 1]^2 under (f[outX], f[outY]), via the
    // shoelace formula over the box-boundary polygon.
    template < class T, int N, int M, class Storage, int D >
    [[nodiscard]] double imageArea(
        const Eigen::Matrix< tax::TaylorExpansion< T, N, M, Storage >, D, 1 >& f ) const
    {
        static_assert( M == 2, "AreaRatioCriterion measures the image area of a 2-D box" );
        const int n = nEdge > 0 ? nEdge : 1;

        std::vector< std::array< double, 2 > > pts;
        pts.reserve( static_cast< std::size_t >( 4 * n ) );
        for ( int edge = 0; edge < 4; ++edge )
        {
            for ( int i = 0; i < n; ++i )
            {
                const double s = static_cast< double >( i ) / static_cast< double >( n );
                double a = 0.0, b = 0.0;
                switch ( edge )
                {
                    case 0:
                        a = -1.0 + 2.0 * s;
                        b = +1.0;
                        break;
                    case 1:
                        a = +1.0;
                        b = +1.0 - 2.0 * s;
                        break;
                    case 2:
                        a = +1.0 - 2.0 * s;
                        b = -1.0;
                        break;
                    case 3:
                        a = -1.0;
                        b = -1.0 + 2.0 * s;
                        break;
                }
                const std::array< T, 2 > d{ static_cast< T >( a ), static_cast< T >( b ) };
                pts.push_back( { static_cast< double >( f( outX ).eval( d ) ),
                                 static_cast< double >( f( outY ).eval( d ) ) } );
            }
        }

        double twice = 0.0;
        const std::size_t m = pts.size();
        for ( std::size_t i = 0; i < m; ++i )
        {
            const auto& p = pts[i];
            const auto& q = pts[( i + 1 ) % m];
            twice += p[0] * q[1] - q[0] * p[1];
        }
        return 0.5 * std::abs( twice );
    }
};

}  // namespace tax::ads
