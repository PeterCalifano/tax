// SPDX-License-Identifier: BSD-3-Clause
//
// `tax.la` submodule: re-exports of `Vec` / `Mat` plus the free
// linear-algebra functions `norm`, `dot`, `cross`.

#include "common.hpp"

namespace tax_py
{

// The `la` submodule is created by `tax_module.cpp` and the Vec / Mat
// classes are registered there directly — this file just adds the free
// linear-algebra functions on top.
void bind_la( nb::module_& la_mod )
{
    la_mod.def(
        "norm",
        []( const TeVec& v ) {
            if ( v.size() == 0 ) throw std::invalid_argument( "norm: empty vector" );
            DynTE acc = tax::square( v( 0 ) );
            for ( Eigen::Index i = 1; i < v.size(); ++i ) acc += tax::square( v( i ) );
            return tax::sqrt( acc );
        },
        nb::arg( "v" ), "Euclidean norm of a Vec (TaylorExpansion result)." );
    la_mod.def(
        "norm",
        []( const TeMat& a ) {
            if ( a.size() == 0 ) throw std::invalid_argument( "norm: empty matrix" );
            DynTE acc = tax::square( a( 0, 0 ) );
            bool first = true;
            for ( Eigen::Index r = 0; r < a.rows(); ++r )
                for ( Eigen::Index c = 0; c < a.cols(); ++c )
                {
                    if ( first )
                    {
                        first = false;
                        continue;
                    }
                    acc += tax::square( a( r, c ) );
                }
            return tax::sqrt( acc );
        },
        nb::arg( "a" ), "Frobenius norm of a Mat (TaylorExpansion result)." );

    la_mod.def(
        "dot",
        []( const TeVec& a, const TeVec& b ) {
            if ( a.size() != b.size() )
                throw std::invalid_argument( "dot: sizes must match" );
            if ( a.size() == 0 ) throw std::invalid_argument( "dot: empty vector" );
            DynTE out = a( 0 ) * b( 0 );
            for ( Eigen::Index i = 1; i < a.size(); ++i ) out += a( i ) * b( i );
            return out;
        },
        nb::arg( "a" ), nb::arg( "b" ), "Dot product of two Vec objects." );
    la_mod.def(
        "dot",
        []( const TeVec& a, const Eigen::Ref< const Eigen::VectorXd >& b ) {
            if ( a.size() != b.size() )
                throw std::invalid_argument( "dot: sizes must match" );
            if ( a.size() == 0 ) throw std::invalid_argument( "dot: empty vector" );
            DynTE out = a( 0 ) * b( 0 );
            for ( Eigen::Index i = 1; i < a.size(); ++i ) out += a( i ) * b( i );
            return out;
        },
        nb::arg( "a" ), nb::arg( "b" ),
        "Dot product of a Vec and a 1-D numpy float array." );

    la_mod.def(
        "cross",
        []( const TeVec& a, const TeVec& b ) {
            if ( a.size() != 3 || b.size() != 3 )
                throw std::invalid_argument( "cross: both operands must have size 3" );
            TeVec out( 3 );
            out( 0 ) = a( 1 ) * b( 2 ) - a( 2 ) * b( 1 );
            out( 1 ) = a( 2 ) * b( 0 ) - a( 0 ) * b( 2 );
            out( 2 ) = a( 0 ) * b( 1 ) - a( 1 ) * b( 0 );
            return out;
        },
        nb::arg( "a" ), nb::arg( "b" ),
        "3-D cross product `a × b` for two Vec objects." );
    la_mod.def(
        "cross",
        []( const TeVec& a, const Eigen::Ref< const Eigen::VectorXd >& b ) {
            if ( a.size() != 3 || b.size() != 3 )
                throw std::invalid_argument( "cross: both operands must have size 3" );
            TeVec out( 3 );
            out( 0 ) = a( 1 ) * b( 2 ) - a( 2 ) * b( 1 );
            out( 1 ) = a( 2 ) * b( 0 ) - a( 0 ) * b( 2 );
            out( 2 ) = a( 0 ) * b( 1 ) - a( 1 ) * b( 0 );
            return out;
        },
        nb::arg( "a" ), nb::arg( "b" ),
        "3-D cross product against a 1-D numpy float array of length 3." );

    // -----------------------------------------------------------------------
    // Reductions exposed as free functions (Vec methods are the same code).
    // -----------------------------------------------------------------------
    la_mod.def( "sum", []( const TeVec& v ) {
        if ( v.size() == 0 ) throw std::invalid_argument( "sum: empty vector" );
        DynTE acc = v( 0 );
        for ( Eigen::Index i = 1; i < v.size(); ++i ) acc += v( i );
        return acc;
    } );
    la_mod.def( "mean", []( const TeVec& v ) {
        if ( v.size() == 0 ) throw std::invalid_argument( "mean: empty vector" );
        DynTE acc = v( 0 );
        for ( Eigen::Index i = 1; i < v.size(); ++i ) acc += v( i );
        acc *= 1.0 / double( v.size() );
        return acc;
    } );
    la_mod.def( "prod", []( const TeVec& v ) {
        if ( v.size() == 0 ) throw std::invalid_argument( "prod: empty vector" );
        DynTE acc = v( 0 );
        for ( Eigen::Index i = 1; i < v.size(); ++i ) acc = acc * v( i );
        return acc;
    } );
    la_mod.def( "min", []( const TeVec& v ) {
        if ( v.size() == 0 ) throw std::invalid_argument( "min: empty vector" );
        Eigen::Index k = 0;
        double best = v( 0 ).value();
        for ( Eigen::Index i = 1; i < v.size(); ++i )
        {
            const double cur = v( i ).value();
            if ( cur < best )
            {
                best = cur;
                k = i;
            }
        }
        return v( k );
    } );
    la_mod.def( "max", []( const TeVec& v ) {
        if ( v.size() == 0 ) throw std::invalid_argument( "max: empty vector" );
        Eigen::Index k = 0;
        double best = v( 0 ).value();
        for ( Eigen::Index i = 1; i < v.size(); ++i )
        {
            const double cur = v( i ).value();
            if ( cur > best )
            {
                best = cur;
                k = i;
            }
        }
        return v( k );
    } );

    la_mod.def(
        "normalize",
        []( const TeVec& v ) {
            if ( v.size() == 0 )
                throw std::invalid_argument( "normalize: empty vector" );
            DynTE acc = tax::square( v( 0 ) );
            for ( Eigen::Index i = 1; i < v.size(); ++i ) acc += tax::square( v( i ) );
            const DynTE inv_n = tax::pow( tax::sqrt( acc ), -1 );
            TeVec out( v.size() );
            for ( Eigen::Index i = 0; i < v.size(); ++i ) out( i ) = v( i ) * inv_n;
            return out;
        },
        nb::arg( "v" ),
        "Return `v / norm(v)` as a fresh Vec." );

    // -----------------------------------------------------------------------
    // Mat-side free reductions to mirror the method API.
    // -----------------------------------------------------------------------
    la_mod.def( "sum", []( const TeMat& a ) {
        if ( a.size() == 0 ) throw std::invalid_argument( "sum: empty matrix" );
        DynTE acc = a( 0, 0 );
        bool first = true;
        for ( Eigen::Index r = 0; r < a.rows(); ++r )
            for ( Eigen::Index c = 0; c < a.cols(); ++c )
            {
                if ( first )
                {
                    first = false;
                    continue;
                }
                acc += a( r, c );
            }
        return acc;
    } );
    la_mod.def( "trace", []( const TeMat& a ) {
        if ( a.rows() != a.cols() )
            throw std::invalid_argument( "trace: matrix must be square" );
        if ( a.rows() == 0 ) throw std::invalid_argument( "trace: empty matrix" );
        DynTE acc = a( 0, 0 );
        for ( Eigen::Index i = 1; i < a.rows(); ++i ) acc += a( i, i );
        return acc;
    } );
    la_mod.def( "diagonal", []( const TeMat& a ) {
        const Eigen::Index n = std::min( a.rows(), a.cols() );
        TeVec out{ n };
        for ( Eigen::Index i = 0; i < n; ++i ) out( i ) = a( i, i );
        return out;
    } );

    // -----------------------------------------------------------------------
    // Vec / Mat factories. `zeros` is overloaded (n -> Vec, rows/cols -> Mat).
    // -----------------------------------------------------------------------
    la_mod.def(
        "zeros",
        []( std::size_t n, std::size_t order, std::size_t size ) {
            TeVec out{ Eigen::Index( n ) };
            for ( Eigen::Index i = 0; i < out.size(); ++i )
                out( i ) = DynTE::zero( order, size );
            return out;
        },
        nb::arg( "n" ), nb::arg( "order" ), nb::arg( "size" ),
        "Build a Vec of `n` zero TaylorExpansions of shape (order, size)." );
    la_mod.def(
        "zeros",
        []( std::size_t rows, std::size_t cols, std::size_t order, std::size_t size ) {
            TeMat out{ Eigen::Index( rows ), Eigen::Index( cols ) };
            for ( Eigen::Index r = 0; r < out.rows(); ++r )
                for ( Eigen::Index c = 0; c < out.cols(); ++c )
                    out( r, c ) = DynTE::zero( order, size );
            return out;
        },
        nb::arg( "rows" ), nb::arg( "cols" ), nb::arg( "order" ), nb::arg( "size" ),
        "Build a Mat of zero TaylorExpansions with the given shape." );

    la_mod.def(
        "identity",
        []( std::size_t n, std::size_t order, std::size_t size ) {
            const Eigen::Index N = Eigen::Index( n );
            TeMat out{ N, N };
            const DynTE zero_te = DynTE::zero( order, size );
            const DynTE one_te = DynTE::one( order, size );
            for ( Eigen::Index r = 0; r < out.rows(); ++r )
                for ( Eigen::Index c = 0; c < out.cols(); ++c )
                    out( r, c ) = ( r == c ) ? one_te : zero_te;
            return out;
        },
        nb::arg( "n" ), nb::arg( "order" ), nb::arg( "size" ),
        "Build an n × n identity Mat (ones on the diagonal, zeros elsewhere)." );

    la_mod.def(
        "diag",
        []( const TeVec& d ) {
            if ( d.size() == 0 )
                throw std::invalid_argument( "diag: empty diagonal vector" );
            const std::size_t order = d( 0 ).order();
            const std::size_t size = d( 0 ).size();
            const DynTE zero_te = DynTE::zero( order, size );
            TeMat out{ d.size(), d.size() };
            for ( Eigen::Index r = 0; r < out.rows(); ++r )
                for ( Eigen::Index c = 0; c < out.cols(); ++c )
                    out( r, c ) = ( r == c ) ? d( r ) : zero_te;
            return out;
        },
        nb::arg( "d" ),
        "Diagonal Mat with `d` on the main diagonal (size inferred from `d.size()`)." );
}

}  // namespace tax_py
