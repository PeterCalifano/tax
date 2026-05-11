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
}

}  // namespace tax_py
