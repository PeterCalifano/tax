// SPDX-License-Identifier: BSD-3-Clause
//
// Module-level factories and numerical derivative helpers:
//   zero / one / constant / variable / variables / from_coeffs
//   gradient / hessian / jacobian

#include "common.hpp"

namespace tax_py
{

void bind_factories( nb::module_& m )
{
    // -----------------------------------------------------------------------
    // Factories.
    // -----------------------------------------------------------------------
    m.def(
        "zero",
        []( std::size_t order, std::size_t size ) { return DynTE::zero( order, size ); },
        nb::arg( "order" ), nb::arg( "size" ),
        "Construct a zero polynomial of the given shape." );
    m.def(
        "one",
        []( std::size_t order, std::size_t size ) { return DynTE::one( order, size ); },
        nb::arg( "order" ), nb::arg( "size" ),
        "Construct a constant 1 polynomial of the given shape." );
    m.def(
        "constant",
        []( double v, std::size_t order, std::size_t size ) {
            return DynTE::constant( v, order, size );
        },
        nb::arg( "value" ), nb::arg( "order" ), nb::arg( "size" ),
        "Construct a constant `value` polynomial." );

    m.def(
        "variable",
        []( double x0, std::size_t var_idx, std::size_t order, std::size_t size ) {
            return DynTE::variable( x0, var_idx, order, size );
        },
        nb::arg( "x0" ), nb::arg( "var_idx" ), nb::arg( "order" ), nb::arg( "size" ),
        "Coordinate variable: constant term `x0`, e_{var_idx} coefficient = 1." );

    m.def( "variables", &makeVariables, nb::arg( "x0" ), nb::arg( "order" ),
           "Return all `len(x0)` coordinate variables at the given expansion point." );

    m.def(
        "from_coeffs",
        []( const Eigen::Ref< const Eigen::VectorXd >& coeffs, std::size_t order,
            std::size_t size ) {
            const std::size_t expected = tax::detail::numMonomials( order, size );
            if ( static_cast< std::size_t >( coeffs.size() ) != expected )
                throw std::invalid_argument(
                    "from_coeffs: coeffs.size() must equal numMonomials(order, size)" );
            std::vector< double > data( std::size_t( coeffs.size() ) );
            for ( Eigen::Index i = 0; i < coeffs.size(); ++i ) data[std::size_t( i )] = coeffs( i );
            return DynTE( order, size, std::move( data ) );
        },
        nb::arg( "coeffs" ), nb::arg( "order" ), nb::arg( "size" ),
        "Construct a TaylorExpansion directly from a numpy coefficient array." );

    // -----------------------------------------------------------------------
    // Numerical gradient / hessian / jacobian — return numpy arrays directly
    // via nanobind's Eigen ↔ numpy bridge.
    // -----------------------------------------------------------------------
    m.def(
        "gradient",
        []( const DynTE& f ) { return tax::gradient( f ); },
        nb::arg( "f" ),
        "Gradient vector [df/dx_0, ..., df/dx_{M-1}] at the expansion point." );

    m.def(
        "hessian",
        []( const DynTE& f ) { return tax::hessian( f ); },
        nb::arg( "f" ),
        "Hessian matrix H(i,j) = d^2 f / (dx_i dx_j) at the expansion point." );

    m.def(
        "jacobian",
        []( const std::vector< DynTE >& vec ) {
            if ( vec.empty() )
                throw std::invalid_argument( "jacobian: empty list" );
            const Eigen::Index K = Eigen::Index( vec.size() );
            const std::size_t M = vec[0].size();
            Eigen::MatrixXd out( K, Eigen::Index( M ) );
            std::vector< int > alpha( M, 0 );
            for ( Eigen::Index r = 0; r < K; ++r )
            {
                for ( std::size_t j = 0; j < M; ++j )
                {
                    alpha[j] = 1;
                    out( r, Eigen::Index( j ) ) = vec[std::size_t( r )].derivative(
                        std::span< const int >( alpha.data(), M ) );
                    alpha[j] = 0;
                }
            }
            return out;
        },
        nb::arg( "fs" ),
        "Jacobian matrix J(r, j) = df_r / dx_j for a list of TaylorExpansion components." );
}

}  // namespace tax_py
