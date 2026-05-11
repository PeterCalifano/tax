// SPDX-License-Identifier: BSD-3-Clause
//
// `tax.TaylorExpansion` bindings — the dynamic-shape `tax::DynTE<double>`
// class. Properties, accessors, evaluation, deriv/integ, derivative,
// norms, arithmetic, repr/str.

#include "common.hpp"

namespace tax_py
{

void bind_te( nb::module_& m, nb::class_< DynTE >& cls )
{
    ( void )m;

    // ---- properties ------------------------------------------------------
    // Wrapped in lambdas because `order()` / `size()` come from privately-
    // inherited `OrderHolder` / `VarsHolder` bases; member-function pointers
    // would not be accessible to nanobind.
    cls.def_prop_ro( "order", []( const DynTE& t ) { return t.order(); },
                     "Truncation order N (largest total monomial degree)." );
    cls.def_prop_ro( "size", []( const DynTE& t ) { return t.size(); },
                     "Number of variables M." );
    cls.def_prop_ro( "n_coefficients",
                     []( const DynTE& t ) { return t.coeffs().size(); },
                     "Number of stored coefficients = C(N+M, M)." );

    // ---- coefficients ----------------------------------------------------
    cls.def( "value", []( const DynTE& t ) { return t.value(); },
             "Value at the expansion point (constant term)." );
    cls.def(
        "coeff",
        []( const DynTE& t, const std::vector< int >& alpha ) {
            if ( alpha.size() != t.size() )
                throw std::invalid_argument( "alpha length must equal TaylorExpansion.size" );
            return t.coeff( spanOf( alpha ) );
        },
        nb::arg( "alpha" ),
        "Raw Taylor coefficient at the given multi-index (no factorial scaling)." );
    cls.def(
        "coeffs",
        []( const DynTE& t ) {
            const auto& cs = t.coeffs();
            Eigen::VectorXd out( Eigen::Index( cs.size() ) );
            for ( std::size_t i = 0; i < cs.size(); ++i ) out( Eigen::Index( i ) ) = cs[i];
            return out;
        },
        "Return the coefficient vector as a 1D numpy array (graded-lex order)." );

    // ---- evaluation ------------------------------------------------------
    cls.def(
        "at",
        []( const DynTE& t, double dx ) {
            if ( t.size() != 1 )
                throw std::invalid_argument(
                    "at(scalar) requires TaylorExpansion.size == 1; "
                    "use at(list) for the multivariate case." );
            return t.eval( dx );
        },
        nb::arg( "dx" ), "Evaluate the univariate polynomial at displacement `dx`." );
    cls.def(
        "at",
        []( const DynTE& t, const std::vector< double >& dx ) {
            if ( dx.size() != t.size() )
                throw std::invalid_argument(
                    "at(list): displacement length must equal TaylorExpansion.size" );
            return t.eval( std::span< const double >( dx.data(), dx.size() ) );
        },
        nb::arg( "dx" ), "Evaluate the polynomial at the displacement vector `dx`." );

    // ---- symbolic differentiation / integration --------------------------
    cls.def(
        "deriv", []( const DynTE& t, std::size_t var ) { return t.deriv( var ); },
        nb::arg( "var" ),
        "Symbolic partial derivative polynomial w.r.t. variable `var`." );
    cls.def(
        "integ", []( const DynTE& t, std::size_t var ) { return t.integ( var ); },
        nb::arg( "var" ),
        "Symbolic indefinite integral polynomial w.r.t. variable `var` "
        "(top-degree terms are truncated)." );

    // ---- numerical derivatives at the expansion point --------------------
    cls.def(
        "derivative",
        []( const DynTE& t, const std::vector< int >& alpha ) {
            if ( alpha.size() != t.size() )
                throw std::invalid_argument( "alpha length must equal TaylorExpansion.size" );
            return t.derivative( spanOf( alpha ) );
        },
        nb::arg( "alpha" ),
        "Partial derivative of total degree |alpha| at the expansion point." );
    cls.def(
        "derivatives",
        []( const DynTE& t ) {
            auto v = t.derivatives();
            Eigen::VectorXd out( Eigen::Index( v.size() ) );
            for ( std::size_t i = 0; i < v.size(); ++i ) out( Eigen::Index( i ) ) = v[i];
            return out;
        },
        "All partial derivatives in graded-lex order (numpy array, each c_i * alpha!)." );

    // ---- coefficient norms ------------------------------------------------
    cls.def(
        "coeffs_norm_inf", []( const DynTE& t ) { return t.coeffsNormInf(); },
        "max_i |c_i| of the coefficient vector." );
    cls.def(
        "coeffs_norm",
        []( const DynTE& t, unsigned int p ) { return t.coeffsNorm( p ); },
        nb::arg( "p" ), "p-norm of the coefficient vector (p > 0)." );

    // ---- arithmetic operators --------------------------------------------
    cls.def( nb::self + nb::self );
    cls.def( nb::self - nb::self );
    cls.def( nb::self * nb::self );
    cls.def( nb::self / nb::self );
    cls.def( nb::self += nb::self );
    cls.def( nb::self -= nb::self );
    cls.def( nb::self *= nb::self );
    cls.def( nb::self /= nb::self );

    cls.def( nb::self + double() );
    cls.def( double() + nb::self );
    cls.def( nb::self - double() );
    cls.def( double() - nb::self );
    cls.def( nb::self * double() );
    cls.def( double() * nb::self );
    cls.def( nb::self / double() );
    cls.def( nb::self += double() );
    cls.def( nb::self -= double() );
    cls.def( nb::self *= double() );
    cls.def( nb::self /= double() );

    cls.def( -nb::self );

    // ---- repr / str ------------------------------------------------------
    // Both forms return the polynomial form (e.g.
    //   "1.5 + 2·dx₀ - 0.3·dx₀·dx₁ + 0.07·dx₁² + O(||dx||⁴)").
    cls.def( "__repr__", &formatStr );
    cls.def( "__str__", &formatStr );
}

}  // namespace tax_py
