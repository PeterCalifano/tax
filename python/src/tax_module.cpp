// SPDX-License-Identifier: BSD-3-Clause
//
// Python bindings for tax via nanobind.
//
// Exposes the fully-dynamic `tax::DynTE<double>` as `tax.TaylorExpansion` —
// runtime-fixed order and number of variables. The static-shape C++ path
// (`TaylorExpansionT<T, N, M>`) is intentionally not exposed: it would
// require either std::variant dispatch over a (Order, Vars) grid or a
// type-per-shape explosion, neither of which is useful from Python.
//
// Construction goes through module-level factories (`zero`, `one`,
// `constant`, `variable`, `variables`). Arithmetic operators and math
// functions evaluate eagerly into a fresh `TaylorExpansion` on every call.

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <span>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "tax/tax.hpp"

namespace nb = nanobind;
using DynTE = tax::DynTE< double >;

namespace
{

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

[[nodiscard]] std::span< const int > spanOf( const std::vector< int >& v ) noexcept
{
    return std::span< const int >( v.data(), v.size() );
}

[[nodiscard]] std::vector< DynTE > makeVariables( const std::vector< double >& x0,
                                                  std::size_t order )
{
    return DynTE::variables( std::span< const double >( x0.data(), x0.size() ), order );
}

[[nodiscard]] std::string formatRepr( const DynTE& t )
{
    std::ostringstream os;
    os.precision( 17 );
    os << "TaylorExpansion(order=" << t.order() << ", size=" << t.size() << ", coeffs=[";
    for ( std::size_t i = 0; i < t.coeffs().size(); ++i )
    {
        if ( i != 0 ) os << ", ";
        os << t.coeffs()[i];
    }
    os << "])";
    return os.str();
}

}  // namespace

NB_MODULE( _tax, m )
{
    m.doc() =
        "Truncated multivariate Taylor expansions (runtime order and size). "
        "The class is constructed via `zero`, `one`, `constant`, `variable`, "
        "or `variables`; arithmetic and math functions evaluate eagerly.";

    // -----------------------------------------------------------------------
    // tax.TaylorExpansion — backed by tax::DynTE<double>.
    // -----------------------------------------------------------------------
    auto cls = nb::class_< DynTE >( m, "TaylorExpansion",
                                    R"doc(Truncated multivariate Taylor expansion.

Order and number of variables are fixed at construction. Use the
module-level factories `zero`, `one`, `constant`, `variable`, or
`variables` to build instances.
)doc" );

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
                throw std::invalid_argument(
                    "alpha length must equal TaylorExpansion.size" );
            return t.coeff( spanOf( alpha ) );
        },
        nb::arg( "alpha" ),
        "Raw Taylor coefficient at the given multi-index (no factorial scaling)." );
    cls.def(
        "coeffs",
        []( const DynTE& t ) {
            return std::vector< double >( t.coeffs().begin(), t.coeffs().end() );
        },
        "Return a copy of the coefficient vector (graded-lex order)." );

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
                throw std::invalid_argument(
                    "alpha length must equal TaylorExpansion.size" );
            return t.derivative( spanOf( alpha ) );
        },
        nb::arg( "alpha" ),
        "Partial derivative of total degree |alpha| at the expansion point." );
    cls.def(
        "derivatives",
        []( const DynTE& t ) {
            auto v = t.derivatives();
            return std::vector< double >( v.begin(), v.end() );
        },
        "All partial derivatives in graded-lex order (each c_i scaled by alpha!)." );

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

    // ---- repr ------------------------------------------------------------
    cls.def( "__repr__", &formatRepr );
    cls.def( "__str__", &formatRepr );

    // -----------------------------------------------------------------------
    // Module-level factories.
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

    // -----------------------------------------------------------------------
    // Module-level math functions.
    // -----------------------------------------------------------------------
#define TAX_BIND_UNARY( name )                                                \
    m.def( #name,                                                             \
           []( const DynTE& a ) { return tax::name( a ); }, nb::arg( "a" ) )

    TAX_BIND_UNARY( sin );
    TAX_BIND_UNARY( cos );
    TAX_BIND_UNARY( tan );
    TAX_BIND_UNARY( sinh );
    TAX_BIND_UNARY( cosh );
    TAX_BIND_UNARY( tanh );
    TAX_BIND_UNARY( asin );
    TAX_BIND_UNARY( acos );
    TAX_BIND_UNARY( atan );
    TAX_BIND_UNARY( asinh );
    TAX_BIND_UNARY( acosh );
    TAX_BIND_UNARY( atanh );
    TAX_BIND_UNARY( exp );
    TAX_BIND_UNARY( log );
    TAX_BIND_UNARY( log10 );
    TAX_BIND_UNARY( sqrt );
    TAX_BIND_UNARY( cbrt );
    TAX_BIND_UNARY( square );
    TAX_BIND_UNARY( cube );
    TAX_BIND_UNARY( abs );
    TAX_BIND_UNARY( erf );

#undef TAX_BIND_UNARY

    m.def(
        "pow",
        []( const DynTE& a, double c ) { return tax::pow( a, c ); },
        nb::arg( "a" ), nb::arg( "c" ),
        "Real-exponent power: `a ** c`." );
    m.def(
        "pow",
        []( const DynTE& a, int n ) { return tax::pow( a, n ); },
        nb::arg( "a" ), nb::arg( "n" ),
        "Integer-exponent power via binary exponentiation; negative `n` allowed." );

    m.def(
        "atan2",
        []( const DynTE& y, const DynTE& x ) { return tax::atan2( y, x ); },
        nb::arg( "y" ), nb::arg( "x" ),
        "atan2(y, x) — see `numpy.arctan2`." );

    m.def(
        "hypot",
        []( const DynTE& x, const DynTE& y ) { return tax::hypot( x, y ); },
        nb::arg( "x" ), nb::arg( "y" ),
        "sqrt(x*x + y*y), computed via the existing kernels." );
}
