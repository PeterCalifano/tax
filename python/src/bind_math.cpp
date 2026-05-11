// SPDX-License-Identifier: BSD-3-Clause
//
// `tax.math` submodule: math functions on TaylorExpansion. All eager —
// each call materialises a fresh `TaylorExpansion`.

#include "common.hpp"

namespace tax_py
{

void bind_math( nb::module_& m )
{
    nb::module_ math_mod = m.def_submodule(
        "math", "Math functions on `TaylorExpansion`. All eager — each call "
                "materialises a fresh `TaylorExpansion`." );

#define TAX_BIND_UNARY( name )                                                \
    math_mod.def( #name,                                                      \
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

    math_mod.def(
        "pow",
        []( const DynTE& a, double c ) { return tax::pow( a, c ); },
        nb::arg( "a" ), nb::arg( "c" ),
        "Real-exponent power: `a ** c`." );
    math_mod.def(
        "pow",
        []( const DynTE& a, int n ) { return tax::pow( a, n ); },
        nb::arg( "a" ), nb::arg( "n" ),
        "Integer-exponent power via binary exponentiation; negative `n` allowed." );

    math_mod.def(
        "atan2",
        []( const DynTE& y, const DynTE& x ) { return tax::atan2( y, x ); },
        nb::arg( "y" ), nb::arg( "x" ),
        "atan2(y, x) — see `numpy.arctan2`." );

    math_mod.def(
        "hypot",
        []( const DynTE& x, const DynTE& y ) { return tax::hypot( x, y ); },
        nb::arg( "x" ), nb::arg( "y" ),
        "sqrt(x*x + y*y), computed via the existing kernels." );
}

}  // namespace tax_py
