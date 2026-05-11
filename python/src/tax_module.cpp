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

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <Eigen/Core>
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

// Pretty polynomial form via operator<<:
//   1.5 + 2·dx₀ - 0.3·dx₀·dx₁ + 0.07·dx₁² + O(||dx||⁴)
[[nodiscard]] std::string formatStr( const DynTE& t )
{
    std::ostringstream os;
    os << t;
    return os.str();
}

// Repr is the polynomial form wrapped with shape diagnostics.
[[nodiscard]] std::string formatRepr( const DynTE& t )
{
    std::ostringstream os;
    os << t ;
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
    //  __repr__ wraps the polynomial in a "TaylorExpansion<order=..,size=..>"
    //  envelope; __str__ is just the polynomial form (matching `print(te)`
    //  output of `std::cout << te;` in C++).
    cls.def( "__repr__", &formatRepr );
    cls.def( "__str__", &formatStr );

    // -----------------------------------------------------------------------
    // Container aliases — hoisted so Vec's `__matmul__` can reference Mat
    // before its class is declared.
    // -----------------------------------------------------------------------
    using TeVec = Eigen::Matrix< DynTE, Eigen::Dynamic, 1 >;
    using TeMat = Eigen::Matrix< DynTE, Eigen::Dynamic, Eigen::Dynamic >;

    // -----------------------------------------------------------------------
    // tax.Vec — Eigen::Matrix<DynTE, Dynamic, 1> wrapper.
    //
    // Useful when you want to operate on a vector-valued TaylorExpansion
    // function as a single object (e.g. for `value()` / `eval()` / `jacobian()`
    // queries). Backed by Eigen so the existing C++ helpers work directly.
    // -----------------------------------------------------------------------
    auto vec_cls = nb::class_< TeVec >(
        m, "Vec",
        R"doc(Vector of `TaylorExpansion` objects — a 1-D collection.

Backed by `Eigen::Matrix<DynTE, Dynamic, 1>`. All elements must share the
same shape `(order, size)`; this is asserted by the underlying kernels
when arithmetic / derivative queries fire.
)doc" );

    vec_cls.def(
        "__init__",
        []( TeVec* self, const std::vector< DynTE >& fs ) {
            new ( self ) TeVec( Eigen::Index( fs.size() ) );
            for ( std::size_t i = 0; i < fs.size(); ++i )
                ( *self )( Eigen::Index( i ) ) = fs[i];
        },
        nb::arg( "fs" ),
        "Build from a list of TaylorExpansion components." );

    vec_cls.def( "__len__", []( const TeVec& v ) { return std::size_t( v.size() ); } );
    vec_cls.def(
        "__getitem__",
        []( const TeVec& v, Eigen::Index i ) -> DynTE {
            if ( i < 0 || i >= v.size() )
                throw std::out_of_range( "Vec index" );
            return v( i );
        },
        nb::arg( "i" ) );
    vec_cls.def(
        "__setitem__",
        []( TeVec& v, Eigen::Index i, const DynTE& f ) {
            if ( i < 0 || i >= v.size() )
                throw std::out_of_range( "Vec index" );
            v( i ) = f;
        },
        nb::arg( "i" ), nb::arg( "f" ) );

    vec_cls.def(
        "value",
        []( const TeVec& v ) { return tax::value( v ).eval(); },
        "Constant terms as a 1-D numpy array of shape (len,)." );

    vec_cls.def(
        "eval",
        []( const TeVec& v, const std::vector< double >& dx ) {
            Eigen::Map< const Eigen::VectorXd > edx( dx.data(),
                                                     Eigen::Index( dx.size() ) );
            return tax::eval( v, edx ).eval();
        },
        nb::arg( "dx" ),
        "Evaluate every component at the displacement `dx` (list or 1-D numpy "
        "array); numpy 1-D output." );

    vec_cls.def(
        "derivative",
        []( const TeVec& v, const std::vector< int >& alpha ) {
            return tax::derivative( v, std::span< const int >( alpha.data(), alpha.size() ) )
                .eval();
        },
        nb::arg( "alpha" ),
        "Per-component numerical partial derivative at the expansion point." );

    vec_cls.def(
        "jacobian",
        []( const TeVec& v ) { return tax::jacobian( v ).eval(); },
        "Jacobian matrix J(r, j) = d v[r] / dx_j as a numpy 2-D array." );

    // ---- slicing (segment) ----
    vec_cls.def(
        "segment",
        []( const TeVec& v, Eigen::Index start, Eigen::Index length ) {
            if ( start < 0 || length < 0 || start + length > v.size() )
                throw std::out_of_range( "Vec.segment: start/length out of range" );
            return TeVec( v.segment( start, length ) );
        },
        nb::arg( "start" ), nb::arg( "length" ),
        "Return a fresh Vec containing `length` consecutive elements starting at `start`." );

    // ---- norms (treating the Vec as a TE-valued tuple) ----
    vec_cls.def(
        "squared_norm",
        []( const TeVec& v ) {
            if ( v.size() == 0 )
                throw std::invalid_argument( "Vec.squared_norm on empty vector" );
            DynTE acc = tax::square( v( 0 ) );
            for ( Eigen::Index i = 1; i < v.size(); ++i ) acc += tax::square( v( i ) );
            return acc;
        },
        "Sum of squares of every element: `Σ v[i]²`. Returns a TaylorExpansion." );

    vec_cls.def(
        "norm",
        []( const TeVec& v ) {
            if ( v.size() == 0 )
                throw std::invalid_argument( "Vec.norm on empty vector" );
            DynTE acc = tax::square( v( 0 ) );
            for ( Eigen::Index i = 1; i < v.size(); ++i ) acc += tax::square( v( i ) );
            return tax::sqrt( acc );
        },
        "Euclidean (2-)norm `sqrt(Σ v[i]²)`. Returns a TaylorExpansion.\n"
        "Requires the value of the squared norm at the expansion point to be > 0." );

    // -----------------------------------------------------------------------
    // Vec arithmetic: element-wise +/-/*//, broadcasting against a scalar
    // (TaylorExpansion or float), and numpy 1-D arrays of floats. Matrix-
    // multiplication `@` is exposed via `__matmul__`:
    //   vec @ vec  -> TaylorExpansion (dot product)
    //   vec @ mat  -> vec (row-vector times matrix)
    // -----------------------------------------------------------------------

    // ---- in-place ----
    vec_cls.def( "__iadd__",
                 []( TeVec& a, const TeVec& b ) -> TeVec& {
                     if ( a.size() != b.size() )
                         throw std::invalid_argument( "Vec sizes must match" );
                     for ( Eigen::Index i = 0; i < a.size(); ++i ) a( i ) += b( i );
                     return a;
                 },
                 nb::rv_policy::reference );
    vec_cls.def( "__isub__",
                 []( TeVec& a, const TeVec& b ) -> TeVec& {
                     if ( a.size() != b.size() )
                         throw std::invalid_argument( "Vec sizes must match" );
                     for ( Eigen::Index i = 0; i < a.size(); ++i ) a( i ) -= b( i );
                     return a;
                 },
                 nb::rv_policy::reference );
    vec_cls.def( "__imul__",
                 []( TeVec& a, double s ) -> TeVec& {
                     for ( Eigen::Index i = 0; i < a.size(); ++i ) a( i ) *= s;
                     return a;
                 },
                 nb::rv_policy::reference );
    vec_cls.def( "__itruediv__",
                 []( TeVec& a, double s ) -> TeVec& {
                     for ( Eigen::Index i = 0; i < a.size(); ++i ) a( i ) /= s;
                     return a;
                 },
                 nb::rv_policy::reference );

    // ---- Vec ↔ Vec ----
    vec_cls.def( "__add__", []( const TeVec& a, const TeVec& b ) {
        if ( a.size() != b.size() )
            throw std::invalid_argument( "Vec sizes must match" );
        TeVec out( a.size() );
        for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = a( i ) + b( i );
        return out;
    } );
    vec_cls.def( "__sub__", []( const TeVec& a, const TeVec& b ) {
        if ( a.size() != b.size() )
            throw std::invalid_argument( "Vec sizes must match" );
        TeVec out( a.size() );
        for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = a( i ) - b( i );
        return out;
    } );
    vec_cls.def( "__mul__", []( const TeVec& a, const TeVec& b ) {
        if ( a.size() != b.size() )
            throw std::invalid_argument( "Vec sizes must match" );
        TeVec out( a.size() );
        for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = a( i ) * b( i );
        return out;
    } );
    vec_cls.def( "__truediv__", []( const TeVec& a, const TeVec& b ) {
        if ( a.size() != b.size() )
            throw std::invalid_argument( "Vec sizes must match" );
        TeVec out( a.size() );
        for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = a( i ) / b( i );
        return out;
    } );

    // ---- Vec ↔ TaylorExpansion (broadcast) ----
    vec_cls.def( "__add__", []( const TeVec& a, const DynTE& s ) {
        TeVec out( a.size() );
        for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = a( i ) + s;
        return out;
    } );
    vec_cls.def( "__radd__",
                 []( const TeVec& a, const DynTE& s ) {
                     TeVec out( a.size() );
                     for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = s + a( i );
                     return out;
                 } );
    vec_cls.def( "__sub__", []( const TeVec& a, const DynTE& s ) {
        TeVec out( a.size() );
        for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = a( i ) - s;
        return out;
    } );
    vec_cls.def( "__rsub__",
                 []( const TeVec& a, const DynTE& s ) {
                     TeVec out( a.size() );
                     for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = s - a( i );
                     return out;
                 } );
    vec_cls.def( "__mul__", []( const TeVec& a, const DynTE& s ) {
        TeVec out( a.size() );
        for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = a( i ) * s;
        return out;
    } );
    vec_cls.def( "__rmul__",
                 []( const TeVec& a, const DynTE& s ) {
                     TeVec out( a.size() );
                     for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = s * a( i );
                     return out;
                 } );
    vec_cls.def( "__truediv__", []( const TeVec& a, const DynTE& s ) {
        TeVec out( a.size() );
        for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = a( i ) / s;
        return out;
    } );

    // ---- Vec ↔ float (broadcast) ----
    vec_cls.def( "__add__", []( const TeVec& a, double s ) {
        TeVec out( a.size() );
        for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = a( i ) + s;
        return out;
    } );
    vec_cls.def( "__radd__",
                 []( const TeVec& a, double s ) {
                     TeVec out( a.size() );
                     for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = s + a( i );
                     return out;
                 } );
    vec_cls.def( "__sub__", []( const TeVec& a, double s ) {
        TeVec out( a.size() );
        for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = a( i ) - s;
        return out;
    } );
    vec_cls.def( "__rsub__",
                 []( const TeVec& a, double s ) {
                     TeVec out( a.size() );
                     for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = s - a( i );
                     return out;
                 } );
    vec_cls.def( "__mul__", []( const TeVec& a, double s ) {
        TeVec out( a.size() );
        for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = a( i ) * s;
        return out;
    } );
    vec_cls.def( "__rmul__",
                 []( const TeVec& a, double s ) {
                     TeVec out( a.size() );
                     for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = s * a( i );
                     return out;
                 } );
    vec_cls.def( "__truediv__", []( const TeVec& a, double s ) {
        TeVec out( a.size() );
        for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = a( i ) / s;
        return out;
    } );

    // ---- Vec ↔ numpy 1-D float array (per-element scalar adds) ----
    vec_cls.def( "__add__", []( const TeVec& a, const std::vector< double >& v ) {
        if ( Eigen::Index( v.size() ) != a.size() )
            throw std::invalid_argument( "Vec / 1-D array sizes must match" );
        TeVec out( a.size() );
        for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = a( i ) + v[std::size_t( i )];
        return out;
    } );
    vec_cls.def( "__sub__", []( const TeVec& a, const std::vector< double >& v ) {
        if ( Eigen::Index( v.size() ) != a.size() )
            throw std::invalid_argument( "Vec / 1-D array sizes must match" );
        TeVec out( a.size() );
        for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = a( i ) - v[std::size_t( i )];
        return out;
    } );
    vec_cls.def( "__mul__", []( const TeVec& a, const std::vector< double >& v ) {
        if ( Eigen::Index( v.size() ) != a.size() )
            throw std::invalid_argument( "Vec / 1-D array sizes must match" );
        TeVec out( a.size() );
        for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = a( i ) * v[std::size_t( i )];
        return out;
    } );
    vec_cls.def( "__truediv__",
                 []( const TeVec& a, const std::vector< double >& v ) {
                     if ( Eigen::Index( v.size() ) != a.size() )
                         throw std::invalid_argument( "Vec / 1-D array sizes must match" );
                     TeVec out( a.size() );
                     for ( Eigen::Index i = 0; i < a.size(); ++i )
                         out( i ) = a( i ) / v[std::size_t( i )];
                     return out;
                 } );

    // ---- unary negation ----
    vec_cls.def( "__neg__", []( const TeVec& a ) {
        TeVec out( a.size() );
        for ( Eigen::Index i = 0; i < a.size(); ++i ) out( i ) = -a( i );
        return out;
    } );

    // ---- dot product via @ ----
    vec_cls.def( "__matmul__", []( const TeVec& a, const TeVec& b ) {
        if ( a.size() != b.size() )
            throw std::invalid_argument( "Vec @ Vec sizes must match" );
        if ( a.size() == 0 ) throw std::invalid_argument( "Vec @ Vec on empty vectors" );
        DynTE out = a( 0 ) * b( 0 );
        for ( Eigen::Index i = 1; i < a.size(); ++i ) out += a( i ) * b( i );
        return out;
    } );

    // ---- row-vector × matrix product (vec @ mat -> vec) ----
    vec_cls.def( "__matmul__", []( const TeVec& v, const TeMat& m_ ) {
        if ( v.size() != m_.rows() )
            throw std::invalid_argument( "Vec @ Mat inner dimension must match" );
        if ( v.size() == 0 || m_.cols() == 0 )
            throw std::invalid_argument( "Vec @ Mat on empty operand" );
        TeVec out( m_.cols() );
        for ( Eigen::Index c = 0; c < m_.cols(); ++c )
        {
            DynTE accum = v( 0 ) * m_( 0, c );
            for ( Eigen::Index k = 1; k < v.size(); ++k ) accum += v( k ) * m_( k, c );
            out( c ) = accum;
        }
        return out;
    } );

    vec_cls.def( "__repr__", []( const TeVec& v ) {
        std::ostringstream os;
        os << "Vec(len=" << v.size() << ")[";
        for ( Eigen::Index i = 0; i < v.size(); ++i )
        {
            os << "\n  " << i << ": " << v( i );
            if ( i + 1 < v.size() ) os << ',';
        }
        os << ( v.size() > 0 ? "\n]" : "]" );
        return os.str();
    } );

    // -----------------------------------------------------------------------
    // tax.Mat — Eigen::Matrix<DynTE, Dynamic, Dynamic>.
    // -----------------------------------------------------------------------
    auto mat_cls = nb::class_< TeMat >(
        m, "Mat",
        R"doc(Matrix of `TaylorExpansion` objects — a 2-D collection.

Backed by `Eigen::Matrix<DynTE, Dynamic, Dynamic>`. All elements must
share the same shape `(order, size)`.
)doc" );

    mat_cls.def(
        "__init__",
        []( TeMat* self, const std::vector< std::vector< DynTE > >& rows ) {
            if ( rows.empty() )
                throw std::invalid_argument( "Mat: empty rows" );
            const Eigen::Index R = Eigen::Index( rows.size() );
            const Eigen::Index C = Eigen::Index( rows[0].size() );
            for ( const auto& row : rows )
                if ( Eigen::Index( row.size() ) != C )
                    throw std::invalid_argument(
                        "Mat: rows must have equal length" );
            new ( self ) TeMat( R, C );
            for ( Eigen::Index r = 0; r < R; ++r )
                for ( Eigen::Index c = 0; c < C; ++c )
                    ( *self )( r, c ) = rows[std::size_t( r )][std::size_t( c )];
        },
        nb::arg( "rows" ),
        "Build from a list-of-lists of TaylorExpansion components." );

    mat_cls.def_prop_ro( "rows", []( const TeMat& m_ ) { return std::size_t( m_.rows() ); } );
    mat_cls.def_prop_ro( "cols", []( const TeMat& m_ ) { return std::size_t( m_.cols() ); } );
    mat_cls.def_prop_ro( "shape", []( const TeMat& m_ ) {
        return std::pair< std::size_t, std::size_t >( m_.rows(), m_.cols() );
    } );

    mat_cls.def(
        "__getitem__",
        []( const TeMat& m_, std::pair< Eigen::Index, Eigen::Index > rc ) -> DynTE {
            const auto [r, c] = rc;
            if ( r < 0 || r >= m_.rows() || c < 0 || c >= m_.cols() )
                throw std::out_of_range( "Mat index" );
            return m_( r, c );
        },
        nb::arg( "rc" ) );
    mat_cls.def(
        "__setitem__",
        []( TeMat& m_, std::pair< Eigen::Index, Eigen::Index > rc, const DynTE& f ) {
            const auto [r, c] = rc;
            if ( r < 0 || r >= m_.rows() || c < 0 || c >= m_.cols() )
                throw std::out_of_range( "Mat index" );
            m_( r, c ) = f;
        },
        nb::arg( "rc" ), nb::arg( "f" ) );

    mat_cls.def(
        "value",
        []( const TeMat& m_ ) { return tax::value( m_ ).eval(); },
        "Constant terms as a 2-D numpy array of shape (rows, cols)." );

    mat_cls.def(
        "eval",
        []( const TeMat& m_, const std::vector< double >& dx ) {
            Eigen::Map< const Eigen::VectorXd > edx( dx.data(),
                                                     Eigen::Index( dx.size() ) );
            return tax::eval( m_, edx ).eval();
        },
        nb::arg( "dx" ),
        "Evaluate every element at the displacement `dx` (list or 1-D numpy "
        "array); numpy 2-D output." );

    mat_cls.def(
        "derivative",
        []( const TeMat& m_, const std::vector< int >& alpha ) {
            return tax::derivative( m_,
                                   std::span< const int >( alpha.data(), alpha.size() ) )
                .eval();
        },
        nb::arg( "alpha" ),
        "Per-element numerical partial derivative at the expansion point." );

    // ---- slicing (block / row / col) ----
    mat_cls.def(
        "block",
        []( const TeMat& m_, Eigen::Index r0, Eigen::Index c0, Eigen::Index rows,
            Eigen::Index cols ) {
            if ( r0 < 0 || c0 < 0 || rows < 0 || cols < 0 || r0 + rows > m_.rows()
                 || c0 + cols > m_.cols() )
                throw std::out_of_range( "Mat.block: indices out of range" );
            return TeMat( m_.block( r0, c0, rows, cols ) );
        },
        nb::arg( "row" ), nb::arg( "col" ), nb::arg( "rows" ), nb::arg( "cols" ),
        "Return a fresh Mat containing a `(rows × cols)` sub-block starting at "
        "`(row, col)`." );

    mat_cls.def(
        "row",
        []( const TeMat& m_, Eigen::Index r ) {
            if ( r < 0 || r >= m_.rows() ) throw std::out_of_range( "Mat.row index" );
            return TeVec( m_.row( r ).transpose() );
        },
        nb::arg( "r" ), "Return row `r` as a Vec." );

    mat_cls.def(
        "col",
        []( const TeMat& m_, Eigen::Index c ) {
            if ( c < 0 || c >= m_.cols() ) throw std::out_of_range( "Mat.col index" );
            return TeVec( m_.col( c ) );
        },
        nb::arg( "c" ), "Return column `c` as a Vec." );

    // -----------------------------------------------------------------------
    // Mat arithmetic: element-wise +/-/*//, broadcasting against a scalar
    // (TaylorExpansion or float), and numpy 2-D arrays of floats.
    //   mat @ mat -> mat (matrix product)
    //   mat @ vec -> vec (matrix–vector product)
    //   mat.T     -> transpose
    // -----------------------------------------------------------------------

    // ---- in-place ----
    mat_cls.def( "__iadd__",
                 []( TeMat& a, const TeMat& b ) -> TeMat& {
                     if ( a.rows() != b.rows() || a.cols() != b.cols() )
                         throw std::invalid_argument( "Mat shapes must match" );
                     for ( Eigen::Index r = 0; r < a.rows(); ++r )
                         for ( Eigen::Index c = 0; c < a.cols(); ++c ) a( r, c ) += b( r, c );
                     return a;
                 },
                 nb::rv_policy::reference );
    mat_cls.def( "__isub__",
                 []( TeMat& a, const TeMat& b ) -> TeMat& {
                     if ( a.rows() != b.rows() || a.cols() != b.cols() )
                         throw std::invalid_argument( "Mat shapes must match" );
                     for ( Eigen::Index r = 0; r < a.rows(); ++r )
                         for ( Eigen::Index c = 0; c < a.cols(); ++c ) a( r, c ) -= b( r, c );
                     return a;
                 },
                 nb::rv_policy::reference );
    mat_cls.def( "__imul__",
                 []( TeMat& a, double s ) -> TeMat& {
                     for ( Eigen::Index r = 0; r < a.rows(); ++r )
                         for ( Eigen::Index c = 0; c < a.cols(); ++c ) a( r, c ) *= s;
                     return a;
                 },
                 nb::rv_policy::reference );

    // ---- Mat ↔ Mat (element-wise) ----
    auto mat_binop = []( const TeMat& a, const TeMat& b, auto op ) {
        if ( a.rows() != b.rows() || a.cols() != b.cols() )
            throw std::invalid_argument( "Mat shapes must match" );
        TeMat out( a.rows(), a.cols() );
        for ( Eigen::Index r = 0; r < a.rows(); ++r )
            for ( Eigen::Index c = 0; c < a.cols(); ++c ) out( r, c ) = op( a( r, c ), b( r, c ) );
        return out;
    };
    mat_cls.def( "__add__", [=]( const TeMat& a, const TeMat& b ) {
        return mat_binop( a, b, []( const DynTE& x, const DynTE& y ) { return x + y; } );
    } );
    mat_cls.def( "__sub__", [=]( const TeMat& a, const TeMat& b ) {
        return mat_binop( a, b, []( const DynTE& x, const DynTE& y ) { return x - y; } );
    } );
    mat_cls.def( "__mul__", [=]( const TeMat& a, const TeMat& b ) {
        return mat_binop( a, b, []( const DynTE& x, const DynTE& y ) { return x * y; } );
    } );
    mat_cls.def( "__truediv__", [=]( const TeMat& a, const TeMat& b ) {
        return mat_binop( a, b, []( const DynTE& x, const DynTE& y ) { return x / y; } );
    } );

    // ---- Mat ↔ TaylorExpansion (broadcast) ----
    auto mat_scalar_te_op = []( const TeMat& a, const DynTE& s, auto op ) {
        TeMat out( a.rows(), a.cols() );
        for ( Eigen::Index r = 0; r < a.rows(); ++r )
            for ( Eigen::Index c = 0; c < a.cols(); ++c ) out( r, c ) = op( a( r, c ), s );
        return out;
    };
    mat_cls.def( "__add__", [=]( const TeMat& a, const DynTE& s ) {
        return mat_scalar_te_op( a, s, []( const DynTE& x, const DynTE& y ) { return x + y; } );
    } );
    mat_cls.def( "__radd__", [=]( const TeMat& a, const DynTE& s ) {
        return mat_scalar_te_op( a, s, []( const DynTE& x, const DynTE& y ) { return y + x; } );
    } );
    mat_cls.def( "__sub__", [=]( const TeMat& a, const DynTE& s ) {
        return mat_scalar_te_op( a, s, []( const DynTE& x, const DynTE& y ) { return x - y; } );
    } );
    mat_cls.def( "__rsub__", [=]( const TeMat& a, const DynTE& s ) {
        return mat_scalar_te_op( a, s, []( const DynTE& x, const DynTE& y ) { return y - x; } );
    } );
    mat_cls.def( "__mul__", [=]( const TeMat& a, const DynTE& s ) {
        return mat_scalar_te_op( a, s, []( const DynTE& x, const DynTE& y ) { return x * y; } );
    } );
    mat_cls.def( "__rmul__", [=]( const TeMat& a, const DynTE& s ) {
        return mat_scalar_te_op( a, s, []( const DynTE& x, const DynTE& y ) { return y * x; } );
    } );
    mat_cls.def( "__truediv__", [=]( const TeMat& a, const DynTE& s ) {
        return mat_scalar_te_op( a, s, []( const DynTE& x, const DynTE& y ) { return x / y; } );
    } );

    // ---- Mat ↔ float (broadcast) ----
    auto mat_scalar_d_op = []( const TeMat& a, double s, auto op ) {
        TeMat out( a.rows(), a.cols() );
        for ( Eigen::Index r = 0; r < a.rows(); ++r )
            for ( Eigen::Index c = 0; c < a.cols(); ++c ) out( r, c ) = op( a( r, c ), s );
        return out;
    };
    mat_cls.def( "__add__", [=]( const TeMat& a, double s ) {
        return mat_scalar_d_op( a, s, []( const DynTE& x, double y ) { return x + y; } );
    } );
    mat_cls.def( "__radd__", [=]( const TeMat& a, double s ) {
        return mat_scalar_d_op( a, s, []( const DynTE& x, double y ) { return y + x; } );
    } );
    mat_cls.def( "__sub__", [=]( const TeMat& a, double s ) {
        return mat_scalar_d_op( a, s, []( const DynTE& x, double y ) { return x - y; } );
    } );
    mat_cls.def( "__rsub__", [=]( const TeMat& a, double s ) {
        return mat_scalar_d_op( a, s, []( const DynTE& x, double y ) { return y - x; } );
    } );
    mat_cls.def( "__mul__", [=]( const TeMat& a, double s ) {
        return mat_scalar_d_op( a, s, []( const DynTE& x, double y ) { return x * y; } );
    } );
    mat_cls.def( "__rmul__", [=]( const TeMat& a, double s ) {
        return mat_scalar_d_op( a, s, []( const DynTE& x, double y ) { return y * x; } );
    } );
    mat_cls.def( "__truediv__", [=]( const TeMat& a, double s ) {
        return mat_scalar_d_op( a, s, []( const DynTE& x, double y ) { return x / y; } );
    } );

    // ---- Mat ↔ numpy 2-D float array (per-element scalar adds) ----
    auto mat_numpy_op = []( const TeMat& a, const Eigen::Ref< const Eigen::MatrixXd >& B,
                            auto op ) {
        if ( a.rows() != B.rows() || a.cols() != B.cols() )
            throw std::invalid_argument( "Mat / 2-D array shapes must match" );
        TeMat out( a.rows(), a.cols() );
        for ( Eigen::Index r = 0; r < a.rows(); ++r )
            for ( Eigen::Index c = 0; c < a.cols(); ++c )
                out( r, c ) = op( a( r, c ), B( r, c ) );
        return out;
    };
    mat_cls.def( "__add__", [=]( const TeMat& a, const Eigen::Ref< const Eigen::MatrixXd >& B ) {
        return mat_numpy_op( a, B, []( const DynTE& x, double y ) { return x + y; } );
    } );
    mat_cls.def( "__sub__", [=]( const TeMat& a, const Eigen::Ref< const Eigen::MatrixXd >& B ) {
        return mat_numpy_op( a, B, []( const DynTE& x, double y ) { return x - y; } );
    } );
    mat_cls.def( "__mul__", [=]( const TeMat& a, const Eigen::Ref< const Eigen::MatrixXd >& B ) {
        return mat_numpy_op( a, B, []( const DynTE& x, double y ) { return x * y; } );
    } );
    mat_cls.def( "__truediv__",
                 [=]( const TeMat& a, const Eigen::Ref< const Eigen::MatrixXd >& B ) {
                     return mat_numpy_op( a, B,
                                          []( const DynTE& x, double y ) { return x / y; } );
                 } );

    // ---- unary negation ----
    mat_cls.def( "__neg__", []( const TeMat& a ) {
        TeMat out( a.rows(), a.cols() );
        for ( Eigen::Index r = 0; r < a.rows(); ++r )
            for ( Eigen::Index c = 0; c < a.cols(); ++c ) out( r, c ) = -a( r, c );
        return out;
    } );

    // ---- matrix product (mat @ mat) ----
    mat_cls.def( "__matmul__", []( const TeMat& a, const TeMat& b ) {
        if ( a.cols() != b.rows() )
            throw std::invalid_argument( "Mat @ Mat inner dimensions must match" );
        if ( a.rows() == 0 || b.cols() == 0 )
            throw std::invalid_argument( "Mat @ Mat on empty matrix" );
        TeMat out( a.rows(), b.cols() );
        for ( Eigen::Index r = 0; r < a.rows(); ++r )
            for ( Eigen::Index c = 0; c < b.cols(); ++c )
            {
                DynTE accum = a( r, 0 ) * b( 0, c );
                for ( Eigen::Index k = 1; k < a.cols(); ++k ) accum += a( r, k ) * b( k, c );
                out( r, c ) = accum;
            }
        return out;
    } );

    // ---- matrix–vector product (mat @ vec) ----
    mat_cls.def( "__matmul__", []( const TeMat& a, const TeVec& v ) {
        if ( a.cols() != v.size() )
            throw std::invalid_argument( "Mat @ Vec inner dimension must match" );
        if ( a.rows() == 0 || v.size() == 0 )
            throw std::invalid_argument( "Mat @ Vec on empty operand" );
        TeVec out( a.rows() );
        for ( Eigen::Index r = 0; r < a.rows(); ++r )
        {
            DynTE accum = a( r, 0 ) * v( 0 );
            for ( Eigen::Index k = 1; k < a.cols(); ++k ) accum += a( r, k ) * v( k );
            out( r ) = accum;
        }
        return out;
    } );

    // ---- transpose ----
    mat_cls.def_prop_ro(
        "T", []( const TeMat& a ) -> TeMat { return a.transpose(); },
        "Transpose. Returns a fresh Mat with shape (cols, rows)." );

    mat_cls.def( "__repr__", []( const TeMat& m_ ) {
        std::ostringstream os;
        os << "Mat(rows=" << m_.rows() << ", cols=" << m_.cols() << ", raveled)[";
        const Eigen::Index R = m_.rows();
        const Eigen::Index C = m_.cols();
        bool first = true;
        for ( Eigen::Index r = 0; r < R; ++r )
        {
            for ( Eigen::Index c = 0; c < C; ++c )
            {
                if ( !first ) os << ',';
                os << "\n  (" << r << ',' << c << "): " << m_( r, c );
                first = false;
            }
        }
        os << ( R * C > 0 ? "\n]" : "]" );
        return os.str();
    } );

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
    // Math functions live under the `tax.math` submodule.
    // -----------------------------------------------------------------------
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
                    out( r, Eigen::Index( j ) ) =
                        vec[std::size_t( r )].derivative(
                            std::span< const int >( alpha.data(), M ) );
                    alpha[j] = 0;
                }
            }
            return out;
        },
        nb::arg( "fs" ),
        "Jacobian matrix J(r, j) = df_r / dx_j for a list of TaylorExpansion components." );
}
