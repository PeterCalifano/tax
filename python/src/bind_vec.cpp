// SPDX-License-Identifier: BSD-3-Clause
//
// `tax.Vec` bindings — `Eigen::Matrix<DynTE, Dynamic, 1>`. Methods,
// indexing, slicing, norms, arithmetic, matmul (incl. numpy interop),
// dot/cross.

#include "common.hpp"

namespace tax_py
{

void bind_vec( nb::module_& m, nb::class_< TeVec >& vec_cls,
               nb::class_< TeMat >& mat_cls )
{
    ( void )m;
    ( void )mat_cls;  // Used implicitly through TeMat overload resolution.

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

    // ---- numpy array × Vec dispatch ----
    // `vec @ np_1d` and `np_1d @ vec` both compute a dot product returning a
    // TaylorExpansion. `vec @ np_2d` and `np_2d @ vec` route to matrix-vector
    // products. nanobind's eigen plugin accepts numpy arrays here.
    vec_cls.def( "__matmul__",
                 []( const TeVec& a, const Eigen::Ref< const Eigen::VectorXd >& b ) {
                     if ( a.size() != b.size() )
                         throw std::invalid_argument( "Vec @ ndarray sizes must match" );
                     if ( a.size() == 0 )
                         throw std::invalid_argument( "Vec @ ndarray on empty operand" );
                     DynTE out = a( 0 ) * b( 0 );
                     for ( Eigen::Index i = 1; i < a.size(); ++i ) out += a( i ) * b( i );
                     return out;
                 } );
    vec_cls.def( "__rmatmul__",
                 []( const TeVec& a, const Eigen::Ref< const Eigen::VectorXd >& b ) {
                     // np_1d @ vec — symmetric, same dot product.
                     if ( a.size() != b.size() )
                         throw std::invalid_argument( "ndarray @ Vec sizes must match" );
                     if ( a.size() == 0 )
                         throw std::invalid_argument( "ndarray @ Vec on empty operand" );
                     DynTE out = b( 0 ) * a( 0 );
                     for ( Eigen::Index i = 1; i < a.size(); ++i ) out += b( i ) * a( i );
                     return out;
                 } );
    vec_cls.def( "__matmul__",
                 []( const TeVec& v, const Eigen::Ref< const Eigen::MatrixXd >& M ) {
                     // vec @ np_2d: row-vector times matrix -> Vec(M.cols()).
                     if ( v.size() != M.rows() )
                         throw std::invalid_argument( "Vec @ ndarray inner dim must match" );
                     TeVec out( M.cols() );
                     for ( Eigen::Index c = 0; c < M.cols(); ++c )
                     {
                         DynTE accum = v( 0 ) * M( 0, c );
                         for ( Eigen::Index k = 1; k < v.size(); ++k )
                             accum += v( k ) * M( k, c );
                         out( c ) = accum;
                     }
                     return out;
                 } );
    vec_cls.def( "__rmatmul__",
                 []( const TeVec& v, const Eigen::Ref< const Eigen::MatrixXd >& M ) {
                     // np_2d @ vec: matrix-vector product -> Vec(M.rows()).
                     if ( v.size() != M.cols() )
                         throw std::invalid_argument( "ndarray @ Vec inner dim must match" );
                     TeVec out( M.rows() );
                     for ( Eigen::Index r = 0; r < M.rows(); ++r )
                     {
                         DynTE accum = v( 0 ) * M( r, 0 );
                         for ( Eigen::Index k = 1; k < v.size(); ++k )
                             accum += v( k ) * M( r, k );
                         out( r ) = accum;
                     }
                     return out;
                 } );

    // Tell numpy not to handle Vec via its ufunc protocol — defer to the
    // reflected `__rmatmul__` / `__radd__` / ... above. Without this, numpy
    // would iterate the Vec, build an object-dtype array, and return one
    // from `np.eye(N) @ vec`.
    vec_cls.attr( "__array_ufunc__" ) = nb::none();

    // -----------------------------------------------------------------------
    // Named linear-algebra methods (also exposed as free functions under
    // `tax.la`):
    //   `vec.dot(other)`  -> TaylorExpansion (same as `vec @ other`)
    //   `vec.cross(other)` -> Vec (3-D only)
    // -----------------------------------------------------------------------

    vec_cls.def(
        "dot",
        []( const TeVec& a, const TeVec& b ) {
            if ( a.size() != b.size() )
                throw std::invalid_argument( "Vec.dot: sizes must match" );
            if ( a.size() == 0 ) throw std::invalid_argument( "Vec.dot: empty vector" );
            DynTE out = a( 0 ) * b( 0 );
            for ( Eigen::Index i = 1; i < a.size(); ++i ) out += a( i ) * b( i );
            return out;
        },
        nb::arg( "other" ),
        "Dot product `Σ a[i] * b[i]` as a TaylorExpansion." );
    vec_cls.def(
        "dot",
        []( const TeVec& a, const Eigen::Ref< const Eigen::VectorXd >& b ) {
            if ( a.size() != b.size() )
                throw std::invalid_argument( "Vec.dot: sizes must match" );
            if ( a.size() == 0 ) throw std::invalid_argument( "Vec.dot: empty vector" );
            DynTE out = a( 0 ) * b( 0 );
            for ( Eigen::Index i = 1; i < a.size(); ++i ) out += a( i ) * b( i );
            return out;
        },
        nb::arg( "other" ),
        "Dot product against a 1-D numpy float array." );

    // Cross product for 3-D vectors.
    auto cross_te = []( const TeVec& a, const TeVec& b ) {
        if ( a.size() != 3 || b.size() != 3 )
            throw std::invalid_argument( "Vec.cross: both operands must have size 3" );
        TeVec out( 3 );
        out( 0 ) = a( 1 ) * b( 2 ) - a( 2 ) * b( 1 );
        out( 1 ) = a( 2 ) * b( 0 ) - a( 0 ) * b( 2 );
        out( 2 ) = a( 0 ) * b( 1 ) - a( 1 ) * b( 0 );
        return out;
    };
    vec_cls.def( "cross", cross_te, nb::arg( "other" ),
                 "3-D cross product `a × b`. Requires both vectors to have size 3." );
    vec_cls.def(
        "cross",
        []( const TeVec& a, const Eigen::Ref< const Eigen::VectorXd >& b ) {
            if ( a.size() != 3 || b.size() != 3 )
                throw std::invalid_argument( "Vec.cross: both operands must have size 3" );
            TeVec out( 3 );
            out( 0 ) = a( 1 ) * b( 2 ) - a( 2 ) * b( 1 );
            out( 1 ) = a( 2 ) * b( 0 ) - a( 0 ) * b( 2 );
            out( 2 ) = a( 0 ) * b( 1 ) - a( 1 ) * b( 0 );
            return out;
        },
        nb::arg( "other" ),
        "3-D cross product against a 1-D numpy float array of length 3." );

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
}

}  // namespace tax_py
