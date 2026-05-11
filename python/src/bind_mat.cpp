// SPDX-License-Identifier: BSD-3-Clause
//
// `tax.Mat` bindings — `Eigen::Matrix<DynTE, Dynamic, Dynamic>`. Indexing,
// slicing (block / row / col), Frobenius norm, arithmetic, matmul (incl.
// numpy interop), transpose.

#include "common.hpp"

namespace tax_py
{

void bind_mat( nb::class_< TeVec >& vec_cls, nb::class_< TeMat >& mat_cls )
{
    ( void )vec_cls;  // referenced implicitly through TeVec lambdas

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

    // ---- numpy interop matmul ----
    mat_cls.def( "__matmul__",
                 []( const TeMat& a, const Eigen::Ref< const Eigen::MatrixXd >& B ) {
                     // mat @ np_2d -> Mat(a.rows(), B.cols()).
                     if ( a.cols() != B.rows() )
                         throw std::invalid_argument( "Mat @ ndarray inner dim must match" );
                     TeMat out( a.rows(), B.cols() );
                     for ( Eigen::Index r = 0; r < a.rows(); ++r )
                         for ( Eigen::Index c = 0; c < B.cols(); ++c )
                         {
                             DynTE accum = a( r, 0 ) * B( 0, c );
                             for ( Eigen::Index k = 1; k < a.cols(); ++k )
                                 accum += a( r, k ) * B( k, c );
                             out( r, c ) = accum;
                         }
                     return out;
                 } );
    mat_cls.def( "__rmatmul__",
                 []( const TeMat& a, const Eigen::Ref< const Eigen::MatrixXd >& B ) {
                     // np_2d @ mat -> Mat(B.rows(), a.cols()).
                     if ( B.cols() != a.rows() )
                         throw std::invalid_argument( "ndarray @ Mat inner dim must match" );
                     TeMat out( B.rows(), a.cols() );
                     for ( Eigen::Index r = 0; r < B.rows(); ++r )
                         for ( Eigen::Index c = 0; c < a.cols(); ++c )
                         {
                             DynTE accum = B( r, 0 ) * a( 0, c );
                             for ( Eigen::Index k = 1; k < a.rows(); ++k )
                                 accum += B( r, k ) * a( k, c );
                             out( r, c ) = accum;
                         }
                     return out;
                 } );
    mat_cls.def( "__matmul__",
                 []( const TeMat& a, const Eigen::Ref< const Eigen::VectorXd >& v ) {
                     // mat @ np_1d -> Vec.
                     if ( a.cols() != v.size() )
                         throw std::invalid_argument( "Mat @ ndarray inner dim must match" );
                     TeVec out( a.rows() );
                     for ( Eigen::Index r = 0; r < a.rows(); ++r )
                     {
                         DynTE accum = a( r, 0 ) * v( 0 );
                         for ( Eigen::Index k = 1; k < a.cols(); ++k ) accum += a( r, k ) * v( k );
                         out( r ) = accum;
                     }
                     return out;
                 } );
    mat_cls.def( "__rmatmul__",
                 []( const TeMat& a, const Eigen::Ref< const Eigen::VectorXd >& v ) {
                     // np_1d @ mat (row-vector × matrix) -> Vec.
                     if ( v.size() != a.rows() )
                         throw std::invalid_argument( "ndarray @ Mat inner dim must match" );
                     TeVec out( a.cols() );
                     for ( Eigen::Index c = 0; c < a.cols(); ++c )
                     {
                         DynTE accum = v( 0 ) * a( 0, c );
                         for ( Eigen::Index k = 1; k < v.size(); ++k )
                             accum += v( k ) * a( k, c );
                         out( c ) = accum;
                     }
                     return out;
                 } );

    // Same numpy-ufunc opt-out as for Vec — defer to reflected ops.
    mat_cls.attr( "__array_ufunc__" ) = nb::none();

    // Frobenius norm `sqrt(Σ a(r,c)²)` returning a TaylorExpansion.
    mat_cls.def(
        "squared_norm",
        []( const TeMat& a ) {
            if ( a.size() == 0 )
                throw std::invalid_argument( "Mat.squared_norm: empty matrix" );
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
            return acc;
        },
        "Sum of squares of every element. Returns a TaylorExpansion." );
    mat_cls.def(
        "norm",
        []( const TeMat& a ) {
            if ( a.size() == 0 )
                throw std::invalid_argument( "Mat.norm: empty matrix" );
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
        "Frobenius norm `sqrt(Σ a(r,c)²)`. Returns a TaylorExpansion." );

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

}

}  // namespace tax_py
