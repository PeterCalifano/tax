#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <tax/storage/tte_static.hpp>
#include <tax/utils/combinatorics.hpp>
#include <tax/utils/degree_of.hpp>
#include <tax/utils/fwd.hpp>

namespace tax
{

/**
 * @brief Sparse-storage truncated Taylor polynomial in `M` variables up to order `N`.
 * @tparam T Scalar coefficient type.
 * @tparam N Maximum total polynomial order (non-negative).
 * @tparam M Number of variables (`>= 1`).
 *
 * Sibling type to `TaylorExpansionT<T, N, M>`. Stores only the nonzero
 * monomials as two parallel sorted vectors:
 *   - `indices()[k]`: flat graded-lex index of the k-th nonzero, strictly
 *     increasing
 *   - `values()[k]`:  the coefficient at that flat index
 *
 * Sorted-by-flat-index lets the binary operators merge in
 * `O(nnz(a) + nnz(b))` and gives SIMD-friendly sequential access on
 * `values()` when the index column is not needed. Intended for the case
 * where the user knows (or expects) `nnz << numMonomials(N, M)` — typically
 * structured perturbation problems where the operand is `c + a*x_i`.
 *
 * The class does not participate in tax's expression-template fusion:
 * operators are eager and return materialised `SparseTaylorExpansionT`
 * results. Mixed sparse↔dense arithmetic requires explicit conversion via
 * `toDense()` / `SparseTaylorExpansionT(dense)`.
 */
template < typename T, int N, int M = 1 >
class SparseTaylorExpansionT
{
   public:
    static_assert( N >= 0, "Sparse TTE order must be non-negative" );
    static_assert( M >= 1, "Sparse TTE variable count must be at least 1" );

    /// @brief Number of monomial slots available in this shape.
    static constexpr std::size_t nCoefficients = detail::numMonomials( N, M );

    static constexpr std::size_t order() noexcept { return std::size_t( N ); }
    static constexpr std::size_t size() noexcept { return std::size_t( M ); }

    using Dense = TaylorExpansionT< T, N, M >;
    using Input = std::array< T, M >;

    // -- Constructors ---------------------------------------------------------

    /// @brief Construct the zero polynomial (no nonzero monomials).
    SparseTaylorExpansionT() = default;

    /// @brief Construct the constant polynomial with value `c`.
    /*implicit*/ SparseTaylorExpansionT( T c )
    {
        if ( c != T{ 0 } )
        {
            idx_.push_back( 0 );
            val_.push_back( c );
        }
    }

    /// @brief Lift a dense TTE into sparse storage (drops exact zeros).
    explicit SparseTaylorExpansionT( const Dense& d )
    {
        idx_.reserve( nCoefficients );
        val_.reserve( nCoefficients );
        for ( std::size_t k = 0; k < nCoefficients; ++k )
        {
            const T v = d[k];
            if ( v != T{ 0 } )
            {
                idx_.push_back( std::uint16_t( k ) );
                val_.push_back( v );
            }
        }
    }

    // -- Variable factories ---------------------------------------------------

    /**
     * @brief Univariate variable expanded at `x0` (`x = x0 + 1*dx`).
     */
    [[nodiscard]] static SparseTaylorExpansionT variable( T x0 )
        requires( M == 1 )
    {
        SparseTaylorExpansionT s;
        if ( x0 != T{ 0 } )
        {
            s.idx_.push_back( 0 );
            s.val_.push_back( x0 );
        }
        if constexpr ( N >= 1 )
        {
            s.idx_.push_back( 1 );
            s.val_.push_back( T{ 1 } );
        }
        return s;
    }

    /**
     * @brief Coordinate variable `x_I` expanded at point `x0` (compile-time index).
     */
    template < int I >
    [[nodiscard]] static SparseTaylorExpansionT variable( const Input& x0 )
    {
        static_assert( I >= 0 && I < M, "Variable index out of range" );
        return variable( x0, I );
    }

    /**
     * @brief Coordinate variable `x_var` expanded at point `x0` (runtime index).
     */
    [[nodiscard]] static SparseTaylorExpansionT variable( const Input& x0, int var )
    {
        SparseTaylorExpansionT s;
        if ( x0[std::size_t( var )] != T{ 0 } )
        {
            s.idx_.push_back( 0 );
            s.val_.push_back( x0[std::size_t( var )] );
        }
        if constexpr ( N >= 1 )
        {
            MultiIndex< M > ei{};
            ei[std::size_t( var )] = 1;
            s.idx_.push_back( std::uint16_t( detail::flatIndex< M >( ei ) ) );
            s.val_.push_back( T{ 1 } );
        }
        // Variable factories only ever produce two slots; if both got
        // emitted the smaller (0) preceded the larger (e_I), so the array
        // is already sorted by construction.
        return s;
    }

    /**
     * @brief All coordinate variables `(x_0, ..., x_{M-1})` at expansion point `x0`.
     */
    [[nodiscard]] static auto variables( const Input& x0 )
    {
        return [&]< std::size_t... I >( std::index_sequence< I... > ) {
            return std::tuple{ variable< int( I ) >( x0 )... };
        }( std::make_index_sequence< std::size_t( M ) >{} );
    }

    /// @brief Constant polynomial with value `v`.
    [[nodiscard]] static SparseTaylorExpansionT constant( T v ) { return SparseTaylorExpansionT{ v }; }

    /// @brief Zero polynomial.
    [[nodiscard]] static SparseTaylorExpansionT zero() { return SparseTaylorExpansionT{}; }

    /// @brief One polynomial.
    [[nodiscard]] static SparseTaylorExpansionT one() { return SparseTaylorExpansionT{ T{ 1 } }; }

    // -- Conversion -----------------------------------------------------------

    /// @brief Materialise a dense `TaylorExpansionT<T, N, M>` from this sparse polynomial.
    [[nodiscard]] Dense toDense() const noexcept
    {
        Dense out;
        for ( std::size_t k = 0; k < val_.size(); ++k ) out[idx_[k]] = val_[k];
        return out;
    }

    // -- Element access -------------------------------------------------------

    /// @brief Number of nonzero monomials currently stored.
    [[nodiscard]] std::size_t nnz() const noexcept { return val_.size(); }

    /// @brief Constant term of the polynomial.
    [[nodiscard]] T value() const noexcept
    {
        if ( !idx_.empty() && idx_.front() == 0 ) return val_.front();
        return T{ 0 };
    }

    /**
     * @brief Coefficient at flat index `k` (graded-lex), zero if missing.
     * @details `O(log nnz)` binary search; intended for tests and inspection,
     *          not hot loops.
     */
    [[nodiscard]] T coeff( std::size_t k ) const noexcept
    {
        const auto it = std::lower_bound( idx_.begin(), idx_.end(), std::uint16_t( k ) );
        if ( it == idx_.end() || *it != std::uint16_t( k ) ) return T{ 0 };
        return val_[ std::size_t( it - idx_.begin() ) ];
    }

    /// @brief Coefficient at multi-index `alpha`, zero if missing.
    [[nodiscard]] T coeff( const MultiIndex< M >& alpha ) const noexcept
    {
        return coeff( detail::flatIndex< M >( alpha ) );
    }

    /// @brief Read-only view of the sorted flat indices of all nonzero slots.
    [[nodiscard]] std::span< const std::uint16_t > indices() const noexcept { return idx_; }
    /// @brief Read-only view of the values aligned with `indices()`.
    [[nodiscard]] std::span< const T > values() const noexcept { return val_; }

    // -- Builder primitives (used by kernels) ---------------------------------

    /// @brief Reserve capacity in both index and value buffers.
    void reserve( std::size_t cap )
    {
        idx_.reserve( cap );
        val_.reserve( cap );
    }

    /// @brief Drop all nonzero entries (becomes the zero polynomial).
    void clear() noexcept
    {
        idx_.clear();
        val_.clear();
    }

    /**
     * @brief Append a `(flat_index, value)` pair to the back of the storage.
     * @warning Caller is responsible for emitting in strictly increasing
     *          `flat_index` order and only with nonzero `value`. Used by
     *          kernels that already enforce these invariants by construction.
     */
    void emplaceBack( std::uint16_t flat_index, T value )
    {
        idx_.push_back( flat_index );
        val_.push_back( value );
    }

    // -- Arithmetic -----------------------------------------------------------

    /// @brief Sparse + sparse (two-pointer merge over sorted indices).
    [[nodiscard]] friend SparseTaylorExpansionT operator+( const SparseTaylorExpansionT& a,
                                                            const SparseTaylorExpansionT& b )
    {
        SparseTaylorExpansionT out;
        out.reserve( a.idx_.size() + b.idx_.size() );
        std::size_t i = 0;
        std::size_t j = 0;
        while ( i < a.idx_.size() && j < b.idx_.size() )
        {
            const auto ia = a.idx_[i];
            const auto ib = b.idx_[j];
            if ( ia < ib )
            {
                out.idx_.push_back( ia );
                out.val_.push_back( a.val_[i] );
                ++i;
            } else if ( ib < ia )
            {
                out.idx_.push_back( ib );
                out.val_.push_back( b.val_[j] );
                ++j;
            } else
            {
                const T sum = a.val_[i] + b.val_[j];
                if ( sum != T{ 0 } )
                {
                    out.idx_.push_back( ia );
                    out.val_.push_back( sum );
                }
                ++i;
                ++j;
            }
        }
        for ( ; i < a.idx_.size(); ++i )
        {
            out.idx_.push_back( a.idx_[i] );
            out.val_.push_back( a.val_[i] );
        }
        for ( ; j < b.idx_.size(); ++j )
        {
            out.idx_.push_back( b.idx_[j] );
            out.val_.push_back( b.val_[j] );
        }
        return out;
    }

    /// @brief Sparse - sparse (two-pointer merge over sorted indices).
    [[nodiscard]] friend SparseTaylorExpansionT operator-( const SparseTaylorExpansionT& a,
                                                            const SparseTaylorExpansionT& b )
    {
        SparseTaylorExpansionT out;
        out.reserve( a.idx_.size() + b.idx_.size() );
        std::size_t i = 0;
        std::size_t j = 0;
        while ( i < a.idx_.size() && j < b.idx_.size() )
        {
            const auto ia = a.idx_[i];
            const auto ib = b.idx_[j];
            if ( ia < ib )
            {
                out.idx_.push_back( ia );
                out.val_.push_back( a.val_[i] );
                ++i;
            } else if ( ib < ia )
            {
                out.idx_.push_back( ib );
                out.val_.push_back( -b.val_[j] );
                ++j;
            } else
            {
                const T diff = a.val_[i] - b.val_[j];
                if ( diff != T{ 0 } )
                {
                    out.idx_.push_back( ia );
                    out.val_.push_back( diff );
                }
                ++i;
                ++j;
            }
        }
        for ( ; i < a.idx_.size(); ++i )
        {
            out.idx_.push_back( a.idx_[i] );
            out.val_.push_back( a.val_[i] );
        }
        for ( ; j < b.idx_.size(); ++j )
        {
            out.idx_.push_back( b.idx_[j] );
            out.val_.push_back( -b.val_[j] );
        }
        return out;
    }

    /// @brief Unary negation.
    [[nodiscard]] friend SparseTaylorExpansionT operator-( const SparseTaylorExpansionT& a )
    {
        SparseTaylorExpansionT out;
        out.reserve( a.idx_.size() );
        for ( std::size_t k = 0; k < a.val_.size(); ++k )
        {
            out.idx_.push_back( a.idx_[k] );
            out.val_.push_back( -a.val_[k] );
        }
        return out;
    }

    /// @brief Scalar multiplication (drops to zero if `s == 0`).
    [[nodiscard]] friend SparseTaylorExpansionT operator*( const SparseTaylorExpansionT& a, T s )
    {
        if ( s == T{ 0 } ) return SparseTaylorExpansionT{};
        SparseTaylorExpansionT out;
        out.reserve( a.idx_.size() );
        for ( std::size_t k = 0; k < a.val_.size(); ++k )
        {
            out.idx_.push_back( a.idx_[k] );
            out.val_.push_back( a.val_[k] * s );
        }
        return out;
    }

    [[nodiscard]] friend SparseTaylorExpansionT operator*( T s, const SparseTaylorExpansionT& a )
    {
        return a * s;
    }

    /// @brief Scalar division (delegates to scalar multiplication by `1/s`).
    [[nodiscard]] friend SparseTaylorExpansionT operator/( const SparseTaylorExpansionT& a, T s )
    {
        return a * ( T{ 1 } / s );
    }

    /// @brief Add a scalar to the constant term.
    [[nodiscard]] friend SparseTaylorExpansionT operator+( const SparseTaylorExpansionT& a, T s )
    {
        if ( s == T{ 0 } ) return a;
        SparseTaylorExpansionT out;
        out.reserve( a.idx_.size() + 1 );
        if ( a.idx_.empty() || a.idx_.front() != 0 )
        {
            out.idx_.push_back( 0 );
            out.val_.push_back( s );
            for ( std::size_t k = 0; k < a.val_.size(); ++k )
            {
                out.idx_.push_back( a.idx_[k] );
                out.val_.push_back( a.val_[k] );
            }
        } else
        {
            const T sum = a.val_.front() + s;
            if ( sum != T{ 0 } )
            {
                out.idx_.push_back( 0 );
                out.val_.push_back( sum );
            }
            for ( std::size_t k = 1; k < a.val_.size(); ++k )
            {
                out.idx_.push_back( a.idx_[k] );
                out.val_.push_back( a.val_[k] );
            }
        }
        return out;
    }

    [[nodiscard]] friend SparseTaylorExpansionT operator+( T s, const SparseTaylorExpansionT& a )
    {
        return a + s;
    }

    /// @brief Subtract a scalar from the constant term.
    [[nodiscard]] friend SparseTaylorExpansionT operator-( const SparseTaylorExpansionT& a, T s )
    {
        return a + ( -s );
    }

    /// @brief Scalar minus polynomial.
    [[nodiscard]] friend SparseTaylorExpansionT operator-( T s, const SparseTaylorExpansionT& a )
    {
        return ( -a ) + s;
    }

   private:
    std::vector< std::uint16_t > idx_;
    std::vector< T > val_;
};

/// @brief Univariate sparse TE alias (`double`, order `N`, one variable).
template < int N >
using STE = SparseTaylorExpansionT< double, N, 1 >;

/// @brief Multivariate sparse TE alias (`double`, order `N`, `M` variables).
template < int N, int M >
using STEn = SparseTaylorExpansionT< double, N, M >;

}  // namespace tax
