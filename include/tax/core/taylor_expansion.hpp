#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <tax/core/concepts.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/storage/dense.hpp>
#include <tax/core/storage/sparse.hpp>
#include <tax/la/types.hpp>

namespace tax
{

// Primary template (forward declaration for partial specialisations).
template < typename T, int N, int M = 1, typename Storage = storage::Dense >
class TaylorExpansion;

// ---------------------------------------------------------------------------
// Dense specialisation
// ---------------------------------------------------------------------------

/// A truncated Taylor expansion in M variables of order N with dense storage.
template < typename T, int N, int M >
    requires Scalar< T >
class TaylorExpansion< T, N, M, storage::Dense >
{
   public:
    static_assert( N >= 0, "TaylorExpansion order must be non-negative" );
    static_assert( M >= 1, "TaylorExpansion variable count must be at least 1" );

    // ------------------------------------------------------------------
    // Associated types
    // ------------------------------------------------------------------
    using scalar_type = T;
    using container_t = storage::DenseContainer< T, N, M >;
    using Input = std::array< T, std::size_t( M ) >;
    using Data = Coeffs< T, N, M >;

    // ------------------------------------------------------------------
    // Compile-time properties
    // ------------------------------------------------------------------
    static constexpr int order_v = N;
    static constexpr int vars_v = M;
    static constexpr std::size_t nCoefficients = numMonomials( N, M );

    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Zero-initialise all coefficients.
    constexpr TaylorExpansion() noexcept = default;

    /// Constant expansion: value `val`, all higher-order coefficients zero.
    /*implicit*/ constexpr TaylorExpansion( T val ) noexcept { c_.set( 0, val ); }

    /// Construct directly from a raw coefficient array.
    explicit constexpr TaylorExpansion( Data c ) noexcept : c_{ c } {}

    // ------------------------------------------------------------------
    // Named factories
    // ------------------------------------------------------------------

    [[nodiscard]] static constexpr TaylorExpansion zero() noexcept { return {}; }

    [[nodiscard]] static constexpr TaylorExpansion constant( T v ) noexcept
    {
        return TaylorExpansion{ v };
    }

    /// Univariate variable: `x = x0 + 1*dx`.
    [[nodiscard]] static constexpr TaylorExpansion variable( T x0 ) noexcept
        requires( M == 1 )
    {
        TaylorExpansion r{ x0 };
        if constexpr ( N >= 1 ) r.c_.set( 1, T{ 1 } );
        return r;
    }

    /// Multivariate variable: the I-th coordinate variable at point `p`.
    template < int I >
    [[nodiscard]] static constexpr TaylorExpansion variable( const Input& p ) noexcept
        requires( M >= 1 && I >= 0 && I < M )
    {
        TaylorExpansion r{};
        r.c_.set( 0, p[std::size_t( I )] );
        if constexpr ( N >= 1 )
        {
            MultiIndex< M > alpha{};
            alpha[std::size_t( I )] = 1;
            r.c_.set( flatIndex< M >( alpha ), T{ 1 } );
        }
        return r;
    }

    /// Runtime-indexed coordinate variable: `x_i = x0 + 1*dx_i`.
    [[nodiscard]] static TaylorExpansion variable( T x0, int var_idx )
    {
        if ( var_idx < 0 || var_idx >= M )
            throw std::out_of_range( "variable(): var_idx out of range" );
        TaylorExpansion r{};
        r.c_.set( 0, x0 );
        if constexpr ( N >= 1 )
        {
            MultiIndex< M > alpha{};
            alpha[std::size_t( var_idx )] = 1;
            r.c_.set( flatIndex< M >( alpha ), T{ 1 } );
        }
        return r;
    }

    // ------------------------------------------------------------------
    // Element access
    // ------------------------------------------------------------------

    /// Constant (zeroth) coefficient, i.e. f(x0).
    [[nodiscard]] constexpr T value() const noexcept { return c_.value(); }

    /// Read coefficient at flat index `k`.
    [[nodiscard]] constexpr T operator[]( std::size_t k ) const noexcept { return c_[k]; }

    /// Write coefficient at flat index `k`.
    [[nodiscard]] constexpr T& operator[]( std::size_t k ) noexcept { return c_[k]; }

    /// Runtime multi-index coefficient lookup.
    [[nodiscard]] constexpr T coeff( const MultiIndex< M >& alpha ) const noexcept
    {
        return c_[flatIndex< M >( alpha )];
    }

    /// Compile-time multi-index coefficient lookup.
    template < int... Alpha >
    [[nodiscard]] constexpr T coeff() const noexcept
    {
        static_assert( sizeof...( Alpha ) == std::size_t( M ),
                       "coeff<Alpha...>(): arity must match variable count" );
        static_assert( ( ( Alpha >= 0 ) && ... ), "coeff<Alpha...>(): negative exponent" );
        constexpr int total = ( Alpha + ... + 0 );
        static_assert( total <= N, "coeff<Alpha...>(): total degree exceeds N" );
        constexpr MultiIndex< M > a{ Alpha... };
        return c_[flatIndex< M >( a )];
    }

    // ------------------------------------------------------------------
    // Derivative accessors (apply k! scaling to raw coefficients)
    // ------------------------------------------------------------------

    /// Runtime partial derivative value `d^|alpha| f / dx^alpha` at x0.
    [[nodiscard]] constexpr T derivative( const MultiIndex< M >& alpha ) const noexcept
    {
        // Accumulate the factorial in T: std::size_t overflows at 21! on 64-bit,
        // silently corrupting high-order derivatives.
        T fac = T{ 1 };
        for ( int i = 0; i < M; ++i )
            for ( int j = 1; j <= alpha[std::size_t( i )]; ++j ) fac *= T( j );
        return coeff( alpha ) * fac;
    }

    /// Compile-time partial derivative value.
    template < int... Alpha >
    [[nodiscard]] constexpr T derivative() const noexcept
    {
        static_assert( sizeof...( Alpha ) == std::size_t( M ),
                       "derivative<Alpha...>(): arity must match variable count" );
        static_assert( ( ( Alpha >= 0 ) && ... ), "derivative<Alpha...>(): negative exponent" );
        constexpr int total = ( Alpha + ... + 0 );
        static_assert( total <= N, "derivative<Alpha...>(): total degree exceeds N" );

        // Accumulate in T to avoid std::size_t factorial overflow (21! > UINT64_MAX).
        constexpr auto factorial = []( int n ) constexpr noexcept -> T {
            T r = T{ 1 };
            for ( int i = 2; i <= n; ++i ) r *= T( i );
            return r;
        };
        constexpr T fac = ( factorial( Alpha ) * ... * T( 1 ) );
        return coeff< Alpha... >() * fac;
    }

    // ------------------------------------------------------------------
    // Polynomial evaluation at a displacement from expansion point
    // ------------------------------------------------------------------

    /// Evaluate the polynomial at displacement `dx` from the expansion point.
    [[nodiscard]] constexpr T eval( const Input& dx ) const noexcept
    {
        if constexpr ( M == 1 )
        {
            // Horner's method
            T result = c_[N];
            for ( int i = N - 1; i >= 0; --i ) result = result * dx[0] + c_[i];
            return result;
        } else
        {
            T result{};

            // Power table pw[i][j] = dx_i^j: each monomial then costs one
            // multiply via the partial product carried down the recursion,
            // instead of rebuilding dx^alpha from |alpha| factors.
            std::array< std::array< T, std::size_t( N ) + 1 >, std::size_t( M ) > pw{};
            for ( int i = 0; i < M; ++i )
            {
                pw[std::size_t( i )][0] = T{ 1 };
                for ( int j = 1; j <= N; ++j )
                    pw[std::size_t( i )][std::size_t( j )] =
                        pw[std::size_t( i )][std::size_t( j - 1 )] * dx[std::size_t( i )];
            }

            // Degree-by-degree accumulation: enumerate all monomials of total degree d
            // and accumulate c_alpha * dx^alpha.
            auto accumulate = [&]( auto& self, int var, int rem, MultiIndex< M > alpha,
                                   T partial ) constexpr -> void {
                if ( var == M - 1 )
                {
                    alpha[std::size_t( var )] = rem;
                    result += c_[flatIndex< M >( alpha )] * partial *
                              pw[std::size_t( var )][std::size_t( rem )];
                    return;
                }
                for ( int k = rem; k >= 0; --k )
                {
                    auto a2 = alpha;
                    a2[std::size_t( var )] = k;
                    self( self, var + 1, rem - k, a2,
                          partial * pw[std::size_t( var )][std::size_t( k )] );
                }
            };

            for ( int d = 0; d <= N; ++d )
                accumulate( accumulate, 0, d, MultiIndex< M >{}, T{ 1 } );
            return result;
        }
    }

    /// Evaluate the polynomial at displacement given as an Eigen vector.
    template < typename DxDerived >
    [[nodiscard]] T eval( const Eigen::MatrixBase< DxDerived >& dx ) const
    {
        static_assert(
            DxDerived::SizeAtCompileTime == M || DxDerived::SizeAtCompileTime == Eigen::Dynamic,
            "eval(Eigen): size must match number of variables M" );
        Input p{};
        for ( int i = 0; i < M; ++i ) p[std::size_t( i )] = T( dx( i ) );
        return eval( p );
    }

    // ------------------------------------------------------------------
    // Differentiation and integration
    // ------------------------------------------------------------------

    /// Partial derivative polynomial with respect to variable `I`.
    template < int I >
    [[nodiscard]] constexpr TaylorExpansion deriv() const noexcept
        requires( I >= 0 && I < M )
    {
        Data out{};
        for ( std::size_t i = 0; i < nCoefficients; ++i )
        {
            if ( c_[i] == T{} ) continue;
            auto alpha = unflatIndex< M >( i );
            const int exp = alpha[std::size_t( I )];
            if ( exp == 0 ) continue;
            alpha[std::size_t( I )] = exp - 1;
            out[flatIndex< M >( alpha )] += c_[i] * T( exp );
        }
        return TaylorExpansion{ out };
    }

    /// Partial derivative polynomial with respect to variable `var`. Throws std::out_of_range if
    /// `var` is outside [0, M).
    [[nodiscard]] TaylorExpansion deriv( int var ) const
    {
        if ( var < 0 || var >= M )
            throw std::out_of_range( "tax::TaylorExpansion::deriv(var): var must be in [0, M)" );
        Data out{};
        for ( std::size_t i = 0; i < nCoefficients; ++i )
        {
            if ( c_[i] == T{} ) continue;
            auto alpha = unflatIndex< M >( i );
            const int exp = alpha[std::size_t( var )];
            if ( exp == 0 ) continue;
            alpha[std::size_t( var )] = exp - 1;
            out[flatIndex< M >( alpha )] += c_[i] * T( exp );
        }
        return TaylorExpansion{ out };
    }

    /// Indefinite integral polynomial with respect to variable `I`.
    template < int I >
    [[nodiscard]] constexpr TaylorExpansion integ() const noexcept
        requires( I >= 0 && I < M )
    {
        Data out{};
        for ( std::size_t i = 0; i < nCoefficients; ++i )
        {
            if ( c_[i] == T{} ) continue;
            auto alpha = unflatIndex< M >( i );
            if ( totalDegree( alpha ) >= N ) continue;  // would exceed order N
            const int exp = alpha[std::size_t( I )];
            alpha[std::size_t( I )] = exp + 1;
            out[flatIndex< M >( alpha )] = c_[i] / T( exp + 1 );
        }
        return TaylorExpansion{ out };
    }

    /// Indefinite integral polynomial with respect to variable `var`. Throws std::out_of_range if
    /// `var` is outside [0, M).
    [[nodiscard]] TaylorExpansion integ( int var ) const
    {
        if ( var < 0 || var >= M )
            throw std::out_of_range( "tax::TaylorExpansion::integ(var): var must be in [0, M)" );
        Data out{};
        for ( std::size_t i = 0; i < nCoefficients; ++i )
        {
            if ( c_[i] == T{} ) continue;
            auto alpha = unflatIndex< M >( i );
            if ( totalDegree( alpha ) >= N ) continue;  // would exceed order N
            const int exp = alpha[std::size_t( var )];
            alpha[std::size_t( var )] = exp + 1;
            out[flatIndex< M >( alpha )] = c_[i] / T( exp + 1 );
        }
        return TaylorExpansion{ out };
    }

    // ------------------------------------------------------------------
    // Truncation
    // ------------------------------------------------------------------

    /// Order-reducing truncation: drop monomials of degree > N2, yielding a lower-order expansion.
    template < int N2 >
    [[nodiscard]] constexpr TaylorExpansion< T, N2, M, storage::Dense > truncate() const noexcept
        requires( N2 >= 0 && N2 <= N )
    {
        typename TaylorExpansion< T, N2, M, storage::Dense >::Data out{};
        // Graded-lex: degree-<=N2 monomials are a shared prefix of the order-N layout.
        for ( std::size_t k = 0; k < numMonomials( N2, M ); ++k ) out[k] = c_[k];
        return TaylorExpansion< T, N2, M, storage::Dense >{ out };
    }

    /// Same-order truncation: zero every coefficient of total degree > d (d>=N copies, d<0 zeroes).
    [[nodiscard]] constexpr TaylorExpansion truncate( int d ) const noexcept
    {
        if ( d >= N ) return *this;
        Data out{};
        if ( d >= 0 )
            for ( std::size_t k = 0; k < numMonomials( d, M ); ++k ) out[k] = c_[k];
        return TaylorExpansion{ out };
    }

    // ------------------------------------------------------------------
    // Gradient and Hessian (require Eigen/Core, already included above)
    // ------------------------------------------------------------------

    /// Compute the gradient vector `[df/dx_0, ..., df/dx_{M-1}]` at the expansion point.
    [[nodiscard]] tax::la::VecNT< M, T > gradient() const noexcept
    {
        tax::la::VecNT< M, T > g;
        MultiIndex< M > alpha{};
        for ( int i = 0; i < M; ++i )
        {
            alpha[std::size_t( i )] = 1;
            g( i ) = derivative( alpha );
            alpha[std::size_t( i )] = 0;
        }
        return g;
    }

    /// Compute the Hessian matrix `H(i,j) = d^2 f / (dx_i dx_j)` at the expansion point.
    [[nodiscard]] tax::la::MatNT< M, T > hessian() const noexcept
    {
        tax::la::MatNT< M, T > H;
        for ( int i = 0; i < M; ++i )
        {
            for ( int j = 0; j < M; ++j )
            {
                MultiIndex< M > alpha{};
                alpha[std::size_t( i )] += 1;
                alpha[std::size_t( j )] += 1;
                H( i, j ) = derivative( alpha );
            }
        }
        return H;
    }

    // ------------------------------------------------------------------
    // Container access
    // ------------------------------------------------------------------

    [[nodiscard]] constexpr const container_t& container() const noexcept { return c_; }
    [[nodiscard]] constexpr container_t& container() noexcept { return c_; }

    /// Raw coefficient array — convenience accessor used by kernels.
    [[nodiscard]] constexpr const Data& coefficients() const noexcept { return c_.data; }
    [[nodiscard]] constexpr Data& coefficients() noexcept { return c_.data; }

   private:
    container_t c_{};
};

// ---------------------------------------------------------------------------
// Convenience aliases  (dense)
// ---------------------------------------------------------------------------

/// `TE<N>` — univariate `double` expansion of order N.
template < int N, int M = 1 >
using TE = TaylorExpansion< double, N, M, storage::Dense >;

/// `TEn<N, M>` — explicit M-variate alias, same as `TE<N, M>`.
template < int N, int M >
using TEn = TaylorExpansion< double, N, M, storage::Dense >;

// ---------------------------------------------------------------------------
// Sparse specialisation
// ---------------------------------------------------------------------------

/// A truncated Taylor expansion in M variables of order N with sparse storage.
template < typename T, int N, int M >
    requires Scalar< T >
class TaylorExpansion< T, N, M, storage::Sparse >
{
   public:
    static_assert( N >= 0, "TaylorExpansion<Sparse> order must be non-negative" );
    static_assert( M >= 1, "TaylorExpansion<Sparse> variable count must be at least 1" );

    // ------------------------------------------------------------------
    // Associated types
    // ------------------------------------------------------------------
    using scalar_type = T;
    using container_t = storage::SparseContainer< T, N, M >;
    using Input = std::array< T, std::size_t( M ) >;
    using Dense = TaylorExpansion< T, N, M, storage::Dense >;

    // ------------------------------------------------------------------
    // Compile-time properties
    // ------------------------------------------------------------------
    static constexpr int order_v = N;
    static constexpr int vars_v = M;
    /// Dense-equivalent upper bound on the number of monomials.
    static constexpr std::size_t nCoefficients = numMonomials( N, M );

    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Zero-polynomial — no nonzero monomials.
    constexpr TaylorExpansion() = default;

    /// Constant polynomial with value `c`.
    /*implicit*/ TaylorExpansion( T c )
    {
        if ( c != T{ 0 } ) c_.set( 0, c );
    }

    /// Lift a dense polynomial into sparse storage (drops exact zeros).
    explicit TaylorExpansion( const Dense& d )
    {
        for ( std::size_t k = 0; k < Dense::nCoefficients; ++k )
        {
            if ( d[k] != T{ 0 } ) c_.set( k, d[k] );
        }
    }

    // ------------------------------------------------------------------
    // Named factories
    // ------------------------------------------------------------------

    [[nodiscard]] static TaylorExpansion zero() noexcept { return {}; }
    [[nodiscard]] static TaylorExpansion constant( T c ) { return TaylorExpansion{ c }; }

    /// Univariate variable: `x = x0 + 1*dx`.
    [[nodiscard]] static TaylorExpansion variable( T x0 ) noexcept
        requires( M == 1 )
    {
        TaylorExpansion r;
        if ( x0 != T{ 0 } ) r.c_.set( 0, x0 );
        if constexpr ( N >= 1 ) r.c_.set( 1, T{ 1 } );
        return r;
    }

    /// Coordinate variable `x_I` at expansion point `p` (compile-time index).
    template < int I >
    [[nodiscard]] static TaylorExpansion variable( const Input& p ) noexcept
        requires( M >= 1 && I >= 0 && I < M )
    {
        TaylorExpansion r;
        if ( p[std::size_t( I )] != T{ 0 } ) r.c_.set( 0, p[std::size_t( I )] );
        if constexpr ( N >= 1 )
        {
            MultiIndex< M > alpha{};
            alpha[std::size_t( I )] = 1;
            r.c_.set( flatIndex< M >( alpha ), T{ 1 } );
        }
        return r;
    }

    // ------------------------------------------------------------------
    // Element access
    // ------------------------------------------------------------------

    /// Number of currently stored nonzero monomials.
    [[nodiscard]] std::size_t nnz() const noexcept { return c_.nnz(); }

    /// Constant (zeroth) coefficient; returns 0 if the slot is absent.
    [[nodiscard]] T value() const noexcept { return c_.value(); }

    /// Runtime multi-index coefficient lookup (O(log nnz)).
    [[nodiscard]] T coeff( const MultiIndex< M >& alpha ) const noexcept
    {
        return c_.coeffAtFlat( flatIndex< M >( alpha ) );
    }

    /// Compile-time multi-index coefficient lookup (O(log nnz)).
    template < int... Alpha >
    [[nodiscard]] T coeff() const noexcept
    {
        static_assert( sizeof...( Alpha ) == std::size_t( M ),
                       "coeff<Alpha...>(): arity must match variable count" );
        static_assert( ( ( Alpha >= 0 ) && ... ), "coeff<Alpha...>(): negative exponent" );
        constexpr int total = ( Alpha + ... + 0 );
        static_assert( total <= N, "coeff<Alpha...>(): total degree exceeds N" );
        constexpr MultiIndex< M > a{ Alpha... };
        return c_.coeffAtFlat( flatIndex< M >( a ) );
    }

    // ------------------------------------------------------------------
    // Derivative accessors (apply k! scaling to raw coefficients)
    // ------------------------------------------------------------------

    /// Runtime partial derivative value `d^|alpha| f / dx^alpha` at x0.
    [[nodiscard]] T derivative( const MultiIndex< M >& alpha ) const noexcept
    {
        // Accumulate in T: std::size_t overflows at 21! (see dense variant).
        T fac = T{ 1 };
        for ( int i = 0; i < M; ++i )
            for ( int j = 1; j <= alpha[std::size_t( i )]; ++j ) fac *= T( j );
        return coeff( alpha ) * fac;
    }

    /// Compile-time partial derivative value.
    template < int... Alpha >
    [[nodiscard]] T derivative() const noexcept
    {
        static_assert( sizeof...( Alpha ) == std::size_t( M ) );
        static_assert( ( ( Alpha >= 0 ) && ... ) );
        constexpr int total = ( Alpha + ... + 0 );
        static_assert( total <= N );

        // Accumulate in T to avoid std::size_t factorial overflow (21! > UINT64_MAX).
        constexpr auto factorial = []( int n ) constexpr noexcept -> T {
            T r = T{ 1 };
            for ( int i = 2; i <= n; ++i ) r *= T( i );
            return r;
        };
        constexpr T fac = ( factorial( Alpha ) * ... * T( 1 ) );
        return coeff< Alpha... >() * fac;
    }

    // ------------------------------------------------------------------
    // Sparse-specific accessors
    // ------------------------------------------------------------------

    /// Read-only view of the sorted flat indices of all nonzero slots.
    [[nodiscard]] std::span< const storage::flat_index_t > support() const noexcept
    {
        return c_.support();
    }

    /// Read-only view of the coefficient values aligned with `support()`.
    [[nodiscard]] std::span< const T > values() const noexcept { return c_.values(); }

    // ------------------------------------------------------------------
    // Conversion
    // ------------------------------------------------------------------

    /// Materialise a dense `TaylorExpansion<T, N, M, Dense>` from this sparse polynomial.
    [[nodiscard]] Dense dense() const noexcept
    {
        Dense r;
        c_.forEachNonzero( [&]( std::size_t k, T v ) { r[k] = v; } );
        return r;
    }

    // ------------------------------------------------------------------
    // Truncation
    // ------------------------------------------------------------------

    /// Order-reducing truncation: drop monomials of degree > N2, yielding a lower-order expansion.
    template < int N2 >
    [[nodiscard]] TaylorExpansion< T, N2, M, storage::Sparse > truncate() const noexcept
        requires( N2 >= 0 && N2 <= N )
    {
        return truncatedBelow< TaylorExpansion< T, N2, M, storage::Sparse > >(
            numMonomials( N2, M ) );
    }

    /// Same-order truncation: zero every coefficient of total degree > d (d>=N copies, d<0 zeroes).
    [[nodiscard]] TaylorExpansion truncate( int d ) const noexcept
    {
        if ( d >= N ) return *this;
        return truncatedBelow< TaylorExpansion >( d >= 0 ? numMonomials( d, M ) : 0 );
    }

    // ------------------------------------------------------------------
    // Container access
    // ------------------------------------------------------------------

    [[nodiscard]] const container_t& container() const noexcept { return c_; }
    [[nodiscard]] container_t& container() noexcept { return c_; }

   private:
    /// Copy the sorted prefix of stored slots with flat index < `limit` into a fresh result.
    template < typename Result >
    [[nodiscard]] Result truncatedBelow( std::size_t limit ) const noexcept
    {
        Result r;
        const auto cap = storage::flat_index_t( limit );
        auto& oi = r.container().rawIndices();
        auto& ov = r.container().rawValues();
        const auto sup = support();
        const auto vals = values();
        for ( std::size_t i = 0; i < sup.size(); ++i )
        {
            if ( sup[i] >= cap ) break;  // support is sorted ascending
            oi.push_back( sup[i] );
            ov.push_back( vals[i] );
        }
        return r;
    }

    container_t c_;
};

// ---------------------------------------------------------------------------
// Convenience aliases  (sparse)
// ---------------------------------------------------------------------------

/// `STE<N>` — univariate sparse `double` expansion of order N.
/// `STE<N, M>` — M-variate sparse `double` expansion of order N.
template < int N, int M = 1 >
using STE = TaylorExpansion< double, N, M, storage::Sparse >;

// ---------------------------------------------------------------------------
// Conversion helper: dense -> sparse
// ---------------------------------------------------------------------------

/// Convert a dense polynomial to sparse storage (drops exact zeros).
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, storage::Sparse > sparse(
    const TaylorExpansion< T, N, M, storage::Dense >& d ) noexcept
{
    return TaylorExpansion< T, N, M, storage::Sparse >( d );
}

}  // namespace tax
