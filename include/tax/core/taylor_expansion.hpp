#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <tax/core/concepts.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/storage/dense.hpp>
#include <tax/core/storage/sparse.hpp>
#include <Eigen/Core>

namespace tax
{

// Primary template (forward declaration for partial specialisations).
template < typename T, int N, int M = 1, typename Storage = storage::Dense >
class TaylorExpansion;

// ---------------------------------------------------------------------------
// Dense specialisation
// ---------------------------------------------------------------------------

/**
 * @brief A truncated Taylor expansion in M variables of order N with dense storage.
 *
 * Coefficients are stored in graded-lexicographic order in a `std::array` of
 * size `numMonomials(N, M)` (stack-allocated, no heap).
 *
 * @tparam T  Scalar type (must satisfy `tax::Scalar`).
 * @tparam N  Truncation order (non-negative compile-time integer).
 * @tparam M  Number of independent variables (>= 1).
 */
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
    using Input       = std::array< T, std::size_t( M ) >;
    using Data        = Coeffs< T, N, M >;

    // ------------------------------------------------------------------
    // Compile-time properties
    // ------------------------------------------------------------------
    static constexpr int         order_v        = N;
    static constexpr int         vars_v         = M;
    static constexpr std::size_t nCoefficients  = numMonomials( N, M );

    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// @brief Zero-initialise all coefficients.
    constexpr TaylorExpansion() noexcept = default;

    /// @brief Constant expansion: value `val`, all higher-order coefficients zero.
    /*implicit*/ constexpr TaylorExpansion( T val ) noexcept { c_.set( 0, val ); }

    /// @brief Construct directly from a raw coefficient array.
    explicit constexpr TaylorExpansion( Data c ) noexcept : c_{ c } {}

    // ------------------------------------------------------------------
    // Named factories
    // ------------------------------------------------------------------

    [[nodiscard]] static constexpr TaylorExpansion zero() noexcept { return {}; }

    [[nodiscard]] static constexpr TaylorExpansion constant( T v ) noexcept
    {
        return TaylorExpansion{ v };
    }

    /**
     * @brief Univariate variable: `x = x0 + 1*dx`.
     * @note Only available when `M == 1`.
     */
    [[nodiscard]] static constexpr TaylorExpansion variable( T x0 ) noexcept
        requires( M == 1 )
    {
        TaylorExpansion r{ x0 };
        if constexpr ( N >= 1 )
            r.c_.set( 1, T{ 1 } );
        return r;
    }

    /**
     * @brief Multivariate variable: the I-th coordinate variable at point `p`.
     *
     * Sets `coeff(e_I) = 1` where `e_I` is the I-th unit multi-index.
     *
     * @tparam I  Variable index in `[0, M)`.
     * @param  p  Expansion point.
     */
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

    // ------------------------------------------------------------------
    // Element access
    // ------------------------------------------------------------------

    /// @brief Constant (zeroth) coefficient, i.e. f(x0).
    [[nodiscard]] constexpr T value() const noexcept { return c_.value(); }

    /// @brief Read coefficient at flat index `k`.
    [[nodiscard]] constexpr T operator[]( std::size_t k ) const noexcept { return c_[k]; }

    /// @brief Write coefficient at flat index `k`.
    [[nodiscard]] constexpr T& operator[]( std::size_t k ) noexcept { return c_[k]; }

    /// @brief Runtime multi-index coefficient lookup.
    [[nodiscard]] constexpr T coeff( const MultiIndex< M >& alpha ) const noexcept
    {
        return c_[flatIndex< M >( alpha )];
    }

    /**
     * @brief Compile-time multi-index coefficient lookup.
     *
     * Usage: `f.coeff<2, 0>()` retrieves the coefficient of `x^2 y^0`.
     */
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

    /**
     * @brief Runtime partial derivative value `d^|alpha| f / dx^alpha` at x0.
     *
     * Multiplies the stored Taylor coefficient by the multinomial factorial
     * `alpha!  =  alpha_0! * alpha_1! * ... * alpha_{M-1}!`.
     */
    [[nodiscard]] constexpr T derivative( const MultiIndex< M >& alpha ) const noexcept
    {
        std::size_t fac = 1;
        for ( int i = 0; i < M; ++i )
            for ( int j = 1; j <= alpha[std::size_t( i )]; ++j )
                fac *= std::size_t( j );
        return coeff( alpha ) * T( fac );
    }

    /**
     * @brief Compile-time partial derivative value.
     *
     * Usage: `f.derivative<2>()` gives `d^2 f / dx^2` at x0 (univariate).
     *        `f.derivative<1, 0>()` gives `df/dx` (multivariate, M==2).
     */
    template < int... Alpha >
    [[nodiscard]] constexpr T derivative() const noexcept
    {
        static_assert( sizeof...( Alpha ) == std::size_t( M ),
                       "derivative<Alpha...>(): arity must match variable count" );
        static_assert( ( ( Alpha >= 0 ) && ... ),
                       "derivative<Alpha...>(): negative exponent" );
        constexpr int total = ( Alpha + ... + 0 );
        static_assert( total <= N, "derivative<Alpha...>(): total degree exceeds N" );

        constexpr auto factorial = []( int n ) constexpr noexcept -> std::size_t
        {
            std::size_t r = 1;
            for ( int i = 2; i <= n; ++i )
                r *= std::size_t( i );
            return r;
        };
        constexpr std::size_t fac = ( factorial( Alpha ) * ... * std::size_t( 1 ) );
        return coeff< Alpha... >() * T( fac );
    }

    // ------------------------------------------------------------------
    // Polynomial evaluation at a displacement from expansion point
    // ------------------------------------------------------------------

    /**
     * @brief Evaluate the polynomial at displacement `dx` from the expansion point.
     *
     * Computes `f(x0 + dx)` truncated to order N using the Horner scheme for
     * univariate (M == 1) and a degree-by-degree monomial-accumulation scheme for
     * multivariate cases.
     *
     * @param dx Displacement vector (same layout as `Input`).
     * @return `f(x0 + dx)` as a scalar of type `T`.
     */
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

            // Degree-by-degree accumulation: enumerate all monomials of total degree d,
            // compute dx^alpha and accumulate c_alpha * dx^alpha.
            auto accumulate = [&]( auto& self, int var, int rem,
                                   MultiIndex< M > alpha ) constexpr -> void
            {
                if ( var == M - 1 )
                {
                    alpha[std::size_t( var )] = rem;
                    T monomial{ 1 };
                    for ( int i = 0; i < M; ++i )
                        for ( int j = 0; j < alpha[std::size_t( i )]; ++j ) monomial *= dx[std::size_t( i )];
                    result += c_[flatIndex< M >( alpha )] * monomial;
                    return;
                }
                for ( int k = rem; k >= 0; --k )
                {
                    auto a2               = alpha;
                    a2[std::size_t( var )] = k;
                    self( self, var + 1, rem - k, a2 );
                }
            };

            for ( int d = 0; d <= N; ++d ) accumulate( accumulate, 0, d, MultiIndex< M >{} );
            return result;
        }
    }

    /**
     * @brief Evaluate the polynomial at displacement given as an Eigen vector.
     *
     * Converts the Eigen vector to an `Input` array and delegates to `eval(Input)`.
     *
     * @tparam DxDerived Eigen expression type with `SizeAtCompileTime == M`.
     * @param  dx        Displacement Eigen vector.
     * @return `f(x0 + dx)` as a scalar of type `T`.
     */
    template < typename DxDerived >
    [[nodiscard]] T eval( const Eigen::MatrixBase< DxDerived >& dx ) const noexcept
    {
        static_assert( DxDerived::SizeAtCompileTime == M || DxDerived::SizeAtCompileTime == Eigen::Dynamic,
                       "eval(Eigen): size must match number of variables M" );
        Input p{};
        for ( int i = 0; i < M; ++i ) p[std::size_t( i )] = T( dx( i ) );
        return eval( p );
    }

    // ------------------------------------------------------------------
    // Differentiation and integration
    // ------------------------------------------------------------------

    /**
     * @brief Partial derivative polynomial with respect to variable `I`.
     *
     * @tparam I Variable index (0-based, must be in `[0, M)`).
     * @details For each term `c_alpha * x^alpha` with `alpha[I] > 0`, contributes
     *          `c_alpha * alpha[I] * x^(alpha - e_I)` to the result.
     *          Terms where `alpha[I] == 0` vanish.  Shape (N, M) is preserved.
     * @return New polynomial representing `d/dx_I` of this polynomial.
     */
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

    /**
     * @brief Partial derivative polynomial with respect to variable `var`.
     *
     * @param var Variable index (0-based, must be in `[0, M)`).
     * @details Runtime-index overload of `deriv<I>()`.
     * @return New polynomial representing `d/dx_var` of this polynomial.
     * @throws std::out_of_range if `var < 0` or `var >= M`.
     */
    [[nodiscard]] TaylorExpansion deriv( int var ) const
    {
        if ( var < 0 || var >= M )
            throw std::out_of_range(
                "tax::TaylorExpansion::deriv(var): var must be in [0, M)" );
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

    /**
     * @brief Indefinite integral polynomial with respect to variable `I`.
     *
     * @tparam I Variable index (0-based, must be in `[0, M)`).
     * @details For each term `c_alpha * x^alpha` with `|alpha| < N`, contributes
     *          `c_alpha / (alpha[I] + 1) * x^(alpha + e_I)` to the result.
     *          Terms of degree `N` are dropped (result stays order N).
     *          The constant of integration is zero.
     * @return New polynomial representing `integral ... dx_I` of this polynomial.
     */
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

    /**
     * @brief Indefinite integral polynomial with respect to variable `var`.
     *
     * @param var Variable index (0-based, must be in `[0, M)`).
     * @details Runtime-index overload of `integ<I>()`.
     * @return New polynomial representing `integral ... dx_var` of this polynomial.
     * @throws std::out_of_range if `var < 0` or `var >= M`.
     */
    [[nodiscard]] TaylorExpansion integ( int var ) const
    {
        if ( var < 0 || var >= M )
            throw std::out_of_range(
                "tax::TaylorExpansion::integ(var): var must be in [0, M)" );
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
    // Gradient and Hessian (require Eigen/Core, already included above)
    // ------------------------------------------------------------------

    /**
     * @brief Compute the gradient vector `[df/dx_0, ..., df/dx_{M-1}]` at the expansion point.
     * @return `Eigen::Matrix<T, M, 1>` of first-order partial derivatives.
     */
    [[nodiscard]] Eigen::Matrix< T, M, 1 > gradient() const noexcept
    {
        Eigen::Matrix< T, M, 1 > g;
        MultiIndex< M > alpha{};
        for ( int i = 0; i < M; ++i )
        {
            alpha[std::size_t( i )] = 1;
            g( i ) = derivative( alpha );
            alpha[std::size_t( i )] = 0;
        }
        return g;
    }

    /**
     * @brief Compute the Hessian matrix `H(i,j) = d^2 f / (dx_i dx_j)` at the expansion point.
     * @return `Eigen::Matrix<T, M, M>` of second-order mixed partial derivatives.
     */
    [[nodiscard]] Eigen::Matrix< T, M, M > hessian() const noexcept
    {
        Eigen::Matrix< T, M, M > H;
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

    /// @brief Raw coefficient array — convenience accessor used by kernels.
    [[nodiscard]] constexpr const Data& coefficients() const noexcept { return c_.data; }
    [[nodiscard]] constexpr Data& coefficients() noexcept { return c_.data; }

private:
    container_t c_{};
};

// ---------------------------------------------------------------------------
// Convenience aliases  (dense)
// ---------------------------------------------------------------------------

/**
 * @brief `TE<N>` — univariate `double` expansion of order N.
 * @brief `TE<N, M>` — M-variate `double` expansion of order N.
 */
template < int N, int M = 1 >
using TE = TaylorExpansion< double, N, M, storage::Dense >;

// ---------------------------------------------------------------------------
// Self-check: verify the dense TaylorExpansion satisfies its own concepts.
// ---------------------------------------------------------------------------
static_assert( TaylorPolynomial< TE< 3 > >,
               "TE<3> must satisfy TaylorPolynomial" );
static_assert( DensePolynomial< TE< 3 > >,
               "TE<3> must satisfy DensePolynomial" );
static_assert( TaylorPolynomial< TE< 3, 2 > >,
               "TE<3,2> must satisfy TaylorPolynomial" );
static_assert( DensePolynomial< TE< 3, 2 > >,
               "TE<3,2> must satisfy DensePolynomial" );

// ---------------------------------------------------------------------------
// Sparse specialisation
// ---------------------------------------------------------------------------

/**
 * @brief A truncated Taylor expansion in M variables of order N with sparse storage.
 *
 * Stores only nonzero monomials as two parallel sorted vectors of (flat-index, value)
 * pairs.  Element access is O(log nnz) via binary search; arithmetic is O(nnz_a + nnz_b)
 * via a sorted merge walk.
 *
 * @tparam T  Scalar type (must satisfy `tax::Scalar`).
 * @tparam N  Truncation order (non-negative compile-time integer).
 * @tparam M  Number of independent variables (>= 1).
 */
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
    using Input       = std::array< T, std::size_t( M ) >;
    using Dense       = TaylorExpansion< T, N, M, storage::Dense >;

    // ------------------------------------------------------------------
    // Compile-time properties
    // ------------------------------------------------------------------
    static constexpr int         order_v       = N;
    static constexpr int         vars_v        = M;
    /// Dense-equivalent upper bound on the number of monomials.
    static constexpr std::size_t nCoefficients = numMonomials( N, M );

    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// @brief Zero-polynomial — no nonzero monomials.
    constexpr TaylorExpansion() = default;

    /// @brief Constant polynomial with value `c`.
    /*implicit*/ TaylorExpansion( T c )
    {
        if ( c != T{ 0 } )
            c_.set( 0, c );
    }

    /**
     * @brief Lift a dense polynomial into sparse storage (drops exact zeros).
     *
     * @param d  Source dense polynomial.
     */
    explicit TaylorExpansion( const Dense& d )
    {
        for ( std::size_t k = 0; k < Dense::nCoefficients; ++k )
        {
            if ( d[k] != T{ 0 } )
                c_.set( k, d[k] );
        }
    }

    // ------------------------------------------------------------------
    // Named factories
    // ------------------------------------------------------------------

    [[nodiscard]] static TaylorExpansion zero() noexcept { return {}; }
    [[nodiscard]] static TaylorExpansion constant( T c ) { return TaylorExpansion{ c }; }

    /**
     * @brief Univariate variable: `x = x0 + 1*dx`.
     * @note Only available when `M == 1`.
     */
    [[nodiscard]] static TaylorExpansion variable( T x0 ) noexcept
        requires( M == 1 )
    {
        TaylorExpansion r;
        if ( x0 != T{ 0 } )
            r.c_.set( 0, x0 );
        if constexpr ( N >= 1 )
            r.c_.set( 1, T{ 1 } );
        return r;
    }

    /**
     * @brief Coordinate variable `x_I` at expansion point `p` (compile-time index).
     *
     * @tparam I  Variable index in `[0, M)`.
     * @param  p  Expansion point.
     */
    template < int I >
    [[nodiscard]] static TaylorExpansion variable( const Input& p ) noexcept
        requires( M >= 1 && I >= 0 && I < M )
    {
        TaylorExpansion r;
        if ( p[std::size_t( I )] != T{ 0 } )
            r.c_.set( 0, p[std::size_t( I )] );
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

    /// @brief Number of currently stored nonzero monomials.
    [[nodiscard]] std::size_t nnz() const noexcept { return c_.nnz(); }

    /// @brief Constant (zeroth) coefficient; returns 0 if the slot is absent.
    [[nodiscard]] T value() const noexcept { return c_.value(); }

    /// @brief Runtime multi-index coefficient lookup (O(log nnz)).
    [[nodiscard]] T coeff( const MultiIndex< M >& alpha ) const noexcept
    {
        return c_.coeffAtFlat( flatIndex< M >( alpha ) );
    }

    /**
     * @brief Compile-time multi-index coefficient lookup (O(log nnz)).
     *
     * Usage: `f.coeff<2, 0>()` retrieves the coefficient of `x^2 y^0`.
     */
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

    /**
     * @brief Runtime partial derivative value `d^|alpha| f / dx^alpha` at x0.
     */
    [[nodiscard]] T derivative( const MultiIndex< M >& alpha ) const noexcept
    {
        std::size_t fac = 1;
        for ( int i = 0; i < M; ++i )
            for ( int j = 1; j <= alpha[std::size_t( i )]; ++j )
                fac *= std::size_t( j );
        return coeff( alpha ) * T( fac );
    }

    /**
     * @brief Compile-time partial derivative value.
     */
    template < int... Alpha >
    [[nodiscard]] T derivative() const noexcept
    {
        static_assert( sizeof...( Alpha ) == std::size_t( M ) );
        static_assert( ( ( Alpha >= 0 ) && ... ) );
        constexpr int total = ( Alpha + ... + 0 );
        static_assert( total <= N );

        constexpr auto factorial = []( int n ) constexpr noexcept -> std::size_t
        {
            std::size_t r = 1;
            for ( int i = 2; i <= n; ++i )
                r *= std::size_t( i );
            return r;
        };
        constexpr std::size_t fac = ( factorial( Alpha ) * ... * std::size_t( 1 ) );
        return coeff< Alpha... >() * T( fac );
    }

    // ------------------------------------------------------------------
    // Sparse-specific accessors
    // ------------------------------------------------------------------

    /// @brief Read-only view of the sorted flat indices of all nonzero slots.
    [[nodiscard]] std::span< const storage::flat_index_t > support() const noexcept
    {
        return c_.support();
    }

    /// @brief Read-only view of the coefficient values aligned with `support()`.
    [[nodiscard]] std::span< const T > values() const noexcept { return c_.values(); }

    // ------------------------------------------------------------------
    // Conversion
    // ------------------------------------------------------------------

    /**
     * @brief Materialise a dense `TaylorExpansion<T, N, M, Dense>` from this sparse polynomial.
     *
     * Absent slots are filled with `T{0}` (dense default-initialisation).
     */
    [[nodiscard]] Dense dense() const noexcept
    {
        Dense r;
        c_.forEachNonzero( [&]( std::size_t k, T v ) { r[k] = v; } );
        return r;
    }

    // ------------------------------------------------------------------
    // Container access
    // ------------------------------------------------------------------

    [[nodiscard]] const container_t& container() const noexcept { return c_; }
    [[nodiscard]] container_t&       container()       noexcept { return c_; }

private:
    container_t c_;
};

// ---------------------------------------------------------------------------
// Convenience aliases  (sparse)
// ---------------------------------------------------------------------------

/// @brief `STE<N>` — univariate sparse `double` expansion of order N.
/// @brief `STE<N, M>` — M-variate sparse `double` expansion of order N.
template < int N, int M = 1 >
using STE = TaylorExpansion< double, N, M, storage::Sparse >;

// ---------------------------------------------------------------------------
// Conversion helper: dense -> sparse
// ---------------------------------------------------------------------------

/**
 * @brief Convert a dense polynomial to sparse storage (drops exact zeros).
 *
 * @tparam T Scalar type.
 * @tparam N Truncation order.
 * @tparam M Number of variables.
 * @param  d Source dense polynomial.
 * @return   Equivalent sparse polynomial.
 */
template < typename T, int N, int M >
[[nodiscard]] TaylorExpansion< T, N, M, storage::Sparse >
    sparse( const TaylorExpansion< T, N, M, storage::Dense >& d ) noexcept
{
    return TaylorExpansion< T, N, M, storage::Sparse >( d );
}

// ---------------------------------------------------------------------------
// Self-check: verify the sparse TaylorExpansion satisfies its own concepts.
// ---------------------------------------------------------------------------
static_assert( TaylorPolynomial< STE< 3 > >,
               "STE<3> must satisfy TaylorPolynomial" );
static_assert( TaylorPolynomial< STE< 3, 2 > >,
               "STE<3,2> must satisfy TaylorPolynomial" );

}  // namespace tax
