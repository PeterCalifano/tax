#pragma once

#include <array>
#include <cstddef>
#include <tax/core/concepts.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/storage/dense.hpp>

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
// Convenience aliases
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

}  // namespace tax
