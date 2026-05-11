#pragma once

// Fully-dynamic partial specialisation of `tax::TaylorExpansionT<T, N, M>`:
// runtime order, runtime size, `std::vector<T>` coefficient storage.
//
// The static template (in `tax/storage/tte_static.hpp`) keeps every existing optimisation:
// stack-resident `std::array`, expression-template fusion, zero allocation.
// This file adds an eager-evaluation alternative for runtime-resolved shapes
// (Python bindings, REPLs, exploratory work). Both share the kernel layer.

#include <cassert>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <tax/kernels.hpp>
#include <tax/storage/shape.hpp>
#include <tax/utils/combinatorics.hpp>
#include <tax/utils/enumeration.hpp>
#include <tax/utils/fwd.hpp>

namespace tax
{

template < typename T, int N, int M >
class TaylorExpansionT;

// =============================================================================
// TaylorExpansionT<T, Dynamic, Dynamic>: fully-dynamic specialisation.
// =============================================================================

template < typename T >
class TaylorExpansionT< T, Dynamic, Dynamic > : public detail::ShapeBase< Dynamic, Dynamic >
{
    static_assert( std::floating_point< T >, "Scalar must satisfy std::floating_point" );

   public:
    using scalar_type = T;
    using Data = std::vector< T >;

    static constexpr int order_ct = Dynamic;
    static constexpr int size_ct = Dynamic;

    using ShapeBaseT = detail::ShapeBase< Dynamic, Dynamic >;
    using ShapeBaseT::coeffsSize;
    using ShapeBaseT::size;
    using ShapeBaseT::order;

    // -- Constructors --------------------------------------------------------

    /// @brief Construct an empty (order=0, size=0) expansion.
    TaylorExpansionT() noexcept = default;

    /// @brief Construct zero polynomial of given shape.
    TaylorExpansionT( std::size_t order_v, std::size_t size_v )
        : ShapeBaseT( order_v, size_v ), c_( detail::numMonomials( order_v, size_v ), T{ 0 } )
    {
        assert( size_v >= 1 && "TaylorExpansionT requires size >= 1" );
    }

    /// @brief Construct from explicit (order, size, coefficient vector). Vector size must match.
    TaylorExpansionT( std::size_t order_v, std::size_t size_v, Data c )
        : ShapeBaseT( order_v, size_v ), c_( std::move( c ) )
    {
        assert( size_v >= 1 );
        assert( c_.size() == detail::numMonomials( order_v, size_v ) );
    }

    // -- Factories -----------------------------------------------------------

    /// @brief Zero polynomial.
    [[nodiscard]] static TaylorExpansionT zero( std::size_t order_v, std::size_t size_v )
    {
        return TaylorExpansionT( order_v, size_v );
    }

    /// @brief Constant polynomial equal to `c`.
    [[nodiscard]] static TaylorExpansionT constant( T c, std::size_t order_v, std::size_t size_v )
    {
        TaylorExpansionT out( order_v, size_v );
        out.c_[0] = c;
        return out;
    }

    /// @brief Polynomial with value `1`.
    [[nodiscard]] static TaylorExpansionT one( std::size_t order_v, std::size_t size_v )
    {
        return constant( T{ 1 }, order_v, size_v );
    }

    /**
     * @brief Variable `x_{var_idx}` expanded around `x0`.
     * @details Constant term is `x0`; the `e_{var_idx}` coefficient is `1`.
     */
    [[nodiscard]] static TaylorExpansionT variable( T x0, std::size_t var_idx, std::size_t order_v,
                                                    std::size_t size_v )
    {
        if ( var_idx >= size_v )
            throw std::out_of_range( "tax::TaylorExpansionT::variable: var_idx >= size" );
        TaylorExpansionT out( order_v, size_v );
        out.c_[0] = x0;
        if ( order_v >= 1 )
        {
            std::vector< int > ei( size_v, 0 );
            ei[var_idx] = 1;
            const std::size_t fi =
                detail::flatIndex( std::span< const int >( ei.data(), size_v ) );
            out.c_[fi] = T{ 1 };
        }
        return out;
    }

    /// @brief Build all `size_v` coordinate variables at expansion point `x0`.
    [[nodiscard]] static std::vector< TaylorExpansionT > variables( std::span< const T > x0,
                                                                    std::size_t order_v )
    {
        const std::size_t size_v = x0.size();
        std::vector< TaylorExpansionT > out;
        out.reserve( size_v );
        for ( std::size_t i = 0; i < size_v; ++i )
            out.push_back( variable( x0[i], i, order_v, size_v ) );
        return out;
    }

    // -- Accessors -----------------------------------------------------------

    [[nodiscard]] const Data& coeffs() const noexcept { return c_; }
    [[nodiscard]] Data& coeffs() noexcept { return c_; }
    [[nodiscard]] T value() const noexcept { return c_[0]; }
    [[nodiscard]] T operator[]( std::size_t i ) const noexcept { return c_[i]; }
    [[nodiscard]] T& operator[]( std::size_t i ) noexcept { return c_[i]; }

    /// @brief Coefficient at multi-index `alpha` (size must equal `size()`).
    [[nodiscard]] T coeff( std::span< const int > alpha ) const
    {
        assert( alpha.size() == this->size() );
        return c_[detail::flatIndex( alpha )];
    }

    /// @brief Coefficient at multi-index given by an initializer list.
    [[nodiscard]] T coeff( std::initializer_list< int > alpha ) const
    {
        std::vector< int > buf( alpha );
        return coeff( std::span< const int >( buf.data(), buf.size() ) );
    }

    // -- In-place arithmetic -------------------------------------------------

    TaylorExpansionT& operator+=( const TaylorExpansionT& r )
    {
        assertSameShape( r );
        detail::addInPlace( c_.data(), r.c_.data(), c_.size() );
        return *this;
    }
    TaylorExpansionT& operator-=( const TaylorExpansionT& r )
    {
        assertSameShape( r );
        detail::subInPlace( c_.data(), r.c_.data(), c_.size() );
        return *this;
    }
    TaylorExpansionT& operator*=( const TaylorExpansionT& r )
    {
        assertSameShape( r );
        Data tmp( c_.size(), T{ 0 } );
        detail::cauchyProduct( tmp.data(), c_.data(), r.c_.data(), this->order(), this->size() );
        c_.swap( tmp );
        return *this;
    }
    TaylorExpansionT& operator/=( const TaylorExpansionT& r )
    {
        assertSameShape( r );
        Data rec( c_.size(), T{ 0 } );
        detail::seriesReciprocal( rec.data(), r.c_.data(), this->order(), this->size() );
        Data prod( c_.size(), T{ 0 } );
        detail::cauchyProduct( prod.data(), c_.data(), rec.data(), this->order(), this->size() );
        c_.swap( prod );
        return *this;
    }

    TaylorExpansionT& operator+=( T s ) noexcept
    {
        c_[0] += s;
        return *this;
    }
    TaylorExpansionT& operator-=( T s ) noexcept
    {
        c_[0] -= s;
        return *this;
    }
    TaylorExpansionT& operator*=( T s ) noexcept
    {
        detail::scaleInPlace( c_.data(), s, c_.size() );
        return *this;
    }
    TaylorExpansionT& operator/=( T s ) noexcept { return *this *= ( T{ 1 } / s ); }

    [[nodiscard]] TaylorExpansionT operator-() const
    {
        TaylorExpansionT out( *this );
        detail::negateInPlace( out.c_.data(), out.c_.size() );
        return out;
    }

   private:
    void assertSameShape( const TaylorExpansionT& r ) const
    {
        assert( this->order() == r.order() && "TaylorExpansionT: order mismatch" );
        assert( this->size() == r.size() && "TaylorExpansionT: size mismatch" );
        ( void )r;
    }

    Data c_;
};

/// @brief Fully-dynamic TaylorExpansionT alias (runtime order and size).
template < typename T = double >
using DynTE = TaylorExpansionT< T, Dynamic, Dynamic >;

// =============================================================================
// Binary arithmetic (eager, no expression templates).
// =============================================================================

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > operator+(
    TaylorExpansionT< T, Dynamic, Dynamic > l,
    const TaylorExpansionT< T, Dynamic, Dynamic >& r )
{
    l += r;
    return l;
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > operator-(
    TaylorExpansionT< T, Dynamic, Dynamic > l,
    const TaylorExpansionT< T, Dynamic, Dynamic >& r )
{
    l -= r;
    return l;
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > operator*(
    TaylorExpansionT< T, Dynamic, Dynamic > l,
    const TaylorExpansionT< T, Dynamic, Dynamic >& r )
{
    l *= r;
    return l;
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > operator/(
    TaylorExpansionT< T, Dynamic, Dynamic > l,
    const TaylorExpansionT< T, Dynamic, Dynamic >& r )
{
    l /= r;
    return l;
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > operator+(
    TaylorExpansionT< T, Dynamic, Dynamic > l, T s )
{
    l += s;
    return l;
}
template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > operator+(
    T s, TaylorExpansionT< T, Dynamic, Dynamic > r )
{
    r += s;
    return r;
}
template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > operator-(
    TaylorExpansionT< T, Dynamic, Dynamic > l, T s )
{
    l -= s;
    return l;
}
template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > operator-(
    T s, TaylorExpansionT< T, Dynamic, Dynamic > r )
{
    auto out = -r;
    out += s;
    return out;
}
template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > operator*(
    TaylorExpansionT< T, Dynamic, Dynamic > l, T s )
{
    l *= s;
    return l;
}
template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > operator*(
    T s, TaylorExpansionT< T, Dynamic, Dynamic > r )
{
    r *= s;
    return r;
}
template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > operator/(
    TaylorExpansionT< T, Dynamic, Dynamic > l, T s )
{
    l /= s;
    return l;
}

// =============================================================================
// Math free functions (eager, runtime kernels).
// =============================================================================

namespace detail
{
template < typename T, typename K >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > applyUnary(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a, K&& kernel )
{
    TaylorExpansionT< T, Dynamic, Dynamic > out( a.order(), a.size() );
    kernel( out.coeffs().data(), a.coeffs().data(), a.order(), a.size() );
    return out;
}
}  // namespace detail

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > sin(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    return detail::applyUnary( a, []( T* o, const T* x, std::size_t N, std::size_t M ) {
        detail::seriesSin( o, x, N, M );
    } );
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > cos(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    return detail::applyUnary( a, []( T* o, const T* x, std::size_t N, std::size_t M ) {
        detail::seriesCos( o, x, N, M );
    } );
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > exp(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    return detail::applyUnary( a, []( T* o, const T* x, std::size_t N, std::size_t M ) {
        detail::seriesExp( o, x, N, M );
    } );
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > log(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    return detail::applyUnary( a, []( T* o, const T* x, std::size_t N, std::size_t M ) {
        detail::seriesLog( o, x, N, M );
    } );
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > sqrt(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    return detail::applyUnary( a, []( T* o, const T* x, std::size_t N, std::size_t M ) {
        detail::seriesSqrt( o, x, N, M );
    } );
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > square(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    return detail::applyUnary( a, []( T* o, const T* x, std::size_t N, std::size_t M ) {
        detail::seriesSquare( o, x, N, M );
    } );
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > pow(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a, T c )
{
    TaylorExpansionT< T, Dynamic, Dynamic > out( a.order(), a.size() );
    detail::seriesPow( out.coeffs().data(), a.coeffs().data(), c, a.order(), a.size() );
    return out;
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > tan(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    return detail::applyUnary( a, []( T* o, const T* x, std::size_t N, std::size_t M ) {
        detail::seriesTan( o, x, N, M );
    } );
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > sinh(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    return detail::applyUnary( a, []( T* o, const T* x, std::size_t N, std::size_t M ) {
        detail::seriesSinh( o, x, N, M );
    } );
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > cosh(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    return detail::applyUnary( a, []( T* o, const T* x, std::size_t N, std::size_t M ) {
        detail::seriesCosh( o, x, N, M );
    } );
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > tanh(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    return detail::applyUnary( a, []( T* o, const T* x, std::size_t N, std::size_t M ) {
        detail::seriesTanh( o, x, N, M );
    } );
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > asin(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    return detail::applyUnary( a, []( T* o, const T* x, std::size_t N, std::size_t M ) {
        detail::seriesAsin( o, x, N, M );
    } );
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > acos(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    return detail::applyUnary( a, []( T* o, const T* x, std::size_t N, std::size_t M ) {
        detail::seriesAcos( o, x, N, M );
    } );
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > atan(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    return detail::applyUnary( a, []( T* o, const T* x, std::size_t N, std::size_t M ) {
        detail::seriesAtan( o, x, N, M );
    } );
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > cube(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    return detail::applyUnary( a, []( T* o, const T* x, std::size_t N, std::size_t M ) {
        detail::seriesCube( o, x, N, M );
    } );
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > cbrt(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    return detail::applyUnary( a, []( T* o, const T* x, std::size_t N, std::size_t M ) {
        detail::seriesCbrt( o, x, N, M );
    } );
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > abs(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    TaylorExpansionT< T, Dynamic, Dynamic > out( a.order(), a.size() );
    detail::seriesAbs( out.coeffs().data(), a.coeffs().data(), a.coeffs().size() );
    return out;
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > log10(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    using std::log;
    auto out = tax::log( a );
    out *= T{ 1 } / log( T{ 10 } );
    return out;
}

// =============================================================================
// Streaming
// =============================================================================

template < typename T >
inline std::ostream& operator<<( std::ostream& os, const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    os << "DynTE(order=" << a.order() << ", size=" << a.size() << ", coeffs=[";
    for ( std::size_t i = 0; i < a.coeffs().size(); ++i )
    {
        if ( i != 0 ) os << ", ";
        os << a.coeffs()[i];
    }
    os << "])";
    return os;
}

}  // namespace tax
