#pragma once

// Fully-dynamic partial specialisation of `tax::TaylorExpansionT<T, N, M>`:
// runtime order, runtime nvars, `std::vector<T>` coefficient storage.
//
// The static template (in `tax/tte.hpp`) keeps every existing optimisation:
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
    static constexpr int vars_ct = Dynamic;

    using ShapeBaseT = detail::ShapeBase< Dynamic, Dynamic >;
    using ShapeBaseT::coeffsSize;
    using ShapeBaseT::nvars;
    using ShapeBaseT::order;

    // -- Constructors --------------------------------------------------------

    /// @brief Construct an empty (order=0, nvars=0) expansion.
    TaylorExpansionT() noexcept = default;

    /// @brief Construct zero polynomial of given shape.
    TaylorExpansionT( std::size_t order_v, std::size_t nvars_v )
        : ShapeBaseT( order_v, nvars_v ), c_( detail::numMonomials( order_v, nvars_v ), T{ 0 } )
    {
        assert( nvars_v >= 1 && "TaylorExpansionT requires nvars >= 1" );
    }

    /// @brief Construct from explicit (order, nvars, coefficient vector). Vector size must match.
    TaylorExpansionT( std::size_t order_v, std::size_t nvars_v, Data c )
        : ShapeBaseT( order_v, nvars_v ), c_( std::move( c ) )
    {
        assert( nvars_v >= 1 );
        assert( c_.size() == detail::numMonomials( order_v, nvars_v ) );
    }

    // -- Factories -----------------------------------------------------------

    /// @brief Zero polynomial.
    [[nodiscard]] static TaylorExpansionT zero( std::size_t order_v, std::size_t nvars_v )
    {
        return TaylorExpansionT( order_v, nvars_v );
    }

    /// @brief Constant polynomial equal to `c`.
    [[nodiscard]] static TaylorExpansionT constant( T c, std::size_t order_v, std::size_t nvars_v )
    {
        TaylorExpansionT out( order_v, nvars_v );
        out.c_[0] = c;
        return out;
    }

    /// @brief Polynomial with value `1`.
    [[nodiscard]] static TaylorExpansionT one( std::size_t order_v, std::size_t nvars_v )
    {
        return constant( T{ 1 }, order_v, nvars_v );
    }

    /**
     * @brief Variable `x_{var_idx}` expanded around `x0`.
     * @details Constant term is `x0`; the `e_{var_idx}` coefficient is `1`.
     */
    [[nodiscard]] static TaylorExpansionT variable( T x0, std::size_t var_idx, std::size_t order_v,
                                                    std::size_t nvars_v )
    {
        if ( var_idx >= nvars_v )
            throw std::out_of_range( "tax::TaylorExpansionT::variable: var_idx >= nvars" );
        TaylorExpansionT out( order_v, nvars_v );
        out.c_[0] = x0;
        if ( order_v >= 1 )
        {
            std::vector< int > ei( nvars_v, 0 );
            ei[var_idx] = 1;
            const std::size_t fi =
                detail::flatIndex( std::span< const int >( ei.data(), nvars_v ) );
            out.c_[fi] = T{ 1 };
        }
        return out;
    }

    /// @brief Build all `nvars_v` coordinate variables at expansion point `x0`.
    [[nodiscard]] static std::vector< TaylorExpansionT > variables( std::span< const T > x0,
                                                                    std::size_t order_v )
    {
        const std::size_t nvars_v = x0.size();
        std::vector< TaylorExpansionT > out;
        out.reserve( nvars_v );
        for ( std::size_t i = 0; i < nvars_v; ++i )
            out.push_back( variable( x0[i], i, order_v, nvars_v ) );
        return out;
    }

    // -- Accessors -----------------------------------------------------------

    [[nodiscard]] const Data& coeffs() const noexcept { return c_; }
    [[nodiscard]] Data& coeffs() noexcept { return c_; }
    [[nodiscard]] T value() const noexcept { return c_[0]; }
    [[nodiscard]] T operator[]( std::size_t i ) const noexcept { return c_[i]; }
    [[nodiscard]] T& operator[]( std::size_t i ) noexcept { return c_[i]; }

    /// @brief Coefficient at multi-index `alpha` (size must equal `nvars()`).
    [[nodiscard]] T coeff( std::span< const int > alpha ) const
    {
        assert( alpha.size() == this->nvars() );
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
        detail::addInPlaceRT( c_.data(), r.c_.data(), c_.size() );
        return *this;
    }
    TaylorExpansionT& operator-=( const TaylorExpansionT& r )
    {
        assertSameShape( r );
        detail::subInPlaceRT( c_.data(), r.c_.data(), c_.size() );
        return *this;
    }
    TaylorExpansionT& operator*=( const TaylorExpansionT& r )
    {
        assertSameShape( r );
        Data tmp( c_.size(), T{ 0 } );
        detail::cauchyProductRT( tmp.data(), c_.data(), r.c_.data(), this->order(), this->nvars() );
        c_.swap( tmp );
        return *this;
    }
    TaylorExpansionT& operator/=( const TaylorExpansionT& r )
    {
        assertSameShape( r );
        Data rec( c_.size(), T{ 0 } );
        detail::seriesReciprocalRT( rec.data(), r.c_.data(), this->order(), this->nvars() );
        Data prod( c_.size(), T{ 0 } );
        detail::cauchyProductRT( prod.data(), c_.data(), rec.data(), this->order(), this->nvars() );
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
        detail::scaleInPlaceRT( c_.data(), s, c_.size() );
        return *this;
    }
    TaylorExpansionT& operator/=( T s ) noexcept { return *this *= ( T{ 1 } / s ); }

    [[nodiscard]] TaylorExpansionT operator-() const
    {
        TaylorExpansionT out( *this );
        detail::negateInPlaceRT( out.c_.data(), out.c_.size() );
        return out;
    }

   private:
    void assertSameShape( const TaylorExpansionT& r ) const
    {
        assert( this->order() == r.order() && "TaylorExpansionT: order mismatch" );
        assert( this->nvars() == r.nvars() && "TaylorExpansionT: nvars mismatch" );
        ( void )r;
    }

    Data c_;
};

/// @brief Fully-dynamic TaylorExpansionT alias (runtime order and nvars).
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
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > applyUnaryRT(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a, K&& kernel )
{
    TaylorExpansionT< T, Dynamic, Dynamic > out( a.order(), a.nvars() );
    kernel( out.coeffs().data(), a.coeffs().data(), a.order(), a.nvars() );
    return out;
}
}  // namespace detail

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > sin(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    return detail::applyUnaryRT( a, []( T* o, const T* x, std::size_t N, std::size_t M ) {
        detail::seriesSinRT( o, x, N, M );
    } );
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > cos(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    return detail::applyUnaryRT( a, []( T* o, const T* x, std::size_t N, std::size_t M ) {
        detail::seriesCosRT( o, x, N, M );
    } );
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > exp(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    return detail::applyUnaryRT( a, []( T* o, const T* x, std::size_t N, std::size_t M ) {
        detail::seriesExpRT( o, x, N, M );
    } );
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > log(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    return detail::applyUnaryRT( a, []( T* o, const T* x, std::size_t N, std::size_t M ) {
        detail::seriesLogRT( o, x, N, M );
    } );
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > sqrt(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    return detail::applyUnaryRT( a, []( T* o, const T* x, std::size_t N, std::size_t M ) {
        detail::seriesSqrtRT( o, x, N, M );
    } );
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > square(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    return detail::applyUnaryRT( a, []( T* o, const T* x, std::size_t N, std::size_t M ) {
        detail::seriesSquareRT( o, x, N, M );
    } );
}

template < typename T >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, Dynamic > pow(
    const TaylorExpansionT< T, Dynamic, Dynamic >& a, T c )
{
    TaylorExpansionT< T, Dynamic, Dynamic > out( a.order(), a.nvars() );
    detail::seriesPowRT( out.coeffs().data(), a.coeffs().data(), c, a.order(), a.nvars() );
    return out;
}

// =============================================================================
// Streaming
// =============================================================================

template < typename T >
inline std::ostream& operator<<( std::ostream& os, const TaylorExpansionT< T, Dynamic, Dynamic >& a )
{
    os << "DynTE(order=" << a.order() << ", nvars=" << a.nvars() << ", coeffs=[";
    for ( std::size_t i = 0; i < a.coeffs().size(); ++i )
    {
        if ( i != 0 ) os << ", ";
        os << a.coeffs()[i];
    }
    os << "])";
    return os;
}

}  // namespace tax
