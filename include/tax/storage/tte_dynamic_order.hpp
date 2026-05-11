#pragma once

// Mixed-dynamism partial specialisation of `tax::TaylorExpansionT<T, N, M>`:
// runtime order `N`, compile-time number of variables `M`. Storage is
// `std::vector<T>` (size depends on runtime order), but the variable index
// stays a compile-time concept (so `variable<I>(x0, order)` works and
// `variables(x0, order)` returns a `std::array<..., M>`).
//
// This is the natural choice when M is fixed by the problem (e.g. a 6-D
// state vector in orbital mechanics) but the truncation order is chosen at
// runtime — by far the most common need outside the fully-static path.

#include <array>
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
// TaylorExpansionT<T, Dynamic, M>: runtime order, compile-time variable count.
// =============================================================================

template < typename T, int M >
class TaylorExpansionT< T, Dynamic, M > : public detail::ShapeBase< Dynamic, M >
{
    static_assert( M >= 1, "TaylorExpansionT requires M >= 1 (or M = Dynamic)" );
    static_assert( std::floating_point< T >, "Scalar must satisfy std::floating_point" );

   public:
    using scalar_type = T;
    using Data = std::vector< T >;
    /// @brief Expansion point type — fixed-size since `M` is compile-time.
    using Input = std::array< T, std::size_t( M ) >;

    static constexpr int order_ct = Dynamic;
    static constexpr int size_ct = M;

    using ShapeBaseT = detail::ShapeBase< Dynamic, M >;
    using ShapeBaseT::coeffsSize;
    using ShapeBaseT::order;
    using ShapeBaseT::size;

    // -- Constructors --------------------------------------------------------

    /// @brief Construct an empty (order=0) expansion. `size()` still returns `M`.
    TaylorExpansionT() noexcept = default;

    /// @brief Construct zero polynomial of the given order.
    explicit TaylorExpansionT( std::size_t order_v )
        : ShapeBaseT( order_v, std::size_t( M ) ),
          c_( detail::numMonomials( order_v, std::size_t( M ) ), T{ 0 } )
    {
    }

    /// @brief Construct from explicit `(order, coefficient vector)`. Sizes must match.
    TaylorExpansionT( std::size_t order_v, Data c )
        : ShapeBaseT( order_v, std::size_t( M ) ), c_( std::move( c ) )
    {
        assert( c_.size() == detail::numMonomials( order_v, std::size_t( M ) ) );
    }

    // -- Factories -----------------------------------------------------------

    [[nodiscard]] static TaylorExpansionT zero( std::size_t order_v )
    {
        return TaylorExpansionT( order_v );
    }

    [[nodiscard]] static TaylorExpansionT constant( T c, std::size_t order_v )
    {
        TaylorExpansionT out( order_v );
        out.c_[0] = c;
        return out;
    }

    [[nodiscard]] static TaylorExpansionT one( std::size_t order_v )
    {
        return constant( T{ 1 }, order_v );
    }

    /**
     * @brief Univariate variable expanded at `x0`. Only available when `M == 1`.
     */
    [[nodiscard]] static TaylorExpansionT variable( T x0, std::size_t order_v )
        requires( M == 1 )
    {
        TaylorExpansionT out( order_v );
        out.c_[0] = x0;
        if ( order_v >= 1 ) out.c_[1] = T{ 1 };
        return out;
    }

    /**
     * @brief Variable `x_I` expanded at `x0` (compile-time index, like the static TTE).
     */
    template < int I >
        requires( I >= 0 && I < M )
    [[nodiscard]] static TaylorExpansionT variable( const Input& x0, std::size_t order_v )
    {
        TaylorExpansionT out( order_v );
        out.c_[0] = x0[I];
        if ( order_v >= 1 )
        {
            tax::MultiIndex< M > ei{};
            ei[I] = 1;
            out.c_[detail::flatIndex< M >( ei )] = T{ 1 };
        }
        return out;
    }

    /**
     * @brief Variable `x_{var_idx}` at `x0` (runtime index).
     */
    [[nodiscard]] static TaylorExpansionT variable( T x0, std::size_t var_idx,
                                                    std::size_t order_v )
    {
        if ( var_idx >= std::size_t( M ) )
            throw std::out_of_range( "tax::TaylorExpansionT::variable: var_idx >= M" );
        TaylorExpansionT out( order_v );
        out.c_[0] = x0;
        if ( order_v >= 1 )
        {
            tax::MultiIndex< M > ei{};
            ei[var_idx] = 1;
            out.c_[detail::flatIndex< M >( ei )] = T{ 1 };
        }
        return out;
    }

    /**
     * @brief All `M` coordinate variables at expansion point `x0`, given runtime order.
     * @details Returned as `std::array<TaylorExpansionT, M>` for structured-binding access.
     */
    [[nodiscard]] static std::array< TaylorExpansionT, std::size_t( M ) > variables(
        const Input& x0, std::size_t order_v )
    {
        return [&]< std::size_t... I >( std::index_sequence< I... > ) {
            return std::array< TaylorExpansionT, std::size_t( M ) >{
                variable< int( I ) >( x0, order_v )... };
        }( std::make_index_sequence< std::size_t( M ) >{} );
    }

    // -- Accessors -----------------------------------------------------------

    [[nodiscard]] const Data& coeffs() const noexcept { return c_; }
    [[nodiscard]] Data& coeffs() noexcept { return c_; }
    [[nodiscard]] T value() const noexcept { return c_[0]; }
    [[nodiscard]] T operator[]( std::size_t i ) const noexcept { return c_[i]; }
    [[nodiscard]] T& operator[]( std::size_t i ) noexcept { return c_[i]; }

    /// @brief Coefficient at multi-index `alpha`.
    [[nodiscard]] T coeff( const tax::MultiIndex< M >& alpha ) const noexcept
    {
        return c_[detail::flatIndex< M >( alpha )];
    }

    // -- In-place arithmetic -------------------------------------------------

    TaylorExpansionT& operator+=( const TaylorExpansionT& r )
    {
        assertSameOrder( r );
        detail::addInPlace( c_.data(), r.c_.data(), c_.size() );
        return *this;
    }
    TaylorExpansionT& operator-=( const TaylorExpansionT& r )
    {
        assertSameOrder( r );
        detail::subInPlace( c_.data(), r.c_.data(), c_.size() );
        return *this;
    }
    TaylorExpansionT& operator*=( const TaylorExpansionT& r )
    {
        assertSameOrder( r );
        Data tmp( c_.size(), T{ 0 } );
        detail::cauchyProduct( tmp.data(), c_.data(), r.c_.data(), this->order(),
                               std::size_t( M ) );
        c_.swap( tmp );
        return *this;
    }
    TaylorExpansionT& operator/=( const TaylorExpansionT& r )
    {
        assertSameOrder( r );
        Data rec( c_.size(), T{ 0 } );
        detail::seriesReciprocal( rec.data(), r.c_.data(), this->order(), std::size_t( M ) );
        Data prod( c_.size(), T{ 0 } );
        detail::cauchyProduct( prod.data(), c_.data(), rec.data(), this->order(),
                               std::size_t( M ) );
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
    void assertSameOrder( const TaylorExpansionT& r ) const
    {
        assert( this->order() == r.order() && "TaylorExpansionT: order mismatch" );
        ( void )r;
    }

    Data c_;
};

/// @brief Dynamic-order, static-size Taylor expansion alias.
template < int M, typename T = double >
using DynOrderTE = TaylorExpansionT< T, Dynamic, M >;

// =============================================================================
// Binary arithmetic (eager, no expression templates).
// =============================================================================

template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > operator+(
    TaylorExpansionT< T, Dynamic, M > l, const TaylorExpansionT< T, Dynamic, M >& r )
{
    l += r;
    return l;
}

template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > operator-(
    TaylorExpansionT< T, Dynamic, M > l, const TaylorExpansionT< T, Dynamic, M >& r )
{
    l -= r;
    return l;
}

template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > operator*(
    TaylorExpansionT< T, Dynamic, M > l, const TaylorExpansionT< T, Dynamic, M >& r )
{
    l *= r;
    return l;
}

template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > operator/(
    TaylorExpansionT< T, Dynamic, M > l, const TaylorExpansionT< T, Dynamic, M >& r )
{
    l /= r;
    return l;
}

template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > operator+(
    TaylorExpansionT< T, Dynamic, M > l, T s )
{
    l += s;
    return l;
}
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > operator+(
    T s, TaylorExpansionT< T, Dynamic, M > r )
{
    r += s;
    return r;
}
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > operator-(
    TaylorExpansionT< T, Dynamic, M > l, T s )
{
    l -= s;
    return l;
}
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > operator-(
    T s, TaylorExpansionT< T, Dynamic, M > r )
{
    auto out = -r;
    out += s;
    return out;
}
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > operator*(
    TaylorExpansionT< T, Dynamic, M > l, T s )
{
    l *= s;
    return l;
}
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > operator*(
    T s, TaylorExpansionT< T, Dynamic, M > r )
{
    r *= s;
    return r;
}
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > operator/(
    TaylorExpansionT< T, Dynamic, M > l, T s )
{
    l /= s;
    return l;
}

// =============================================================================
// Math free functions (eager, runtime kernels with compile-time M).
// =============================================================================

namespace detail
{
template < typename T, int M, typename K >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > applyUnaryDynOrder(
    const TaylorExpansionT< T, Dynamic, M >& a, K&& kernel )
{
    TaylorExpansionT< T, Dynamic, M > out( a.order() );
    kernel( out.coeffs().data(), a.coeffs().data(), a.order(), std::size_t( M ) );
    return out;
}
}  // namespace detail

#define TAX_DEFINE_DYN_ORDER_UNARY( name )                                                       \
    template < typename T, int M >                                                               \
    [[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > name(                                 \
        const TaylorExpansionT< T, Dynamic, M >& a )                                             \
    {                                                                                            \
        return detail::applyUnaryDynOrder( a, []( T* o, const T* x, std::size_t N,               \
                                                  std::size_t Mr ) {                             \
            detail::series##name( o, x, N, Mr );                                                 \
        } );                                                                                     \
    }

// Trig
TAX_DEFINE_DYN_ORDER_UNARY( Sin )
TAX_DEFINE_DYN_ORDER_UNARY( Cos )
TAX_DEFINE_DYN_ORDER_UNARY( Tan )

// Hyperbolic
TAX_DEFINE_DYN_ORDER_UNARY( Sinh )
TAX_DEFINE_DYN_ORDER_UNARY( Cosh )
TAX_DEFINE_DYN_ORDER_UNARY( Tanh )

// Inverse trig
TAX_DEFINE_DYN_ORDER_UNARY( Asin )
TAX_DEFINE_DYN_ORDER_UNARY( Acos )
TAX_DEFINE_DYN_ORDER_UNARY( Atan )

// Transcendental
TAX_DEFINE_DYN_ORDER_UNARY( Exp )
TAX_DEFINE_DYN_ORDER_UNARY( Log )

// Algebra
TAX_DEFINE_DYN_ORDER_UNARY( Sqrt )
TAX_DEFINE_DYN_ORDER_UNARY( Cbrt )
TAX_DEFINE_DYN_ORDER_UNARY( Square )
TAX_DEFINE_DYN_ORDER_UNARY( Cube )

// Inverse hyperbolic + erf
TAX_DEFINE_DYN_ORDER_UNARY( Asinh )
TAX_DEFINE_DYN_ORDER_UNARY( Acosh )
TAX_DEFINE_DYN_ORDER_UNARY( Atanh )
TAX_DEFINE_DYN_ORDER_UNARY( Erf )

#undef TAX_DEFINE_DYN_ORDER_UNARY

// Lowercase aliases matching the std math conventions.
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > sin(
    const TaylorExpansionT< T, Dynamic, M >& a )
{
    return Sin( a );
}
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > cos(
    const TaylorExpansionT< T, Dynamic, M >& a )
{
    return Cos( a );
}
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > tan(
    const TaylorExpansionT< T, Dynamic, M >& a )
{
    return Tan( a );
}
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > sinh(
    const TaylorExpansionT< T, Dynamic, M >& a )
{
    return Sinh( a );
}
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > cosh(
    const TaylorExpansionT< T, Dynamic, M >& a )
{
    return Cosh( a );
}
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > tanh(
    const TaylorExpansionT< T, Dynamic, M >& a )
{
    return Tanh( a );
}
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > asin(
    const TaylorExpansionT< T, Dynamic, M >& a )
{
    return Asin( a );
}
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > acos(
    const TaylorExpansionT< T, Dynamic, M >& a )
{
    return Acos( a );
}
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > atan(
    const TaylorExpansionT< T, Dynamic, M >& a )
{
    return Atan( a );
}
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > exp(
    const TaylorExpansionT< T, Dynamic, M >& a )
{
    return Exp( a );
}
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > log(
    const TaylorExpansionT< T, Dynamic, M >& a )
{
    return Log( a );
}
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > sqrt(
    const TaylorExpansionT< T, Dynamic, M >& a )
{
    return Sqrt( a );
}
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > cbrt(
    const TaylorExpansionT< T, Dynamic, M >& a )
{
    return Cbrt( a );
}
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > square(
    const TaylorExpansionT< T, Dynamic, M >& a )
{
    return Square( a );
}
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > cube(
    const TaylorExpansionT< T, Dynamic, M >& a )
{
    return Cube( a );
}

template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > pow(
    const TaylorExpansionT< T, Dynamic, M >& a, T c )
{
    TaylorExpansionT< T, Dynamic, M > out( a.order() );
    detail::seriesPow( out.coeffs().data(), a.coeffs().data(), c, a.order(), std::size_t( M ) );
    return out;
}

template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > abs(
    const TaylorExpansionT< T, Dynamic, M >& a )
{
    TaylorExpansionT< T, Dynamic, M > out( a.order() );
    detail::seriesAbs( out.coeffs().data(), a.coeffs().data(), a.coeffs().size() );
    return out;
}

template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > log10(
    const TaylorExpansionT< T, Dynamic, M >& a )
{
    using std::log;
    auto out = tax::log( a );
    out *= T{ 1 } / log( T{ 10 } );
    return out;
}

// Lowercase aliases for the inverse-hyperbolic + erf set.
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > asinh(
    const TaylorExpansionT< T, Dynamic, M >& a )
{
    return Asinh( a );
}
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > acosh(
    const TaylorExpansionT< T, Dynamic, M >& a )
{
    return Acosh( a );
}
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > atanh(
    const TaylorExpansionT< T, Dynamic, M >& a )
{
    return Atanh( a );
}
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > erf(
    const TaylorExpansionT< T, Dynamic, M >& a )
{
    return Erf( a );
}

/// @brief Integer power: `a^n` via binary exponentiation.
template < int n, typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > pow(
    const TaylorExpansionT< T, Dynamic, M >& a )
{
    TaylorExpansionT< T, Dynamic, M > out( a.order() );
    detail::seriesIntPow( out.coeffs().data(), a.coeffs().data(), n, a.order(),
                          std::size_t( M ) );
    return out;
}

template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > pow(
    const TaylorExpansionT< T, Dynamic, M >& a, int n )
{
    TaylorExpansionT< T, Dynamic, M > out( a.order() );
    detail::seriesIntPow( out.coeffs().data(), a.coeffs().data(), n, a.order(),
                          std::size_t( M ) );
    return out;
}

template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > atan2(
    const TaylorExpansionT< T, Dynamic, M >& y,
    const TaylorExpansionT< T, Dynamic, M >& x )
{
    assert( y.order() == x.order() );
    TaylorExpansionT< T, Dynamic, M > out( y.order() );
    detail::seriesAtan2( out.coeffs().data(), y.coeffs().data(), x.coeffs().data(), y.order(),
                        std::size_t( M ) );
    return out;
}

/// @brief `hypot(x, y) = sqrt(x^2 + y^2)`. Composed from existing kernels.
template < typename T, int M >
[[nodiscard]] inline TaylorExpansionT< T, Dynamic, M > hypot(
    const TaylorExpansionT< T, Dynamic, M >& x,
    const TaylorExpansionT< T, Dynamic, M >& y )
{
    return tax::sqrt( tax::square( x ) + tax::square( y ) );
}

// =============================================================================
// Streaming
// =============================================================================

template < typename T, int M >
inline std::ostream& operator<<( std::ostream& os,
                                 const TaylorExpansionT< T, Dynamic, M >& a )
{
    os << "TaylorExpansionT<order=" << a.order() << ", size=" << M << ", coeffs=[";
    for ( std::size_t i = 0; i < a.coeffs().size(); ++i )
    {
        if ( i != 0 ) os << ", ";
        os << a.coeffs()[i];
    }
    os << "])";
    return os;
}

}  // namespace tax
