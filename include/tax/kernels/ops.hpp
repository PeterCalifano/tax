#pragma once

#include <tax/utils/fwd.hpp>

namespace tax::detail
{

template < typename T, std::size_t S >
/// @brief In-place element-wise addition: `o += r`.
constexpr void addInPlace( std::array< T, S >& o, const std::array< T, S >& r ) noexcept
{
    for ( std::size_t i = 0; i < S; ++i ) o[i] += r[i];
}

template < typename T, std::size_t S >
/// @brief In-place element-wise subtraction: `o -= r`.
constexpr void subInPlace( std::array< T, S >& o, const std::array< T, S >& r ) noexcept
{
    for ( std::size_t i = 0; i < S; ++i ) o[i] -= r[i];
}

template < typename T, std::size_t S >
/// @brief In-place sign flip.
constexpr void negateInPlace( std::array< T, S >& o ) noexcept
{
    for ( auto& v : o ) v = -v;
}

template < typename T, std::size_t S >
/// @brief In-place scalar multiply.
constexpr void scaleInPlace( std::array< T, S >& o, T s ) noexcept
{
    for ( auto& v : o ) v *= s;
}

// =============================================================================
// Runtime-shape variants (used by the dynamic-shape `TaylorExpansionT`).
// =============================================================================

/// @brief Runtime overload of `addInPlace`. Caller owns `o.size() == r.size()`.
template < typename T >
inline void addInPlaceRT( T* o, const T* r, std::size_t S ) noexcept
{
    for ( std::size_t i = 0; i < S; ++i ) o[i] += r[i];
}

/// @brief Runtime overload of `subInPlace`.
template < typename T >
inline void subInPlaceRT( T* o, const T* r, std::size_t S ) noexcept
{
    for ( std::size_t i = 0; i < S; ++i ) o[i] -= r[i];
}

/// @brief Runtime overload of `negateInPlace`.
template < typename T >
inline void negateInPlaceRT( T* o, std::size_t S ) noexcept
{
    for ( std::size_t i = 0; i < S; ++i ) o[i] = -o[i];
}

/// @brief Runtime overload of `scaleInPlace`.
template < typename T >
inline void scaleInPlaceRT( T* o, T s, std::size_t S ) noexcept
{
    for ( std::size_t i = 0; i < S; ++i ) o[i] *= s;
}

namespace detail
{

/// @brief Extract the innermost scalar value for sign comparison.
template < typename U >
[[nodiscard]] constexpr auto extractValue( const U& v ) noexcept
{
    if constexpr ( requires { v.value(); } )
        return extractValue( v.value() );
    else
        return v;
}

}  // namespace detail

template < typename T, std::size_t S >
/// @brief Absolute value: `out = |a|`. Requires `a[0] != 0`.
constexpr void seriesAbs( std::array< T, S >& out, const std::array< T, S >& a ) noexcept
{
    out = a;
    if ( detail::extractValue( a[0] ) < 0 ) negateInPlace< T, S >( out );
}

/// @brief Runtime overload of `seriesAbs`. Requires `a[0] != 0`.
template < typename T >
inline void seriesAbsRT( T* out, const T* a, std::size_t S ) noexcept
{
    for ( std::size_t i = 0; i < S; ++i ) out[i] = a[i];
    if ( detail::extractValue( a[0] ) < 0 ) negateInPlaceRT( out, S );
}

}  // namespace tax::detail
