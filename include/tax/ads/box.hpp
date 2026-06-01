// include/tax/ads/box.hpp
//
// Box<T, M> — axis-aligned hyperrectangle in M-dimensional space.
// Used as the geometric primitive of the ADS tree: every leaf owns a
// Box describing the subdomain of initial conditions for which its
// payload (typically a DA-valued flow map) is valid.
//
// Storage is std::array<T, M> on both center and halfWidth so the
// type is constexpr-friendly and trivially copyable. Eigen overloads
// exist for ergonomic interop with the tax::la vector aliases.

#pragma once

#include <array>
#include <cstddef>
#include <tax/la/types.hpp>
#include <utility>

namespace tax::ads
{

template < class T, int M >
struct Box
{
    static_assert( M >= 1, "Box dimension must be at least 1" );

    std::array< T, M > center{};
    std::array< T, M > halfWidth{};

    constexpr Box() noexcept = default;

    constexpr Box( std::array< T, M > c, std::array< T, M > hw ) noexcept
        : center( c ), halfWidth( hw )
    {
    }

    template < class CenterDerived, class HalfDerived >
    Box( const Eigen::MatrixBase< CenterDerived >& c, const Eigen::MatrixBase< HalfDerived >& hw )
    {
        for ( int i = 0; i < M; ++i )
        {
            center[static_cast< std::size_t >( i )] = c( i );
            halfWidth[static_cast< std::size_t >( i )] = hw( i );
        }
    }

    [[nodiscard]] constexpr bool contains( const std::array< T, M >& pt ) const noexcept
    {
        for ( int i = 0; i < M; ++i )
        {
            const std::size_t k = static_cast< std::size_t >( i );
            const T d = pt[k] - center[k];
            if ( d > halfWidth[k] ) return false;
            if ( d < -halfWidth[k] ) return false;
        }
        return true;
    }

    [[nodiscard]] constexpr std::pair< Box, Box > split( int dim ) const noexcept
    {
        Box L{ *this };
        Box R{ *this };
        const std::size_t d = static_cast< std::size_t >( dim );
        const T h = halfWidth[d] * T{ 0.5 };
        L.halfWidth[d] = h;
        R.halfWidth[d] = h;
        L.center[d] = center[d] - h;
        R.center[d] = center[d] + h;
        return { L, R };
    }

    [[nodiscard]] constexpr std::array< T, M > denormalize(
        const std::array< T, M >& d ) const noexcept
    {
        std::array< T, M > out{};
        for ( int i = 0; i < M; ++i )
        {
            const std::size_t k = static_cast< std::size_t >( i );
            out[k] = center[k] + halfWidth[k] * d[k];
        }
        return out;
    }

    [[nodiscard]] tax::la::VecNT< M, T > centerEigen() const noexcept
    {
        tax::la::VecNT< M, T > v;
        for ( int i = 0; i < M; ++i ) v( i ) = center[static_cast< std::size_t >( i )];
        return v;
    }

    [[nodiscard]] tax::la::VecNT< M, T > halfWidthEigen() const noexcept
    {
        tax::la::VecNT< M, T > v;
        for ( int i = 0; i < M; ++i ) v( i ) = halfWidth[static_cast< std::size_t >( i )];
        return v;
    }

    template < class Derived >
    [[nodiscard]] bool contains( const Eigen::MatrixBase< Derived >& pt ) const noexcept
    {
        for ( int i = 0; i < M; ++i )
        {
            const T d = pt( i ) - center[static_cast< std::size_t >( i )];
            if ( d > halfWidth[static_cast< std::size_t >( i )] ) return false;
            if ( d < -halfWidth[static_cast< std::size_t >( i )] ) return false;
        }
        return true;
    }

    template < class Derived >
    [[nodiscard]] tax::la::VecNT< M, T > denormalize( const Eigen::MatrixBase< Derived >& d ) const noexcept
    {
        tax::la::VecNT< M, T > out;
        for ( int i = 0; i < M; ++i )
        {
            const std::size_t k = static_cast< std::size_t >( i );
            out( i ) = center[k] + halfWidth[k] * d( i );
        }
        return out;
    }
};

}  // namespace tax::ads
