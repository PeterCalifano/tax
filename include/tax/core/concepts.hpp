#pragma once

#include <concepts>
#include <cstddef>

namespace tax
{

/// @brief Scalar constraint used for DA coefficients and function values.
template < typename T >
concept Scalar = std::floating_point< T >;

/**
 * @brief Concept for any truncated Taylor polynomial type.
 *
 * A type `P` satisfies `TaylorPolynomial` if it exposes:
 *   - `scalar_type`     — the coefficient/value scalar type
 *   - `container_t`     — the underlying storage container type
 *   - `order_v`         — compile-time truncation order (int)
 *   - `vars_v`          — compile-time variable count (int)
 *   - `nCoefficients`   — total number of stored coefficients (std::size_t)
 *   - `p.value()`       — the constant (zeroth) coefficient
 */
template < typename P >
concept TaylorPolynomial = requires( const P& p, std::size_t k )
{
    typename P::scalar_type;
    typename P::container_t;
    { P::order_v } -> std::convertible_to< int >;
    { P::vars_v } -> std::convertible_to< int >;
    { P::nCoefficients } -> std::convertible_to< std::size_t >;
    { p.value() } -> std::convertible_to< typename P::scalar_type >;
};

/**
 * @brief Refinement of `TaylorPolynomial` for dense (flat-index) storage.
 *
 * Additionally requires `p[k]` — coefficient access by flat storage index.
 */
template < typename P >
concept DensePolynomial = TaylorPolynomial< P > && requires( const P& p, std::size_t k )
{
    { p[k] } -> std::convertible_to< typename P::scalar_type >;
};

}  // namespace tax
