#pragma once

#include <type_traits>

namespace da {

// =============================================================================
// §4  Leaf tag and stored_t trait
// =============================================================================

/// Tag for DA<T,N,M> leaves.  ET nodes do not inherit from this.
struct DALeaf {};

} // namespace da

namespace da::detail {

/// Storage policy for ET node members:
///   — DA<T,N,M> (a leaf) is stored by const& (zero-copy reference to a named variable)
///   — any ET node      is stored by value    (prevents dangling refs to temporaries)
template <typename E>
using stored_t = std::conditional_t<
    std::is_base_of_v<da::DALeaf, std::remove_cvref_t<E>>,
    const std::remove_cvref_t<E>&,
    std::remove_cvref_t<E>>;

/// Compile-time leaf predicate.  Resolved from the same base-class check that
/// drives stored_t — no extra machinery.  Used in if constexpr branches inside
/// BinExpr to select the zero-temp path when an operand is a DA leaf.
template <typename E>
static constexpr bool is_leaf_v = std::is_base_of_v<da::DALeaf, std::remove_cvref_t<E>>;

} // namespace da::detail
