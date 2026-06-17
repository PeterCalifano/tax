#pragma once

// include/tax/core/named.hpp
//
// Named, sliceable, composable Taylor expansions (dense, static case).
//
// A `tax::named::NamedTaylorExpansion< T, N, Axes... >` wraps a dense
// `TaylorExpansion< T, N, M >` and attaches a compile-time list of *named
// axes* to it.  Each axis is a contiguous block of the underlying M
// variables identified by a compile-time string (`Axis< "x", 3 >`).
//
// The named layer adds three capabilities on top of the bare expansion:
//
//   * embed   — inject a value over a subset of axes into a superset by
//               remapping multi-indices (value-preserving, same order N);
//   * compose — `+`, `-`, `*`, `/` between expansions over *different* axis
//               sets run in the union of the two sets: both operands are
//               embedded first, then the existing dense kernels do the work.
//               The result type carries the union of axes, so the dependency
//               set is derived automatically from the operands;
//   * slice   — project an expansion back onto a subset of its axes by
//               keeping the monomials that do not depend on the dropped axes
//               (i.e. restricting the dropped axes to their expansion point).
//
// The axis list of every `NamedTaylorExpansion` is kept in canonical order (sorted by
// name, unique), so `x * p` and `p * x` produce the *same* type.

#include <array>
#include <cstddef>
#include <tax/core/concepts.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/operators/arithmetic.hpp>
#include <tax/operators/math_binary.hpp>
#include <tax/operators/math_unary.hpp>
#include <utility>

namespace tax::named
{

// ---------------------------------------------------------------------------
// FixedString — a structural compile-time string usable as an NTTP
// ---------------------------------------------------------------------------

/**
 * @brief A null-terminated compile-time string suitable as a non-type template
 *        parameter (e.g. `Axis< "x", 3 >`).
 *
 * @tparam K  Size of the backing array including the terminating null.
 */
template < std::size_t K >
struct FixedString
{
    char data[K]{};

    /*implicit*/ constexpr FixedString( const char ( &s )[K] ) noexcept
    {
        for ( std::size_t i = 0; i < K; ++i ) data[i] = s[i];
    }

    /// @brief Length of the string excluding the terminating null.
    [[nodiscard]] static constexpr std::size_t size() noexcept { return K - 1; }

    [[nodiscard]] constexpr char operator[]( std::size_t i ) const noexcept { return data[i]; }
};

/// @brief Three-way lexicographic comparison of two FixedStrings (-1 / 0 / 1).
template < std::size_t A, std::size_t B >
[[nodiscard]] constexpr int compareFixed( const FixedString< A >& a,
                                          const FixedString< B >& b ) noexcept
{
    const std::size_t la = a.size();
    const std::size_t lb = b.size();
    const std::size_t n = la < lb ? la : lb;
    for ( std::size_t i = 0; i < n; ++i )
    {
        // Compare as unsigned char: plain char signedness is implementation-defined,
        // so signed comparison would order non-ASCII axis names inconsistently and
        // could make the canonical merged-type ordering platform-dependent.
        const unsigned char ca = static_cast< unsigned char >( a[i] );
        const unsigned char cb = static_cast< unsigned char >( b[i] );
        if ( ca != cb ) return ca < cb ? -1 : 1;
    }
    if ( la == lb ) return 0;
    return la < lb ? -1 : 1;
}

/// @brief Three-way comparison of two FixedString NTTP values (-1 / 0 / 1).
template < FixedString A, FixedString B >
inline constexpr int fixedCompare = compareFixed( A, B );

// ---------------------------------------------------------------------------
// Axis — a named block of `Dim` consecutive variables
// ---------------------------------------------------------------------------

/**
 * @brief A named axis: the compile-time string `Name` labels a block of `Dim`
 *        consecutive variables of the underlying expansion.
 *
 * @tparam Name  Compile-time axis name.
 * @tparam Dim   Number of variables in the block (>= 1).
 */
template < FixedString Name, int Dim >
struct Axis
{
    static constexpr auto name = Name;
    static constexpr int dim = Dim;
    static_assert( Dim >= 1, "Axis dimension must be at least 1" );
};

/// @brief Sign of the name comparison of two axes (-1 / 0 / 1).
/// `fixedCompare` already returns exactly -1/0/1, so no further clamping is needed.
template < typename A, typename B >
inline constexpr int axisSign = fixedCompare< A::name, B::name >;

// ---------------------------------------------------------------------------
// Compile-time axis-list machinery
// ---------------------------------------------------------------------------

namespace detail
{

/// @brief A list of axis types.
template < typename... Ts >
struct TypeList
{
    static constexpr std::size_t size = sizeof...( Ts );
};

template < typename Head, typename List >
struct Prepend;
template < typename Head, typename... Ts >
struct Prepend< Head, TypeList< Ts... > >
{
    using type = TypeList< Head, Ts... >;
};

// --- Merge two name-sorted axis lists into one sorted, unique list ----------

template < int Cmp, typename A, typename B >
struct MergeChoose;

template < typename A, typename B >
struct Merge;

template < typename... Bs >
struct Merge< TypeList<>, TypeList< Bs... > >
{
    using type = TypeList< Bs... >;
};

template < typename A0, typename... As >
struct Merge< TypeList< A0, As... >, TypeList<> >
{
    using type = TypeList< A0, As... >;
};

template < typename A0, typename... As, typename B0, typename... Bs >
struct Merge< TypeList< A0, As... >, TypeList< B0, Bs... > >
    : MergeChoose< axisSign< A0, B0 >, TypeList< A0, As... >, TypeList< B0, Bs... > >
{
};

// A0 < B0 : take A0
template < typename A0, typename... As, typename B0, typename... Bs >
struct MergeChoose< -1, TypeList< A0, As... >, TypeList< B0, Bs... > >
{
    using type =
        typename Prepend< A0,
                          typename Merge< TypeList< As... >, TypeList< B0, Bs... > >::type >::type;
};

// A0 > B0 : take B0
template < typename A0, typename... As, typename B0, typename... Bs >
struct MergeChoose< 1, TypeList< A0, As... >, TypeList< B0, Bs... > >
{
    using type =
        typename Prepend< B0,
                          typename Merge< TypeList< A0, As... >, TypeList< Bs... > >::type >::type;
};

// A0 == B0 (same name) : require identical dimension, take one, advance both
template < typename A0, typename... As, typename B0, typename... Bs >
struct MergeChoose< 0, TypeList< A0, As... >, TypeList< B0, Bs... > >
{
    static_assert( A0::dim == B0::dim,
                   "named axis used with inconsistent dimension across operands" );
    using type =
        typename Prepend< A0, typename Merge< TypeList< As... >, TypeList< Bs... > >::type >::type;
};

/// @brief Left-fold `Merge` over a pack of (singleton) axis lists.
template < typename Acc, typename... Rest >
struct MergeFold
{
    using type = Acc;
};
template < typename Acc, typename First, typename... Rest >
struct MergeFold< Acc, First, Rest... >
{
    using type = typename MergeFold< typename Merge< Acc, First >::type, Rest... >::type;
};

// --- Lookups ----------------------------------------------------------------

/// @brief Variable offset of an axis (matched by name) within a list, or -1.
template < typename List, typename Ax >
struct OffsetOf;
template < typename Ax >
struct OffsetOf< TypeList<>, Ax >
{
    static constexpr int value = -1;
};
template < typename H, typename... Ts, typename Ax >
struct OffsetOf< TypeList< H, Ts... >, Ax >
{
   private:
    static constexpr int tail = OffsetOf< TypeList< Ts... >, Ax >::value;

   public:
    static constexpr int value = ( axisSign< H, Ax > == 0 ) ? 0 : ( tail < 0 ? -1 : H::dim + tail );
};

/// @brief Dimension of the axis named `Name` within a list, or -1 if absent.
template < typename List, FixedString Name >
struct DimOfName;
template < FixedString Name >
struct DimOfName< TypeList<>, Name >
{
    static constexpr int value = -1;
};
template < typename H, typename... Ts, FixedString Name >
struct DimOfName< TypeList< H, Ts... >, Name >
{
    static constexpr int value = ( fixedCompare< H::name, Name > == 0 )
                                     ? H::dim
                                     : DimOfName< TypeList< Ts... >, Name >::value;
};

/// @brief Total number of variables (sum of axis dimensions) in a list.
template < typename List >
struct TotalDim;
template < typename... Axes >
struct TotalDim< TypeList< Axes... > >
{
    static constexpr int value = ( Axes::dim + ... + 0 );
};

/// @brief True if the axes are sorted by name with no duplicates.
template < typename List >
struct IsCanonical : std::true_type
{
};
template < typename A0 >
struct IsCanonical< TypeList< A0 > > : std::true_type
{
};
template < typename A0, typename A1, typename... Rest >
struct IsCanonical< TypeList< A0, A1, Rest... > >
    : std::bool_constant< ( axisSign< A0, A1 > < 0 ) &&
                          IsCanonical< TypeList< A1, Rest... > >::value >
{
};

/// @brief True if every axis of `Sub` is present in `Super` with the same dim.
template < typename Sub, typename Super >
struct IsSubsetOf;
template < typename Super, typename... Bs >
struct IsSubsetOf< TypeList< Bs... >, Super >
    : std::bool_constant< ( ( DimOfName< Super, Bs::name >::value == Bs::dim ) && ... ) >
{
};

// --- Source -> target variable index map -----------------------------------

template < typename Tgt, bool allowDrop, typename... SrcAxes >
[[nodiscard]] constexpr auto buildAxisMapImpl( TypeList< SrcAxes... > ) noexcept
{
    constexpr int Msrc = ( SrcAxes::dim + ... + 0 );
    std::array< int, std::size_t( Msrc ) > map{};
    int so = 0;
    auto place = [&]< typename Ax >() constexpr {
        constexpr int to = OffsetOf< Tgt, Ax >::value;
        static_assert( allowDrop || to >= 0,
                       "embed(): target axis set is not a superset of the source" );
        for ( int l = 0; l < Ax::dim; ++l )
            map[std::size_t( so + l )] = ( to < 0 ) ? -1 : ( to + l );
        so += Ax::dim;
    };
    ( place.template operator()< SrcAxes >(), ... );
    return map;
}

/**
 * @brief Build the per-variable index map from a source axis layout to a
 *        target axis layout.
 *
 * Returns `std::array<int, TotalDim<Src>>` where entry `j` is the target
 * variable index of source variable `j`, or -1 when the source variable's
 * axis is absent from the target.  With `allowDrop == false` a missing axis is
 * a hard error (used for embedding into a superset).
 */
template < typename Src, typename Tgt, bool allowDrop >
[[nodiscard]] constexpr auto buildAxisMap() noexcept
{
    return buildAxisMapImpl< Tgt, allowDrop >( Src{} );
}

}  // namespace detail

// Forward declaration so `detail::Rebind` can name the named type.
template < typename T, int N, typename... Axes >
    requires Scalar< T >
class NamedTaylorExpansion;

namespace detail
{

/// @brief Rebind a `TypeList` of axes into an `NamedTaylorExpansion< T, N, Axes... >`.
template < typename T, int N, typename List >
struct Rebind;
template < typename T, int N, typename... Axes >
struct Rebind< T, N, TypeList< Axes... > >
{
    using type = NamedTaylorExpansion< T, N, Axes... >;
};

/// @brief The named type over the merged (union) axis set of two operands.
template < typename T, int N, typename ListA, typename ListB >
using MergedNamedTaylorExpansion =
    typename Rebind< T, N, typename Merge< ListA, ListB >::type >::type;

}  // namespace detail

// ---------------------------------------------------------------------------
// NamedTaylorExpansion — a named Taylor expansion
// ---------------------------------------------------------------------------

template < typename T, int N, typename... Axes >
    requires Scalar< T >
class NamedTaylorExpansion
{
   public:
    using axis_list = detail::TypeList< Axes... >;
    using scalar_type = T;

    static_assert( detail::IsCanonical< axis_list >::value,
                   "NamedTaylorExpansion axes must be sorted by name and unique; build via "
                   "variables()/composition rather than spelling them by hand" );

    /// @brief Number of underlying variables (sum of axis dimensions).
    static constexpr int vars_v = detail::TotalDim< axis_list >::value;
    static constexpr int order_v = N;

    /// @brief Underlying anonymous dense expansion type.
    using Inner = TaylorExpansion< T, N, vars_v, storage::Dense >;
    using Input = typename Inner::Input;

    // Mirror the underlying storage traits so NamedTaylorExpansion satisfies the
    // tax::TaylorPolynomial concept and can flow through concept-constrained helpers.
    using container_t = typename Inner::container_t;
    static constexpr std::size_t nCoefficients = Inner::nCoefficients;

    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    constexpr NamedTaylorExpansion() noexcept = default;

    /// @brief Constant expansion (value in every axis direction is flat).
    /*implicit*/ constexpr NamedTaylorExpansion( T v ) noexcept : inner_{ v } {}

    /// @brief Wrap an existing anonymous expansion carrying these axes.
    explicit constexpr NamedTaylorExpansion( const Inner& inner ) noexcept : inner_{ inner } {}

    /**
     * @brief Implicit promotion from an expansion over a subset of these axes.
     *
     * A value that depends on fewer axes embeds into this (wider) axis set
     * automatically, so e.g. an `{x}` value can be assigned where an `{x, p}`
     * value is expected — no manual `+ 0 * p` padding required.  The promotion
     * is the value-preserving multi-index embedding (absent axes get zero
     * exponents).
     */
    template < typename... B >
        requires( !std::is_same_v< detail::TypeList< B... >, axis_list > &&
                  detail::IsSubsetOf< detail::TypeList< B... >, axis_list >::value )
    /*implicit*/ constexpr NamedTaylorExpansion(
        const NamedTaylorExpansion< T, N, B... >& other ) noexcept
        : inner_{ other.template embed< NamedTaylorExpansion >().inner() }
    {
    }

    // ------------------------------------------------------------------
    // Coordinate variables
    // ------------------------------------------------------------------

    /**
     * @brief The I-th coordinate variable of the joint variable space at `p`.
     *
     * Equivalent to `Inner::variable<I>(p)` lifted into the named type.
     */
    template < int I >
    [[nodiscard]] static constexpr NamedTaylorExpansion variable( const Input& p ) noexcept
        requires( I >= 0 && I < vars_v )
    {
        return NamedTaylorExpansion{ Inner::template variable< I >( p ) };
    }

    // ------------------------------------------------------------------
    // Access
    // ------------------------------------------------------------------

    /// @brief Constant (zeroth) coefficient.
    [[nodiscard]] constexpr T value() const noexcept { return inner_.value(); }

    /// @brief The underlying anonymous expansion.
    [[nodiscard]] constexpr const Inner& inner() const noexcept { return inner_; }
    [[nodiscard]] constexpr Inner& inner() noexcept { return inner_; }

    // ------------------------------------------------------------------
    // Embedding and slicing
    // ------------------------------------------------------------------

    /**
     * @brief Embed into the target named type `R`, whose axes must be a
     *        superset of this expansion's axes.
     *
     * Each monomial multi-index is remapped from this expansion's variable
     * layout to `R`'s; absent axes receive zero exponents.  Value-preserving.
     */
    template < typename R >
    [[nodiscard]] constexpr R embed() const noexcept
    {
        constexpr auto map =
            detail::buildAxisMap< axis_list, typename R::axis_list, /*allowDrop=*/false >();
        typename R::Inner out{};
        for ( std::size_t k = 0; k < Inner::nCoefficients; ++k )
        {
            const T c = inner_[k];
            if ( c == T{} ) continue;
            const auto a_src = unflatIndex< vars_v >( k );
            MultiIndex< R::vars_v > a_dst{};
            for ( int j = 0; j < vars_v; ++j )
                a_dst[std::size_t( map[std::size_t( j )] )] = a_src[std::size_t( j )];
            out[flatIndex< R::vars_v >( a_dst )] = c;
        }
        return R{ out };
    }

    /**
     * @brief Project onto the subset of axes named by `Names...`.
     *
     * Keeps only the monomials whose exponents on the dropped axes are all
     * zero — i.e. the restriction of the polynomial obtained by setting every
     * dropped axis to its expansion point.  The result type carries exactly
     * the requested axes (canonicalised), with their original dimensions.
     */
    template < FixedString... Names >
    [[nodiscard]] constexpr auto slice() const noexcept
    {
        static_assert( sizeof...( Names ) >= 1, "slice() needs at least one axis name" );
        // Check name existence *before* forming Axis< Name, DimOfName::value >: an
        // absent name yields Dim == -1, which would otherwise trip Axis's own
        // "dimension must be at least 1" assert with a confusing message.
        static_assert( ( ( detail::DimOfName< axis_list, Names >::value >= 1 ) && ... ),
                       "slice(): every requested axis name must exist in this expansion" );
        using Tgt = typename detail::MergeFold<
            detail::TypeList<>,
            detail::TypeList< Axis< Names, detail::DimOfName< axis_list, Names >::value > >... >::
            type;
        using R = typename detail::Rebind< T, N, Tgt >::type;

        constexpr auto map = detail::buildAxisMap< axis_list, Tgt, /*allowDrop=*/true >();
        typename R::Inner out{};
        for ( std::size_t k = 0; k < Inner::nCoefficients; ++k )
        {
            const T c = inner_[k];
            if ( c == T{} ) continue;
            const auto a_src = unflatIndex< vars_v >( k );
            MultiIndex< R::vars_v > a_dst{};
            bool keep = true;
            for ( int j = 0; j < vars_v; ++j )
            {
                const int to = map[std::size_t( j )];
                if ( to < 0 )
                {
                    if ( a_src[std::size_t( j )] != 0 )
                    {
                        keep = false;
                        break;
                    }
                } else
                {
                    a_dst[std::size_t( to )] = a_src[std::size_t( j )];
                }
            }
            if ( keep ) out[flatIndex< R::vars_v >( a_dst )] += c;
        }
        return R{ out };
    }

    // ------------------------------------------------------------------
    // Per-axis differentiation and integration (axis set preserved)
    // ------------------------------------------------------------------

    /// @brief Global variable index of local coordinate `Local` of axis `Name`.
    template < FixedString Name, int Local >
    static constexpr int axisVar() noexcept
    {
        constexpr int dim = detail::DimOfName< axis_list, Name >::value;
        static_assert( dim >= 1, "axis name not present in this expansion" );
        static_assert( Local >= 0 && Local < dim, "local axis index out of range" );
        return detail::OffsetOf< axis_list, Axis< Name, dim > >::value + Local;
    }

    /**
     * @brief Partial derivative with respect to one coordinate of a named axis.
     *
     * `Name` selects the axis, `Local` (default 0) the coordinate within that
     * axis.  The axis set is preserved.  Composing with `slice()` yields the
     * "sub-derivative" projection: `f.deriv<"p">().slice<"x">()` is the
     * p-derivative of `f` viewed as a function of `x` at the expansion point.
     */
    template < FixedString Name, int Local = 0 >
    [[nodiscard]] constexpr NamedTaylorExpansion deriv() const noexcept
    {
        return NamedTaylorExpansion{ inner_.template deriv< axisVar< Name, Local >() >() };
    }

    /**
     * @brief Indefinite integral with respect to one coordinate of a named axis.
     *
     * Mirrors `deriv()`; the axis set is preserved and the order stays `N`
     * (degree-`N` terms are dropped, matching `TaylorExpansion::integ`).
     */
    template < FixedString Name, int Local = 0 >
    [[nodiscard]] constexpr NamedTaylorExpansion integ() const noexcept
    {
        return NamedTaylorExpansion{ inner_.template integ< axisVar< Name, Local >() >() };
    }

   private:
    Inner inner_{};
};

// ---------------------------------------------------------------------------
// Composition operators
// ---------------------------------------------------------------------------

#define TAX_NAMED_BINARY_OP( OP )                                                       \
    template < typename T, int N, typename... A, typename... B >                        \
    [[nodiscard]] constexpr auto operator OP(                                           \
        const NamedTaylorExpansion< T, N, A... >& a,                                    \
        const NamedTaylorExpansion< T, N, B... >& b ) noexcept                          \
    {                                                                                   \
        using R = detail::MergedNamedTaylorExpansion< T, N, detail::TypeList< A... >,   \
                                                      detail::TypeList< B... > >;       \
        return R{ a.template embed< R >().inner() OP b.template embed< R >().inner() }; \
    }

TAX_NAMED_BINARY_OP( +)
TAX_NAMED_BINARY_OP( -)
TAX_NAMED_BINARY_OP( * )
TAX_NAMED_BINARY_OP( / )

#undef TAX_NAMED_BINARY_OP

// --- Scalar combinations (axis set unchanged) ------------------------------

#define TAX_NAMED_SCALAR_OP( OP )                                                           \
    template < typename T, int N, typename... A >                                           \
    [[nodiscard]] constexpr NamedTaylorExpansion< T, N, A... > operator OP(                 \
        const NamedTaylorExpansion< T, N, A... >& a, std::type_identity_t< T > s ) noexcept \
    {                                                                                       \
        return NamedTaylorExpansion< T, N, A... >{ a.inner() OP s };                        \
    }

TAX_NAMED_SCALAR_OP( +)
TAX_NAMED_SCALAR_OP( -)
TAX_NAMED_SCALAR_OP( * )
TAX_NAMED_SCALAR_OP( / )

#undef TAX_NAMED_SCALAR_OP

template < typename T, int N, typename... A >
[[nodiscard]] constexpr NamedTaylorExpansion< T, N, A... > operator+(
    std::type_identity_t< T > s, const NamedTaylorExpansion< T, N, A... >& a ) noexcept
{
    return a + s;
}

template < typename T, int N, typename... A >
[[nodiscard]] constexpr NamedTaylorExpansion< T, N, A... > operator*(
    std::type_identity_t< T > s, const NamedTaylorExpansion< T, N, A... >& a ) noexcept
{
    return a * s;
}

template < typename T, int N, typename... A >
[[nodiscard]] constexpr NamedTaylorExpansion< T, N, A... > operator-(
    std::type_identity_t< T > s, const NamedTaylorExpansion< T, N, A... >& a ) noexcept
{
    return NamedTaylorExpansion< T, N, A... >{ s - a.inner() };
}

template < typename T, int N, typename... A >
[[nodiscard]] constexpr NamedTaylorExpansion< T, N, A... > operator-(
    const NamedTaylorExpansion< T, N, A... >& a ) noexcept
{
    return NamedTaylorExpansion< T, N, A... >{ -a.inner() };
}

// ---------------------------------------------------------------------------
// Unary math functions (forwarded to the inner expansion, axis set preserved)
// ---------------------------------------------------------------------------

// Each wrapper applies the corresponding `tax::` series function to the inner
// anonymous expansion and rewraps the result with the same axis list, so
// transcendental functions of named expansions keep their named structure.
#define TAX_NAMED_UNARY_FN( FN )                                           \
    template < typename T, int N, typename... A >                          \
    [[nodiscard]] NamedTaylorExpansion< T, N, A... > FN(                   \
        const NamedTaylorExpansion< T, N, A... >& a ) noexcept             \
    {                                                                      \
        return NamedTaylorExpansion< T, N, A... >{ tax::FN( a.inner() ) }; \
    }

TAX_NAMED_UNARY_FN( square )
TAX_NAMED_UNARY_FN( cube )
TAX_NAMED_UNARY_FN( sqrt )
TAX_NAMED_UNARY_FN( cbrt )
TAX_NAMED_UNARY_FN( reciprocal )
TAX_NAMED_UNARY_FN( exp )
TAX_NAMED_UNARY_FN( log )
TAX_NAMED_UNARY_FN( sin )
TAX_NAMED_UNARY_FN( cos )
TAX_NAMED_UNARY_FN( tan )
TAX_NAMED_UNARY_FN( asin )
TAX_NAMED_UNARY_FN( acos )
TAX_NAMED_UNARY_FN( atan )
TAX_NAMED_UNARY_FN( sinh )
TAX_NAMED_UNARY_FN( cosh )
TAX_NAMED_UNARY_FN( tanh )
TAX_NAMED_UNARY_FN( asinh )
TAX_NAMED_UNARY_FN( acosh )
TAX_NAMED_UNARY_FN( atanh )
TAX_NAMED_UNARY_FN( erf )

#undef TAX_NAMED_UNARY_FN

// ---------------------------------------------------------------------------
// Binary math functions
// ---------------------------------------------------------------------------

/// @brief `x^n` for an integer exponent (axis set preserved).
template < typename T, int N, typename... A >
[[nodiscard]] NamedTaylorExpansion< T, N, A... > pow( const NamedTaylorExpansion< T, N, A... >& x,
                                                      int n ) noexcept
{
    return NamedTaylorExpansion< T, N, A... >{ tax::pow( x.inner(), n ) };
}

/// @brief `x^p` for a real exponent (axis set preserved; requires x.value() != 0).
template < typename T, int N, typename... A >
[[nodiscard]] NamedTaylorExpansion< T, N, A... > pow( const NamedTaylorExpansion< T, N, A... >& x,
                                                      std::type_identity_t< T > p ) noexcept
{
    return NamedTaylorExpansion< T, N, A... >{ tax::pow( x.inner(), p ) };
}

/// @brief `atan2(y, x)` over the union of the two operands' axis sets.
template < typename T, int N, typename... A, typename... B >
[[nodiscard]] auto atan2( const NamedTaylorExpansion< T, N, A... >& y,
                          const NamedTaylorExpansion< T, N, B... >& x ) noexcept
{
    using R = detail::MergedNamedTaylorExpansion< T, N, detail::TypeList< A... >,
                                                  detail::TypeList< B... > >;
    return R{ tax::atan2( y.template embed< R >().inner(), x.template embed< R >().inner() ) };
}

// ---------------------------------------------------------------------------
// Coordinate-variable factory for a single named axis
// ---------------------------------------------------------------------------

/**
 * @brief Build the `D` coordinate variables of a single named axis `Name`.
 *
 * Returns an array of `D` named expansions, each over just the axis
 * `Axis< Name, D >`, expanded about the point `x0`.  These single-axis values
 * compose freely with variables of other axes — the result type tracks the
 * union of the axes involved.
 *
 * @tparam Name  Axis name.
 * @tparam N     Truncation order.
 * @param  x0    Expansion point for the `D` coordinates.
 */
template < FixedString Name, int N, typename T, std::size_t D >
[[nodiscard]] constexpr auto variables( const std::array< T, D >& x0 ) noexcept
{
    using Ax = Axis< Name, int( D ) >;
    using E = NamedTaylorExpansion< T, N, Ax >;
    std::array< E, D > out{};
    [&]< std::size_t... I >( std::index_sequence< I... > ) {
        ( ( out[I] = E::template variable< int( I ) >( x0 ) ), ... );
    }( std::make_index_sequence< D >{} );
    return out;
}

/**
 * @brief Build the single coordinate variable of a 1-D named axis `Name`.
 *
 * Convenience factory for a scalar expansion point: returns one
 * `NamedTaylorExpansion` over `Axis< Name, 1 >`.  Named in the singular (cf.
 * the plural `variables` factories that return a collection) since the result
 * is a single value.
 *
 * @tparam Name  Axis name.
 * @tparam N     Truncation order.
 * @param  x0    Expansion point.
 */
template < FixedString Name, int N, typename T >
    requires Scalar< T >
[[nodiscard]] constexpr auto variable( T x0 ) noexcept
{
    using E = NamedTaylorExpansion< T, N, Axis< Name, 1 > >;
    typename E::Input p{ x0 };
    return E::template variable< 0 >( p );
}

// ---------------------------------------------------------------------------
// Convenience alias (double-valued)
// ---------------------------------------------------------------------------

/// @brief `NE< N, Axes... >` — double-valued named expansion of order N.
template < int N, typename... Axes >
using NE = NamedTaylorExpansion< double, N, Axes... >;

}  // namespace tax::named

// ---------------------------------------------------------------------------
// Public re-exports: the named API is reachable directly under `tax`
// ---------------------------------------------------------------------------

namespace tax
{
using named::Axis;
using named::FixedString;
using named::NamedTaylorExpansion;
using named::NE;
using named::variable;
using named::variables;
}  // namespace tax
