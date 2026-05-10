#pragma once

#include <tax/expr/func_expr.hpp>
#include <tax/operators/common.hpp>

namespace tax
{

template < typename E >
[[nodiscard]] constexpr auto abs( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpAbs< E::order_ct, E::vars_ct > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto square( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpSquare< E::order_ct, E::vars_ct > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto cube( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpCube< E::order_ct, E::vars_ct > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto sqrt( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpSqrt< E::order_ct, E::vars_ct > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto cbrt( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpCbrt< E::order_ct, E::vars_ct > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto sin( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpSin< E::order_ct, E::vars_ct > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto cos( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpCos< E::order_ct, E::vars_ct > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto tan( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpTan< E::order_ct, E::vars_ct > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto asin( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpAsin< E::order_ct, E::vars_ct > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto acos( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpAcos< E::order_ct, E::vars_ct > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto atan( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpAtan< E::order_ct, E::vars_ct > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto sinh( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpSinh< E::order_ct, E::vars_ct > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto cosh( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpCosh< E::order_ct, E::vars_ct > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto tanh( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpTanh< E::order_ct, E::vars_ct > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto asinh( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpAsinh< E::order_ct, E::vars_ct > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto acosh( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpAcosh< E::order_ct, E::vars_ct > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto atanh( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpAtanh< E::order_ct, E::vars_ct > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto log( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpLog< E::order_ct, E::vars_ct > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto log10( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpLog10< E::order_ct, E::vars_ct > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto exp( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpExp< E::order_ct, E::vars_ct > >{ e.self() };
}

template < typename E >
[[nodiscard]] constexpr auto erf( const ExprBase< E >& e ) noexcept
{
    return detail::FuncExpr< E, detail::OpErf< E::order_ct, E::vars_ct > >{ e.self() };
}

}  // namespace tax
