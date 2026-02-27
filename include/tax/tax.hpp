#pragma once

// Umbrella header — includes every layer of the DA library in dependency order.
//
// Lay-out of include/tax/:
//
//   fwd.hpp              — Scalar concept, MultiIndex
//   combinatorics.hpp    — binom, numMonomials, totalDegree, flatIndex
//   kernels.hpp          — array arithmetic, Cauchy product/accumulate, series kernels
//   leaf.hpp             — DALeaf tag, stored_t, is_leaf_v
//   expr/
//     base.hpp           — DAExpr<Derived,T,N,M> CRTP base
//     ops.hpp            — OpAdd, OpSub, OpMul, OpDiv, OpScalar*, OpNeg
//     bin_expr.hpp       — BinExpr<L,R,Op>
//     unary_expr.hpp     — UnaryExpr<E,Op>
//     scalar_expr.hpp    — ScalarExpr<E,Op>
//     scalar_div_l_expr.hpp — ScalarDivLExpr<E>
//     square_expr.hpp    — SquareExpr<E>
//     cube_expr.hpp      — CubeExpr<E>
//     sqrt_expr.hpp      — SqrtExpr<E>
//     sum_expr.hpp       — SumExpr<Es...>
//     product_expr.hpp   — ProductExpr<Es...>
//   da_type.hpp          — DA<T,N,M> leaf / materialised type
//   operators.hpp        — operator overloads, square/cube/sqrt free functions
//   aliases.hpp          — DAd, DAf, DAMd, DAMf

#include <tax/fwd.hpp>
#include <tax/combinatorics.hpp>
#include <tax/kernels.hpp>
#include <tax/leaf.hpp>
#include <tax/expr/base.hpp>
#include <tax/expr/ops.hpp>
#include <tax/expr/bin_expr.hpp>
#include <tax/expr/unary_expr.hpp>
#include <tax/expr/scalar_expr.hpp>
#include <tax/expr/scalar_div_l_expr.hpp>
#include <tax/expr/square_expr.hpp>
#include <tax/expr/cube_expr.hpp>
#include <tax/expr/sqrt_expr.hpp>
#include <tax/expr/sum_expr.hpp>
#include <tax/expr/product_expr.hpp>
#include <tax/da_type.hpp>
#include <tax/operators.hpp>
#include <tax/aliases.hpp>
