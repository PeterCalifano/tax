// include/tax/la.hpp
//
// Linear-algebra umbrella header. Pulls in:
//
//   types       — Vec, Mat, VecNT<N,T>, MatNT<N,T>, MatNMT<N,M,T>.
//   num_traits  — Eigen::NumTraits<TaylorExpansion> + internal traits.
//   values      — variables, value, eval.
//   derivatives — derivative, gradient, hessian, jacobian.
//   invert      — formal polynomial-map inversion (Picard iteration).
//
// Everything public lives in namespace `tax::la`.

#pragma once

#include <tax/la/derivatives.hpp>
#include <tax/la/invert.hpp>
#include <tax/la/num_traits.hpp>
#include <tax/la/types.hpp>
#include <tax/la/values.hpp>
