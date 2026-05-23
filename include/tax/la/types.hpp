// include/tax/la/types.hpp
//
// Linear-algebra type aliases. Thin wrapper around Eigen that gives
// the rest of the library a single vocabulary for vectors and
// matrices.  Both `Vec` and `Mat` are dynamic-size doubles; the
// fixed-size aliases require the scalar type explicitly so a user
// reading the spelling sees exactly what is being stored.

#pragma once

#include <Eigen/Core>

namespace tax::la
{

/// @brief Dynamic-size column vector of `double` (Eigen::VectorXd).
using Vec = Eigen::VectorXd;

/// @brief Dynamic-size dense matrix of `double` (Eigen::MatrixXd).
using Mat = Eigen::MatrixXd;

/// @brief Fixed-size column vector with `N` rows of scalar `T`.
template < int N, class T >
using VecNT = Eigen::Matrix< T, N, 1 >;

/// @brief Fixed-size square matrix `N x N` of scalar `T`.
template < int N, class T >
using MatNT = Eigen::Matrix< T, N, N >;

/// @brief Fixed-size dense matrix `N x M` of scalar `T`.
template < int N, int M, class T >
using MatNMT = Eigen::Matrix< T, N, M >;

}  // namespace tax::la
