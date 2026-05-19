#include <gtest/gtest.h>
#include "../testUtils.hpp"

TEST(SparseConv, DenseToSparseToDenseRoundTrip)
{
    using TE2 = tax::TE<3, 2>;
    using Input2 = TE2::Input;

    auto d = TE2::constant(2.0);
    const Input2 p{1.0, 2.0};
    auto x = TE2::variable<0>(p);
    auto f = d + x;     // f = (2 + 1) + dx, constant=3, coeff of x=1

    auto s = tax::sparse(f);
    EXPECT_GE(s.nnz(), 1u);

    auto back = s.dense();
    tax::test::ExpectCoeffsNear(back, f);
}

TEST(SparseConv, DropExactZeros)
{
    auto d = tax::TE<3>::zero();
    d[0] = 1.0;
    d[2] = 3.0;
    auto s = tax::sparse(d);
    EXPECT_EQ(s.nnz(), 2u);
}
