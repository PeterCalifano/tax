#include <gtest/gtest.h>
#include "../testUtils.hpp"

TEST(SparseArith, AddMatchesDense)
{
    auto x = tax::TE<5>::variable(0.5);
    auto y = tax::TE<5>::variable(0.3);
    auto sx = tax::sparse(x);
    auto sy = tax::sparse(y);
    auto sum = sx + sy;
    tax::test::ExpectCoeffsNear(sum.dense(), x + y);
}

TEST(SparseArith, SubMatchesDense)
{
    auto x = tax::TE<5>::variable(0.5);
    auto y = tax::TE<5>::variable(0.3);
    auto d = tax::sparse(x) - tax::sparse(y);
    tax::test::ExpectCoeffsNear(d.dense(), x - y);
}

TEST(SparseArith, ScalarMul)
{
    auto sx = tax::sparse(tax::TE<5>::variable(0.5));
    auto m = 3.0 * sx;
    EXPECT_NEAR(m.value(), 1.5, 1e-12);
    EXPECT_EQ(m.nnz(), sx.nnz());
}

TEST(SparseArith, UnaryNegateMatchesDense)
{
    auto x = tax::TE<4>::variable(1.0);
    auto sx = tax::sparse(x);
    tax::test::ExpectCoeffsNear((-sx).dense(), -x);
}

TEST(SparseArith, ScalarDivMatchesDense)
{
    auto x = tax::TE<4>::variable(2.0);
    auto sx = tax::sparse(x);
    tax::test::ExpectCoeffsNear((sx / 4.0).dense(), x / 4.0);
}

TEST(SparseArith, AddScalarMatchesDense)
{
    auto x = tax::TE<4>::variable(1.0);
    auto sx = tax::sparse(x);
    tax::test::ExpectCoeffsNear((sx + 5.0).dense(), x + 5.0);
}

TEST(SparseArith, ScalarSubMatchesDense)
{
    auto x = tax::TE<4>::variable(1.0);
    auto sx = tax::sparse(x);
    tax::test::ExpectCoeffsNear((3.0 - sx).dense(), 3.0 - x);
}
