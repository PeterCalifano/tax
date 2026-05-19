#include <gtest/gtest.h>
#include "../testUtils.hpp"

TEST(SparseCauchy, MulMatchesDense)
{
    using TE3  = tax::TE<4, 3>;
    using Input3 = TE3::Input;
    const Input3 p{0.3, -0.2, 0.5};
    auto x = TE3::variable<0>(p);
    auto y = TE3::variable<1>(p);
    auto sxy = tax::sparse(x) * tax::sparse(y);
    auto dxy = x * y;
    tax::test::ExpectCoeffsNear(sxy.dense(), dxy, 1e-12);
}

TEST(SparseCauchy, MulUnivariateMatchesDense)
{
    auto x = tax::TE<6>::variable(1.5);
    auto y = tax::TE<6>::variable(2.0);
    auto sx = tax::sparse(x);
    auto sy = tax::sparse(y);
    tax::test::ExpectCoeffsNear((sx * sy).dense(), x * y, 1e-12);
}

TEST(SparseCauchy, MulByZeroIsZero)
{
    auto x = tax::TE<4>::variable(1.0);
    auto sx = tax::sparse(x);
    auto zero = tax::STE<4>{};
    auto result = sx * zero;
    EXPECT_EQ(result.nnz(), 0u);
}

TEST(SparseCauchy, MulByOneIsIdentity)
{
    auto x = tax::TE<4>::variable(2.0);
    auto sx = tax::sparse(x);
    auto one = tax::STE<4>{ 1.0 };
    tax::test::ExpectCoeffsNear((sx * one).dense(), x, 1e-12);
}
