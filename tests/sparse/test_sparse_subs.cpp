#include <gtest/gtest.h>
#include "../testUtils.hpp"

TEST(SparseSubs, SqrtMatchesDense)
{
    auto x = tax::TE<5>::variable(4.0);
    auto sx = tax::sparse(x);
    tax::test::ExpectCoeffsNear(tax::sqrt(sx).dense(), tax::sqrt(x), 1e-12);
}

TEST(SparseSubs, ReciprocalMatchesDense)
{
    auto x = tax::TE<5>::variable(2.0);
    auto sx = tax::sparse(x);
    tax::test::ExpectCoeffsNear(tax::reciprocal(sx).dense(), tax::reciprocal(x), 1e-12);
}

TEST(SparseSubs, DivisionMatchesDense)
{
    auto x = tax::TE<5>::variable(3.0);
    auto y = tax::TE<5>::variable(2.0);
    auto sx = tax::sparse(x), sy = tax::sparse(y);
    tax::test::ExpectCoeffsNear((sx / sy).dense(), x / y, 1e-12);
}

TEST(SparseSubs, IntegerPowMatchesDense)
{
    auto x = tax::TE<6>::variable(1.5);
    auto sx = tax::sparse(x);
    tax::test::ExpectCoeffsNear(tax::pow(sx, 3).dense(), tax::pow(x, 3), 1e-12);
}

TEST(SparseSubs, PowZeroIsOne)
{
    auto x = tax::TE<4>::variable(2.0);
    auto sx = tax::sparse(x);
    auto one = tax::pow(sx, 0);
    EXPECT_NEAR(one.value(), 1.0, 1e-12);
    EXPECT_EQ(one.nnz(), 1u);
}

TEST(SparseSubs, PowOneIsIdentity)
{
    auto x = tax::TE<4>::variable(2.0);
    auto sx = tax::sparse(x);
    tax::test::ExpectCoeffsNear(tax::pow(sx, 1).dense(), x, 1e-12);
}

TEST(SparseSubs, SqrtMultivarMatchesDense)
{
    using TE2 = tax::TE<4, 2>;
    using Input2 = TE2::Input;
    const Input2 p{4.0, 0.5};
    auto x = TE2::variable<0>(p);
    auto y = TE2::variable<1>(p);
    auto f = x + y;
    auto sf = tax::sparse(f);
    tax::test::ExpectCoeffsNear(tax::sqrt(sf).dense(), tax::sqrt(f), 1e-12);
}
