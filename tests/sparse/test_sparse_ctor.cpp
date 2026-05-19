#include <gtest/gtest.h>
#include "../testUtils.hpp"

TEST(SparseCtor, ZeroIsEmpty)
{
    tax::STE<3> z;
    EXPECT_EQ(z.nnz(), 0u);
    EXPECT_EQ(z.value(), 0.0);
}

TEST(SparseCtor, ConstantHasOneNonzero)
{
    tax::STE<3> c{2.5};
    EXPECT_EQ(c.nnz(), 1u);
    EXPECT_EQ(c.value(), 2.5);
}

TEST(SparseCtor, VariableHasTwoNonzeros)
{
    auto x = tax::STE<3>::variable(1.5);
    EXPECT_EQ(x.nnz(), 2u);
    EXPECT_EQ(x.value(), 1.5);
    EXPECT_EQ(x.coeff(tax::MultiIndex<1>{1}), 1.0);
}
