#include <gtest/gtest.h>
#include "../testUtils.hpp"

TEST(Arith, AddUni) {
    auto a = tax::TE<3>::variable(1.0);
    auto b = tax::TE<3>::variable(2.0);
    auto c = a + b;
    EXPECT_DOUBLE_EQ(c.value(), 3.0);
    EXPECT_DOUBLE_EQ(c[1], 2.0);  // both contribute +1
}

TEST(Arith, SubMulti) {
    typename tax::TE<2, 2>::Input p{1.0, 2.0};
    auto x = tax::TE<2, 2>::variable<0>(p);
    auto y = tax::TE<2, 2>::variable<1>(p);
    auto d = x - y;
    EXPECT_DOUBLE_EQ(d.value(), -1.0);
    EXPECT_DOUBLE_EQ((d.coeff<1, 0>()), 1.0);
    EXPECT_DOUBLE_EQ((d.coeff<0, 1>()), -1.0);
}

TEST(Arith, ScalarAddBothSides) {
    auto x = tax::TE<3>::variable(0.5);
    EXPECT_DOUBLE_EQ((x + 2.0).value(), 2.5);
    EXPECT_DOUBLE_EQ((2.0 + x).value(), 2.5);
    EXPECT_DOUBLE_EQ((x - 2.0).value(), -1.5);
    EXPECT_DOUBLE_EQ((2.0 - x).value(), 1.5);
}

TEST(Arith, ScalarMulBothSides) {
    auto x = tax::TE<3>::variable(0.5);
    auto m = 3.0 * x;
    EXPECT_DOUBLE_EQ(m.value(), 1.5);
    EXPECT_DOUBLE_EQ(m[1], 3.0);
}

TEST(Arith, UnaryNegate) {
    auto x = tax::TE<3>::variable(2.0);
    auto n = -x;
    EXPECT_DOUBLE_EQ(n.value(), -2.0);
    EXPECT_DOUBLE_EQ(n[1], -1.0);
}

TEST(Arith, MulUniSquares) {
    auto x = tax::TE<4>::variable(0.0);
    auto y = x * x;
    // y(t) = t^2 → coeffs [0, 0, 1, 0, 0]
    EXPECT_DOUBLE_EQ(y[0], 0.0);
    EXPECT_DOUBLE_EQ(y[2], 1.0);
    EXPECT_DOUBLE_EQ(y[4], 0.0);
}

TEST(Arith, SubUni) {
    auto a = tax::TE<3>::variable(5.0);
    auto b = tax::TE<3>::variable(3.0);
    auto d = a - b;
    EXPECT_DOUBLE_EQ(d.value(), 2.0);
    EXPECT_DOUBLE_EQ(d[1], 0.0);  // both have d/dx = 1, cancel
}

TEST(Arith, DivScalar) {
    auto x = tax::TE<3>::variable(0.5);
    auto d = x / 2.0;
    EXPECT_DOUBLE_EQ(d.value(), 0.25);
    EXPECT_DOUBLE_EQ(d[1], 0.5);
}

TEST(Arith, ScalarMinusTE) {
    auto x = tax::TE<3>::variable(1.0);
    auto r = 5.0 - x;
    EXPECT_DOUBLE_EQ(r.value(), 4.0);
    EXPECT_DOUBLE_EQ(r[1], -1.0);
}

TEST(Arith, MulMultiBilinear) {
    typename tax::TE<2, 2>::Input p{1.0, 2.0};
    auto x = tax::TE<2, 2>::variable<0>(p);
    auto y = tax::TE<2, 2>::variable<1>(p);
    auto z = x * y;
    EXPECT_DOUBLE_EQ(z.value(), 2.0);
    EXPECT_DOUBLE_EQ((z.coeff<1, 0>()), 2.0);
    EXPECT_DOUBLE_EQ((z.coeff<0, 1>()), 1.0);
    EXPECT_DOUBLE_EQ((z.coeff<1, 1>()), 1.0);
}

TEST(Arith, MixedScalarLiterals) {
    // Integer literals must convert to the coefficient scalar type:
    // the scalar parameter is a non-deduced context (std::type_identity_t).
    auto x = tax::TE<3>::variable(2.0);
    auto a = x + 1;
    auto b = 1 + x;
    auto c = x - 1;
    auto d = 2 - x;
    auto e = x * 2;
    auto f = 2 * x;
    auto g = x / 2;
    EXPECT_DOUBLE_EQ(a.value(), 3.0);
    EXPECT_DOUBLE_EQ(b.value(), 3.0);
    EXPECT_DOUBLE_EQ(c.value(), 1.0);
    EXPECT_DOUBLE_EQ(d.value(), 0.0);
    EXPECT_DOUBLE_EQ(e.value(), 4.0);
    EXPECT_DOUBLE_EQ(f.value(), 4.0);
    EXPECT_DOUBLE_EQ(g.value(), 1.0);
    EXPECT_DOUBLE_EQ(e[1], 2.0);
    EXPECT_DOUBLE_EQ(d[1], -1.0);
}
