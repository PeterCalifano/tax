#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <cmath>

TEST(Exp, AtZero) {
    auto x = tax::TE<5>::variable(0.0);
    auto e = tax::exp(x);
    // exp(x) at 0: coeffs [1, 1, 1/2, 1/6, 1/24, 1/120]
    EXPECT_NEAR(e[0], 1.0, 1e-15);
    EXPECT_NEAR(e[1], 1.0, 1e-15);
    EXPECT_NEAR(e[2], 0.5, 1e-15);
    EXPECT_NEAR(e[3], 1.0/6.0, 1e-15);
    EXPECT_NEAR(e[4], 1.0/24.0, 1e-15);
    EXPECT_NEAR(e[5], 1.0/120.0, 1e-15);
}

TEST(Log, RoundTripWithExp) {
    auto x = tax::TE<5>::variable(0.7);
    tax::test::ExpectCoeffsNear(tax::log(tax::exp(x)), x, 1e-12);
}

TEST(Log, ConstantOne) {
    auto one = tax::TE<5>::constant(1.0);
    auto l = tax::log(one);
    EXPECT_NEAR(l.value(), 0.0, 1e-15);
}
