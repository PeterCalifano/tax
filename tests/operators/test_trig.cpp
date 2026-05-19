#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <cmath>

TEST(Sin, AtZero) {
    auto x = tax::TE<5>::variable(0.0);
    auto s = tax::sin(x);
    // sin(t) at 0: [0, 1, 0, -1/6, 0, 1/120]
    EXPECT_NEAR(s[0], 0.0, 1e-15);
    EXPECT_NEAR(s[1], 1.0, 1e-15);
    EXPECT_NEAR(s[3], -1.0/6.0, 1e-15);
    EXPECT_NEAR(s[5], 1.0/120.0, 1e-15);
}

TEST(SinCos, PythagoreanIdentity) {
    auto x = tax::TE<5>::variable(0.7);
    auto p = tax::sin(x) * tax::sin(x) + tax::cos(x) * tax::cos(x);
    EXPECT_NEAR(p.value(), 1.0, 1e-12);
    for (std::size_t k = 1; k < p.nCoefficients; ++k) EXPECT_NEAR(p[k], 0.0, 1e-12);
}

TEST(Tan, RatioSinCos) {
    auto x = tax::TE<5>::variable(0.5);
    tax::test::ExpectCoeffsNear(tax::tan(x), tax::sin(x) / tax::cos(x), 1e-12);
}
