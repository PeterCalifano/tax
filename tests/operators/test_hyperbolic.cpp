#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <cmath>

TEST(Sinh, IdentityWithCosh) {
    auto x = tax::TE<5>::variable(0.3);
    auto s = tax::sinh(x);
    auto c = tax::cosh(x);
    auto sum = c*c - s*s;          // cosh^2 - sinh^2 = 1
    EXPECT_NEAR(sum.value(), 1.0, 1e-12);
    for (std::size_t k = 1; k < sum.nCoefficients; ++k) EXPECT_NEAR(sum[k], 0.0, 1e-12);
}

TEST(Tanh, RatioSinhOverCosh) {
    auto x = tax::TE<5>::variable(0.4);
    tax::test::ExpectCoeffsNear(tax::tanh(x), tax::sinh(x) / tax::cosh(x), 1e-12);
}

TEST(Asinh, RoundTrip) {
    auto x = tax::TE<5>::variable(0.5);
    tax::test::ExpectCoeffsNear(tax::asinh(tax::sinh(x)), x, 1e-10);
}

TEST(Acosh, RoundTrip) {
    auto x = tax::TE<5>::variable(2.0);
    tax::test::ExpectCoeffsNear(tax::acosh(tax::cosh(x)), x, 1e-10);
}

TEST(Atanh, RoundTrip) {
    auto x = tax::TE<5>::variable(0.4);
    tax::test::ExpectCoeffsNear(tax::atanh(tax::tanh(x)), x, 1e-10);
}
