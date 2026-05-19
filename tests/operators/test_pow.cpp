#include <gtest/gtest.h>
#include "../testUtils.hpp"

TEST(Pow, IntegerExponent) {
    auto x = tax::TE<6>::variable(2.0);
    auto a = tax::pow(x, 3);
    auto b = x * x * x;
    tax::test::ExpectCoeffsNear(a, b, 1e-12);
}

TEST(Pow, RealExponent) {
    auto x = tax::TE<5>::variable(4.0);
    auto a = tax::pow(x, 0.5);   // sqrt
    auto b = tax::sqrt(x);
    tax::test::ExpectCoeffsNear(a, b, 1e-12);
}

TEST(Pow, NegativeRealExponent) {
    auto x = tax::TE<5>::variable(2.0);
    auto a = tax::pow(x, -1.0);
    auto b = tax::reciprocal(x);
    tax::test::ExpectCoeffsNear(a, b, 1e-12);
}
