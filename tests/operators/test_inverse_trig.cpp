#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <cmath>

TEST(Asin, RoundTrip) {
    auto x = tax::TE<5>::variable(0.3);
    tax::test::ExpectCoeffsNear(tax::asin(tax::sin(x)), x, 1e-10);
}

TEST(Acos, RoundTrip) {
    auto x = tax::TE<5>::variable(0.3);
    tax::test::ExpectCoeffsNear(tax::acos(tax::cos(x)), x, 1e-10);
}

TEST(Atan, RoundTrip) {
    auto x = tax::TE<5>::variable(0.4);
    tax::test::ExpectCoeffsNear(tax::atan(tax::tan(x)), x, 1e-10);
}

TEST(Atan2, ConsistentWithAtan) {
    auto x = tax::TE<5>::variable(0.6);
    auto y = tax::TE<5>::variable(0.8);
    auto a = tax::atan2(y, x);
    auto b = tax::atan(y / x);
    tax::test::ExpectCoeffsNear(a, b, 1e-12);
}
