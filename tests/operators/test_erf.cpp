#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <cmath>

TEST(Erf, AtZero) {
    auto x = tax::TE<3>::variable(0.0);
    auto e = tax::erf(x);
    EXPECT_NEAR(e[0], 0.0, 1e-15);
    EXPECT_NEAR(e[1], 2.0/std::sqrt(M_PI), 1e-12);
}
