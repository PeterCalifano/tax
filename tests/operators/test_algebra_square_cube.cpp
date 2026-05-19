#include <gtest/gtest.h>
#include "../testUtils.hpp"

TEST(Square, MatchesMul) {
    auto x = tax::TE<5>::variable(0.5);
    auto a = tax::square(x);
    auto b = x * x;
    tax::test::ExpectCoeffsNear(a, b);
}

TEST(Cube, MatchesMul) {
    typename tax::TE<4, 2>::Input p{0.3, -0.2};
    auto x = tax::TE<4, 2>::variable<0>(p);
    auto a = tax::cube(x);
    auto b = x * x * x;
    tax::test::ExpectCoeffsNear(a, b);
}
