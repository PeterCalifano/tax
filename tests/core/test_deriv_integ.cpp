#include <gtest/gtest.h>
#include <tax/tax.hpp>
#include "../testUtils.hpp"

TEST(Deriv, UnivariateMatchesDx) {
    // f = sin(x), df/dx = cos(x)
    auto x = tax::TE<5>::variable(0.3);
    auto f = tax::sin(x);
    auto df = f.deriv<0>();
    auto expected = tax::cos(x);
    // Compare up to order N-1.
    for (std::size_t k = 0; k + 1 < f.nCoefficients; ++k) EXPECT_NEAR(df[k], expected[k], 1e-12);
}

TEST(Integ, RoundTripWithDeriv) {
    // d/dx (integ f) = f
    auto x = tax::TE<5>::variable(0.0);
    auto f = tax::cos(x);
    auto F = f.integ<0>();
    auto dF = F.deriv<0>();
    for (std::size_t k = 0; k + 1 < f.nCoefficients; ++k) EXPECT_NEAR(dF[k], f[k], 1e-12);
}

TEST(Deriv, MultivariatePartial) {
    // f = x*y, df/dx = y
    typename tax::TE<3, 2>::Input p{1.0, 2.0};
    auto x = tax::TE<3, 2>::variable<0>(p);
    auto y = tax::TE<3, 2>::variable<1>(p);
    auto f = x * y;
    auto df_dx = f.deriv<0>();
    tax::test::ExpectCoeffsNear(df_dx, y, 1e-12);
}
