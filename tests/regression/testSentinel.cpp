// tests/regression/testSentinel.cpp
//
// Slice 1 wiring test. Verifies that TAX_BUILD_REGRESSIONS=ON builds the
// regression subdir, links Google Test, and finds the tax target. No DACE.
// Replaced in slice 2 by the ported testUnivariate / testMultivariate.

#include <gtest/gtest.h>
#include <tax/tax.hpp>

TEST(RegressionSentinel, TaxHeaderIncludes)
{
    constexpr int N = 4;
    auto x = tax::TE<N>::variable(0.5);
    EXPECT_DOUBLE_EQ(x.value(), 0.5);
}
