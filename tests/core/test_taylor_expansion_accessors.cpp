#include <gtest/gtest.h>
#include <tax/tax.hpp>

TEST(Accessors, CompileTimeCoeff) {
    typename tax::TE<3, 2>::Input p{1.0, 2.0};
    auto x = tax::TE<3, 2>::variable<0>(p);
    EXPECT_EQ((x.coeff<0, 0>()), 1.0);
    EXPECT_EQ((x.coeff<1, 0>()), 1.0);
    EXPECT_EQ((x.coeff<0, 1>()), 0.0);
}

TEST(Accessors, CompileTimeDerivativeMultipliesByFactorial) {
    tax::TE<3> f;
    f[0] = 1.0;
    f[1] = 2.0;
    f[2] = 3.0;
    EXPECT_EQ(f.coeff(tax::MultiIndex<1>{2}), 3.0);
    EXPECT_EQ(f.derivative(tax::MultiIndex<1>{2}), 6.0);
    EXPECT_EQ(f.template derivative<2>(), 6.0);
}

TEST(Accessors, RuntimeDerivativeUni) {
    tax::TE<4> f;
    for (std::size_t k = 0; k < f.nCoefficients; ++k) f[k] = double(k);
    EXPECT_EQ(f.derivative(tax::MultiIndex<1>{0}), 0.0);
    EXPECT_EQ(f.derivative(tax::MultiIndex<1>{1}), 1.0);
    EXPECT_EQ(f.derivative(tax::MultiIndex<1>{2}), 4.0);
    EXPECT_EQ(f.derivative(tax::MultiIndex<1>{3}), 18.0);
}
